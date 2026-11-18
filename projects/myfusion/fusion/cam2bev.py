import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

import logging
logger = logging.getLogger(__name__)


@MODELS.register_module()
class Cam2BEVFusionModule(nn.Module):
    """Projects camera image features into a true bird's-eye-view grid
    using a flat-ground-plane assumption (IPM-style), then fuses with the
    LiDAR backbone's native BEV features via channel concatenation + 1x1
    conv.

    Every earlier fusion module in this codebase (TransformerFusionModule,
    LidarOnlyFusionModule/NoPool) treated the camera branch as an
    unstructured set of tokens pooled/projected with no regard to actual
    3D geometry -- the detection head received an uninterpretable blob
    rather than a spatially-aligned camera signal, regardless of how
    correctly everything else was wired (see project_running_log.csv,
    2026-08-26 heatmap-alignment diagnostic: even after every geometry/
    loss-normalization bug was found and fixed, the raw classification
    heatmap's peak activations were still a median 8m from any real car --
    an upstream feature-representation problem, not a wiring bug).

    This module instead explicitly projects each LiDAR BEV grid cell's
    real-world (x, y, z_ground) position into the camera image using the
    sample's own calibration (lidar2cam/cam2img/lidar2img -- already
    populated by the base KittiDataset and carried through automatically
    via Pack3DDetInputs' default meta_keys, confirmed present with no
    dataset changes needed), and bilinearly samples the image feature map
    at that projected pixel -- producing a camera feature map at the SAME
    spatial grid as the LiDAR BEV features, ready for direct channel-wise
    fusion.
    """

    def __init__(self, lidar_channels=768, img_channels=256, d_model=256,
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 voxel_size=(0.16, 0.16, 4),
                 out_size_factor=2,
                 ground_height=-1.7):
        super().__init__()
        self.d_model = d_model
        self.pc_range = point_cloud_range
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor
        self.ground_height = ground_height

        self.lidar_proj = nn.Conv2d(lidar_channels, d_model, kernel_size=1)
        self.img_proj = nn.Conv2d(img_channels, d_model, kernel_size=1)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )
        self.native_hw = None
        self._diag_valid_frac = None  # last-batch fraction of in-image BEV cells, for sanity checks

    def _build_bev_grid_xy(self, H, W, device):
        xs = self.pc_range[0] + (torch.arange(W, device=device, dtype=torch.float32) + 0.5) \
            * self.out_size_factor * self.voxel_size[0]
        ys = self.pc_range[1] + (torch.arange(H, device=device, dtype=torch.float32) + 0.5) \
            * self.out_size_factor * self.voxel_size[1]
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
        return grid_x, grid_y

    def project_img_to_bev(self, img_feat, lidar2img, img_shape, H, W):
        """img_feat: [C, h, w] single-sample image feature map.
        lidar2img: [4, 4] projection matrix (LiDAR frame -> raw image pixels
        of the ORIGINAL, pre-resize KITTI image).
        img_shape: (resized_h, resized_w) -- the resolution the image
        pipeline's Resize step produced (and which img_backbone actually
        saw), needed to convert raw-image pixel projections into the
        [-1, 1] grid_sample coordinate space.
        """
        device = img_feat.device
        grid_x, grid_y = self._build_bev_grid_xy(H, W, device)  # [H, W] each
        grid_z = torch.full_like(grid_x, self.ground_height)
        ones = torch.ones_like(grid_x)
        pts_lidar = torch.stack([grid_x, grid_y, grid_z, ones], dim=0).view(4, -1)  # [4, H*W]

        l2i = lidar2img.to(device=device, dtype=torch.float32)
        pts_img = l2i @ pts_lidar  # [4, H*W]
        eps = 1e-5
        depth = pts_img[2]
        valid = depth > eps
        u = pts_img[0] / depth.clamp(min=eps)
        v = pts_img[1] / depth.clamp(min=eps)

        resized_h, resized_w = img_shape[0], img_shape[1]
        u_norm = (u / resized_w) * 2 - 1
        v_norm = (v / resized_h) * 2 - 1
        valid = valid & (u_norm > -1) & (u_norm < 1) & (v_norm > -1) & (v_norm < 1)

        sample_grid = torch.stack([u_norm, v_norm], dim=-1).view(1, H, W, 2)
        sampled = F.grid_sample(img_feat.unsqueeze(0), sample_grid,
                                 mode='bilinear', padding_mode='zeros',
                                 align_corners=False)  # [1, C, H, W]
        mask = valid.view(1, 1, H, W).float()
        return sampled * mask, valid.float().mean()

    def forward(self, img_feats, lidar_feats, batch_data_samples=None):
        """img_feats: [B, C_img, h, w] (image backbone/neck output, single
        level). lidar_feats: [B, C_lidar, H, W] (LiDAR backbone/neck native
        BEV output). batch_data_samples: list of Det3DDataSample, used only
        to read each sample's calibration + resized img_shape.
        """
        lidar_proj = self.lidar_proj(lidar_feats)  # [B, d_model, H, W]
        B, _, H, W = lidar_proj.shape
        self.native_hw = (H, W)

        img_proj = self.img_proj(img_feats)  # [B, d_model, h, w]

        assert batch_data_samples is not None and len(batch_data_samples) == B, \
            "Cam2BEVFusionModule requires batch_data_samples (for calibration) matching batch size"

        cam_bev_list = []
        valid_fracs = []
        for b in range(B):
            meta = batch_data_samples[b].metainfo
            lidar2img = meta.get('lidar2img', None)
            if lidar2img is None:
                lidar2cam = torch.as_tensor(meta['lidar2cam'], dtype=torch.float32)
                cam2img_raw = torch.as_tensor(meta['cam2img'], dtype=torch.float32)
                cam2img = torch.eye(4, dtype=torch.float32)
                cam2img[:cam2img_raw.shape[0], :cam2img_raw.shape[1]] = cam2img_raw
                lidar2img = cam2img @ lidar2cam
            else:
                lidar2img = torch.as_tensor(lidar2img, dtype=torch.float32)
                if lidar2img.shape[0] < 4 or lidar2img.shape[1] < 4:
                    padded = torch.eye(4, dtype=torch.float32)
                    padded[:lidar2img.shape[0], :lidar2img.shape[1]] = lidar2img
                    lidar2img = padded
            img_shape = meta.get('img_shape', meta.get('pad_shape'))
            cam_bev, valid_frac = self.project_img_to_bev(img_proj[b], lidar2img, img_shape, H, W)
            cam_bev_list.append(cam_bev)
            valid_fracs.append(valid_frac)
        cam_bev = torch.cat(cam_bev_list, dim=0)  # [B, d_model, H, W]
        self._diag_valid_frac = torch.stack(valid_fracs).mean().item()

        fused = self.fuse_conv(torch.cat([lidar_proj, cam_bev], dim=1))
        fused_flat = fused.flatten(2).permute(0, 2, 1)
        lidar_flat = lidar_proj.flatten(2).permute(0, 2, 1)
        img_flat = cam_bev.flatten(2).permute(0, 2, 1)
        return lidar_flat, img_flat, fused_flat
