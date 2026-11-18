"""TokenLearner-based query head for camera-LiDAR BEV 3D detection.

Contribution 2 of the thesis, Stage 1 (static token count).

Where this sits in the pipeline
-------------------------------
    Cam2BEVFusionModule  ->  fused_flat [B, N, C]  ->  THIS HEAD  ->  3D boxes

The fused BEV map has N = 53,568 cells (a 248 x 216 grid at 0.32 m per cell).
More than 98% of those cells are empty road, so running a transformer over the
whole map is both wasteful and computationally impossible (53,568^2 pairs).

This head therefore:
  1. prepares the inputs            (list  -> grid)
  2. scores every cell, S times     (learned, the paper's alpha)
  3. pools each score map           (grid  -> S tokens)
  4. takes each map's centre        (gives every token a position in metres)
  5. [Stage 2, not here] gates tokens to make the count adaptive
  6. refines the tokens             (transformer decoder, tokens act as queries)
  7. predicts one 3D box per token

Steps 2-3 are TokenLearner (Ryoo et al., NeurIPS 2021, Equation 1). Step 4 is our
addition: the paper did classification and never needed positions, we need them to
place boxes.

Stage 1 keeps the token count fixed. The adaptive gate and the budget loss come
later, as a separate step, so that a failure can be traced to one change at a time.

Two changes on top of the plain TokenLearner (both switchable, both ON by
default; set them to False for the ablation baseline):
  - use_token_prior : each token has a learnable home position on the map, so a
                      map can pick the object NEAR its home instead of averaging
                      over every similar-looking object, and the tokens start
                      spread over the map instead of all at its centre.
  - use_ref_loss    : an extra loss on the reference points themselves, so the
                      offset branch cannot absorb the whole centre error and the
                      attention maps are forced to move onto the objects.

Steps 8 and 9 (loss and predict) are at the bottom of the file:
  - loss()    : Hungarian matching + focal/L1 losses
  - predict() : decoding into 3D boxes
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The registry is only available inside mmengine/mmdet3d. We still want this file to
# be importable (and unit-testable) on a plain machine with just PyTorch installed,
# so fall back to a decorator that does nothing.
try:
    from mmengine.registry import MODELS

    _HAS_MMENGINE = True
except ImportError:  # pragma: no cover - only taken outside the mm* environment
    _HAS_MMENGINE = False

    class _DummyRegistry:
        """Stand-in so that @MODELS.register_module() is a no-op outside mmengine."""

        @staticmethod
        def register_module(*args, **kwargs):
            def _decorator(cls):
                return cls

            return _decorator

    MODELS = _DummyRegistry()
    logger.warning(
        "mmengine not found - TokenLearnerQueryHead will not be registered in MODELS. "
        "This is expected when running the standalone unit test."
    )

# Hungarian matching. scipy ships with every mmdet3d install; the import is guarded
# only so that a missing scipy fails with a clear message at the call site instead
# of breaking the import of this whole module.
try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    linear_sum_assignment = None

# Optional mmdet3d structures, used only by predict() to return boxes in the
# format the evaluator expects.
try:
    from mmdet3d.structures import LiDARInstance3DBoxes
    from mmengine.structures import InstanceData

    _HAS_MMDET3D = True
except ImportError:  # pragma: no cover
    _HAS_MMDET3D = False
    LiDARInstance3DBoxes = None
    InstanceData = None


class MLP(nn.Module):
    """A plain multi-layer perceptron applied to the last dimension.

    Used for the query position encoder and the box regression branch. Kept as its
    own module so both call sites read the same way.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # No activation after the final layer: the output is a raw prediction.
            if i < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


@MODELS.register_module()
class TokenLearnerQueryHead(nn.Module):
    """Learns S tokens from a fused BEV map and predicts one 3D box per token.

    Args:
        in_channels: channels of the fused BEV map (``d_model`` of the fusion
            module). 256 in the current pipeline.
        num_tokens: S, the number of tokens/queries. Fixed in Stage 1. Each token
            can produce at most one detection, so S must exceed the largest number
            of ground-truth objects in any frame.
        grid_hw: (H, W) of the BEV map the head receives. Read from the config and
            NOT from ``fusion_module.native_hw``: that attribute is None until the
            first forward pass, so it cannot be used while building the head.
        pc_range: (x_min, y_min, z_min, x_max, y_max, z_max) in metres.
        voxel_size: (x, y, z) voxel size in metres, before the backbone stride.
        out_size_factor: how much the LiDAR backbone shrinks the pillar canvas.
            Together with voxel_size this gives the real size of one cell:
            0.16 * 2 = 0.32 m in the current pipeline.
        num_classes: 3 for KITTI (Car, Pedestrian, Cyclist).
        embed_dims: working width of the decoder.
        num_decoder_layers: start at 1. More layers cost memory linearly.
        num_heads: attention heads in the decoder.
        ffn_dims: hidden width of the decoder's feed-forward block.
        dropout: dropout inside the decoder.
        alpha_groups: grouped-convolution groups inside the score network. Keeps
            the parameter count down. Must divide ``in_channels``.
        memory_pool_size: if > 1, average-pool the BEV map by this factor before
            using it as the decoder's memory. Cross-attention memory grows with
            S * number_of_cells, so set this to 2 if you hit OOM. 1 = full
            resolution, which is what the design assumes.
        offset_scale: metres per unit of the predicted centre offset. The network
            outputs a small number and we multiply by this, so both the prediction
            and its target stay around 1 instead of being tens of metres.
        focal_alpha, focal_gamma: focal loss settings for classification.
        loss_cls_weight, loss_bbox_weight: weights of the two training losses.
        cost_cls_weight, cost_center_weight, cost_dim_weight: weights used only
            inside the Hungarian matcher. They decide which prediction is paired
            with which ground-truth box, not how much each loss counts.
        score_threshold: at inference, drop detections below this score. Kept
            low (0.05) because KITTI AP benefits from low-confidence boxes.
        max_detections: at inference, keep at most this many boxes per frame.
        use_token_prior: CHANGE 1. Give every token a learnable home position on
            the map (a Gaussian prior added to its attention logits). Without it
            the score network only sees local features, so one map cannot choose
            WHICH of several similar objects to look at, and all maps start near
            the centre of the map.
        prior_sigma: starting width of each token's prior, in normalised map
            units (0.15 = roughly 10 m). Learnable per token.
        use_ref_loss: CHANGE 2. Add 'loss_ref', an L1 loss pulling each matched
            token's reference point onto its ground-truth centre. Without it the
            offset branch can absorb the whole centre error and the attention
            maps never need to move.
        loss_ref_weight: weight of 'loss_ref'.
    """

    # Read by MyFusionTransformer. Dense heads (CenterHead) take the reshaped
    # grid wrapped in a list; this head takes the flat [B, N, C] tensor and does
    # its own reshaping, because it needs both forms.
    expects_flat_bev = True

    def __init__(
        self,
        in_channels: int = 256,
        num_tokens: int = 100,
        grid_hw: Sequence[int] = (248, 216),
        pc_range: Sequence[float] = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
        voxel_size: Sequence[float] = (0.16, 0.16, 4.0),
        out_size_factor: int = 2,
        num_classes: int = 3,
        embed_dims: int = 256,
        num_decoder_layers: int = 1,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        dropout: float = 0.1,
        alpha_groups: int = 8,
        memory_pool_size: int = 1,
        offset_scale: float = 10.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 2.0,
        cost_cls_weight: float = 1.0,
        cost_center_weight: float = 2.0,
        cost_dim_weight: float = 0.5,
        score_threshold: float = 0.05,
        max_detections: int = 50,
        use_token_prior: bool = True,
        prior_sigma: float = 0.15,
        use_ref_loss: bool = True,
        loss_ref_weight: float = 1.0,
        train_cfg: Optional[dict] = None,
        # STAGE 2: the adaptive token budget. Off by default, so every Stage 1
        # config and result is unaffected.
        use_token_budget: bool = False,
        budget_weight: float = 0.01,
        target_tokens: int = 30,
        gate_hidden: int = 128,
        gate_init_bias: float = 2.0,
        gate_weight: float = 1.0,
        test_cfg: Optional[dict] = None,
    ):
        super().__init__()
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # ------------------------------------------------------------------
        # Validate the configuration early. A wrong grid size does NOT raise on
        # its own later on (248*216 and 216*248 are the same count), it just
        # silently scrambles every cell, so it is worth being strict here.
        # ------------------------------------------------------------------
        if len(grid_hw) != 2:
            raise ValueError(f"grid_hw must be (H, W), got {grid_hw}")
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
            )
        if memory_pool_size < 1:
            raise ValueError(f"memory_pool_size must be >= 1, got {memory_pool_size}")

        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.grid_h, self.grid_w = int(grid_hw[0]), int(grid_hw[1])
        self.num_cells = self.grid_h * self.grid_w
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.memory_pool_size = memory_pool_size

        self.pc_range = tuple(float(v) for v in pc_range)
        # One cell of the map the head sees, in metres. The pillar canvas is built
        # at voxel_size, then the backbone shrinks it by out_size_factor.
        self.cell_size = float(voxel_size[0]) * float(out_size_factor)

        # Loss / matching / inference settings.
        self.offset_scale = float(offset_scale)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.loss_cls_weight = float(loss_cls_weight)
        self.loss_bbox_weight = float(loss_bbox_weight)
        self.cost_cls_weight = float(cost_cls_weight)
        self.cost_center_weight = float(cost_center_weight)
        self.cost_dim_weight = float(cost_dim_weight)
        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        self.use_token_prior = bool(use_token_prior)
        self.use_ref_loss = bool(use_ref_loss)
        self.loss_ref_weight = float(loss_ref_weight)

        if alpha_groups < 1 or in_channels % alpha_groups != 0:
            logger.warning(
                "alpha_groups=%d does not divide in_channels=%d; falling back to 1.",
                alpha_groups,
                in_channels,
            )
            alpha_groups = 1

        # ------------------------------------------------------------------
        # Step 2: the score network (the paper's alpha).
        # Convolutions, not a Linear layer: with 53,568 positions a dense
        # projection is not practical, and a conv lets each score see the cells
        # around it. Two 3x3 layers reach ~5 cells = ~1.6 m, which covers a
        # pedestrian (2x2 cells) and a useful part of a car (12x6 cells).
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # CHANGE 1: one learnable home position (and width) per token.
        # Centres start on an even grid over the map, in normalised [0, 1]
        # coordinates, so the S tokens begin spread out instead of all sitting
        # at the map centre. Both are learned, so tokens can drift to where
        # objects usually are.
        # ------------------------------------------------------------------
        if self.use_token_prior:
            cols = math.ceil(math.sqrt(num_tokens))
            rows = math.ceil(num_tokens / cols)
            u_x = (torch.arange(cols, dtype=torch.float32) + 0.5) / cols
            u_y = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
            grid_y, grid_x = torch.meshgrid(u_y, u_x, indexing="ij")
            centres = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)[:num_tokens]
            self.token_centres = nn.Parameter(centres)                       # [S, 2]
            self.token_log_sigma = nn.Parameter(
                torch.full((num_tokens,), math.log(prior_sigma))             # [S]
            )
            # Bounds for the clamp in _token_prior; see that method for why.
            self._log_sigma_min = math.log(0.02)    # ~1.4 m across this map
            self._log_sigma_max = math.log(0.50)    # ~35 m, i.e. nearly no prior

        self.alpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      groups=alpha_groups),
            nn.GroupNorm(alpha_groups, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, num_tokens, kernel_size=3, padding=1),
        )

        # If the fusion map is not already embed_dims wide, project it once.
        self.input_proj = (
            nn.Identity() if in_channels == embed_dims
            else nn.Linear(in_channels, embed_dims)
        )

        # ------------------------------------------------------------------
        # Step 6: the decoder. Each layer = self-attention between the queries,
        # then cross-attention from the queries into the BEV map, then an FFN.
        # ------------------------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=ffn_dims,
            dropout=dropout,
            batch_first=True,   # we work with [B, L, C] everywhere
            norm_first=True,    # pre-norm trains more stably at low layer counts
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(embed_dims),
        )

        # Position encoders. The queries get a learned encoding of their reference
        # point; the memory cells get a fixed sine encoding (buffer, see below).
        self.query_pos_mlp = MLP(2, embed_dims, embed_dims, num_layers=2)

        # ------------------------------------------------------------------
        # Step 7: prediction branches.
        # 8 box values: dx, dy, z, log_l, log_w, log_h, sin(yaw), cos(yaw)
        # dx, dy are an OFFSET from the token's reference point, in metres.
        # ------------------------------------------------------------------
        self.cls_head = nn.Linear(embed_dims, num_classes)
        self.reg_head = MLP(embed_dims, embed_dims, 8, num_layers=3)

        # ------------------------------------------------------------------
        # Constant geometry, built once. register_buffer so that:
        #   - it follows the model to the GPU
        #   - the optimizer never touches it
        #   - persistent=False keeps it out of checkpoints (no load/save mismatch)
        # ------------------------------------------------------------------
        cell_xy, cell_xy_grid = self._build_cell_coordinates()
        self.register_buffer("cell_xy", cell_xy, persistent=False)
        # Normalised copy, used by the token prior on every forward pass. Caching
        # it avoids recomputing the same [N, 2] tensor thousands of times.
        self.register_buffer(
            "cell_xy_norm", self._normalize_xy(cell_xy), persistent=False
        )

        memory_xy = self._pool_coordinates(cell_xy_grid)
        self.register_buffer(
            "memory_pos_embed",
            self._build_sine_position_embedding(memory_xy),
            persistent=False,
        )

        self._init_weights()
        self.use_token_budget = bool(use_token_budget)
        self.budget_weight = float(budget_weight)
        self.target_tokens = int(target_tokens)
        self.gate_weight = float(gate_weight)
        if self.use_token_budget:
            # Reads the pooled token (in_channels wide), not the decoder's
            # output, because the decision has to be made before the decoder
            # for the saving to be real.
            self.gate_mlp = MLP(in_channels, gate_hidden, 1, num_layers=2)
            nn.init.constant_(self.gate_mlp.layers[-1].bias, float(gate_init_bias))
            logger.info(
                "token budget ON: target=%d tokens, budget_weight=%.4g, "
                "gate_weight=%.4g",
                self.target_tokens, self.budget_weight, self.gate_weight,
            )

        logger.info(
            "TokenLearnerQueryHead ready: grid=%dx%d (%d cells, %.2f m each), "
            "C=%d, S=%d, decoder_layers=%d, memory_pool=%d",
            self.grid_h, self.grid_w, self.num_cells, self.cell_size,
            in_channels, num_tokens, num_decoder_layers, memory_pool_size,
        )

    # ======================================================================
    # Construction helpers
    # ======================================================================

    def _build_cell_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Real-world (x, y) centre of every BEV cell, in metres.

        Mirrors ``Cam2BEVFusionModule._build_bev_grid_xy`` exactly, so the two
        modules agree on what a cell index means.

            x = x_min + (col + 0.5) * cell_size      -> forward,  W = 216 columns
            y = y_min + (row + 0.5) * cell_size      -> lateral,  H = 248 rows

        The +0.5 puts the coordinate at the centre of the cell, not its corner.

        Returns:
            cell_xy:      [N, 2]      flattened, index = row * W + col
            cell_xy_grid: [1, 2, H, W] same values, kept for pooling
        """
        xs = self.pc_range[0] + (torch.arange(self.grid_w, dtype=torch.float32) + 0.5) * self.cell_size
        ys = self.pc_range[1] + (torch.arange(self.grid_h, dtype=torch.float32) + 0.5) * self.cell_size

        # indexing='ij' -> rows follow ys, columns follow xs. This is the same
        # convention the fusion module used when it flattened the map.
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # [H, W] each

        cell_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)   # [N, 2]
        cell_xy_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        return cell_xy, cell_xy_grid

    def _pool_coordinates(self, cell_xy_grid: torch.Tensor) -> torch.Tensor:
        """Coordinates of the memory cells, after optional average pooling.

        Args:
            cell_xy_grid: [1, 2, H, W]
        Returns:
            [M, 2] where M = N when memory_pool_size == 1
        """
        if self.memory_pool_size == 1:
            return cell_xy_grid.squeeze(0).flatten(1).transpose(0, 1).contiguous()

        pooled = F.avg_pool2d(cell_xy_grid, kernel_size=self.memory_pool_size)
        return pooled.squeeze(0).flatten(1).transpose(0, 1).contiguous()

    def _normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Map metres to [0, 1] using the point cloud range.

        Works for any trailing shape [..., 2]. Keeping positions in [0, 1] makes
        the position encoders independent of the scene size.
        """
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        norm_x = (xy[..., 0] - x_min) / max(x_max - x_min, 1e-6)
        norm_y = (xy[..., 1] - y_min) / max(y_max - y_min, 1e-6)
        return torch.stack([norm_x, norm_y], dim=-1)

    def _build_sine_position_embedding(self, xy: torch.Tensor) -> torch.Tensor:
        """Fixed sine/cosine position encoding for the memory cells.

        A learned embedding for 53,568 cells would add ~13.7M parameters for no
        real benefit, so the memory uses a fixed encoding. Only the queries get a
        learned one (``query_pos_mlp``), because their positions move as the
        attention maps change.

        Args:
            xy: [M, 2] in metres
        Returns:
            [M, embed_dims]
        """
        num_feats = self.embed_dims // 2        # half the width per axis
        temperature = 10000.0
        scale = 2 * math.pi                     # math.pi, not torch.pi: older torch lacks it

        xy_norm = self._normalize_xy(xy) * scale                      # [M, 2]

        dim_t = torch.arange(num_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_feats)

        pos_x = xy_norm[:, 0:1] / dim_t                               # [M, num_feats]
        pos_y = xy_norm[:, 1:2] / dim_t

        # interleave sin on even indices, cos on odd indices
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)

        return torch.cat([pos_y, pos_x], dim=1)                       # [M, embed_dims]

    def _init_weights(self) -> None:
        """Small, sane initialisation.

        The regression head's last layer starts at zero so that, on the very first
        forward pass, every box centre equals its reference point and every size is
        exp(0) = 1 m. Training then only has to learn the correction, which is a
        much easier starting point than a random box.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.reg_head.layers[-1].weight)
        nn.init.zeros_(self.reg_head.layers[-1].bias)

        # Classification starts pessimistic: most queries find nothing, so a low
        # prior probability avoids a huge loss in the first iterations (the usual
        # focal-loss bias trick).
        prior_prob = 0.01
        bias_value = -float(torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        nn.init.constant_(self.cls_head.bias, bias_value)

    # ======================================================================
    # Step 1 - prepare inputs
    # ======================================================================
    @staticmethod
    def _to_flat(features: Any) -> torch.Tensor:
        """Accept whatever the surrounding detector hands the head.

        MyFusionTransformer passes the flat [B, N, C] map directly, because
        this head sets ``expects_flat_bev = True``. mmdet3d's own detectors
        (VoxelNet, via SingleStage3DDetector) pass the neck's output instead:
        a list holding one [B, C, H, W] grid. Both end up as [B, N, C] with
        cell n at row = n // W, col = n % W, the order every later step
        assumes.
        """
        if isinstance(features, (list, tuple)):
            if len(features) != 1:
                raise ValueError(
                    f"expected a single feature level, got {len(features)}. "
                    "This head reads one BEV map; if the neck returns several "
                    "levels, select the stride that matches grid_hw."
                )
            features = features[0]

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                "expected a tensor or a one-element list of tensors, got "
                f"{type(features).__name__}"
            )

        if features.dim() == 4:
            # [B, C, H, W] -> [B, N, C]. Exact inverse of the reshape in
            # _prepare_inputs, so both forms describe the same map.
            features = features.flatten(2).transpose(1, 2)

        return features
    def _prepare_inputs(self, fused_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Turn the flat list of cells back into a 2D map.

        The fusion module flattened the map with ``flatten(2)``, so cell n sits at
        row = n // W, col = n % W. We undo exactly that.

        Args:
            fused_flat: [B, N, C]
        Returns:
            bev_grid: [B, C, H, W]  - for the convolutions in step 2
            bev_flat: [B, N, C]     - for pooling (step 3) and the decoder (step 6)

        Nothing is learned here and no value changes; only the arrangement does.
        """
        fused_flat = self._to_flat(fused_flat)

        if fused_flat.dim() != 3:
            raise ValueError(
                f"expected fused_flat with 3 dims [B, N, C], got shape {tuple(fused_flat.shape)}"
            )

        batch, num_cells, channels = fused_flat.shape

        if num_cells != self.num_cells:
            raise ValueError(
                f"got {num_cells} cells but the head is configured for "
                f"{self.grid_h}x{self.grid_w}={self.num_cells}. "
                "Check grid_hw against the LiDAR backbone's actual output stride."
            )
        if channels != self.in_channels:
            raise ValueError(
                f"got {channels} channels but the head is configured for {self.in_channels}. "
                "Check d_model in the fusion module config."
            )

        try:
            # permute first so that channels become the leading axis, then split
            # the cell axis back into (H, W). reshape (not view) because permute
            # leaves the tensor non-contiguous.
            bev_grid = fused_flat.permute(0, 2, 1).reshape(
                batch, channels, self.grid_h, self.grid_w
            )
        except RuntimeError as err:
            # Re-raise with the numbers that actually matter; the native message
            # only mentions element counts.
            raise RuntimeError(
                f"failed to reshape {tuple(fused_flat.shape)} into "
                f"[{batch}, {channels}, {self.grid_h}, {self.grid_w}]: {err}"
            ) from err

        logger.debug("step 1 | bev_grid=%s bev_flat=%s",
                     tuple(bev_grid.shape), tuple(fused_flat.shape))
        return bev_grid, fused_flat

    # ======================================================================
    # Step 2 - attention maps (the paper's alpha + normalisation)
    # ======================================================================

    def _token_prior(self) -> torch.Tensor:
        """CHANGE 1: log of a Gaussian around each token's home position.

        Returns:
            [1, S, N] to be added to the attention logits before the softmax

        Cells far from a token's home get a large negative score, so that token
        picks the object NEAR its home instead of averaging over every object that
        looks alike.

        Two details that matter in practice:

        * ``log_sigma`` is clamped. The gradient on a token's centre scales with
          1/sigma^2, so a shrinking sigma makes it explode: measured at
          sigma=2e-9 the centre gradient reaches 1e16, which trips AMP's
          inf check and silently skips optimizer steps. The clamp bounds sigma to
          roughly 1.4 m .. 35 m on this map, which covers everything useful.
        * the distance is computed per axis instead of stacking a [S, N, 2]
          difference, which would hold 43 MB at S=100 (128 MB at S=300) through
          the whole backward pass for no reason.
        """
        cell_norm = self.cell_xy_norm                                   # [N, 2]
        log_sigma = self.token_log_sigma.float().clamp(
            min=self._log_sigma_min, max=self._log_sigma_max
        )
        centres = self.token_centres.float()                            # [S, 2]

        dx = centres[:, 0:1] - cell_norm[None, :, 0]                    # [S, N]
        dy = centres[:, 1:2] - cell_norm[None, :, 1]                    # [S, N]
        dist2 = dx * dx + dy * dy

        sigma2 = log_sigma.mul(2).exp()[:, None]                        # [S, 1]
        return (-dist2 / (2 * sigma2)).unsqueeze(0)                     # [1, S, N]

    def _compute_attention_maps(self, bev_grid: torch.Tensor) -> torch.Tensor:
        """Score every cell, once per token.

        Args:
            bev_grid: [B, C, H, W]
        Returns:
            attn: [B, S, N], every map sums to 1 over the cells

        The softmax runs over the CELL axis, not over tokens: each map has to be a
        distribution over space ("where do I look"), not a choice between tokens.

        This is the only learned part of steps 1-4.

        Mixed precision: the softmax covers 53,568 values, which is far too many to
        normalise reliably in fp16, so it is always computed in fp32 and cast back
        afterwards. Without this, AMP training produces maps that do not sum to 1.
        """
        logits = self.alpha(bev_grid)                       # [B, S, H, W]
        logits = logits.flatten(2)                          # [B, S, N]

        if self.use_token_prior:
            attn = (logits.float() + self._token_prior()).softmax(dim=-1).to(logits.dtype)
        else:
            attn = logits.float().softmax(dim=-1).to(logits.dtype)

        if logger.isEnabledFor(logging.DEBUG):
            # Cheap sanity numbers; only computed when debug logging is on.
            logger.debug(
                "step 2 | attn=%s sum=%.4f max=%.4f",
                tuple(attn.shape),
                attn.sum(-1).mean().item(),
                attn.max().item(),
            )
        return attn

    # ======================================================================
    # Step 3 - pool the tokens (the paper's gamma, elementwise product, rho)
    # ======================================================================

    @staticmethod
    def _pool_tokens(attn: torch.Tensor, bev_flat: torch.Tensor) -> torch.Tensor:
        """Weighted average of the cells, once per token.

        Args:
            attn:     [B, S, N]
            bev_flat: [B, N, C]
        Returns:
            tokens: [B, S, C]

        This single einsum is the paper's broadcast, elementwise product and
        spatial average pooling together. Doing them literally would build a
        [B, S, N, C] tensor - about 5.5 GB per sample at S=100 - which is why the
        contraction is done in one go instead.

        Nothing is learned here.
        """
        tokens = torch.einsum("bsn,bnc->bsc", attn, bev_flat)
        logger.debug("step 3 | tokens=%s", tuple(tokens.shape))
        return tokens

    # ======================================================================
    # Step 4 - reference points (our addition, not in the paper)
    # ======================================================================

    def _compute_reference_points(self, attn: torch.Tensor) -> torch.Tensor:
        """Centre of mass of each attention map, in metres.

        A token is just a vector and carries no position, but its attention map
        does. The same weights used to pool the features are applied to the cell
        coordinates instead.

        Args:
            attn: [B, S, N]
        Returns:
            ref_xy: [B, S, 2] in metres (x forward, y lateral)

        Known failure mode: if one map splits its weight between two objects, the
        centre lands on empty road between them. Only worth fixing (with a
        concentration penalty) if the plotted maps actually show it.

        Nothing is learned here.
        """
        # The buffer is fp32; cast it to whatever precision the batch is running
        # in, otherwise matmul refuses to mix half and float.
        cell_xy = self.cell_xy.to(dtype=attn.dtype)
        ref_xy = torch.matmul(attn, cell_xy)                # [B,S,N] @ [N,2]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "step 4 | ref_xy=%s x:[%.2f, %.2f] y:[%.2f, %.2f]",
                tuple(ref_xy.shape),
                ref_xy[..., 0].min().item(), ref_xy[..., 0].max().item(),
                ref_xy[..., 1].min().item(), ref_xy[..., 1].max().item(),
            )
        return ref_xy

    # ======================================================================
    # Step 6 - decoder
    # ======================================================================

    def _build_memory(self, bev_grid: torch.Tensor) -> torch.Tensor:
        """The map the queries read from, with its position encoding added.

        Args:
            bev_grid: [B, C, H, W]
        Returns:
            memory: [B, M, embed_dims]
        """
        if self.memory_pool_size > 1:
            bev_grid = F.avg_pool2d(bev_grid, kernel_size=self.memory_pool_size)

        memory = bev_grid.flatten(2).transpose(1, 2)        # [B, M, C]
        memory = self.input_proj(memory)                    # [B, M, embed_dims]

        if memory.shape[1] != self.memory_pos_embed.shape[0]:
            raise RuntimeError(
                f"memory has {memory.shape[1]} cells but the position embedding has "
                f"{self.memory_pos_embed.shape[0]}. These are built from the same "
                "config, so this means grid_hw or memory_pool_size changed after init."
            )

        # Without a position encoding the decoder cannot tell one cell from another.
        # Cast the fp32 buffer to the batch's dtype so AMP does not silently
        # promote the whole memory tensor back to fp32 (it is the largest tensor
        # in the head, so that costs real memory).
        return memory + self.memory_pos_embed.to(dtype=memory.dtype).unsqueeze(0)

    def _decode(
        self,
        tokens: torch.Tensor,
        ref_xy: torch.Tensor,
        bev_grid: torch.Tensor,
        keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Let the tokens talk to each other and read the map again.

        Args:
            tokens:   [B, S, C]
            ref_xy:   [B, S, 2] in metres
            bev_grid: [B, C, H, W]
            keep:     [B, S] with 1 = open, 0 = closed. Stage 2 only; None means
                      every token is open.
        Returns:
            query_feats: [B, S, embed_dims]

        Self-attention (S x S) stops two tokens from chasing the same object.
        Cross-attention (S x M) lets a blurry pooled token recover detail from the
        map around its reference point.

        Memory note: the cross-attention weights are B * heads * S * M floats. At
        S=100, M=53,568, 8 heads that is ~171 MB per sample per layer. Start with
        one layer; raise memory_pool_size if that is too much.
        """
        query = self.input_proj(tokens)                                  # [B,S,E]
        query = query + self.query_pos_mlp(self._normalize_xy(ref_xy))   # add position

        memory = self._build_memory(bev_grid)                            # [B,M,E]

        # A sample with every token closed makes nn.MultiheadAttention return
        # NaN for the whole row, which then spreads through the batch.
        # _token_gate already prevents it, but _decode must not depend on its
        # caller for that: measured, an all-closed keep produced non-finite
        # output here. Reopening the first token costs one query and keeps the
        # batch usable.
        if keep is not None:
            empty = (keep > 0.5).sum(dim=1) == 0
            if bool(empty.any()):
                keep = keep.clone()
                keep[empty, 0] = 1.0
                logger.debug("step 6 | %d sample(s) had no open token; "
                             "reopened token 0", int(empty.sum()))

        # STAGE 2, inference: actually drop the closed tokens rather than
        # masking them.
        #
        # Masking alone saves nothing. tgt_key_padding_mask only removes a
        # closed token from the SELF-attention keys; all S queries still run
        # cross-attention over every one of the M memory cells, which is by
        # far the most expensive part of the head. Measured on the real
        # 248x216 grid: decoding 100 queries with 95 of them masked took
        # 0.209 s, exactly the same as decoding 100 with none masked, while
        # a head built with S=5 ran the whole forward pass in 0.438 s against
        # 0.851 s at S=100. The saving is real but masking does not collect
        # it.
        #
        # Gathering gives bit-identical results for the kept tokens: an open
        # token's self-attention already ignores the closed ones through the
        # mask, cross-attention treats each query independently, and the
        # closed tokens' outputs are discarded anyway. So this changes speed
        # and nothing else, and needs no retraining.
        #
        # Only at inference and only for a single sample, where the kept
        # count is one number rather than a ragged set per batch element.
        # Training keeps the masked path so every token still receives a
        # gradient.
        if (keep is not None and not self.training and query.shape[0] == 1):
            index = (keep[0] > 0.5).nonzero(as_tuple=True)[0]
            if 0 < index.numel() < query.shape[1]:
                gathered = self.decoder(
                    tgt=query[:, index],
                    memory=memory,
                    tgt_key_padding_mask=None,
                )
                # Scatter back so the caller still sees [B, S, E] and every
                # index means the same token it always did. Closed rows stay
                # zero; predict() drops them on the gate anyway.
                query_feats = query.new_zeros(
                    query.shape[0], query.shape[1], gathered.shape[-1])
                query_feats[:, index] = gathered
                logger.debug("step 6 | decoded %d of %d tokens",
                             index.numel(), query.shape[1])
                return query_feats

        # nn.Transformer wants True where a position must be IGNORED.
        tgt_key_padding_mask = None
        if keep is not None:
            tgt_key_padding_mask = keep < 0.5

        query_feats = self.decoder(
            tgt=query,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logger.debug("step 6 | query_feats=%s memory=%s",
                     tuple(query_feats.shape), tuple(memory.shape))
        return query_feats

    # ======================================================================
    # Step 7 - predictions
    # ======================================================================

    def _predict_boxes(
        self, query_feats: torch.Tensor, ref_xy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """One class vector and one 3D box per token.

        Args:
            query_feats: [B, S, embed_dims]
            ref_xy:      [B, S, 2]
        Returns:
            dict with
                cls_logits [B, S, num_classes]  raw scores (apply sigmoid to read)
                bbox_raw   [B, S, 8]            the network's raw output
                center_xy  [B, S, 2]            ref_xy + predicted offset, metres

        bbox_raw layout:
            0, 1  dx, dy  offset from the reference point, in offset_scale units
            2     z       centre height
            3,4,5 log l, log w, log h
            6,7   sin(yaw), cos(yaw)

        The offset is multiplied by ``offset_scale`` (10 m by default) so the raw
        number stays around 1. Without that, a query matched to a box 30 m away
        would have to output 30, and the gradients would be badly scaled.

        Sizes are predicted in log space so they stay positive and the scale stays
        stable. The yaw is a sin/cos pair to avoid the discontinuity at +/- pi.

        There is no explicit "background" class: a query that finds nothing simply
        produces low scores for all three classes (the focal-loss convention).
        """
        cls_logits = self.cls_head(query_feats)             # [B, S, num_classes]
        bbox_raw = self.reg_head(query_feats)               # [B, S, 8]

        center_xy = ref_xy + bbox_raw[..., 0:2] * self.offset_scale

        logger.debug("step 7 | cls=%s bbox=%s",
                     tuple(cls_logits.shape), tuple(bbox_raw.shape))
        return {
            "cls_logits": cls_logits,
            "bbox_raw": bbox_raw,
            "center_xy": center_xy,
        }

    # ======================================================================
    # Full forward pass
    # ======================================================================
        # ======================================================================
    # Step 5 - the adaptive token budget (Stage 2)
    # ======================================================================

    def _token_gate(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decide, per token, whether it is worth spending.

        Args:
            tokens: [B, S, C] straight from the pooling step
        Returns:
            keep: [B, S], exactly 0 or 1 going forward, with the sigmoid's
                  gradient coming back
            soft: [B, S], the raw keep probability, used by the budget loss

        Hard forward, soft backward -- the usual straight-through estimator.
        A purely soft gate would train a model that never drops anything and
        then drop tokens at inference, which is a different model from the one
        that was trained.
        """
        soft = torch.sigmoid(self.gate_mlp(tokens).squeeze(-1))       # [B, S]
        hard = (soft > 0.5).to(soft.dtype)

        # nn.MultiheadAttention returns NaN for a query row whose key mask is
        # entirely True, so a sample that closed everything would poison the
        # batch. Reopen its most confident token instead.
        empty = hard.sum(dim=1) == 0
        if bool(empty.any()):
            best = soft.argmax(dim=1)
            hard[empty, best[empty]] = 1.0

        keep = hard + soft - soft.detach()   # forward: hard, backward: soft
        return keep, soft
    def forward(self, fused_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run steps 1-4, 6 and 7.

        Args:
            fused_flat: [B, N, C] straight from Cam2BEVFusionModule
        Returns:
            dict with cls_logits, bbox_raw, center_xy, tokens, attn, ref_xy

        ``attn`` is returned because the attention maps are the main thing to plot
        when checking whether the tokens are landing on real objects. It is a
        [B, S, N] tensor, so drop it from the dict if memory ever gets tight.
        """
        bev_grid, bev_flat = self._prepare_inputs(fused_flat)

        attn = self._compute_attention_maps(bev_grid)
        tokens = self._pool_tokens(attn, bev_flat)
        ref_xy = self._compute_reference_points(attn)

        # Stage 2 will insert the budget gate here and pass `keep` into _decode.
        if self.use_token_budget:
            keep, keep_soft = self._token_gate(tokens)
        else:
            keep, keep_soft = None, None

        query_feats = self._decode(tokens, ref_xy, bev_grid, keep=keep)
        outputs = self._predict_boxes(query_feats, ref_xy)

        if keep is not None:
            # The gate deliberately does NOT touch the class logits.
            #
            # An earlier version subtracted a penalty from a closed token's
            # logits, so that closing a token the matcher needed would blow up
            # the classification loss and push the gate back open. In practice
            # it did the opposite: about 88 of the 100 tokens match nothing,
            # and for every one of those, lowering the logit LOWERS the focal
            # loss. The classification loss therefore paid every token to
            # close, independently of the budget. Measured on this pipeline:
            # with a budget of 60 the gate settled at 11.8 open tokens while
            # loss_budget never left zero, grad_norm rose from ~30 to ~85, and
            # loss_bbox and loss_ref both ended worse than Stage 1.
            #
            # The gate is trained by its own loss instead (see
            # _compute_losses), which leaves the detection gradients exactly as
            # Stage 1 has them.
            outputs["keep"] = keep
            outputs["keep_soft"] = keep_soft

        outputs.update({"tokens": tokens, "attn": attn, "ref_xy": ref_xy})
        return outputs

    # ======================================================================
    # Ground truth handling (shared by the matcher and the losses)
    # ======================================================================

    def _extract_ground_truth(
        self, batch_data_samples: Sequence[Any], device: torch.device
    ) -> List[Dict[str, torch.Tensor]]:
        """Pull centres, sizes, yaws and labels out of whatever we were given.

        Two input forms are accepted:
          * mmdet3d ``Det3DDataSample`` with ``gt_instances_3d`` - the real pipeline
          * a plain dict ``{'boxes': [M, 7], 'labels': [M]}`` - used by the unit test

        Returns one dict per sample with
            center [M, 3], dims [M, 3] (l, w, h), yaw [M], labels [M]

        Two filters are applied, both of which matter on real KITTI data:

        * labels outside [0, num_classes) are dropped. KITTI marks DontCare and
          any class we are not training on with -1. Keeping them would index the
          classification cost matrix out of bounds.
        * boxes whose centre falls outside point_cloud_range are dropped. The
          dataset pipeline usually removes these already, but a box just outside
          the map has no query anywhere near it, so its target offset would be a
          large meaningless number pulling on the regression loss.

        Note on z: mmdet3d stores a box's z at the BOTTOM of the box, while we
        predict the centre. ``gravity_center`` is the converted value, so we use
        that and never ``tensor[:, 2]`` directly.
        """
        ground_truth: List[Dict[str, torch.Tensor]] = []
        x_min, y_min, _, x_max, y_max, _ = self.pc_range

        for index, sample in enumerate(batch_data_samples):
            try:
                if hasattr(sample, "gt_instances_3d"):
                    instances = sample.gt_instances_3d
                    boxes = instances.bboxes_3d
                    center = boxes.gravity_center.to(device=device, dtype=torch.float32)
                    dims = boxes.dims.to(device=device, dtype=torch.float32)
                    yaw = boxes.yaw.to(device=device, dtype=torch.float32)
                    labels = instances.labels_3d.to(device).long()
                elif isinstance(sample, dict):
                    boxes = sample["boxes"].to(device=device, dtype=torch.float32)
                    center, dims, yaw = boxes[:, 0:3], boxes[:, 3:6], boxes[:, 6]
                    labels = sample["labels"].to(device).long()
                else:
                    raise TypeError(
                        f"sample {index} is a {type(sample).__name__}; expected a "
                        "Det3DDataSample or a dict with 'boxes' and 'labels'"
                    )
            except (AttributeError, KeyError) as err:
                raise ValueError(
                    f"could not read ground truth from sample {index}: {err}"
                ) from err

            valid = (labels >= 0) & (labels < self.num_classes)
            valid &= (center[:, 0] >= x_min) & (center[:, 0] <= x_max)
            valid &= (center[:, 1] >= y_min) & (center[:, 1] <= y_max)

            dropped = int((~valid).sum())
            if dropped:
                logger.debug(
                    "sample %d: dropped %d/%d ground-truth boxes "
                    "(ignored class or centre outside the map)",
                    index, dropped, labels.numel(),
                )

            ground_truth.append({
                "center": center[valid],
                "dims": dims[valid],
                "yaw": yaw[valid],
                "labels": labels[valid],
            })

        return ground_truth

    def _box_regression_loss(
        self,
        bbox_raw: torch.Tensor,
        center_xy: torch.Tensor,
        gt: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """L1 loss between one sample's matched boxes and their ground truth.

        Args:
            bbox_raw:  [M, 8] raw predictions of the matched queries
            center_xy: [M, 2] their decoded centres (reference point + offset)
            gt:        the M ground-truth boxes they were matched to
        Returns:
            scalar, summed over the M boxes and all components

        The centre term compares the ABSOLUTE centre, not the offset. This matters:
        ``center_xy = ref_xy + offset``, and ``ref_xy`` comes from the attention
        maps, so the gradient of this term reaches the score network as well as the
        box head. That is what teaches the attention maps to look at real objects
        rather than at empty road. Comparing offsets against a target built from
        ``ref_xy`` would instead let the two sides chase each other.

        The division by offset_scale keeps the centre term in the same numeric
        range as the other components, so no single term dominates.

        Sizes are clamped before the log: a zero-sized annotation does occur in
        KITTI and log(0) would poison the whole batch with -inf.
        """
        gt_center = gt["center"]                                    # [M, 3]
        yaw = gt["yaw"]

        loss_center = F.l1_loss(
            center_xy / self.offset_scale,
            gt_center[:, 0:2] / self.offset_scale,
            reduction="sum",
        )
        loss_height = F.l1_loss(bbox_raw[:, 2], gt_center[:, 2], reduction="sum")
        loss_dims = F.l1_loss(
            bbox_raw[:, 3:6], torch.log(gt["dims"].clamp(min=1e-3)), reduction="sum"
        )
        loss_rotation = F.l1_loss(
            bbox_raw[:, 6:8],
            torch.stack([yaw.sin(), yaw.cos()], dim=-1),
            reduction="sum",
        )

        return loss_center + loss_height + loss_dims + loss_rotation

    # ======================================================================
    # Step 8 - matching and losses
    # ======================================================================

    def _sigmoid_focal_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss, summed over everything. Written out to avoid a mmdet import.

        Args:
            logits:  [*, num_classes] raw scores
            targets: [*, num_classes] 1.0 for the true class, 0.0 elsewhere
        """
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.focal_gamma) * ce

        return loss.sum()

    @torch.no_grad()
    def _match_one_sample(
        self,
        cls_logits: torch.Tensor,
        center_xy: torch.Tensor,
        bbox_raw: torch.Tensor,
        gt: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pair each ground-truth box with exactly one query (Hungarian matching).

        Args:
            cls_logits: [S, num_classes]
            center_xy:  [S, 2] predicted centres, metres
            bbox_raw:   [S, 8] raw predictions (we use the log sizes)
            gt:         one sample's ground truth, M boxes
        Returns:
            (query_indices, gt_indices), both [min(S, M)]

        One-to-one matching is what removes the need for NMS: two queries can never
        be assigned to the same object, so the model is trained not to duplicate.

        No gradient flows through the matching itself - it only decides who is
        compared with whom.
        """
        num_gt = gt["labels"].numel()
        empty = torch.zeros(0, dtype=torch.long, device=cls_logits.device)
        if num_gt == 0:
            return empty, empty

        if not _HAS_SCIPY:
            raise RuntimeError(
                "scipy is required for Hungarian matching "
                "(pip install scipy). It ships with mmdet3d environments."
            )

        # --- classification cost -----------------------------------------
        # Standard DETR focal cost: how much the loss would improve if this query
        # were assigned this label, instead of staying background.
        prob = cls_logits.sigmoid()                                   # [S, K]
        neg_cost = (1 - self.focal_alpha) * (prob ** self.focal_gamma) \
            * -torch.log1p(-prob.clamp(max=1 - 1e-6))
        pos_cost = self.focal_alpha * ((1 - prob) ** self.focal_gamma) \
            * -torch.log(prob.clamp(min=1e-6))
        cost_cls = pos_cost[:, gt["labels"]] - neg_cost[:, gt["labels"]]  # [S, M]

        # --- geometric cost ----------------------------------------------
        # Centres are compared in normalised units so that 1 metre forward and
        # 1 metre sideways count the same regardless of the map's extent.
        pred_center = self._normalize_xy(center_xy)                   # [S, 2]
        gt_center = self._normalize_xy(gt["center"][:, 0:2])          # [M, 2]
        cost_center = torch.cdist(pred_center, gt_center, p=1)        # [S, M]

        pred_log_dims = bbox_raw[:, 3:6]                              # [S, 3]
        gt_log_dims = torch.log(gt["dims"].clamp(min=1e-3))           # [M, 3]
        cost_dims = torch.cdist(pred_log_dims, gt_log_dims, p=1)      # [S, M]

        cost = (
            self.cost_cls_weight * cost_cls
            + self.cost_center_weight * cost_center
            + self.cost_dim_weight * cost_dims
        )

        # A NaN anywhere makes scipy raise a confusing error, so clean up first.
        cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4)

        # scipy runs on CPU and does not accept fp16, so force float32 here. This
        # is the only CPU-GPU sync in the head; it is small (S x M) and happens
        # once per sample.
        query_idx, gt_idx = linear_sum_assignment(cost.detach().float().cpu().numpy())
        device = cls_logits.device
        return (
            torch.as_tensor(query_idx, dtype=torch.long, device=device),
            torch.as_tensor(gt_idx, dtype=torch.long, device=device),
        )

    def loss(
        self,
        fused_flat: torch.Tensor,
        batch_data_samples: Sequence[Any],
    ) -> Dict[str, torch.Tensor]:
        """Run the head and compare its boxes with the ground truth.

        Args:
            fused_flat:         [B, N, C] from the fusion module
            batch_data_samples: one entry per sample (see _extract_ground_truth)
        Returns:
            {'loss_cls': ..., 'loss_bbox': ...} plus 'loss_ref' when
            use_ref_loss is on - mmengine sums these itself

        Stage 2 will add 'loss_budget' here, pushing the gate to keep fewer tokens.
        """
        fused_flat = self._to_flat(fused_flat)
        if len(batch_data_samples) != fused_flat.shape[0]:
            raise ValueError(
                f"batch size mismatch: {fused_flat.shape[0]} feature maps but "
                f"{len(batch_data_samples)} data samples"
            )

        outputs = self(fused_flat)
        return self._compute_losses(outputs, batch_data_samples)

    def _compute_losses(
        self, outputs: Dict[str, torch.Tensor], batch_data_samples: Sequence[Any]
    ) -> Dict[str, torch.Tensor]:
        """The loss itself, split out so it can be tested without a forward pass.

        Everything is cast to fp32 first. Under AMP the head's outputs are fp16,
        and a focal loss summed over B*S*K values in half precision loses
        accuracy quickly; losses are cheap, so there is no reason to keep them
        in half.
        """
        cls_logits = outputs["cls_logits"].float()  # [B, S, K]
        bbox_raw = outputs["bbox_raw"].float()      # [B, S, 8]
        center_xy = outputs["center_xy"].float()    # [B, S, 2]
        ref_xy = outputs["ref_xy"].float()          # [B, S, 2]

        device = cls_logits.device
        batch_size = cls_logits.shape[0]
        ground_truth = self._extract_ground_truth(batch_data_samples, device)

        # Every query starts as background; matching then fills in the real labels.
        cls_targets = torch.zeros_like(cls_logits)
        # STAGE 2: 1 for a query the matcher chose, 0 otherwise. This is what
        # the gate is trained to predict -- "will this token produce a box?"
        gate_targets = torch.zeros_like(cls_logits[..., 0])
        bbox_losses = []
        ref_losses = []
        num_matched = 0

        for b in range(batch_size):
            gt = ground_truth[b]
            query_idx, gt_idx = self._match_one_sample(
                cls_logits[b], center_xy[b], bbox_raw[b], gt
            )
            if query_idx.numel() == 0:
                continue                             # frame with no annotations

            cls_targets[b, query_idx, gt["labels"][gt_idx]] = 1.0
            gate_targets[b, query_idx] = 1.0

            matched_gt = {k: v[gt_idx] for k, v in gt.items()}
            bbox_losses.append(
                self._box_regression_loss(
                    bbox_raw[b, query_idx], center_xy[b, query_idx], matched_gt
                )
            )
            num_matched += query_idx.numel()

            if self.use_ref_loss:
                # CHANGE 2: pull the attention map's centre of mass itself onto
                # the object, so the offset branch cannot do all the work.
                ref_losses.append(F.l1_loss(
                    ref_xy[b, query_idx] / self.offset_scale,
                    matched_gt["center"][:, 0:2] / self.offset_scale,
                    reduction="sum",
                ))

        # Normalise by the number of matched boxes so that a crowded frame does not
        # dominate an empty one. clamp(min=1) guards the no-object case.
        avg_factor = max(num_matched, 1)

        loss_cls = self.loss_cls_weight * self._sigmoid_focal_loss(
            cls_logits, cls_targets
        ) / avg_factor

        if bbox_losses:
            loss_bbox = self.loss_bbox_weight * torch.stack(bbox_losses).sum() / avg_factor
        else:
            # No ground truth anywhere in this batch. Return a real zero that is
            # still attached to the graph, otherwise DDP complains about a
            # parameter that produced no gradient.
            loss_bbox = bbox_raw.sum() * 0.0

        logger.debug(
            "step 8 | matched=%d loss_cls=%.4f loss_bbox=%.4f",
            num_matched, loss_cls.item(), loss_bbox.item(),
        )
        losses = {"loss_cls": loss_cls, "loss_bbox": loss_bbox}
        if self.use_ref_loss:
            if ref_losses:
                losses["loss_ref"] = (
                    self.loss_ref_weight * torch.stack(ref_losses).sum() / avg_factor
                )
            else:
                losses["loss_ref"] = ref_xy.sum() * 0.0   # keep the graph for DDP

        if self.use_token_budget and "keep_soft" in outputs:
            keep_soft = outputs["keep_soft"].float()

            # The gate's own supervision: open for the tokens the matcher
            # chose, closed for the rest. Averaged over every token, so with
            # ~12 cars among 100 tokens the negatives dominate and the gate
            # settles near the number of objects actually in the frame --
            # which is the adaptive behaviour this stage exists for.
            losses["loss_gate"] = self.gate_weight * F.binary_cross_entropy(
                keep_soft.clamp(1e-6, 1 - 1e-6), gate_targets, reduction="mean")

            # Only the EXCESS over the budget is charged, never every open
            # token. Below the budget this is zero and the gate is left to the
            # loss above; above it, this can close even tokens the matcher
            # wanted, and that is what trades accuracy for token count.
            # Charging every token instead pushes uniformly towards zero.
            over_budget = F.relu(keep_soft.sum(dim=1) - float(self.target_tokens))
            losses["loss_budget"] = self.budget_weight * over_budget.mean()

            # Not a loss: mmengine logs every tensor in this dict but only sums
            # the keys containing "loss", so this is reported each iteration
            # without affecting training. It is the number the Stage 2 result
            # table is built from, and the first thing to watch -- if it sits
            # at S the budget is too weak, if it crashes to 1 it is too strong.
            losses["tokens_open"] = (
                outputs["keep"].float().sum(dim=1).mean().detach())

        return losses

    # ======================================================================
    # Step 9 - inference
    # ======================================================================

    def _decode_boxes(self, bbox_raw: torch.Tensor, center_xy: torch.Tensor) -> torch.Tensor:
        """Turn the raw 8 numbers into real boxes.

        Args:
            bbox_raw:  [S, 8]
            center_xy: [S, 2] already includes the offset
        Returns:
            [S, 7] = x, y, z, l, w, h, yaw   (z is the box centre, not its base)

        The size clamp keeps a wild early prediction from producing a kilometre
        long car, which would otherwise break the evaluator.
        """
        height = bbox_raw[:, 2:3]
        dims = bbox_raw[:, 3:6].clamp(min=-5.0, max=5.0).exp()
        yaw = torch.atan2(bbox_raw[:, 6], bbox_raw[:, 7]).unsqueeze(-1)
        return torch.cat([center_xy, height, dims, yaw], dim=-1)

    @torch.no_grad()
    def predict(
        self,
        fused_flat: torch.Tensor,
        batch_data_samples: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        """Produce final detections, one list entry per sample.

        Args:
            fused_flat:         [B, N, C]
            batch_data_samples: unused for now, kept so the call site matches the
                                dense head's signature
        Returns:
            list of InstanceData (or plain dicts when mmdet3d is absent) with
            bboxes_3d, scores_3d, labels_3d

        No NMS. Hungarian matching trained the model to give one box per object,
        so duplicates are not expected; a score threshold is enough.
        """
        outputs = self(fused_flat)

        cls_logits = outputs["cls_logits"].float()
        bbox_raw = outputs["bbox_raw"].float()
        center_xy = outputs["center_xy"].float()

        # Stage 2: tokens whose gate is closed are dropped here as well.
        keep = outputs.get("keep", None)

        results: List[Any] = []
        for b in range(cls_logits.shape[0]):
            scores, labels = cls_logits[b].sigmoid().max(dim=-1)     # [S]

            selected = scores >= self.score_threshold
            if keep is not None:
                selected = selected & (keep[b] > 0.5)

            if selected.sum() > self.max_detections:
                # Keep the strongest ones if too many survive the threshold.
                top = torch.topk(scores.masked_fill(~selected, -1.0),
                                 k=self.max_detections).indices
                selected = torch.zeros_like(selected)
                selected[top] = True

            boxes = self._decode_boxes(bbox_raw[b][selected], center_xy[b][selected])
            results.append(
                self._wrap_results(boxes, scores[selected], labels[selected])
            )

            logger.debug("step 9 | sample %d: %d detections", b, int(selected.sum()))

        return results

    @staticmethod
    def _wrap_results(
        boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
    ) -> Any:
        """Package one sample's detections for the evaluator.

        ``origin=(0.5, 0.5, 0.5)`` tells mmdet3d that our z is the centre of the
        box; it converts to its own bottom-centre convention internally. Getting
        this wrong shifts every box by half its height.
        """
        if not _HAS_MMDET3D:
            return {"bboxes_3d": boxes, "scores_3d": scores, "labels_3d": labels}

        instances = InstanceData()
        instances.bboxes_3d = LiDARInstance3DBoxes(
            boxes, box_dim=7, origin=(0.5, 0.5, 0.5)
        )
        instances.scores_3d = scores
        instances.labels_3d = labels
        return instances