# TokenLearner query head on the official PointPillars / VoxelNet pipeline.
#
# Why this config exists
# ----------------------
# Every MyFusionTransformer run scores 0.0000% Car 3D AP whatever head it
# carries -- CenterHead included -- which is a known, pre-existing problem in
# that class. Inside it the TokenLearner head cannot be measured at all: its
# losses fall steadily (loss_bbox 4.83 -> 2.66 over 20 epochs, reference-point
# error 5.9 m -> 2.9 m, rotation away from its no-information floor) while AP
# stays pinned at 0.
#
# The same repository's official PointPillars run reaches 85.21 / 72.17 /
# 66.79 Car 3D AP40 through mmdet3d's own VoxelNet class. Putting the same
# head on that pipeline produces a number that means something:
#
#     VoxelNet + Anchor3DHead            85.21 / 72.17 / 66.79   (measured)
#     VoxelNet + TokenLearnerQueryHead     ?                     (this config)
#
# There is no camera branch here. Contribution 2 is about tokenisation, not
# fusion; when MyFusionTransformer is fixed the same head goes back onto it.
#
# The base is the DUMPED config of the 85.21 run, so the backbone, the data
# pipeline and the validation split are exactly the ones that produced that
# number -- nothing is reconstructed from memory.
_base_ = [
    '../../../work_dirs/official_pointpillars_full_run/'
    'pointpillars_kitti_car_noplane.py'
]

custom_imports = dict(
    imports=[
        # Replaces mmdet3d's numba CUDA rotated-IoU kernel, which fails to
        # compile in this environment (NVVM IR 1.6 vs 2.0). Verified exact
        # against this very run: 85.2054 / 72.1725 / 66.7894, matching the
        # original GPU log to four decimals.
        'projects.myfusion.cpu_rotate_iou',
        'projects.myfusion.fusion.TokenLearner',
    ],
    allow_failed_imports=False,
)

# The base is a dump of a FINISHED run, so it carries that run's load_from and
# work_dir. Both must be overridden: otherwise this run would start from the
# Anchor3DHead weights and write over the 85.21 baseline's directory.
load_from = None
resume = False
work_dir = './work_dirs/voxelnet_tokenlearner_s100'

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size = [0.16, 0.16, 4]

model = dict(
    bbox_head=dict(
        # Drop every Anchor3DHead key (anchor_generator, bbox_coder, the three
        # losses, ...); without this mmengine merges them in and the build
        # fails on unexpected arguments.
        _delete_=True,
        type='TokenLearnerQueryHead',
        # SECOND (layer_strides 2, 2, 2) over the 496x432 pillar canvas, then
        # SECONDFPN (upsample_strides 1, 2, 4, out_channels 128 x 3) puts every
        # level back on the stride-2 grid and concatenates them:
        #     [B, 384, 248, 216]
        # Same 248 x 216 grid the head already assumes, but 384 channels wide
        # instead of the fusion module's 256. embed_dims stays 256, so
        # input_proj becomes a real Linear(384, 256) rather than Identity.
        in_channels=384,
        grid_hw=(248, 216),
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        out_size_factor=2,
        num_classes=1,
        num_tokens=100,
        num_decoder_layers=1,
        memory_pool_size=1,
        use_token_prior=True,
        # MEASURED, not guessed: on this grid the token homes sit 5.4 m apart,
        # so a 2.5 m window makes neighbouring priors overlap at ~1 sigma.
        # See fit_token_centres.py. The original 0.15 gave an 11 m window, in
        # which each token averaged ~5,500 BEV cells while a car covers 72.
        prior_sigma=0.036,
        use_ref_loss=True,
    ),
    # Anchor3DHead's assigner and NMS settings. SingleStage3DDetector pushes
    # these into the head's config before building it, whatever they hold.
    # This head does its own Hungarian matching and runs no NMS, so they mean
    # nothing here and are cleared rather than carried along as dead config.
    train_cfg=None,
    test_cfg=None,
)

# The base run used AdamW at lr=1e-3 rising to 1e-2. That suits an anchor head
# and destroys a transformer decoder: the MyFusionTransformer run broke the
# moment its lr crossed 9e-4 -- grad_norm collapsed from 13.7 to 6.9 and the
# model never recovered over the remaining 13 epochs. This run therefore uses
# the schedule that works for this head, which is also the one the
# MyFusionTransformer run used, so the two TokenLearner runs stay directly
# comparable with each other.
#
# This does mean the comparison against 85.21 is best-schedule-per-head rather
# than one shared schedule. A shared one is not available: no single learning
# rate suits both an anchor head and a transformer decoder, and holding the
# schedule fixed would only show which head tolerates the other's.
#
# optim_wrapper is a dict, so this MERGES into the base: betas=(0.95, 0.99)
# and the gradient clip carry over, only lr changes.
lr = 1e-05
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # Weight decay pulls token_centres towards the corner of the map in
            # normalised coordinates, and token_log_sigma towards sigma=1,
            # i.e. no prior at all. With 100 tokens and ~12 cars per frame most
            # tokens get no gradient from the box loss in a given step, so
            # nothing pushes back against the decay.
            'token_centres': dict(decay_mult=0.0),
            'token_log_sigma': dict(decay_mult=0.0),
        }),
)

# A list in a child config REPLACES the parent's list rather than extending it,
# so the momentum entries have to be repeated here even though they are
# unchanged. The base schedule ran to epoch 80 while max_epochs is 20, which
# left the learning rate still climbing when training stopped; these run to 20.
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=8, eta_min=lr * 10,
         begin=0, end=8, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=12, eta_min=lr * 1e-4,
         begin=8, end=20, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=8, eta_min=0.8947368421052632,
         begin=0, end=8, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=12, eta_min=1,
         begin=8, end=20, by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=4)

# The base saves nothing until the very end (interval=-1). Keep every epoch:
# the MyFusionTransformer run's best model was epoch 6 and was lost to
# max_keep_ckpts=3. 20 x ~90 MB is affordable.
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=20))