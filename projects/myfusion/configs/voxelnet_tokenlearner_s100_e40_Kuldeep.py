# Run A: the same head and the same pipeline, trained for 40 epochs instead
# of 20.
#
# Why
# ---
# The 20-epoch run was still climbing when its schedule ran out:
#
#     epoch  4   AP40@0.70 moderate   0.89
#     epoch  8                        0.59
#     epoch 12                        5.27
#     epoch 16                       12.48
#     epoch 20                       12.69   <- lr had decayed to 1e-9 by here
#
# The flat stretch from 16 to 20 is the learning rate going to zero, not the
# model saturating: loss_bbox fell 4.30 -> 2.11 across the same span. So 13 AP
# is where training stopped, not where the head tops out.
#
# This is expected for a query head. An anchor head starts from boxes that are
# already 3.9 x 1.6 m in roughly the right places; a DETR-style head has to
# learn position and size from nothing, which is why the original DETR needs
# 300-500 epochs on COCO. 20 is very short.
#
# Everything except the length of training is inherited unchanged, so any
# difference is attributable to the epoch count alone.
_base_ = ['./voxelnet_tokenlearner_s100_Kuldeep.py']

work_dir = './work_dirs/voxelnet_tokenlearner_s100_e40'

# The one-cycle shape is kept and simply stretched: the rise still occupies the
# first 40% of training and the decay the remaining 60%, exactly as at 20
# epochs. Only the endpoints move (8/20 -> 16/40), so the peak learning rate
# and the curve the model experiences are the same, just slower.
#
# A list in a child config replaces the parent's list rather than extending
# it, so all four entries are repeated.
lr = 1e-05
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=16, eta_min=lr * 10,
         begin=0, end=16, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=24, eta_min=lr * 1e-4,
         begin=16, end=40, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=16, eta_min=0.8947368421052632,
         begin=0, end=16, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=24, eta_min=1,
         begin=16, end=40, by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=4)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=40))