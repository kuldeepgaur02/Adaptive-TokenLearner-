# The best of everything measured so far, with the reference-point loss dropped.
#
# What the ablations showed, all against Stage 1's 12.69 Car 3D AP40
# (moderate) with 100 fixed tokens, one decoder layer and 20 epochs:
#
#     use_ref_loss   = False      19.94   (+7.25, final)
#     use_token_prior= False      14.65   (+1.96 at epoch 16, run unfinished)
#
# Both of those were additions on top of plain TokenLearner, and both were
# written to fix the same thing: attention that had spread itself over all
# 53,568 BEV cells, with every token reduced to a local mean (5,483 effective
# cells against a car's 72). Two things have changed since. The prior's width
# was measured from the data rather than guessed, and the head moved onto the
# VoxelNet pipeline, whose BEV features are far better than the fusion module
# that was producing 0.0000 AP. The attention now lands on cars on its own, so
# what used to hold it there is now just holding it back.
#
# This config therefore drops the reference-point loss, keeps the token prior, and keeps everything that did help:
#
#     3 decoder layers      +9.6    (more chances to correct a box)
#     token gate, budget 10 +13.1   (9.7 tokens instead of 100)
#     40 epochs             +7.9    (the 20-epoch runs were still climbing)
#
# Two variants run side by side -- this one and the other -- because the two
# removals might not be independent: each may have been compensating for the
# other, in which case dropping both could be worse than dropping either.
# Guessing which is cheaper to avoid than to be wrong about.
_base_ = ['./voxelnet_tokenlearner_s100_Kuldeep.py']

work_dir = './work_dirs/voxelnet_tl_cleanA'

model = dict(
    bbox_head=dict(
        num_decoder_layers=3,
        use_token_budget=True,
        target_tokens=10,
        budget_weight=0.1,
        # Dropped: measured at +7.25 AP without it.
        use_ref_loss=False,
        # Kept, at the width measured from the data.
        use_token_prior=True,
    ),
)

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