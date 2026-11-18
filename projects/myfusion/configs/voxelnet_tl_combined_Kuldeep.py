# Everything that helped, in one run: 3 decoder layers + the token gate with a
# budget of 10 + 40 epochs.
#
# Each was measured separately against the same Stage 1 baseline, which scored
# 12.69 Car 3D AP40 (moderate) with 100 fixed tokens, one decoder layer and
# 20 epochs:
#
#     40 epochs                100 tokens   20.63   (+7.9)
#     3 decoder layers         100 tokens   22.33   (+9.6)
#     gate, budget 10          9.7 tokens   25.75   (+13.1)
#
# The three act on different things -- how long the model trains, how many
# chances a query gets to correct its box, and how many queries produce a box
# at all -- so there is reason to expect them to add rather than overlap. This
# run is the test of that. If they do not add, that is worth knowing too: it
# would mean they were all fixing the same underlying problem.
#
# Cost: three decoder layers run at ~1.33 s/iter against ~1.0, so 40 epochs is
# about 30 hours. That fits the 48-hour limit on train_any.sh with room to
# spare.
_base_ = ['./voxelnet_tokenlearner_s100_Kuldeep.py']

work_dir = './work_dirs/voxelnet_tl_combined'

model = dict(
    bbox_head=dict(
        # From the 3-layer run. One layer gives each query a single pass at its
        # box; the extra layers let it look at the map again and correct
        # itself, which is what the IoU-0.70 score is short of.
        num_decoder_layers=3,
        # From the budget-10 run, which was the best of the three. Left alone
        # the gate settles near the number of cars in the frame (10.8); a
        # budget of 10 nudges it slightly below that, to 9.7, and scored
        # higher than the free gate. Not 5: that weight collapsed the gate to
        # the one-token floor and AP fell to 3.72.
        use_token_budget=True,
        target_tokens=10,
        budget_weight=0.1,
    ),
)

# From the 40-epoch run. The one-cycle shape is unchanged, only stretched: the
# rise still takes the first 40% of training and the decay the rest.
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