# Stage 2: adaptive token count with no budget pressure.
#
# Stage 1 spends all 100 tokens on every frame, on an empty stretch of road as
# much as in a traffic jam, and reached 13.07 / 12.69 / 13.04 Car 3D AP40 on
# this pipeline. Stage 2 puts a small gate on each token, read from the pooled
# token BEFORE the decoder, so a closed token never enters cross-attention --
# that is where the saving comes from.
#
# Two forces act on the gate:
#
#   loss_gate   = BCE(gate, "did the matcher choose this token?")
#   loss_budget = budget_weight * relu(open_tokens - target_tokens)
#
# Left alone, the first one settles the gate near the number of objects in the
# frame, which is the adaptive behaviour this stage is for. The second only
# bites when the budget is set BELOW that natural count, and then it forces the
# gate to give something up -- which is where accuracy starts to be traded for
# token count. Measured on the gate smoke test: free, the gate held 19 tokens
# for 12 objects; squeezed, it went to 5.
#
# These three configs run in parallel:
#
#   adaptive (this file's family)  target 100, weight 0.0   the natural count
#   budget10                       target  10, weight 0.1   a mild squeeze
#   budget5                        target   5, weight 0.5   a hard squeeze
#
# Together with Stage 1's fixed 100 tokens they give the curve this
# contribution is about: how far the token count can fall before the accuracy
# does.
#
# FIRST THING TO CHECK once a run starts: "tokens_open" in the log. It starts
# at 100 and should settle near the number of cars in a frame, so roughly 12-20. If it sits at 100 the gate is not
# learning; if it crashes to 1 something is pushing far too hard.
_base_ = ['./voxelnet_tokenlearner_s100_Kuldeep.py']

work_dir = './work_dirs/voxelnet_tl_adaptive'

model = dict(
    bbox_head=dict(
        use_token_budget=True,
        target_tokens=100,
        budget_weight=0.0,
    ),
)