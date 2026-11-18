# Ablation: the learnable token prior removed
#
# Both of these are additions on top of plain TokenLearner (Ryoo et al., 2021),
# which does classification and never needed object positions. They are
# switchable precisely so that they can be removed and measured, and this is
# that measurement.
#
# The comparison is against Stage 1 on this pipeline -- 100 fixed tokens, one
# decoder layer, 20 epochs -- which scored 13.07 / 12.69 / 13.04 Car 3D AP40
# with BOTH additions on. This config is that run with use_token_prior turned off and
# nothing else changed, so the difference is attributable to it alone.
#
# Without the prior the attention is a plain softmax over all 53,568 BEV
# cells, with nothing to keep a token local. That is the configuration the
# first run of this head used, and it measured 5,483 effective cells per
# token against a car's 72 -- each token was the local mean of the map and
# carried no evidence that an object was there. The rotation term sat at
# 1.201, which is its exact no-information value. If that is what the prior
# fixes, this run should be far worse than 12.69.
_base_ = ['./voxelnet_tokenlearner_s100_Kuldeep.py']

work_dir = './work_dirs/voxelnet_tl_abl_noprior'

model = dict(
    bbox_head=dict(
        use_token_prior=False,
    ),
)