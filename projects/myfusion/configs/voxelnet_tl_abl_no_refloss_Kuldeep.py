# Ablation: the reference-point loss removed
#
# Both of these are additions on top of plain TokenLearner (Ryoo et al., 2021),
# which does classification and never needed object positions. They are
# switchable precisely so that they can be removed and measured, and this is
# that measurement.
#
# The comparison is against Stage 1 on this pipeline -- 100 fixed tokens, one
# decoder layer, 20 epochs -- which scored 13.07 / 12.69 / 13.04 Car 3D AP40
# with BOTH additions on. This config is that run with use_ref_loss turned off and
# nothing else changed, so the difference is attributable to it alone.
#
# Without this loss, nothing pulls the attention map's centre of mass onto
# the object: the offset branch can absorb the whole centre error on its own,
# and the attention maps never need to move. The claim being tested is that
# the box loss alone is not enough to teach the score network where to look.
_base_ = ['./voxelnet_tokenlearner_s100_Kuldeep.py']

work_dir = './work_dirs/voxelnet_tl_abl_norefloss'

model = dict(
    bbox_head=dict(
        use_ref_loss=False,
    ),
)