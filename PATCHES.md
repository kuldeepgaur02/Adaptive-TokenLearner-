# Changes made to files that already existed

This repository is a **changeset**, not a runnable fork of MMDetection3D. Most
of the work is in new files, listed in the README. One pre-existing file was
also modified, and this document records every change made to it.

That file — `projects/myfusion/fusion/my_fusion.py` — belongs to the lab's
wider fusion project and is not this project's work, so its contents are not
reproduced here. What follows is each change made to it: where it goes, what it
does, and why. Only the lines written for this project are quoted.

---

## `projects/myfusion/fusion/my_fusion.py`

All five changes are inside `class MyFusionTransformer(Base3DDetector)`. Each
is additive and guarded, so every configuration that existed before this
project behaves exactly as it did.

### 1. New method `_format_bbox_head_input` — the Contribution 2 integration point

A dense head such as `CenterHead` wants the BEV map as `[B, C, H, W]` wrapped
in a list. A query head needs the flat `[B, N, C]` tensor, because it uses both
the grid and the flat token list and does the reshape itself. Rather than
branch on the head's class name, the head declares what it wants.

Added as a new method on the class:

```python
def _format_bbox_head_input(self, bev_grid):
    """Give the bbox head the BEV map in the form it expects.

    Dense heads (CenterHead) take the [B, C, H, W] grid wrapped in a list.
    Query heads (TokenLearnerQueryHead, Contribution 2) set
    ``expects_flat_bev = True`` and take the flat [B, N, C] tensor, because
    they need both the grid and the flat list and do the reshape themselves.

    flatten(2).transpose(1, 2) is the exact inverse of the
    permute(0, 2, 1).view(B, C, H, W) in prepare_fusedfeature_for_bbhead,
    so cell n is still row = n // W, col = n % W, the order the head assumes.

    Heads without the attribute (every config before TokenLearner) take the
    else branch, so their behaviour is unchanged.
    """
    if getattr(self.bbox_head, 'expects_flat_bev', False):
        return bev_grid.flatten(2).transpose(1, 2)    # [B, N, C]
    return [bev_grid]                                 # [[B, C, H, W]]
```

`TokenLearnerQueryHead` in `projects/myfusion/fusion/TokenLearner.py` sets
`expects_flat_bev = True`. The inverse-of-the-reshape argument in the docstring
is the part that matters: if the flattening order did not match the reshape the
head performs, every token would read a different BEV cell than the one its
attention map selected, and the reference points would be meaningless.

### 2. `loss()` and `predict()` feed the **fused** output to the head

Both methods previously passed `bbhead_input_lidar` — the LiDAR-only,
pre-fusion branch — into the bbox head, so the fusion module's real output was
computed and then silently discarded on every training step, in every fusion
architecture that had been tested. A gradient-flow diagnostic confirmed it:
`img_backbone` and the fusion module's image-side parameters reported
`grad=None`.

In `loss()`, replacing the old call:

```python
# bbhead_input_fusion is the module's actual fused LiDAR+camera
# output. Every earlier version of this method fed bbhead_input_lidar
# (LiDAR-only, pre-fusion) into bbox_head.loss instead -- meaning the
# fusion module's real output was computed and silently discarded on
# every training step, in every fusion architecture tested
# (TransformerFusionModule, bottleneck fusion, now Cam2BEVFusionModule).
# Confirmed via a direct gradient-flow diagnostic: img_backbone and
# the fusion module's own image-side params showed grad=None with the
# old code, for the exact same reason. See project_running_log.csv,
# 2026-08-26.
fusion_bbox_head_input = self._format_bbox_head_input(bbhead_input_fusion)

losses = self.bbox_head.loss(fusion_bbox_head_input, batch_data_samples)
```

and the matching change in `predict()`:

```python
# See loss() above -- use the actual fused output, not the LiDAR-only
# pre-fusion branch, so predictions reflect what was trained.
fusion_bbox_head_input = self._format_bbox_head_input(bbhead_input_fusion)

raw_predictions3d = self.bbox_head.predict(fusion_bbox_head_input, batch_data_samples)
```

This is the discarded-fusion-output bug discussed in the report.

### 3. `prepare_fusedfeature_for_bbhead` — stop guessing a square BEV grid

The method recovered the grid shape with `H = W = int(N ** 0.5)`. That silently
held for every fusion module tested earlier, because each pooled to a square
grid. It breaks for the geometrically correct, non-square KITTI canvas:
`int(sqrt(53568)) = 231`, and `231 × 231 ≠ 53568`, so the reshape crashes.

Replacing the single `H = W = int(N ** 0.5)` line:

```python
# H=W=sqrt(N) assumes a square grid, which silently held for every
# fusion module tested before 2026-08-24 (512x512 canvas -> square
# 256x256 native, or pooled to square 64x64). It breaks for the
# geometrically-correct, non-square canvas ([496,432] -> native
# 248x216): int(sqrt(53568))=231, but 231*231 != 53568, crash.
# A fusion module can set self.fusion_module.native_hw = (H, W) to
# bypass the guess; falls back to the old behavior otherwise so
# every already-validated experiment is unaffected.
native_hw = getattr(self.fusion_module, 'native_hw', None)
if native_hw is not None:
    H, W = native_hw
else:
    H = W = int(N ** 0.5)  # Assume square grid (32x32)
```

`Cam2BEVFusionModule` sets `self.native_hw = (H, W)` from the LiDAR branch's
own output shape, so the true 248 × 216 grid survives to the head.

### 4. `_forward()` passes `batch_data_samples` to the fusion module

`Cam2BEVFusionModule` needs each sample's calibration to project the image into
BEV. Older fusion modules do not accept the keyword, so the call falls back.

Replacing the single `self.fusion_module(img_feats, lidar_feats)` call:

```python
# Some fusion modules (e.g. Cam2BEVFusionModule) need per-sample
# calibration from batch_data_samples to geometrically project the
# image into BEV; older fusion modules don't accept this kwarg, so
# fall back for backward compatibility.
try:
    lidar_feats_flat, img_feats_flat, fused_feats = self.fusion_module(
        img_feats, lidar_feats, batch_data_samples=batch_data_samples)
except TypeError:
    lidar_feats_flat, img_feats_flat, fused_feats = self.fusion_module(img_feats, lidar_feats)
```

### 5. New `skip_img_branch` argument, and a `None` guard in `predict()`

Every `MyFusionTransformer` run scored exactly 0.0000% Car 3D AP whatever head
was attached, while the same architecture, schedule and data trained through
the official `VoxelNet` class reached real accuracy. Every shared piece had
already been verified identical, so the cause had to be specific to this class.
One concrete, untested difference: it always computes the image branch and
always runs `img_bbox_head` in `predict()`, even when a LiDAR-only fusion
module never uses the result. This flag turns both off so that can be tested.

Added as a constructor argument, defaulting to `False` so no existing config
changes behaviour:

```python
def __init__(self, img_backbone,
            ...
            skip_img_branch=False):
    ...
    self.skip_img_branch = skip_img_branch
```

Used in `extract_feat`:

```python
if self.skip_img_branch:
    img_feats = None
elif img is not None:
    ...
```

and in `predict`:

```python
if self.skip_img_branch or self.img_bbox_head is None:
    raw_predictions2d = [None] * len(raw_predictions3d)
else:
    image_bbox_head_input = [bbhead_input_img]
    raw_predictions2d = self.img_bbox_head.predict(image_bbox_head_input, batch_data_samples)
```

The `self.img_bbox_head is None` half of that condition is also new: the
VoxelNet-line configs build no 2D head at all.

**This change did not fix the problem.** `MyFusionTransformer` still scores
0.0000 with the image branch skipped, which is why every number in the report
comes from the VoxelNet path instead. It is recorded here because it was run,
not because it worked.

---

## Files that were deliberately **not** changed

Worth stating, because a reader would reasonably expect them to have been:

| File | Why no change was needed |
| --- | --- |
| `projects/myfusion/fusion/__init__.py` | It imports `MyFusionTransformer` only. `TokenLearnerQueryHead` is registered through the config's `custom_imports` instead, so no import line had to be added here. |
| `projects/myfusion/__init__.py` | Package glue; untouched. |
| `projects/myfusion/datasets/kitti_fusion_dataset.py` | The calibration matrices `Cam2BEVFusionModule` needs (`lidar2cam`, `cam2img`, `lidar2img`) are already populated by the base `KittiDataset` and carried through by `Pack3DDetInputs`' default `meta_keys`. Confirmed present; no dataset change required. |
| anything under `mmdet3d/` | The rotated-IoU kernel that would not compile was replaced by a **new** file, `projects/myfusion/cpu_rotate_iou.py`, loaded through `custom_imports`. No package was upgraded and no upstream file was edited. |

---

## Prerequisite for running anything from this repository

Because `projects/myfusion/__init__.py` imports `.fusion`, and
`projects/myfusion/fusion/__init__.py` imports `my_fusion`, importing
`projects.myfusion.fusion.TokenLearner` — which is what the configs'
`custom_imports` does — loads `my_fusion.py` as a side effect.

So these files must be in place in the target checkout before any config here
will run:

- `projects/myfusion/__init__.py`
- `projects/myfusion/fusion/__init__.py`
- `projects/myfusion/fusion/my_fusion.py`, with the five changes above applied
- `projects/myfusion/datasets/` (the directory must exist; it is a namespace
  package and is imported by `projects/myfusion/__init__.py`)

All four are already present in the lab checkout this work was done in.
