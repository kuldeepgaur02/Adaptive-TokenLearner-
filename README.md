# Adaptive Token Budgeting for BEV 3D Object Detection

The number of queries in a DETR-style 3D detection head is normally a
hyperparameter, fixed before training and applied unchanged to every frame.
This project makes it a **per-frame decision of the model**.

On the KITTI validation split that takes Car 3D AP40 (moderate, IoU 0.70) from
**12.69 to 34.13** while cutting the token count from **100 to 4.24 per
frame**, and the resulting count correlates **r = 0.82** with the number of
cars actually present in the scene.

**Author:** Kuldeep Raj Gour · Indian Institute of Technology Hyderabad
**Report:** see the accompanying project report for the full method and analysis.

---

## Results

All numbers are Car 3D AP40, moderate difficulty, IoU 0.70, on the KITTI
validation split (1,481 frames, 5,646 cars). Each row is a single training run.

| Configuration | Tokens/frame | AP40 |
| --- | ---: | ---: |
| Head inside the fusion transformer | 100 | 0.00 |
| **Stage 1** — 100 fixed tokens, 1 layer, 20 epochs | 100 | 12.69 |
| + 40 epochs | 100 | 20.63 |
| + 3 decoder layers | 100 | 22.33 |
| Gate, no budget | 10.8 | 21.47 |
| Gate, budget T=10 | 9.7 | 25.75 |
| Gate, budget T=5 (collapses) | 1.0 | 3.72 |
| **Proposed** — 3 layers, 40 epochs, gate T=10 | **9.2** | **34.13** |
| VoxelNet + Anchor3DHead (reference) | — | 72.17 |

Token counts for gated rows are the training-time mean; on validation the
proposed model uses 4.24.

---

## What is in this repository

The work was built inside
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d), in a lab checkout
that also carries the lab's own fusion project. This repository is therefore a
**changeset, not a runnable fork**: it holds the files written for this project
plus a record of the changes made to one file that already existed. Upstream
MMDetection3D is unmodified and is not reproduced here, and neither is the
lab's own work.

```
projects/myfusion/
    fusion/TokenLearner.py      the query head and the token gate
    fusion/cam2bev.py           camera -> BEV projection, earlier fusion attempt
    cpu_rotate_iou.py           CPU replacement for the evaluator's IoU kernel
    configs/                    12 configs: proposed, baselines, ablations
scripts/                        training, diagnostics and analysis
baseline/                       the anchor-head reference config
PATCHES.md                      every change made to a pre-existing file
```

### Where each file goes in MMDetection3D

| In this repository | Copy to |
| --- | --- |
| `projects/myfusion/fusion/TokenLearner.py` | `projects/myfusion/fusion/` |
| `projects/myfusion/fusion/cam2bev.py` | `projects/myfusion/fusion/` |
| `projects/myfusion/cpu_rotate_iou.py` | `projects/myfusion/` |
| `projects/myfusion/configs/*.py` | `projects/myfusion/configs/` |
| `scripts/*` | repository root |
| `baseline/pointpillars_kitti_car_noplane.py` | `work_dirs/official_pointpillars_full_run/` |

The baseline config must keep that exact path, because every config here
reaches it through `_base_` as a relative path:

```python
_base_ = ['../../../work_dirs/official_pointpillars_full_run/'
          'pointpillars_kitti_car_noplane.py']
```

It is kept in `baseline/` in this repository only because `work_dirs/` is a
runtime output directory and is git-ignored; one `cp` puts it back, and the
command is in *Reproducing the main result* below.

### The camera → BEV module

`projects/myfusion/fusion/cam2bev.py` is the camera–LiDAR fusion module from
the attempt that preceded the query head, included because one config here uses
it. Every earlier fusion module in the lab directory treated the camera branch
as an unstructured set of tokens, pooled and projected with no regard to 3D
geometry, so the head received a spatially meaningless camera signal however
correctly the rest was wired — a heatmap-alignment diagnostic put the
classification peak a median 8 m from any real car even after every geometry
and loss-normalisation bug had been fixed. This module instead projects each
LiDAR BEV cell's real-world position into the image with the sample's own
calibration and bilinearly samples the image feature map there, producing a
camera feature map on the *same* grid as the LiDAR features. It did not rescue
the fusion path — those runs still score 0.00 — and the reported results are
LiDAR-only.

### Changes to a file that already existed

One pre-existing file, `projects/myfusion/fusion/my_fusion.py`, was modified
for this project. It belongs to the lab's wider fusion project, so its contents
are not reproduced here. **[PATCHES.md](PATCHES.md)** records every change made
to it — five additive, guarded changes — along with the files that were
deliberately left alone and why, and the prerequisites for running anything
here.

### Attribution

Everything in this repository was written for this project except
`baseline/pointpillars_kitti_car_noplane.py`, which is a stock MMDetection3D
config, included only because the other configs inherit from it and will not
run without it. The lab's own files in the same directory are described in
`PATCHES.md` where this project touched them, and are otherwise not reproduced.

---

## The method

Everything new lives in `projects/myfusion/fusion/TokenLearner.py`.

### 1. TokenLearner as a query head

The backbone gives a bird's-eye-view feature map of 248 × 216 cells (53,568 in
all, 0.32 m each, 384 channels). A small network produces S = 100 spatial
attention maps over it; each map is pooled to a single vector, and those
vectors become the decoder's object queries. Each map's centroid is also
converted to BEV metres and used as that query's **reference point**, so the
decoder refines an offset from a data-dependent starting position rather than
regressing a position from nothing.

### 2. The token gate — the contribution

A two-layer MLP reads each pooled token and emits one logit; a sigmoid turns
it into a keep probability and a threshold at 0.5 turns that into a hard
keep-or-drop decision. The number of surviving queries is therefore decided
per frame, by the model.

A threshold has zero gradient everywhere, so a **straight-through estimator**
carries the hard decision forward and the smooth gradient backward:

```python
keep = hard + soft - soft.detach()     # value is hard, gradient is soft
```

Two practical details: if a frame closes every token the attention becomes
non-finite, so the strongest token is reopened — on validation that floor is
reached exactly on frames with no objects. And at inference the closed tokens
are gathered out before the decoder runs, so the decoder genuinely processes
fewer queries; this path was checked against the masked path and the outputs
agree to 4.6e-7.

### 3. Two losses

**Gate supervision.** Hungarian matching already determines which queries were
assigned to a ground-truth object, so that is reused directly as the gate's
target: binary cross-entropy against "was this token matched?". The
supervision is free, because the matching was already being computed.

**Budget.** A one-sided hinge against a budget T: opening more than T tokens
is penalised, opening fewer costs nothing. That asymmetry is deliberate — it
lets the gate fall below the budget on sparse frames, which it does, all the
way down to the one-token floor on empty ones.

An earlier design instead suppressed the classification logits of closed
tokens. That failed in an instructive way: focal loss already pushes unmatched
queries towards background, so suppressing a closed token's logits reduced the
classification loss for free, independently of the budget. With T = 60 the
gate collapsed to 11.8 tokens while the budget loss sat at exactly 0.0000 —
the budget was never the active term. Decoupling the gate from the
classification head removed the shortcut.

---

## Configs

### The proposed method

| Config | What it is |
| --- | --- |
| `voxelnet_tl_combined_Kuldeep.py` | **the main result**: 3 decoder layers, 40 epochs, gate at budget 10 → **34.13** |

### Baselines

| Config | What it is | AP40 |
| --- | --- | ---: |
| `baseline/pointpillars_kitti_car_noplane.py` | anchor-head reference (stock MMDetection3D, not written here) | 72.17 |
| `voxelnet_tokenlearner_s100_Kuldeep.py` | Stage 1: 100 fixed tokens, 1 layer, 20 epochs — the baseline the contribution is measured against | 12.69 |
| `voxelnet_tokenlearner_s100_e40_Kuldeep.py` | Stage 1 at 40 epochs | 20.63 |
| `voxelnet_tokenlearner_s100_L3_Kuldeep.py` | Stage 1 with 3 decoder layers | 22.33 |
| `my_fusion_transformer_cam2bev_tokenlearner_s100.py` | the earlier camera–LiDAR attempt, through `MyFusionTransformer` with `cam2bev.py` as the fusion module; kept because it carries the measured attention width and the corrected learning-rate schedule | 0.00 |

### Ablations

| Config | What is removed or changed | AP40 |
| --- | --- | ---: |
| `voxelnet_tl_adaptive_Kuldeep.py` | gate with no budget at all | 21.47 |
| `voxelnet_tl_budget10_Kuldeep.py` | gate at budget 10 | 25.75 |
| `voxelnet_tl_budget5_Kuldeep.py` | gate at budget 5 — collapses to the one-token floor | 3.72 |
| `voxelnet_tl_abl_no_prior_Kuldeep.py` | token prior removed, 1 decoder layer | 13.95 |
| `voxelnet_tl_abl_no_refloss_Kuldeep.py` | reference-point loss removed, 1 decoder layer | 22.88 |
| `voxelnet_tl_cleanA_Kuldeep.py` | reference-point loss removed, full setup | 27.53 |
| `voxelnet_tl_cleanB_Kuldeep.py` | both additions removed, full setup | 27.78 |

**The ablation that reverses.** With one decoder layer, removing the
reference-point loss *improves* accuracy by 10.19 AP (12.69 → 22.88). Run
again on top of the full configuration, removing it *costs* 6.60 AP (34.13 →
27.53). With a single decoder layer a query gets one shot at its box, so
pinning its starting position is pure constraint; with three layers it can
refine across passes and a good starting point pays for itself. An ablation is
only valid in the configuration it was run in.

---

## Scripts

Each of these was written to answer a specific question, and each produced a
number that appears in the report.

| Script | Why it exists | What it produced |
| --- | --- | --- |
| `cpu_rotate_iou.py` | The KITTI evaluator computes rotated IoU — the overlap of two tilted 3D boxes — in a numba CUDA kernel that would not compile in this environment. Rather than change any package version, that one function was reimplemented on CPU | **Without it no AP could be measured at all.** Checked against a run with known numbers it returned 85.2054 / 72.1725 / 66.7894, reproducing them exactly. Every AP in this project comes from the verified implementation |
| `diagnose_tokenlearner.py` | The box loss sat flat for three epochs with no error. This decomposes it per term to find which part was stuck | Showed the size term learning while the rotation term did not — which points at the tokens, not the regressor, because size has a guessable mean and rotation does not. Found each token was effectively averaging **5,483 BEV cells** against the 72 a car occupies |
| `fit_token_centres.py` | The attention width had been set by assumption. This measures what it should be from the data | **σ = 0.036**, from 19,734 ground-truth car positions, against the 0.15 that had been guessed. Reference error fell from 6.4 m to 3.9 m |
| `smoke_test_budget.py` | A gate can be wired up wrongly in ways that look fine for hours. This checks the mechanism on CPU in about a minute: that the decision is genuinely hard, that gradient reaches the gate, that no sample closes every token, that nothing goes non-finite, and that a tighter budget really does leave fewer tokens | **Caught the first gate design in 20 minutes** instead of after a 15-hour training run |
| `token_histogram.py` | A mean token count proves nothing on its own: a gate spending exactly 4.24 tokens on every frame would report the same mean. This records the count per frame against the objects actually present | Range **1 to 14** tokens, and correlation **r = +0.82** with the true car count — the evidence that the budget is adaptive rather than merely smaller. It cross-checks its own object count against the dataset's figure (5,646 both ways) before trusting the correlation |
| `make_plots.py` | mmengine records every number but draws nothing | The loss, AP, token-count and trade-off curves |
| `dump_real_frames.py` | The report's figures should show real output, not illustrations | Real LiDAR points, ground-truth and predicted boxes, camera projections and token counts for three validation frames |
| `train_any.sh` | SLURM submission; takes a config path and resumes automatically | Every training run |

---

## Reproducing the main result

Copy the files into an MMDetection3D checkout at the paths given above, put the
baseline config where `_base_` expects to find it, then submit:

```bash
mkdir -p work_dirs/official_pointpillars_full_run
cp baseline/pointpillars_kitti_car_noplane.py work_dirs/official_pointpillars_full_run/

sbatch train_any.sh projects/myfusion/configs/voxelnet_tl_combined_Kuldeep.py
```

The changes recorded in [PATCHES.md](PATCHES.md) all sit inside
`MyFusionTransformer`, which the `voxelnet_*` configs do not use as their
detector. They are listed there because they were part of the work, and because
the camera–LiDAR config does need them. Its *Prerequisite* section is the part
that applies to every config here: `my_fusion.py` and the two `__init__.py`
files must exist in the target checkout, because importing
`projects.myfusion.fusion.TokenLearner` loads them as a side effect.

About 30 hours on one GPU. Validation runs every fourth epoch and prints Car
3D AP40 at IoU 0.70 and 0.50. The run converges: epoch 36 gives 34.21 and
epoch 40 gives 34.13.

Before starting a long run, check the gate mechanism in about a minute:

```bash
python smoke_test_budget.py
```

Afterwards, for the token statistics behind the adaptivity claim:

```bash
python token_histogram.py \
  projects/myfusion/configs/voxelnet_tl_combined_Kuldeep.py \
  work_dirs/voxelnet_tl_combined/epoch_40.pth
```
