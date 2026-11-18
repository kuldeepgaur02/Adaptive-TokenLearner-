"""Why is loss_bbox flat at ~5.0?

Loads epoch_2.pth, runs ONE training batch through the head, and prints the
numbers that decide between the competing explanations. Read-only: nothing is
trained, nothing is written.

Run on a GPU node (NOT the login node), while the real job keeps running:

    srun -p cse-gpu-all --gres=gpu:1 --time=00:30:00 --cpus-per-task=4 --pty bash
    cd ~/mmdetection3d
    python diagnose_tokenlearner.py

Everything it prints comes with the number to compare against, so the output
can be pasted back as-is.
"""

import sys
import traceback

import torch

CFG = "projects/myfusion/configs/my_fusion_transformer_cam2bev_tokenlearner_s100.py"
CKPT = "work_dirs/my_fusion_transformer_cam2bev_tokenlearner_s100/epoch_20.pth"


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main():
    from mmengine.config import Config
    from mmengine.registry import MODELS, init_default_scope
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(CFG)

    # custom_imports normally runs inside Runner; we build the model by hand,
    # so the project modules have to be imported explicitly or the registry
    # will not know TokenLearnerQueryHead / Cam2BEVFusionModule.
    if "custom_imports" in cfg:
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    banner("1. BUILD + LOAD")
    model = MODELS.build(cfg.model)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"checkpoint: {CKPT}")
    print(f"  epoch in checkpoint : {ckpt.get('meta', {}).get('epoch', '?')}")
    print(f"  missing keys        : {len(missing)}")
    print(f"  unexpected keys     : {len(unexpected)}")
    if missing:
        print(f"  (first few missing) : {missing[:5]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    head = model.bbox_head
    print(f"  device              : {device}")

    banner("2. ONE TRAINING BATCH")
    loader = Runner.build_dataloader(cfg.train_dataloader)
    raw = next(iter(loader))
    data = model.data_preprocessor(raw, training=True)
    inputs, samples = data["inputs"], data["data_samples"]
    print(f"batch size: {len(samples)}")

    with torch.no_grad():
        _, _, fused = model._forward(inputs, samples)
        flat = model._format_bbox_head_input(fused)
        out = head(flat)

    attn = out["attn"].float()           # [B, S, N]
    ref_xy = out["ref_xy"].float()       # [B, S, 2]
    center_xy = out["center_xy"].float()  # [B, S, 2]
    bbox_raw = out["bbox_raw"].float()   # [B, S, 8]
    cls_logits = out["cls_logits"].float()
    B, S, N = attn.shape
    print(f"attn {tuple(attn.shape)}  ref_xy {tuple(ref_xy.shape)}")

    # ------------------------------------------------------------------
    banner("3. ARE THE ATTENTION MAPS SHARP OR BLURRED?")
    # Participation ratio: 1 / sum(w^2). For a map concentrated on k cells with
    # equal weight this equals k. A car covers ~72 cells of this grid.
    eff = 1.0 / (attn ** 2).sum(-1)
    print(f"effective cells per token : mean {eff.mean():8.0f}   "
          f"min {eff.min():8.0f}   max {eff.max():8.0f}")
    print(f"largest single cell weight: {attn.max().item():.3e}  "
          f"(uniform over {N} cells would be {1.0 / N:.3e})")
    print()
    print("  ~72      -> maps sit on one car. Ideal.")
    print("  ~500     -> workable.")
    print("  >5000    -> each token is the local mean of the map; the car's")
    print("              signal is diluted ~100x and the token carries almost")
    print("              no evidence that an object is there.")

    # ------------------------------------------------------------------
    banner("4. HAVE THE TOKEN PRIORS MOVED SINCE INIT?")
    if head.use_token_prior:
        import math
        sigma = head.token_log_sigma.detach().float().exp().cpu()
        print(f"sigma     : mean {sigma.mean():.4f}  min {sigma.min():.4f}  "
              f"max {sigma.max():.4f}   (init 0.15)")
        x_min, y_min, _, x_max, y_max, _ = head.pc_range
        print(f"            = {sigma.mean() * (x_max - x_min):.1f} m across x, "
              f"{sigma.mean() * (y_max - y_min):.1f} m across y")

        cols = math.ceil(math.sqrt(head.num_tokens))
        rows = math.ceil(head.num_tokens / cols)
        u_x = (torch.arange(cols, dtype=torch.float32) + 0.5) / cols
        u_y = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
        gy, gx = torch.meshgrid(u_y, u_x, indexing="ij")
        init = torch.stack([gx, gy], -1).reshape(-1, 2)[: head.num_tokens]

        now = head.token_centres.detach().float().cpu()
        drift = (now - init).abs().sum(-1)
        span_m = torch.tensor([x_max - x_min, y_max - y_min])
        drift_m = ((now - init) * span_m).abs().sum(-1)
        print(f"centre drift from init grid: mean {drift_m.mean():.2f} m   "
              f"max {drift_m.max():.2f} m")
        print()
        print("  <1 m  -> the token homes are frozen; ref_xy is just the init")
        print("           grid, and the matcher only ever picks the nearest home.")
        print("  >5 m  -> the homes are genuinely learning where cars are.")
    else:
        print("use_token_prior is OFF")

    # ------------------------------------------------------------------
    banner("5. WHERE DOES loss_bbox ACTUALLY GO?")
    gts = head._extract_ground_truth(samples, attn.device)

    tot = {"center": 0.0, "height": 0.0, "dims": 0.0, "rotation": 0.0}
    ref_err_m, ctr_err_m, off_m, n_box = [], [], [], 0

    for b in range(B):
        gt = gts[b]
        if gt["labels"].numel() == 0:
            continue
        qi, gi = head._match_one_sample(
            cls_logits[b], center_xy[b], bbox_raw[b], gt)
        if qi.numel() == 0:
            continue
        n_box += qi.numel()

        mgt = {k: v[gi] for k, v in gt.items()}
        br = bbox_raw[b, qi]
        cx = center_xy[b, qi]
        gc = mgt["center"]
        yaw = mgt["yaw"]

        import torch.nn.functional as F
        s = head.offset_scale
        tot["center"] += F.l1_loss(cx / s, gc[:, 0:2] / s, reduction="sum").item()
        tot["height"] += F.l1_loss(br[:, 2], gc[:, 2], reduction="sum").item()
        tot["dims"] += F.l1_loss(
            br[:, 3:6], torch.log(mgt["dims"].clamp(min=1e-3)),
            reduction="sum").item()
        tot["rotation"] += F.l1_loss(
            br[:, 6:8], torch.stack([yaw.sin(), yaw.cos()], -1),
            reduction="sum").item()

        ref_err_m.append((ref_xy[b, qi] - gc[:, 0:2]).abs().sum(-1))
        ctr_err_m.append((cx - gc[:, 0:2]).abs().sum(-1))
        off_m.append((br[:, 0:2] * s).abs().sum(-1))

    print(f"ground-truth cars in this batch : {n_box}")
    if n_box == 0:
        print("no matched boxes in this batch - rerun to get another one")
        return

    print(f"per-box loss terms (x{head.loss_bbox_weight} = the logged loss_bbox):")
    w = head.loss_bbox_weight
    total = sum(tot.values()) / n_box
    for k, v in tot.items():
        per = v / n_box
        print(f"  {k:9s} {per:7.3f}   -> contributes {per * w:6.3f} "
              f"({100 * per / total:4.1f}% of loss_bbox)")
    print(f"  {'TOTAL':9s} {total:7.3f}   -> loss_bbox {total * w:6.3f} "
          f"(logged value is ~5.0)")

    ref_err_m = torch.cat(ref_err_m)
    ctr_err_m = torch.cat(ctr_err_m)
    off_m = torch.cat(off_m)
    print()
    print(f"reference point -> gt car : {ref_err_m.mean():6.2f} m  (L1, |dx|+|dy|)")
    print(f"final centre    -> gt car : {ctr_err_m.mean():6.2f} m")
    print(f"offset branch moved it by : {off_m.mean():6.2f} m   (max {off_m.max():.2f})")
    print()
    print("  If 'offset branch moved it by' is near 0 while the centre error is")
    print("  several metres, the regression branch is dead and every centre is")
    print("  just the attention map's centre of mass.")
    print("  If the offset is large but the centre error did not shrink, the")
    print("  offset is firing in the wrong direction - a matching problem.")

    # ------------------------------------------------------------------
    banner("6. IS THE MATCHING STABLE?")
    # Same batch, matched twice: identical by construction. What matters is how
    # many DIFFERENT tokens win across the frames of this batch - if the same
    # few tokens always win, the head has collapsed onto a positional prior.
    winners = []
    for b in range(B):
        gt = gts[b]
        if gt["labels"].numel() == 0:
            continue
        qi, _ = head._match_one_sample(
            cls_logits[b], center_xy[b], bbox_raw[b], gt)
        winners.append(qi.cpu())
    if winners:
        allw = torch.cat(winners)
        print(f"matched queries: {allw.numel()} matches across {len(winners)} frames, "
              f"{allw.unique().numel()} distinct tokens out of {S}")
        probs = cls_logits.sigmoid()
        print(f"score of matched queries  : {probs.flatten(0, 1)[0, 0]:.3f} (sample)")
        print(f"max score anywhere        : {probs.max():.3f}")
        print(f"mean score over all tokens: {probs.mean():.3f}")

    banner("DONE - paste everything above")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)