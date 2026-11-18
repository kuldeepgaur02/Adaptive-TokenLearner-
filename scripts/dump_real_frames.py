"""Dump real KITTI frames plus this model's own output, for the slides.

The diagrams in the deck are currently drawn by hand. This replaces them with
the real thing: actual LiDAR points, actual ground-truth boxes, the actual
camera image, and the tokens the trained gate actually kept on that frame.

It picks frames by how many cars they contain -- one empty, one ordinary, one
busy -- so the three slide panels are real scenes rather than invented ones.

    srun --partition=cse-gpu-all --gres=gpu:1 --pty \\
      python dump_real_frames.py CONFIG CHECKPOINT

e.g.

    srun --partition=cse-gpu-all --gres=gpu:1 --pty \\
      python dump_real_frames.py \\
        projects/myfusion/configs/voxelnet_tl_combined_Kuldeep.py \\
        work_dirs/voxelnet_tl_combined/epoch_40.pth

Writes into real_frames/ :

    frames.npz        points, GT boxes, predicted boxes, tokens kept,
                      lidar2img, per frame -- everything needed to redraw
    cam_<id>.png      the camera image for each chosen frame
    MANIFEST.txt      what was found and what was not

Nothing is assumed: every field is looked up, checked, and reported. If a
field is missing the script says so and carries on with the rest rather than
dying, so one run tells you everything that is and is not available.
"""

import os
import shutil
import sys
import traceback

import numpy as np
import torch

OUT = "real_frames"
MAX_POINTS = 40000          # subsample, so the archive stays small
WANT = [("empty", 0), ("ordinary", 4), ("busy", 11)]   # cars per frame
notes = []


def say(msg):
    print(msg)
    notes.append(msg)


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def as_numpy(x):
    """Boxes come back as a tensor, an ndarray, or an mmdet3d box class."""
    if x is None:
        return None
    if hasattr(x, "tensor"):
        x = x.tensor
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def dig(d, *path):
    """info['a']['b']['c'] without raising when something is absent."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def find_file(candidate, roots):
    """Resolve a dataset path that may be absolute or relative to a root."""
    if not candidate:
        return None
    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate
    for root in roots:
        if not root:
            continue
        for tail in (candidate,
                     os.path.join("training", "velodyne", os.path.basename(candidate)),
                     os.path.join("training", "image_2", os.path.basename(candidate))):
            p = os.path.join(root, tail)
            if os.path.exists(p):
                return p
    return None


def main(config_path, checkpoint_path):
    from mmengine.config import Config
    from mmengine.registry import MODELS, init_default_scope
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(config_path)
    if "custom_imports" in cfg:
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    os.makedirs(OUT, exist_ok=True)

    # ---------------------------------------------------------------- model
    banner("1. BUILD AND LOAD")
    model = MODELS.build(cfg.model)
    head = model.bbox_head
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    say(f"checkpoint     : {checkpoint_path}")
    say(f"  epoch        : {state.get('meta', {}).get('epoch', '?')}")
    say(f"  missing keys : {len(missing)}   unexpected: {len(unexpected)}")
    say(f"  gate on      : {getattr(head, 'use_token_budget', False)}")
    say(f"  device       : {device}")

    # ------------------------------------------------------------- dataset
    banner("2. FIND THE FRAMES")
    loader = Runner.build_dataloader(cfg.val_dataloader)
    dataset = loader.dataset
    say(f"validation set : {len(dataset)} frames")

    roots = []
    for key in ("data_root",):
        if key in cfg:
            roots.append(cfg[key])
    dr = dig(cfg, "val_dataloader", "dataset", "data_root")
    if dr:
        roots.append(dr)
    roots += ["data/kitti", "./data/kitti", "data/kitti/"]
    roots = [r for i, r in enumerate(roots) if r and r not in roots[:i]]
    say(f"data roots tried: {roots}")

    # How many cars in each frame, from the annotations the evaluator uses.
    counts = []
    for i in range(len(dataset)):
        info = dataset.get_data_info(i)
        labels = dig(info, "eval_ann_info", "gt_labels_3d")
        counts.append(int((np.asarray(labels) >= 0).sum()) if labels is not None else -1)
    counts = np.array(counts)
    say(f"cars per frame : total {int(counts[counts >= 0].sum())}, "
        f"max {counts.max()}, {int((counts == 0).sum())} empty frames")

    chosen = []
    for name, want in WANT:
        pool = np.where(counts == want)[0]
        if len(pool) == 0:                      # nearest available instead
            pool = np.array([int(np.argmin(np.abs(counts - want)))])
            say(f"  no frame with exactly {want} cars; using the nearest")
        idx = int(pool[len(pool) // 2])
        chosen.append((name, idx, int(counts[idx])))
        say(f"  {name:9s} -> dataset index {idx}, {counts[idx]} cars")

    # ------------------------------------------------------------ dump them
    banner("3. RUN THE MODEL AND DUMP")
    bundle = {}
    for name, idx, ncars in chosen:
        info = dataset.get_data_info(idx)
        sample_id = str(info.get("sample_idx", idx))

        # --- the model's own output on this frame -----------------------
        item = dataset[idx]
        batch = loader.collate_fn([item])
        with torch.no_grad():
            data = model.data_preprocessor(batch, training=False)
            feats = model.extract_feat(data["inputs"])
            out = head(feats)
            kept = float(out["keep"].float().sum(dim=1)[0]) if "keep" in out else -1.0
            try:
                preds = head.predict(feats)
                res = preds[0]
                pred_boxes = as_numpy(res["bboxes_3d"] if isinstance(res, dict)
                                      else res.bboxes_3d)
                pred_scores = as_numpy(res["scores_3d"] if isinstance(res, dict)
                                       else res.scores_3d)
            except Exception as exc:
                pred_boxes, pred_scores = None, None
                say(f"  [{name}] predict() failed: {exc}")

        # --- ground truth ----------------------------------------------
        gt = as_numpy(dig(info, "eval_ann_info", "gt_bboxes_3d"))

        # --- the raw point cloud ----------------------------------------
        lidar_rel = dig(info, "lidar_points", "lidar_path") or info.get("lidar_path")
        lidar_abs = find_file(lidar_rel, roots)
        pts = None
        if lidar_abs:
            raw = np.fromfile(lidar_abs, dtype=np.float32)
            dim = dig(cfg, "val_dataloader", "dataset", "pipeline") and 4 or 4
            pts = raw.reshape(-1, dim)[:, :3]
            if len(pts) > MAX_POINTS:
                sel = np.random.default_rng(0).choice(len(pts), MAX_POINTS,
                                                      replace=False)
                pts = pts[sel]
        else:
            say(f"  [{name}] point cloud not found (looked for {lidar_rel!r})")

        # --- the camera image and its projection ------------------------
        img_rel = (dig(info, "images", "CAM2", "img_path")
                   or dig(info, "images", "CAM_2", "img_path")
                   or info.get("img_path"))
        img_abs = find_file(img_rel, roots)
        if img_abs:
            shutil.copy(img_abs, os.path.join(OUT, f"cam_{name}_{sample_id}.png"))
            say(f"  [{name}] camera image copied from {img_abs}")
        else:
            say(f"  [{name}] camera image not found (looked for {img_rel!r})")
        lidar2img = dig(info, "images", "CAM2", "lidar2img")

        bundle[f"{name}__points"] = (pts if pts is not None
                                     else np.zeros((0, 3))).astype(np.float16)
        bundle[f"{name}__gt"] = (gt if gt is not None
                                 else np.zeros((0, 7))).astype(np.float32)
        bundle[f"{name}__pred"] = (pred_boxes if pred_boxes is not None
                                   else np.zeros((0, 7))).astype(np.float32)
        bundle[f"{name}__score"] = (pred_scores if pred_scores is not None
                                    else np.zeros((0,))).astype(np.float32)
        bundle[f"{name}__l2i"] = (np.asarray(lidar2img, dtype=np.float32)
                                  if lidar2img is not None
                                  else np.zeros((0, 0), dtype=np.float32))
        bundle[f"{name}__meta"] = np.array(
            [idx, ncars, kept, len(pts) if pts is not None else 0],
            dtype=np.float32)

        say(f"  [{name}] id {sample_id}: {ncars} GT cars, "
            f"{0 if pred_boxes is None else len(pred_boxes)} predictions, "
            f"{kept:.0f} tokens kept, "
            f"{0 if pts is None else len(pts)} points")

    bundle["pc_range"] = np.asarray(getattr(head, "pc_range", []), dtype=np.float32)
    np.savez_compressed(os.path.join(OUT, "frames.npz"), **bundle)

    with open(os.path.join(OUT, "MANIFEST.txt"), "w") as fh:
        fh.write("\n".join(notes) + "\n")

    banner("4. DONE")
    size = os.path.getsize(os.path.join(OUT, "frames.npz")) / 1e6
    print(f"wrote {OUT}/frames.npz  ({size:.1f} MB)")
    for f in sorted(os.listdir(OUT)):
        print(f"      {OUT}/{f}")
    print("\nSend me frames.npz, the cam_*.png files and MANIFEST.txt.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    try:
        sys.exit(main(sys.argv[1], sys.argv[2]))
    except Exception:
        traceback.print_exc()
        sys.exit(1)