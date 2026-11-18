"""Where are the cars actually, and where should the 100 token homes sit?

The diagnostic showed only 18 of 100 tokens ever match a car, because the homes
were spread evenly over the whole 69 x 79 m map while KITTI cars sit in a much
narrower region. This script measures the real distribution and places the homes
on it with k-means, instead of guessing.

It reads ground truth through the TRAINING dataloader, so the distribution
includes the augmentation the model actually sees (flip, global rotation,
database sampling) - not the raw annotations.

Run (CPU is fine, no GPU needed):

    cd ~/mmdetection3d
    python fit_token_centres.py

Writes:
    token_centres_init.pt   normalised [S, 2] homes, to be loaded by the head
    token_centres_init.png  scatter plot (cars + chosen homes), if matplotlib
                            is available

Nothing is trained and no existing file is touched.
"""

import sys
import traceback

import numpy as np
import torch

CFG = "projects/myfusion/configs/my_fusion_transformer_cam2bev_tokenlearner_s100.py"
NUM_TOKENS = 100
MAX_BATCHES = 400          # ~1600 frames, ~15k cars. Enough, and takes ~1 min.
OUT_PT = "token_centres_init.pt"
OUT_PNG = "token_centres_init.png"
SEED = 0


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def kmeans(points: torch.Tensor, k: int, iters: int = 100, seed: int = 0):
    """Lloyd's algorithm with k-means++ seeding. Pure torch, no sklearn.

    Args:
        points: [P, 2]
        k: number of clusters
    Returns:
        centres [k, 2]
    """
    generator = torch.Generator().manual_seed(seed)
    num_points = points.shape[0]

    # --- k-means++ seeding: spread the initial centres out, otherwise Lloyd's
    # converges to a poor local optimum and leaves clusters empty.
    centres = torch.empty(k, 2, dtype=points.dtype)
    first = torch.randint(num_points, (1,), generator=generator).item()
    centres[0] = points[first]
    closest = ((points - centres[0]) ** 2).sum(-1)
    for i in range(1, k):
        probs = closest.clamp(min=0)
        total = probs.sum()
        if total <= 0:
            pick = torch.randint(num_points, (1,), generator=generator).item()
        else:
            pick = torch.multinomial(probs / total, 1, generator=generator).item()
        centres[i] = points[pick]
        closest = torch.minimum(closest, ((points - centres[i]) ** 2).sum(-1))

    # --- Lloyd's iterations
    for step in range(iters):
        dist = torch.cdist(points, centres)              # [P, k]
        assign = dist.argmin(dim=1)
        moved = 0.0
        for i in range(k):
            members = points[assign == i]
            if members.numel() == 0:
                # Empty cluster: move it onto the worst-served point so that
                # every one of the k tokens stays useful.
                worst = dist.min(dim=1).values.argmax()
                new = points[worst]
            else:
                new = members.mean(dim=0)
            moved = max(moved, float((new - centres[i]).abs().sum()))
            centres[i] = new
        if moved < 1e-4:
            print(f"  k-means converged after {step + 1} iterations")
            break
    return centres


def mean_nearest_distance(points: torch.Tensor, centres: torch.Tensor) -> float:
    """Mean L1 distance from each point to its nearest centre.

    This is the floor on the reference-point error: a token's reference point
    cannot start closer to a car than its home is.
    Chunked, because points x centres can be large.
    """
    totals = []
    for start in range(0, points.shape[0], 4096):
        chunk = points[start:start + 4096]
        dist = torch.cdist(chunk, centres, p=1)
        totals.append(dist.min(dim=1).values)
    return float(torch.cat(totals).mean())


def main():
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(CFG)
    if "custom_imports" in cfg:
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    pc_range = cfg.model.bbox_head.pc_range
    x_min, y_min, _, x_max, y_max, _ = pc_range
    span = torch.tensor([x_max - x_min, y_max - y_min], dtype=torch.float32)
    origin = torch.tensor([x_min, y_min], dtype=torch.float32)

    banner("1. COLLECT CAR CENTRES FROM THE TRAINING DATALOADER")
    loader = Runner.build_dataloader(cfg.train_dataloader)

    collected = []
    frames = 0
    for index, batch in enumerate(loader):
        if index >= MAX_BATCHES:
            break
        for sample in batch["data_samples"]:
            boxes = sample.gt_instances_3d.bboxes_3d
            labels = sample.gt_instances_3d.labels_3d
            if len(boxes) == 0:
                frames += 1
                continue
            keep = labels >= 0
            if keep.any():
                xy = boxes.gravity_center[:, 0:2][keep]
                collected.append(xy.float().cpu())
            frames += 1
        if (index + 1) % 100 == 0:
            done = sum(c.shape[0] for c in collected)
            print(f"  {index + 1}/{MAX_BATCHES} batches, {done} cars so far")

    if not collected:
        print("No ground-truth boxes found. Check the config's train_dataloader.")
        return

    cars = torch.cat(collected)                      # [P, 2] in metres
    print(f"\n{cars.shape[0]} car centres from {frames} frames "
          f"({cars.shape[0] / max(frames, 1):.1f} per frame)")

    banner("2. WHERE ARE THEY?")
    for axis, name, lo, hi in ((0, "x (forward)", x_min, x_max),
                               (1, "y (lateral)", y_min, y_max)):
        values = cars[:, axis]
        qs = torch.quantile(
            values, torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))
        print(f"{name}:  map covers {lo:7.2f} .. {hi:7.2f} m")
        print(f"              cars    1% {qs[0]:7.2f}   5% {qs[1]:7.2f}   "
              f"25% {qs[2]:7.2f}   50% {qs[3]:7.2f}")
        print(f"                     75% {qs[4]:7.2f}  95% {qs[5]:7.2f}  "
              f"99% {qs[6]:7.2f}")
        used = (qs[6] - qs[0]) / (hi - lo)
        print(f"              cars occupy {100 * used:.0f}% of the map along "
              f"this axis\n")

    banner("3. THE CURRENT (UNIFORM GRID) HOMES")
    import math
    cols = math.ceil(math.sqrt(NUM_TOKENS))
    rows = math.ceil(NUM_TOKENS / cols)
    u_x = (torch.arange(cols, dtype=torch.float32) + 0.5) / cols
    u_y = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    gy, gx = torch.meshgrid(u_y, u_x, indexing="ij")
    grid_norm = torch.stack([gx, gy], -1).reshape(-1, 2)[:NUM_TOKENS]
    grid_m = grid_norm * span + origin

    grid_err = mean_nearest_distance(cars, grid_m)
    dist_to_grid = torch.cdist(cars, grid_m, p=1)
    grid_used = dist_to_grid.argmin(dim=1).unique().numel()
    print(f"mean distance from a car to its nearest home : {grid_err:6.2f} m (L1)")
    print(f"homes that are ever the nearest one          : {grid_used}/{NUM_TOKENS}")
    print("\n(the diagnostic measured a 6.39 m reference-point error and")
    print(" 18 active tokens, so these two numbers should look familiar)")

    banner("4. FIT NEW HOMES WITH K-MEANS")
    torch.manual_seed(SEED)
    centres_m = kmeans(cars, NUM_TOKENS, seed=SEED)

    km_err = mean_nearest_distance(cars, centres_m)
    dist_to_km = torch.cdist(cars, centres_m, p=1)
    km_used = dist_to_km.argmin(dim=1).unique().numel()
    print(f"mean distance from a car to its nearest home : {km_err:6.2f} m (L1)")
    print(f"homes that are ever the nearest one          : {km_used}/{NUM_TOKENS}")
    print(f"\nimprovement: {grid_err:.2f} m -> {km_err:.2f} m  "
          f"({100 * (1 - km_err / grid_err):.0f}% lower)")

    banner("5. HOW WIDE SHOULD THE PRIOR BE?")
    centres_norm = (centres_m - origin) / span
    pair = torch.cdist(centres_norm, centres_norm)
    pair.fill_diagonal_(float("inf"))
    nearest = pair.min(dim=1).values
    spacing = float(nearest.median())
    sigma = spacing / 2.0

    print(f"median spacing between neighbouring homes : {spacing:.4f} "
          f"normalised  (~{spacing * float(span.mean()):.1f} m)")
    print(f"suggested prior_sigma                     : {sigma:.4f} "
          f"(half the spacing, so neighbours overlap at ~1 sigma)")
    print(f"                                          = "
          f"{sigma * (x_max - x_min):.1f} m across x, "
          f"{sigma * (y_max - y_min):.1f} m across y")
    print(f"\ncurrent value is 0.15 (10.4 m x, 11.9 m y)")
    area_ratio = (sigma / 0.15) ** 2
    print(f"this shrinks each token's window to {100 * area_ratio:.1f}% of its "
          f"current area,")
    print(f"so ~5483 effective cells becomes roughly "
          f"{int(5483 * area_ratio)} (a car is ~72)")

    banner("6. SAVE")
    torch.save({
        "centres_norm": centres_norm,
        "centres_m": centres_m,
        "suggested_sigma": sigma,
        "num_tokens": NUM_TOKENS,
        "pc_range": list(pc_range),
        "num_cars": int(cars.shape[0]),
    }, OUT_PT)
    print(f"wrote {OUT_PT}  (normalised [{NUM_TOKENS}, 2] homes + sigma)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sub = cars[torch.randperm(cars.shape[0])[:8000]]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        for ax, homes, title, err, used in (
                (axes[0], grid_m, "current: uniform grid", grid_err, grid_used),
                (axes[1], centres_m, "proposed: k-means on cars", km_err, km_used)):
            ax.scatter(sub[:, 1], sub[:, 0], s=1, alpha=0.12,
                       color="#4a7fb5", label="car centres")
            ax.scatter(homes[:, 1], homes[:, 0], s=45, marker="x",
                       color="#c2410c", linewidths=1.6, label="token homes")
            ax.set_title(f"{title}\n{err:.2f} m mean error, "
                         f"{used}/{NUM_TOKENS} homes used")
            ax.set_xlabel("y, lateral (m)")
            ax.set_ylabel("x, forward (m)")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", markerscale=3)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=110)
        print(f"wrote {OUT_PNG}")
    except Exception as err:  # matplotlib missing or headless trouble
        print(f"(no plot: {err})")

    banner("DONE - paste sections 2 to 5")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)