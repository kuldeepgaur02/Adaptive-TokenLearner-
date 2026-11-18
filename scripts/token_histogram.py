"""How many tokens does the gate actually spend, frame by frame?

The training log reports an average -- 9.7 tokens. An average alone does not
show the method is adaptive: a gate that spent exactly 9.7 tokens on every
frame would report the same number and would not be adaptive at all. What
makes the case is the SPREAD: few tokens on an empty road, more in traffic.

This runs a trained checkpoint over the validation set and records, per frame,
how many tokens the gate kept and how many cars were actually there. It writes:

    figures/5_token_histogram.png   the distribution of tokens per frame
    figures/6_tokens_vs_cars.png    tokens kept against cars present
    figures/token_counts.csv        the raw numbers

Usage (needs a GPU; about five minutes on the 1,481-frame val set):

    python token_histogram.py CONFIG CHECKPOINT [LABEL]

e.g.

    python token_histogram.py \\
        projects/myfusion/configs/voxelnet_tl_budget10_Kuldeep.py \\
        work_dirs/voxelnet_tl_budget10/epoch_20.pth \\
        "budget 10"

The second figure is the one that proves the point. If tokens rise with the
number of cars, the gate is reading the scene. If it is flat, the gate has
only learned a constant and "adaptive" would be the wrong word for it.
"""

import os
import sys
import traceback

import torch

OUT_DIR = "figures"

# Car in KITTI's full category list (Pedestrian, Cyclist, Car, Van, ...).
# Only used as a fallback; see count_cars.
CAR_LABEL = 2


def banner(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def count_cars(dataset, index):
    """Ground-truth cars in one frame, or None if they cannot be read.

    Two sources, in order of trust:

    1. ``eval_ann_info`` -- the annotations the evaluator itself uses, already
       filtered to the dataset's own classes. With classes=['Car'] every label
       there is a Car, so the count is just how many there are.

    2. ``instances`` -- the raw annotations. Their ``bbox_label_3d`` indexes
       the FULL KITTI category list (Pedestrian, Cyclist, Car, Van, Truck,
       ...), NOT the dataset's class list, so Car is 2 and not 0. Counting 0
       here returns pedestrians: on this val split that is 850 against the
       5,646 cars actually present, and a correlation computed from it is
       meaningless. The total is checked against the dataset's own figure
       before any of it is used.
    """
    try:
        info = dataset.get_data_info(index)
    except Exception:
        return None
    if not isinstance(info, dict):
        return None

    ann = info.get("eval_ann_info")
    if isinstance(ann, dict) and ann.get("gt_labels_3d") is not None:
        labels = torch.as_tensor(ann["gt_labels_3d"])
        return int((labels >= 0).sum())

    instances = info.get("instances")
    if isinstance(instances, list):
        return sum(1 for item in instances
                   if item.get("bbox_label_3d", -1) == CAR_LABEL)
    return None


def main(config_path, checkpoint_path, label):
    from mmengine.config import Config
    from mmengine.registry import MODELS, init_default_scope
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(config_path)
    if "custom_imports" in cfg:
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    banner("1. BUILD AND LOAD")
    model = MODELS.build(cfg.model)
    head = model.bbox_head
    if not getattr(head, "use_token_budget", False):
        print("This checkpoint has no token gate -- it spends every token on "
              "every frame, so there is no distribution to plot.")
        return 1

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"checkpoint : {checkpoint_path}")
    print(f"  epoch    : {state.get('meta', {}).get('epoch', '?')}")
    print(f"  missing  : {len(missing)}   unexpected: {len(unexpected)}")
    print(f"  device   : {device}")
    print(f"  budget   : {head.target_tokens} tokens, "
          f"weight {head.budget_weight}")

    banner("2. RUN OVER THE VALIDATION SET")
    loader = Runner.build_dataloader(cfg.val_dataloader)
    dataset = loader.dataset
    print(f"{len(dataset)} frames")

    tokens, cars = [], []
    index = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            data = model.data_preprocessor(batch, training=False)
            feats = model.extract_feat(data["inputs"])
            out = head(feats)
            per_frame = out["keep"].float().sum(dim=1).cpu().tolist()

            for value in per_frame:
                tokens.append(value)
                cars.append(count_cars(dataset, index))
                index += 1

            if (step + 1) % 200 == 0:
                print(f"  {index}/{len(dataset)} frames")

    tokens_t = torch.tensor(tokens)
    have_cars = [c for c in cars if c is not None]

    # Verify the ground-truth count against the dataset's own figure before
    # drawing any conclusion from it. An earlier version of this script read
    # the wrong label and counted 850 pedestrians as cars; the correlation it
    # produced looked plausible and was meaningless. Better to refuse the
    # number than to present a wrong one.
    cars_ok = False
    if have_cars:
        counted = sum(have_cars)
        expected = None
        try:
            expected = sum(
                1
                for i in range(len(dataset))
                for item in (dataset.get_data_info(i).get("instances") or [])
                if item.get("bbox_label_3d", -1) == CAR_LABEL
            )
        except Exception:
            pass
        if expected and abs(counted - expected) <= max(1, 0.01 * expected):
            cars_ok = True
            print(f"ground-truth check : counted {counted} cars, "
                  f"dataset has {expected} -- agrees")
        else:
            print(f"ground-truth check : counted {counted} cars but the "
                  f"dataset has {expected}. These do not agree, so the "
                  f"cars-versus-tokens figure is NOT trustworthy and is "
                  f"skipped. The token histogram is unaffected.")

    banner("3. WHAT THE GATE SPENT")
    print(f"frames            : {len(tokens)}")
    print(f"tokens per frame  : mean {tokens_t.mean():.2f}   "
          f"min {tokens_t.min():.0f}   max {tokens_t.max():.0f}   "
          f"std {tokens_t.std():.2f}")
    print(f"distinct counts   : {len(tokens_t.unique())}")
    pool = head.num_tokens        # read from the head, not written in here
    print(f"token pool        : {pool}  (what Stage 1 spends on every frame)")
    print(f"against Stage 1   : {tokens_t.mean():.2f} of {pool}, "
          f"a {100 * (1 - tokens_t.mean() / pool):.1f}% reduction in tokens "
          f"that produce a box")
    print()
    if float(tokens_t.std()) < 0.5:
        print("  The spread is almost nothing. The gate has settled on a")
        print("  constant rather than reading each frame, so this is a smaller")
        print("  fixed count, not an adaptive one. Worth saying plainly.")
    else:
        print("  The count varies frame to frame, which is what makes this")
        print("  adaptive rather than simply a smaller fixed number.")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "token_counts.csv"), "w") as handle:
        handle.write("frame,tokens_kept,cars_present\n")
        for i, (t, c) in enumerate(zip(tokens, cars)):
            handle.write(f"{i},{t:.0f},{'' if c is None else c}\n")
    print(f"\nwrote {OUT_DIR}/token_counts.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; numbers written, no figures")
        return 0

    # --- histogram -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    high = int(tokens_t.max()) + 2
    ax.hist(tokens, bins=range(0, high), color="#4a7fb5",
            edgecolor="white", align="left")
    ax.axvline(float(tokens_t.mean()), color="#c2410c", linestyle="--",
               linewidth=1.6,
               label=f"mean {tokens_t.mean():.1f}")
    ax.axvline(head.target_tokens, color="0.4", linestyle=":", linewidth=1.4,
               label=f"budget {head.target_tokens}")
    ax.set_xlabel("tokens kept in a frame")
    ax.set_ylabel("frames")
    ax.set_title(f"Tokens spent per frame — {label}\n"
                 f"(Stage 1 spends all {pool} on every one of these frames)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "5_token_histogram.png"), dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/5_token_histogram.png")

    # --- tokens against cars -------------------------------------------
    if len(have_cars) < 10 or not cars_ok:
        print("  (ground-truth counts unusable, skipping 6_tokens_vs_cars)")
        return 0

    pairs = [(c, t) for c, t in zip(cars, tokens) if c is not None]
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=10, alpha=0.25, color="#4a7fb5",
               label="one frame")

    # Mean tokens at each car count, which is easier to read than the cloud.
    by_count = {}
    for c, t in pairs:
        by_count.setdefault(c, []).append(t)
    keys = sorted(k for k in by_count if len(by_count[k]) >= 5)
    if keys:
        ax.plot(keys, [sum(by_count[k]) / len(by_count[k]) for k in keys],
                color="#c2410c", linewidth=2, marker="o", markersize=4,
                label="mean (counts with 5+ frames)")

    ax.set_xlabel("cars present in the frame")
    ax.set_ylabel("tokens kept")
    ax.set_title(f"Does the gate read the scene? — {label}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "6_tokens_vs_cars.png"), dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/6_tokens_vs_cars.png")

    # Pearson's r, written out rather than taken from statistics.correlation,
    # which only exists from Python 3.10 and the cluster runs 3.8.
    if len(xs) > 2:
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        if var_x > 0 and var_y > 0:
            r = cov / (var_x ** 0.5 * var_y ** 0.5)
            print(f"\ncorrelation between cars present and tokens kept: {r:+.3f}")
            if r > 0.3:
                print("  The gate is responding to the scene: more cars, more")
                print("  tokens. This is the sentence the slide is for.")
            elif r > 0.1:
                print("  A weak relationship. The gate varies, but only")
                print("  loosely with how much is actually in the frame.")
            else:
                print("  Essentially no relationship. The count varies, but not")
                print("  with the number of cars, so 'adaptive' overstates it.")
                print("  Better to say so than to have it asked.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    try:
        sys.exit(main(sys.argv[1], sys.argv[2],
                      sys.argv[3] if len(sys.argv) > 3 else "trained gate"))
    except Exception:
        traceback.print_exc()
        sys.exit(1)