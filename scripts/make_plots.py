"""Turn the training logs into the four figures the results rest on.

mmengine records every number but draws nothing, so this reads the .log files
straight out of work_dirs and writes PNGs to figures/.

    python make_plots.py

Reads nothing but logs, writes nothing but figures. Safe to run while training.

Figures:
  1_loss.png     loss_bbox against epoch, every run
  2_ap.png       Car AP40 @ IoU 0.70 (moderate) against epoch, every run
  3_tokens.png   tokens_open against epoch, the Stage 2 runs
  4_tradeoff.png final AP against the number of tokens actually used

Anything missing is skipped with a note rather than crashing, so it works just
as well part-way through a set of runs.
"""

import glob
import os
import re
import sys
from collections import OrderedDict

OUT_DIR = "figures"

# work_dir -> the label that goes in the legend. Order sets the legend order.
RUNS = OrderedDict([
    ("voxelnet_tokenlearner_s100", "Stage 1: 100 fixed, 1 layer, 20 ep"),
    ("voxelnet_tokenlearner_s100_e40", "100 fixed, 1 layer, 40 ep"),
    ("voxelnet_tokenlearner_s100_L3", "100 fixed, 3 layers, 20 ep"),
    ("voxelnet_tl_adaptive", "Stage 2: adaptive, no budget"),
    ("voxelnet_tl_budget10", "Stage 2: budget 10"),
    ("voxelnet_tl_budget5", "Stage 2: budget 5 (collapsed)"),
    ("voxelnet_tl_combined", "Combined: 3 layers + budget 10 + 40 ep"),
    ("voxelnet_tl_abl_noprior", "Ablation: no token prior"),
    ("voxelnet_tl_abl_norefloss", "Ablation: no reference-point loss"),
    ("voxelnet_tl_cleanA", "Clean A: no ref loss, 3 layers, budget 10, 40 ep"),
    ("voxelnet_tl_cleanB", "Clean B: no ref loss, no prior, 3 layers, budget 10, 40 ep"),
])

# The Anchor3DHead reference is READ from its own run rather than written in
# here, so the line on the chart cannot drift away from what that run actually
# scored.
REFERENCE_RUN = ("official_pointpillars_full_run", "VoxelNet + Anchor3DHead")

# Only used when a run's config cannot be found, and reported when it happens.
DEFAULT_NUM_TOKENS = 100

TRAIN_RE = re.compile(
    r"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/\s*(\d+)\]")
VAL_RE = re.compile(r"Epoch\(val\)\s+\[(\d+)\]")
FIELD_RE = re.compile(r"(\w+):\s*(-?[\d.]+(?:e[-+]?\d+)?)")
AP_HEAD_RE = re.compile(r"Car AP40@(0\.\d+)")
AP_3D_RE = re.compile(r"^3d\s+AP40:\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)")


def config_num_tokens(work_dir):
    """How many tokens this run was configured with, from its own config dump.

    Every work_dir keeps a copy of the config it ran with, so the token pool
    is read from there rather than assumed. A run without a gate spends all of
    them, which is what the figures use as its x position.
    """
    for path in glob.glob(os.path.join("work_dirs", work_dir, "*.py")):
        try:
            with open(path, errors="replace") as handle:
                found = re.search(r"num_tokens\s*=\s*(\d+)", handle.read())
            if found:
                return int(found.group(1))
        except OSError:
            continue
    return None


def read_run(work_dir):
    """Pull the curves out of one run's logs.

    A run restarted from a checkpoint leaves several timestamped folders, so
    all of them are read and the results keyed by epoch; a later run of the
    same epoch simply overwrites the earlier one.
    """
    logs = sorted(glob.glob(os.path.join("work_dirs", work_dir, "*", "*.log")))
    if not logs:
        return None

    loss = {}        # epoch -> loss_bbox at the end of that epoch
    tokens = {}      # epoch -> tokens_open at the end of that epoch
    ap70 = {}        # epoch -> (easy, moderate, hard)
    ap50 = {}

    for path in logs:
        epoch = None
        pending = None          # which AP table we are inside
        with open(path, errors="replace") as handle:
            for line in handle:
                train = TRAIN_RE.search(line)
                if train:
                    epoch_now, step, total = (int(g) for g in train.groups())
                    if step == total:          # only the end-of-epoch line
                        fields = dict(FIELD_RE.findall(line))
                        if "loss_bbox" in fields:
                            loss[epoch_now] = float(fields["loss_bbox"])
                        if "tokens_open" in fields:
                            tokens[epoch_now] = float(fields["tokens_open"])
                    continue

                val = VAL_RE.search(line)
                if val:
                    epoch = int(val.group(1))
                    continue

                head = AP_HEAD_RE.search(line)
                if head:
                    pending = head.group(1)
                    continue

                if pending is not None:
                    numbers = AP_3D_RE.match(line.strip())
                    if numbers:
                        values = tuple(float(v) for v in numbers.groups())
                        target = ap70 if pending == "0.70" else ap50
                        target[epoch if epoch is not None else -1] = values
                        pending = None

    if not loss and not ap70:
        return None
    return {"loss": loss, "tokens": tokens, "ap70": ap70, "ap50": ap50}


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; cannot draw anything.")
        return 1

    data = OrderedDict()
    for work_dir, label in RUNS.items():
        parsed = read_run(work_dir)
        if parsed is None:
            print(f"  skipped  {work_dir:36s} (no log found)")
            continue
        data[label] = parsed
        configured = config_num_tokens(work_dir)
        if configured is None:
            configured = DEFAULT_NUM_TOKENS
            print(f"  (no config found in {work_dir}, assuming "
                  f"{DEFAULT_NUM_TOKENS} tokens)")
        parsed["num_tokens"] = configured
        # A gated run logs what it actually kept; an ungated one spends its
        # whole pool on every frame.
        tok = (parsed["tokens"][max(parsed["tokens"])]
               if parsed["tokens"] else float(configured))
        parsed["used"] = tok
        summary = (f"  read     {work_dir:36s} {len(parsed['loss'])} epochs, "
                   f"{len(parsed['ap70'])} evaluations, tokens {tok:5.1f}")
        if parsed["ap70"]:
            summary += f", final AP {parsed['ap70'][max(parsed['ap70'])][1]:6.2f}"
        print(summary)

    if not data:
        print("nothing to plot")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    colours = plt.cm.tab10.colors

    def save(figure, name):
        path = os.path.join(OUT_DIR, name)
        figure.tight_layout()
        figure.savefig(path, dpi=130)
        plt.close(figure)
        print(f"wrote {path}")

    # --- 1. box loss ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, run) in enumerate(data.items()):
        if not run["loss"]:
            continue
        epochs = sorted(run["loss"])
        ax.plot(epochs, [run["loss"][e] for e in epochs],
                label=label, color=colours[i % 10], linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss_bbox")
    ax.set_title("Box regression loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "1_loss.png")

    # --- 2. accuracy ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, run) in enumerate(data.items()):
        if not run["ap70"]:
            continue
        epochs = sorted(e for e in run["ap70"] if e >= 0)
        if not epochs:
            continue
        ax.plot(epochs, [run["ap70"][e][1] for e in epochs],
                marker="o", markersize=4,
                label=label, color=colours[i % 10], linewidth=1.8)
    reference = read_run(REFERENCE_RUN[0])
    if reference and reference["ap70"]:
        value = reference["ap70"][max(reference["ap70"])][1]
        ax.axhline(value, linestyle="--", color="0.4", linewidth=1.2)
        ax.text(0.99, value, f" {REFERENCE_RUN[1]}: {value:.2f}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8, color="0.3")
    else:
        print(f"  (no {REFERENCE_RUN[0]} log, drawing no reference line)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Car 3D AP40 @ IoU 0.70 (moderate)")
    ax.set_title("Detection accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "2_ap.png")

    # --- 3. how many tokens the gate keeps -----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = False
    for i, (label, run) in enumerate(data.items()):
        if not run["tokens"]:
            continue
        epochs = sorted(run["tokens"])
        ax.plot(epochs, [run["tokens"][e] for e in epochs],
                label=label, color=colours[i % 10], linewidth=1.8)
        drawn = True
    if drawn:
        # The pool every one of these runs was configured with, read from their
        # configs rather than assumed. A run without a gate sits on this line.
        pools = {run["num_tokens"] for run in data.values()
                 if run.get("num_tokens")}
        for pool in sorted(pools):
            ax.axhline(pool, linestyle="--", color="0.4", linewidth=1.2)
            ax.text(0.99, pool, f" token pool: {pool} (what Stage 1 spends)",
                    transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=8, color="0.3")
        ax.set_xlabel("epoch")
        ax.set_ylabel("tokens kept per frame")
        ax.set_title("What the gate decides to spend")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        save(fig, "3_tokens.png")
    else:
        plt.close(fig)
        print("  (no Stage 2 run yet, skipping 3_tokens.png)")

    # --- 4. the trade-off ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, run) in enumerate(data.items()):
        if not run["ap70"]:
            continue
        epochs = [e for e in run["ap70"] if e >= 0]
        if not epochs:
            continue
        final_ap = run["ap70"][max(epochs)][1]
        used = run["used"]
        ax.scatter([used], [final_ap], s=90, color=colours[i % 10],
                   zorder=3, label=label)
        ax.annotate(f"{used:.1f} tok\n{final_ap:.1f} AP",
                    (used, final_ap), textcoords="offset points",
                    xytext=(8, -4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("tokens used per frame (log scale)")
    ax.set_ylabel("Car 3D AP40 @ IoU 0.70 (moderate)")
    ax.set_title("Fewer tokens, better detection")
    ax.grid(alpha=0.3, which="both")
    # Lower LEFT is where the collapsed-gate run sits (1 token, low AP), so
    # the legend goes on the right to leave it visible.
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "4_tradeoff.png")

    print(f"\n{len(data)} runs plotted into {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())