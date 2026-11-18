"""Does the Stage 2 token gate actually work?

Checks the things that fail silently and would otherwise cost a 15-hour run:

  1. the gate is genuinely hard (0 or 1), not a soft blend
  2. gradient reaches the gate network at all
  3. no sample ends up with every token closed (that makes attention NaN)
  4. the budget loss falls and the token count actually drops over a few steps
  5. nothing goes NaN or inf
  6. a larger lambda ends with fewer tokens open -- the whole premise

Runs on CPU in about a minute, on random data. It says nothing about
accuracy; it only says the mechanism is wired up.

    python smoke_test_budget.py
"""

import sys
import traceback

import torch

sys.path.insert(0, ".")


def banner(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def make_head(num_tokens=100, budget_weight=0.01, target_tokens=30, grid=(24, 20)):
    """A small head: the real grid would make this test slow for no benefit."""
    from projects.myfusion.fusion.TokenLearner import TokenLearnerQueryHead

    return TokenLearnerQueryHead(
        in_channels=64,
        embed_dims=64,
        num_tokens=num_tokens,
        grid_hw=grid,
        pc_range=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
        voxel_size=(0.16, 0.16, 4.0),
        out_size_factor=2,
        num_classes=1,
        num_heads=4,
        ffn_dims=128,
        alpha_groups=8,
        use_token_prior=True,
        prior_sigma=0.036,
        use_ref_loss=True,
        use_token_budget=True,
        budget_weight=budget_weight,
        target_tokens=target_tokens,
    )


def fake_batch(head, batch=2, cars=12, seed=0):
    """A random BEV map and some ground-truth boxes inside the map."""
    generator = torch.Generator().manual_seed(seed)
    feats = torch.randn(batch, head.num_cells, head.in_channels,
                        generator=generator)

    x_min, y_min, _, x_max, y_max, _ = head.pc_range
    samples = []
    for _ in range(batch):
        centre = torch.stack([
            torch.rand(cars, generator=generator) * (x_max - x_min) + x_min,
            torch.rand(cars, generator=generator) * (y_max - y_min) + y_min,
            torch.rand(cars, generator=generator) * 2 - 1.5,
        ], dim=1)
        dims = torch.stack([
            3.5 + torch.rand(cars, generator=generator),
            1.5 + torch.rand(cars, generator=generator) * 0.3,
            1.4 + torch.rand(cars, generator=generator) * 0.3,
        ], dim=1)
        yaw = (torch.rand(cars, generator=generator) * 2 - 1) * 3.14159
        samples.append({
            "boxes": torch.cat([centre, dims, yaw[:, None]], dim=1),
            "labels": torch.zeros(cars, dtype=torch.long),
        })
    return feats, samples


def run(head, feats, samples, steps=40, lr=1e-3):
    """Train the toy head for a few steps and record what the gate does."""
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr)
    history = []
    for step in range(steps):
        losses = head.loss(feats, samples)
        total = sum(v for k, v in losses.items() if "loss" in k)

        optimiser.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 35.0)
        optimiser.step()

        history.append({
            "step": step,
            "total": float(total),
            "budget": float(losses.get("loss_budget", torch.tensor(0.0))),
            "cls": float(losses["loss_cls"]),
            "open": float(losses["tokens_open"]),
        })
    return history


def main():
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"{'PASS' if passed else 'FAIL'}  {name:46s} {detail}")

    banner("1. BUILD")
    head = make_head()
    print(f"tokens={head.num_tokens}  cells={head.num_cells}  "
          f"budget_weight={head.budget_weight}  target={head.target_tokens}")
    check("gate network exists", hasattr(head, "gate_mlp"))
    check("budget is on", head.use_token_budget)

    feats, samples = fake_batch(head)

    banner("2. ONE FORWARD PASS")
    out = head(feats)
    keep = out["keep"]
    soft = out["keep_soft"]
    print(f"keep {tuple(keep.shape)}  soft {tuple(soft.shape)}")

    # Not an exact equality: keep = hard + soft - soft.detach() is hard in
    # exact arithmetic, but in float32 the two soft terms leave ~1e-7 behind,
    # so a kept token reads 0.9999999 rather than 1.0. What matters is that
    # the value is one or the other and not a blend, so round and compare.
    hard_error = float((keep - keep.round()).abs().max())
    check("keep is hard (0 or 1)", hard_error < 1e-5,
          f"max deviation from 0/1 = {hard_error:.1e}")
    check("soft is a probability", bool((soft >= 0).all() and (soft <= 1).all()),
          f"[{soft.min():.3f}, {soft.max():.3f}]")
    per_sample = keep.sum(1)
    check("no sample is fully closed", bool((per_sample > 0).all()),
          f"open per sample {per_sample.tolist()}")
    check("starts nearly all open", bool(per_sample.min() > 0.8 * head.num_tokens),
          f"{per_sample.tolist()} of {head.num_tokens}")
    check("no NaN in outputs",
          not any(torch.isnan(v).any() for v in out.values()
                  if isinstance(v, torch.Tensor)))

    banner("3. DOES GRADIENT REACH THE GATE?")
    losses = head.loss(feats, samples)
    print("loss keys:", sorted(losses))
    check("loss_budget present", "loss_budget" in losses)
    check("tokens_open present", "tokens_open" in losses)
    check("tokens_open is not summed into the total",
          "loss" not in "tokens_open")

    total = sum(v for k, v in losses.items() if "loss" in k)
    head.zero_grad()
    total.backward()

    gate_grads = [p.grad for p in head.gate_mlp.parameters() if p.grad is not None]
    gate_norm = sum(float(g.norm()) for g in gate_grads)
    check("gate network receives gradient", gate_norm > 0, f"|grad| = {gate_norm:.3e}")
    check("gate gradient is finite",
          all(torch.isfinite(g).all() for g in gate_grads))

    alpha_norm = sum(float(p.grad.norm()) for p in head.alpha.parameters()
                     if p.grad is not None)
    check("score network still receives gradient", alpha_norm > 0,
          f"|grad| = {alpha_norm:.3e}")

    banner("4. DOES THE TOKEN COUNT ACTUALLY FALL?")
    torch.manual_seed(0)
    head = make_head(budget_weight=0.01, target_tokens=30)
    hist = run(head, feats, samples, steps=40)
    for h in hist[::8] + [hist[-1]]:
        print(f"  step {h['step']:3d}   open {h['open']:6.1f}   "
              f"budget {h['budget']:8.4f}   cls {h['cls']:7.4f}   "
              f"total {h['total']:8.4f}")
    first, last = hist[0]["open"], hist[-1]["open"]
    check("token count moved", abs(last - first) > 1e-6,
          f"{first:.1f} -> {last:.1f}")
    check("token count fell", last < first, f"{first:.1f} -> {last:.1f}")
    check("all losses stayed finite",
          all(torch.isfinite(torch.tensor(h["total"])) for h in hist))

    banner("5. DOES LAMBDA CONTROL THE COUNT?")
    results = {}
    for weight in (0.0, 0.01, 0.1):
        torch.manual_seed(0)
        h = make_head(budget_weight=weight, target_tokens=30)
        hist = run(h, feats, samples, steps=40)
        results[weight] = hist[-1]["open"]
        print(f"  lambda = {weight:<5}  ->  {hist[-1]['open']:6.1f} tokens open "
              f"after 40 steps")
    check("more pressure leaves fewer tokens",
          results[0.1] <= results[0.01] <= results[0.0] + 1e-6,
          f"{results[0.0]:.1f} >= {results[0.01]:.1f} >= {results[0.1]:.1f}")

    banner("6. INFERENCE DROPS THE CLOSED TOKENS")
    torch.manual_seed(0)
    head = make_head(budget_weight=0.1, target_tokens=10)
    run(head, feats, samples, steps=40)
    head.eval()
    with torch.no_grad():
        out = head(feats)
        results_list = head.predict(feats)
    open_now = float(out["keep"].sum(1).mean())
    detections = [len(r["scores_3d"]) if isinstance(r, dict) else len(r.scores_3d)
                  for r in results_list]
    print(f"  tokens open: {open_now:.1f}   detections per sample: {detections}")
    check("detections never exceed open tokens",
          all(d <= out["keep"][i].sum() for i, d in enumerate(detections)),
          f"{detections} vs {out['keep'].sum(1).tolist()}")

    banner("ALL PASS" if ok else "SOMETHING FAILED - do not start a training run")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)