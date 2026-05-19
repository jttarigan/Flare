"""Overfit-on-fixed-batch smoke test.

Standard "does the architecture learn?" diagnostic: take 8 samples from
session_01_idle, train against that single batch for ~80 iterations, watch
the masked-L1 drive below the trivial "predict all lit" baseline. If it
doesn't, something is broken (dead gradients, mask/target mismatch, class
weighting overpowering signal).

This is a 5–10s test on RTX 4060 Ti and should be re-run any time model.py,
loss.py, or dataset.py changes shape.

Pass criteria (rough, for v0.1):
  - L1 drops below the naive baseline within ~30 iters.
  - SSIM climbs above 0.90 within ~50 iters.
  - No NaN/Inf in any metric.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import FlareSampleDataset  # noqa: E402
from loss import VisibilityLoss, masked_ssim  # noqa: E402
from model import VisibilityUNet  # noqa: E402

REPO_ROOT = HERE.parent.parent
SHARD = REPO_ROOT / "Flare" / "training" / "data" / "session_01_idle" / "samples.zst"
N_SAMPLES_IN_BATCH = 8
N_ITERS = 80
LR = 3e-3  # high; we want to see fast convergence on the fixed batch


def main() -> int:
    if not SHARD.is_file():
        sys.exit(f"missing shard: {SHARD}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    torch.manual_seed(0)

    print(f"loading {SHARD.name}...")
    t0 = time.time()
    ds = FlareSampleDataset([SHARD])
    print(f"  {len(ds)} samples in {time.time() - t0:.1f}s")

    samples = [ds[i] for i in range(N_SAMPLES_IN_BATCH)]
    inp = torch.stack([s[0] for s in samples]).to(device)
    tgt = torch.stack([s[1] for s in samples]).to(device)
    mask = inp[:, 8:9]
    print(f"  fixed batch: inp {tuple(inp.shape)}  mask coverage {mask.mean().item():.3f}")

    model = VisibilityUNet().to(device)
    loss_fn = VisibilityLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # Baselines.
    model.eval()
    with torch.no_grad():
        pred = model(inp)
        b = loss_fn(pred, tgt, mask)
        b_ssim = masked_ssim(pred, tgt, mask)
        naive_l1 = (torch.abs(torch.ones_like(tgt) - tgt) * mask).sum() / mask.sum().clamp_min(1.0)
    print(f"\nbaseline      l1={b['l1'].item():.4f}  sobel={b['sobel'].item():.4f}  "
          f"ssim={b_ssim.item():.4f}")
    print(f"naive (all 1) l1={naive_l1.item():.4f}  (network must beat this)")

    print(f"\noverfit loop, {N_ITERS} iters @ lr={LR}:")
    model.train()
    t0 = time.time()
    final_l1 = float("inf")
    final_ssim = 0.0
    for it in range(N_ITERS):
        optim.zero_grad(set_to_none=True)
        pred = model(inp)
        out = loss_fn(pred, tgt, mask)
        out["total"].backward()
        optim.step()
        if (it + 1) % 10 == 0 or it == 0:
            with torch.no_grad():
                ssim = masked_ssim(pred, tgt, mask)
            print(f"  iter {it + 1:3d}  total={out['total'].item():.4f}  "
                  f"l1={out['l1'].item():.4f}  sobel={out['sobel'].item():.4f}  "
                  f"ssim={ssim.item():.4f}")
            final_l1 = out["l1"].item()
            final_ssim = ssim.item()

    dt = time.time() - t0
    print(f"\nwall: {dt:.1f}s ({dt * 1000 / N_ITERS:.1f} ms/iter)")
    print(f"final l1 = {final_l1:.4f}  vs naive = {naive_l1.item():.4f}  "
          f"({'PASS' if final_l1 < naive_l1.item() else 'FAIL'})")
    print(f"final ssim = {final_ssim:.4f}  "
          f"({'PASS' if final_ssim > 0.9 else 'FAIL'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
