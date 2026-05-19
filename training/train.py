"""Train the visibility predictor — per MODEL_SPEC §6/§7.

Train sessions:  01_idle, 02_slow, 03_chase, 04_scrum (90/10 random split for
                 early-stopping validation).
Held-out test:   05_empty (left alone until final eval).

Augmentation: random horizontal flip with world-x channel + delta-x channel
sign-negation (preserves world coordinate semantics under the flip).

Logs per epoch to stdout: train loss components, val L1, val SSIM. Checkpoints
the best-val-L1 model + the latest model under training/checkpoints/.

CLI:
    python train.py --data-root <path> --epochs 100 --batch 32 [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, random_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import FlareSampleDataset  # noqa: E402
from loss import VisibilityLoss, masked_ssim  # noqa: E402
from model import VisibilityUNet, count_parameters  # noqa: E402

TRAIN_SESSIONS = ["session_01_idle", "session_02_slow",
                  "session_03_chase", "session_04_scrum"]
TEST_SESSION = "session_05_empty"

NUM_INPUT_CHANNELS = 9
VALID_MASK_CHANNEL = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=HERE.parent / "training" / "data")
    p.add_argument("--ckpt-dir", type=Path,
                   default=HERE / "checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-flip", action="store_true")
    return p.parse_args()


def horizontal_flip_batch(inp: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-place-ish horizontal flip of input + target. world-x (ch 0) and
    delta-x (ch 4) sign-negate after the spatial flip so coordinate semantics
    survive."""
    inp = torch.flip(inp, dims=[-1])
    tgt = torch.flip(tgt, dims=[-1])
    inp[:, 0] = -inp[:, 0]
    inp[:, 4] = -inp[:, 4]
    return inp, tgt


def split_train_val(ds, val_frac: float, seed: int) -> Tuple[Subset, Subset]:
    n = len(ds)
    n_val = max(1, int(round(n * val_frac)))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=gen)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: VisibilityLoss,
    *,
    device: torch.device,
    optim: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    do_flip: bool,
    amp_dtype: torch.dtype | None,
) -> dict:
    is_train = optim is not None
    model.train(mode=is_train)

    sums = {"total": 0.0, "l1": 0.0, "sobel": 0.0, "ssim": 0.0}
    n_batches = 0

    for inp, tgt in loader:
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        if is_train and do_flip and random.random() < 0.5:
            inp, tgt = horizontal_flip_batch(inp, tgt)
        mask = inp[:, VALID_MASK_CHANNEL:VALID_MASK_CHANNEL + 1]

        if amp_dtype is not None:
            with autocast(device_type=device.type, dtype=amp_dtype):
                pred = model(inp)
                losses = loss_fn(pred, tgt, mask)
        else:
            pred = model(inp)
            losses = loss_fn(pred, tgt, mask)

        if is_train:
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(losses["total"]).backward()
                scaler.step(optim)
                scaler.update()
            else:
                losses["total"].backward()
                optim.step()
        else:
            with torch.no_grad():
                sums["ssim"] += float(masked_ssim(pred.float(), tgt.float(), mask.float()))

        sums["total"] += float(losses["total"].detach())
        sums["l1"] += float(losses["l1"])
        sums["sobel"] += float(losses["sobel"])
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"device: {device}")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve shard paths.
    train_shards: List[Path] = []
    for s in TRAIN_SESSIONS:
        p = args.data_root / s / "samples.zst"
        if not p.is_file():
            sys.exit(f"missing shard: {p}")
        train_shards.append(p)
    test_shard = args.data_root / TEST_SESSION / "samples.zst"
    if not test_shard.is_file():
        sys.exit(f"missing test shard: {test_shard}")

    print(f"loading {len(train_shards)} train shards into memory...")
    t0 = time.time()
    train_ds = FlareSampleDataset(train_shards)
    print(f"  {len(train_ds)} samples in {time.time() - t0:.1f}s")

    train_subset, val_subset = split_train_val(train_ds, args.val_frac, args.seed)
    print(f"split: {len(train_subset)} train / {len(val_subset)} val")

    train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_subset, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))

    model = VisibilityUNet(in_channels=NUM_INPUT_CHANNELS).to(device)
    n_params = count_parameters(model)
    print(f"model params: {n_params:,}")

    loss_fn = VisibilityLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)

    use_amp = (not args.no_amp) and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else None
    scaler = GradScaler(device.type) if use_amp else None
    print(f"amp: {'fp16' if use_amp else 'off'}")

    best_val_l1 = math.inf
    history = []
    train_t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_metrics = run_epoch(
            model, train_loader, loss_fn,
            device=device, optim=optim, scaler=scaler,
            do_flip=not args.no_flip, amp_dtype=amp_dtype,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, loss_fn,
                device=device, optim=None, scaler=None,
                do_flip=False, amp_dtype=amp_dtype,
            )
        sched.step()
        ep_dt = time.time() - ep_t0

        row = {
            "epoch": epoch,
            "lr": optim.param_groups[0]["lr"],
            "train_total": train_metrics["total"],
            "train_l1": train_metrics["l1"],
            "train_sobel": train_metrics["sobel"],
            "val_l1": val_metrics["l1"],
            "val_sobel": val_metrics["sobel"],
            "val_ssim": val_metrics["ssim"],
            "epoch_sec": ep_dt,
        }
        history.append(row)
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"lr={row['lr']:.2e}  "
            f"train_l1={row['train_l1']:.4f}  train_sobel={row['train_sobel']:.4f}  "
            f"val_l1={row['val_l1']:.4f}  val_ssim={row['val_ssim']:.4f}  "
            f"({ep_dt:.1f}s)"
        )

        # Save latest + best.
        latest = {"epoch": epoch, "model": model.state_dict(),
                  "optim": optim.state_dict(), "args": vars(args)}
        torch.save(latest, args.ckpt_dir / "latest.pt")
        if val_metrics["l1"] < best_val_l1:
            best_val_l1 = val_metrics["l1"]
            torch.save(latest, args.ckpt_dir / "best.pt")
            print(f"  -> new best val_l1 {best_val_l1:.4f}, saved best.pt")

    with open(args.ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"done in {(time.time() - train_t0) / 60:.1f} min. best val_l1 = {best_val_l1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
