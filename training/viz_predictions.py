"""Render prediction-vs-GT mosaics from a trained checkpoint.

For each picked sample, lays out a 4-panel strip (256 x 1024):
    [ depth (lin_z) | GT shadow | predicted shadow | abs diff (masked) ]

Background pixels (valid_mask == 0) are drawn in a neutral grey across all
panels so the eye can ignore them. Mosaics for "seen" (session_01_idle)
and "unseen" (session_05_empty) samples are saved side-by-side so any
generalization gap is visible at a glance.

Usage:
    python viz_predictions.py --ckpt checkpoints_shake/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import FlareSampleDataset  # noqa: E402
from model import VisibilityUNet  # noqa: E402

REPO_ROOT = HERE.parent.parent
DATA_ROOT = HERE / "data"
CAPTURES = REPO_ROOT / "Flare" / "captures"

SEEN_SHARD = DATA_ROOT / "session_01_idle" / "samples.zst"
UNSEEN_SHARD = DATA_ROOT / "session_05_empty" / "samples.zst"
SAMPLE_INDICES = [0, 450, 900, 1350, 1800]


def to_grey255(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def apply_mask(panel: np.ndarray, mask: np.ndarray, bg: int = 90) -> np.ndarray:
    out = panel.copy()
    out[mask < 0.5] = bg
    return out


def build_strip(sample: dict, pred: np.ndarray) -> np.ndarray:
    """Return a [256, 1024] uint8 image: depth | GT | pred | diff."""
    lin_z = np.clip(sample["depth"] * 0.0, 0, 1)  # placeholder; replaced below
    # Use eye-z normalized for visualisation — same as dataset channel 3.
    from dataset import extract_near_far, linearize_depth, SCENE_DEPTH_NORM
    znear, zfar = extract_near_far(sample["proj"])
    eye_z = linearize_depth(sample["depth"], znear, zfar)
    lin_z = np.clip(eye_z / SCENE_DEPTH_NORM, 0.0, 1.0)
    # Invert so closer = brighter, easier to read.
    lin_z_panel = 1.0 - lin_z

    gt = sample["shadow"]
    diff = np.abs(pred - gt)

    valid = (sample["depth"] < 0.9995).astype(np.float32)

    p_depth = apply_mask(to_grey255(lin_z_panel), valid)
    p_gt = apply_mask(to_grey255(gt), valid)
    p_pred = apply_mask(to_grey255(pred), valid)
    p_diff = apply_mask(to_grey255(diff * 4.0), valid)  # 4x to amplify

    return np.concatenate([p_depth, p_gt, p_pred, p_diff], axis=1)


def render_mosaic(ds: FlareSampleDataset, model: torch.nn.Module,
                  device: torch.device, indices: list[int]) -> np.ndarray:
    """Vertical stack of per-sample 4-panel strips."""
    rows = []
    for i in indices:
        sample = ds._idx.get(i)  # use the raw numpy dict so we can grab depth
        # Build a 9-channel input via the same pipeline.
        from dataset import build_input_tensor
        inp = build_input_tensor(sample)
        with torch.no_grad():
            pred = model(torch.from_numpy(inp).unsqueeze(0).to(device))
        pred_np = pred[0, 0].cpu().numpy()
        strip = build_strip(sample, pred_np)
        rows.append(strip)
    return np.concatenate(rows, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-prefix", type=str, default="viz_shakedown")
    args = ap.parse_args()

    CAPTURES.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading ckpt {args.ckpt}...")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = VisibilityUNet().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"  ckpt epoch = {state['epoch']}  best val_l1 reachable from history.json")

    for tag, shard in [("seen", SEEN_SHARD), ("unseen", UNSEEN_SHARD)]:
        if not shard.is_file():
            print(f"  skipping {tag}: {shard} missing")
            continue
        print(f"loading {tag} shard {shard.name}...")
        ds = FlareSampleDataset([shard])
        idx = [i for i in SAMPLE_INDICES if i < len(ds)]
        mosaic = render_mosaic(ds, model, device, idx)
        out = CAPTURES / f"{args.out_prefix}_{tag}.png"
        Image.fromarray(mosaic, mode="L").save(out)
        print(f"  wrote {out}  ({mosaic.shape[1]}x{mosaic.shape[0]})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
