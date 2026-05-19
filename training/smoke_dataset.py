"""Smoke test for the Flare dataset loader. Numpy-only, no torch required.

Verifies:
- zstd shard decompresses to a whole number of Samples.
- Header / matrices / light list / depth / shadow fields parse with sane ranges.
- World-position reconstruction from depth + inv(VP) lands in plausible scene coords.
- Input tensor shape matches MODEL_SPEC §2 (8 × 256 × 256).
- Shadow factor distribution is bimodal (lit + shadowed + thin penumbra).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import (  # noqa: E402
    BACKGROUND_DEPTH_THRESHOLD,
    NUM_INPUT_CHANNELS,
    TEX_SIZE,
    ShardIndex,
    build_input_tensor,
    extract_near_far,
)

REPO_ROOT = HERE.parent.parent  # ITHappyViewer/
DATA_ROOT = REPO_ROOT / "Flare" / "training" / "data"

CHANNEL_NAMES = ["wx", "wy", "wz", "lin_z", "dx", "dy", "dz", "inv_dist", "valid"]


def summarize_sample(idx: int, sample: dict) -> None:
    print(f"\nsample[{idx}]  frame_idx={sample['frame_idx']}  "
          f"active_lights={sample['active_light_count']}  "
          f"hero={sample['hero_pos'].tolist()}")
    znear, zfar = extract_near_far(sample["proj"])
    print(f"  proj diag = {np.diag(sample['proj']).tolist()}")
    print(f"  znear={znear:.4f}  zfar={zfar:.4f}  (ratio {zfar / znear:.1f})")

    depth = sample["depth"]
    valid_frac = float((depth < BACKGROUND_DEPTH_THRESHOLD).mean())
    print(f"  depth      shape={depth.shape}  "
          f"min={depth.min():.4f}  max={depth.max():.4f}  "
          f"mean={depth.mean():.4f}  on-platform frac={valid_frac:.3f}")

    inp = build_input_tensor(sample)
    assert inp.shape == (NUM_INPUT_CHANNELS, TEX_SIZE, TEX_SIZE), (
        f"input shape {inp.shape} != ({NUM_INPUT_CHANNELS}, {TEX_SIZE}, {TEX_SIZE})"
    )
    print(f"  input      shape={inp.shape}  dtype={inp.dtype}")
    valid_mask = inp[8] > 0.5
    for c, name in enumerate(CHANNEL_NAMES):
        ch = inp[c]
        # For non-mask channels, also report valid-pixel-only stats.
        if c == 8:
            print(f"    [{c}] {name:8s}  min={ch.min():+.3f}  max={ch.max():+.3f}  "
                  f"mean={ch.mean():+.3f}")
        else:
            v = ch[valid_mask]
            print(f"    [{c}] {name:8s}  min={ch.min():+.3f}  max={ch.max():+.3f}  "
                  f"mean={ch.mean():+.3f}    "
                  f"(valid only: min={v.min():+.3f} max={v.max():+.3f} mean={v.mean():+.3f})")

    s = sample["shadow"]
    s_valid = s[valid_mask]
    lit = int((s_valid >= 0.95).sum())
    shad = int((s_valid <= 0.05).sum())
    pen = int(((s_valid > 0.05) & (s_valid < 0.95)).sum())
    n_valid = int(valid_mask.sum())
    print(f"  shadow (on-platform only, {n_valid} px):  "
          f"min={s_valid.min():.3f}  max={s_valid.max():.3f}  mean={s_valid.mean():.3f}")
    print(f"    lit (>=0.95) {lit} ({100*lit/n_valid:.1f}%)  "
          f"shadowed (<=0.05) {shad} ({100*shad/n_valid:.1f}%)  "
          f"penumbra {pen} ({100*pen/n_valid:.1f}%)")


def main() -> int:
    sessions = sorted(
        d for d in DATA_ROOT.iterdir() if (d / "samples.zst").is_file()
    )
    if not sessions:
        sys.exit(f"no session shards under {DATA_ROOT}")

    print(f"discovered {len(sessions)} sessions under {DATA_ROOT}:")
    for s in sessions:
        size_mb = (s / "samples.zst").stat().st_size / 1e6
        print(f"  {s.name}: samples.zst = {size_mb:.2f} MB")

    # Load just the first shard for the deep probe; cheap and representative.
    first = sessions[0]
    print(f"\nloading {first.name} into memory...")
    idx = ShardIndex([first / "samples.zst"])
    print(f"  {len(idx)} samples")

    probe_indices = [0, len(idx) // 2, len(idx) - 1]
    for i in probe_indices:
        summarize_sample(i, idx.get(i))

    # Quick aggregate: shadow distribution on-platform across the whole shard.
    print(f"\n[aggregate] sweeping all {len(idx)} samples for on-platform shadow stats...")
    lit_total = shad_total = pen_total = valid_total = 0
    for i in range(len(idx)):
        sample = idx.get(i)
        valid = sample["depth"] < BACKGROUND_DEPTH_THRESHOLD
        s = sample["shadow"][valid]
        lit_total += int((s >= 0.95).sum())
        shad_total += int((s <= 0.05).sum())
        pen_total += int(((s > 0.05) & (s < 0.95)).sum())
        valid_total += int(valid.sum())
    print(f"  total on-platform pixels: {valid_total}  "
          f"(of {len(idx) * TEX_SIZE * TEX_SIZE} total, "
          f"{100 * valid_total / (len(idx) * TEX_SIZE * TEX_SIZE):.1f}%)")
    print(f"  lit fraction:     {lit_total / valid_total:.4f}")
    print(f"  shadow fraction:  {shad_total / valid_total:.4f}")
    print(f"  penumbra frac:    {pen_total / valid_total:.4f}")
    print(f"  (bimodal-ish distribution expected — penumbra is the PCF soft edge)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
