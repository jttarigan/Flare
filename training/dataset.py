"""Flare visibility-predictor dataset.

Decodes the DATA_SPEC v0.2 wire format from zstd-compressed Sample shards
produced by `FlareLab --capture`, and assembles the 9-channel input tensor
and 1-channel shadow-factor target the U-Net consumes (per MODEL_SPEC §2/§3).

Input channels (per pixel):
    0..2  world_pos.xyz    (reconstructed from captured depth + inverse VP;
                            zeroed for background pixels)
    3     linear_eye_z      (normalized eye-space z in [0, 1]; raw GL window
                             depth sits in [0.989, 1.000] for this scene, so
                             linearization is mandatory to expose dynamic range)
    4..6  hero_pos - world  (vector from receiver toward hero light; zeroed
                             for background)
    7     1 / max(eps, ||hero_pos - world||)  (zeroed for background)
    8     valid_mask        (1.0 where pixel hit scene geometry, 0.0 at
                             far-clip background; ~85% of pixels are background)

Target channel:
    0     shadow_factor     (1 = lit, 0 = shadowed; matches SKINNED_FS convention)

Numpy-only at parse time; the torch `Dataset` wrapper imports torch lazily so
this module is usable in environments without torch (e.g., quick smoke tests).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import zstandard

SAMPLE_BYTES = 263336
HEADER_BYTES = 168
LIGHTS_BYTES = 1024
TEX_SIZE = 256
PIXEL_COUNT = TEX_SIZE * TEX_SIZE
DEPTH_BYTES = PIXEL_COUNT * 2
SHADOW_BYTES = PIXEL_COUNT * 2
NUM_INPUT_CHANNELS = 9

assert HEADER_BYTES + LIGHTS_BYTES + DEPTH_BYTES + SHADOW_BYTES == SAMPLE_BYTES

EPS = 1e-4

# Window-depth threshold below which a pixel is considered "real geometry"
# (vs. the far-clip background). The scene's farthest legitimate surface sits
# around window-depth 0.998; 0.9995 leaves clear margin.
BACKGROUND_DEPTH_THRESHOLD = 0.9995

# Scene scale for normalizing linearized eye-space z into [0, 1]. Picked from
# the dataset smoke test: world bounds y∈[-3,+3], camera ~25 units above the
# platform, so eye-z range is roughly [0, 50].
SCENE_DEPTH_NORM = 50.0


def decompress_shard(zst_path: Path) -> bytes:
    """Decompress a samples.zst shard fully into memory.

    ~480 MB raw per session — fits comfortably on training hosts and avoids
    streaming-decompressor cost on every __getitem__.
    """
    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, "rb") as f:
        return dctx.decompress(f.read(), max_output_size=2_000_000_000)


def parse_sample(raw: bytes, offset: int) -> dict:
    """Parse one Sample record into numpy arrays + scalars.

    Matrices are converted from cglm column-major flat storage to mathematical
    [row, col] arrays by reshaping then transposing.
    """
    base = offset
    frame_idx, uuid_lo, uuid_hi = struct.unpack_from("<QQQ", raw, base)
    view = (
        np.frombuffer(raw, dtype=np.float32, count=16, offset=base + 24)
        .reshape(4, 4)
        .T.copy()
    )
    proj = (
        np.frombuffer(raw, dtype=np.float32, count=16, offset=base + 88)
        .reshape(4, 4)
        .T.copy()
    )
    hero_pos = np.frombuffer(
        raw, dtype=np.float32, count=3, offset=base + 152
    ).copy()
    (active,) = struct.unpack_from("<I", raw, base + 164)

    light_off = base + HEADER_BYTES
    lights = (
        np.frombuffer(raw, dtype=np.float32, count=32 * 8, offset=light_off)
        .reshape(32, 8)
        .copy()
    )

    depth_off = light_off + LIGHTS_BYTES
    depth_f16 = np.frombuffer(
        raw, dtype=np.float16, count=PIXEL_COUNT, offset=depth_off
    )
    depth = depth_f16.astype(np.float32).reshape(TEX_SIZE, TEX_SIZE)

    shadow_off = depth_off + DEPTH_BYTES
    shadow_f16 = np.frombuffer(
        raw, dtype=np.float16, count=PIXEL_COUNT, offset=shadow_off
    )
    shadow = shadow_f16.astype(np.float32).reshape(TEX_SIZE, TEX_SIZE)

    return {
        "frame_idx": int(frame_idx),
        "uuid": (int(uuid_lo), int(uuid_hi)),
        "view": view,
        "proj": proj,
        "hero_pos": hero_pos,
        "active_light_count": int(active),
        "lights": lights,
        "depth": depth,
        "shadow": shadow,
    }


_pixel_grid_cache: dict = {}


def _pixel_grid(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel-center NDC xy for a `size`×`size` viewport. Cached."""
    cached = _pixel_grid_cache.get(size)
    if cached is not None:
        return cached
    coord = (np.arange(size, dtype=np.float32) + 0.5) / size  # [0,1]
    uu, vv = np.meshgrid(coord, coord, indexing="xy")
    ndc_x = 2.0 * uu - 1.0
    ndc_y = 2.0 * vv - 1.0
    _pixel_grid_cache[size] = (ndc_x, ndc_y)
    return ndc_x, ndc_y


def reconstruct_world_pos(
    depth: np.ndarray, view: np.ndarray, proj: np.ndarray
) -> np.ndarray:
    """Per-pixel world position from a captured depth buffer.

    depth: [H, W] float32 in [0, 1] (GL window-space depth).
    Returns: [H, W, 3] float32 world-space position.
    """
    h, w = depth.shape
    ndc_x, ndc_y = _pixel_grid(h)
    ndc_z = 2.0 * depth - 1.0
    ones = np.ones_like(ndc_x)
    ndc = np.stack([ndc_x, ndc_y, ndc_z, ones], axis=-1)  # [H, W, 4]

    inv_vp = np.linalg.inv(proj @ view).astype(np.float32)
    world_h = ndc @ inv_vp.T  # [H, W, 4]
    w_comp = world_h[..., 3:4]
    safe_w = np.where(np.abs(w_comp) < EPS, EPS, w_comp)
    return (world_h[..., :3] / safe_w).astype(np.float32)


def extract_near_far(proj: np.ndarray) -> Tuple[float, float]:
    """Recover (znear, zfar) from a standard GL perspective projection matrix.

    Math matrix layout: P[2,2] = -(zf+zn)/(zf-zn), P[2,3] = -2*zf*zn/(zf-zn).
    """
    A = float(proj[2, 2])
    B = float(proj[2, 3])
    znear = B / (A - 1.0)
    zfar = B / (A + 1.0)
    return znear, zfar


def linearize_depth(window_depth: np.ndarray, znear: float, zfar: float) -> np.ndarray:
    """Convert GL window-space depth in [0,1] to positive eye-space z (distance
    from camera along view direction). Returns float32 of the same shape."""
    ndc_z = 2.0 * window_depth - 1.0
    eye_z = (2.0 * znear * zfar) / (zfar + znear - ndc_z * (zfar - znear))
    return eye_z.astype(np.float32)


def build_input_tensor(sample: dict) -> np.ndarray:
    """Assemble the 9-channel input from a parsed Sample. Returns [9, H, W] float32.

    Background pixels (depth at far clip) have their world_pos / delta /
    inv_dist channels zeroed; the validity mask (channel 8) tells the network
    which pixels carry real signal.
    """
    depth = sample["depth"]
    world = reconstruct_world_pos(depth, sample["view"], sample["proj"])

    znear, zfar = extract_near_far(sample["proj"])
    eye_z = linearize_depth(depth, znear, zfar)
    eye_z_norm = np.clip(eye_z / SCENE_DEPTH_NORM, 0.0, 1.0).astype(np.float32)

    hero = sample["hero_pos"].reshape(1, 1, 3)
    delta = hero - world
    dist = np.linalg.norm(delta, axis=-1, keepdims=True)
    inv_dist = 1.0 / np.maximum(EPS, dist)

    valid = (depth < BACKGROUND_DEPTH_THRESHOLD).astype(np.float32)
    valid_b = valid[..., None]
    world = world * valid_b
    delta = delta * valid_b
    inv_dist = inv_dist * valid_b

    chans = np.concatenate(
        [
            world,                   # 3
            eye_z_norm[..., None],   # 1
            delta,                   # 3
            inv_dist,                # 1
            valid_b,                 # 1
        ],
        axis=-1,
    ).astype(np.float32)
    return np.transpose(chans, (2, 0, 1)).copy()


class ShardIndex:
    """In-memory shard pool + flat sample index. Numpy-only; no torch."""

    def __init__(self, zst_paths: Sequence[Path]):
        self.paths: List[Path] = [Path(p) for p in zst_paths]
        self.shards: List[bytes] = []
        self.index: List[Tuple[int, int]] = []
        for shard_idx, p in enumerate(self.paths):
            raw = decompress_shard(p)
            if len(raw) % SAMPLE_BYTES != 0:
                raise ValueError(
                    f"shard {p} size {len(raw)} not a multiple of {SAMPLE_BYTES}"
                )
            n = len(raw) // SAMPLE_BYTES
            self.shards.append(raw)
            for sample_idx in range(n):
                self.index.append((shard_idx, sample_idx))

    def __len__(self) -> int:
        return len(self.index)

    def get(self, idx: int) -> dict:
        shard_idx, sample_idx = self.index[idx]
        return parse_sample(self.shards[shard_idx], sample_idx * SAMPLE_BYTES)


class FlareSampleDataset:
    """torch.utils.data.Dataset wrapper. Returns (input[8,H,W], target[1,H,W]) tensors.

    Torch is imported lazily so the rest of this module is importable without it.
    """

    def __init__(self, zst_paths: Iterable[Path]):
        import torch  # noqa: F401  (lazy: validate torch is importable now)

        self._torch = __import__("torch")
        self._idx = ShardIndex(list(zst_paths))

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, idx: int):
        sample = self._idx.get(idx)
        inp = build_input_tensor(sample)
        tgt = sample["shadow"][None, ...]
        return (
            self._torch.from_numpy(inp),
            self._torch.from_numpy(tgt),
        )
