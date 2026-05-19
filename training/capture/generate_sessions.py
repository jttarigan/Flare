"""Synthetic FlareLab cursor-script generator.

Emits 5 deterministic seeded `.bin` files in the cursor_script wire format
(see FlareLab/src/cursor_script.h):
    u32 frame_count
    N x { f32 world_x, f32 world_z }

Per-frame cursor world positions are written directly into Game::cursorPos
by `flarelab.exe --cursor-script <path>`. This bypasses the touch/joystick
input layer (which the v1 generator unknowingly fed sub-threshold deflections
to — all 5 sessions ended up capturing the same scene state). The squad's
chase/formation AI and the hero light's player-tracking still respond
normally to cursor motion.

Five archetypes (cursor positions in world coordinates, platform centered
at origin, enemy at world (3.0, 0, 0)):
    01_idle    cursor parked at (0, 0)
    02_slow    figure-8 around origin, amplitude 1.5
    03_chase   circle around origin, radius 2.5, period 20 s
    04_scrum   parked near enemy at (2.5, 0.3)
    05_empty   parked far from action at (-2.5, -1.5)

Each session is 7200 frames at 60 FPS (2 min wall-clock under --capture).
Gaussian noise sigma=0.05 (5 cm) jitters every frame so character formation
gets gentle stimulation even in "parked" sessions.
"""

import math
import os
import random
import struct
import sys

FRAMES_PER_SESSION = 7200
ENTRY_FMT = "<ff"  # 8 bytes: f32 x, f32 z
HEADER_FMT = "<I"  # 4 bytes: u32 frame_count

assert struct.calcsize(ENTRY_FMT) == 8
assert struct.calcsize(HEADER_FMT) == 4

OUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "FlareLab", "cursor_scripts",
))

NOISE_SIGMA = 0.05  # world-space jitter, ~5 cm


def write_script(path: str, trajectory):
    """trajectory yields (x, z) tuples — one per frame."""
    pts = list(trajectory)
    n = len(pts)
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, n))
        for x, z in pts:
            f.write(struct.pack(ENTRY_FMT, float(x), float(z)))
    return n


def gen_idle(rng):
    for _ in range(FRAMES_PER_SESSION):
        yield (rng.gauss(0.0, NOISE_SIGMA), rng.gauss(0.0, NOISE_SIGMA))


def gen_slow(rng):
    amp = 1.5
    period = 600.0  # 10 s per cycle
    for frame in range(FRAMES_PER_SESSION):
        t = frame / period
        x = amp * math.sin(2 * math.pi * t)
        z = amp * math.sin(4 * math.pi * t) * 0.5  # figure-8: x at f, z at 2f
        yield (x + rng.gauss(0.0, NOISE_SIGMA),
               z + rng.gauss(0.0, NOISE_SIGMA))


def gen_chase(rng):
    radius = 2.5
    period = 1200.0  # 20 s per revolution
    for frame in range(FRAMES_PER_SESSION):
        theta = 2 * math.pi * (frame / period)
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        yield (x + rng.gauss(0.0, NOISE_SIGMA),
               z + rng.gauss(0.0, NOISE_SIGMA))


def gen_scrum(rng):
    # Enemy at world (3.0, 0, 0); park cursor just shy of it so the squad
    # piles up at attack range.
    bx, bz = 2.5, 0.3
    for _ in range(FRAMES_PER_SESSION):
        yield (bx + rng.gauss(0.0, NOISE_SIGMA),
               bz + rng.gauss(0.0, NOISE_SIGMA))


def gen_empty(rng):
    bx, bz = -2.5, -1.5
    for _ in range(FRAMES_PER_SESSION):
        yield (bx + rng.gauss(0.0, NOISE_SIGMA),
               bz + rng.gauss(0.0, NOISE_SIGMA))


SESSIONS = [
    ("session_01_idle.bin",  gen_idle,  0xC0FFEE_001),
    ("session_02_slow.bin",  gen_slow,  0xC0FFEE_002),
    ("session_03_chase.bin", gen_chase, 0xC0FFEE_003),
    ("session_04_scrum.bin", gen_scrum, 0xC0FFEE_004),
    ("session_05_empty.bin", gen_empty, 0xC0FFEE_005),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, gen, seed in SESSIONS:
        rng = random.Random(seed)
        path = os.path.join(OUT_DIR, name)
        n = write_script(path, gen(rng))
        size = os.path.getsize(path)
        print(f"  {name}: {n} frames ({size} B)")
    print(f"wrote {len(SESSIONS)} cursor scripts to {OUT_DIR}")


if __name__ == "__main__":
    main()
