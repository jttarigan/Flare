"""Synthetic FlareLab input-log generator.

Emits 5 deterministic seeded `.bin` files matching the Phase-2-step-2 wire
format (see FlareLab/src/input_log.h):
    u32 frame_idx, u8 type, u8 pointer_id, f32 x, f32 y

One Down at frame 0 → Move every frame for FRAMES_PER_SESSION → Up at end.
Window is locked at 1280x720 in replay mode; coordinates are window pixels.
Gaussian noise is added per-frame so each pattern covers a neighbourhood of
the nominal trajectory.

Five archetypes per DATA_SPEC v0.2 Section 1:
    01_idle   — cursor parked near platform centre
    02_slow   — slow sine walk around centre
    03_chase  — full perimeter circle
    04_scrum  — cursor sits on enemy spawn region
    05_empty  — cursor parked at far platform edge
"""

import math
import os
import random
import struct
import sys

WIDTH = 1280
HEIGHT = 720
CX = WIDTH / 2.0   # 640
CY = HEIGHT / 2.0  # 360

FRAMES_PER_SESSION = 7200  # 2 min at 60 FPS
POINTER_ID = 0

EV_DOWN = 0
EV_MOVE = 1
EV_UP   = 2

EVENT_FMT = "<IBBff"  # 14 bytes: u32, u8, u8, f32, f32
assert struct.calcsize(EVENT_FMT) == 14

SESSIONS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "FlareLab", "sessions",
)


def write_log(path, trajectory):
    """trajectory is an iterable of (frame, x, y). Emits Down+Moves+Up."""
    last_frame = -1
    last_xy = None
    with open(path, "wb") as f:
        for i, (frame, x, y) in enumerate(trajectory):
            x = max(0.0, min(float(WIDTH  - 1), float(x)))
            y = max(0.0, min(float(HEIGHT - 1), float(y)))
            ev_type = EV_DOWN if i == 0 else EV_MOVE
            f.write(struct.pack(EVENT_FMT, frame, ev_type, POINTER_ID, x, y))
            last_frame = frame
            last_xy = (x, y)
        if last_xy is None:
            raise RuntimeError("empty trajectory")
        f.write(struct.pack(EVENT_FMT, last_frame, EV_UP, POINTER_ID,
                            last_xy[0], last_xy[1]))
    return last_frame + 1  # event count + 1 for the Up


def gen_idle(rng):
    sigma = 2.0
    for frame in range(FRAMES_PER_SESSION):
        yield (frame,
               CX + rng.gauss(0.0, sigma),
               CY + rng.gauss(0.0, sigma))


def gen_slow(rng):
    sigma = 2.0
    amp = 120.0
    period_frames = 600.0  # 10 s per cycle
    for frame in range(FRAMES_PER_SESSION):
        t = frame / period_frames
        x = CX + amp * math.sin(2 * math.pi * t)
        y = CY + amp * math.sin(2 * math.pi * t * 0.5)
        yield (frame,
               x + rng.gauss(0.0, sigma),
               y + rng.gauss(0.0, sigma))


def gen_chase(rng):
    sigma = 3.0
    radius = 300.0
    period_frames = 1200.0  # 20 s per revolution
    for frame in range(FRAMES_PER_SESSION):
        theta = 2 * math.pi * (frame / period_frames)
        x = CX + radius * math.cos(theta)
        y = CY + radius * math.sin(theta)
        yield (frame,
               x + rng.gauss(0.0, sigma),
               y + rng.gauss(0.0, sigma))


def gen_scrum(rng):
    # Enemy spawn is roughly upper-centre in screen space. Park cursor near it.
    sigma = 4.0
    bx, by = CX, 200.0
    for frame in range(FRAMES_PER_SESSION):
        yield (frame,
               bx + rng.gauss(0.0, sigma),
               by + rng.gauss(0.0, sigma))


def gen_empty(rng):
    sigma = 2.0
    bx, by = 100.0, 100.0
    for frame in range(FRAMES_PER_SESSION):
        yield (frame,
               bx + rng.gauss(0.0, sigma),
               by + rng.gauss(0.0, sigma))


SESSIONS = [
    ("session_01_idle.bin",  gen_idle,  0xF1A8E_001),
    ("session_02_slow.bin",  gen_slow,  0xF1A8E_002),
    ("session_03_chase.bin", gen_chase, 0xF1A8E_003),
    ("session_04_scrum.bin", gen_scrum, 0xF1A8E_004),
    ("session_05_empty.bin", gen_empty, 0xF1A8E_005),
]


def main():
    out_dir = os.path.normpath(SESSIONS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    for name, gen, seed in SESSIONS:
        rng = random.Random(seed)
        path = os.path.join(out_dir, name)
        n = write_log(path, gen(rng))
        size = os.path.getsize(path)
        print(f"  {name}: {n} events ({size} B)")
    print(f"wrote {len(SESSIONS)} sessions to {out_dir}")


if __name__ == "__main__":
    main()
