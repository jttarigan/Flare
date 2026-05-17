# DATA_SPEC — Flare Phase 2 training-data pipeline

**Status:** DRAFT v0.2 — desktop-port architecture. Once accepted, the wire format below becomes load-bearing for every downstream phase. Do not start the desktop port build until this is signed off.

## What changed from v0.1

v0.1 had mobile-side capture with Mali shadow as both training GT and paper baseline. After user pushback, **v0.2 splits the two roles**:

- **Paper baseline (Phase 1, frozen):** the Mali real-time cube-shadow on the Poco. Measured at shadow_cast 5.9 ms / sum 11.2 ms. This is what the Phase-5 eval compares the NPU model against.
- **Training GT (Phase 2, new):** rendered on a desktop port of the game at *higher* quality (4096² PCF cube-shadow or ray-traced, TBD). The model learns to predict this. At deployment, model output is compared back against both the Mali baseline (cost win) and the desktop GT (quality target).

Capture, GT rendering, and training now all live on the desktop. Mobile is silent through Phases 2–3 — it only re-enters at Phase 4 (NNAPI deployment) and Phase 5 (eval sweep).

## Purpose

Produce two tensor-format datasets that Phase 3 trains against:

- **Dataset V** — visibility predictor input/output pairs. Replaces the 6-pass cube-shadow pre-pass (Phase 1 baseline: 5.9 ms on Mali).
- **Dataset L** — light-field encoder input/output pairs. Replaces the per-fragment N-light loop (Phase 1 baseline: ~3.2 ms skinned + 2.0 ms platform).

Both come from one capture run on the desktop port — each frame contributes one sample to each dataset.

## The desktop port (`FlareLab/`)

A sibling folder at `E:/OneDrive/MyProject/ITHappyViewer/FlareLab/`. Hosts the desktop entry point + scripted-input replay + high-quality GT renderer. Shares source with `ITHappyGame/` via CMake source-file references — no fork, no copy.

Minimum platform abstraction:

- **Asset I/O.** Behind a tiny `flare_asset_*` interface. Android impl uses `AAssetManager`; desktop impl uses `fopen` against `FlareLab/assets/` (a copy of the mobile assets).
- **Logging.** `LOGI` / `LOGE` macros that route to `__android_log_print` on Android, `printf(stderr)` on desktop.
- **Windowing.** GLFW on desktop, `NativeActivity` on mobile. Lives in each project's `main.cpp` only; never crosses into `game.cpp`.
- **Input.** Touch on mobile, mouse/keyboard + scripted-input on desktop. Both feed the existing `Game::onTouchDown` / `onTouchMove` / `onTouchUp` so the gameplay layer is identical.

The shaders compile unchanged: GLES 3.0 `#version 300 es` → desktop GL 3.3 with a one-line preprocessor swap (or use ANGLE to host GLES directly on desktop).

Why this works without a sim-to-real gap: the model's inputs (depth buffer, camera matrices, light list) are functions of game state, not driver. If desktop renders the same scene with the same projection, the depth buffer is bit-near-identical to Mali's. Light list and camera are literally just numbers passed in.

**Sanity check before deployment** (deferred to Phase 4): pull a few mobile-captured (depth, camera, lights) samples on the Poco, run them through the trained model, verify predictions match what they were during training. Drift = driver bug, caught before user-facing release.

## The five design decisions (re-answered for desktop)

### 1. Sampling strategy — *recommend scripted replay only, perturbations deferred*

**Decision.** 100% scripted replay from 5 input sessions, no random perturbation in v0.2. Add perturbation in v0.3 only if Phase-3 validation loss reveals coverage gaps.

**Why simpler now.** Desktop iteration cycle is seconds, not minutes. We can re-capture quickly if needed. Pre-emptively adding perturbation buys robustness we may not need and complicates the data spec.

**Input sessions to record** (mouse + keyboard, saved to `inputs.bin` per session):
1. Idle — cursor stationary, observe ambient.
2. Slow movement — cursor walks the platform.
3. Full chase — cursor runs around perimeter.
4. Attack scrum — cursor sits on enemy.
5. Large empty area — cursor at platform edge with no enemy nearby.

### 2. Visibility ground truth — *recommend high-quality PCF cube-shadow at 4096²*

**Decision.** GT for Dataset V is **4096² cube shadow with 16-sample PCF (Percentage Closer Filtering)**, captured as a per-fragment shadow factor `[0,1]`, downsampled to 256×256 f16 for storage.

**Why PCF and not ray-traced (yet).** PCF on a 4096² cube map is implementable in the existing shader with ~30 lines of GLSL. Ray-traced GT requires a separate pipeline (nvdiffrast or pyOpenGL) and ~1–2 extra weeks. Pick PCF for v0.2; revisit ray-traced if PCF quality doesn't beat the Mali baseline by enough margin to make a strong paper claim.

**Format per sample:** `(camera_view, camera_proj, hero_light_pos, depth_256x256_f16, shadow_factor_256x256_f16)`.

**Open:** confirm 256² is enough resolution. Eval will validate by comparing 256² model output upsampled to native vs. native PCF.

### 3. Scene encoding — *recommend depth + light list*

**Decision.** Model input is `(depth_256x256_f16, camera_view, camera_proj, light_list[32])`. Each light is `(pos_xyz, radius, color_rgb, falloff_scale)`.

**Why depth + light list and not voxels/bones.** Same rationale as v0.1: maps cleanly to U-Net + light-list concat, deployable to NNAPI INT8 op set. Bones are encoded in the depth buffer (post-skinning). Voxels need 3D conv which NNAPI struggles with.

**Always send the full `light_list[32]`** — zero-padded slots cost nothing. Active count is implicit in non-zero radii.

### 4. Storage budget — *recommend every 4th frame, 5 sessions × 2 min*

**Decision.**

- **Capture rate:** every 4th frame at 60 FPS = 15 samples/sec.
- **Per-sample size:**
  - depth buffer: 256 × 256 × 2 = 128 KB
  - shadow factor: 256 × 256 × 2 = 128 KB
  - light list: 32 × 32 = 1 KB
  - matrices: 256 B
  - **Total: ~258 KB raw, ~80 KB after zstd-1 compression.**
- **Per session:** 2 min × 15 samples/sec = 1800 samples = ~145 MB compressed.
- **Total corpus (5 sessions):** ~725 MB compressed.

Generous compared to v0.1's 520 MB because we dropped perturbation but kept the same per-sample resolution. Add perturbation later if needed.

### 5. Where data lives — *recommend `Flare/training/data/`, gitignored, manifest in git*

**Decision** (unchanged from v0.1):

```
Flare/training/
├── DATA_SPEC.md           (this file — in git)
├── capture/               (desktop-side dumper script — in git)
│   ├── run_session.py     (driver: starts FlareLab in capture mode, plays back inputs.bin, pulls samples.zst)
│   └── inputs/            (saved input sessions — in git, small)
│       ├── session_001_idle.bin
│       └── ...
├── data/                  (gitignored, generated)
│   ├── session_001_idle/
│   │   ├── manifest.json  (frame → byte_offset lookup)
│   │   ├── samples.zst    (concatenated samples)
│   │   └── meta.json      (capture date, FlareLab SHA, ITHappyGame SHA, GT settings)
│   └── ...
└── models/                (gitignored — PyTorch outputs, ONNX exports)
```

`FlareLab/` lives at the workspace root (sibling of `ITHappyGame/`), not inside `Flare/` — it's implementation, not research artifact.

## Wire format (unchanged from v0.1)

```
struct Sample {           // little-endian, packed, fixed size 264960 bytes
  uint64  frame_index;
  uint64  session_uuid_low;
  uint64  session_uuid_high;
  float32 camera_view[16];
  float32 camera_proj[16];
  float32 hero_light_pos[3];
  uint32  active_light_count;
  Light   light_list[32];                  // 1024 B
  float16 depth[256 * 256];                 // 131072 B
  float16 shadow_factor[256 * 256];         // 131072 B (PCF GT, captured on desktop)
}
struct Light {
  float32 pos[3];
  float32 radius;
  float32 color[3];
  float32 falloff_scale;
}
```

`samples.zst` per session: N back-to-back `Sample` structs, zstd-compressed. `manifest.json` is `[ {frame_index, byte_offset, byte_length}, ... ]` for random-access shuffled training.

## Dataset L addendum

Reuses the same captured samples — selects different fields:

- **Input:** `(camera_view, camera_proj, light_list)`.
- **GT:** per-fragment **lit color** = existing N-light diffuse output × (1 − shadow_factor) baked into `lit_rgb_256x256_f16`.
- Adds ~196 KB raw / ~70 KB compressed per sample. Total corpus grows to ~1.3 GB.

Captured in the same desktop session as Dataset V.

## Out of scope (Phase 3+)

- Model architecture, loss, training loop.
- NNAPI shaping, quantization, TFLite delegate choice.
- Eval sweep harness (`Flare/eval/`).
- User study (`Flare/study/`).
- Mobile re-deployment / sanity-check (Phase 4 task).

## Concrete sequence after sign-off

1. **Build `FlareLab/` desktop port.** Folder, CMake, GLFW window, asset glue, shader version-swap, render same Phase-1 scene on desktop. Goal: same render as the Poco screenshot.
2. **Add scripted-input record + replay.** Save mouse/keyboard timeline to `inputs.bin`; replay deterministically.
3. **Add capture mode.** Compile flag `FLARE_CAPTURE_TRAINING_DATA=1` that, when on, dumps every 4th frame's `Sample` struct to `samples.zst`. Off by default — desktop port still runs fine for free-flight testing.
4. **Add desktop-only high-quality shadow renderer** (4096² PCF). New shader path, only used when capture mode is on.
5. **Record 5 input sessions, run capture, compress.** End of Phase 2.

That's the order. Don't merge step 3 before step 2 has a replay path. Don't merge step 4 before step 3 has somewhere to dump the GT.

## Reviewer prompt

This is v0.2 — desktop-port architecture. Sanity-check the open items:

- **§1** (replay-only, no perturbation in v0.2): comfortable, or want to add perturbation upfront?
- **§2** (PCF 4096², not ray-traced): pick PCF for now and revisit, or invest in ray-traced from the start?
- **Step 4 in the sequence:** the high-quality shadow renderer is new desktop code. Confirm scope is OK (~30 lines of GLSL on top of existing cube-shadow).
