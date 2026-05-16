# CLAUDE.md — Flare

Context for Claude Code working on the **Flare** research project: NPU-accelerated lighting and visibility approximation for action games, targeting a Q1 journal submission.

## What Flare is (and isn't)

Flare is research, hosted in this `Flare/` folder under the wider `ITHappyViewer/` repo. The implementation testbed is the sibling **`ITHappyGame/`** Android NDK app — see `../ITHappyGame/CLAUDE.md` for the game's code architecture (skinning, squad logic, asset pipeline). Don't duplicate those here.

- **Research artifacts (this folder):** plans, PROJECT_STATE, training pipeline (PyTorch + ONNX/TFLite), eval scripts, paper drafts, figures.
- **Implementation target (`../ITHappyGame/`):** all C++/GLSL changes go here.
- **Out of scope for Flare:** `../renderer.js`, `../acscull/`, `../CullViewer/`, `../ITMapEditor/`, `../Asset/`. Leave them alone.

## Research thesis

Redistribute lighting and visibility computation from the contended GPU to the idle NPU via learned approximation, decoupling rendering cost from light count.

**Two technical contributions:**
1. **Neural light-field encoder (NPU):** turns an arbitrary light set into a compact feature grid, sampled cheaply in the fragment shader. Constant GPU cost regardless of light count.
2. **Learned visibility predictor (NPU):** replaces point-light cube shadow maps for action gameplay where perceptual thresholds justify approximation.

**Target:** Q1 journal.

## Tech stack

- **Game runtime:** Android NDK, OpenGL ES 3.0, C++17, CMake + Gradle (in `../ITHappyGame/`).
- **Training pipeline:** PyTorch (RTX desktop) → ONNX → TFLite → NNAPI.
- **Eval harness:** TBD. Sweeping run script invoked from desktop driving an instrumented APK.

## Device matrix

- **Poco X6 Pro** — MediaTek Dimensity 8300, APU 780. **Primary dev device.**
- **Samsung Galaxy A56** — Exynos 1580 NPU.
- **Samsung Galaxy Z Fold 6** — Snapdragon 8 Gen 3, Hexagon NPU. **Borrowed.**
- **Budget no-NPU device** — fallback case (must still run baseline + a CPU-emulation path for the NPU stages).

## Phase plan

| Phase | Weeks | Topic |
|---|---|---|
| 1 | 3 | Baseline and instrumentation |
| 2 | 2 | Data generation pipeline |
| 3 | 4 | Model design and training |
| 4 | 2 | Mobile deployment (NNAPI) |
| 5 | 3 | Evaluation |
| 6 | 3 (∥5) | User study |
| 7 | 4 | Writing |

**Current:** Phase 1, Step 1 — integrate forward N-light shading into the existing top-down game as the research baseline.

## Baseline decisions (Phase 1)

Locked in at planning and shape the upcoming N-light implementation:
- **UBO** for the light list. Baseline **cap 8 active lights**. Evaluation will sweep 1 / 4 / 8 / 16 / 32.
- **Per-fragment lighting for units** (skinned characters, props). **Per-vertex for terrain / ground.**
- **Quadratic falloff (atten²)** for cartoon punch — not physical inverse-square.
- **Explosion sprites are self-luminous** — they skip the light loop entirely.

## Commands

Game build / install / run lives in `../ITHappyGame/`:
- `cd ../ITHappyGame && ./gradlew installDebug` — build + install debug APK on attached device.
- `cd ../ITHappyGame && ./gradlew assembleRelease` — release APK for measurement.

Training / data / eval pipelines: **not yet created**. They will land under `Flare/` as Phase 2+ work begins.

## Code conventions (ITHappyGame, the implementation target)

Filled from a Phase-1 read of the native source. Cross-reference `../ITHappyGame/CLAUDE.md` for architecture; this section is the *style / patterns* a Flare change needs to match.

**Build / language**
- C++17, NDK, CMake (`../ITHappyGame/app/src/main/cpp/CMakeLists.txt`). Three sources: `main.cpp`, `game.cpp`, `gltf_model.cpp`. Single-header `cgltf.h` vendored.
- ABIs: `arm64-v8a` + `x86_64`. No Java/Kotlin — pure `NativeActivity` (`android:hasCode="false"`).
- GLES 3.0 (`EGL_OPENGL_ES3_BIT`, `#version 300 es`). UBOs, instancing, MRT, texture arrays available.
- No linter, no unit tests. Iteration loop: edit → `./gradlew installDebug` → manual device test → `adb logcat -s ITHappyGame`.

**Engine layout**
- `main.cpp` owns EGL and the `android_app` main loop; one `Engine` struct holds `Game` by value. Per-frame `dt` capped at 50 ms.
- `Game::init` compiles shaders inline as `R"(#version 300 es ... )"` raw-string literals and links 3 programs at startup: `shaderProgram` (SKINNED_VS/FS), `gridShader` (GRID_VS/FS), `ringShader` (RING_VS/FS). A 4th `joyShader` is lazily linked inside `renderJoystick`.
- `glEnable(GL_DEPTH_TEST)` on. `GL_CULL_FACE` deliberately **off** — upstream materials are DoubleSide.

**Per-frame render order** (`Game::render`, `game.cpp:922`)
1. `glClearColor(0.12, 0.12, 0.2, 1.0)` + depth clear.
2. `gridShader` → `renderPlatform()` (filled map cells) + line-grid floor.
3. `ringShader` → attack range ring + cursor indicator.
4. `shaderProgram` (skinned) → all `chars` + `enemy` + `renderPropPlacements()`. Props re-use the skinned program with `u_hasBones=0`.
5. `renderJoystick()` (lazy 2D shader).
6. `renderTriCount()` (7-segment HUD via `glScissor` + `glClear`).

**Uniform discovery**
- `glGetUniformLocation` is called per-draw inside the render path (not cached). Cheap to add new uniforms; cache once-per-program if profiling shows it hot.
- Skinning cap: `u_bones[128]` per mesh. `GltfModel::draw` clamps to 128.

**Current lighting (what Flare replaces)**
- `SKINNED_FS` (`game.cpp:53–76`) does a single hardcoded directional: `lightDir = normalize(vec3(0.4, 1.0, 0.3))`, `ambient = 0.35`, `diffuse = 0.65`.
- `u_colorOverride` is the only material knob: `(-1,-1,-1)` keeps vertex color; non-negative rgb replaces it (used for the red-tinted reduced-GLB toggle).
- Ground / platform: `GRID_FS` is pure unlit vertex color, no normals on the geometry.
- No specular, no shadows, no point lights, no falloff.

**Asset I/O**
- `loadGltfAsset()` reads via `AAssetManager` (zip-backed APK assets). GLBs in `../ITHappyGame/app/src/main/assets/`. Re-author through the viewer; **don't hand-edit GLBs**.
- `map.bin` / `map.json` authored by `../ITMapEditor`. Per-cell type byte: 0=void, 1=floor, 2=blocked-floor (prop).

**Skinning invariants — do not break** (full detail in `../ITHappyGame/CLAUDE.md`)
- Per-mesh `inverseBindMatrices` (don't share across meshes).
- Full node hierarchy walk (586 nodes for player.glb, only 44 are bones — wrapper nodes carry rest offsets).
- Per-TRS-path sampler times (T / R / S keyframes don't share timing).

## Flare-specific working conventions

- **Plan-before-implement** for non-trivial changes. Write the design in chat, wait for approval, then code. Applies to game-side and training-side.
- **Checkpoint commits** in `../ITHappyGame/` before risky refactors so `git reset --hard <hash>` is always a clean retreat.
- **No `Co-Authored-By: Claude` trailer** — josta is sole author. (Inherited from repo-wide memory.)
- **Single foregrounded Electron** if any viewer work is unavoidable — same rule as the wider repo.
- **PROJECT_STATE.md updated at end of every session** — what was done, open questions, next concrete step.

## Cross-references

- `PROJECT_STATE.md` — current session log + next concrete step.
- `../ITHappyGame/CLAUDE.md` — implementation-target architecture (authoritative for game internals).
- `../CLAUDE.md` — viewer-side context (mostly orthogonal to Flare).
