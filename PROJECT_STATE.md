# PROJECT_STATE — Flare

Single source of truth for "where is the research right now". Update at the end of every session.

## Current position

- **Phase 1 — closed.** Baseline forward N-light shading + hero cube-shadow + per-stage GPU timing all running on Poco X6 Pro.
- **Phase 2 — started 2026-05-18.** Steps 1, 2, and 3 closed and verified. Step 4 (4096² PCF GT renderer) next.
- **Step 1 closed and device-verified** (1a–1d).
- **Step 2 closed and device-verified.** Point-light cube shadow mapping for the hero light is the GPU baseline the learned visibility predictor (contribution #2) gets benchmarked against in Phase 5.
- **Step 3 closed and device-verified.** Per-stage GPU timing via `EXT_disjoint_timer_query`. Mali r44p1 quirk: extension is omitted from `glGetString(GL_EXTENSIONS)` but `eglGetProcAddress` still returns working entry points — probe via the entry-point pointers + a `glGenQueriesEXT` smoke test, ignore the extension string.
- **Phase 1 baseline numbers (Poco, 8 active lights, hero shadow on):** shadow_cast ≈ 5.9 ms, platform ≈ 2.0 ms, skinned ≈ 3.2 ms, sum ≈ 11.2 ms. Shadow pre-pass dominates — confirms the Phase-3 learned visibility predictor (contribution #2) is the right target.
- **Phase 2 next: training-data generation.** Decoupled from `ITHappyGame/` — lives under `Flare/training/` (not yet created).
- **Game-side commits (in `../ITHappyGame/`):**
  - `0934f78` — Checkpoint (pre-Flare asset refresh; retreat point).
  - `1160bea` — Flare 1a (single hero point light, 0.5× chars, dim ambient, 20% zoom).
  - `8526ba1` — Flare 1b (std140 UBO `LightBlock`, 8-light cap, deterministic colored seed scene).
  - `5a7ad76` — Flare 1c (`PLATFORM_VS/FS` per-vertex ground lighting sharing the UBO; unlit line grid kept on `gridShader`).
  - `544e6a8` — Flare 1d (top-right corner tap cycles `activeLightCount` 1/4/8/16/32; green HUD digit).
  - `47269ea` — Flare Step 2 (512² depth cube map, 6-pass shadow caster for hero light only; SKINNED_FS + PLATFORM_VS/FS split hero from non-hero so per-fragment shadow factor cleanly attenuates only its contribution).
  - `ebdf549` — Flare Step 3 (`EXT_disjoint_timer_query` with 4-slot ring, 3 stages, cyan µs HUD). Probe via `eglGetProcAddress` + smoke test, not the extension string (Mali quirk).
  - `15b8170` — Add `platform.h` shim (FlareLab enabler). Android build bit-identical; on desktop, shim provides AAssetManager/AAsset over fopen + GLEW in place of GLES3 headers.

- **FlareLab/ commits** (separate local repo, sibling to `ITHappyGame/`):
  - `0b4007a` — Phase 2 step 1 desktop port scaffold. GLFW + GL 3.3 + GLEW. Shares game.cpp/gltf_model.cpp via CMake source references. Renders the full Phase-1 scene on desktop (RTX 4060 Ti).
  - `00ef870` — Phase 2 step 2 scripted input record + replay. `--record <path>` / `--replay <path>` CLI; 14-byte/event binary log (u32 frame, u8 type, u8 id, f32 x, f32 y) written field-by-field. Replay forces dt=1/60, non-resizable window, 2 s grace after last event. Verified: 1055-event session replayed faithfully.
  - `0b07403` — Phase 2 step 3 capture mode. `--capture <path>` writes one 263336-byte Sample (DATA_SPEC v0.2 wire format) every 4th frame. 256² depth captured via `glBlitFramebuffer` from default FBO → 256² capture FBO → readback → hand-rolled f32→f16. Shadow factor is placeholder zeros until step 4. Smoke test: 38 samples × 263336 B exactly.

## Session log

### Session 1 — 2026-05-16

**Goal:** Set up research workspace, read the existing testbed, propose N-light forward shading.

**Done**
- Created `Flare/CLAUDE.md` (research thesis, tech stack, device matrix, phase plan, baseline decisions, ITHappyGame code conventions).
- Created this `Flare/PROJECT_STATE.md`.
- Persisted research scope to auto-memory: `project_flare_research`, `reference_flare_workspace`, `feedback_flare_plan_before_implement`. Updated `MEMORY.md` index.
- Read `ITHappyGame/app/src/main/cpp/{main.cpp, game.h, game.cpp, gltf_model.h, gltf_model.cpp, CMakeLists.txt}`. Located the current lighting (hardcoded directional in `SKINNED_FS`, `game.cpp:53–76`) and the per-frame uniform upload points (`Game::render`, `game.cpp:922–1027`).
- Delivered the N-light forward shading plan in chat. User approved.
- **Checkpoint commit `0934f78`** in `../ITHappyGame/` (bundled latest GLB asset refresh + retreat marker).
- **Implemented sub-step 1a** (`1160bea`): single hero point light tracking the player. Replaced hardcoded directional in `SKINNED_FS` with `u_lightPos / u_lightColor / u_lightRadius / u_ambient` uniforms, quadratic falloff. Added `Game::{ambientColor, heroLightColor, heroLightRadius, heroLightYOffset}` and per-frame upload at top of `Game::render` (one upload covers chars + enemy + props since they share `shaderProgram`). Visual tweaks bundled: 0.5× character scale (with bounding box), 2× cursor offsets for wider follower spread, ambient dropped to `(0.12, 0.12, 0.14)`, camera 20% closer.
- **Implemented sub-step 1b** (`8526ba1`): replaced sub-step 1a's per-frame `glUniform` calls with a 1056-byte std140 `LightBlock` UBO holding `ambient + lightCount + Light[32]` (interleaved `posR`, `colAtt`). One `glBufferSubData` + `glBindBufferBase` per frame; same compiled shader handles every eval sweep (1/4/8/16/32) via runtime-bounded loop on `u_lightCount.x`. Seeded 8-light Phase-1 scene: hero (slot 0, player-tracked) + 7 static colored points (R/C/G/Y/M/B/O) at y=1.5 around origin. Block index resolved via `glGetUniformBlockIndex` + `glUniformBlockBinding` (ES 3.0 has no `layout(binding=)` on uniform blocks).
- Verified clean on Mali-G615 (Poco X6 Pro, OpenGL ES 3.2 driver r44p1): zero shader compile errors, multi-color falloff visible on characters as the player traverses the seeded scene.
- **Implemented sub-step 1c** (`5a7ad76`): per-vertex ground lighting via new `PLATFORM_VS/FS` pair. VS runs the N-light loop with a constant `(0,1,0)` up-normal, reads the same `LightBlock` UBO as `SKINNED_FS`; FS multiplies vertex base color by the lit term. UBO upload relocated to the top of `Game::render` so both shader programs see fresh light data before any draw — one `glBindBufferBase` persists across `glUseProgram` calls. `cleanup()` extended to free `platformShader` + `lightUbo`. User-confirmed visually on device.
- **Established device-screenshot workflow.** `adb exec-out screencap -p > Flare/captures/<name>.png` produces a PNG Claude can read directly (multimodal); `Flare/captures/` is gitignored. First attempt in session 1 returned an all-black frame because the phone screen had slept between user confirmation and capture — visual feedback worked but the assistant-loop visual check is screen-state-dependent.
- **Implemented sub-step 1d** (`544e6a8`): top-right corner tap (x > 0.75·W, y < 0.25·H) cycles `activeLightCount` through `1 → 4 → 8 → 16 → 32 → 1` — the exact Phase-5 eval sweep order, so the sweep becomes data collection on the same APK without per-count rebuilds. New `renderLightCount` adds a green 5th HUD line below the four tri-count lines so screenshots are self-labelling. Build-verified; **not yet device-tested** (Poco disconnected).
- **Implemented Step 2** (`47269ea`): point-light cube shadow mapping for the hero light only. 512² `GL_DEPTH_COMPONENT24` cube map + 6-pass depth-only pre-pass via new `shadowCastShader` (shares SKINNED_VS layout/skinning). Cast set = chars + enemy + props; platform doesn't cast (only ground in scene). `SHADOW_CAST_FS` writes linear distance / `shadowFar` to `gl_FragDepth` so the receiver compares distances directly — no per-face unprojection. `SKINNED_FS` and `PLATFORM_VS/FS` split hero from non-hero lights so the per-fragment shadow factor only attenuates the hero's contribution; the seven colored seed lights stay unshadowed (Phase-1 scope decision). Build-verified for both ABIs; **not yet device-tested**.

**Decisions recorded for posterity** (made in planning before the session)
- UBO for lights, baseline cap 8, eval sweep 1 / 4 / 8 / 16 / 32.
- Per-fragment lighting for units, per-vertex for terrain/ground.
- Quadratic falloff (atten²), cartoon punch over physical correctness.
- Explosion sprites self-luminous, skip light loop.

**Open questions raised this session**
- **Light source authoring.** Where do lights come from gameplay-side for the Phase-1 baseline? Static seeded scene? Tied to props? Spawned by explosions? A deterministic 8-light layout is needed for reproducible eval; gameplay-coupled lights belong to Phase 5 stress runs.
- **Per-vertex "terrain" target.** The platform is drawn with `gridShader` (unlit, no normals). I'd add a dedicated `PLATFORM_VS/FS` rather than retrofit `GRID_VS`. Confirm before sub-step 1c.
- **Are props units or ground?** `renderPropPlacements` re-uses the skinned shader with `u_hasBones=0`. My read: treat as units (per-fragment) — already happening with sub-step 1a. Confirm.
- **UBO size headroom.** GLES 3.0 guarantees `MAX_UNIFORM_BLOCK_SIZE ≥ 16 KiB`; 32 lights × 32 B = 1 KiB. Plenty.
- **Calibration on device.** `heroLightRadius=5.5` and `heroLightColor={1.2, 1.0, 0.8}` are eyeballed. Tweak after seeing the Poco render.

### Session 2 — 2026-05-17

**Done**
- Reconnected Poco X6 Pro, `installDebug` from `Flare/`. Clean boot: GL Vendor ARM, Mali-G615 MC6, ES 3.2 r44p1. No `Shader compile error`, no `Shadow FBO incomplete`, no `LightBlock not found`.
- Walked the player. Hero contact shadow renders correctly on the platform under the cluster, shape morphs per-pose so the cube map is being sampled live. No acne, no detachment — bias 0.01 holds at this scale.
- Top-right corner tap cycled the green HUD digit through the sweep (`8 → 16 → 32 …`). Scene is visually identical past 8 (only 8 lights seeded; extra iterations are zero-contribution loops, which is exactly what the GPU-cost eval will measure).
- Captures: `Flare/captures/step2_first.png` (default 8 lights, hero shadow visible) and `Flare/captures/step2_cycle1.png` (HUD at 32).
- **Step 1d and Step 2 are now device-verified.** Phase-1 baseline closes after Step 3 (GPU timing).
- **Implemented Step 3** (`ebdf549`): `EXT_disjoint_timer_query` with a 4-slot ring × 3 stages (shadow_cast / platform / skinned). Cyan 7-segment µs HUD below the green light-count digit, plus a 4th sum line. `GL_GPU_DISJOINT_EXT` guard skips stale results during thermal events.
- **Mali quirk discovered.** First attempt used `glGetString(GL_EXTENSIONS)` / `glGetStringi` — neither lists `GL_EXT_disjoint_timer_query` on r44p1 (110 extensions enumerated, none match). But `eglGetProcAddress("glGenQueriesEXT")` returns a valid pointer, the smoke test `fGenQueriesEXT(1, &probe)` succeeds with `glGetError == GL_NO_ERROR`, and queries return real µs numbers. **Fix: skip the extension string entirely, probe via `eglGetProcAddress` + a runtime smoke test.** Worth remembering for the other test devices — Mali drivers will likely repeat this trick for other extensions too.
- **Phase 1 baseline measured.** Poco X6 Pro, 8 active lights, hero shadow on, idle scene: shadow_cast 5910 µs, platform 1999 µs, skinned 3237 µs, sum 11208 µs. Shadow pre-pass is 53% of GPU lighting work — confirms learned visibility predictor (Phase 3 contribution #2) targets the right cost center. Capture: `Flare/captures/step3_timers.png`.

### Session 3 — 2026-05-18

**Done**
- **Phase 2 design pass.** Wrote `Flare/training/DATA_SPEC.md` v0.1 (mobile-capture + Mali GT), then revised to v0.2 (desktop port + higher-quality desktop-rendered GT) after user pushback. v0.2 splits the two roles: **paper baseline** = frozen Phase-1 Mali shadow at 5.9 ms; **training GT** = higher-quality (4096² PCF) shadow rendered on the desktop port. Paper claim shifts from "match Mali quality, cheaper" to "deliver cleaner shadows than the Mali baseline at NPU cost."
- **Mali extension quirk saved to memory.** Driver omits `GL_EXT_disjoint_timer_query` from `glGetString` but `eglGetProcAddress` works. New `project_flare_mali_extension_quirk` memory + MEMORY.md entry so future device-matrix debugging short-circuits.
- **`platform.h` shim landed in `ITHappyGame/`** (`15b8170`). Tiny header that on Android is a forwarder to the existing Android API + GLES3 headers (mobile build unchanged), and on desktop provides `LOGI`/`LOGE` over stderr + an `AAssetManager`/`AAsset` reimpl over `fopen` + `<GL/glew.h>` in place of `<GLES3/gl3.h>`. `game.cpp`, `gltf_model.cpp`, `game.h`, `gltf_model.h` all now include `"platform.h"` instead of Android-specific headers directly. Android `assembleDebug` verified clean.
- **FlareLab/ desktop port stood up** (sibling to `ITHappyGame/`, separate local git repo). CMakeLists with FetchContent for GLFW 3.4 + glew-cmake 2.2.0 (needed `CMAKE_POLICY_VERSION_MINIMUM=3.5` for CMake 4.x compatibility). `main.cpp` opens GLFW window + GL 3.3 core context, calls `glewInit`, and runs the same `Game` lifecycle (init → loop → cleanup). `platform_desktop.cpp` implements the `AAssetManager` shim against `ITHappyGame/app/src/main/assets/` (path resolved at configure time via `FLARE_ASSET_BASE_DIR`). Touch events synthesized from mouse so `Game::onTouchDown/Move/Up` reach the gameplay layer unchanged.
- **First successful FlareLab run on RTX 4060 Ti.** All 5 GLBs loaded, animations parsed, map.bin read, scene renders at 1280×720 visually identical to the Poco baseline. NVIDIA accepts `#version 300 es` shaders on desktop GL out of the box — formal GLES→GL shader prelude swap deferred until AMD/Intel testing matters.
- **FlareLab Step 2 — scripted input record + replay** (`00ef870`). New `src/input_log.{h,cpp}`: `RecordSink` (append-only file) + `ReplaySource` (load-once, drain-per-frame). 14-byte/event format written field-by-field to dodge struct padding. `main.cpp` extended with `Mode::{Live,Record,Replay}` and `--record` / `--replay` CLI flags. Replay forces dt = 1/60 s, makes the window non-resizable so logged pixel coords stay valid, and auto-exits 2 s (120 frames @ 60 Hz) after the last event. Verified end-to-end: 1055-event session captured → replayed → playback visually faithful to recording.

### Session 4 — 2026-05-19

**Done**
- **FlareLab Step 3 — capture mode** (`0b07403`). `--capture <path>` flag writes one 263336-byte Sample (DATA_SPEC v0.2 wire format) every 4th frame to a raw `.bin` (zstd deferred to step 5). New `src/sample_writer.{h,cpp}` owns a 256² depth FBO, blits from default FBO depth via `glBlitFramebuffer`, reads back as f32, packs to f16 with a hand-rolled IEEE binary32→binary16 conversion. Header / light list / depth fields written by-field (no struct padding). Shadow factor field is placeholder zeros until step 4 ships the PCF GT.
- **`Game::lastView` / `Game::lastProj` exposed** (`ITHappyGame@b9a5bfe`). Mirrored at the top of `Game::render` every frame so SampleWriter can read the per-frame camera/projection without poking into the render path. No behavior change on Android; build verified for arm64-v8a + x86_64.
- **DATA_SPEC.md fixed**: wire-format size note corrected from 264960 B → 263336 B (the 264960 was a math error; field-by-field correctly sums to 168 + 1024 + 131072 + 131072).
- **Smoke test passed end-to-end.** Synthetic 2-event replay log (tiny.bin, 28 B) → flarelab.exe `--replay tiny.bin --capture samples_smoke.bin` → 38 samples × 263336 B = 10006768 B exactly, zero remainder. Header fields parse back: `frame_index=0`, UUID matches startup log, view/proj diag matches m4_perspective, `hero_light_pos=(0.05, 1.5, 0)` (player just started moving), `active_light_count=8`, light[0] = warm hero light, light[1] = first seeded colored light. Depth is 95% non-zero bytes (real geometry); shadow_factor is all zeros (placeholder).

## Next concrete step — FlareLab Phase 2 step 4: 4096² PCF cube-shadow GT renderer

The capture pipeline writes `shadow_factor` as placeholder zeros today (step 3). Step 4 fills that field with the high-quality ground truth the Phase-3 visibility predictor learns to imitate. Per DATA_SPEC §2: **4096² depth cube map + 16-sample PCF**, captured as a per-fragment shadow factor in [0,1], downsampled to 256² f16 for storage.

**Design questions to resolve before code:**
- **Render-graph placement.** Existing Phase-1 cube-shadow runs at 512² inside `Game::render`. Cleanest is a *second* cube-shadow path that only fires on capture frames (`g_frameIdx % 4 == 0`), uses a 4096² FBO, and writes its sampled per-fragment factor into a 256² FBO that SampleWriter reads back. The Phase-1 512² path stays untouched so the Mali baseline cost numbers don't shift.
- **PCF kernel choice.** 16-sample stratified Poisson disk vs. fixed 4×4 grid vs. learned sampling pattern? Poisson disk is the conventional pick; 4×4 grid is cheaper but more banding-prone. Probably Poisson.
- **Where the shadow_factor pass renders.** Need a third FBO sized 256×256 (RGBA8 or R8) that runs a full-screen quad over the captured 256² depth + light list and writes shadow factor per texel. Or: render at the native 1280×720 resolution and downsample (like the depth path).
- **Cost.** 4096² is 64× the per-face area of the Phase-1 512² map; 6 faces × 4096² is 100M pixels per capture frame. On RTX 4060 Ti this is ~2 ms — fine for offline GT generation, not realtime.

**Files this will touch:**
- `ITHappyGame/app/src/main/cpp/game.cpp` and `.h` — likely a new opt-in path or a separate renderer class living next to `Game` (TBD in design).
- `FlareLab/src/sample_writer.cpp` — call into the high-quality shadow path, read back 256² shadow factor, replace the zeros in the Sample.

**Phase 2 remaining steps after this:**
- step 5: record 5 input sessions + run capture + zstd-compress (`samples.bin` → `samples.zst`, manifest.json).

**Out of scope until later phases:** model architecture (Phase 3), NNAPI/TFLite (Phase 4), eval harness (Phase 5), user study (Phase 6), paper (Phase 7).

---

## Phase 2 kickoff archive (historical, kept for context)

Phase 2 (2 weeks) is **data generation**: produce the (scene, light_set) → (visibility_per_fragment) and (scene, light_set) → (light_field_grid) training datasets that the Phase-3 NPU models learn from.

**Open design questions to settle before any code:**

1. **Sampling strategy.** Random scenes vs. replay of real gameplay traces? Replay gives realistic light distributions but couples training to the game's current behavior. Random gives coverage but may oversample unlikely configurations. Recommend a *mix*: 70% replay (recorded from instrumented APK runs) + 30% randomized perturbations of replayed scenes.
2. **Ground-truth format for visibility.** Per-fragment shadow factor `[0,1]` from the cube-shadow sampler? Or per-light visibility from the player's POV? Affects model input/output shape decisively.
3. **Scene encoding.** What does the model see as input? Voxel grid? Bone positions + light list? Camera-space depth + light list? Decision drives the encoder architecture.
4. **Storage budget.** A single 1080p frame's per-fragment shadow GT is 2 MB raw. At 30 FPS × 1 minute × 100 sessions = 360 GB. Need lossy compression or per-frame subsampling.
5. **Where data lives.** `Flare/training/data/` (gitignored)? External S3? Local NAS? Persistence story matters because re-running data gen is expensive.

**Concrete first task next session:** write a 1-page design doc covering 1–5, agree on the answers, *then* start the instrumentation pass on `ITHappyGame/` to dump (scene, light_set, GT) tuples per frame. The dump format will live forever — get it right before generating gigabytes.

**Out of scope until later phases:** model architecture (Phase 3), NNAPI/TFLite (Phase 4), user study (Phase 6), paper (Phase 7).
