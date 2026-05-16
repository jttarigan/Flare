# PROJECT_STATE — Flare

Single source of truth for "where is the research right now". Update at the end of every session.

## Current position

- **Phase 1** — Baseline and instrumentation (week 1 of 3)
- **Step 1 — Forward N-light shading: shader work complete (1a + 1b + 1c shipped to device, user-confirmed on Mali-G615 / Poco X6 Pro).** Only the runtime `activeLightCount` knob + HUD digit remains before Step 1 closes.
- **Game-side commits (in `../ITHappyGame/`):**
  - `0934f78` — Checkpoint (pre-Flare asset refresh; retreat point).
  - `1160bea` — Flare 1a (single hero point light, 0.5× chars, dim ambient, 20% zoom).
  - `8526ba1` — Flare 1b (std140 UBO `LightBlock`, 8-light cap, deterministic colored seed scene).
  - `5a7ad76` — Flare 1c (`PLATFORM_VS/FS` per-vertex ground lighting sharing the UBO; unlit line grid kept on `gridShader`).

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

## Next concrete step

**Sub-step 1d — runtime `activeLightCount` knob + HUD digit.** Closes Step 1 by making the eval sweep on-device-toggleable without a rebuild.
1. Map an unused input gesture (e.g. triple-tap on the right HUD area, or three-finger tap anywhere — joystick owns the left half, double-tap right already does the ACSCull toggle) to cycle `activeLightCount` through 1 → 4 → 8 → 16 → 32 → 1.
2. Add a 5th line to `renderTriCount` (or a separate 1-digit scissor block) showing the current count, so screenshots from Phase 5 sweeps are self-labelling.
3. Sanity-check: with `activeLightCount = 0`, scene should fall back to pure ambient on both ground and characters (good regression check that the shader paths share their fallback correctly).

Once that lands, **Step 1 closes** and Phase 1, Step 2 (point-light cube shadow mapping) opens.

## Later — Phase 1, Step 2

**Topic:** Point-light cube shadow mapping.

**Pre-reads**
- Whatever N-light implementation lands from Step 1.
- ES 3.0 depth cubemap path: `GL_TEXTURE_CUBE_MAP` + `GL_DEPTH_COMPONENT24` + per-face attachment (no geometry shaders in ES 3.0 → six render passes, one per cube face).

**First task:** Single-light cube shadow map at fixed res (e.g. 256² × 6 faces) for one designated "hero" light, hard PCF. Then sweep light count × shadow res for the per-light/per-pixel cost curve that motivates the visibility predictor.

**Out of scope until later phases:** NPU integration, training-data generation, ONNX/TFLite/NNAPI, the user study.
