# PROJECT_STATE — Flare

Single source of truth for "where is the research right now". Update at the end of every session.

## Current position

- **Phase 1** — Baseline and instrumentation (week 1 of 3)
- **Step 1 closed** (all of 1a–1d shipped; 1a–1c device-confirmed, 1d build-verified pending device test).
- **Step 2 landed in code, pending device test.** Point-light cube shadow mapping for the hero light is the GPU baseline the learned visibility predictor (contribution #2) gets benchmarked against in Phase 5.
- **Game-side commits (in `../ITHappyGame/`):**
  - `0934f78` — Checkpoint (pre-Flare asset refresh; retreat point).
  - `1160bea` — Flare 1a (single hero point light, 0.5× chars, dim ambient, 20% zoom).
  - `8526ba1` — Flare 1b (std140 UBO `LightBlock`, 8-light cap, deterministic colored seed scene).
  - `5a7ad76` — Flare 1c (`PLATFORM_VS/FS` per-vertex ground lighting sharing the UBO; unlit line grid kept on `gridShader`).
  - `544e6a8` — Flare 1d (top-right corner tap cycles `activeLightCount` 1/4/8/16/32; green HUD digit).
  - `47269ea` — Flare Step 2 (512² depth cube map, 6-pass shadow caster for hero light only; SKINNED_FS + PLATFORM_VS/FS split hero from non-hero so per-fragment shadow factor cleanly attenuates only its contribution).

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

## Next concrete step — device verification of 1d + Step 2

The Poco was disconnected mid-session, so the last two commits are build-verified only. First task next session:
1. Reconnect Poco X6 Pro, `cd ../ITHappyGame && ./gradlew installDebug`, capture initial logcat (`adb logcat -d -s ITHappyGame:V`) — look for `Shader compile error`, `Shadow FBO incomplete`, or `LightBlock not found` lines.
2. Walk the player. Hero light should cast a contact shadow on the platform under the player's feet and on followers when they line up between the player and any of the seven seeded colored lights. The seven colored lights themselves stay unshadowed (full color contribution to visible casters in their range).
3. Tap top-right corner to cycle `activeLightCount`. The green HUD digit should tick `1 → 4 → 8 → 16 → 32 → 1`.
4. Capture a screenshot via `adb exec-out screencap -p > Flare/captures/step2.png` — Claude can read it directly.

Open issues to expect on first run:
- **Shadow bias.** Hardcoded at `0.01` against normalized `length(fragToLight) / shadowFar`. Likely needs a pass: too low → acne, too high → floating shadows.
- **`gl_FragDepth` precision.** Mali-G615 should be fine but flicker at shadow edges = raise to `highp` or pack depth into a color attachment.
- **6-pass cost.** Subjective on-device. If FPS visibly drops, halve `shadowMapSize` to 256.

After verification, **Step 3** opens: GPU timing queries (`EXT_disjoint_timer_query`) so the cost curve is measurable instead of subjective. That closes Phase 1.

**Out of scope until later phases:** NPU integration, training-data generation, ONNX/TFLite/NNAPI, the user study.
