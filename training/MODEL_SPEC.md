# MODEL_SPEC — Visibility Predictor (Phase 3, draft v0.1.1)

> **v0.1.1 changes (smoke-test findings, see §10.1):** input grew to 9 channels
> (linearized eye-space z replaces raw window depth + validity mask added);
> §5 loss is masked + class-weighted to handle the 94/4.5/1.9 class imbalance.

Best-guess design draft for the **learned visibility predictor** (research contribution #2). Replaces the 5.9 ms Mali cube-shadow pre-pass with NPU inference. **Quality target: tight to the 4096² PCF GT.** **Latency target: < 2 ms on Poco X6 Pro APU 780.** Implementation is gated on a planning round before code lands.

## 1. Problem statement

Given a per-frame camera-space depth field and the hero light position, predict the 256² shadow factor that matches the 4096² PCF cube-shadow GT (DATA_SPEC v0.2, `shadow_factor` field).

- Single-light scope. The seven static colored lights stay unshadowed in the baseline (per Phase-1 decision); the predictor targets the hero light only.
- Per-frame inference. No temporal accumulation in v0.1 (defer to v0.2 if quality regresses on motion-heavy sessions).

## 2. Inputs (256² × 9 channels)

Assembled by the data-loader from the captured Sample. **The wire format does not change** — everything is derived from the existing DATA_SPEC v0.2 fields.

| Channels | Source | Notes |
|---|---|---|
| 3 | `world_pos = invVP · ndc(depth)` | Reconstructed per-pixel. Zeroed on background pixels (see ch 8) so the network doesn't see far-plane outliers. |
| 1 | `linear_eye_z / 50` (clipped to [0, 1]) | Eye-space z, derived from captured window depth + extracted (znear, zfar) from the projection matrix. **Raw window depth is unusable** — the scene's zfar/znear ≈ 1000:1 packs all real geometry into window-depth [0.989, 1.000]. Linearization restores dynamic range to ~[0.17, 0.26] on valid pixels. |
| 3 | `(hero_light_pos − world_pos)` | Per-pixel vector toward hero light. Encodes the shadow-ray geometry. Zeroed on background. |
| 1 | `1 / max(ε, ‖hero_light_pos − world_pos‖)` | Inverse distance. Cheap falloff-correlated occlusion signal. Zeroed on background. |
| 1 | `valid_mask = (depth < 0.9995)` | 1 where pixel hit real geometry, 0 at far-clip background. The captured 256² spans the full screen, so ~14% of pixels are background — the network is told explicitly. |

## 3. Output (256² × 1 channel)

- `shadow_factor ∈ [0, 1]`, f16 at output, sigmoid head.
- Convention matches `SKINNED_FS::shadowFactor` (1 = lit, 0 = shadowed) so the drop-in replacement at the fragment shader is a tensor sample without re-binding.

## 4. Architecture — lightweight U-Net (NPU-friendly first)

Picked U-Net over flat dilated-conv stack and per-light-aggregate because: (a) widest NNAPI delegate op coverage; (b) skip connections preserve shadow-edge crispness, which is exactly the quality axis under test; (c) depthwise-separable variants quantize to INT8 with minimal SSIM loss in published mobile-shadow work.

- **Encoder:** 4 down-blocks, channel widths `[32, 48, 64, 96, 128]`. Each block = Conv3×3 + BN + ReLU6 + DepthwiseConv3×3 + BN + ReLU6, then MaxPool2×2.
- **Bottleneck:** 1×1 conv at 128 channels, two depthwise blocks.
- **Decoder:** 4 up-blocks. Bilinear-resize (NOT transposed conv — better delegate coverage on Hexagon + APU 780 + Exynos NPU) → concat skip → DepthwiseSeparable block.
- **Head:** 1×1 conv → sigmoid.
- **Param budget:** target < 500K. MAC budget at 256² input: target < 30M MACs/frame.
- **Ops used (all standard NNAPI 1.2+):** `Conv2D`, `DepthwiseConv2D`, `Add`, `Concat`, `BN` (folded at export), `ReLU6`, `ResizeBilinear`, `Logistic` (sigmoid).
- **Avoided:** GroupNorm, LayerNorm, attention, transposed conv, SiLU/Swish. All known to fall back to CPU on at least one of the three target NPUs.

## 5. Loss

- **Masked L1 on shadow factor** (primary). Computed only over `valid_mask == 1` pixels — background carries no shadow signal and isn't part of the perceptual task. f16 GT, f32 prediction.
- **Class-weighted scaling.** On-platform pixel distribution is **~93.7% lit / 4.5% shadowed / 1.9% penumbra** (measured across `session_01_idle`, 1830 samples). Naive unweighted L1 lets the network collapse to "output 1 everywhere" for L1 ≈ 0.06. Mitigation: weight shadowed (≤0.05) and penumbra (0.05–0.95) pixels at **3×** lit-pixel weight in the loss reduction. Re-evaluate after first training run; switch to focal loss if collapse persists.
- **Masked L1 on Sobel gradient** (weight 0.3). Drives shadow-edge sharpness — the perceptual axis that distinguishes "good" shadow approximations from "blurry blob" failure modes. Gradient computed before masking, then the gradient itself is masked.
- **No perceptual loss in v0.1.** SSIM is the evaluation metric, not the training metric, to avoid optimizing on the validator.

## 6. Train / val / test split

| Session | Role | Why |
|---|---|---|
| `session_01_idle`  | train | Static cursor → varied skinned-pose distribution |
| `session_02_slow`  | train | Slow walk → mid-range receiver motion |
| `session_03_chase` | train | Perimeter chase → full receiver-position coverage |
| `session_04_scrum` | train | Cursor on enemy → tight clustering, lots of overlap shadowing |
| `session_05_empty` | **held out** | Far-corner cursor → cleanest "did the network actually generalize" signal |

Within the 4 train sessions: 90/10 random split for early-stopping validation. ~6588 train + 732 val + 1830 test samples.

## 7. Training recipe

- PyTorch on RTX 4060 Ti, FP16 mixed precision.
- AdamW, lr 3e-4 cosine → 1e-5, 100 epochs, batch 32.
- Augmentation: horizontal flip (scene is symmetric under x-flip around the platform axis). No rotation or crop — would invalidate the world-pos channels.

## 8. Mobile deployment path (Phase 4 preview, locked here)

1. Train FP32 PyTorch checkpoint.
2. Export ONNX (opset 13). Run `onnx-simplifier`.
3. Convert ONNX → TFLite via `onnx2tflite`. Sanity-check no CPU fallback ops.
4. Post-training INT8 quantization with `tf.lite.TFLiteConverter` + representative dataset (100 samples from `session_05_empty`).
5. NNAPI delegate at runtime; CPU XNNPACK fallback path validated separately for the no-NPU budget device.

## 9. Success criteria

| Axis | Target | Source |
|---|---|---|
| **Quality (FP32)** | SSIM ≥ 0.98 vs 4096² PCF GT on held-out `session_05_empty` | Placeholder; Phase 5 perceptual study sets final |
| **Quality (INT8)**| SSIM ≥ 0.975 (post-quantization regression < 0.005) | Standard mobile-NN tolerance |
| **Latency (APU 780, Poco X6 Pro)** | < 2 ms / frame | < 1/3 of the 5.9 ms Mali shadow pre-pass |
| **Latency (Exynos 1580, Hexagon)** | < 3 ms / frame | Budget margin for cross-device variance |
| **Parameter count** | < 500K | Disk size, cold-start load time |

## 10. Open research risk — captured depth is camera-POV, not light-POV

The network has to infer the geometry between the receiver point and the hero light from a top-down camera depth image. The platform is flat, occluders are 4 skinned characters + 1 boss enemy with known general locations, and the hero light tracks the player — so the space of "what could occlude" is small and learnable. Published mobile neural-shadow work succeeds with similar setups.

**But if v0.1 bottoms out below SSIM 0.95**, the likely fix is to extend DATA_SPEC to include a low-res (64² × 6 faces ≈ 25 KB) cube shadow depth as additional input. That would invalidate Dataset V and require re-capture. Fallback plan, not a v0.1 plan.

## 10.1 Smoke-test findings (v0.1.1 pivot)

`Flare/training/smoke_dataset.py` parsed all 1830 samples of `session_01_idle` and exposed two issues with the v0.1 input plan:

1. **Captured depth has no usable dynamic range.** Projection matrix decode shows zfar/znear ≈ 1000:1. All real-scene pixels sit in window-depth `[0.989, 1.000]`. Feeding raw depth to the network would give it near-constant signal. **Fix:** linearize to eye-space z (channel 3 in v0.1.1) using `(znear, zfar)` extracted per-sample from `proj[2,2]` / `proj[2,3]`. On valid pixels the resulting normalized channel has range `[0.17, 0.26]` — proper variance.
2. **~14% of every captured frame is far-clip background** (off-platform sky region). World-pos reconstruction for those pixels blows up to ±80 units (far plane). Without explicit signalling the network would learn the degenerate "if `world.y < -10` output `1`" rule and bypass the actual visibility problem. **Fix:** add channel 8 `valid_mask`, zero out world / delta / inv_dist on background pixels, mask the training loss.
3. **Even with masking, on-platform classes are skewed 94/4.5/1.9.** Shadowed and penumbra pixels are rare. Loss must be class-weighted or focal; tracked in §5.

These are encoding / loss issues, not architectural or wire-format issues. Dataset V stays frozen.

## 11. Implementation gate

Per `CLAUDE.md` "plan-before-implement": discuss this spec, lock decisions, then write the training loop. No `model.py` until §2/§4/§5/§9 are signed off.
