# MODEL_SPEC — Visibility Predictor (Phase 3, draft v0.1)

Best-guess design draft for the **learned visibility predictor** (research contribution #2). Replaces the 5.9 ms Mali cube-shadow pre-pass with NPU inference. **Quality target: tight to the 4096² PCF GT.** **Latency target: < 2 ms on Poco X6 Pro APU 780.** Implementation is gated on a planning round before code lands.

## 1. Problem statement

Given a per-frame camera-space depth field and the hero light position, predict the 256² shadow factor that matches the 4096² PCF cube-shadow GT (DATA_SPEC v0.2, `shadow_factor` field).

- Single-light scope. The seven static colored lights stay unshadowed in the baseline (per Phase-1 decision); the predictor targets the hero light only.
- Per-frame inference. No temporal accumulation in v0.1 (defer to v0.2 if quality regresses on motion-heavy sessions).

## 2. Inputs (256² × 8 channels)

Assembled by the data-loader from the captured Sample. **The wire format does not change** — everything is derived from the existing DATA_SPEC v0.2 fields.

| Channels | Source | Notes |
|---|---|---|
| 3 | `world_pos = invVP · ndc(depth)` | Reconstructed per-pixel in the data-loader. Cheap matrix mul; no inference-time cost. |
| 1 | `depth_f16` | The captured eye-space depth, passed through as-is. |
| 3 | `(hero_light_pos − world_pos)` broadcast | Per-pixel vector to the hero light. Encodes the "ray we'd shoot at the shadow map" geometrically. |
| 1 | `1 / max(ε, length(hero_light_pos − world_pos))` | Inverse distance. Helps the network learn falloff-correlated occlusion patterns cheaply. |

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

- **L1 on shadow factor** (primary). f16 GT, f32 prediction.
- **L1 on Sobel gradient of shadow factor** (weight 0.3). Drives edge sharpness explicitly — the perceptual axis that distinguishes "good" shadow approximations from "blurry blob" failure modes.
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

## 11. Implementation gate

Per `CLAUDE.md` "plan-before-implement": discuss this spec, lock decisions, then write the training loop. No `model.py` until §2/§4/§5/§9 are signed off.
