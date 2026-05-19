"""Visibility-predictor loss — per MODEL_SPEC §5.

Combined objective:
    total = masked_class_weighted_L1(pred, target)
          + sobel_weight * masked_L1(sobel(pred), sobel(target))

Masking: only pixels with `mask == 1` (real scene geometry, channel 8 of the
dataset input) contribute. Background pixels are out-of-task and would
otherwise bias the network toward predicting "lit" by default.

Class weighting: on-platform shadow distribution is ~94% lit / 4.5% shadowed
/ 1.9% penumbra (measured on session_01_idle). Without weighting the network
collapses to "output 1 everywhere" for L1 ≈ 0.06. Shadowed + penumbra pixels
(target < 0.95) are weighted at `hard_class_weight` × the lit-pixel weight.

Sobel-gradient term: drives shadow-edge sharpness, the perceptual axis that
distinguishes a clean PCF replica from a blurry-blob shortcut.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_SOBEL_X = torch.tensor(
    [[[[-1.0, 0.0, 1.0],
       [-2.0, 0.0, 2.0],
       [-1.0, 0.0, 1.0]]]]
)
_SOBEL_Y = torch.tensor(
    [[[[-1.0, -2.0, -1.0],
       [ 0.0,  0.0,  0.0],
       [ 1.0,  2.0,  1.0]]]]
)


def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    """Returns [B, 2, H, W] sobel gradients (x then y) for [B, 1, H, W] input."""
    kx = _SOBEL_X.to(x.device, x.dtype)
    ky = _SOBEL_Y.to(x.device, x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.cat([gx, gy], dim=1)


class VisibilityLoss(nn.Module):
    def __init__(
        self,
        sobel_weight: float = 0.3,
        hard_class_weight: float = 3.0,
        lit_threshold: float = 0.95,
    ):
        super().__init__()
        self.sobel_weight = sobel_weight
        self.hard_class_weight = hard_class_weight
        self.lit_threshold = lit_threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """All inputs are [B, 1, H, W]. Returns dict with backward-able 'total'
        and detached 'l1' / 'sobel' for logging."""
        is_hard = (target < self.lit_threshold).to(pred.dtype)
        weight = mask * (1.0 + (self.hard_class_weight - 1.0) * is_hard)

        denom = weight.sum().clamp_min(1.0)
        l1 = (torch.abs(pred - target) * weight).sum() / denom

        grad_pred = sobel_grad(pred)
        grad_tgt = sobel_grad(target)
        mask_grad = mask.expand_as(grad_pred)
        denom_g = mask_grad.sum().clamp_min(1.0)
        l1_sobel = (torch.abs(grad_pred - grad_tgt) * mask_grad).sum() / denom_g

        total = l1 + self.sobel_weight * l1_sobel
        return {
            "total": total,
            "l1": l1.detach(),
            "sobel": l1_sobel.detach(),
        }


def masked_ssim(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Per-image masked SSIM, averaged over the batch. Returns a scalar.

    Wang et al. 2004 SSIM with a Gaussian window. Background pixels (mask==0)
    are excluded from the per-image mean of the SSIM map; the local SSIM
    computation itself still runs convolutionally over the full image so the
    window remains shift-invariant (no edge artefacts from masking inside the
    Gaussian).
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    coords = torch.arange(window, dtype=pred.dtype, device=pred.device) - (window - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * (1.5 ** 2)))
    g = (g / g.sum()).view(1, 1, window, 1)
    g_kernel_x = g
    g_kernel_y = g.transpose(2, 3)

    def _blur(x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, g_kernel_x, padding=(window // 2, 0))
        x = F.conv2d(x, g_kernel_y, padding=(0, window // 2))
        return x

    mu_p = _blur(pred)
    mu_t = _blur(target)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t
    sig_p2 = _blur(pred * pred) - mu_p2
    sig_t2 = _blur(target * target) - mu_t2
    sig_pt = _blur(pred * target) - mu_pt

    ssim_map = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / \
               ((mu_p2 + mu_t2 + c1) * (sig_p2 + sig_t2 + c2))

    # Per-image masked mean.
    m = mask.to(ssim_map.dtype)
    denom = m.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return ((ssim_map * m).sum(dim=(1, 2, 3)) / denom).mean()


if __name__ == "__main__":
    # Smoke test on synthetic tensors.
    torch.manual_seed(0)
    B = 2
    pred = torch.rand(B, 1, 64, 64)
    target = torch.rand(B, 1, 64, 64)
    mask = (torch.rand(B, 1, 64, 64) > 0.2).float()
    loss = VisibilityLoss()
    out = loss(pred, target, mask)
    s = masked_ssim(pred, target, mask)
    print(f"total={out['total'].item():.4f}  l1={out['l1'].item():.4f}  "
          f"sobel={out['sobel'].item():.4f}  ssim={s.item():.4f}")
