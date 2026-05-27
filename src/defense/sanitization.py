"""
defense/sanitization.py
-----------------------
Layer 1 — Input Sanitization (Image Processing)

Implements the pre-inference sanitization pipeline described in the proposal:

  1. Adaptive Median Filtering   — removes high-freq adversarial noise
  2. Bilateral Denoising         — edge-preserving smoothing
  3. Bit-Depth Squeezing         — quantizes away fine perturbations
  4. Spatial Transformations     — random crop/rotate disrupts alignment

All transforms operate on PyTorch tensors (B, C, H, W) in [0.0, 1.0].
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.restoration import denoise_bilateral


# ──────────────────────────────────────────────────────────────────────────────
# 1. Adaptive Median Filter
# ──────────────────────────────────────────────────────────────────────────────
def adaptive_median_filter(
    images: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Apply a median filter per-channel using a sliding window.

    Targets salt-and-pepper and high-frequency adversarial noise by replacing
    each pixel with the median of its local neighbourhood.

    Args:
        images      : (B, C, H, W) float tensor in [0, 1]
        kernel_size : neighbourhood size (must be odd)

    Returns:
        filtered tensor, same shape
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    B, C, H, W = images.shape
    pad = kernel_size // 2

    # Pad with reflection to preserve border pixels
    x = F.pad(images, [pad]*4, mode="reflect")

    # Unfold into patches: (B, C, H, W, k, k)
    x = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # Flatten patch → median
    x = x.contiguous().view(B, C, H, W, -1).median(dim=-1).values

    return x.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Bilateral Denoising
# ──────────────────────────────────────────────────────────────────────────────
def bilateral_denoise(
    images: torch.Tensor,
    sigma_color: float = 0.05,
    sigma_spatial: float = 3.0,
) -> torch.Tensor:
    """
    Edge-preserving bilateral filter via scikit-image.

    Reduces high-frequency adversarial noise while preserving semantic
    boundaries (edges) that are important for correct classification.

    Args:
        images         : (B, C, H, W) float tensor in [0, 1]
        sigma_color    : range sigma — controls which intensities are averaged
        sigma_spatial  : spatial sigma — controls neighbourhood radius

    Returns:
        denoised tensor, same shape
    """
    B, C, H, W = images.shape
    out = torch.zeros_like(images)

    for i in range(B):
        # Convert to numpy (H, W, C) for skimage
        img_np = images[i].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        if C == 1:
            img_np = img_np[:, :, 0]

        denoised = denoise_bilateral(
            img_np,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            channel_axis=-1 if C > 1 else None,
        )

        if C == 1:
            denoised = denoised[:, :, np.newaxis]

        out[i] = torch.from_numpy(denoised.astype(np.float32)).permute(2, 0, 1)

    return out.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Bit-Depth Squeezing
# ──────────────────────────────────────────────────────────────────────────────
def bit_depth_squeeze(
    images: torch.Tensor,
    bits: int = 5,
) -> torch.Tensor:
    """
    Reduce colour depth by quantising pixel values to `bits` bits.

    Adversarial perturbations rely on fine-grained pixel manipulations
    (often < 1/255 per channel). Squeezing to 5-bit (32 levels) eliminates
    perturbations smaller than 1/32, which is above the perturbation budget ε.

    Args:
        images : (B, C, H, W) float tensor in [0, 1]
        bits   : target bit depth (typical: 4–6)

    Returns:
        squeezed tensor in [0, 1]
    """
    assert 1 <= bits <= 8, "bits must be in [1, 8]"
    levels = 2 ** bits - 1
    return (images * levels).round() / levels


# ──────────────────────────────────────────────────────────────────────────────
# 4. Spatial Transformations
# ──────────────────────────────────────────────────────────────────────────────
def spatial_transform(
    images: torch.Tensor,
    crop_frac: float = 0.9,
    max_rotate_deg: float = 5.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Apply random crop and small rotation to disrupt adversarial alignment.

    Adversarial perturbations are computed for a specific spatial layout;
    even a small crop or rotation breaks their pixel-level alignment.

    Args:
        images          : (B, C, H, W) float tensor in [0, 1]
        crop_frac       : fraction of original size to keep (e.g., 0.9 → 90%)
        max_rotate_deg  : maximum rotation in degrees
        seed            : optional RNG seed for reproducibility

    Returns:
        transformed tensor, same spatial shape as input (resized back)
    """
    if seed is not None:
        torch.manual_seed(seed)

    B, C, H, W = images.shape
    out = torch.zeros_like(images)

    crop_h = int(H * crop_frac)
    crop_w = int(W * crop_frac)

    for i in range(B):
        img = images[i]  # (C, H, W)

        # Random crop
        top  = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
        img  = img[:, top:top+crop_h, left:left+crop_w]

        # Resize back to original dimensions
        img = F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

        # Random rotation via OpenCV
        angle   = (torch.rand(1).item() * 2 - 1) * max_rotate_deg
        img_np  = img.permute(1, 2, 0).numpy()
        M       = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        rotated = cv2.warpAffine(img_np, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if C == 1:
            rotated = rotated[:, :, np.newaxis]
        out[i] = torch.from_numpy(rotated).permute(2, 0, 1)

    return out.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Full Sanitization Pipeline
# ──────────────────────────────────────────────────────────────────────────────
class SanitizationPipeline:
    """
    Composes all four sanitization transforms in sequence.

    The pipeline is applied before any model inference:
        raw input → median filter → bilateral denoise → bit squeeze → spatial transform → model

    Args:
        median_kernel    : kernel size for adaptive median filter (default 3)
        sigma_color      : bilateral filter range sigma
        sigma_spatial    : bilateral filter spatial sigma
        squeeze_bits     : bit depth for squeezing (default 5)
        crop_frac        : spatial crop fraction
        max_rotate_deg   : max rotation angle
        use_median       : toggle median filter on/off
        use_bilateral    : toggle bilateral denoise on/off
        use_squeeze      : toggle bit-depth squeeze on/off
        use_spatial      : toggle spatial transform on/off
    """

    def __init__(
        self,
        median_kernel:  int   = 3,
        sigma_color:    float = 0.05,
        sigma_spatial:  float = 3.0,
        squeeze_bits:   int   = 5,
        crop_frac:      float = 0.9,
        max_rotate_deg: float = 5.0,
        use_median:     bool  = True,
        use_bilateral:  bool  = True,
        use_squeeze:    bool  = True,
        use_spatial:    bool  = False,   # off by default (stochastic at eval time)
    ):
        self.median_kernel  = median_kernel
        self.sigma_color    = sigma_color
        self.sigma_spatial  = sigma_spatial
        self.squeeze_bits   = squeeze_bits
        self.crop_frac      = crop_frac
        self.max_rotate_deg = max_rotate_deg
        self.use_median     = use_median
        self.use_bilateral  = use_bilateral
        self.use_squeeze    = use_squeeze
        self.use_spatial    = use_spatial

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the full pipeline to a batch of images."""
        x = images.float()

        if self.use_median:
            x = adaptive_median_filter(x, kernel_size=self.median_kernel)

        if self.use_bilateral:
            x = bilateral_denoise(x, sigma_color=self.sigma_color,
                                     sigma_spatial=self.sigma_spatial)

        if self.use_squeeze:
            x = bit_depth_squeeze(x, bits=self.squeeze_bits)

        if self.use_spatial:
            x = spatial_transform(x, crop_frac=self.crop_frac,
                                      max_rotate_deg=self.max_rotate_deg)

        return x

    def __repr__(self) -> str:
        active = []
        if self.use_median:   active.append(f"MedianFilter(k={self.median_kernel})")
        if self.use_bilateral: active.append(f"BilateralDenoise(σc={self.sigma_color}, σs={self.sigma_spatial})")
        if self.use_squeeze:  active.append(f"BitSqueeze(bits={self.squeeze_bits})")
        if self.use_spatial:  active.append(f"SpatialTransform(crop={self.crop_frac}, rot={self.max_rotate_deg}°)")
        return f"SanitizationPipeline([{', '.join(active)}])"

# ──────────────────────────────────────────────────────────────────────────────
# Canonical pipeline order — used by SAS metric in evaluation/evaluator.py
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_PIPELINE_ORDER = [
    "adaptive_median_filter",
    "bilateral_denoise",
    "bit_depth_squeeze",
]

