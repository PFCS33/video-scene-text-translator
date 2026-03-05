"""Image processing utilities: sharpness, contrast, histogram matching."""

from __future__ import annotations

import cv2
import numpy as np


def compute_sharpness(image: np.ndarray) -> float:
    """Compute image sharpness via Laplacian variance.

    Higher values = sharper. Normalized to [0, 1] via tanh mapping.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(np.tanh(variance / 1000.0))


def compute_contrast(image: np.ndarray) -> float:
    """Compute local contrast as coefficient of variation (std/mean).

    Text regions with high contrast (dark on light or vice versa) score higher.
    Returns value in [0, 1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean = gray.mean()
    if mean < 1e-6:
        return 0.0
    std = gray.std()
    return float(np.clip(std / mean, 0, 1))


def match_histogram_luminance(
    source: np.ndarray,
    reference: np.ndarray,
    color_space: str = "YCrCb",
) -> np.ndarray:
    """Match the luminance histogram of source to reference.

    Only the luminance (Y/L) channel is matched; chrominance channels
    from the source are preserved to keep the translated text's color.

    Args:
        source: BGR image to adjust (the translated ROI).
        reference: BGR image to match against (original ROI from target frame).
        color_space: "YCrCb" or "LAB".

    Returns:
        Adjusted BGR image with matched luminance.
    """
    if color_space == "YCrCb":
        src_cvt = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
        ref_cvt = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb)
        back_cvt = cv2.COLOR_YCrCb2BGR
    elif color_space == "LAB":
        src_cvt = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        ref_cvt = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        back_cvt = cv2.COLOR_LAB2BGR
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    src_channels = list(cv2.split(src_cvt))
    ref_channels = cv2.split(ref_cvt)

    src_channels[0] = _match_single_channel_histogram(
        src_channels[0], ref_channels[0]
    )

    result_cvt = cv2.merge(src_channels)
    return cv2.cvtColor(result_cvt, back_cvt)


def _match_single_channel_histogram(
    source: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Match histogram of a single-channel image using CDF mapping.

    Builds a 256-entry lookup table from source CDF to reference CDF.
    """
    src_hist, _ = np.histogram(source.flatten(), bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(reference.flatten(), bins=256, range=(0, 256))

    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)

    src_cdf /= src_cdf[-1] + 1e-10
    ref_cdf /= ref_cdf[-1] + 1e-10

    lookup = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        diff = np.abs(ref_cdf - src_cdf[src_val])
        lookup[src_val] = np.argmin(diff)

    return lookup[source]
