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


def compute_contrast_otsu(image: np.ndarray) -> float:
    """Compute contrast via Otsu's interclass variance, normalized to [0, 1].

    Otsu's method finds the threshold that maximizes the between-class variance
    of foreground and background pixel intensities. This directly measures how
    well-separated text is from its background — aligned with STRIVE's approach.

    The interclass variance is normalized by dividing by the maximum possible
    variance (occurs when pixels are split between 0 and 255).

    Returns:
        Score in [0, 1], where higher = better text/background separation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    if gray.size == 0:
        return 0.0

    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Compute interclass variance at the Otsu threshold
    fg = gray[gray > threshold]
    bg = gray[gray <= threshold]

    if fg.size == 0 or bg.size == 0:
        return 0.0

    w_fg = fg.size / gray.size
    w_bg = bg.size / gray.size
    mean_fg = fg.mean()
    mean_bg = bg.mean()

    interclass_variance = w_fg * w_bg * (mean_fg - mean_bg) ** 2

    # Maximum possible interclass variance: when pixels are split 50/50
    # between 0 and 255 → 0.25 * 255^2 = 16256.25
    max_variance = 0.25 * 255.0 ** 2
    return float(np.clip(interclass_variance / max_variance, 0, 1))


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
