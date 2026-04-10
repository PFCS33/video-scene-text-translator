"""Pure-logic helpers for AnyText2 adaptive mask sizing.

When the source text mask is much wider than the translated text needs,
AnyText2 fills the empty space with gibberish characters. This module
provides the math to:

1. Estimate the target text's natural pixel width at a given height
2. Compute a centered, aspect-matched mask rectangle inside the canonical
3. Restore a "middle strip" of original pixels onto an inpainted background
   so the mask region contains a valid text-to-replace anchor

The functions here have zero external dependencies beyond numpy and are
unit-tested in isolation without any gradio client or inpainter.

See plan.md and docs/sessions/ for design rationale and the STE length-
mismatch limitation this works around.
"""

from __future__ import annotations

import numpy as np

# Width of each character class as a multiple of canonical height.
# Values are approximate defaults that avoid font discovery; they trade
# ±15% accuracy for zero font dependencies. See plan.md D3.
_CJK_WIDTH = 1.00       # Full-width ideographs, kana, hangul
_LATIN_UPPER_WIDTH = 0.60
_LATIN_LOWER_WIDTH = 0.50
_DIGIT_WIDTH = 0.55
_SPACE_WIDTH = 0.30
_OTHER_WIDTH = 0.55     # Punctuation, symbols, unsupported scripts


def _is_cjk(codepoint: int) -> bool:
    """Return True if a Unicode codepoint is a full-width CJK character.

    Covers the common CJK Unified Ideographs blocks, Hiragana, Katakana,
    Hangul, and CJK punctuation. Not exhaustive — but good enough for the
    scripts this pipeline actually targets.
    """
    return (
        0x3040 <= codepoint <= 0x309F    # Hiragana
        or 0x30A0 <= codepoint <= 0x30FF  # Katakana
        or 0x3400 <= codepoint <= 0x4DBF  # CJK Unified Ideographs Ext A
        or 0x4E00 <= codepoint <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= codepoint <= 0xD7AF  # Hangul Syllables
        or 0xF900 <= codepoint <= 0xFAFF  # CJK Compatibility Ideographs
        or 0xFF00 <= codepoint <= 0xFF60  # Fullwidth ASCII variants
        or 0xFFE0 <= codepoint <= 0xFFE6  # Fullwidth signs
    )


def _char_width_ratio(char: str) -> float:
    """Return the width-to-height ratio for a single character."""
    cp = ord(char)
    if _is_cjk(cp):
        return _CJK_WIDTH
    if char == " ":
        return _SPACE_WIDTH
    if char.isdigit():
        return _DIGIT_WIDTH
    if char.isupper():
        return _LATIN_UPPER_WIDTH
    if char.islower():
        return _LATIN_LOWER_WIDTH
    return _OTHER_WIDTH


def estimate_target_width(text: str, height: int) -> float:
    """Estimate the natural pixel width of *text* rendered at *height*.

    Uses a character-class heuristic — no font files required. Each
    character contributes a fraction of *height* based on its class
    (CJK, Latin upper/lower, digit, space, or other).

    Args:
        text: The translated string.
        height: Target render height in pixels (usually the canonical height).

    Returns:
        Estimated width in pixels, as a float.  Returns 0 for empty input or
        non-positive height.  Caller is responsible for rounding / clamping.

    Example:
        >>> estimate_target_width("我是示", height=80)  # 3 × 1.0 × 80
        240.0
        >>> estimate_target_width("DANGER", height=80)   # 6 × 0.60 × 80
        288.0
    """
    if not text or height <= 0:
        return 0
    total_ratio = sum(_char_width_ratio(c) for c in text)
    return total_ratio * height


def compute_adaptive_mask_rect(
    canonical_w: int,
    canonical_h: int,
    target_text: str,
    tolerance: float,
) -> tuple[int, int, int, int] | None:
    """Compute a centered mask rectangle for AnyText2 adaptive mask sizing.

    Returns the ``(top, bottom, left, right)`` slice of a shrunk mask
    horizontally centered inside the canonical text area, or ``None`` if
    the adaptive flow should be skipped for this track.

    Returns ``None`` when:
    - *target_text* is empty (nothing to render)
    - Canonical dimensions are degenerate (``≤ 0``)
    - Aspect mismatch between source and target is within *tolerance*
      (close enough to the source aspect — current behaviour is fine)
    - Target's natural aspect is **wider** than the source canonical
      (long→short only; we never grow the mask — see plan.md D9)

    When a rectangle is returned, its width is clamped to at most
    *canonical_w* (never wider than the source canonical).

    Args:
        canonical_w: Width of the canonical (frontalized) text area in pixels.
        canonical_h: Height of the canonical text area in pixels.
        target_text: The translated string that will be rendered.
        tolerance: If ``|target_aspect - source_aspect| / source_aspect``
            is strictly less than this, skip the adaptive flow.

    Returns:
        ``(top, bottom, left, right)`` integer slice or ``None``.
    """
    if not target_text or canonical_w <= 0 or canonical_h <= 0:
        return None

    source_aspect = canonical_w / canonical_h
    target_width_px = estimate_target_width(target_text, canonical_h)
    if target_width_px <= 0:
        return None

    target_aspect = target_width_px / canonical_h

    # Long→short only: never grow the mask
    if target_aspect >= source_aspect:
        return None

    # Close enough to source aspect → skip (fast path)
    mismatch = abs(target_aspect - source_aspect) / source_aspect
    if mismatch < tolerance:
        return None

    # Clamp to at most canonical_w
    mask_w = int(round(target_width_px))
    mask_w = min(mask_w, canonical_w)

    # Centered horizontally
    left = (canonical_w - mask_w) // 2
    right = left + mask_w
    return (0, canonical_h, left, right)


def restore_middle_strip(
    inpainted: np.ndarray,
    original: np.ndarray,
    mask_rect: tuple[int, int, int, int],
    feather_px: int = 3,
) -> np.ndarray:
    """Composite a strip of *original* pixels onto an *inpainted* background.

    Used by the adaptive mask flow: after SRNet erases all source text from
    the canonical, this restores the center strip (the part that will be
    masked for AnyText2) so that AnyText2 sees a valid "text to replace"
    anchor inside the mask and a clean background outside.

    Args:
        inpainted: The fully-inpainted canonical ROI (source text erased).
            Shape: ``(H, W, 3)``, uint8.
        original: The un-inpainted canonical ROI with source text intact.
            Must match *inpainted* in shape and dtype.
        mask_rect: ``(top, bottom, left, right)`` of the strip to restore,
            in the coordinate system of the two input arrays.
        feather_px: Linear alpha feather width at the left/right strip
            boundaries, in pixels.  ``0`` = hard cut, no blending.

    Returns:
        A new BGR array (never a view or mutation of the inputs) with the
        strip composited onto the inpainted background.

    Raises:
        ValueError: If *inpainted* and *original* shapes differ.
    """
    if inpainted.shape != original.shape:
        msg = (
            f"inpainted shape {inpainted.shape} does not match "
            f"original shape {original.shape}"
        )
        raise ValueError(msg)

    top, bottom, left, right = mask_rect
    h, w = inpainted.shape[:2]

    # Clamp to canvas bounds
    top = max(0, top)
    bottom = min(h, bottom)
    left = max(0, left)
    right = min(w, right)

    if right <= left or bottom <= top:
        # Zero-area strip → nothing to restore
        return inpainted.copy()

    # Build a per-column alpha mask: 1.0 inside the strip, 0.0 far outside,
    # linear ramp across the feather zone at each boundary.
    alpha_row = np.zeros(w, dtype=np.float32)
    alpha_row[left:right] = 1.0

    if feather_px > 0:
        # Left feather: ramp from 0 to 1 across [left-feather_px, left]
        for i in range(1, feather_px + 1):
            col = left - i
            if col < 0:
                break
            alpha_row[col] = max(alpha_row[col], 1.0 - i / (feather_px + 1))
        # Right feather: ramp from 1 to 0 across [right, right+feather_px]
        for i in range(1, feather_px + 1):
            col = right + i - 1
            if col >= w:
                break
            alpha_row[col] = max(alpha_row[col], 1.0 - i / (feather_px + 1))

    # Broadcast to (H, W, 3) alpha cube
    alpha = np.broadcast_to(alpha_row[None, :, None], inpainted.shape)

    blended = (
        original.astype(np.float32) * alpha
        + inpainted.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
