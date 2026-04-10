"""Unit tests for AnyText2 adaptive mask helpers.

Pure-logic tests for the functions in `src.models.anytext2_mask` that compute
target text width, adaptive mask rectangles, and middle-strip restoration.
Does not touch any gradio client, server, or inpainter.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.anytext2_mask import (
    compute_adaptive_mask_rect,
    estimate_target_width,
    restore_middle_strip,
)


class TestEstimateTargetWidth:
    """Character-class-based width estimator.

    Uses these multipliers of canonical height:
        CJK:          1.00
        Latin upper:  0.60
        Latin lower:  0.50
        Digit:        0.55
        Space:        0.30
        Other:        0.55
    """

    def test_empty_string_is_zero(self):
        assert estimate_target_width("", height=80) == 0

    def test_single_cjk_char(self):
        # 1 full-width char ≈ 1.0 × height
        assert estimate_target_width("汉", height=80) == pytest.approx(80.0)

    def test_single_japanese_hiragana(self):
        assert estimate_target_width("あ", height=80) == pytest.approx(80.0)

    def test_single_korean_hangul(self):
        assert estimate_target_width("한", height=80) == pytest.approx(80.0)

    def test_single_latin_upper(self):
        assert estimate_target_width("A", height=80) == pytest.approx(48.0)  # 80 * 0.60

    def test_single_latin_lower(self):
        assert estimate_target_width("a", height=80) == pytest.approx(40.0)  # 80 * 0.50

    def test_single_digit(self):
        assert estimate_target_width("5", height=80) == pytest.approx(44.0)  # 80 * 0.55

    def test_single_space(self):
        assert estimate_target_width(" ", height=80) == pytest.approx(24.0)  # 80 * 0.30

    def test_cjk_phrase_three_chars(self):
        # "我是示" → 3 × 80 = 240
        assert estimate_target_width("我是示", height=80) == pytest.approx(240.0)

    def test_all_upper_latin_word(self):
        # "DANGER" → 6 × 0.60 × 80 = 288
        assert estimate_target_width("DANGER", height=80) == pytest.approx(288.0)

    def test_all_lower_latin_word(self):
        # "peligro" → 7 × 0.50 × 80 = 280
        assert estimate_target_width("peligro", height=80) == pytest.approx(280.0)

    def test_mixed_case_latin_word(self):
        # "Hello" → 0.60 + 0.50*4 = 2.6 × 80 = 208
        assert estimate_target_width("Hello", height=80) == pytest.approx(208.0)

    def test_mixed_cjk_and_latin(self):
        # "Hello世界" → H(0.60) + ello(0.50*4) + 世界(1.0*2) = 0.60 + 2.0 + 2.0 = 4.6 × 80 = 368
        assert estimate_target_width("Hello世界", height=80) == pytest.approx(368.0)

    def test_digits_in_text(self):
        # "Room 101" → R(0.60) + oom(0.50*3) + space(0.30) + 101(0.55*3) = 0.60 + 1.5 + 0.30 + 1.65 = 4.05 × 80 = 324
        assert estimate_target_width("Room 101", height=80) == pytest.approx(324.0)

    def test_height_scales_linearly(self):
        # Same text, double height → double width
        w1 = estimate_target_width("DANGER", height=40)
        w2 = estimate_target_width("DANGER", height=80)
        assert w2 == pytest.approx(2 * w1)

    def test_other_punctuation_fallback(self):
        # Punctuation falls through to "other" = 0.55
        assert estimate_target_width("!", height=80) == pytest.approx(44.0)

    def test_height_zero_returns_zero(self):
        # Degenerate height → 0 (defensive guard)
        assert estimate_target_width("DANGER", height=0) == 0


class TestComputeAdaptiveMaskRect:
    """Compute (top, bottom, left, right) mask rect inside a canonical area.

    Returns None when the adaptive flow should be skipped entirely:
    - Empty target text
    - Degenerate canonical dimensions
    - Aspect mismatch is within tolerance (close enough to skip)
    - Target is wider than source (long→short only, never grow)

    Otherwise returns a horizontally centered rect with height = canonical_h
    and width clamped to canonical_w (never wider than source).
    """

    def _rect(self, canonical_w, canonical_h, target_text,
              tolerance=0.15):
        return compute_adaptive_mask_rect(
            canonical_w=canonical_w,
            canonical_h=canonical_h,
            target_text=target_text,
            tolerance=tolerance,
        )

    # --- Skip cases (return None) ---

    def test_empty_target_text_returns_none(self):
        assert self._rect(700, 80, "") is None

    def test_zero_canonical_returns_none(self):
        assert self._rect(0, 80, "DANGER") is None
        assert self._rect(700, 0, "DANGER") is None

    def test_within_tolerance_returns_none(self):
        # source = 7:1 (560/80), target "我是示" = 3:1 (240/80) → ignore this test path
        # Use matching aspects: source 3:1, target exactly 3 CJK chars
        # source_aspect = 240/80 = 3.0, target_aspect = 240/80 = 3.0, mismatch = 0
        assert self._rect(240, 80, "我是示") is None

    def test_just_below_tolerance_returns_none(self):
        # source_aspect = 3.0 (240/80), target "Hello" ≈ 2.6 (208/80)
        # mismatch = |3.0 - 2.6| / 3.0 = 0.133 < 0.15 → skip
        assert self._rect(240, 80, "Hello") is None

    def test_target_wider_than_source_returns_none(self):
        # source = "A" sized canonical (short), target "CONGRATULATIONS" (long)
        # target_aspect >> source_aspect → don't grow
        assert self._rect(100, 80, "CONGRATULATIONS") is None

    # --- Shrink cases (return rect) ---

    def test_long_cjk_to_short_cjk_shrinks(self):
        # source = 560:80 = 7:1 (7 CJK width), target "我是示" = 3:1 (240 px wide)
        # mismatch = |7 - 3| / 7 = 0.57 ≫ 0.15 → shrink
        rect = self._rect(560, 80, "我是示")
        assert rect is not None
        top, bot, left, right = rect
        assert top == 0
        assert bot == 80
        # Width = 240, centered in 560 → left = (560-240)/2 = 160
        assert right - left == 240
        assert left == 160
        assert right == 400

    def test_just_above_tolerance_shrinks(self):
        # source_aspect = 3.0 (240/80), target "HI" = 0.60 + 0.60 = 1.2 × 80 = 96 wide
        # target_aspect = 96/80 = 1.2, mismatch = |3.0 - 1.2| / 3.0 = 0.60 → shrink
        rect = self._rect(240, 80, "HI")
        assert rect is not None
        top, bot, left, right = rect
        assert top == 0
        assert bot == 80
        assert right - left == 96
        assert left == 72  # (240 - 96) // 2

    def test_very_narrow_target_not_clamped(self):
        # source = 800, target "a" (40 px) → mask_w = 40, no floor clamping
        rect = self._rect(800, 80, "a")
        assert rect is not None
        _, _, left, right = rect
        assert right - left == 40
        assert left == (800 - 40) // 2  # 380

    def test_centered_rect_odd_width(self):
        # canonical_w = 701 odd, target produces even width → centered (integer floor)
        rect = self._rect(701, 80, "我是示")  # width = 240
        assert rect is not None
        _, _, left, right = rect
        assert right - left == 240
        assert left == (701 - 240) // 2  # 230

    def test_target_width_rounded_to_int(self):
        # "Hello世" → 0.60 + 0.50*4 + 1.0 = 3.6 × 80 = 288
        rect = self._rect(560, 80, "Hello世")
        assert rect is not None
        _, _, left, right = rect
        width = right - left
        assert isinstance(width, int)
        assert width == 288

    def test_tolerance_parameter_controls_skip_threshold(self):
        # Default tolerance 0.15: mismatch 0.133 skips
        assert self._rect(240, 80, "Hello", tolerance=0.15) is None
        # Tighter tolerance 0.10: same case now triggers shrink
        rect = self._rect(240, 80, "Hello", tolerance=0.10)
        assert rect is not None

    def test_returned_rect_is_tuple_of_ints(self):
        rect = self._rect(560, 80, "我是示")
        assert rect is not None
        assert len(rect) == 4
        for v in rect:
            assert isinstance(v, int)


class TestRestoreMiddleStrip:
    """Composite a middle strip of `original` onto `inpainted`.

    Pixels inside the strip: taken from `original`.
    Pixels outside the strip: taken from `inpainted`.
    With `feather_px > 0`, a linear alpha gradient is applied at the
    left/right strip boundaries so there's no visible seam.
    """

    def _make_arrays(self, h=80, w=560):
        # original: all 255 (e.g. white text), inpainted: all 0 (e.g. black bg)
        original = np.full((h, w, 3), 255, dtype=np.uint8)
        inpainted = np.zeros((h, w, 3), dtype=np.uint8)
        return original, inpainted

    def test_output_has_same_shape(self):
        original, inpainted = self._make_arrays()
        result = restore_middle_strip(
            inpainted, original, (0, 80, 160, 400), feather_px=0,
        )
        assert result.shape == inpainted.shape
        assert result.dtype == inpainted.dtype

    def test_hard_restore_no_feather(self):
        original, inpainted = self._make_arrays()
        # strip: left=160, right=400
        result = restore_middle_strip(
            inpainted, original, (0, 80, 160, 400), feather_px=0,
        )
        # Inside strip: 255 (from original)
        assert np.all(result[:, 160:400] == 255)
        # Outside strip (left side): 0 (from inpainted)
        assert np.all(result[:, :160] == 0)
        # Outside strip (right side): 0 (from inpainted)
        assert np.all(result[:, 400:] == 0)

    def test_empty_strip_returns_inpainted(self):
        original, inpainted = self._make_arrays()
        # left == right → zero-width strip
        result = restore_middle_strip(
            inpainted, original, (0, 80, 280, 280), feather_px=0,
        )
        assert np.all(result == 0)  # all inpainted

    def test_full_width_strip_returns_original(self):
        original, inpainted = self._make_arrays()
        result = restore_middle_strip(
            inpainted, original, (0, 80, 0, 560), feather_px=0,
        )
        assert np.all(result == 255)  # all original

    def test_feather_center_is_original(self):
        original, inpainted = self._make_arrays()
        # strip 160..400 with 3px feather
        result = restore_middle_strip(
            inpainted, original, (0, 80, 160, 400), feather_px=3,
        )
        # Well inside the strip (away from feather zone): pure original
        assert np.all(result[:, 200:360] == 255)

    def test_feather_well_outside_is_inpainted(self):
        original, inpainted = self._make_arrays()
        result = restore_middle_strip(
            inpainted, original, (0, 80, 160, 400), feather_px=3,
        )
        # Well outside the strip (beyond feather zone): pure inpainted
        assert np.all(result[:, :150] == 0)
        assert np.all(result[:, 410:] == 0)

    def test_feather_boundary_has_intermediate_values(self):
        original, inpainted = self._make_arrays()
        result = restore_middle_strip(
            inpainted, original, (0, 80, 160, 400), feather_px=3,
        )
        # Somewhere in the feather zone: values are between 0 and 255
        # (linear blend)
        left_feather_col = result[0, 158]  # in the left feather zone
        assert 0 < left_feather_col[0] < 255

    def test_shape_mismatch_raises(self):
        original = np.full((80, 560, 3), 255, dtype=np.uint8)
        inpainted = np.zeros((80, 400, 3), dtype=np.uint8)  # wrong shape
        with pytest.raises(ValueError):
            restore_middle_strip(
                inpainted, original, (0, 80, 160, 400), feather_px=0,
            )

    def test_does_not_mutate_inputs(self):
        original, inpainted = self._make_arrays()
        original_copy = original.copy()
        inpainted_copy = inpainted.copy()
        restore_middle_strip(
            inpainted, original, (0, 80, 160, 400), feather_px=3,
        )
        assert np.array_equal(original, original_copy)
        assert np.array_equal(inpainted, inpainted_copy)

    def test_feather_clamps_at_left_canvas_edge(self):
        # strip touches left edge of canvas — left feather can't extend left
        original, inpainted = self._make_arrays()
        result = restore_middle_strip(
            inpainted, original, (0, 80, 0, 200), feather_px=5,
        )
        # Leftmost column is entirely inside the strip → should be original
        assert np.all(result[:, 0] == 255)
        # Right feather still applies (strip is not at right edge)
        assert np.all(result[:, 199] == 255)           # last col of strip
        assert 0 < result[0, 200, 0] < 255             # right feather col 1
        assert np.all(result[:, 206:] == 0)            # well past the feather

    def test_feather_clamps_at_right_canvas_edge(self):
        # strip touches right edge of canvas — right feather must not bleed
        # into the strip interior (regression guard for the R2 off-by-one
        # concern raised during review). canvas w=560, strip right==560.
        original, inpainted = self._make_arrays()
        result = restore_middle_strip(
            inpainted, original, (0, 80, 360, 560), feather_px=5,
        )
        # Last column of the strip must stay pure original, not feathered
        assert np.all(result[:, 559] == 255)
        # Left feather applies normally
        assert 0 < result[0, 359, 0] < 255             # left feather col 1
        assert np.all(result[:, :355] == 0)            # well before feather

    def test_feather_at_both_edges(self):
        # Strip covers entire canvas — both feather loops must no-op
        original, inpainted = self._make_arrays()
        result = restore_middle_strip(
            inpainted, original, (0, 80, 0, 560), feather_px=5,
        )
        # Entire canvas is the strip → entire canvas is original
        assert np.all(result == 255)
