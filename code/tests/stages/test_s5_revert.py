"""Tests for Stage 5: Revert (De-Frontalization + Compositing)."""

import cv2
import numpy as np
import pytest

from src.data_types import BBox, PropagatedROI, Quad, TextDetection, TextTrack
from src.stages.s5_revert import RevertStage


@pytest.fixture
def revert_stage(default_config):
    return RevertStage(default_config)


class TestWarpRoiToFrame:
    def test_identity_homography(self, revert_stage):
        """Warp with identity H returns bbox-sized output, not full-frame.

        Note: warp_roi_to_frame expands the target bbox by max(5%, 2 px)
        on each side before clamping. For a 200x100 quad at origin on a
        640x480 frame, expansion_w=10 expansion_h=5 -> pre-clamp bbox
        (-10, -5, 220, 110), clamped to (0, 0, 210, 105).
        """
        roi = np.full((100, 200, 3), 128, dtype=np.uint8)
        alpha = np.ones((100, 200), dtype=np.float32)
        quad = Quad(points=np.array([
            [0, 0], [200, 0], [200, 100], [0, 100]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H_from_frontal = np.eye(3)
        result = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal, (480, 640)
        )
        assert result is not None
        warped_roi, warped_alpha, target_bbox = result
        assert warped_roi.shape == (105, 210, 3)
        assert warped_alpha.shape == (105, 210)
        assert target_bbox == BBox(x=0, y=0, width=210, height=105)

    def test_none_homography_returns_none(self, revert_stage):
        """Passing None as H_from_frontal returns None."""
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        quad = Quad(points=np.zeros((4, 2), dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        result = revert_stage.warp_roi_to_frame(prop, None, (50, 50))
        assert result is None

    def test_bounded_warp_size(self, revert_stage):
        """Warped ROI should be bbox-sized, not full-frame-sized.

        Expansion: expansion_w=max(5, 2)=5, expansion_h=max(2, 2)=2.
        Pre-clamp bbox (295, 198, 110, 54) — fully inside the 1920x1080
        frame, so no clamping.
        """
        roi = np.full((50, 100, 3), 200, dtype=np.uint8)
        alpha = np.ones((50, 100), dtype=np.float32)
        quad = Quad(points=np.array([
            [300, 200], [400, 200], [400, 250], [300, 250]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H_from_frontal = np.eye(3)
        result = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal, (1080, 1920)
        )
        assert result is not None
        warped_roi, warped_alpha, target_bbox = result
        # Key assertion: output is quad-bbox-sized (plus expansion),
        # not 1920x1080
        assert warped_roi.shape == (54, 110, 3)
        assert warped_alpha.shape == (54, 110)
        assert target_bbox == BBox(x=295, y=198, width=110, height=54)

    def test_bbox_clamped_to_frame_bounds(self, revert_stage):
        """Quad partially outside frame should be clamped to frame bounds.

        Expansion: both expansions = max(5, 2) = 5.
        Pre-clamp bbox (145, 145, 110, 110), clamped against (200, 200)
        gives (145, 145, 55, 55).
        """
        roi = np.full((100, 100, 3), 128, dtype=np.uint8)
        alpha = np.ones((100, 100), dtype=np.float32)
        quad = Quad(points=np.array([
            [150, 150], [250, 150], [250, 250], [150, 250]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H_from_frontal = np.eye(3)
        result = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal, (200, 200)
        )
        assert result is not None
        warped_roi, warped_alpha, target_bbox = result
        assert target_bbox == BBox(x=145, y=145, width=55, height=55)
        assert warped_roi.shape == (55, 55, 3)
        assert warped_alpha.shape == (55, 55)

    def test_delta_h_none_matches_legacy_path(self, revert_stage):
        """Passing delta_H=None must produce bit-identical output to
        omitting the argument entirely — pins backward compatibility."""
        rng = np.random.default_rng(1)
        roi = rng.integers(0, 256, (60, 120, 3), dtype=np.uint8)
        alpha = np.ones((60, 120), dtype=np.float32)
        quad = Quad(points=np.array([
            [50, 40], [170, 40], [170, 100], [50, 100]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H = np.eye(3)

        legacy = revert_stage.warp_roi_to_frame(prop, H, (240, 320))
        with_none = revert_stage.warp_roi_to_frame(
            prop, H, (240, 320), delta_H=None,
        )
        assert legacy is not None and with_none is not None
        np.testing.assert_array_equal(legacy[0], with_none[0])
        np.testing.assert_array_equal(legacy[1], with_none[1])
        assert legacy[2] == with_none[2]

    def test_delta_h_identity_matches_none(self, revert_stage):
        """Passing delta_H=I must produce identical output to delta_H=None."""
        rng = np.random.default_rng(2)
        roi = rng.integers(0, 256, (60, 120, 3), dtype=np.uint8)
        alpha = np.ones((60, 120), dtype=np.float32)
        quad = Quad(points=np.array([
            [50, 40], [170, 40], [170, 100], [50, 100]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H = np.eye(3)

        without = revert_stage.warp_roi_to_frame(prop, H, (240, 320))
        with_identity = revert_stage.warp_roi_to_frame(
            prop, H, (240, 320), delta_H=np.eye(3, dtype=np.float64),
        )
        assert without is not None and with_identity is not None
        np.testing.assert_array_equal(without[0], with_identity[0])
        np.testing.assert_array_equal(without[1], with_identity[1])
        assert without[2] == with_identity[2]

    def test_delta_h_direction_translation_sanity(self, revert_stage):
        """**Direction sanity test** — pins the ΔH composition direction.

        Catches the classic ``cv2.warpPerspective`` gotcha: does
        ``warp_roi_to_frame`` compose ``delta_H`` directly (correct,
        matches the refiner's training contract) or ``inv(delta_H)``
        (wrong — would drift the edited ROI in the *opposite* direction
        of CoTracker's error)?

        Setup
        -----
        - 100x100 black edited ROI with a 20x20 white square at
          ``[y in 20, 40), x in 10, 30)``.
        - Axis-aligned quad at origin. After the 5% bbox expansion and
          clamping to frame bounds, ``target_bbox = (0, 0, 105, 105)``,
          so ``T = I``.
        - ``H_from_frontal = I``.
        - Therefore ``H_adjusted = T @ H_from_frontal @ delta_H = delta_H``
          exactly — nothing else contributes to the warp.
        - ``delta_H`` = pure translation by (+5, +2). This is the
          forward homography the trained refiner predicts when the
          target text has drifted +5 right, +2 down in canonical space.

        Contract
        --------
        ``cv2.warpPerspective`` treats its matrix as forward src->dst
        and internally samples source at ``inv(M) @ output``. With
        ``M = delta_H``:

            output (x=25, y=32) <- source (20, 30)  [inside square -> white]
            output (x=10, y=20) <- source ( 5, 18)  [outside square -> black]

        If the code had used ``M = inv(delta_H)`` instead, both sample
        coordinates would swap sign, and (25, 32) would become *black*
        while (10, 20) would become *white*. The asserts below catch
        exactly that swap.
        """
        edited_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        # White 20x20 square: y in [20, 40), x in [10, 30) -> center (20, 30)
        edited_roi[20:40, 10:30, :] = 255
        alpha = np.ones((100, 100), dtype=np.float32)
        quad = Quad(points=np.array([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=edited_roi, alpha_mask=alpha, target_quad=quad,
        )

        # --- Baseline: delta_H=None, marker at its original (20, 30). ---
        baseline = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal=np.eye(3), frame_shape=(200, 200),
        )
        assert baseline is not None
        baseline_roi = baseline[0]
        assert baseline_roi[30, 20, 0] > 200, (
            "baseline marker should be at (x=20, y=30)"
        )

        # --- With delta_H = pure translation (+5, +2) ---
        delta_H = np.array([
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        shifted = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal=np.eye(3), frame_shape=(200, 200),
            delta_H=delta_H,
        )
        assert shifted is not None
        shifted_roi = shifted[0]

        # After correct shift: marker landed at (x=25, y=32).
        assert shifted_roi[32, 25, 0] > 200, (
            f"expected marker at (x=25, y=32) after +5/+2 translation; "
            f"got value {shifted_roi[32, 25, 0]} — suggests wrong direction"
        )
        # Wrong-direction guard: if we'd used inv(delta_H), output
        # (x=10, y=20) would sample source (x=15, y=22), inside the
        # square, giving a bright pixel. With the correct direction it
        # samples source (x=5, y=18), outside the square, giving black.
        assert shifted_roi[20, 10, 0] < 50, (
            f"expected black at (x=10, y=20) with correct direction; "
            f"got value {shifted_roi[20, 10, 0]} — suggests wrong direction"
        )

    def test_delta_h_non_identity_changes_output(self, revert_stage):
        """A non-identity delta_H must actually change the warp output —
        pins that the new parameter is wired into the cv2.warpPerspective
        call rather than being silently ignored."""
        rng = np.random.default_rng(3)
        roi = rng.integers(0, 256, (60, 120, 3), dtype=np.uint8)
        alpha = np.ones((60, 120), dtype=np.float32)
        quad = Quad(points=np.array([
            [50, 40], [170, 40], [170, 100], [50, 100]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H = np.eye(3)

        baseline = revert_stage.warp_roi_to_frame(prop, H, (240, 320))
        # Small pure-translation delta_H: shift content by 5 px in x.
        delta_H = np.array(
            [[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        shifted = revert_stage.warp_roi_to_frame(
            prop, H, (240, 320), delta_H=delta_H,
        )
        assert baseline is not None and shifted is not None
        assert not np.array_equal(baseline[0], shifted[0]), (
            "delta_H did not affect warp_roi output"
        )

    def test_zero_area_bbox_returns_none(self, revert_stage):
        """If clamped bbox has zero area, return None."""
        roi = np.full((50, 50, 3), 128, dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        # Quad entirely outside frame bounds
        quad = Quad(points=np.array([
            [300, 300], [350, 300], [350, 350], [300, 350]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H_from_frontal = np.eye(3)
        result = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal, (100, 100)
        )
        assert result is None


class TestCompositeRoiIntoFrame:
    def test_full_alpha_replaces_region(self, revert_stage):
        """Full alpha compositing replaces only the target bbox region."""
        frame = np.full((200, 200, 3), 0, dtype=np.uint8)
        roi = np.full((50, 50, 3), 200, dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        target_bbox = BBox(x=10, y=20, width=50, height=50)
        result = revert_stage.composite_roi_into_frame(
            frame, roi, alpha, target_bbox
        )
        # Target region should be replaced
        np.testing.assert_array_equal(result[20:70, 10:60], roi)
        # Outside region should be untouched
        assert result[0, 0, 0] == 0

    def test_zero_alpha_preserves_frame(self, revert_stage):
        frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        roi = np.full((50, 50, 3), 200, dtype=np.uint8)
        alpha = np.zeros((50, 50), dtype=np.float32)
        target_bbox = BBox(x=10, y=20, width=50, height=50)
        result = revert_stage.composite_roi_into_frame(
            frame, roi, alpha, target_bbox
        )
        np.testing.assert_array_equal(result, frame)

    def test_half_alpha_blends(self, revert_stage):
        frame = np.full((200, 200, 3), 0, dtype=np.uint8)
        roi = np.full((50, 50, 3), 200, dtype=np.uint8)
        alpha = np.full((50, 50), 0.5, dtype=np.float32)
        target_bbox = BBox(x=10, y=20, width=50, height=50)
        result = revert_stage.composite_roi_into_frame(
            frame, roi, alpha, target_bbox
        )
        region = result[20:70, 10:60]
        expected = 100  # 0 * 0.5 + 200 * 0.5
        assert np.abs(region.mean() - expected) < 2


class TestRevertRun:
    def test_no_rois_returns_original_frames(self, revert_stage):
        frames = {
            0: np.zeros((100, 100, 3), dtype=np.uint8),
            1: np.ones((100, 100, 3), dtype=np.uint8) * 255,
        }
        output = revert_stage.run(frames, {}, [])
        assert len(output) == 2
        np.testing.assert_array_equal(output[0], frames[0])
        np.testing.assert_array_equal(output[1], frames[1])

    def test_run_reads_homography_from_detection(self, revert_stage):
        """run() should read H_from_frontal from TextDetection, not a separate dict.

        Uses textured frame and ROI so ``cv2.seamlessClone`` (Poisson
        blending) inside ``composite_roi_into_frame_seamless`` has gradient
        signal to work with. A uniform frame + uniform ROI produces an
        all-zero output because Poisson blending operates on gradients.
        """
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 128, (200, 200, 3), dtype=np.uint8)
        original_frame = frame.copy()
        frames = {0: frame}

        roi = rng.integers(150, 256, (50, 100, 3), dtype=np.uint8)
        alpha = np.ones((50, 100), dtype=np.float32)
        quad = Quad(points=np.array([
            [10, 20], [110, 20], [110, 70], [10, 70]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )

        det = TextDetection(
            frame_idx=0,
            quad=quad,
            bbox=quad.to_bbox(),
            text="HELLO",
            ocr_confidence=0.95,
            H_from_frontal=np.eye(3),
            homography_valid=True,
        )
        track = TextTrack(
            track_id=0,
            source_text="HELLO",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={0: det},
            reference_frame_idx=0,
        )

        output = revert_stage.run(frames, {0: [prop]}, [track])
        assert len(output) == 1
        # Quad region should have been modified — compared to the same
        # region in the untouched original frame.
        out_region = output[0][20:70, 10:110]
        orig_region = original_frame[20:70, 10:110]
        diff = np.abs(out_region.astype(np.int32) - orig_region.astype(np.int32))
        # Seamless clone produces noticeable changes inside the bbox.
        assert diff.mean() > 10, (
            f"ROI region looks unchanged after compositing (mean diff {diff.mean()})"
        )
        # Region outside the quad should be untouched by the composite.
        outside = output[0][0:15, 0:15]
        orig_outside = original_frame[0:15, 0:15]
        np.testing.assert_array_equal(outside, orig_outside)

    def test_run_skips_invalid_homography(self, revert_stage):
        """run() should skip ROIs where homography_valid is False."""
        frame = np.full((200, 200, 3), 50, dtype=np.uint8)
        frames = {0: frame}

        roi = np.full((50, 100, 3), 200, dtype=np.uint8)
        alpha = np.ones((50, 100), dtype=np.float32)
        quad = Quad(points=np.array([
            [10, 20], [110, 20], [110, 70], [10, 70]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )

        det = TextDetection(
            frame_idx=0,
            quad=quad,
            bbox=quad.to_bbox(),
            text="HELLO",
            ocr_confidence=0.95,
            H_from_frontal=np.eye(3),
            homography_valid=False,  # invalid
        )
        track = TextTrack(
            track_id=0,
            source_text="HELLO",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={0: det},
            reference_frame_idx=0,

        )

        output = revert_stage.run(frames, {0: [prop]}, [track])
        # Frame should be unmodified since homography is invalid
        np.testing.assert_array_equal(output[0], frame)

    def test_run_skips_missing_track(self, revert_stage):
        """run() should skip ROIs whose track_id has no matching track."""
        frame = np.full((200, 200, 3), 50, dtype=np.uint8)
        frames = {0: frame}

        roi = np.full((50, 100, 3), 200, dtype=np.uint8)
        alpha = np.ones((50, 100), dtype=np.float32)
        quad = Quad(points=np.array([
            [10, 20], [110, 20], [110, 70], [10, 70]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=99,  # no track with this ID
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )

        output = revert_stage.run(frames, {0: [prop]}, [])
        np.testing.assert_array_equal(output[0], frame)


class TestRefinerIntegration:
    """Tests that exercise RevertStage.run() with the refiner wired in.

    Uses a fake refiner injected via ``stage._refiner`` so tests don't
    need a real checkpoint. The tests focus on the wiring and control
    flow — the numerical scale/compose math is pinned by
    ``test_s5_refiner.py`` already.
    """

    def _make_track(
        self,
        rng: np.random.Generator,
        *,
        canonical_size: tuple[int, int] = (120, 60),
        frame_idx: int = 0,
    ) -> tuple[TextTrack, TextDetection, PropagatedROI]:
        w_can, h_can = canonical_size
        quad = Quad(points=np.array([
            [30, 40], [150, 40], [150, 100], [30, 100]
        ], dtype=np.float32))
        det = TextDetection(
            frame_idx=frame_idx,
            quad=quad,
            bbox=quad.to_bbox(),
            text="HELLO",
            ocr_confidence=0.9,
            H_to_frontal=np.eye(3, dtype=np.float64),
            H_from_frontal=np.eye(3, dtype=np.float64),
            homography_valid=True,
        )
        track = TextTrack(
            track_id=0,
            source_text="HELLO",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={frame_idx: det},
            reference_frame_idx=frame_idx,
            canonical_size=canonical_size,
        )
        prop = PropagatedROI(
            frame_idx=frame_idx,
            track_id=0,
            roi_image=rng.integers(150, 256, (h_can, w_can, 3), dtype=np.uint8),
            alpha_mask=np.ones((h_can, w_can), dtype=np.float32),
            target_quad=quad,
            target_roi_canonical=rng.integers(
                0, 256, (h_can, w_can, 3), dtype=np.uint8,
            ),
        )
        return track, det, prop

    def test_refiner_not_invoked_when_disabled(self, default_config):
        """With use_refiner=False, predict_delta_H is never called even
        when target_roi_canonical is populated."""
        assert default_config.revert.use_refiner is False
        stage = RevertStage(default_config)
        assert stage._refiner is None

        rng = np.random.default_rng(0)
        track, _, prop = self._make_track(rng)
        frame = rng.integers(0, 128, (200, 240, 3), dtype=np.uint8)
        # The pipeline should execute cleanly — if run() tried to call
        # a None refiner it would raise an AttributeError.
        out = stage.run({0: frame}, {0: [prop]}, [track])
        assert len(out) == 1

    def test_refiner_invoked_when_enabled(self, default_config):
        """With use_refiner=True + target_roi_canonical present, the
        refiner is invoked exactly once per (track, detection) pair."""
        default_config.revert.use_refiner = True
        stage = RevertStage(default_config)

        # Inject a fake refiner that records calls and returns identity.
        calls: list[tuple[np.ndarray, np.ndarray]] = []

        class _FakeRefiner:
            def predict_delta_H(self, ref, tgt):
                calls.append((ref, tgt))
                return np.eye(3, dtype=np.float64)

        stage._refiner = _FakeRefiner()  # type: ignore[assignment]

        rng = np.random.default_rng(1)
        track, _, prop = self._make_track(rng)
        frame = rng.integers(0, 128, (200, 240, 3), dtype=np.uint8)
        stage.run({0: frame}, {0: [prop]}, [track])

        assert len(calls) == 1
        ref_called, tgt_called = calls[0]
        # Reference canonical was pre-built from the ref frame via H_to_frontal
        # (identity here), so it should be the frame warped to canonical_size.
        assert ref_called.shape == (60, 120, 3)
        # Target canonical came straight from PropagatedROI
        np.testing.assert_array_equal(tgt_called, prop.target_roi_canonical)

    def test_refiner_skipped_when_target_canonical_missing(self, default_config):
        """target_roi_canonical=None should leave the refiner uninvoked
        — this is the non-refiner S4 path."""
        default_config.revert.use_refiner = True
        stage = RevertStage(default_config)

        calls: list = []

        class _FakeRefiner:
            def predict_delta_H(self, ref, tgt):
                calls.append((ref, tgt))
                return np.eye(3, dtype=np.float64)

        stage._refiner = _FakeRefiner()  # type: ignore[assignment]

        rng = np.random.default_rng(2)
        track, _, prop = self._make_track(rng)
        prop.target_roi_canonical = None  # S4 flag was off

        frame = rng.integers(0, 128, (200, 240, 3), dtype=np.uint8)
        stage.run({0: frame}, {0: [prop]}, [track])
        assert calls == []  # refiner not called

    def test_refiner_rejection_falls_back_to_identity(self, default_config):
        """When the refiner returns None, run() must fall back to the
        non-refiner warp path (delta_H=None) rather than crashing."""
        default_config.revert.use_refiner = True
        stage = RevertStage(default_config)

        class _NullRefiner:
            def predict_delta_H(self, ref, tgt):
                return None  # reject everything

        stage._refiner = _NullRefiner()  # type: ignore[assignment]

        rng = np.random.default_rng(3)
        track, _, prop = self._make_track(rng)
        frame = rng.integers(0, 128, (200, 240, 3), dtype=np.uint8)
        # Should not raise
        out = stage.run({0: frame}, {0: [prop]}, [track])
        assert len(out) == 1

    def test_refiner_exception_falls_back_to_identity(self, default_config):
        """If predict_delta_H raises, run() must catch, log, and continue."""
        default_config.revert.use_refiner = True
        stage = RevertStage(default_config)

        class _BrokenRefiner:
            def predict_delta_H(self, ref, tgt):
                raise RuntimeError("simulated inference crash")

        stage._refiner = _BrokenRefiner()  # type: ignore[assignment]

        rng = np.random.default_rng(4)
        track, _, prop = self._make_track(rng)
        frame = rng.integers(0, 128, (200, 240, 3), dtype=np.uint8)
        # Should not raise — an exception from the refiner must NEVER
        # crash the pipeline.
        out = stage.run({0: frame}, {0: [prop]}, [track])
        assert len(out) == 1

    def test_refiner_delta_h_reaches_composite(self, default_config):
        """End-to-end plumbing: a non-identity ΔH from the refiner must
        actually change the composited output frame.

        If somewhere between ``predict_delta_H`` and the ``cv2.warpPerspective``
        call ΔH gets dropped or ignored, the identity-refiner and
        translation-refiner outputs would be identical — this test
        catches that regression.
        """
        default_config.revert.use_refiner = True
        stage = RevertStage(default_config)

        class _IdentityRefiner:
            def predict_delta_H(self, ref, tgt):
                return np.eye(3, dtype=np.float64)

        class _ShiftRefiner:
            def predict_delta_H(self, ref, tgt):
                return np.array([
                    [1.0, 0.0, 4.0],
                    [0.0, 1.0, 3.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)

        rng = np.random.default_rng(7)
        track, _, prop = self._make_track(rng)
        frame = rng.integers(0, 128, (200, 240, 3), dtype=np.uint8)

        stage._refiner = _IdentityRefiner()  # type: ignore[assignment]
        identity_out = stage.run(
            {0: frame.copy()}, {0: [prop]}, [track],
        )[0].copy()

        stage._refiner = _ShiftRefiner()  # type: ignore[assignment]
        shift_out = stage.run(
            {0: frame.copy()}, {0: [prop]}, [track],
        )[0]

        # Outputs must differ — the translation ΔH caused the edited
        # ROI to be composited at a slightly different position.
        assert not np.array_equal(identity_out, shift_out), (
            "identity and non-identity ΔH produced byte-identical output; "
            "the predicted delta_H was not reaching warp_roi_to_frame"
        )
        # And the difference should be confined to (roughly) the quad
        # region. The far corners of the frame should be untouched by
        # both runs.
        np.testing.assert_array_equal(identity_out[:5, :5], shift_out[:5, :5])
        np.testing.assert_array_equal(
            identity_out[-5:, -5:], shift_out[-5:, -5:],
        )

    def test_ref_roi_precomputed_once_per_track(self, default_config):
        """Reference canonical should be warped exactly once per track
        even when the track has many detections."""
        default_config.revert.use_refiner = True
        stage = RevertStage(default_config)

        ref_canonicals: list[np.ndarray] = []

        class _TrackingRefiner:
            def predict_delta_H(self, ref, tgt):
                ref_canonicals.append(ref)
                return np.eye(3, dtype=np.float64)

        stage._refiner = _TrackingRefiner()  # type: ignore[assignment]

        rng = np.random.default_rng(5)
        track, det0, prop0 = self._make_track(rng, frame_idx=0)

        # Add two more detections on the same track at frames 1 and 2,
        # each with their own PropagatedROI.
        h_can, w_can = 60, 120
        for fi in (1, 2):
            det = TextDetection(
                frame_idx=fi,
                quad=det0.quad,
                bbox=det0.bbox,
                text="HELLO",
                ocr_confidence=0.9,
                H_to_frontal=np.eye(3, dtype=np.float64),
                H_from_frontal=np.eye(3, dtype=np.float64),
                homography_valid=True,
            )
            track.detections[fi] = det

        frames_dict = {
            0: rng.integers(0, 128, (200, 240, 3), dtype=np.uint8),
            1: rng.integers(0, 128, (200, 240, 3), dtype=np.uint8),
            2: rng.integers(0, 128, (200, 240, 3), dtype=np.uint8),
        }
        props = {
            0: [prop0],
            1: [PropagatedROI(
                frame_idx=1, track_id=0,
                roi_image=rng.integers(150, 256, (h_can, w_can, 3), dtype=np.uint8),
                alpha_mask=np.ones((h_can, w_can), dtype=np.float32),
                target_quad=det0.quad,
                target_roi_canonical=rng.integers(
                    0, 256, (h_can, w_can, 3), dtype=np.uint8,
                ),
            )],
            2: [PropagatedROI(
                frame_idx=2, track_id=0,
                roi_image=rng.integers(150, 256, (h_can, w_can, 3), dtype=np.uint8),
                alpha_mask=np.ones((h_can, w_can), dtype=np.float32),
                target_quad=det0.quad,
                target_roi_canonical=rng.integers(
                    0, 256, (h_can, w_can, 3), dtype=np.uint8,
                ),
            )],
        }

        stage.run(frames_dict, props, [track])

        # 3 calls, one per detection — but they should all receive the
        # same ref_canonical object (verifying the precompute happens once).
        assert len(ref_canonicals) == 3
        assert ref_canonicals[0] is ref_canonicals[1] is ref_canonicals[2]


class TestTemporalSmoothing:
    """Tests for the temporal corner smoothing feature."""

    @staticmethod
    def _make_multi_frame_track(
        rng: np.random.Generator,
        n_frames: int = 10,
        canonical_size: tuple[int, int] = (120, 60),
    ) -> tuple[TextTrack, dict[int, list[PropagatedROI]], dict[int, np.ndarray]]:
        """Build a track spanning ``n_frames`` with identity homographies
        and textured frames/ROIs for compositing."""
        w_can, h_can = canonical_size
        quad = Quad(points=np.array([
            [30, 40], [150, 40], [150, 100], [30, 100]
        ], dtype=np.float32))
        track = TextTrack(
            track_id=0,
            source_text="HI",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={},
            reference_frame_idx=0,
            canonical_size=canonical_size,
        )
        frames: dict[int, np.ndarray] = {}
        props: dict[int, list[PropagatedROI]] = {}
        for fi in range(n_frames):
            det = TextDetection(
                frame_idx=fi,
                quad=quad,
                bbox=quad.to_bbox(),
                text="HI",
                ocr_confidence=0.9,
                H_to_frontal=np.eye(3, dtype=np.float64),
                H_from_frontal=np.eye(3, dtype=np.float64),
                homography_valid=True,
            )
            track.detections[fi] = det
            frames[fi] = rng.integers(0, 128, (200, 240, 3), dtype=np.uint8)
            props[fi] = [PropagatedROI(
                frame_idx=fi,
                track_id=0,
                roi_image=rng.integers(150, 256, (h_can, w_can, 3), dtype=np.uint8),
                alpha_mask=np.ones((h_can, w_can), dtype=np.float32),
                target_quad=quad,
            )]
        return track, props, frames

    def test_smoothing_disabled_by_default(self, default_config):
        assert default_config.revert.temporal_smooth_window == 1

    def test_smoothing_window_1_is_noop(self, default_config):
        """Window=1 should produce identical output to baseline."""
        default_config.revert.temporal_smooth_window = 1
        stage = RevertStage(default_config)
        rng = np.random.default_rng(0)
        track, props, frames = self._make_multi_frame_track(rng, n_frames=5)
        out = stage.run(frames, props, [track])
        assert len(out) == 5

    def test_smoothing_runs_without_crash(self, default_config):
        """Window=5 should complete without errors."""
        default_config.revert.temporal_smooth_window = 5
        default_config.revert.temporal_smooth_sigma = 2.0
        stage = RevertStage(default_config)
        rng = np.random.default_rng(1)
        track, props, frames = self._make_multi_frame_track(rng, n_frames=10)
        out = stage.run(frames, props, [track])
        assert len(out) == 10

    def test_smoothing_reduces_corner_jitter(self, default_config):
        """Inject artificial jitter into H_from_frontal and verify that
        smoothing produces smaller frame-to-frame corner variation.

        Uses ``_project_canonical_to_frame`` directly to measure geometric
        jitter on the projected corners — avoids conflating ROI content
        differences with geometric instability.
        """
        rng = np.random.default_rng(2)
        n_frames = 20
        canonical_size = (120, 60)
        w_can, h_can = canonical_size
        can_corners = np.array(
            [[0, 0], [w_can, 0], [w_can, h_can], [0, h_can]],
            dtype=np.float64,
        )

        # Build jittered H_from_frontal per frame
        raw_corners: dict[int, np.ndarray] = {}
        H_list: list[np.ndarray] = []
        for fi in range(n_frames):
            jx = rng.normal(0, 3.0)
            jy = rng.normal(0, 2.0)
            H = np.array([
                [1.0, 0.0, jx],
                [0.0, 1.0, jy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            H_list.append(H)
            proj = RevertStage._project_canonical_to_frame(H, can_corners)
            raw_corners[fi] = proj

        # Smooth
        fi_sorted = list(range(n_frames))
        smoothed = RevertStage._smooth_corner_trajectories(
            raw_corners, fi_sorted, window=7, sigma=2.0,
        )

        # Measure frame-to-frame corner displacement
        def _corner_jitter(corners_dict: dict[int, np.ndarray]) -> float:
            diffs = []
            for i in range(n_frames - 1):
                d = np.linalg.norm(corners_dict[i] - corners_dict[i + 1], axis=1)
                diffs.append(d.mean())
            return float(np.mean(diffs))

        jitter_raw = _corner_jitter(raw_corners)
        jitter_smooth = _corner_jitter(smoothed)

        assert jitter_smooth < jitter_raw, (
            f"smoothing did not reduce corner jitter: raw={jitter_raw:.3f}, "
            f"smooth={jitter_smooth:.3f}"
        )

    def test_smooth_corner_trajectories_unit(self):
        """Direct test of the static smoothing helper."""
        # 5 frames with a spike at frame 2
        trajectories = {
            0: np.array([[10, 20], [110, 20], [110, 70], [10, 70]], dtype=np.float64),
            1: np.array([[10, 20], [110, 20], [110, 70], [10, 70]], dtype=np.float64),
            2: np.array([[20, 30], [120, 30], [120, 80], [20, 80]], dtype=np.float64),  # spike
            3: np.array([[10, 20], [110, 20], [110, 70], [10, 70]], dtype=np.float64),
            4: np.array([[10, 20], [110, 20], [110, 70], [10, 70]], dtype=np.float64),
        }
        smoothed = RevertStage._smooth_corner_trajectories(
            trajectories, [0, 1, 2, 3, 4], window=3, sigma=1.0,
        )
        # The spike at frame 2 should be attenuated toward the
        # surrounding values (10, 20).
        spike_raw = trajectories[2][0, 0]    # 20
        spike_smooth = smoothed[2][0, 0]
        baseline = trajectories[0][0, 0]     # 10
        assert baseline < spike_smooth < spike_raw, (
            f"expected spike attenuation: baseline={baseline}, "
            f"smooth={spike_smooth}, raw={spike_raw}"
        )
        # Non-spike frames should be nearly unchanged.
        assert np.allclose(smoothed[0], trajectories[0], atol=2.0)
        assert np.allclose(smoothed[4], trajectories[4], atol=2.0)


class TestPreInpaintBackendDispatch:
    """RevertStage._get_pre_inpainter dispatch for srnet vs hisam backends.

    The pre-inpaint path is separate from S4's LCM inpainter — S5 has its
    own checkpoint + backend so users can tune boundary scrubbing
    independently of lighting-correction backgrounds.
    """

    def test_srnet_backend_constructs_srnet_inpainter(self, default_config, monkeypatch):
        import src.stages.s4_propagation.srnet_inpainter as srnet_mod

        class _StubSRNetInpainter:
            def __init__(self, checkpoint_path, device):
                self.checkpoint_path = checkpoint_path
                self.device = device

        monkeypatch.setattr(srnet_mod, "SRNetInpainter", _StubSRNetInpainter)

        default_config.revert.pre_inpaint_backend = "srnet"
        default_config.revert.pre_inpaint_checkpoint = "/fake/srnet.model"
        default_config.revert.pre_inpaint_device = "cpu"

        stage = RevertStage(default_config)
        inp = stage._get_pre_inpainter()

        assert isinstance(inp, _StubSRNetInpainter)
        assert inp.checkpoint_path == "/fake/srnet.model"
        assert inp.device == "cpu"

    def test_hisam_backend_constructs_segmentation_inpainter(
        self, default_config, monkeypatch,
    ):
        import src.stages.s4_propagation.segmentation_inpainter as seg_mod

        class _StubHiSAMInpainter:
            def __init__(self, checkpoint_path, device, model_type,
                         mask_dilation_px, inpaint_method, use_patch_mode):
                self.checkpoint_path = checkpoint_path
                self.device = device
                self.model_type = model_type
                self.mask_dilation_px = mask_dilation_px
                self.inpaint_method = inpaint_method
                self.use_patch_mode = use_patch_mode

        monkeypatch.setattr(
            seg_mod, "SegmentationBasedInpainter", _StubHiSAMInpainter,
        )

        default_config.revert.pre_inpaint_backend = "hisam"
        default_config.revert.pre_inpaint_checkpoint = "/fake/hisam.pth"
        default_config.revert.pre_inpaint_device = "cpu"
        default_config.revert.pre_inpaint_hisam_model_type = "vit_b"
        default_config.revert.pre_inpaint_hisam_mask_dilation_px = 5
        default_config.revert.pre_inpaint_hisam_inpaint_method = "telea"
        default_config.revert.pre_inpaint_hisam_use_patch_mode = True

        stage = RevertStage(default_config)
        inp = stage._get_pre_inpainter()

        assert isinstance(inp, _StubHiSAMInpainter)
        assert inp.checkpoint_path == "/fake/hisam.pth"
        assert inp.device == "cpu"
        assert inp.model_type == "vit_b"
        assert inp.mask_dilation_px == 5
        assert inp.inpaint_method == "telea"
        assert inp.use_patch_mode is True

    def test_unknown_backend_raises(self, default_config):
        default_config.revert.pre_inpaint_backend = "bogus"
        stage = RevertStage(default_config)

        with pytest.raises(ValueError, match="pre_inpaint_backend"):
            stage._get_pre_inpainter()

    def test_cached_after_first_call(self, default_config, monkeypatch):
        """Second _get_pre_inpainter call reuses the cached instance."""
        import src.stages.s4_propagation.srnet_inpainter as srnet_mod

        call_count = {"n": 0}

        class _CountingSRNetInpainter:
            def __init__(self, checkpoint_path, device):
                call_count["n"] += 1

        monkeypatch.setattr(srnet_mod, "SRNetInpainter", _CountingSRNetInpainter)

        default_config.revert.pre_inpaint_backend = "srnet"
        default_config.revert.pre_inpaint_checkpoint = "/fake/srnet.model"

        stage = RevertStage(default_config)
        first = stage._get_pre_inpainter()
        second = stage._get_pre_inpainter()

        assert first is second
        assert call_count["n"] == 1


class TestSeamlessCenterStability:
    """Regression guard for seamlessClone center jitter.

    The old fallback ``target_bbox.x + target_bbox.width // 2`` rounds
    ``x_min`` and ``x_max - x_min`` independently via ``int(round(...))``.
    Across frames, these two roundings can disagree by ±1 px even when
    the underlying float quad midpoint barely moves, producing visible
    seamlessClone seed-pixel jitter. ``_seamless_center_from_corners``
    rounds the float midpoint exactly once.
    """

    def test_equal_float_midpoint_yields_equal_center(self):
        """Two quads with equal float midpoints must share a center.

        ``corners_a`` and ``corners_b`` both have bbox midpoint
        ``(15.5, 10.5)``; only the distribution of rounding error
        between x_min/x_max (and y_min/y_max) differs. The int-bbox
        path would send them to different centers because
        ``round(x_min)`` and ``round(width)`` disagree between the two.
        """
        corners_a = np.array([
            [10.4, 5.4], [20.6, 5.4], [20.6, 15.6], [10.4, 15.6],
        ], dtype=np.float64)
        corners_b = np.array([
            [10.6, 5.6], [20.4, 5.6], [20.4, 15.4], [10.6, 15.4],
        ], dtype=np.float64)
        assert RevertStage._seamless_center_from_corners(corners_a) == \
            RevertStage._seamless_center_from_corners(corners_b)

    def test_matches_float_midpoint_rounded_once(self):
        """Output must equal int(round(float_midpoint)) componentwise."""
        corners = np.array([
            [100.3, 50.9], [300.7, 50.9], [300.7, 110.1], [100.3, 110.1],
        ], dtype=np.float64)
        cx = (100.3 + 300.7) / 2.0
        cy = (50.9 + 110.1) / 2.0
        expected = (int(round(cx)), int(round(cy)))
        assert RevertStage._seamless_center_from_corners(corners) == expected

    def test_beats_int_bbox_on_known_jitter_case(self):
        """New center is stable where int-bbox center jumps by 1 px.

        Baseline quad at frame N; same quad drifted ~0.4 px in x and
        ~0.3 px in y between frames N and N+1 (sub-pixel optical flow
        noise). The int-bbox fallback (``bbox.x + bbox.width // 2``)
        produces different centers for the two frames; the round-once
        helper stays within sample-noise sanity.
        """
        # Frame N
        corners_a = np.array([
            [10.3, 5.3], [20.8, 5.3], [20.8, 15.8], [10.3, 15.8],
        ], dtype=np.float64)
        # Frame N+1 (sub-pixel drift)
        corners_b = np.array([
            [10.7, 5.6], [21.0, 5.6], [21.0, 16.1], [10.7, 16.1],
        ], dtype=np.float64)

        def int_bbox_center(corners: np.ndarray) -> tuple[int, int]:
            """Simulate the pre-fix ``target_bbox.x + bbox.width // 2``."""
            xs, ys = corners[:, 0], corners[:, 1]
            x_min = int(round(float(xs.min())))
            y_min = int(round(float(ys.min())))
            width = int(round(float(xs.max()) - float(xs.min())))
            height = int(round(float(ys.max()) - float(ys.min())))
            return (x_min + width // 2, y_min + height // 2)

        old_a = int_bbox_center(corners_a)
        old_b = int_bbox_center(corners_b)
        new_a = RevertStage._seamless_center_from_corners(corners_a)
        new_b = RevertStage._seamless_center_from_corners(corners_b)

        # The old path jumps by ≥1 px on this input (pins the bug).
        old_jump = max(abs(old_a[0] - old_b[0]), abs(old_a[1] - old_b[1]))
        new_jump = max(abs(new_a[0] - new_b[0]), abs(new_a[1] - new_b[1]))
        assert old_jump >= 1, (
            "Test input no longer exercises the int-bbox jitter case; "
            "adjust corners_a / corners_b."
        )
        # New path must not jitter more than the underlying float drift.
        assert new_jump <= old_jump

    def test_run_uses_float_center_when_canonical_size_set(
        self, default_config,
    ):
        """End-to-end: ``run()`` passes a round-once center to seamlessClone.

        Stubs ``composite_roi_into_frame_seamless`` and verifies the
        ``src_center`` argument matches what ``_seamless_center_from_corners``
        would compute from the effective frame corners — i.e., the
        int-bbox fallback path was NOT used.
        """
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 128, (200, 200, 3), dtype=np.uint8)
        frames = {0: frame}

        # Non-integer quad corners — chosen so the float midpoint and
        # the int-bbox midpoint disagree.
        quad = Quad(points=np.array([
            [10.3, 20.4], [110.7, 20.4], [110.7, 70.6], [10.3, 70.6],
        ], dtype=np.float32))
        roi = rng.integers(150, 256, (50, 100, 3), dtype=np.uint8)
        alpha = np.ones((50, 100), dtype=np.float32)
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        det = TextDetection(
            frame_idx=0,
            quad=quad,
            bbox=quad.to_bbox(),
            text="HELLO",
            ocr_confidence=0.95,
            H_from_frontal=np.eye(3),
            homography_valid=True,
        )
        track = TextTrack(
            track_id=0,
            source_text="HELLO",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={0: det},
            reference_frame_idx=0,
            canonical_size=(100, 50),  # required for the fix path
        )

        stage = RevertStage(default_config)
        seen_center: dict[str, tuple[int, int] | None] = {"src_center": None}

        def spy(frame, warped_roi, warped_alpha, target_bbox,
                src_center=None, flags=None):
            seen_center["src_center"] = src_center
            return frame

        stage.composite_roi_into_frame_seamless = spy  # type: ignore[assignment]
        stage.run(frames, {0: [prop]}, [track])

        # Expected: round-once of canonical_corners projected through
        # det.H_from_frontal (= I) and delta_H (= None) = canonical_corners.
        can_corners = np.array(
            [[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64,
        )
        expected = RevertStage._seamless_center_from_corners(can_corners)
        assert seen_center["src_center"] == expected


class TestPreInpaintMaskRasterisation:
    """Regression guard for the mask-boundary jitter in _pre_inpaint_region.

    The pre-fix path rasterised the expanded quad via
    ``cv2.fillConvexPoly(mask, expanded.astype(np.int32), 255)`` followed
    by a 5×5 ``cv2.erode`` and a boolean paste. Two problems:

    1. ``.astype(np.int32)`` quantises each corner to the pixel grid,
       so a sub-pixel quad drift that crosses an integer boundary
       flipped the rasterised edge by a full pixel between frames.
    2. The hard (boolean) paste turned any mismatch between the
       inpainted and original backgrounds into a visible flickering
       1-px ring as that edge oscillated.

    The fix shrinks the expanded quad inward by a constant 2 px
    (replacing the 5×5 erode buffer) and rasterises at ``shift=4,
    lineType=cv2.LINE_AA`` so the boundary tracks sub-pixel motion
    proportionally. The grayscale mask is then used as a soft alpha.
    """

    def test_shrink_moves_corners_toward_centroid_by_exact_px(self):
        """Each corner ends up ``shrink_px`` closer to the centroid."""
        corners = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100],
        ], dtype=np.float32)
        shrunk = RevertStage._shrink_quad_to_centroid(corners, shrink_px=5.0)
        centroid = corners.mean(axis=0)
        for i in range(4):
            orig = float(np.linalg.norm(corners[i] - centroid))
            new = float(np.linalg.norm(shrunk[i] - centroid))
            assert abs((orig - new) - 5.0) < 1e-4

    def test_shrink_zero_is_identity(self):
        corners = np.array([
            [10.5, 20.7], [110.3, 20.7], [110.3, 70.1], [10.5, 70.1],
        ], dtype=np.float32)
        shrunk = RevertStage._shrink_quad_to_centroid(corners, shrink_px=0.0)
        np.testing.assert_allclose(shrunk, corners, atol=1e-4)

    def test_aa_mask_has_soft_boundary(self):
        """The rasterised mask must contain intermediate values at the edge."""
        shape = (40, 80)
        corners = np.array([
            [10, 10], [50.5, 10], [50.5, 30], [10, 30],
        ], dtype=np.float32)
        mask = RevertStage._build_antialiased_mask(corners, shape)
        assert mask.shape == shape
        assert mask.dtype == np.uint8
        unique = np.unique(mask)
        intermediate = unique[(unique > 0) & (unique < 255)]
        assert len(intermediate) > 0, (
            f"AA mask has no gradient values; unique={unique}"
        )

    def test_aa_mask_reduces_jitter_vs_int32_rasterisation(self):
        """Sub-pixel drift across a grid line moves AA mask less than hard mask.

        The pre-fix rasterisation truncates corner coords via
        ``.astype(np.int32)``, so 49.9→49 and 50.1→50 — a 0.2-px
        drift flips a full column of ~height pixels between the two
        masks. The AA path spreads the change across a few adjacent
        columns with small per-pixel deltas. Summed absolute diff
        must be strictly smaller for AA.
        """
        shape = (40, 80)
        corners_a = np.array([
            [10, 10], [49.9, 10], [49.9, 30], [10, 30],
        ], dtype=np.float32)
        corners_b = np.array([
            [10, 10], [50.1, 10], [50.1, 30], [10, 30],
        ], dtype=np.float32)

        # Old hard rasterisation, for comparison.
        hard_a = np.zeros(shape, dtype=np.uint8)
        hard_b = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(hard_a, corners_a.astype(np.int32), 255)
        cv2.fillConvexPoly(hard_b, corners_b.astype(np.int32), 255)
        hard_diff = int(np.abs(
            hard_a.astype(np.int32) - hard_b.astype(np.int32)
        ).sum())

        aa_a = RevertStage._build_antialiased_mask(corners_a, shape)
        aa_b = RevertStage._build_antialiased_mask(corners_b, shape)
        aa_diff = int(np.abs(
            aa_a.astype(np.int32) - aa_b.astype(np.int32)
        ).sum())

        assert hard_diff > 0, (
            "Test input no longer exercises the int32-quantisation "
            "jitter case; adjust corners_a / corners_b."
        )
        assert aa_diff < hard_diff, (
            f"AA mask jitters as much as hard mask: "
            f"hard_diff={hard_diff}, aa_diff={aa_diff}"
        )
