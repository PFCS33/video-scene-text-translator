"""Tests for Stage 5: Revert (De-Frontalization + Compositing)."""

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
