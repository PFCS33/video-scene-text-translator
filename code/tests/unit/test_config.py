"""Tests for config loading and validation."""

import tempfile

import yaml

from src.config import PipelineConfig


class TestPipelineConfig:
    def test_default_values(self):
        config = PipelineConfig()
        assert config.detection.ocr_confidence_threshold == 0.3
        assert config.translation.source_lang == "en"
        assert config.detection.optical_flow_method == "farneback"
        assert config.text_editor.backend == "placeholder"

    def test_validate_missing_input(self):
        config = PipelineConfig()
        errors = config.validate()
        assert any("input_video" in e for e in errors)

    def test_validate_missing_output(self):
        config = PipelineConfig()
        config.input_video = "test.mp4"
        errors = config.validate()
        assert any("output_video" in e for e in errors)

    def test_validate_valid(self, default_config):
        errors = default_config.validate()
        assert errors == []

    def test_validate_bad_detection_weights(self):
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.detection.weight_ocr_confidence = 0.5
        config.detection.weight_sharpness = 0.5
        config.detection.weight_contrast = 0.5
        config.detection.weight_frontality = 0.5
        errors = config.validate()
        assert any("Detection scoring weights must sum to 1.0" in e for e in errors)

    def test_validate_bad_ref_weights(self):
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.detection.ref_weight_contrast = 0.5
        config.detection.ref_weight_frontality = 0.8
        errors = config.validate()
        assert any("Reference selection weights must sum to 1.0" in e for e in errors)

    def test_validate_bad_ref_sharpness_top_k(self):
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.detection.ref_sharpness_top_k = 0
        errors = config.validate()
        assert any("ref_sharpness_top_k" in e for e in errors)

    def test_default_ref_selection_values(self):
        config = PipelineConfig()
        assert config.detection.ref_ocr_min_confidence == 0.7
        assert config.detection.ref_sharpness_top_k == 10
        assert config.detection.ref_weight_contrast == 0.7
        assert config.detection.ref_weight_frontality == 0.3

    def test_validate_bad_confidence(self):
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.detection.ocr_confidence_threshold = 1.5
        errors = config.validate()
        assert any("ocr_confidence_threshold" in e for e in errors)

    def test_refiner_requires_s4_target_canonical_flag(self):
        """Enabling the refiner without S4's save flag must fail validation."""
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.revert.use_refiner = True
        config.revert.refiner_checkpoint_path = "some/path.pt"
        config.propagation.save_target_canonical_roi = False
        errors = config.validate()
        assert any("save_target_canonical_roi" in e for e in errors)

    def test_refiner_requires_checkpoint_path(self):
        """Enabling the refiner without a checkpoint path must fail."""
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.revert.use_refiner = True
        config.revert.refiner_checkpoint_path = ""
        config.propagation.save_target_canonical_roi = True
        errors = config.validate()
        assert any("refiner_checkpoint_path" in e for e in errors)

    def test_refiner_bad_rejection_threshold(self):
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.revert.use_refiner = True
        config.propagation.save_target_canonical_roi = True
        config.revert.refiner_rejection_warn_threshold = 1.5
        errors = config.validate()
        assert any("refiner_rejection_warn_threshold" in e for e in errors)

    def test_refiner_bad_max_corner_offset(self):
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.revert.use_refiner = True
        config.propagation.save_target_canonical_roi = True
        config.revert.refiner_max_corner_offset_px = 0.0
        errors = config.validate()
        assert any("refiner_max_corner_offset_px" in e for e in errors)

    def test_refiner_valid_setup_passes(self):
        """A fully-wired refiner config must pass validation."""
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.revert.use_refiner = True
        config.revert.refiner_checkpoint_path = "checkpoints/refiner/refiner_v0.pt"
        config.propagation.save_target_canonical_roi = True
        errors = config.validate()
        assert errors == []

    def test_refiner_off_doesnt_require_s4_flag(self):
        """With use_refiner=False, S4's save flag can be anything."""
        config = PipelineConfig()
        config.input_video = "in.mp4"
        config.output_video = "out.mp4"
        config.revert.use_refiner = False
        config.propagation.save_target_canonical_roi = False
        errors = config.validate()
        assert errors == []

    def test_from_yaml(self):
        data = {
            "input_video": "/path/to/video.mp4",
            "output_video": "/path/to/output.mp4",
            "translation": {
                "source_lang": "en",
                "target_lang": "zh-cn",
            },
            "detection": {
                "ocr_confidence_threshold": 0.5,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            tmp_path = f.name

        config = PipelineConfig.from_yaml(tmp_path)
        assert config.input_video == "/path/to/video.mp4"
        assert config.translation.target_lang == "zh-cn"
        assert config.detection.ocr_confidence_threshold == 0.5
        # Defaults preserved for unspecified fields
        assert config.detection.optical_flow_method == "farneback"

    def test_from_yaml_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            tmp_path = f.name

        config = PipelineConfig.from_yaml(tmp_path)
        assert config.input_video == ""
        assert config.detection.ocr_confidence_threshold == 0.3

    def test_from_yaml_adv_parses_refiner_fields(self):
        """Load the checked-in adv.yaml and verify the refiner block
        parses cleanly into RevertConfig."""
        config = PipelineConfig.from_yaml("config/adv.yaml")
        assert config.revert.use_refiner is True
        assert config.revert.refiner_checkpoint_path.endswith("refiner_v0.pt")
        assert tuple(config.revert.refiner_image_size) == (64, 128)
        assert config.revert.refiner_max_corner_offset_px == 16.0
        assert config.revert.refiner_rejection_warn_threshold == 0.1
        # And S4's save flag must be on so the validator can accept it.
        assert config.propagation.save_target_canonical_roi is True

    def test_from_yaml_adv_validates(self):
        """The refiner rules inside validate() must not reject adv.yaml.

        This is a drift guard: if someone later adds a stricter rule to
        validate() without updating adv.yaml (or vice versa), this fails.
        """
        config = PipelineConfig.from_yaml("config/adv.yaml")
        config.input_video = "dummy.mp4"
        config.output_video = "dummy_out.mp4"
        errors = config.validate()
        assert errors == [], f"adv.yaml failed validation: {errors}"
