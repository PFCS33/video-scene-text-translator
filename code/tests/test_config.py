"""Tests for config loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import PipelineConfig, DetectionConfig


class TestPipelineConfig:
    def test_default_values(self):
        config = PipelineConfig()
        assert config.detection.ocr_confidence_threshold == 0.3
        assert config.translation.source_lang == "en"
        assert config.frontalization.optical_flow_method == "farneback"
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
        assert config.frontalization.optical_flow_method == "farneback"

    def test_from_yaml_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            tmp_path = f.name

        config = PipelineConfig.from_yaml(tmp_path)
        assert config.input_video == ""
        assert config.detection.ocr_confidence_threshold == 0.3
