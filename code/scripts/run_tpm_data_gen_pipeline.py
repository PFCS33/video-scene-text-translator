#!/usr/bin/env python3
"""CLI entry point for the video text replacement pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so `src` is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import PipelineConfig  # noqa: E402
from src.tpm_data_gen_pipeline import TPMDataGenPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-language scene text replacement in video"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument("--input", type=str, help="Input video path")
    parser.add_argument("--output_dir", type=str, help="Output directory for extracted ROIs")
    parser.add_argument(
        "--source-lang", type=str, default=None,
        help="Source language code (e.g., en)",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--debug-dir", type=str, default=None,
        help="Directory to save intermediate debug outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config from YAML, or use defaults
    config_path = args.config or str(project_root / "config" / "default.yaml")
    config = PipelineConfig.from_yaml(config_path)

    # CLI overrides
    if args.input:
        config.input_video = args.input
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.source_lang:
        config.translation.source_lang = args.source_lang
    config.translation.target_lang = None  # We only want to extract ROIs, so translation is not needed
    if args.log_level:
        config.log_level = args.log_level
    if args.debug_dir:
        config.debug_output_dir = args.debug_dir

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    pipeline = TPMDataGenPipeline(config)
    extraction_info = pipeline.run()

    if extraction_info:
        print(f"Done. {len(extraction_info)} text tracks extracted.")
        print(f"Output saved to: {config.output_dir}")
    else:
        print("No text tracks extracted. Check logs for details.")


if __name__ == "__main__":
    main()
