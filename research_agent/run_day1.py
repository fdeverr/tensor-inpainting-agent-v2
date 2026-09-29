"""Command-line entry point for the Day 1 baseline."""

from __future__ import annotations

import argparse

from .cli_options import add_optional_evaluation

from .core.pipeline import run_day1_baseline
from .schemas import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reproducible Day 1 interpolation baseline."
    )
    parser.add_argument("--image", required=True, help="Path to a complete benchmark image")
    parser.add_argument("--mat-key", help="MAT variable name; auto-detected when omitted")
    parser.add_argument(
        "--output-dir",
        default="research_agent/outputs",
        help="Directory that will contain one subdirectory per run",
    )
    parser.add_argument(
        "--mask-type",
        choices=("random", "block"),
        default="block",
    )
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Resize the largest side to this value; use 0 to keep source size",
    )
    parser.add_argument("--missing-fill-value", type=float, default=0.0)
    add_optional_evaluation(parser, "full-reference-metrics")
    add_optional_evaluation(parser, "no-reference-metrics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        image_path=args.image,
        output_dir=args.output_dir,
        mask_type=args.mask_type,
        missing_rate=args.missing_rate,
        seed=args.seed,
        image_size=args.image_size or None,
        mat_key=args.mat_key,
        missing_fill_value=args.missing_fill_value,
        full_reference_metrics=args.full_reference_metrics,
        no_reference_metrics=args.no_reference_metrics,
    )
    result = run_day1_baseline(config)

    missing_psnr_text = (
        "infinite (perfect)"
        if result.missing_psnr is None
        else "%.4f dB" % result.missing_psnr
    )
    full_psnr_text = (
        "infinite (perfect)"
        if result.full_psnr is None
        else "%.4f dB" % result.full_psnr
    )
    print("Day 1 interpolation baseline completed")
    print("  run_id: %s" % result.run_id)
    print("  actual_missing_rate: %.4f" % result.actual_missing_rate)
    print("  missing_region_psnr: %s" % missing_psnr_text)
    print("  full_image_psnr: %s" % full_psnr_text)
    print("  composite_ssim: %.6f" % result.composite_ssim)
    if args.full_reference_metrics:
        print("  lpips: %s" % ("N/A" if result.lpips is None else "%.6f" % result.lpips))
    if args.no_reference_metrics:
        print("  maniqa: %s" % ("N/A" if result.maniqa is None else "%.6f" % result.maniqa))
        print("  clip_iqa: %s" % ("N/A" if result.clip_iqa is None else "%.6f" % result.clip_iqa))
        print("  musiq: %s" % ("N/A" if result.musiq is None else "%.6f" % result.musiq))
    print("  artifacts: %s" % result.artifacts["metrics"])


if __name__ == "__main__":
    main()
