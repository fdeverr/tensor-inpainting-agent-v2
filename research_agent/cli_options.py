"""Shared switches for optional, expensive evaluation stages."""

import argparse


def parse_float_csv(value: str) -> tuple[float, ...]:
    """Parse a non-empty comma-separated numeric CLI value."""

    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def add_optional_evaluation(parser: argparse.ArgumentParser, option: str) -> None:
    descriptions = {
        "full-reference-metrics": "full-reference LPIPS evaluation (PSNR/SSIM are always on)",
        "no-reference-metrics": "MANIQA/CLIP-IQA/MUSIQ evaluation",
        "selection-visual-assessment": (
            "one-shot multimodal analysis before decomposition selection"
        ),
        "mutation-visual-assessment": (
            "multimodal observations supplied to candidate mutation"
        ),
        "siren-comparison": "independent SIREN implicit-neural baseline",
    }
    destination = option.replace("-", "_")
    enabled_by_default = option in {"full-reference-metrics", "siren-comparison"}
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--" + option, dest=destination, action="store_true",
        help="enable %s%s"
        % (descriptions[option], " (default)" if enabled_by_default else ""),
    )
    group.add_argument(
        "--skip-" + option, dest=destination, action="store_false",
        help="disable %s%s"
        % (descriptions[option], "" if enabled_by_default else " (default)"),
    )
    parser.set_defaults(**{destination: enabled_by_default})
