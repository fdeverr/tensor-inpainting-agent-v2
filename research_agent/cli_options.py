"""Shared switches for optional, expensive evaluation stages."""

import argparse
import os

from .prompt_context import ContextBudget


def add_llm_context_budget(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-context-tokens", type=int,
                        default=os.getenv("LLM_CONTEXT_TOKENS", "131072"),
                        help="configured model context window (input plus reserved output)")
    parser.add_argument("--llm-output-reserve-tokens", type=int,
                        default=os.getenv("LLM_OUTPUT_RESERVE_TOKENS", "16384"),
                        help="output reserve within the context window; caps generation output")


def apply_llm_context_budget(args) -> None:
    budget = ContextBudget(args.llm_context_tokens, args.llm_output_reserve_tokens)
    os.environ["LLM_CONTEXT_TOKENS"] = str(budget.context_tokens)
    os.environ["LLM_OUTPUT_RESERVE_TOKENS"] = str(budget.output_reserve_tokens)


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
    enabled_by_default = option == "siren-comparison"
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
