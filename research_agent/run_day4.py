"""Command-line entry point for Day 4 method selection."""

from __future__ import annotations

import argparse

from .cli_options import add_optional_evaluation, parse_float_csv, add_llm_context_budget, apply_llm_context_budget

from .workflow_day4 import Day4WorkflowConfig, run_day4_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve tensor knowledge, select a method, and run it."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--mat-key", help="MAT variable name; auto-detected when omitted")
    parser.add_argument("--output-dir", default="research_agent/outputs")
    parser.add_argument("--mask-type", choices=("random", "block", "slices", "sildes"), default="block")
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--base-model",
        choices=(
            "auto",
            "matrix",
            "mode3",
            "cp",
            "nonnegative_cp",
            "tucker",
            "btd",
            "tsvd",
            "nonnegative_tucker",
            "hierarchical_tucker",
            "tt",
            "tensor_ring",
        ),
        default="auto",
        help="automatically select or explicitly fix the base tensor decomposition",
    )
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-steps-ceiling", type=int, default=6000)
    parser.add_argument("--tuning-near-limit-ratio", type=float, default=0.9)
    parser.add_argument("--tuning-expansion-factor", type=float, default=2.0)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_optional_evaluation(parser, "full-reference-metrics")
    add_optional_evaluation(parser, "no-reference-metrics")
    add_optional_evaluation(parser, "selection-visual-assessment")
    add_optional_evaluation(parser, "siren-comparison")
    parser.add_argument("--method-shortlist-size", type=int, default=3)
    parser.add_argument("--screening-trials", type=int, default=2)
    parser.add_argument("--screening-max-steps", type=int, default=400)
    parser.add_argument("--screening-patience", type=int, default=10)
    parser.add_argument("--siren-max-steps", type=int, default=4000)
    parser.add_argument("--siren-tuning-trials", type=int, default=4)
    parser.add_argument("--siren-learning-rates", type=parse_float_csv, dest="siren_learning_rate_candidates", default=(5e-5, 1e-4, 3e-4))
    parser.add_argument("--siren-validation-interval", type=int, default=25)
    parser.add_argument("--siren-patience", type=int, default=20)
    parser.add_argument(
        "--llm-mode",
        choices=("auto", "off", "required"),
        default="auto",
        help="auto falls back to deterministic rules when LLM env vars are absent",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    add_llm_context_budget(parser)
    return parser


def _metric_text(value):
    return "infinite (perfect)" if value is None else "%.4f dB" % value


def main() -> None:
    args = build_parser().parse_args()
    apply_llm_context_budget(args)
    state = run_day4_workflow(
        Day4WorkflowConfig(
            image_path=args.image,
            output_dir=args.output_dir,
            mask_type=args.mask_type,
            missing_rate=args.missing_rate,
            seed=args.seed,
            image_size=args.image_size or None,
            mat_key=args.mat_key,
            model_name=args.base_model,
            max_steps=args.max_steps,
            max_steps_ceiling=args.max_steps_ceiling,
            tuning_near_limit_ratio=args.tuning_near_limit_ratio,
            tuning_expansion_factor=args.tuning_expansion_factor,
            validation_interval=args.validation_interval,
            patience=args.patience,
            device=args.device,
            full_reference_metrics=args.full_reference_metrics,
            no_reference_metrics=args.no_reference_metrics,
            llm_mode=args.llm_mode,
            retrieval_top_k=args.retrieval_top_k,
            selection_visual_assessment=args.selection_visual_assessment,
            siren_comparison=args.siren_comparison,
            method_shortlist_size=args.method_shortlist_size,
            screening_trials=args.screening_trials,
            screening_max_steps=args.screening_max_steps,
            screening_patience=args.screening_patience,
            siren_max_steps=args.siren_max_steps,
            siren_tuning_trials=args.siren_tuning_trials,
            siren_learning_rate_candidates=args.siren_learning_rate_candidates,
            siren_validation_interval=args.siren_validation_interval,
            siren_patience=args.siren_patience,
        )
    )
    plan = state["results"]["method_plan"]
    diagnostics = state["results"]["selector_diagnostics"]
    interpolation = state["results"]["interpolation_metrics"]
    tensor = state["results"]["tensor_metrics"]
    siren = state["results"].get("siren_comparison", {})
    comparison = state["results"].get(
        "baseline_comparison", state["results"]["comparison"]
    )
    print("Day 4 method-selection workflow completed")
    print("  run_id: %s" % state["run_id"])
    print("  selected_model: %s" % plan["method"])
    print("  selection_mode: %s" % plan["selection_mode"])
    print("  confidence: %.4f" % plan["confidence"])
    print("  fallback_reason: %s" % diagnostics["fallback_reason"])
    print(
        "  interpolation_missing_psnr: %s"
        % _metric_text(interpolation["missing_psnr"])
    )
    print("  tensor_missing_psnr: %s" % _metric_text(tensor["missing_psnr"]))
    if siren.get("status") == "completed":
        print(
            "  siren_missing_psnr: %s"
            % _metric_text(siren["metrics"]["missing_psnr"])
        )
    print("  interpolation_full_psnr: %s" % _metric_text(interpolation["full_psnr"]))
    print("  tensor_full_psnr: %s" % _metric_text(tensor["full_psnr"]))
    enabled_optional_metrics = []
    if args.full_reference_metrics:
        enabled_optional_metrics.append("lpips")
    if args.no_reference_metrics:
        enabled_optional_metrics.extend(("maniqa", "clip_iqa", "musiq"))
    for key in enabled_optional_metrics:
        value = tensor.get(key)
        print("  tensor_%s: %s" % (key, "N/A" if value is None else "%.6f" % value))
    print("  winner: %s" % comparison["winner"])
    print("  method_plan: %s" % state["artifacts"]["method_plan"])
    print("  state: %s" % state["artifacts"]["state"])


if __name__ == "__main__":
    main()
