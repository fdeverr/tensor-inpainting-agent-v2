"""Command-line entry point for Day 6 fair evaluation and improvement."""

from __future__ import annotations

import argparse

from .cli_options import add_optional_evaluation, parse_float_csv

from .workflow_day6 import Day6WorkflowConfig, run_day6_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fairly evaluate candidates in a configurable iterative evolution loop."
    )
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--candidate-root", default="research_agent/algorithms/candidates")
    parser.add_argument("--approved-root", default="research_agent/algorithms/approved")
    parser.add_argument(
        "--knowledge-root",
        help="cross-run reusable experience root; defaults beside candidate-root",
    )
    parser.add_argument("--output-dir", default="research_agent/outputs")
    parser.add_argument("--llm-mode", choices=("auto", "off", "required"), default="auto")
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=4,
        help="maximum distinct structure configurations searched independently per model",
    )
    parser.add_argument(
        "--learning-rates",
        type=parse_float_csv,
        default=(0.001, 0.01, 0.1),
        help="comma-separated coarse learning-rate grid",
    )
    parser.add_argument("--learning-rate-refinement-factor", type=float, default=3.0)
    parser.add_argument(
        "--skip-learning-rate-refinement", action="store_true"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="hard ceiling for the LLM-requested shared training steps",
    )
    parser.add_argument("--max-improvement-rounds", type=int, default=5)
    parser.add_argument("--ablation-screen-trials", type=int, default=1)
    parser.add_argument("--ablation-screen-max-steps", type=int, default=300)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_optional_evaluation(parser, "full-reference-metrics")
    add_optional_evaluation(parser, "no-reference-metrics")
    add_optional_evaluation(parser, "mutation-visual-assessment")
    parser.add_argument(
        "--minimum-psnr-delta",
        type=float,
        default=0.2,
        help="minimum missing-region PSNR gain required for promotion",
    )
    parser.add_argument("--ssim-tolerance", type=float, default=0.002)
    parser.add_argument("--smoke-timeout", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = run_day6_workflow(
        Day6WorkflowConfig(
            base_run_dir=args.base_run_dir,
            initial_candidate_dir=args.candidate_dir,
            candidate_root=args.candidate_root,
            approved_root=args.approved_root,
            knowledge_root=args.knowledge_root,
            output_dir=args.output_dir,
            llm_mode=args.llm_mode,
            tuning_trials=args.tuning_trials,
            learning_rate_candidates=args.learning_rates,
            refine_learning_rate=not args.skip_learning_rate_refinement,
            learning_rate_refinement_factor=(
                args.learning_rate_refinement_factor
            ),
            max_steps=args.max_steps,
            max_improvement_rounds=args.max_improvement_rounds,
            ablation_screen_trials=args.ablation_screen_trials,
            ablation_screen_max_steps=args.ablation_screen_max_steps,
            validation_interval=args.validation_interval,
            patience=args.patience,
            device=args.device,
            full_reference_metrics=args.full_reference_metrics,
            no_reference_metrics=args.no_reference_metrics,
            visual_assessment=args.mutation_visual_assessment,
            minimum_psnr_delta=args.minimum_psnr_delta,
            ssim_tolerance=args.ssim_tolerance,
            smoke_timeout_seconds=args.smoke_timeout,
        )
    )
    print("Day 6 fair experiment completed")
    print("  workflow_id: %s" % state["workflow_id"])
    print("  rounds: %d" % len(state["rounds"]))
    print("  accepted: %s" % state["accepted"])
    print("  stop_reason: %s" % state["stop_reason"])
    print("  best_algorithm: %s" % state["best_available"]["algorithm"])
    best_metrics = state["best_available"]["metrics"]
    for name in ("missing_psnr", "full_psnr"):
        value = best_metrics.get(name)
        print(
            "  best_%s: %s"
            % (
                name,
                "infinite (perfect)" if value is None else "%.4f dB" % value,
            )
        )
    print("  reconstruction: %s" % state["best_available"]["reconstruction"])
    print(
        "  run_practice: %s"
        % state["artifacts"]["run_practice"]["practice_markdown"]
    )
    print(
        "  global_experience: %s"
        % state["artifacts"]["global_experience"]["experience_markdown"]
    )


if __name__ == "__main__":
    main()
