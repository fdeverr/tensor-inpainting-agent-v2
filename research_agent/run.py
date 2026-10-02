"""Public one-command entry point for the complete Tensor Inpainting Agent."""

from __future__ import annotations

import argparse

from .cli_options import add_optional_evaluation, parse_float_csv

from .workflow_full import (
    DEFAULT_RESEARCH_PROMPT,
    FullWorkflowConfig,
    run_full_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run method selection, candidate generation, fair evaluation, and reporting."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--mat-key", help="MAT variable name; auto-detected when omitted")
    parser.add_argument("--prompt", default=DEFAULT_RESEARCH_PROMPT)
    parser.add_argument("--output-dir", default="research_agent/outputs")
    parser.add_argument("--candidate-root", default="research_agent/algorithms/candidates")
    parser.add_argument("--approved-root", default="research_agent/algorithms/approved")
    parser.add_argument(
        "--knowledge-root",
        help=argparse.SUPPRESS,  # Legacy no-op; retained for older launch commands.
    )
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
    parser.add_argument("--method-max-steps", type=int, default=1500)
    parser.add_argument("--method-max-steps-ceiling", type=int, default=6000)
    parser.add_argument("--tuning-near-limit-ratio", type=float, default=0.9)
    parser.add_argument("--tuning-expansion-factor", type=float, default=2.0)
    parser.add_argument(
        "--fair-max-steps",
        type=int,
        default=3000,
        help="hard ceiling for the LLM-requested shared Day 6 training steps",
    )
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=4,
        help="maximum distinct structure configurations searched independently per model",
    )
    parser.add_argument(
        "--fair-learning-rates",
        type=parse_float_csv,
        default=(0.001, 0.01, 0.1),
        help="comma-separated Day 6 coarse learning-rate grid",
    )
    parser.add_argument(
        "--fair-learning-rate-refinement-factor", type=float, default=3.0
    )
    parser.add_argument(
        "--skip-fair-learning-rate-refinement",
        action="store_true",
        help="disable the two local Day 6 learning-rate refinement trials",
    )
    parser.add_argument("--max-improvement-rounds", type=int, default=5)
    parser.add_argument("--ablation-screen-trials", type=int, default=1)
    parser.add_argument("--ablation-screen-max-steps", type=int, default=300)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_optional_evaluation(parser, "full-reference-metrics")
    add_optional_evaluation(parser, "no-reference-metrics")
    add_optional_evaluation(parser, "selection-visual-assessment")
    add_optional_evaluation(parser, "mutation-visual-assessment")
    add_optional_evaluation(parser, "siren-comparison")
    parser.add_argument("--method-shortlist-size", type=int, default=3)
    parser.add_argument("--screening-trials", type=int, default=2)
    parser.add_argument("--screening-max-steps", type=int, default=400)
    parser.add_argument("--screening-patience", type=int, default=10)
    parser.add_argument("--siren-max-steps", type=int, default=4000)
    parser.add_argument("--siren-tuning-trials", type=int, default=4)
    parser.add_argument("--siren-validation-interval", type=int, default=25)
    parser.add_argument("--siren-patience", type=int, default=20)
    parser.add_argument("--llm-mode", choices=("auto", "off", "required"), default="auto")
    parser.add_argument("--retrieval-top-k", type=int, default=8)
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
    state = run_full_workflow(
        FullWorkflowConfig(
            image_path=args.image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            candidate_root=args.candidate_root,
            approved_root=args.approved_root,
            knowledge_root=args.knowledge_root,
            mask_type=args.mask_type,
            missing_rate=args.missing_rate,
            seed=args.seed,
            image_size=args.image_size or None,
            mat_key=args.mat_key,
            base_model=args.base_model,
            method_max_steps=args.method_max_steps,
            method_max_steps_ceiling=args.method_max_steps_ceiling,
            tuning_near_limit_ratio=args.tuning_near_limit_ratio,
            tuning_expansion_factor=args.tuning_expansion_factor,
            fair_max_steps=args.fair_max_steps,
            tuning_trials=args.tuning_trials,
            fair_learning_rate_candidates=args.fair_learning_rates,
            fair_refine_learning_rate=(
                not args.skip_fair_learning_rate_refinement
            ),
            fair_learning_rate_refinement_factor=(
                args.fair_learning_rate_refinement_factor
            ),
            max_improvement_rounds=args.max_improvement_rounds,
            ablation_screen_trials=args.ablation_screen_trials,
            ablation_screen_max_steps=args.ablation_screen_max_steps,
            validation_interval=args.validation_interval,
            patience=args.patience,
            device=args.device,
            full_reference_metrics=args.full_reference_metrics,
            no_reference_metrics=args.no_reference_metrics,
            selection_visual_assessment=args.selection_visual_assessment,
            mutation_visual_assessment=args.mutation_visual_assessment,
            siren_comparison=args.siren_comparison,
            method_shortlist_size=args.method_shortlist_size,
            screening_trials=args.screening_trials,
            screening_max_steps=args.screening_max_steps,
            screening_patience=args.screening_patience,
            siren_max_steps=args.siren_max_steps,
            siren_tuning_trials=args.siren_tuning_trials,
            siren_validation_interval=args.siren_validation_interval,
            siren_patience=args.siren_patience,
            llm_mode=args.llm_mode,
            retrieval_top_k=args.retrieval_top_k,
            minimum_psnr_delta=args.minimum_psnr_delta,
            ssim_tolerance=args.ssim_tolerance,
            smoke_timeout_seconds=args.smoke_timeout,
        )
    )
    print("Tensor Inpainting Agent completed")
    print("  run_id: %s" % state["run_id"])
    print("  candidate_accepted: %s" % state["candidate_accepted"])
    print("  best_algorithm: %s" % state["best_available"]["algorithm"])
    best_metrics = state["best_available"]["metrics"]
    best_missing_psnr = best_metrics.get("missing_psnr")
    best_full_psnr = best_metrics.get("full_psnr")
    print(
        "  best_missing_psnr: %s"
        % (
            "infinite (perfect)"
            if best_missing_psnr is None
            else "%.4f dB" % best_missing_psnr
        )
    )
    print(
        "  best_full_psnr: %s"
        % (
            "infinite (perfect)"
            if best_full_psnr is None
            else "%.4f dB" % best_full_psnr
        )
    )
    print("  best_ssim: %.4f" % state["best_available"]["metrics"]["composite_ssim"])
    enabled_optional_metrics = []
    if args.full_reference_metrics:
        enabled_optional_metrics.append(("lpips", "LPIPS"))
    if args.no_reference_metrics:
        enabled_optional_metrics.extend(
            [
                ("maniqa", "MANIQA"),
                ("clip_iqa", "CLIP-IQA"),
                ("musiq", "MUSIQ"),
            ]
        )
    for key, label in enabled_optional_metrics:
        value = best_metrics.get(key)
        print(
            "  best_%s: %s"
            % (label.lower(), "N/A" if value is None else "%.4f" % value)
        )
    print("  completion: %s" % state["artifacts"]["best_completion"])
    print("  completion_data: %s" % state["artifacts"]["best_completion_data"])
    if "best_completion_mat" in state["artifacts"]:
        print("  completion_mat: %s" % state["artifacts"]["best_completion_mat"])
    print("  comparison_images:")
    for role, path in state["artifacts"].get("comparison_images", {}).items():
        print("    %s: %s" % (role, path))
    print("  report: %s" % state["artifacts"]["report"])
    print("  state: %s" % state["artifacts"]["state"])
    if "run_practice" in state["artifacts"]:
        print(
            "  run_practice: %s"
            % state["artifacts"]["run_practice"]["practice_markdown"]
        )


if __name__ == "__main__":
    main()
