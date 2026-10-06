"""Run with python -m research_agent.run_recovery --help."""

import argparse

from .recovery import RecoveryConfig, run_recovery
from .recovery_data import DATA_TYPES
from .cli_options import add_llm_context_budget, apply_llm_context_budget


TRAINING_OPTIONS = (
    ("validation_interval", int, "研发训练：每 N 步用 GT 验证"),
    ("patience", int, "研发训练：连续 N 次验证未改善后早停（不是步数）"),
    ("evaluation_validation_interval", int, "数据集评测：每 N 步用 GT 验证"),
    ("evaluation_patience", int, "数据集评测早停耐心；0 关闭早停，默认 0"),
    ("method_max_steps_ceiling", int, "方法选择扩展硬上限；省略时等于 evolution-steps"),
    ("fair_max_steps", int, "进化候选公平比较预算；省略时等于 evolution-steps"),
    ("tuning_near_limit_ratio", float, "方法选择自动扩展触发比例"),
    ("tuning_expansion_factor", float, "方法选择自动扩展倍数"),
    ("fair_learning_rate_refinement_factor", float, "公平比较学习率精搜倍数"),
    ("ablation_screen_trials", int, "消融预赛 trial 数"),
    ("ablation_screen_max_steps", int, "消融预赛训练步数"),
    ("method_shortlist_size", int, "LLM 推荐并参与整类预赛的分解家族数，3–5"),
    ("screening_trials", int, "基线预赛 trial 数，1–3"),
    ("screening_max_steps", int, "基线预赛步数；省略时 min(200, evolution-steps)"),
    ("screening_patience", int, "基线预赛早停耐心"),
    ("siren_max_steps", int, "SIREN 每个调参 trial 的步数；省略时等于 evaluation-steps"),
    ("siren_tuning_trials", int, "SIREN 调参次数，1–4"),
    ("siren_validation_interval", int, "SIREN GT 验证间隔"),
    ("siren_patience", int, "SIREN 早停耐心"),
    ("retrieval_top_k", int, "知识检索数量"),
    ("minimum_psnr_delta", float, "候选晋级最小 PSNR 增益"),
    ("minimum_nmse_delta", float, "音频候选晋级最小 NMSE 降低量（线性值，默认 0，需严格降低）"),
    ("ssim_tolerance", float, "候选晋级允许的 SSIM 下降"),
)


def learning_rates(value):
    try:
        rates = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("学习率应是逗号分隔的数值") from error
    if not rates or any(not 1e-5 <= rate <= 1 for rate in rates):
        raise argparse.ArgumentTypeError("学习率必须在 [1e-5, 1] 内")
    return rates


def on_off(value):
    if value not in {"on", "off"}:
        raise argparse.ArgumentTypeError("开关值必须是 on 或 off")
    return value == "on"


def image_size(value):
    if value == "original":
        return 0
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("image-size 必须是整数或 original") from error


def add_switch(parser, positive, negative, dest, default, help_text=None):
    # Both the run_ai.sh 'on/off' syntax and the existing flag-only syntax work.
    parser.add_argument(*positive, dest=dest, nargs="?", const=True, type=on_off,
                        default=default, help=help_text)
    parser.add_argument(*negative, dest=dest, action="store_false", default=argparse.SUPPRESS)


def build_parser():
    defaults = RecoveryConfig(dataset_root="")
    parser = argparse.ArgumentParser(description="One development sample per modality, then fixed-algorithm whole-dataset comparison")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default="research_agent/outputs/recovery")
    parser.add_argument("--history-root", default="research_agent/algorithms/history")
    for name in ("candidate_root", "approved_root"):
        parser.add_argument("--" + name.replace("_", "-"), default=getattr(defaults, name))
    parser.add_argument("--knowledge-root", default=defaults.knowledge_root,
                        help="global experience root, partitioned by modality/base method; one summary per complete evolution run")
    parser.add_argument("--data-types", nargs="+", choices=tuple(DATA_TYPES), default=list(DATA_TYPES))
    parser.add_argument("--representative", action="append", default=[], metavar="TYPE=FILE")
    parser.add_argument("--mask-type", choices=("random", "block", "slices", "sildes"), default="random")
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=image_size, default=128, help="spatial resolution for non-audio; 0/original preserves source size")
    parser.add_argument("--audio-frame-size", type=int, default=256)
    parser.add_argument("--evolution-steps", "--method-max-steps", type=int, default=1500, dest="evolution_steps")
    parser.add_argument("--evaluation-steps", type=int, default=1000)
    parser.add_argument("--improvement-rounds", "--max-improvement-rounds", type=int, default=5, dest="improvement_rounds")
    parser.add_argument("--tuning-trials", type=int, default=4)
    for name, value_type, help_text in TRAINING_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=value_type,
                            default=getattr(defaults, name), help=help_text)
    parser.add_argument("--fair-learning-rates", type=learning_rates, dest="fair_learning_rate_candidates",
                        default=defaults.fair_learning_rate_candidates, help="逗号分隔的学习率，如 0.001,0.01,0.1")
    parser.add_argument("--siren-learning-rates", type=learning_rates, dest="siren_learning_rate_candidates",
                        default=defaults.siren_learning_rate_candidates, help="SIREN 调参学习率，默认 0.00005,0.0001,0.0003")
    add_switch(parser, ("--fair-learning-rate-refinement",),
               ("--no-fair-learning-rate-refinement", "--skip-fair-learning-rate-refinement"),
               "fair_refine_learning_rate", defaults.fair_refine_learning_rate)
    parser.add_argument("--smoke-timeout", type=float, dest="smoke_timeout_seconds",
                        default=defaults.smoke_timeout_seconds)
    parser.add_argument("--base-model", default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--llm-mode", choices=("auto", "off", "required"), default="auto")
    add_llm_context_budget(parser)
    add_switch(parser, ("--lpips", "--full-reference-metrics"),
               ("--no-lpips", "--skip-full-reference-metrics"), "lpips", False, "Image only")
    add_switch(parser, ("--siren-comparison",),
               ("--no-siren-comparison", "--skip-siren-comparison"), "siren_comparison", True)
    for name in ("no_reference_metrics", "selection_visual_assessment", "mutation_visual_assessment"):
        option = name.replace("_", "-")
        add_switch(parser, ("--" + option,), ("--skip-" + option,), name, False, "Image only; optional on/off")
    parser.add_argument("--prompt", default="", help="附加算法研发要求（应用于每个所选类型）")
    parser.add_argument("--inventory-only", action="store_true", help="validate and classify GT without training or LLM calls")
    parser.add_argument("--evaluate-only", action="store_true", help="evaluate archived champions without evolution or LLM calls")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    apply_llm_context_budget(args)
    representatives = {}
    for item in args.representative:
        if "=" not in item:
            parser.error("--representative must be TYPE=FILE")
        kind, path = item.split("=", 1)
        if kind in representatives:
            parser.error("duplicate representative for %s" % kind)
        representatives[kind] = path
    config = RecoveryConfig(
        dataset_root=args.dataset_root, output_dir=args.output_dir, history_root=args.history_root,
        data_types=tuple(args.data_types), representatives=representatives,
        mask_type=args.mask_type, missing_rate=args.missing_rate, seed=args.seed,
        image_size=args.image_size or None, audio_frame_size=args.audio_frame_size,
        evolution_steps=args.evolution_steps, evaluation_steps=args.evaluation_steps,
        improvement_rounds=args.improvement_rounds, tuning_trials=args.tuning_trials,
        **{name: getattr(args, name) for name, _, _ in TRAINING_OPTIONS},
        fair_learning_rate_candidates=args.fair_learning_rate_candidates,
        siren_learning_rate_candidates=args.siren_learning_rate_candidates,
        fair_refine_learning_rate=args.fair_refine_learning_rate,
        smoke_timeout_seconds=args.smoke_timeout_seconds,
        candidate_root=args.candidate_root, approved_root=args.approved_root,
        knowledge_root=args.knowledge_root,
        base_model=args.base_model, device=args.device, llm_mode=args.llm_mode,
        lpips=args.lpips, siren_comparison=args.siren_comparison,
        no_reference_metrics=args.no_reference_metrics,
        selection_visual_assessment=args.selection_visual_assessment,
        mutation_visual_assessment=args.mutation_visual_assessment, prompt=args.prompt)
    if args.inventory_only and args.evaluate_only:
        parser.error("--inventory-only and --evaluate-only are mutually exclusive")
    state = run_recovery(config, inventory_only=args.inventory_only, evaluate_only=args.evaluate_only)
    print("Status: %s\nOutput: %s" % (state["stage"], state["run_dir"]))
    for kind, samples in state["inventory"]["groups"].items():
        print("%s: %d valid samples" % (kind, len(samples)))
    for failure in state["inventory"]["failures"]:
        print("Excluded: %s — %s" % (failure["source"], failure["error"]))
    if state["stage"] == "COMPLETED_WITH_FAILURES":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
