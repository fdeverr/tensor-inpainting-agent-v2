"""Small reproducible benchmark driver for complete research-agent cases."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schemas import SUPPORTED_TENSOR_MODEL_NAMES
from .workflow import _make_run_id, _write_json
from .workflow_full import FullWorkflowConfig, run_full_workflow


@dataclass(frozen=True)
class BenchmarkConfig:
    image_paths: Sequence[str]
    mask_types: Sequence[str] = ("random", "block")
    missing_rates: Sequence[float] = (0.4,)
    output_dir: str = "research_agent/outputs"
    candidate_root: str = "research_agent/algorithms/candidates"
    approved_root: str = "research_agent/algorithms/approved"
    seed: int = 42
    image_size: int = 64
    mat_key: Optional[str] = None
    base_model: str = "auto"
    method_max_steps: int = 50
    method_max_steps_ceiling: int = 50
    tuning_near_limit_ratio: float = 0.9
    tuning_expansion_factor: float = 2.0
    fair_max_steps: int = 50
    tuning_trials: int = 2
    fair_learning_rate_candidates: tuple[float, ...] = (0.01,)
    fair_refine_learning_rate: bool = False
    fair_learning_rate_refinement_factor: float = 3.0
    max_improvement_rounds: int = 5
    screening_max_steps: int = 50
    screening_patience: int = 5
    siren_max_steps: int = 50
    siren_tuning_trials: int = 2
    siren_validation_interval: int = 10
    siren_patience: int = 5
    device: str = "auto"
    llm_mode: str = "off"
    full_reference_metrics: bool = True
    no_reference_metrics: bool = False
    selection_visual_assessment: bool = False
    mutation_visual_assessment: bool = False
    siren_comparison: bool = True

    def validate(self) -> None:
        if not self.image_paths:
            raise ValueError("at least one image is required")
        missing = [path for path in self.image_paths if not Path(path).is_file()]
        if missing:
            raise ValueError("benchmark images do not exist: %s" % missing)
        if not self.mask_types or any(item not in {"random", "block"} for item in self.mask_types):
            raise ValueError("mask_types may contain only random and block")
        if not self.missing_rates or any(not 0.0 < rate < 1.0 for rate in self.missing_rates):
            raise ValueError("missing_rates must be in (0, 1)")
        if self.base_model not in {"auto", *SUPPORTED_TENSOR_MODEL_NAMES}:
            raise ValueError(
                "base_model must be auto or one of %s"
                % sorted(SUPPORTED_TENSOR_MODEL_NAMES)
            )
        if not isinstance(self.full_reference_metrics, bool):
            raise ValueError("full_reference_metrics must be a bool")
        if not isinstance(self.no_reference_metrics, bool):
            raise ValueError("no_reference_metrics must be a bool")
        if not isinstance(self.selection_visual_assessment, bool):
            raise ValueError("selection_visual_assessment must be a bool")
        if not isinstance(self.mutation_visual_assessment, bool):
            raise ValueError("mutation_visual_assessment must be a bool")
        if not isinstance(self.siren_comparison, bool):
            raise ValueError("siren_comparison must be a bool")
        if min(
            self.method_max_steps,
            self.method_max_steps_ceiling,
            self.fair_max_steps,
            self.screening_max_steps,
            self.screening_patience,
            self.siren_max_steps,
            self.siren_validation_interval,
            self.siren_patience,
        ) < 1:
            raise ValueError("benchmark training budgets must be positive")
        if self.method_max_steps_ceiling < self.method_max_steps:
            raise ValueError(
                "method_max_steps_ceiling must be at least method_max_steps"
            )
        if not 0.0 < self.tuning_near_limit_ratio <= 1.0:
            raise ValueError("tuning_near_limit_ratio must be in (0, 1]")
        if self.tuning_expansion_factor <= 1.0:
            raise ValueError("tuning_expansion_factor must be greater than 1")
        if not self.fair_learning_rate_candidates or any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not 1e-5 <= float(rate) <= 1.0
            for rate in self.fair_learning_rate_candidates
        ):
            raise ValueError(
                "fair_learning_rate_candidates must contain numbers in [1e-5, 1.0]"
            )
        if not isinstance(self.fair_refine_learning_rate, bool):
            raise ValueError("fair_refine_learning_rate must be a bool")
        if self.fair_learning_rate_refinement_factor <= 1.0:
            raise ValueError(
                "fair_learning_rate_refinement_factor must be greater than 1"
            )
        if self.fair_refine_learning_rate and any(
            float(rate) / self.fair_learning_rate_refinement_factor < 1e-5
            or float(rate) * self.fair_learning_rate_refinement_factor > 1.0
            for rate in self.fair_learning_rate_candidates
        ):
            raise ValueError(
                "fair_learning_rate_candidates must remain in [1e-5, 1.0] "
                "after local refinement"
            )
        if not 1 <= self.siren_tuning_trials <= 4:
            raise ValueError("siren_tuning_trials must be in [1, 4]")
        if self.mat_key is not None and not self.mat_key.strip():
            raise ValueError("mat_key must be a non-empty string or None")


def aggregate_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate successful normalized method rows by semantic role."""

    successful = [case for case in cases if case["status"] == "completed"]
    roles: Dict[str, List[Dict[str, Any]]] = {}
    for case in successful:
        for method in case["method_results"]:
            roles.setdefault(method["role"], []).append(method)
    methods = {}
    for role, rows in roles.items():
        def values(metric_name: str) -> List[float]:
            return [
                float(row["metrics"][metric_name])
                for row in rows
                if row.get("metrics", {}).get(metric_name) is not None
            ]

        def summary(metric_values: List[float]) -> tuple:
            if not metric_values:
                return None, None
            return (
                statistics.fmean(metric_values),
                statistics.pstdev(metric_values)
                if len(metric_values) > 1
                else 0.0,
            )

        full_psnr = values("full_psnr")
        missing_psnr = values("missing_psnr")
        ssim = values("composite_ssim")
        lpips = values("lpips")
        maniqa = values("maniqa")
        clip_iqa = values("clip_iqa")
        musiq = values("musiq")
        mean_full_psnr, std_full_psnr = summary(full_psnr)
        mean_missing_psnr, std_missing_psnr = summary(missing_psnr)
        mean_ssim, std_ssim = summary(ssim)
        mean_lpips, std_lpips = summary(lpips)
        mean_maniqa, std_maniqa = summary(maniqa)
        mean_clip_iqa, std_clip_iqa = summary(clip_iqa)
        mean_musiq, std_musiq = summary(musiq)
        runtimes = [
            row["runtime_seconds"]
            for row in rows
            if row["runtime_seconds"] is not None
        ]
        parameters = [row["parameter_count"] for row in rows]
        methods[role] = {
            "evaluated_cases": len(rows),
            "unavailable_or_failed_cases": len(cases) - len(rows),
            "mean_full_psnr": mean_full_psnr,
            "std_full_psnr": std_full_psnr,
            "mean_missing_psnr": mean_missing_psnr,
            "std_missing_psnr": std_missing_psnr,
            "perfect_reconstruction_cases": sum(
                bool(row["metrics"].get("perfect_reconstruction", False))
                for row in rows
            ),
            "mean_composite_ssim": mean_ssim,
            "std_composite_ssim": std_ssim,
            "mean_lpips": mean_lpips,
            "std_lpips": std_lpips,
            "mean_maniqa": mean_maniqa,
            "std_maniqa": std_maniqa,
            "mean_clip_iqa": mean_clip_iqa,
            "std_clip_iqa": std_clip_iqa,
            "mean_musiq": mean_musiq,
            "std_musiq": std_musiq,
            "mean_runtime_seconds": statistics.fmean(runtimes) if runtimes else None,
            "mean_parameter_count": statistics.fmean(parameters),
        }
    return {
        "total_cases": len(cases),
        "successful_cases": len(successful),
        "failed_cases": len(cases) - len(successful),
        "candidate_acceptance_count": sum(
            bool(case.get("candidate_accepted")) for case in successful
        ),
        "methods_by_role": methods,
    }


def _benchmark_markdown(state: Dict[str, Any]) -> str:
    def mean_std(item: Dict[str, Any], name: str) -> str:
        mean = item.get("mean_%s" % name)
        std = item.get("std_%s" % name)
        return "N/A" if mean is None else "%.4f ± %.4f" % (mean, std)

    metric_columns = [
        ("missing_psnr", "Missing-region PSNR ↑"),
        ("full_psnr", "Full-image PSNR ↑"),
        ("composite_ssim", "SSIM ↑"),
    ]
    if state["config"].get("full_reference_metrics", True):
        metric_columns.append(("lpips", "LPIPS ↓"))
    if state["config"].get("no_reference_metrics", False):
        metric_columns.extend(
            [
                ("maniqa", "MANIQA ↑"),
                ("clip_iqa", "CLIP-IQA ↑"),
                ("musiq", "MUSIQ ↑"),
            ]
        )
    headers = ["角色", "完成", "未评估/失败"] + [
        label for _, label in metric_columns
    ] + ["平均训练时间", "平均参数量"]
    lines = [
        "# Tensor Inpainting Agent Benchmark",
        "",
        "## 范围",
        "",
        "- Cases：%d" % state["aggregate"]["total_cases"],
        "- 成功：%d" % state["aggregate"]["successful_cases"],
        "- 失败：%d" % state["aggregate"]["failed_cases"],
        "- 候选晋升次数：%d" % state["aggregate"]["candidate_acceptance_count"],
        "",
        "## 聚合结果",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for role, item in state["aggregate"]["methods_by_role"].items():
        runtime = (
            "N/A"
            if item["mean_runtime_seconds"] is None
            else "%.4f s" % item["mean_runtime_seconds"]
        )
        full_psnr_text = (
            "perfect"
            if item["mean_full_psnr"] is None
            else "%.4f ± %.4f"
            % (item["mean_full_psnr"], item["std_full_psnr"])
        )
        missing_psnr_text = (
            "perfect"
            if item["mean_missing_psnr"] is None
            else "%.4f ± %.4f"
            % (item["mean_missing_psnr"], item["std_missing_psnr"])
        )
        metric_values = {
            "missing_psnr": missing_psnr_text,
            "full_psnr": full_psnr_text,
            "composite_ssim": mean_std(item, "composite_ssim"),
            "lpips": mean_std(item, "lpips"),
            "maniqa": mean_std(item, "maniqa"),
            "clip_iqa": mean_std(item, "clip_iqa"),
            "musiq": mean_std(item, "musiq"),
        }
        cells = [
            role,
            str(item["evaluated_cases"]),
            str(item["unavailable_or_failed_cases"]),
        ] + [metric_values[name] for name, _ in metric_columns] + [
            runtime,
            "%.1f" % item["mean_parameter_count"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Case 明细",
            "",
            "| Image | Mask | Missing rate | 状态 | 最终算法 | 总时间 |",
            "|---|---|---:|---|---|---:|",
        ]
    )
    for case in state["cases"]:
        lines.append(
            "| %s | %s | %.2f | %s | %s | %.3f s |"
            % (
                case["image_path"],
                case["mask_type"],
                case["missing_rate"],
                case["status"],
                case.get("best_algorithm", "N/A"),
                case["wall_runtime_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "> 注意：少量图片上的 quick benchmark 只用于端到端回归，不足以支持泛化结论。",
            "> 默认全参考组包含 PSNR、SSIM、LPIPS；MANIQA、CLIP-IQA、MUSIQ 属于默认关闭的无参考组。N/A 表示指标不可用或对应权重加载失败。",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    config.validate()
    benchmark_id = _make_run_id("benchmark")
    directory = Path(config.output_dir) / benchmark_id
    directory.mkdir(parents=True, exist_ok=False)
    cases = []
    case_index = 0
    for image_path in config.image_paths:
        for mask_type in config.mask_types:
            for missing_rate in config.missing_rates:
                started_at = time.perf_counter()
                case_seed = config.seed + case_index
                case_index += 1
                try:
                    result = run_full_workflow(
                        FullWorkflowConfig(
                            image_path=image_path,
                            output_dir=config.output_dir,
                            candidate_root=config.candidate_root,
                            approved_root=config.approved_root,
                            mask_type=mask_type,
                            missing_rate=missing_rate,
                            seed=case_seed,
                            image_size=config.image_size,
                            mat_key=config.mat_key,
                            base_model=config.base_model,
                            method_max_steps=config.method_max_steps,
                            method_max_steps_ceiling=(
                                config.method_max_steps_ceiling
                            ),
                            tuning_near_limit_ratio=(
                                config.tuning_near_limit_ratio
                            ),
                            tuning_expansion_factor=(
                                config.tuning_expansion_factor
                            ),
                            fair_max_steps=config.fair_max_steps,
                            tuning_trials=config.tuning_trials,
                            fair_learning_rate_candidates=(
                                config.fair_learning_rate_candidates
                            ),
                            fair_refine_learning_rate=(
                                config.fair_refine_learning_rate
                            ),
                            fair_learning_rate_refinement_factor=(
                                config.fair_learning_rate_refinement_factor
                            ),
                            max_improvement_rounds=config.max_improvement_rounds,
                            screening_max_steps=config.screening_max_steps,
                            screening_patience=config.screening_patience,
                            siren_max_steps=config.siren_max_steps,
                            siren_tuning_trials=config.siren_tuning_trials,
                            siren_validation_interval=(
                                config.siren_validation_interval
                            ),
                            siren_patience=config.siren_patience,
                            device=config.device,
                            full_reference_metrics=config.full_reference_metrics,
                            no_reference_metrics=config.no_reference_metrics,
                            selection_visual_assessment=(
                                config.selection_visual_assessment
                            ),
                            mutation_visual_assessment=(
                                config.mutation_visual_assessment
                            ),
                            siren_comparison=config.siren_comparison,
                            llm_mode=config.llm_mode,
                        )
                    )
                    cases.append(
                        {
                            "image_path": image_path,
                            "mask_type": mask_type,
                            "missing_rate": missing_rate,
                            "seed": case_seed,
                            "status": "completed",
                            "run_id": result["run_id"],
                            "state_path": result["artifacts"]["state"],
                            "best_algorithm": result["best_available"]["algorithm"],
                            "candidate_accepted": result["candidate_accepted"],
                            "method_results": result["method_results"],
                            "wall_runtime_seconds": time.perf_counter() - started_at,
                        }
                    )
                except Exception as error:
                    cases.append(
                        {
                            "image_path": image_path,
                            "mask_type": mask_type,
                            "missing_rate": missing_rate,
                            "seed": case_seed,
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "wall_runtime_seconds": time.perf_counter() - started_at,
                        }
                    )
    state = {
        "benchmark_id": benchmark_id,
        "created_at": datetime.now().isoformat(),
        "config": asdict(config),
        "cases": cases,
        "aggregate": aggregate_cases(cases),
        "artifacts": {
            "state": str(directory / "benchmark.json"),
            "report": str(directory / "benchmark.md"),
        },
    }
    _write_json(directory / "benchmark.json", state)
    (directory / "benchmark.md").write_text(
        _benchmark_markdown(state), encoding="utf-8"
    )
    return state
