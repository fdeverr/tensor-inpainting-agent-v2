"""One-command orchestration of the complete seven-day research workflow."""

from __future__ import annotations

import shutil
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_tools.framework import TraceLogger
from .core.audio_metrics import audio_metric_context, audio_metadata_for_gt, metric_score
from .dataset_evolution import compact_case_feedback, evaluate_across_cases
from .reporting import build_method_results, write_research_report
from .schemas import SUPPORTED_TENSOR_MODEL_NAMES
from .workflow import _make_run_id, _write_json
from .workflow_day4 import Day4WorkflowConfig, run_day4_workflow
from .workflow_day5 import Day5WorkflowConfig, run_day5_workflow
from .workflow_day6 import Day6WorkflowConfig, run_day6_workflow


DEFAULT_RESEARCH_PROMPT = "请分析彩图、MSI、视频或音频波形张量及其缺失模式，选择合适的张量分解，并在公平实验下提出、验证和改进一个补全算法。"


@dataclass(frozen=True)
class FullWorkflowConfig:
    image_path: str
    prompt: str = DEFAULT_RESEARCH_PROMPT
    output_dir: str = "research_agent/outputs"
    candidate_root: str = "research_agent/algorithms/candidates"
    approved_root: str = "research_agent/algorithms/approved"
    knowledge_root: Optional[str] = None
    mask_type: str = "block"
    missing_rate: float = 0.4
    seed: int = 42
    image_size: Optional[int] = 128
    mat_key: Optional[str] = None
    observation_mask_path: Optional[str] = None
    data_type: Optional[str] = None
    evolution_cases: Optional[List[Dict[str, Any]]] = None
    dataset_evaluation_steps: int = 1000
    dataset_evaluation_validation_interval: int = 10
    dataset_evaluation_patience: int = 0
    historical_algorithm_reference: Optional[List[Dict[str, Any]]] = None
    dataset_algorithm_reference: Optional[Dict[str, Any]] = None
    valid_element_count: Optional[int] = None
    base_model: str = "auto"
    method_max_steps: int = 1500
    method_max_steps_ceiling: int = 6000
    tuning_near_limit_ratio: float = 0.9
    tuning_expansion_factor: float = 2.0
    fair_max_steps: int = 3000
    tuning_trials: int = 4
    ablation_screen_trials: int = 1
    ablation_screen_max_steps: int = 300
    fair_learning_rate_candidates: tuple[float, ...] = (0.001, 0.01, 0.1)
    fair_refine_learning_rate: bool = True
    fair_learning_rate_refinement_factor: float = 3.0
    max_improvement_rounds: int = 5
    validation_interval: int = 10
    patience: int = 20
    device: str = "auto"
    llm_mode: str = "auto"
    retrieval_top_k: int = 8
    minimum_psnr_delta: float = 0.2
    minimum_nmse_delta: float = 0.0
    ssim_tolerance: float = 0.002
    smoke_timeout_seconds: float = 10.0
    full_reference_metrics: bool = False
    no_reference_metrics: bool = False
    selection_visual_assessment: bool = False
    mutation_visual_assessment: bool = False
    method_shortlist_size: int = 3
    screening_trials: int = 2
    screening_max_steps: int = 400
    screening_patience: int = 10
    siren_comparison: bool = True
    siren_max_steps: int = 4000
    siren_tuning_trials: int = 4
    siren_validation_interval: int = 25
    siren_patience: int = 20

    def validate(self) -> None:
        if not Path(self.image_path).is_file():
            raise ValueError("image_path does not point to a file: %s" % self.image_path)
        if self.evolution_cases and (self.dataset_evaluation_steps < 1 or self.dataset_evaluation_validation_interval < 1 or self.dataset_evaluation_patience < 0):
            raise ValueError("invalid whole-dataset evaluation budget")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if self.mask_type not in {"random", "block", "slices", "sildes"}:
            raise ValueError("mask_type must be random or block")
        if not 0.0 < self.missing_rate < 1.0:
            raise ValueError("missing_rate must be in (0, 1)")
        if self.image_size is not None and self.image_size < 8:
            raise ValueError("image_size must be at least 8 or None")
        if self.mat_key is not None and not self.mat_key.strip():
            raise ValueError("mat_key must be a non-empty string or None")
        if self.base_model not in {"auto", *SUPPORTED_TENSOR_MODEL_NAMES}:
            raise ValueError(
                "base_model must be auto or one of %s"
                % sorted(SUPPORTED_TENSOR_MODEL_NAMES)
            )
        if min(self.method_max_steps, self.fair_max_steps) < 1:
            raise ValueError("training steps must be positive")
        if self.method_max_steps_ceiling < self.method_max_steps:
            raise ValueError(
                "method_max_steps_ceiling must be at least method_max_steps"
            )
        if not 0.0 < self.tuning_near_limit_ratio <= 1.0:
            raise ValueError("tuning_near_limit_ratio must be in (0, 1]")
        if self.tuning_expansion_factor <= 1.0:
            raise ValueError("tuning_expansion_factor must be greater than 1")
        if not 1 <= self.tuning_trials <= 5:
            raise ValueError("tuning_trials must be in [1, 5]")
        if not 1 <= self.ablation_screen_trials <= 3:
            raise ValueError("ablation_screen_trials must be in [1, 3]")
        if self.ablation_screen_max_steps < 1:
            raise ValueError("ablation_screen_max_steps must be positive")
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
        if not 1 <= self.max_improvement_rounds <= 100:
            raise ValueError("max_improvement_rounds must be in [1, 100]")
        if self.minimum_nmse_delta < 0:
            raise ValueError("minimum_nmse_delta must be nonnegative")
        if not isinstance(self.full_reference_metrics, bool):
            raise ValueError("full_reference_metrics must be a bool")
        if not isinstance(self.no_reference_metrics, bool):
            raise ValueError("no_reference_metrics must be a bool")
        if not isinstance(self.selection_visual_assessment, bool):
            raise ValueError("selection_visual_assessment must be a bool")
        if not isinstance(self.mutation_visual_assessment, bool):
            raise ValueError("mutation_visual_assessment must be a bool")
        if not 3 <= self.method_shortlist_size <= 5:
            raise ValueError("method_shortlist_size must be in [3, 5]")
        if not 1 <= self.screening_trials <= 3:
            raise ValueError("screening_trials must be in [1, 3]")
        if self.screening_max_steps < 1:
            raise ValueError("screening_max_steps must be positive")
        if self.screening_patience < 1:
            raise ValueError("screening_patience must be positive")
        if not isinstance(self.siren_comparison, bool):
            raise ValueError("siren_comparison must be a bool")
        if self.siren_max_steps < 1:
            raise ValueError("siren_max_steps must be positive")
        if not 1 <= self.siren_tuning_trials <= 4:
            raise ValueError("siren_tuning_trials must be in [1, 4]")
        if self.siren_validation_interval < 1 or self.siren_patience < 1:
            raise ValueError(
                "siren_validation_interval and siren_patience must be positive"
            )


def _score(metrics: Dict[str, Any]) -> float:
    return metric_score(metrics)


def _export_comparison_images(
    run_dir: Path,
    day4: Dict[str, Any],
    method_results: list[Dict[str, Any]],
) -> Dict[str, str]:
    """Copy comparison-ready images into one predictable top-level directory."""

    directory = run_dir / "comparison_images"
    directory.mkdir(parents=True, exist_ok=True)
    sources = {
        "corrupted_input": day4["artifacts"].get(
            "corrupted_preview", day4["artifacts"]["corrupted"]
        )
    }
    output_names = {
        "corrupted_input": "00_corrupted_input.png",
        "interpolation_baseline": "01_manhattan_interpolation.png",
        "implicit_neural_baseline": "02_siren_baseline.png",
        "tensor_baseline": "03_tensor_baseline.png",
        "candidate": "04_candidate.png",
    }
    for result in method_results:
        role = result["role"]
        if role in output_names:
            sources[role] = result.get("preview", result["reconstruction"])

    exported = {}
    for role, source in sources.items():
        destination = directory / output_names[role]
        shutil.copy2(source, destination)
        exported[role] = str(destination)
    return exported


class FullResearchWorkflow:
    """Compose stable child workflows while retaining their independent traces."""

    def __init__(self, config: FullWorkflowConfig) -> None:
        config.validate()
        self.config = config
        self.run_id = _make_run_id("research-agent")
        self.run_dir = Path(config.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.state_path = self.run_dir / "state.json"
        self.report_path = self.run_dir / "report.md"
        self.trace = TraceLogger(output_dir=str(self.run_dir / "traces"), sanitize=True)
        self.state: Dict[str, Any] = {
            "run_id": self.run_id,
            "stage": "CREATED",
            "prompt": config.prompt,
            "config": asdict(config),
            "child_runs": {},
            "method_results": [],
            "best_available": None,
            "candidate_accepted": False,
            "artifacts": {
                "run_dir": str(self.run_dir),
                "state": str(self.state_path),
                "report": str(self.report_path),
                "trace_jsonl": str(self.trace.jsonl_path),
                "trace_html": str(self.trace.html_path),
            },
            "last_error": None,
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, self.state)

    def _child_completed(self, day: str, state: Dict[str, Any]) -> None:
        state_path = state["artifacts"]["state"]
        self.state["child_runs"][day] = {
            "id": state.get("run_id", state.get("workflow_id")),
            "stage": state["stage"],
            "state": state_path,
        }
        self.state["artifacts"]["%s_state" % day] = state_path
        self._save()
        self.trace.log_event(
            "child_workflow_completed",
            {
                "day": day,
                "child_id": self.state["child_runs"][day]["id"],
                "stage": state["stage"],
            },
        )

    def run(self) -> Dict[str, Any]:
        self.trace.log_event(
            "session_start",
            {
                "workflow": "complete_research_agent",
                "run_id": self.run_id,
                "prompt": self.config.prompt,
            },
        )
        day4: Dict[str, Any]
        day5: Dict[str, Any]
        day6: Optional[Dict[str, Any]] = None
        try:
            # ── 阶段 1：方法选择（Day 4 工作流）──────────────────────────
            # 输入：图片、mask 类型/缺失率、训练预算（Day4WorkflowConfig）
            # 作用：分析图像 → 最近邻插值 → 自动/手动选择十一种分解之一 → 训练基础张量模型
            # 输出：day4 state（selected_model、image_profile、插值/张量指标、run 目录路径）
            self.state["stage"] = "METHOD_SELECTION"
            self._save()
            print("\n" + "=" * 72, flush=True)
            print("📌 阶段 1/4：图像分析、知识检索与基础方法自动调参", flush=True)
            print("=" * 72, flush=True)
            day4 = run_day4_workflow(
                Day4WorkflowConfig(
                    image_path=self.config.image_path,
                    output_dir=self.config.output_dir,
                    mask_type=self.config.mask_type,
                    missing_rate=self.config.missing_rate,
                    seed=self.config.seed,
                    image_size=self.config.image_size,
                    mat_key=self.config.mat_key,
                    observation_mask_path=self.config.observation_mask_path,
                    data_type=self.config.data_type,
                    historical_algorithm_reference=self.config.historical_algorithm_reference,
                    valid_element_count=self.config.valid_element_count,
                    model_name=self.config.base_model,
                    max_steps=self.config.method_max_steps,
                    max_steps_ceiling=self.config.method_max_steps_ceiling,
                    tuning_near_limit_ratio=self.config.tuning_near_limit_ratio,
                    tuning_expansion_factor=self.config.tuning_expansion_factor,
                    validation_interval=self.config.validation_interval,
                    patience=self.config.patience,
                    device=self.config.device,
                    full_reference_metrics=self.config.full_reference_metrics,
                    no_reference_metrics=self.config.no_reference_metrics,
                    llm_mode=self.config.llm_mode,
                    retrieval_top_k=self.config.retrieval_top_k,
                    selection_visual_assessment=(
                        self.config.selection_visual_assessment
                    ),
                    method_shortlist_size=self.config.method_shortlist_size,
                    screening_cases=self.config.evolution_cases,
                    dataset_algorithm_reference=self.config.dataset_algorithm_reference,
                    screening_trials=self.config.screening_trials,
                    screening_max_steps=self.config.screening_max_steps,
                    screening_patience=self.config.screening_patience,
                    siren_comparison=self.config.siren_comparison,
                    siren_max_steps=self.config.siren_max_steps,
                    siren_tuning_trials=self.config.siren_tuning_trials,
                    siren_validation_interval=(
                        self.config.siren_validation_interval
                    ),
                    siren_patience=self.config.siren_patience,
                )
            )
            self._child_completed("day4", day4)

            dataset_reference = copy.deepcopy(self.config.dataset_algorithm_reference)
            if self.config.evolution_cases:
                selected = day4["results"]["selected_trial"]
                selected_incumbent = evaluate_across_cases(
                    day4["selected_model"], None, selected,
                    self.config.evolution_cases,
                    self.run_dir / "pre_candidate_incumbent_dataset",
                    self.config.dataset_evaluation_steps,
                    self.config.dataset_evaluation_validation_interval,
                    self.config.dataset_evaluation_patience,
                    self.config.device,
                )
                dataset_reference = dataset_reference or {}
                dataset_reference["selected_incumbent_before_evolution"] = {
                    "algorithm": day4["selected_model"],
                    "configuration_source": "Day 4 selected configuration; Day 6 may retune the incumbent",
                    "summary": selected_incumbent["summary"],
                    "samples": compact_case_feedback(selected_incumbent),
                }
                reference_path = self.run_dir / "pre_candidate_incumbent_dataset.json"
                _write_json(reference_path, selected_incumbent)
                self.state["artifacts"]["pre_candidate_incumbent_dataset"] = str(reference_path)
                self._save()

            # ── 阶段 2：候选生成与验证（Day 5 工作流）────────────────────
            # 输入：day4 的 run 目录（base_run_dir，作为改进起点）
            # 作用：读 Day4 上下文 → 生成候选改进（LLM/确定性模板）→ Schema + AST + smoke 三级验证
            # 输出：day5 state（candidate_dir、validation 状态、eligible_for_training 标志）
            self.state["stage"] = "CANDIDATE_GENERATION"
            self._save()
            print("\n" + "=" * 72, flush=True)
            print("📌 阶段 2/4：LLM 设计候选架构并执行代码验证", flush=True)
            print("=" * 72, flush=True)
            day5 = run_day5_workflow(
                Day5WorkflowConfig(
                    base_run_dir=day4["artifacts"]["run_dir"],
                    candidate_root=self.config.candidate_root,
                    output_dir=self.config.output_dir,
                    knowledge_root=self.config.knowledge_root,
                    llm_mode=self.config.llm_mode,
                    smoke_timeout_seconds=self.config.smoke_timeout_seconds,
                    visual_assessment=self.config.mutation_visual_assessment,
                    dataset_algorithm_reference=dataset_reference,
                )
            )
            self._child_completed("day5", day5)

            # ── 阶段 3：公平实验与晋升（Day 6 工作流，条件执行）──────────
            # 输入：day4 的 run 目录（基线）+ day5 的候选目录
            # 作用：成对调参 → 缺失区 GT 选优 → 确定性 Judge → 接受则晋升 / 拒绝则带反馈改进
            # 输出：day6 state（rounds、accepted、promotion）；候选验证失败时跳过，day6 保持 None
            if day5["validation"]["eligible_for_training"]:
                self.state["stage"] = "FAIR_EVALUATION"
                self._save()
                print("\n" + "=" * 72, flush=True)
                print("📌 阶段 3/4：基线/候选配对调参、重训与 Judge 评估", flush=True)
                print("=" * 72, flush=True)
                day6 = run_day6_workflow(
                    Day6WorkflowConfig(
                        base_run_dir=day4["artifacts"]["run_dir"],
                        initial_candidate_dir=day5["artifacts"]["candidate_dir"],
                        candidate_root=self.config.candidate_root,
                        approved_root=self.config.approved_root,
                        output_dir=self.config.output_dir,
                        knowledge_root=self.config.knowledge_root,
                        llm_mode=self.config.llm_mode,
                        tuning_trials=self.config.tuning_trials,
                        ablation_screen_trials=self.config.ablation_screen_trials,
                        ablation_screen_max_steps=(
                            self.config.ablation_screen_max_steps
                        ),
                        learning_rate_candidates=(
                            self.config.fair_learning_rate_candidates
                        ),
                        refine_learning_rate=(
                            self.config.fair_refine_learning_rate
                        ),
                        learning_rate_refinement_factor=(
                            self.config.fair_learning_rate_refinement_factor
                        ),
                        max_steps=self.config.fair_max_steps,
                        max_improvement_rounds=self.config.max_improvement_rounds,
                        validation_interval=self.config.validation_interval,
                        patience=self.config.patience,
                        device=self.config.device,
                        full_reference_metrics=self.config.full_reference_metrics,
                        no_reference_metrics=self.config.no_reference_metrics,
                        visual_assessment=self.config.mutation_visual_assessment,
                        minimum_psnr_delta=self.config.minimum_psnr_delta,
                        minimum_nmse_delta=self.config.minimum_nmse_delta,
                        ssim_tolerance=self.config.ssim_tolerance,
                        smoke_timeout_seconds=self.config.smoke_timeout_seconds,
                        evolution_cases=self.config.evolution_cases,
                        dataset_evaluation_steps=self.config.dataset_evaluation_steps,
                        dataset_evaluation_validation_interval=self.config.dataset_evaluation_validation_interval,
                        dataset_evaluation_patience=self.config.dataset_evaluation_patience,
                        dataset_algorithm_reference=dataset_reference,
                    )
                )
                self._child_completed("day6", day6)
                self.state["artifacts"]["run_practice"] = day6["artifacts"][
                    "run_practice"
                ]
            else:
                print(
                    "\n⚠️  阶段 3/4 已跳过：Day 5 候选未通过代码验证。",
                    flush=True,
                )

            # ── 阶段 4：汇总 → 选冠军 → 出报告 ──────────────────────────
            # 单样本按本图指标选输出；Recovery 输出每轮整类 Judge 的最终 incumbent。
            self.state["method_results"] = build_method_results(day4, day6)
            comparison_images = _export_comparison_images(
                self.run_dir,
                day4,
                self.state["method_results"],
            )
            self.state["artifacts"]["comparison_images"] = comparison_images
            print("\n" + "=" * 72, flush=True)
            print("📌 阶段 4/4：汇总结果并生成报告", flush=True)
            print("=" * 72, flush=True)
            print(
                "🖼️  对比效果图已整理到: %s"
                % (self.run_dir / "comparison_images"),
                flush=True,
            )
            eligible = [
                item
                for item in self.state["method_results"]
                if item["eligible_for_final_output"]
            ]
            if self.config.evolution_cases:
                # Recovery evolves one final algorithm per modality. A strong
                # development-only interpolation/SIREN score must not replace it.
                role = "candidate" if day6 and day6.get("best_evolved") else "tensor_baseline"
                winner = next(item for item in eligible if item["role"] == role)
            else:
                winner = max(eligible, key=lambda item: _score(item["metrics"]))
            best_data_path = self.run_dir / "best_completion.npy"
            best_mat_path = self.run_dir / "best_completion.mat"
            best_preview_path = self.run_dir / "best_completion.png"
            shutil.copy2(winner["reconstruction"], best_data_path)
            shutil.copy2(winner["preview"], best_preview_path)
            winner_mat = winner.get("reconstruction_mat")
            if winner_mat:
                shutil.copy2(winner_mat, best_mat_path)
            self.state["best_available"] = {
                **winner,
                "source_reconstruction": winner["reconstruction"],
                "source_preview": winner["preview"],
                "reconstruction": str(best_data_path),
                "preview": str(best_preview_path),
            }
            if self.config.evolution_cases and day6:
                self.state["best_available"]["whole_modality_summary"] = (
                    day6["overall_comparison"].get("whole_modality_summary")
                )
            if winner_mat:
                self.state["best_available"]["reconstruction_mat"] = str(
                    best_mat_path
                )
            self.state["candidate_accepted"] = bool(day6 and day6["accepted"])
            self.state["artifacts"]["best_completion"] = str(best_preview_path)
            self.state["artifacts"]["best_completion_data"] = str(best_data_path)
            if winner_mat:
                self.state["artifacts"]["best_completion_mat"] = str(best_mat_path)
            self.state["stage"] = "COMPLETED"
            self._save()
            write_research_report(
                str(self.report_path), self.state, day4, day5, day6
            )
            self.trace.log_event(
                "session_end",
                {
                    "run_id": self.run_id,
                    "stage": "COMPLETED",
                    "candidate_accepted": self.state["candidate_accepted"],
                    "best_algorithm": winner["algorithm"],
                },
            )
            return self.state
        except Exception as error:
            self.state["stage"] = "FAILED"
            self.state["last_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            self._save()
            self.trace.log_event(
                "error", {"error_type": type(error).__name__, "message": str(error)}
            )
            raise
        finally:
            self.trace.finalize()


def run_full_workflow(config: FullWorkflowConfig) -> Dict[str, Any]:
    with audio_metric_context(audio_metadata_for_gt(config.image_path)):
        return FullResearchWorkflow(config).run()
