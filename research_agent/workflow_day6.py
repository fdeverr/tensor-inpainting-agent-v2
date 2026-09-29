"""Day 6 fair evaluation, bounded improvement loop, and algorithm promotion."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .ablation import (
    build_ablation_arms,
    build_final_removal_audit,
    build_pruning_audit,
    classify_ablation,
)
from .agent_tools.framework import TraceLogger
from .baseline_diagnostics import summarize_baseline_history
from .candidate import (
    CandidateGenerator,
    CandidateValidator,
    candidate_builder,
    load_validated_candidate,
    promote_candidate,
)
from .candidate.generator import load_improver_context
from .candidate.search_contract import (
    declared_search_space_from_manifest,
    evaluate_search_space_contract,
    validate_search_space_contract,
)
from .core.data import load_observation_mask, load_tensor_data
from .core.experiment_judge import judge_candidate
from .core.fair_experiment import (
    final_fit_and_evaluate,
    independent_trial_configurations,
    tune_model_on_observed_pixels,
)
from .core.models.registry import MODEL_CLASSES
from .evolution_knowledge import (
    GlobalExperienceStore,
    RunPracticeStore,
    describe_metrics,
    resolve_knowledge_root,
)
from .method_selector import llm_from_environment
from .schemas import TrainingConfig
from .visual_evaluator import (
    MultimodalQualityEvaluator,
    visual_llm_from_environment,
)
from .workflow_day5 import _identifier, _write_json


@dataclass(frozen=True)
class Day6WorkflowConfig:
    base_run_dir: str
    initial_candidate_dir: str
    candidate_root: str = "research_agent/algorithms/candidates"
    approved_root: str = "research_agent/algorithms/approved"
    output_dir: str = "research_agent/outputs"
    knowledge_root: Optional[str] = None
    llm_mode: str = "auto"
    tuning_trials: int = 4
    learning_rate_candidates: tuple[float, ...] = (0.001, 0.01, 0.1)
    refine_learning_rate: bool = True
    learning_rate_refinement_factor: float = 3.0
    max_steps: int = 3000
    max_improvement_rounds: int = 5
    validation_interval: int = 10
    patience: int = 20
    device: str = "auto"
    minimum_psnr_delta: float = 0.2
    ssim_tolerance: float = 0.002
    smoke_timeout_seconds: float = 10.0
    full_reference_metrics: bool = True
    no_reference_metrics: bool = False
    visual_assessment: bool = False
    ablation_screen_trials: int = 1
    ablation_screen_max_steps: int = 300

    def validate(self) -> None:
        if not (Path(self.base_run_dir) / "state.json").is_file():
            raise ValueError("base_run_dir must contain state.json")
        if not (Path(self.initial_candidate_dir) / "manifest.json").is_file():
            raise ValueError("initial_candidate_dir must contain manifest.json")
        if self.llm_mode not in {"auto", "off", "required"}:
            raise ValueError("llm_mode must be auto, off, or required")
        if not 1 <= self.tuning_trials <= 5:
            raise ValueError("tuning_trials must be in [1, 5]")
        if not self.learning_rate_candidates or any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not 1e-5 <= float(rate) <= 1.0
            for rate in self.learning_rate_candidates
        ):
            raise ValueError(
                "learning_rate_candidates must contain numbers in [1e-5, 1.0]"
            )
        if not isinstance(self.refine_learning_rate, bool):
            raise ValueError("refine_learning_rate must be a bool")
        if self.learning_rate_refinement_factor <= 1.0:
            raise ValueError(
                "learning_rate_refinement_factor must be greater than 1"
            )
        if self.refine_learning_rate and any(
            float(rate) / self.learning_rate_refinement_factor < 1e-5
            or float(rate) * self.learning_rate_refinement_factor > 1.0
            for rate in self.learning_rate_candidates
        ):
            raise ValueError(
                "learning_rate_candidates must remain in [1e-5, 1.0] after "
                "local refinement"
            )
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 1 <= self.max_improvement_rounds <= 100:
            raise ValueError("max_improvement_rounds must be in [1, 100]")
        if self.validation_interval < 1 or self.patience < 1:
            raise ValueError("validation_interval and patience must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not isinstance(self.full_reference_metrics, bool):
            raise ValueError("full_reference_metrics must be a bool")
        if not isinstance(self.no_reference_metrics, bool):
            raise ValueError("no_reference_metrics must be a bool")
        if not isinstance(self.visual_assessment, bool):
            raise ValueError("visual_assessment must be a bool")
        if not 1 <= self.ablation_screen_trials <= 3:
            raise ValueError("ablation_screen_trials must be in [1, 3]")
        if self.ablation_screen_max_steps < 1:
            raise ValueError("ablation_screen_max_steps must be positive")
        if self.minimum_psnr_delta < 0.0 or self.ssim_tolerance < 0.0:
            raise ValueError("judge thresholds must be non-negative")


class Day6Workflow:
    """Evaluate candidates under a fixed per-image GT-selection protocol."""

    def __init__(
        self,
        config: Day6WorkflowConfig,
        generator: Optional[CandidateGenerator] = None,
        validator: Optional[CandidateValidator] = None,
        visual_evaluator: Optional[MultimodalQualityEvaluator] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.workflow_id = _identifier("day6")
        self.run_dir = Path(config.output_dir) / self.workflow_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.state_path = self.run_dir / "state.json"
        if generator is None:
            configured_llm = llm_from_environment(config.llm_mode)
            self.generator = CandidateGenerator(
                configured_llm,
                require_valid_llm_output=config.llm_mode == "required",
            )
        else:
            self.generator = generator
            configured_llm = getattr(generator, "llm", None)
        self.visual_evaluator = visual_evaluator or MultimodalQualityEvaluator(
            visual_llm_from_environment(configured_llm)
            if config.visual_assessment
            else None
        )
        self.validator = validator or CandidateValidator(config.smoke_timeout_seconds)
        self.trace = TraceLogger(output_dir=str(self.run_dir / "traces"), sanitize=True)
        self.base_state = json.loads(
            (Path(config.base_run_dir) / "state.json").read_text(encoding="utf-8")
        )
        if self.base_state.get("stage") != "COMPLETED":
            raise ValueError("base run must be COMPLETED")
        self.state: Dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "stage": "CREATED",
            "config": asdict(config),
            "base_run_id": self.base_state["run_id"],
            "rounds": [],
            "feedback_history": [],
            "candidate_generation_attempts": [],
            "search_space_contract_migrations": [],
            "accepted": False,
            "accepted_rounds": [],
            "stop_reason": None,
            "best_available": None,
            "promotion": None,
            "overall_comparison": None,
            "artifacts": {
                "run_dir": str(self.run_dir),
                "state": str(self.state_path),
                "trace_jsonl": str(self.trace.jsonl_path),
                "trace_html": str(self.trace.html_path),
            },
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, self.state)

    def _resolve_training_budget(
        self,
        manifest: Dict[str, Any],
        learning_rate: float,
    ) -> tuple[TrainingConfig, Dict[str, Any]]:
        """Clamp the LLM request and apply one identical budget to both models."""

        requested = manifest.get("training_budget") or {}

        def positive_int(name: str, fallback: int, upper_bound: int) -> int:
            value = requested.get(name, fallback)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                value = fallback
            return min(int(value), upper_bound)

        requested_steps = positive_int("max_steps", self.config.max_steps, 20_000)
        minimum_steps = min(100, self.config.max_steps)
        effective_steps = min(
            max(requested_steps, minimum_steps),
            self.config.max_steps,
        )
        validation_interval = min(
            positive_int(
                "validation_interval",
                self.config.validation_interval,
                500,
            ),
            effective_steps,
        )
        patience = positive_int(
            "early_stopping_patience",
            self.config.patience,
            500,
        )
        training = TrainingConfig(
            learning_rate=learning_rate,
            max_steps=effective_steps,
            validation_interval=validation_interval,
            early_stopping_patience=patience,
            device=self.config.device,
        )
        audit = {
            "requested_by_llm": {
                "max_steps": requested_steps,
                "validation_interval": requested.get(
                    "validation_interval", self.config.validation_interval
                ),
                "early_stopping_patience": requested.get(
                    "early_stopping_patience", self.config.patience
                ),
                "rationale": requested.get("rationale"),
            },
            "user_max_steps_ceiling": self.config.max_steps,
            "program_minimum_steps": minimum_steps,
            "effective_shared_budget": training.to_dict(),
            "max_steps_was_clamped": requested_steps != effective_steps,
            "fairness_rule": (
                "baseline and candidate use the same resolved max steps, validation "
                "schedule, patience, learning-rate search grid, seed, device, and "
                "observed split; each model selects its own best learning rate"
            ),
            "learning_rate_search": {
                "coarse_candidates": list(self.config.learning_rate_candidates),
                "refinement_enabled": self.config.refine_learning_rate,
                "refinement_factor": self.config.learning_rate_refinement_factor,
            },
        }
        return training, audit

    def _screen_combination_ablation(
        self,
        *,
        candidate_manifest: Dict[str, Any],
        candidate_class: type,
        candidate_search_space: Dict[str, Any],
        incumbent: Dict[str, Any],
        incumbent_search_space: Dict[str, Any],
        observed: Any,
        observed_mask: Any,
        ground_truth: Any,
        training: TrainingConfig,
        seed: int,
        round_dir: Path,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Run a small GT-scored full-factorial screen and return the winning arm space."""

        components = candidate_manifest.get("components") or []
        arms = build_ablation_arms(components)
        screen_steps = min(training.max_steps, self.config.ablation_screen_max_steps)
        screen_training = replace(
            training,
            max_steps=screen_steps,
            validation_interval=min(training.validation_interval, screen_steps),
            early_stopping_patience=min(training.early_stopping_patience, 10),
        )
        arm_records = []
        candidate_builder_fn = candidate_builder(candidate_class)
        base_arm_space = dict(candidate_search_space)
        for switch_name in arms[0]["switch_values"]:
            base_arm_space[switch_name] = [False]
        common_plan = independent_trial_configurations(
            base_search_space=incumbent_search_space,
            candidate_search_space=base_arm_space,
            trial_count=self.config.ablation_screen_trials,
            seed=seed,
            anchor_base_config=incumbent["anchor"],
        )
        common_configurations = common_plan["candidate"]
        for arm in arms:
            arm_configurations = []
            for common_configuration in common_configurations:
                arm_configuration = dict(common_configuration)
                arm_configuration.update(arm["switch_values"])
                arm_configurations.append(arm_configuration)
            tuning = tune_model_on_observed_pixels(
                model_name="%s-ablation-%s"
                % (candidate_manifest["candidate_id"], arm["name"]),
                model_builder=candidate_builder_fn,
                configurations=arm_configurations,
                observed_image=observed,
                observed_mask=observed_mask,
                ground_truth=ground_truth,
                training_config=screen_training,
                seed=seed,
                learning_rate_candidates=[training.learning_rate],
                refine_learning_rate=False,
            )
            record = {
                **arm,
                "best_validation_mse": tuning["best"]["best_validation_mse"],
                "best_hyperparameters": tuning["best"]["hyperparameters"],
                "best_step": tuning["best"]["best_step"],
                "parameter_count": tuning["best"].get("parameter_count"),
                "parameter_count_by_configuration": [
                    {
                        "hyperparameters": {
                            key: value
                            for key, value in trial["hyperparameters"].items()
                            if key not in arm["switch_values"]
                        },
                        "parameter_count": trial.get("parameter_count"),
                    }
                    for trial in tuning.get("trials", [])
                ],
                "stopped_early": tuning["best"].get("stopped_early"),
            }
            arm_records.append(record)
            _write_json(
                round_dir / "ablation" / (arm["name"].replace("+", "_") + ".json"),
                {"arm": record, "tuning": tuning},
            )

        non_base = [record for record in arm_records if record["name"] != "Base"]
        selected = min(non_base, key=lambda item: item["best_validation_mse"])
        selected_space = dict(candidate_search_space)
        for switch_name, enabled in selected["switch_values"].items():
            selected_space[switch_name] = [enabled]
        screening = {
            "protocol": "complete_2_to_the_N_factorial_on_missing_region_ground_truth",
            "isolation_rule": (
                "Every arm uses the same candidate class; Base disables all component "
                "operation switches. The incumbent is evaluated separately by the formal Judge."
            ),
            "ground_truth_used": True,
            "component_count": len(components),
            "arm_count": len(arms),
            "six_proper_ablation_arms_for_three_components": (
                len(components) == 3 and len(arms) == 8
            ),
            "training_config": screen_training.to_dict(),
            "trials_per_arm": self.config.ablation_screen_trials,
            "arms": arm_records,
            "selected_non_base_arm": selected["name"],
            "base_won_screen": min(
                arm_records, key=lambda item: item["best_validation_mse"]
            )["name"] == "Base",
        }
        screening["attribution"] = classify_ablation(screening)
        screening["pruning_audit"] = build_pruning_audit(screening, components)
        _write_json(round_dir / "ablation_screening.json", screening)
        return selected_space, screening

    @staticmethod
    def _validate_candidate_search_contract(
        base_search_space: Dict[str, Any],
        candidate_search_space: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> None:
        """Ensure executable code did not widen the proposal's declared space."""

        validate_search_space_contract(
            base_search_space,
            candidate_search_space,
            declared_search_space_from_manifest(manifest),
        )

    def _make_next_candidate(
        self, context: Dict[str, Any], round_index: int
    ) -> Optional[str]:
        """Generate, validator-repair, then safely fall back for the next round."""

        generation_context = dict(context)
        has_llm_generator = self.generator.llm is not None
        maximum_validation_rounds = 3 if has_llm_generator else 1

        for validation_round in range(1, maximum_validation_rounds + 1):
            use_safe_fallback = (
                has_llm_generator
                and validation_round == maximum_validation_rounds
            )
            print(
                "\n🧠 下一轮候选生成/修复 %d/%d：%s"
                % (
                    validation_round,
                    maximum_validation_rounds,
                    "安全 fallback"
                    if use_safe_fallback
                    else (
                        "等待 LLM 修复完整代码"
                        if validation_round > 1
                        else "等待 LLM 生成完整代码"
                    ),
                ),
                flush=True,
            )
            try:
                if use_safe_fallback:
                    generation = CandidateGenerator(None).generate(generation_context)
                    generation.fallback_reason = (
                        "two LLM candidates failed validation; "
                        "using deterministic safety fallback"
                    )
                else:
                    generation = self.generator.generate(generation_context)
            except Exception as error:
                failure = {
                    "improvement_round": round_index,
                    "validation_round": validation_round,
                    "stage": "generation",
                    "passed": False,
                    "feedback": ["%s: %s" % (type(error).__name__, str(error))],
                }
                self.state["candidate_generation_attempts"].append(failure)
                self.trace.log_event("candidate_regeneration_failed", failure, step=round_index)
                self._save()
                print("   ❌ 候选生成失败: %s" % error, flush=True)
                generation_context = {
                    **context,
                    "previous_failure_feedback": failure["feedback"],
                }
                continue

            proposal = generation.proposal
            if validation_round > 1 and proposal.generation_mode == "llm":
                proposal.generation_mode = "llm_repaired"
            print(
                "   架构: %s | 建议训练: %d 步 | 超参: %s"
                % (
                    proposal.architecture_family,
                    proposal.training_budget.max_steps,
                    ", ".join(sorted(proposal.search_space)),
                ),
                flush=True,
            )

            candidate_id = _identifier("candidate")
            directory = (
                Path(self.config.candidate_root)
                / context["base_method"]
                / candidate_id
            )
            directory.mkdir(parents=True, exist_ok=False)
            model_path = directory / "model.py"
            idea_path = directory / "idea.json"
            manifest_path = directory / "manifest.json"
            validation_path = directory / "validation.json"
            model_path.write_text(proposal.model_code, encoding="utf-8")
            code_hash = hashlib.sha256(proposal.model_code.encode("utf-8")).hexdigest()
            idea = proposal.model_dump(exclude={"model_code"})
            idea["declared_search_space"] = idea.pop("search_space")
            idea.update(
                {
                    "candidate_id": candidate_id,
                    "improvement_round": round_index,
                    "validation_round": validation_round,
                    "previous_failure_feedback": generation_context.get(
                        "previous_failure_feedback", []
                    ),
                    "prompt_version": generation.prompt_version,
                    "generation_attempts": generation.attempts,
                    "fallback_reason": generation.fallback_reason,
                    "schema_validation_errors": generation.validation_errors,
                    "raw_outputs": generation.raw_outputs,
                }
            )
            _write_json(idea_path, idea)
            manifest = {
                "candidate_id": candidate_id,
                "base_run_id": context["base_run_id"],
                "base_method": proposal.base_method,
                "architecture_family": proposal.architecture_family,
                "mutation_goal": proposal.mutation_goal,
                "pain_point": proposal.pain_point,
                "core_difficulty": proposal.core_difficulty,
                "simplified_problem": proposal.simplified_problem,
                "mutation_mode": proposal.mutation_mode,
                "mutation_target": proposal.mutation_target,
                "idea": proposal.idea,
                "single_change": proposal.single_change,
                "components": [
                    component.model_dump() for component in proposal.components
                ],
                "interaction_hypothesis": proposal.interaction_hypothesis,
                "parent_candidate_id": (
                    (context.get("incumbent_candidate") or {}).get("idea", {}).get(
                        "candidate_id"
                    )
                ),
                "created_at": datetime.now().isoformat(),
                "improvement_round": round_index,
                "validation_round": validation_round,
                "generation_mode": proposal.generation_mode,
                "llm_model": getattr(self.generator.llm, "model", None),
                "prompt_version": generation.prompt_version,
                "code_sha256": code_hash,
                "validation_status": "pending",
                "declared_search_space": proposal.search_space,
                "executable_search_space": None,
                "effective_search_space": None,
                "training_budget": proposal.training_budget.model_dump(),
                "eligible_for_training": False,
            }
            _write_json(manifest_path, manifest)
            print("   正在执行 Schema / AST / forward / backward 验证…", flush=True)
            validation = self.validator.validate(str(model_path), str(validation_path))
            manifest.update(
                {
                    "validation_status": validation["status"],
                    "eligible_for_training": bool(validation["passed"]),
                    "validated_at": datetime.now().isoformat(),
                }
            )
            _write_json(manifest_path, manifest)
            if validation["passed"]:
                try:
                    candidate_class, _ = load_validated_candidate(str(directory))
                    image_shape = tuple(
                        int(value)
                        for value in context["image_profile"]["image_shape"]
                    )
                    contract = evaluate_search_space_contract(
                        MODEL_CLASSES[proposal.base_method].search_space(image_shape),
                        candidate_class.search_space(image_shape),
                        manifest["declared_search_space"],
                    )
                    manifest.update(
                        {
                            "declared_search_space": contract[
                                "declared_search_space"
                            ],
                            "executable_search_space": contract[
                                "executable_search_space"
                            ],
                            "effective_search_space": contract[
                                "effective_search_space"
                            ],
                            "search_space_contract": {
                                "passed": contract["passed"],
                                "feedback": contract["feedback"],
                                "checked_at": datetime.now().isoformat(),
                            },
                        }
                    )
                    idea.update(
                        {
                            "declared_search_space": contract[
                                "declared_search_space"
                            ],
                            "executable_search_space": contract[
                                "executable_search_space"
                            ],
                            "effective_search_space": contract[
                                "effective_search_space"
                            ],
                            "search_space_contract": manifest[
                                "search_space_contract"
                            ],
                        }
                    )
                    _write_json(idea_path, idea)
                    _write_json(manifest_path, manifest)
                    validation["search_space_contract"] = {
                        **contract,
                        "image_shape": list(image_shape),
                    }
                    _write_json(validation_path, validation)
                    if not contract["passed"]:
                        raise ValueError(contract["feedback"])
                except Exception as error:
                    feedback = "%s: %s" % (type(error).__name__, str(error))
                    validation["passed"] = False
                    validation["status"] = "rejected"
                    validation.setdefault("feedback", []).append(feedback)
                    validation.setdefault(
                        "search_space_contract",
                        {"passed": False, "feedback": feedback},
                    )
                    manifest["validation_status"] = "rejected"
                    manifest["eligible_for_training"] = False
                    _write_json(validation_path, validation)
                    _write_json(manifest_path, manifest)
            attempt_record = {
                "improvement_round": round_index,
                "validation_round": validation_round,
                "candidate_id": candidate_id,
                "candidate_dir": str(directory),
                "stage": "validation",
                "generation_mode": proposal.generation_mode,
                "fallback_reason": generation.fallback_reason,
                "passed": bool(validation["passed"]),
                "feedback": validation["feedback"],
                "search_space_contract_passed": bool(
                    (validation.get("search_space_contract") or {}).get("passed")
                ),
            }
            self.state["candidate_generation_attempts"].append(attempt_record)
            self.trace.log_event(
                "candidate_regenerated",
                {
                    **attempt_record,
                    "architecture_family": proposal.architecture_family,
                    "training_budget": proposal.training_budget.model_dump(),
                    "experiment_feedback_count": len(
                        context.get("previous_failure_feedback", [])
                    ),
                },
                step=round_index,
            )
            self._save()
            if validation["passed"]:
                print("   ✅ 下一轮候选验证通过", flush=True)
                return str(directory)

            print("   ❌ 下一轮候选验证失败", flush=True)
            for feedback_item in validation["feedback"]:
                print("      - %s" % feedback_item, flush=True)
            generation_context = {
                **context,
                "previous_failure_feedback": validation["feedback"],
                "previous_candidate": proposal.model_dump(),
            }

        return None

    def run(self) -> Dict[str, Any]:
        self.trace.log_event(
            "session_start",
            {
                "workflow": "day6_fair_evaluation_and_bounded_improvement",
                "workflow_id": self.workflow_id,
                "maximum_rounds": self.config.max_improvement_rounds,
            },
        )
        try:
            artifacts = self.base_state["artifacts"]
            observed = load_tensor_data(artifacts["corrupted"], max_size=None)
            observed_mask = load_observation_mask(artifacts["mask"])
            ground_truth_path = Path(self.config.base_run_dir) / "evaluation_ground_truth.npy"
            ground_truth = load_tensor_data(str(ground_truth_path), max_size=None)
            if observed.shape != ground_truth.shape or observed_mask.shape != observed.shape[:2]:
                raise ValueError("base-run tensor artifacts have inconsistent shapes")

            base_method = self.base_state["selected_model"]
            learning_rate = float(self.config.learning_rate_candidates[0])
            seed = int(self.base_state["config"]["seed"])
            improver_context = load_improver_context(self.config.base_run_dir)
            improver_context["mutation_visual_assessment_enabled"] = bool(
                self.config.visual_assessment
            )
            comparison_reference_path = (
                self.run_dir / "algorithm_comparison_reference.json"
            )
            _write_json(
                comparison_reference_path,
                improver_context.get("algorithm_comparison_reference", {}),
            )
            self.state["algorithm_comparison_reference"] = improver_context.get(
                "algorithm_comparison_reference", {}
            )
            self.state["artifacts"]["algorithm_comparison_reference"] = str(
                comparison_reference_path
            )
            global_experience = GlobalExperienceStore(
                resolve_knowledge_root(
                    self.config.candidate_root, self.config.knowledge_root
                ),
                base_method,
            )
            run_practice = RunPracticeStore(
                self.run_dir,
                base_method,
                self.workflow_id,
            )
            self.state["artifacts"]["global_experience"] = (
                global_experience.context()["documents"]
            )
            self.state["artifacts"]["run_practice"] = (
                run_practice.context()["documents"]
            )
            candidate_dir = self.config.initial_candidate_dir
            incumbent: Dict[str, Any] = {
                "name": base_method,
                "class": MODEL_CLASSES[base_method],
                "builder": None,
                "candidate_dir": None,
                "anchor": self.base_state["results"]["selected_trial"]["hyperparameters"],
                "final": None,
                "tuning": None,
                "acceptance_judgment": None,
                "training_budget": None,
                "search_space": None,
                "ablation_screening": None,
                "trained_in_round": None,
            }
            original_final: Optional[Dict[str, Any]] = None
            run_wide_training: Optional[TrainingConfig] = None
            run_wide_budget_audit: Optional[Dict[str, Any]] = None

            for round_index in range(1, self.config.max_improvement_rounds + 1):
                candidate_ready = False
                while not candidate_ready:
                    candidate_class, candidate_manifest = load_validated_candidate(
                        candidate_dir
                    )
                    idea_path = Path(candidate_dir) / "idea.json"
                    idea = json.loads(idea_path.read_text(encoding="utf-8"))
                    incumbent_search_space = incumbent.get("search_space") or incumbent[
                        "class"
                    ].search_space(tuple(observed.shape))
                    candidate_search_space = candidate_class.search_space(
                        tuple(observed.shape)
                    )
                    contract = evaluate_search_space_contract(
                        MODEL_CLASSES[base_method].search_space(
                            tuple(observed.shape)
                        ),
                        candidate_search_space,
                        declared_search_space_from_manifest(candidate_manifest),
                    )
                    if contract["passed"]:
                        candidate_ready = True
                        break

                    feedback = contract["feedback"]
                    candidate_manifest.update(
                        {
                            "validation_status": "rejected",
                            "eligible_for_training": False,
                            "declared_search_space": contract[
                                "declared_search_space"
                            ],
                            "executable_search_space": contract[
                                "executable_search_space"
                            ],
                            "effective_search_space": None,
                            "search_space_contract": {
                                "passed": False,
                                "feedback": feedback,
                                "checked_at": datetime.now().isoformat(),
                            },
                        }
                    )
                    _write_json(
                        Path(candidate_dir) / "manifest.json", candidate_manifest
                    )
                    validation_path = Path(candidate_dir) / "validation.json"
                    if validation_path.is_file():
                        validation_report = json.loads(
                            validation_path.read_text(encoding="utf-8")
                        )
                    else:
                        validation_report = {"feedback": []}
                    validation_report["passed"] = False
                    validation_report["status"] = "rejected"
                    validation_report.setdefault("feedback", []).append(feedback)
                    validation_report["search_space_contract"] = {
                        **contract,
                        "image_shape": list(observed.shape),
                    }
                    _write_json(validation_path, validation_report)
                    rejection_record = {
                        "improvement_round": round_index,
                        "candidate_id": candidate_manifest["candidate_id"],
                        "candidate_dir": str(candidate_dir),
                        "stage": "search_space_contract_preflight",
                        "passed": False,
                        "feedback": [feedback],
                    }
                    self.state["candidate_generation_attempts"].append(
                        rejection_record
                    )
                    self.trace.log_event(
                        "candidate_search_space_contract_rejected",
                        rejection_record,
                        step=round_index,
                    )
                    self._save()
                    print(
                        "❌ 候选声明搜索空间与代码实际搜索空间不一致；"
                        "正在把差异反馈给 LLM 重新生成…",
                        flush=True,
                    )
                    previous_candidate = dict(idea)
                    previous_candidate["search_space"] = contract[
                        "declared_search_space"
                    ]
                    previous_candidate["model_code"] = (
                        Path(candidate_dir) / "model.py"
                    ).read_text(encoding="utf-8")
                    regeneration_context = {
                        **improver_context,
                        "previous_failure_feedback": [feedback],
                        "previous_candidate": previous_candidate,
                    }
                    replacement_dir = self._make_next_candidate(
                        regeneration_context,
                        round_index,
                    )
                    if replacement_dir is None:
                        self.state["stop_reason"] = (
                            "candidate_search_space_contract_repair_failed"
                        )
                        self._save()
                        break
                    candidate_dir = replacement_dir

                if not candidate_ready:
                    break
                print("\n" + "-" * 72, flush=True)
                print(
                    "🔄 进化轮次 %d/%d | 当前最优: %s | 候选: %s"
                    % (
                        round_index,
                        self.config.max_improvement_rounds,
                        incumbent["name"],
                        candidate_manifest["candidate_id"],
                    ),
                    flush=True,
                )
                print("-" * 72, flush=True)
                requested_training, requested_budget_audit = self._resolve_training_budget(
                    candidate_manifest, learning_rate
                )
                if run_wide_training is None:
                    run_wide_training = requested_training
                    run_wide_budget_audit = copy.deepcopy(requested_budget_audit)
                    run_wide_budget_audit["established_in_round"] = round_index
                    run_wide_budget_audit["reuse_policy"] = (
                        "The first fair round establishes one run-wide training protocol. "
                        "Later candidates use it unchanged so the trained incumbent can be reused."
                    )
                training = run_wide_training
                training_budget_audit = copy.deepcopy(run_wide_budget_audit)
                training_budget_audit["current_round"] = round_index
                training_budget_audit["current_candidate_request"] = (
                    requested_budget_audit["requested_by_llm"]
                )
                training_budget_audit["current_candidate_request_was_clamped"] = (
                    requested_budget_audit["max_steps_was_clamped"]
                )
                training_budget_audit["current_candidate_request_applied"] = (
                    round_index == training_budget_audit["established_in_round"]
                )
                training_budget_audit["effective_shared_budget"] = training.to_dict()
                legacy_manifest = "allowed_search_space" in candidate_manifest
                stored_spaces_changed = any(
                    candidate_manifest.get(field) != contract[field]
                    for field in (
                        "declared_search_space",
                        "executable_search_space",
                        "effective_search_space",
                    )
                )
                if legacy_manifest or stored_spaces_changed:
                    migration_record = {
                        "candidate_id": candidate_manifest["candidate_id"],
                        "improvement_round": round_index,
                        "declaration_mismatch": False,
                        "legacy_manifest_migrated": legacy_manifest,
                        "strategy": "migrate_matching_contract_metadata",
                        "reason": None,
                        "updated_at": datetime.now().isoformat(),
                    }
                    candidate_manifest.update(
                        {
                            "declared_search_space": contract[
                                "declared_search_space"
                            ],
                            "executable_search_space": contract[
                                "executable_search_space"
                            ],
                            "effective_search_space": contract[
                                "effective_search_space"
                            ],
                            "search_space_contract": {
                                "passed": True,
                                "feedback": None,
                                "checked_at": datetime.now().isoformat(),
                                "migration": migration_record,
                            },
                        }
                    )
                    candidate_manifest.pop("allowed_search_space", None)
                    candidate_manifest.pop("original_allowed_search_space", None)
                    idea.pop("search_space", None)
                    idea["declared_search_space"] = contract[
                        "declared_search_space"
                    ]
                    idea["executable_search_space"] = contract[
                        "executable_search_space"
                    ]
                    idea["effective_search_space"] = contract[
                        "effective_search_space"
                    ]
                    idea["search_space_contract"] = candidate_manifest[
                        "search_space_contract"
                    ]
                    _write_json(Path(candidate_dir) / "manifest.json", candidate_manifest)
                    _write_json(idea_path, idea)
                    validation_path = Path(candidate_dir) / "validation.json"
                    if validation_path.is_file():
                        validation_report = json.loads(
                            validation_path.read_text(encoding="utf-8")
                        )
                        validation_report["search_space_contract"] = {
                            **contract,
                            "image_shape": list(observed.shape),
                        }
                        _write_json(validation_path, validation_report)
                    self.state.setdefault("search_space_contract_migrations", []).append(
                        migration_record
                    )
                    self.trace.log_event(
                        "candidate_search_space_manifest_migrated",
                        migration_record,
                        step=round_index,
                    )
                    self._save()
                    print(
                        "🔧 候选 manifest 已迁移为 declared/executable/"
                        "effective 三层搜索空间；"
                        "继续验证原研究假设",
                        flush=True,
                    )
                candidate_search_space = contract["effective_search_space"]
                round_dir = self.run_dir / ("round_%d" % round_index)
                ablation_screening = None
                if candidate_manifest.get("mutation_mode") == "combination":
                    print(
                        "🧪 正在进行完整组合消融筛选：Base、单组件、两两组合和完整组合…",
                        flush=True,
                    )
                    candidate_search_space, ablation_screening = (
                        self._screen_combination_ablation(
                            candidate_manifest=candidate_manifest,
                            candidate_class=candidate_class,
                            candidate_search_space=candidate_search_space,
                            incumbent=incumbent,
                            incumbent_search_space=incumbent_search_space,
                            observed=observed,
                            observed_mask=observed_mask,
                            ground_truth=ground_truth,
                            training=training,
                            seed=seed,
                            round_dir=round_dir,
                        )
                    )
                    training_budget_audit["ablation_screening"] = {
                        "arm_count": ablation_screening["arm_count"],
                        "selected_non_base_arm": ablation_screening[
                            "selected_non_base_arm"
                        ],
                        "base_won_screen": ablation_screening["base_won_screen"],
                        "ground_truth_used": True,
                    }
                tuning_configurations = independent_trial_configurations(
                    base_search_space=incumbent_search_space,
                    candidate_search_space=candidate_search_space,
                    trial_count=self.config.tuning_trials,
                    seed=seed,
                    anchor_base_config=incumbent["anchor"],
                )
                training_budget_audit["independent_structure_search"] = {
                    key: tuning_configurations[key]
                    for key in (
                        "tuning_policy",
                        "requested_trial_limit_per_model",
                        "baseline_effective_trial_count",
                        "candidate_effective_trial_count",
                        "baseline_trial_count_reduced",
                        "candidate_trial_count_reduced",
                        "base_available_config_count",
                        "candidate_available_config_count",
                        "baseline_search_coverage",
                        "candidate_search_coverage",
                    )
                }
                if (
                    tuning_configurations["baseline_trial_count_reduced"]
                    or tuning_configurations["candidate_trial_count_reduced"]
                ):
                    print(
                        "ℹ️ 独立结构调优（每个模型上限 %d）："
                        "incumbent %d/%d 组，candidate %d/%d 组；"
                        "搜索空间较小的一方穷举完即停，不限制另一方"
                        % (
                            tuning_configurations[
                                "requested_trial_limit_per_model"
                            ],
                            tuning_configurations[
                                "baseline_effective_trial_count"
                            ],
                            tuning_configurations["base_available_config_count"],
                            tuning_configurations[
                                "candidate_effective_trial_count"
                            ],
                            tuning_configurations[
                                "candidate_available_config_count"
                            ],
                        ),
                        flush=True,
                    )
                incumbent_reused = bool(
                    incumbent.get("tuning") is not None
                    and incumbent.get("final") is not None
                )
                if incumbent_reused:
                    baseline_tuning = incumbent["tuning"]
                    baseline_final = incumbent["final"]
                    print(
                        "♻️ 当前最优已按本次运行的固定公平协议训练，直接复用结果",
                        flush=True,
                    )
                else:
                    baseline_tuning = tune_model_on_observed_pixels(
                        model_name=incumbent["name"],
                        model_builder=incumbent["builder"],
                        configurations=tuning_configurations["baseline"],
                        observed_image=observed,
                        observed_mask=observed_mask,
                        ground_truth=ground_truth,
                        selected_output_dir=str(
                            round_dir / "incumbent_selected"
                        ),
                        training_config=training,
                        seed=seed,
                        learning_rate_candidates=list(
                            self.config.learning_rate_candidates
                        ),
                        refine_learning_rate=self.config.refine_learning_rate,
                        learning_rate_refinement_factor=(
                            self.config.learning_rate_refinement_factor
                        ),
                    )
                    baseline_final = None
                candidate_tuning = tune_model_on_observed_pixels(
                    model_name=candidate_manifest["candidate_id"],
                    model_builder=candidate_builder(candidate_class),
                    configurations=tuning_configurations["candidate"],
                    observed_image=observed,
                    observed_mask=observed_mask,
                    ground_truth=ground_truth,
                    selected_output_dir=str(
                        round_dir / "candidate_selected"
                    ),
                    training_config=training,
                    seed=seed,
                    learning_rate_candidates=list(
                        self.config.learning_rate_candidates
                    ),
                    refine_learning_rate=self.config.refine_learning_rate,
                    learning_rate_refinement_factor=(
                        self.config.learning_rate_refinement_factor
                    ),
                )
                _write_json(
                    round_dir / "tuning_configurations.json",
                    tuning_configurations,
                )
                _write_json(round_dir / "training_budget.json", training_budget_audit)
                _write_json(round_dir / "baseline_tuning.json", baseline_tuning)
                _write_json(round_dir / "candidate_tuning.json", candidate_tuning)

                # GT directly selects hyperparameters and checkpoints for this per-image search.
                if baseline_final is None:
                    baseline_final = final_fit_and_evaluate(
                        incumbent["name"], incumbent["builder"], baseline_tuning["best"],
                        observed, observed_mask, ground_truth, training, seed,
                        str(round_dir / "incumbent_final"),
                        self.config.full_reference_metrics,
                        self.config.no_reference_metrics,
                    )
                candidate_final = final_fit_and_evaluate(
                    candidate_manifest["candidate_id"], candidate_builder(candidate_class),
                    candidate_tuning["best"], observed, observed_mask, ground_truth,
                    training, seed, str(round_dir / "candidate_final"),
                    self.config.full_reference_metrics,
                    self.config.no_reference_metrics,
                )
                if original_final is None:
                    original_final = baseline_final
                judgment = judge_candidate(
                    baseline_tuning, baseline_final, candidate_tuning, candidate_final,
                    self.config.minimum_psnr_delta, self.config.ssim_tolerance,
                )
                judgment["budget_audit"].update(
                    {
                        "shared_training_config": training.to_dict(),
                        "user_max_steps_ceiling": self.config.max_steps,
                        "llm_requested_max_steps": training_budget_audit[
                            "current_candidate_request"
                        ]["max_steps"],
                        "max_steps_was_clamped": training_budget_audit[
                            "current_candidate_request_was_clamped"
                        ],
                        "incumbent_training_reused": incumbent_reused,
                        "incumbent_trained_in_round": incumbent.get("trained_in_round"),
                    }
                )
                candidate_components = candidate_manifest.get("components") or []
                selected_component_ids = None
                if ablation_screening is not None:
                    selected_name = ablation_screening["selected_non_base_arm"]
                    selected_component_ids = selected_name.split("+")
                final_removal_audit = build_final_removal_audit(
                    candidate_components,
                    baseline_final,
                    candidate_final,
                    selected_component_ids=selected_component_ids,
                )
                judgment["removal_audit"] = final_removal_audit
                _write_json(round_dir / "judgment.json", judgment)
                incumbent_name_before = incumbent["name"]
                incumbent["final"] = baseline_final
                incumbent["tuning"] = baseline_tuning
                if incumbent.get("trained_in_round") is None:
                    incumbent["trained_in_round"] = round_index
                if incumbent["candidate_dir"]:
                    incumbent["training_budget"] = training_budget_audit
                if judgment["accepted"]:
                    incumbent = {
                        "name": candidate_manifest["candidate_id"],
                        "class": candidate_class,
                        "builder": candidate_builder(candidate_class),
                        "candidate_dir": candidate_dir,
                        "anchor": candidate_tuning["best"]["hyperparameters"],
                        "final": candidate_final,
                        "tuning": candidate_tuning,
                        "acceptance_judgment": judgment,
                        "training_budget": training_budget_audit,
                        "search_space": candidate_search_space,
                        "ablation_screening": ablation_screening,
                        "trained_in_round": round_index,
                    }
                    self.state["accepted"] = True
                    self.state["accepted_rounds"].append(round_index)

                visual_assessment = {"status": "skipped", "reason": "visual assessment disabled by configuration"}
                if self.config.visual_assessment:
                    visual_evaluator = getattr(
                        self, "visual_evaluator", MultimodalQualityEvaluator(None)
                    )
                    visual_assessment = visual_evaluator.evaluate(
                        image_paths={
                            "incumbent": baseline_final["artifacts"]["preview"],
                            "candidate": candidate_final["artifacts"]["preview"],
                        },
                        tensor_context={
                            "data_type": self.base_state["results"]["image_profile"].get(
                                "data_type", "color_image"
                            ),
                            "tensor_shape": list(observed.shape),
                            "preview_note": (
                                "For MSI or video this is an RGB proxy/middle-frame preview. "
                                "Only incumbent and candidate previews are supplied."
                            ),
                        },
                    )
                _write_json(round_dir / "visual_assessment.json", visual_assessment)
                incumbent_diagnostics = summarize_baseline_history(
                    baseline_tuning.get("best", {}).get("history", [])
                )
                mutation_metric_names = {"missing_psnr", "composite_ssim"}
                if judgment["lpips_delta"] is not None:
                    mutation_metric_names.add("lpips")
                incumbent_metric_summary = describe_metrics(
                    baseline_final["metrics"],
                    include_full_psnr=False,
                )
                candidate_metric_summary = describe_metrics(
                    candidate_final["metrics"],
                    include_full_psnr=False,
                )
                metric_deltas = {
                    "missing_psnr_db": judgment["psnr_delta"],
                    "composite_ssim": judgment["ssim_delta"],
                }
                if judgment["lpips_delta"] is not None:
                    metric_deltas["lpips"] = judgment["lpips_delta"]
                result_summary = {
                    "incumbent_metrics": {
                        name: metric
                        for name, metric in incumbent_metric_summary.items()
                        if name in mutation_metric_names
                    },
                    "candidate_metrics": {
                        name: metric
                        for name, metric in candidate_metric_summary.items()
                        if name in mutation_metric_names
                    },
                    "deltas": metric_deltas,
                    "training_behavior": judgment["training_behavior"],
                    "removal_audit": final_removal_audit,
                }
                if ablation_screening is not None:
                    result_summary["ablation"] = ablation_screening
                if self.config.visual_assessment:
                    result_summary["visual_assessment"] = visual_assessment
                conditions = {
                    "image_shape": list(observed.shape),
                    "mask_type": self.base_state["config"]["mask_type"],
                    "actual_missing_rate": self.base_state["results"]["image_profile"][
                        "actual_missing_rate"
                    ],
                    "seed": seed,
                    "training_budget": training_budget_audit["effective_shared_budget"],
                    "lpips_enabled": bool(self.config.full_reference_metrics),
                }
                practice = {
                    "framework": {
                        "name": incumbent_name_before,
                        "base_method": base_method,
                        "conditions": conditions,
                        "training_diagnostics": incumbent_diagnostics,
                    },
                    "goal": idea["mutation_goal"],
                    "method": {
                        "pain_point": idea.get("pain_point"),
                        "core_difficulty": idea.get("core_difficulty"),
                        "simplified_problem": idea.get("simplified_problem"),
                        "idea": idea["idea"],
                        "mode": idea.get("mutation_mode", "atomic"),
                        "target": idea["mutation_target"],
                        "single_change": idea["single_change"],
                        "components": idea.get("components", []),
                        "interaction_hypothesis": idea.get(
                            "interaction_hypothesis"
                        ),
                    },
                    "result": {
                        **result_summary,
                        "decision": judgment["decision"],
                        "accepted": bool(judgment["accepted"]),
                        "incumbent_after": incumbent["name"],
                    },
                }
                extractor = getattr(self.generator, "extract_experience", None)
                experience = (
                    extractor(practice)
                    if extractor is not None
                    else CandidateGenerator(None).extract_experience(practice)
                )
                _write_json(round_dir / "practice_record.json", practice)
                _write_json(round_dir / "experience_record.json", experience)
                practice_artifacts = run_practice.record(practice)
                experience_artifacts = global_experience.record(experience)
                round_record = {
                    "round": round_index,
                    "incumbent_before": incumbent_name_before,
                    "incumbent_after": incumbent["name"],
                    "candidate_id": candidate_manifest["candidate_id"],
                    "candidate_dir": candidate_dir,
                    "mutation_goal": practice["goal"],
                    "idea": practice["method"]["idea"],
                    "mutation_target": practice["method"]["target"],
                    "mutation_mode": practice["method"]["mode"],
                    "single_change": practice["method"]["single_change"],
                    "components": practice["method"]["components"],
                    "architecture_family": candidate_manifest.get(
                        "architecture_family", "tensor_decomposition"
                    ),
                    "training_budget": training_budget_audit,
                    "tuning_configurations": tuning_configurations,
                    "baseline_tuning": baseline_tuning,
                    "candidate_tuning": candidate_tuning,
                    "baseline_final": baseline_final,
                    "candidate_final": candidate_final,
                    "ablation_screening": ablation_screening,
                    "result_summary": result_summary,
                    "experience": experience,
                    "judgment": judgment,
                    "artifacts": {
                        "round_dir": str(round_dir),
                        "judgment": str(round_dir / "judgment.json"),
                        "practice": str(round_dir / "practice_record.json"),
                        "experience": str(round_dir / "experience_record.json"),
                        "visual_assessment": str(
                            round_dir / "visual_assessment.json"
                        ),
                    },
                }
                self.state["rounds"].append(round_record)
                feedback_record = {
                    "round": round_index,
                    "goal": practice["goal"],
                    "idea": practice["method"]["idea"],
                    "decision": judgment["decision"],
                    "psnr_delta": judgment["psnr_delta"],
                    "ssim_delta": judgment["ssim_delta"],
                    "experience": experience,
                    "suspected_causes": judgment["suspected_causes"],
                    "next_round_constraints": judgment["next_round_constraints"],
                }
                if judgment["lpips_delta"] is not None:
                    feedback_record["lpips_delta"] = judgment["lpips_delta"]
                if self.config.visual_assessment:
                    feedback_record["visual_assessment"] = visual_assessment
                self.state["feedback_history"].append(feedback_record)
                self.state["artifacts"]["run_practice"].update(practice_artifacts)
                self.state["artifacts"]["global_experience"].update(
                    experience_artifacts
                )
                self.trace.log_event(
                    "evolution_round_completed",
                    {
                        "round": round_index,
                        "candidate_id": candidate_manifest["candidate_id"],
                        "decision": judgment["decision"],
                        "incumbent_after": incumbent["name"],
                    },
                    step=round_index,
                )
                self._save()

                if round_index < self.config.max_improvement_rounds:
                    print("\n🧠 正在用最新实践、消融归因与经验生成下一轮变异…", flush=True)
                    next_incumbent_tuning = (
                        candidate_tuning if judgment["accepted"] else baseline_tuning
                    )
                    improver_context["training_curve_summary"] = (
                        summarize_baseline_history(
                            next_incumbent_tuning.get("best", {}).get("history", [])
                        )
                    )
                    comparison_reference = copy.deepcopy(
                        improver_context.get("algorithm_comparison_reference", {})
                    )
                    comparison_reference["latest_evolution_round"] = {
                        "round": round_index,
                        "incumbent_before": incumbent_name_before,
                        "candidate": candidate_manifest["candidate_id"],
                        "incumbent_after": incumbent["name"],
                        "decision": judgment["decision"],
                        "incumbent_metrics": result_summary["incumbent_metrics"],
                        "candidate_metrics": result_summary["candidate_metrics"],
                        "deltas": result_summary["deltas"],
                        "comparison_scope": (
                            "Aggregate evaluation feedback for reference-guided evolution; "
                            "no ground-truth tensor values are included."
                        ),
                    }
                    improver_context["algorithm_comparison_reference"] = (
                        comparison_reference
                    )
                    self.state["algorithm_comparison_reference"] = comparison_reference
                    _write_json(comparison_reference_path, comparison_reference)
                    improver_context["evolution_memory"] = {
                        "global_reusable_experience": global_experience.context(),
                        "current_run_practice": run_practice.context(),
                    }
                    improver_context["previous_round_result"] = practice
                    improver_context["previous_failure_feedback"] = list(
                        self.state["feedback_history"]
                    )
                    improver_context["run_wide_fair_training_contract"] = {
                        "training_config": training.to_dict(),
                        "learning_rate_candidates": list(
                            self.config.learning_rate_candidates
                        ),
                        "refine_learning_rate": self.config.refine_learning_rate,
                        "rule": (
                            "This contract is frozen for the run. The incumbent result is "
                            "reused; later training_budget requests are advisory only."
                        ),
                    }
                    if incumbent["candidate_dir"]:
                        incumbent_dir = Path(incumbent["candidate_dir"])
                        improver_context["incumbent_candidate"] = {
                            "idea": json.loads(
                                (incumbent_dir / "idea.json").read_text(encoding="utf-8")
                            ),
                            "model_code": (incumbent_dir / "model.py").read_text(
                                encoding="utf-8"
                            ),
                            "active_hyperparameters": incumbent["anchor"],
                            "active_ablation_arm": (
                                incumbent.get("ablation_screening") or {}
                            ).get("selected_non_base_arm"),
                            "pruning_audit": (
                                incumbent.get("ablation_screening") or {}
                            ).get("pruning_audit"),
                            "formal_removal_audit": (
                                incumbent.get("acceptance_judgment") or {}
                            ).get("removal_audit"),
                        }
                    else:
                        improver_context["incumbent_candidate"] = None
                    next_candidate_dir = self._make_next_candidate(
                        improver_context, round_index + 1
                    )
                    if next_candidate_dir is None:
                        self.state["stop_reason"] = "next_candidate_validation_failed"
                        self._save()
                        break
                    candidate_dir = next_candidate_dir

            if self.state["stop_reason"] is None:
                self.state["stop_reason"] = "maximum_improvement_rounds_reached"
            if original_final is None:
                raise RuntimeError("no evolution round completed")

            evolved_result = None
            if incumbent["candidate_dir"]:
                algorithm_name = "%s_evolved" % base_method
                self.state["promotion"] = promote_candidate(
                    candidate_dir=incumbent["candidate_dir"],
                    approved_root=self.config.approved_root,
                    algorithm_name=algorithm_name,
                    source_run_id=self.workflow_id,
                    best_config={
                        "hyperparameters": incumbent["tuning"]["best"]["hyperparameters"],
                        "learning_rate": incumbent["tuning"]["best"][
                            "learning_rate"
                        ],
                        "selected_steps": incumbent["tuning"]["best"]["best_step"],
                        "training_budget": incumbent["training_budget"][
                            "effective_shared_budget"
                        ],
                    },
                    comparison=incumbent["acceptance_judgment"],
                    conditions={
                        "base_method": base_method,
                        "evolution_rounds": len(self.state["rounds"]),
                        "accepted_rounds": self.state["accepted_rounds"],
                        "image_shape": list(observed.shape),
                        "mask_type": self.base_state["config"]["mask_type"],
                        "actual_missing_rate": self.base_state["results"]["image_profile"][
                            "actual_missing_rate"
                        ],
                        "seed": seed,
                    },
                )
                evolved_result = {
                    "algorithm": algorithm_name,
                    "role": "accepted_candidate",
                    "reconstruction": incumbent["final"]["artifacts"]["reconstruction"],
                    "metrics": incumbent["final"]["metrics"],
                    "final": incumbent["final"],
                }
                self.state["best_evolved"] = evolved_result

            eligible_results = [
                {
                    "algorithm": "nearest_neighbor_manhattan",
                    "role": "interpolation_baseline",
                    "reconstruction": artifacts["interpolation"],
                    "metrics": self.base_state["results"]["interpolation_metrics"],
                },
                {
                    "algorithm": base_method,
                    "role": "tensor_baseline",
                    "reconstruction": original_final["artifacts"]["reconstruction"],
                    "metrics": original_final["metrics"],
                    "final": original_final,
                },
            ]
            siren = self.base_state.get("results", {}).get(
                "siren_comparison", {}
            )
            siren_reconstruction = self.base_state.get("artifacts", {}).get(
                "siren_reconstruction"
            )
            if siren.get("status") == "completed" and siren_reconstruction:
                eligible_results.append(
                    {
                        "algorithm": "siren",
                        "role": "implicit_neural_baseline",
                        "reconstruction": siren_reconstruction,
                        "metrics": siren["metrics"],
                    }
                )
            if evolved_result:
                eligible_results.append(evolved_result)
            winner = max(
                eligible_results,
                key=lambda item: (
                    float("inf")
                    if item["metrics"].get("missing_psnr") is None
                    else item["metrics"]["missing_psnr"]
                ),
            )
            self.state["overall_comparison"] = {
                "selection_rule": (
                    "highest missing-region PSNR among interpolation, SIREN when "
                    "enabled, the original tensor baseline, and the final accepted "
                    "evolved incumbent"
                ),
                "eligible_results": eligible_results,
                "winner": winner["algorithm"],
            }
            self.state["best_available"] = winner
            self.state["stage"] = "COMPLETED"
            self._save()
            self.trace.log_event(
                "session_end",
                {
                    "workflow_id": self.workflow_id,
                    "stage": "COMPLETED",
                    "accepted": self.state["accepted"],
                    "round_count": len(self.state["rounds"]),
                    "stop_reason": self.state["stop_reason"],
                },
            )
            return self.state
        except Exception as error:
            self.state["stage"] = "FAILED"
            self.state["stop_reason"] = "workflow_error"
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


def run_day6_workflow(
    config: Day6WorkflowConfig,
    generator: Optional[CandidateGenerator] = None,
) -> Dict[str, Any]:
    return Day6Workflow(config=config, generator=generator).run()
