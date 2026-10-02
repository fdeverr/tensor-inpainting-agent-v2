"""Day 5 candidate generation and validation workflow."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_tools.framework import TraceLogger
from .candidate import (
    CandidateGenerator,
    CandidateValidator,
    load_validated_candidate,
)
from .candidate.generator import load_improver_context
from .candidate.search_contract import evaluate_search_space_contract
from .method_selector import llm_from_environment
from .core.models.registry import MODEL_CLASSES
from .visual_evaluator import (
    MultimodalQualityEvaluator,
    visual_llm_from_environment,
)


def _identifier(prefix: str) -> str:
    return "%s-%s-%s" % (
        prefix,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        uuid.uuid4().hex[:6],
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class Day5WorkflowConfig:
    base_run_dir: str
    candidate_root: str = "research_agent/algorithms/candidates"
    output_dir: str = "research_agent/outputs"
    knowledge_root: Optional[str] = None
    llm_mode: str = "auto"
    smoke_timeout_seconds: float = 10.0
    visual_assessment: bool = False
    dataset_algorithm_reference: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        state_path = Path(self.base_run_dir) / "state.json"
        if not state_path.is_file():
            raise ValueError("base_run_dir must contain state.json")
        if self.llm_mode not in {"auto", "off", "required"}:
            raise ValueError("llm_mode must be auto, off, or required")
        if self.smoke_timeout_seconds <= 0:
            raise ValueError("smoke_timeout_seconds must be positive")
        if not isinstance(self.visual_assessment, bool):
            raise ValueError("visual_assessment must be a bool")


class Day5Workflow:
    """Produce an auditable candidate and reject it unless both checks pass."""

    def __init__(
        self,
        config: Day5WorkflowConfig,
        generator: Optional[CandidateGenerator] = None,
        validator: Optional[CandidateValidator] = None,
        visual_evaluator: Optional[MultimodalQualityEvaluator] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.workflow_id = _identifier("day5")
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
        self.validator = validator or CandidateValidator(
            timeout_seconds=config.smoke_timeout_seconds
        )
        self.trace = TraceLogger(output_dir=str(self.run_dir / "traces"), sanitize=True)
        self.state: Dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "stage": "CREATED",
            "config": asdict(config),
            "candidate_id": None,
            "base_run_id": None,
            "artifacts": {
                "state": str(self.state_path),
                "trace_jsonl": str(self.trace.jsonl_path),
                "trace_html": str(self.trace.html_path),
            },
            "generation": None,
            "validation": None,
            "attempt_history": [],
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, self.state)

    def run(self) -> Dict[str, Any]:
        self.trace.log_event(
            "session_start",
            {
                "workflow": "day5_candidate_generation_and_validation",
                "workflow_id": self.workflow_id,
                "llm_used": self.generator.llm is not None,
            },
        )
        try:
            print("\n🧠 正在整理基线模型、图像特征和训练曲线供 LLM 分析…", flush=True)
            context = load_improver_context(self.config.base_run_dir)
            if self.config.dataset_algorithm_reference:
                context["algorithm_comparison_reference"]["whole_modality_algorithms"] = (
                    self.config.dataset_algorithm_reference
                )
            context["evolution_memory"] = {
                "current_run_practice": [],
            }
            context["mutation_visual_assessment_enabled"] = bool(
                self.config.visual_assessment
            )
            comparison_reference_path = self.run_dir / "algorithm_comparison_reference.json"
            _write_json(
                comparison_reference_path,
                context.get("algorithm_comparison_reference", {}),
            )
            self.state["algorithm_comparison_reference"] = context.get(
                "algorithm_comparison_reference", {}
            )
            self.state["artifacts"]["algorithm_comparison_reference"] = str(
                comparison_reference_path
            )
            print(
                "\n🧬 首轮变异将参考插值、SIREN 与张量家族的结构化对比结果…",
                flush=True,
            )
            initial_visual_assessment = {
                "status": "skipped",
                "reason": (
                    "structured numerical algorithm references are enabled; the first "
                    "round does not additionally compare interpolation imagery"
                ),
                "comparison_mode": "structured_algorithm_reference_only",
                "ground_truth_provided": False,
            }
            initial_visual_path = self.run_dir / "initial_interpolation_visual.json"
            _write_json(initial_visual_path, initial_visual_assessment)
            self.state["initial_interpolation_visual_assessment"] = (
                initial_visual_assessment
            )
            self.state["artifacts"]["initial_interpolation_visual_assessment"] = str(
                initial_visual_path
            )
            self.state["base_run_id"] = context["base_run_id"]
            self._save()
            self.trace.log_event(
                "initial_interpolation_visual_assessment",
                {
                    "status": initial_visual_assessment.get("status"),
                    "attempts": initial_visual_assessment.get("attempts", 0),
                    "comparison_mode": "structured_algorithm_reference_only",
                    "ground_truth_provided": False,
                },
                step=1,
            )
            self.trace.log_event(
                "model_improver_input",
                {
                    "base_run_id": context["base_run_id"],
                    "base_method": context["base_method"],
                    "profile": context["image_profile"],
                    "method_plan": context["method_plan"],
                    "base_best_config": context["base_best_config"],
                    "training_curve_summary": context["training_curve_summary"],
                    "base_metrics": context["base_metrics"],
                    "algorithm_comparison_reference": context.get(
                        "algorithm_comparison_reference", {}
                    ),
                    "ground_truth_provided": False,
                    "base_source_character_count": len(context["base_model_source"]),
                },
                step=1,
            )
            generation_context = dict(context)
            total_generation_attempts = 0
            has_llm_generator = self.generator.llm is not None
            maximum_validation_rounds = 3 if has_llm_generator else 1
            candidate_id = ""
            validation: Dict[str, Any] = {}

            for validation_round in range(1, maximum_validation_rounds + 1):
                print(
                    "\n🧠 候选生成/修复 %d/%d：%s"
                    % (
                        validation_round,
                        maximum_validation_rounds,
                        (
                            "安全 fallback"
                            if has_llm_generator
                            and validation_round == maximum_validation_rounds
                            else (
                                "等待 LLM 返回完整代码"
                                if has_llm_generator
                                else "使用确定性 fallback"
                            )
                        ),
                    ),
                    flush=True,
                )
                if has_llm_generator and validation_round == maximum_validation_rounds:
                    generation = CandidateGenerator(None).generate(generation_context)
                    generation.fallback_reason = (
                        "two LLM candidates failed validation; "
                        "using deterministic safety fallback"
                    )
                else:
                    generation = self.generator.generate(generation_context)
                total_generation_attempts += generation.attempts
                proposal = generation.proposal
                print(
                    "   架构: %s | 建议训练: %d 步 | 超参: %s"
                    % (
                        proposal.architecture_family,
                        proposal.training_budget.max_steps,
                        ", ".join(sorted(proposal.search_space)),
                    ),
                    flush=True,
                )
                if validation_round > 1 and proposal.generation_mode == "llm":
                    proposal.generation_mode = "llm_repaired"

                candidate_id = _identifier("candidate")
                candidate_dir = (
                    Path(self.config.candidate_root)
                    / context["base_method"]
                    / candidate_id
                )
                candidate_dir.mkdir(parents=True, exist_ok=False)
                idea_path = candidate_dir / "idea.json"
                model_path = candidate_dir / "model.py"
                manifest_path = candidate_dir / "manifest.json"
                validation_path = candidate_dir / "validation.json"
                model_path.write_text(proposal.model_code, encoding="utf-8")
                code_hash = hashlib.sha256(
                    proposal.model_code.encode("utf-8")
                ).hexdigest()
                idea_payload = proposal.model_dump(exclude={"model_code"})
                idea_payload["declared_search_space"] = idea_payload.pop(
                    "search_space"
                )
                idea_payload.update(
                    {
                        "candidate_id": candidate_id,
                        "prompt_version": generation.prompt_version,
                        "validation_round": validation_round,
                        "generation_attempts": generation.attempts,
                        "fallback_reason": generation.fallback_reason,
                        "schema_validation_errors": generation.validation_errors,
                        "raw_outputs": generation.raw_outputs,
                    }
                )
                _write_json(idea_path, idea_payload)
                llm_model = getattr(self.generator.llm, "model", None)
                manifest = {
                    "candidate_id": candidate_id,
                    "base_run_id": context["base_run_id"],
                    "base_method": proposal.base_method,
                    "architecture_family": proposal.architecture_family,
                    "evolution_round": 1,
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
                    "parent_candidate_id": None,
                    "created_at": datetime.now().isoformat(),
                    "generation_mode": proposal.generation_mode,
                    "llm_model": llm_model,
                    "prompt_version": generation.prompt_version,
                    "validation_round": validation_round,
                    "code_sha256": code_hash,
                    "validation_status": "pending",
                    "declared_search_space": proposal.search_space,
                    "executable_search_space": None,
                    "effective_search_space": None,
                    "training_budget": proposal.training_budget.model_dump(),
                    "eligible_for_training": False,
                }
                _write_json(manifest_path, manifest)
                self.state.update(
                    {
                        "stage": "GENERATED",
                        "candidate_id": candidate_id,
                        "generation": {
                            "mode": proposal.generation_mode,
                            "attempts": generation.attempts,
                            "total_attempts": total_generation_attempts,
                            "validation_round": validation_round,
                            "repair_attempted": validation_round > 1,
                            "fallback_reason": generation.fallback_reason,
                            "hypothesis": proposal.hypothesis,
                            "mutation_goal": proposal.mutation_goal,
                            "mutation_target": proposal.mutation_target,
                            "idea": proposal.idea,
                            "single_change": proposal.single_change,
                            "architecture_family": proposal.architecture_family,
                            "training_budget": proposal.training_budget.model_dump(),
                        },
                    }
                )
                self.state["artifacts"].update(
                    {
                        "candidate_dir": str(candidate_dir),
                        "idea": str(idea_path),
                        "model": str(model_path),
                        "manifest": str(manifest_path),
                        "validation": str(validation_path),
                    }
                )
                self._save()
                self.trace.log_event(
                    "candidate_generated",
                    {
                        "candidate_id": candidate_id,
                        "validation_round": validation_round,
                        "base_method": proposal.base_method,
                        "hypothesis": proposal.hypothesis,
                        "proposed_changes": proposal.proposed_changes,
                        "architecture_family": proposal.architecture_family,
                        "mutation_goal": proposal.mutation_goal,
                        "mutation_target": proposal.mutation_target,
                        "idea": proposal.idea,
                        "single_change": proposal.single_change,
                        "training_budget": proposal.training_budget.model_dump(),
                        "search_space": proposal.search_space,
                        "generation_mode": proposal.generation_mode,
                        "code_sha256": code_hash,
                    },
                    step=2 + (validation_round - 1) * 2,
                )

                print("   正在执行 Schema / AST / forward / backward 验证…", flush=True)
                validation = self.validator.validate(
                    model_path=str(model_path),
                    output_path=str(validation_path),
                )
                manifest["validation_status"] = validation["status"]
                manifest["eligible_for_training"] = bool(validation["passed"])
                manifest["validated_at"] = datetime.now().isoformat()
                _write_json(manifest_path, manifest)
                if validation["passed"]:
                    try:
                        candidate_class, _ = load_validated_candidate(
                            str(candidate_dir)
                        )
                        image_shape = tuple(
                            int(value)
                            for value in context["image_profile"]["image_shape"]
                        )
                        contract = evaluate_search_space_contract(
                            MODEL_CLASSES[proposal.base_method].search_space(
                                image_shape
                            ),
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
                        idea_payload.update(
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
                        self.state["generation"]["search_space_contract_passed"] = contract[
                            "passed"
                        ]
                        _write_json(idea_path, idea_payload)
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
                validation_summary = {
                    "passed": validation["passed"],
                    "status": validation["status"],
                    "feedback": validation["feedback"],
                    "runtime_seconds": validation["runtime_seconds"],
                    "eligible_for_training": validation["passed"],
                }
                self.state["attempt_history"].append(
                    {
                        "validation_round": validation_round,
                        "candidate_id": candidate_id,
                        "generation": dict(self.state["generation"]),
                        "validation": validation_summary,
                        "artifacts": {
                            "candidate_dir": str(candidate_dir),
                            "idea": str(idea_path),
                            "model": str(model_path),
                            "manifest": str(manifest_path),
                            "validation": str(validation_path),
                        },
                    }
                )
                self.state["stage"] = (
                    "VALIDATED" if validation["passed"] else "REJECTED"
                )
                self.state["validation"] = validation_summary
                self._save()
                self.trace.log_event(
                    "candidate_validation",
                    {
                        "candidate_id": candidate_id,
                        "validation_round": validation_round,
                        "passed": validation["passed"],
                        "status": validation["status"],
                        "static_validation": validation["static_validation"],
                        "smoke_test": validation["smoke_test"],
                        "feedback": validation["feedback"],
                        "search_space_contract": validation.get(
                            "search_space_contract"
                        ),
                    },
                    step=3 + (validation_round - 1) * 2,
                )
                print(
                    "   %s 候选验证%s"
                    % (
                        "✅" if validation["passed"] else "❌",
                        "通过" if validation["passed"] else "失败",
                    ),
                    flush=True,
                )
                if not validation["passed"]:
                    for feedback_item in validation["feedback"]:
                        print("      - %s" % feedback_item, flush=True)
                if validation["passed"]:
                    break
                if validation_round < maximum_validation_rounds:
                    self.trace.log_event(
                        "candidate_repair_requested",
                        {
                            "rejected_candidate_id": candidate_id,
                            "next_validation_round": validation_round + 1,
                            "validator_feedback": validation["feedback"],
                        },
                        step=4,
                    )
                    generation_context = {
                        **context,
                        "previous_failure_feedback": validation["feedback"],
                        "previous_candidate": proposal.model_dump(),
                    }

            self.trace.log_event(
                "session_end",
                {
                    "workflow_id": self.workflow_id,
                    "stage": self.state["stage"],
                    "candidate_id": candidate_id,
                    "validation_rounds": len(self.state["attempt_history"]),
                },
            )
            return self.state
        except Exception as error:
            self.state["stage"] = "FAILED"
            self.state["validation"] = {
                "passed": False,
                "status": "workflow_error",
                "feedback": [str(error)],
                "eligible_for_training": False,
            }
            self._save()
            self.trace.log_event(
                "error",
                {"error_type": type(error).__name__, "message": str(error)},
            )
            self.trace.log_event(
                "session_end",
                {"workflow_id": self.workflow_id, "stage": "FAILED"},
            )
            raise
        finally:
            self.trace.finalize()


def run_day5_workflow(
    config: Day5WorkflowConfig,
    generator: Optional[CandidateGenerator] = None,
) -> Dict[str, Any]:
    return Day5Workflow(config=config, generator=generator).run()
