import json
from pathlib import Path

from research_agent.candidate.generator import (
    CandidateGenerator,
    CandidateGenerationResult,
    _algorithm_comparison_reference,
    deterministic_candidate,
)
from research_agent.workflow_day5 import Day5Workflow, Day5WorkflowConfig


class RepairingGenerator:
    def __init__(self):
        self.llm = object()
        self.contexts = []

    def generate(self, context):
        self.contexts.append(context)
        proposal = deterministic_candidate("tucker")
        proposal.generation_mode = "llm"
        if len(self.contexts) == 1:
            proposal.model_code = "from typing import Any\n\n" + proposal.model_code
        return CandidateGenerationResult(
            proposal=proposal,
            attempts=1,
            raw_outputs=["test output"],
            validation_errors=[],
            fallback_reason=None,
            prompt_version="test-feedback-repair",
        )


class StaticResponseLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, messages, temperature=0.0):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


class RecordingVisualEvaluator:
    llm = object()

    def __init__(self):
        self.calls = []

    def evaluate(self, image_paths, tensor_context, comparison_mode):
        self.calls.append(
            {
                "image_paths": image_paths,
                "tensor_context": tensor_context,
                "comparison_mode": comparison_mode,
            }
        )
        return {
            "status": "completed",
            "comparison_mode": comparison_mode,
            "comparison_roles": {
                "incumbent": "tensor_decomposition_baseline",
                "candidate": "manhattan_interpolation_reference_not_ground_truth",
            },
            "ground_truth_provided": False,
            "assessment": {
                "comparison": "插值提供更连贯的低频结构，但存在平滑。",
                "mutation_guidance": ["在张量框架内单独增强二维空间连续性。"],
            },
        }


class RejectThenAcceptValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, model_path, output_path=None):
        self.calls += 1
        passed = self.calls == 2
        feedback = [] if passed else ["from typing import ... is not allowed"]
        result = {
            "passed": passed,
            "status": "validated" if passed else "rejected",
            "static_validation": {
                "passed": passed,
                "checks": [],
                "errors": [],
            },
            "smoke_test": {
                "passed": passed,
                "skipped": not passed,
            },
            "feedback": feedback,
            "runtime_seconds": 0.001,
        }
        if output_path is not None:
            Path(output_path).write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
        return result


class AlwaysRejectLLMGenerator(RepairingGenerator):
    pass


class ManifestMismatchGenerator(RepairingGenerator):
    def generate(self, context):
        result = super().generate(context)
        if len(self.contexts) == 1:
            result.proposal.search_space = {"tv_weight": [0.9]}
        return result


class AlwaysAcceptValidator(RejectThenAcceptValidator):
    def validate(self, model_path, output_path=None):
        self.calls += 1
        result = {
            "passed": True,
            "status": "validated",
            "static_validation": {"passed": True, "checks": [], "errors": []},
            "smoke_test": {"passed": True, "skipped": False},
            "feedback": [],
            "runtime_seconds": 0.001,
        }
        if output_path is not None:
            Path(output_path).write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8"
            )
        return result


class AcceptDeterministicFallbackValidator(RejectThenAcceptValidator):
    def validate(self, model_path, output_path=None):
        self.calls += 1
        source = Path(model_path).read_text(encoding="utf-8")
        passed = self.calls == 3 and "tv_regularization" in source
        feedback = [] if passed else ["generated candidate violates the model contract"]
        result = {
            "passed": passed,
            "status": "validated" if passed else "rejected",
            "static_validation": {
                "passed": passed,
                "checks": [],
                "errors": [],
            },
            "smoke_test": {
                "passed": passed,
                "skipped": not passed,
            },
            "feedback": feedback,
            "runtime_seconds": 0.001,
        }
        if output_path is not None:
            Path(output_path).write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
        return result


def _context():
    return {
        "base_run_id": "base-run-test",
        "base_method": "tucker",
        "image_profile": {"image_shape": [16, 16, 3]},
        "method_plan": {"method": "tucker"},
        "base_best_config": {"model_name": "tucker"},
        "training_curve_summary": {"record_count": 1},
        "base_metrics": {"full_psnr": 14.0, "missing_psnr": 10.0},
        "interpolation_metrics": {"full_psnr": 15.0, "missing_psnr": 11.0},
        "comparison": {"winner": "interpolation"},
        "algorithm_comparison_reference": {
            "evaluation_feedback_reused_for_evolution": True,
            "winner": "nearest_neighbor_manhattan",
            "ranked_results": [
                {
                    "algorithm": "nearest_neighbor_manhattan",
                    "role": "interpolation_baseline",
                    "metrics": {
                        "full_psnr": 15.0,
                        "missing_psnr": 11.0,
                        "composite_ssim": 0.8,
                    },
                },
                {
                    "algorithm": "tucker",
                    "role": "selected_tensor_baseline",
                    "metrics": {
                        "full_psnr": 14.0,
                        "missing_psnr": 10.0,
                        "composite_ssim": 0.7,
                    },
                },
            ],
        },
        "base_model_source": "class TuckerDecomposition: pass",
        "base_class_name": "TuckerDecomposition",
        "previous_failure_feedback": [],
    }


def test_algorithm_comparison_reference_exposes_ranked_baselines_and_screening():
    state = {
        "selected_model": "tucker",
        "results": {
            "interpolation_metrics": {
                "missing_psnr": 18.0,
                "full_psnr": 24.0,
                "composite_ssim": 0.82,
            },
            "tensor_metrics": {
                "missing_psnr": 16.5,
                "full_psnr": 23.0,
                "composite_ssim": 0.78,
                "lpips": 0.31,
            },
            "siren_comparison": {
                "status": "completed",
                "metrics": {
                    "missing_psnr": 17.2,
                    "full_psnr": 23.5,
                    "composite_ssim": 0.80,
                    "lpips": 0.25,
                },
            },
            "method_screening": {
                "ground_truth_used": False,
                "selection_metric": "mask_matched_held_out_observed_mse",
                "winner": "tucker",
                "results": [
                    {
                        "method": "tucker",
                        "best_validation_mse": 0.01,
                        "trial_count": 2,
                        "best_trial": {"large": "payload"},
                    },
                    {
                        "method": "hierarchical_tucker",
                        "best_validation_mse": 0.001,
                        "trial_count": 2,
                        "best_trial": {"losing_training_curve": [1, 2, 3]},
                    },
                ],
            },
        },
    }

    reference = _algorithm_comparison_reference(state)

    assert reference["winner"] == "nearest_neighbor_manhattan"
    assert reference["selected_tensor_gap_to_best_missing_psnr_db"] == -1.5
    assert [item["algorithm"] for item in reference["ranked_results"]] == [
        "nearest_neighbor_manhattan",
        "siren",
        "tucker",
    ]
    assert "full_psnr" not in reference["ranked_results"][0]["metrics"]
    assert reference["tensor_family_screening"]["results"] == [
        {
            "method": "tucker",
            "best_validation_mse": 0.01,
            "trial_count": 2,
        }
    ]
    assert reference["evaluation_feedback_reused_for_evolution"] is True
    assert "hierarchical_tucker" not in json.dumps(reference["tensor_family_screening"])


def test_only_screening_winner_contributes_cohort_scores_and_training_digest():
    def result(method, psnr):
        return {"method": method, "best_validation_mse": 0.01,
                "trial_count": 2, "dataset_evaluation": {
                    "summary": {"complete": True, "mean_missing_psnr": psnr},
                    "results": [{"source": "/private/sample.mat", "status": "completed",
                                 "metrics": {"missing_psnr": psnr, "full_psnr": 99.0},
                                 "training": {"curve_summary": {"record_count": 4}}}],
                }}

    state = {"selected_model": "tucker", "results": {
        "tensor_metrics": {"missing_psnr": 22.0},
        "method_screening": {"winner": "tucker", "ground_truth_used": True,
            "selection_metric": "mean_missing_psnr_then_ssim",
            "results": [result("tucker", 22.0), result("cp", 30.0)]},
    }}
    reference = _algorithm_comparison_reference(state)
    screen = reference["tensor_family_screening"]

    assert [row["method"] for row in screen["results"]] == ["tucker"]
    assert screen["results"][0]["whole_modality_summary"]["mean_missing_psnr"] == 22.0
    assert screen["results"][0]["samples"][0]["training"]["curve_summary"]["record_count"] == 4
    assert "full_psnr" not in screen["results"][0]["samples"][0]["metrics"]
    assert "/private/sample.mat" not in json.dumps(screen)


def test_candidate_prompt_matches_the_executable_model_contract():
    context = _context()
    context["mutation_visual_assessment_enabled"] = True
    context["initial_interpolation_visual_assessment"] = {
        "status": "completed",
        "ground_truth_provided": False,
        "assessment": {"mutation_guidance": ["use spatial continuity"]},
    }
    payload = json.loads(CandidateGenerator._messages(context)[1]["content"])
    contract = payload["fixed_contract"]

    assert contract["trainer_editable"] is False
    assert "TuckerDecomposition" in contract["preloaded_symbols"]
    assert "Mode3Factorization" in contract["preloaded_symbols"]
    assert "from typing import ..." in contract["import_rules"][
        "forbidden_examples"
    ]
    assert not any("scheduling" in change for change in payload["allowed_changes"])
    assert any("coordinate MLP" in change for change in payload["allowed_changes"])
    assert any("Transformer" in change for change in payload["allowed_changes"])
    assert payload["training_budget_policy"]["fairness"].startswith(
        "The resolved per-trial budget"
    )
    assert "independently searches" in payload["training_budget_policy"][
        "fairness"
    ]
    assert "architecture_family" in payload["output_schema"]["properties"]
    assert "training_budget" in payload["output_schema"]["properties"]
    assert "full_psnr" not in payload["context"]["base_metrics"]
    assert payload["context"]["base_metrics"]["missing_psnr"] == 10.0
    assert payload["evolution_protocol"]["psnr_feedback_rule"].startswith(
        "Use missing-region PSNR"
    )
    objective_rule = payload["evolution_protocol"]["optimization_objective_rule"]
    assert "current incumbent" in objective_rule
    assert "interpolation, SIREN" in objective_rule
    assert "Pareto front" in objective_rule
    assert payload["response_style"]["maximum_characters"]["idea"] == 1200
    assert "concise" in payload["response_style"]["instruction"]
    assert "initial_interpolation_visual_assessment" not in payload["context"]
    assert "interpolation_metrics" not in payload["context"]
    assert "comparison" not in payload["context"]
    reference = payload["context"]["algorithm_comparison_reference"]
    assert reference["winner"] == "nearest_neighbor_manhattan"
    assert reference["ranked_results"][0]["metrics"]["missing_psnr"] == 11.0
    assert "full_psnr" not in reference["ranked_results"][0]["metrics"]
    assert "directional benchmark" in payload["evolution_protocol"][
        "algorithm_comparison_rule"
    ]
    guidance = payload["architecture_guidance"]
    assert any("Self-attention" in item for item in guidance["attention_guidance"])
    assert any("Cross-attention" in item for item in guidance["attention_guidance"])
    assert any(
        "dilation" in item for item in guidance["dilated_convolution_guidance"]
    )
    assert "incumbent-versus-candidate" in payload["evolution_protocol"][
        "visual_feedback_rule"
    ]


def test_candidate_prompt_keeps_available_lpips_practice_feedback():
    context = _context()
    context["base_metrics"]["lpips"] = 0.31
    context["previous_round_result"] = {
        "result": {
            "incumbent_metrics": {"lpips": {"value": 0.31}},
            "candidate_metrics": {"lpips": {"value": 0.25}},
            "deltas": {"lpips": -0.06},
        }
    }
    context["previous_failure_feedback"] = [
        {"decision": "reject", "lpips_delta": -0.06, "maniqa": 0.7}
    ]

    payload = json.loads(CandidateGenerator._messages(context)[1]["content"])

    assert payload["context"]["base_metrics"]["lpips"] == 0.31
    assert payload["context"]["previous_round_result"]["result"]["deltas"][
        "lpips"
    ] == -0.06
    assert payload["experiment_feedback"][0]["lpips_delta"] == -0.06
    assert "maniqa" not in payload["experiment_feedback"][0]
    objective_rule = payload["evolution_protocol"]["optimization_objective_rule"]
    assert "secondary perceptual diagnostic" in objective_rule
    assert "negative candidate-minus-incumbent delta" in objective_rule


def test_candidate_prompt_omits_visual_feedback_when_mutation_option_is_off():
    context = _context()
    context["mutation_visual_assessment_enabled"] = False
    payload = json.loads(CandidateGenerator._messages(context)[1]["content"])

    assert "visual_feedback_rule" not in payload["evolution_protocol"]
    assert "initial_interpolation_visual_assessment" not in payload["context"]


def test_overlong_explanatory_prose_is_compacted_without_touching_model_code():
    payload = deterministic_candidate("tucker").model_dump()
    original_model_code = payload["model_code"]
    payload["idea"] = "premise " + ("detail " * 300) + "conclusion"
    llm = StaticResponseLLM(payload)

    result = CandidateGenerator(
        llm=llm,
        require_valid_llm_output=True,
    ).generate(_context())

    assert result.attempts == 1
    assert llm.calls == 1
    assert len(result.proposal.idea) <= 1200
    assert "[middle omitted]" in result.proposal.idea
    assert result.proposal.idea.startswith("premise")
    assert result.proposal.idea.endswith("conclusion")
    assert result.proposal.model_code == original_model_code


def test_candidate_repair_prompt_contains_validator_feedback_and_previous_code():
    context = _context()
    context["previous_failure_feedback"] = [
        "from typing import ... is not allowed"
    ]
    context["previous_candidate"] = deterministic_candidate("tucker").model_dump()
    payload = json.loads(CandidateGenerator._messages(context)[1]["content"])

    assert payload["task"].startswith("Repair the previous candidate")
    assert payload["repair_context"]["validator_feedback"] == [
        "from typing import ... is not allowed"
    ]
    assert "model_code" in payload["repair_context"]["previous_candidate"]


def test_experiment_feedback_is_not_mislabeled_as_validator_repair():
    context = _context()
    context["previous_failure_feedback"] = [
        {"decision": "reject", "next_round_constraints": ["reduce TV weight"]}
    ]
    payload = json.loads(CandidateGenerator._messages(context)[1]["content"])

    assert payload["task"].startswith("Revise the candidate hypothesis")
    assert payload["repair_context"] is None
    assert payload["experiment_feedback"] == context["previous_failure_feedback"]


def test_day5_feeds_validator_errors_back_to_llm_once(tmp_path, monkeypatch):
    base_run_dir = tmp_path / "base-run"
    base_run_dir.mkdir()
    (base_run_dir / "state.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "tensor_preview": "tensor-preview.png",
                    "interpolation_preview": "interpolation-preview.png",
                },
                "results": {
                    "image_profile": {
                        "data_type": "color_image",
                        "image_shape": [16, 16, 3],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "research_agent.workflow_day5.load_improver_context",
        lambda _: _context(),
    )
    generator = RepairingGenerator()
    validator = RejectThenAcceptValidator()
    visual_evaluator = RecordingVisualEvaluator()
    workflow = Day5Workflow(
        Day5WorkflowConfig(
            base_run_dir=str(base_run_dir),
            candidate_root=str(tmp_path / "candidates"),
            output_dir=str(tmp_path / "outputs"),
            llm_mode="off",
            visual_assessment=True,
            smoke_timeout_seconds=10,
        ),
        generator=generator,
        validator=validator,
        visual_evaluator=visual_evaluator,
    )

    state = workflow.run()

    assert state["stage"] == "VALIDATED"
    assert state["validation"]["eligible_for_training"] is True
    assert state["generation"]["mode"] == "llm_repaired"
    assert state["generation"]["repair_attempted"] is True
    assert len(state["attempt_history"]) == 2
    assert state["attempt_history"][0]["validation"]["passed"] is False
    assert state["attempt_history"][1]["validation"]["passed"] is True
    assert len(generator.contexts) == 2
    assert validator.calls == 2
    assert "from typing import ... is not allowed" in generator.contexts[1][
        "previous_failure_feedback"
    ]
    assert "previous_candidate" in generator.contexts[1]
    assert "initial_interpolation_visual_assessment" not in generator.contexts[0]
    assert visual_evaluator.calls == []
    initial_visual_path = Path(
        state["artifacts"]["initial_interpolation_visual_assessment"]
    )
    assert initial_visual_path.is_file()
    initial_visual = json.loads(initial_visual_path.read_text(encoding="utf-8"))
    assert initial_visual["status"] == "skipped"
    assert initial_visual["comparison_mode"] == "structured_algorithm_reference_only"

    final_manifest = json.loads(
        Path(state["artifacts"]["manifest"]).read_text(encoding="utf-8")
    )
    assert final_manifest["validation_round"] == 2
    assert final_manifest["eligible_for_training"] is True
    assert final_manifest["declared_search_space"] == {
        "tv_weight": [0.0, 0.0001, 0.0005, 0.001]
    }
    assert final_manifest["effective_search_space"] == final_manifest[
        "executable_search_space"
    ]
    assert "allowed_search_space" not in final_manifest


def test_day5_feeds_search_space_mismatch_back_to_llm_before_fallback(
    tmp_path, monkeypatch
):
    base_run_dir = tmp_path / "base-run"
    base_run_dir.mkdir()
    (base_run_dir / "state.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "research_agent.workflow_day5.load_improver_context",
        lambda _: _context(),
    )
    generator = ManifestMismatchGenerator()
    validator = AlwaysAcceptValidator()
    workflow = Day5Workflow(
        Day5WorkflowConfig(
            base_run_dir=str(base_run_dir),
            candidate_root=str(tmp_path / "candidates"),
            output_dir=str(tmp_path / "outputs"),
            llm_mode="off",
            smoke_timeout_seconds=10,
        ),
        generator=generator,
        validator=validator,
    )

    state = workflow.run()

    assert state["stage"] == "VALIDATED"
    assert validator.calls == 2
    assert len(generator.contexts) == 2
    assert state["generation"]["mode"] == "llm_repaired"
    assert state["generation"]["search_space_contract_passed"] is True
    assert state["attempt_history"][0]["validation"]["passed"] is False
    assert "SEARCH_SPACE_CONTRACT_MISMATCH" in generator.contexts[1][
        "previous_failure_feedback"
    ][0]
    assert "previous_candidate" in generator.contexts[1]
    manifest = json.loads(
        Path(state["artifacts"]["manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["declared_search_space"]["tv_weight"] == [
        0.0,
        0.0001,
        0.0005,
        0.001,
    ]
    assert manifest["executable_search_space"]["tv_weight"] == [
        0.0,
        0.0001,
        0.0005,
        0.001,
    ]
    assert manifest["effective_search_space"] == manifest[
        "executable_search_space"
    ]
    assert manifest["search_space_contract"]["passed"] is True
    assert "allowed_search_space" not in manifest


def test_day5_uses_safe_fallback_after_two_rejected_llm_candidates(
    tmp_path, monkeypatch
):
    base_run_dir = tmp_path / "base-run"
    base_run_dir.mkdir()
    (base_run_dir / "state.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "research_agent.workflow_day5.load_improver_context",
        lambda _: _context(),
    )
    generator = AlwaysRejectLLMGenerator()
    validator = AcceptDeterministicFallbackValidator()
    workflow = Day5Workflow(
        Day5WorkflowConfig(
            base_run_dir=str(base_run_dir),
            candidate_root=str(tmp_path / "candidates"),
            output_dir=str(tmp_path / "outputs"),
            llm_mode="off",
            smoke_timeout_seconds=10,
        ),
        generator=generator,
        validator=validator,
    )

    state = workflow.run()

    assert state["stage"] == "VALIDATED"
    assert state["generation"]["mode"] == "deterministic_template"
    assert "safety fallback" in state["generation"]["fallback_reason"]
    assert len(state["attempt_history"]) == 3
    assert validator.calls == 3
    assert len(generator.contexts) == 2
