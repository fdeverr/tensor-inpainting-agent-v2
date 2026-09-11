import json
from pathlib import Path

from research_agent.candidate.generator import (
    CandidateGenerator,
    CandidateGenerationResult,
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
        "base_metrics": {"missing_psnr": 10.0},
        "interpolation_metrics": {"missing_psnr": 11.0},
        "comparison": {"winner": "interpolation"},
        "base_model_source": "class TuckerDecomposition: pass",
        "base_class_name": "TuckerDecomposition",
        "previous_failure_feedback": [],
    }


def test_candidate_prompt_matches_the_executable_model_contract():
    payload = json.loads(CandidateGenerator._messages(_context())[1]["content"])
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
        "The resolved budget"
    )
    assert "architecture_family" in payload["output_schema"]["properties"]
    assert "training_budget" in payload["output_schema"]["properties"]


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
    (base_run_dir / "state.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "research_agent.workflow_day5.load_improver_context",
        lambda _: _context(),
    )
    generator = RepairingGenerator()
    validator = RejectThenAcceptValidator()
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

    final_manifest = json.loads(
        Path(state["artifacts"]["manifest"]).read_text(encoding="utf-8")
    )
    assert final_manifest["validation_round"] == 2
    assert final_manifest["eligible_for_training"] is True


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
