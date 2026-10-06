import json
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research_agent.candidate.generator import CandidateGenerator, deterministic_candidate
from research_agent.candidate.approved_registry import promote_candidate
from research_agent.evolution_knowledge import (
    GlobalExperienceStore,
    RunPracticeStore,
    describe_metrics,
    compact_run_practices, prepare_run_experience, commit_run_experience,
    global_experience_context,
)
from research_agent.workflow_full import FullWorkflowConfig
from research_agent.workflow_day6 import Day6WorkflowConfig
from research_agent.workflow_day6 import Day6Workflow


def test_candidate_declares_exactly_one_controlled_change():
    proposal = deterministic_candidate("tucker")

    assert proposal.mutation_target in {"algorithm", "loss"}
    assert len(proposal.proposed_changes) == 1
    CandidateGenerator._validate_framework_lock(proposal)


def test_metric_descriptions_include_value_direction_scope_and_meaning():
    described = describe_metrics(
        {
            "full_psnr": 25.5,
            "missing_psnr": 21.5,
            "composite_ssim": 0.81,
            "lpips": 0.2,
        }
    )

    assert described["full_psnr"]["value"] == 25.5
    assert described["full_psnr"]["scope"].startswith("full completed tensor")
    assert described["missing_psnr"]["value"] == 21.5
    assert described["missing_psnr"]["unit"] == "dB"
    assert described["missing_psnr"]["direction"] == "higher_is_better"
    assert "missing region" in described["missing_psnr"]["physical_meaning"]
    assert described["lpips"]["direction"] == "lower_is_better"

    evolution_described = describe_metrics(
        {"full_psnr": 25.5, "missing_psnr": 21.5, "composite_ssim": 0.81},
        include_full_psnr=False,
    )
    assert "full_psnr" not in evolution_described
    assert evolution_described["missing_psnr"]["value"] == 21.5


def _practice_record():
    metrics = describe_metrics(
        {"full_psnr": 24.0, "missing_psnr": 20.0, "composite_ssim": 0.8}
    )
    return {
        "framework": {
            "name": "tucker",
            "base_method": "tucker",
            "conditions": {"seed": 42},
            "training_diagnostics": {
                "record_count": 1,
                "signals": [{"code": "training_loss_plateau"}],
            },
        },
        "goal": "Improve continuity in the held-out image region.",
        "method": {
            "idea": "Add one bounded total-variation term to the loss.",
            "target": "loss",
            "single_change": "Add total variation to loss_terms.",
        },
        "result": {
            "incumbent_metrics": metrics,
            "candidate_metrics": metrics,
            "deltas": {"missing_psnr_db": 0.0, "composite_ssim": 0.0},
            "decision": "reject",
            "accepted": False,
            "incumbent_after": "tucker",
            "visual_assessment": {"status": "not_configured"},
        },
    }


def _reusable_experience():
    return {
        "experience": (
            "Total-variation regularization can improve spatial continuity, but its "
            "weight must be validated to avoid suppressing image detail."
        ),
        "confidence": "low",
    }


def test_practice_is_run_scoped_and_global_experience_accumulates_across_runs(
    tmp_path,
):
    global_root = tmp_path / "global"
    first_global = GlobalExperienceStore(global_root, "tucker")
    first_practice = RunPracticeStore(tmp_path / "run-1", "tucker", "workflow-1")

    practice_artifacts = first_practice.record(_practice_record())
    experience_artifacts = first_global.record(_reusable_experience())

    second_global = GlobalExperienceStore(global_root, "tucker")
    second_practice = RunPracticeStore(tmp_path / "run-2", "tucker", "workflow-2")
    second_context = second_global.context()

    assert Path(practice_artifacts["practice_markdown"]).is_file()
    assert Path(experience_artifacts["experience_markdown"]).is_file()
    assert second_context["record_count"] == 1
    assert set(second_context["reusable_experience"][0]) == {
        "experience",
        "confidence",
    }
    markdown = Path(experience_artifacts["experience_markdown"]).read_text()
    assert "workflow-1" not in markdown
    assert "candidate-1" not in markdown
    assert "第 1 轮" not in markdown
    assert second_practice.context()["current_run_practice"] == []
    stored_practice = first_practice.context()["current_run_practice"][0]
    assert set(stored_practice) == {"framework", "goal", "method", "result"}
    assert stored_practice["framework"]["name"] == "tucker"
    assert not (global_root / "tucker" / "practice.md").exists()


def test_experience_is_extracted_from_the_complete_practice_tuple():
    class RecordingLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages, temperature=0.0):
            self.messages = messages
            return json.dumps(
                {
                    "experience": (
                        "在低秩张量框架出现训练平台时，小权重平滑先验可能改善连续性，"
                        "但必须检查结构细节是否退化。"
                    ),
                    "confidence": "low",
                },
                ensure_ascii=False,
            )

    llm = RecordingLLM()
    experience = CandidateGenerator(llm).extract_experience(_practice_record())
    payload = json.loads(llm.messages[1]["content"])

    assert set(payload["practice_tuple"]) == {
        "framework",
        "goal",
        "method",
        "result",
    }
    assert set(experience) == {"experience", "confidence"}


def test_legacy_verbose_experience_is_compacted_on_open(tmp_path):
    directory = tmp_path / "global" / "tucker"
    directory.mkdir(parents=True)
    (directory / "reusable_experience.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "record_type": "reusable_experience",
                "source": {
                    "workflow_id": "day6-old",
                    "round": 5,
                    "candidate_id": "candidate-old",
                },
                "problem_pattern": "verbose details",
                "reusable_principle": "Continuity can improve PSNR while smoothing detail.",
                "confidence": "medium",
            }
        )
        + "\n"
    )
    (directory / "reusable_experience.md").write_text("## day6-old / 第 5 轮\n")

    store = GlobalExperienceStore(tmp_path / "global", "tucker")

    record = json.loads(store.experience_jsonl.read_text())
    assert record == {
        "experience": "Continuity can improve PSNR while smoothing detail.",
        "confidence": "medium",
    }
    markdown = store.experience_markdown.read_text()
    assert "day6-old" not in markdown
    assert "candidate-old" not in markdown
    assert "经验：Continuity can improve PSNR" in markdown


def test_duplicate_global_experience_is_not_appended_twice(tmp_path):
    store = GlobalExperienceStore(tmp_path / "global", "tucker")

    store.record(_reusable_experience())
    store.record(_reusable_experience())

    assert store.context()["record_count"] == 1


def test_global_experience_is_separated_by_base_decomposition(tmp_path):
    tucker = GlobalExperienceStore(tmp_path / "global", "tucker")
    cp = GlobalExperienceStore(tmp_path / "global", "cp")

    assert tucker.directory != cp.directory
    assert tucker.context()["record_count"] == 0
    assert cp.context()["record_count"] == 0


def test_evolution_defaults_to_five_rounds(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"placeholder")
    base_run = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base_run.mkdir()
    candidate.mkdir()
    (base_run / "state.json").write_text("{}")
    (candidate / "manifest.json").write_text("{}")

    assert FullWorkflowConfig(image_path=str(image)).max_improvement_rounds == 5
    assert Day6WorkflowConfig(
        base_run_dir=str(base_run), initial_candidate_dir=str(candidate)
    ).max_improvement_rounds == 5


def test_promoted_algorithm_is_grouped_under_base_decomposition(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    code = "class CandidateTensorInpaintingModel: pass\n"
    (candidate / "model.py").write_text(code)
    (candidate / "idea.json").write_text("{}")
    (candidate / "validation.json").write_text(json.dumps({"passed": True}))
    (candidate / "manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": "candidate-1",
                "base_method": "tucker",
                "eligible_for_training": True,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        )
    )

    promoted = promote_candidate(
        candidate_dir=str(candidate),
        approved_root=str(tmp_path / "approved"),
        algorithm_name="tucker_evolved",
        source_run_id="run-1",
        best_config={"rank": 4},
        comparison={"accepted": True},
        conditions={"seed": 42},
    )

    assert Path(promoted["approved_dir"]).parent.parent.name == "tucker"


def test_accepted_candidate_becomes_incumbent_and_loop_continues(tmp_path, monkeypatch):
    class FakeModel:
        @classmethod
        def search_space(cls, image_shape):
            return {"rank": [1]}

    class Trace:
        def log_event(self, *args, **kwargs):
            pass

        def finalize(self):
            pass

    candidates = []
    for name in ("candidate-1", "candidate-2", "candidate-3"):
        directory = tmp_path / "candidates" / "tucker" / name
        directory.mkdir(parents=True)
        (directory / "model.py").write_text("# model")
        (directory / "idea.json").write_text(
            json.dumps(
                {
                    "candidate_id": name,
                    "mutation_goal": "Improve the current tensor reconstruction quality.",
                    "idea": "Test one isolated loss mechanism against the incumbent.",
                    "mutation_target": "loss",
                    "single_change": "Add one bounded regularization mechanism.",
                }
            )
        )
        candidates.append(directory)

    workflow = object.__new__(Day6Workflow)
    workflow.config = SimpleNamespace(
        base_run_dir=str(tmp_path / "base"),
        initial_candidate_dir=str(candidates[0]),
        candidate_root=str(tmp_path / "candidates"),
        approved_root=str(tmp_path / "approved"),
        knowledge_root=str(tmp_path / "knowledge"),
        output_dir=str(tmp_path / "outputs"),
        llm_mode="off",
        tuning_trials=1,
        learning_rate_candidates=(0.01,),
        refine_learning_rate=False,
        learning_rate_refinement_factor=3.0,
        max_steps=5,
        max_improvement_rounds=3,
        dataset_algorithm_reference=None,
        evolution_cases=None,
        validation_interval=1,
        patience=2,
        device="cpu",
        minimum_psnr_delta=0.2,
        ssim_tolerance=0.002,
        smoke_timeout_seconds=1.0,
            full_reference_metrics=True,
            no_reference_metrics=False,
            visual_assessment=False,
    )
    workflow.workflow_id = "workflow-test"
    workflow.run_dir = tmp_path / "run"
    workflow.run_dir.mkdir()
    workflow.state_path = workflow.run_dir / "state.json"
    workflow.trace = Trace()
    summary_calls = []
    prior_store = GlobalExperienceStore(tmp_path / "knowledge", "tucker", "color_image", {
        "mask_type": "block", "requested_missing_rate": 0.4, "actual_missing_rate": 0.4,
        "source_run_id": "prior-run"})
    prior_store.record(_reusable_experience())
    class OncePerRunGenerator:
        def extract_experience(self, practice):
            raise AssertionError("must not summarize each round")

        def extract_run_experience(self, practices, outcome):
            assert len(practices) == 3
            assert [p["result"]["accepted"] for p in practices] == [True, False, False]
            assert prior_store.context()["record_count"] == 1  # No per-round append.
            summary_calls.append(practices)
            return _reusable_experience()
    workflow.generator = OncePerRunGenerator()
    workflow.base_state = {
        "run_id": "base-run",
        "selected_model": "tucker",
        "config": {"seed": 42, "mask_type": "block"},
        "artifacts": {
            "corrupted": "corrupted.npy",
            "mask": "mask.npy",
            "interpolation": "interpolation.npy",
        },
        "results": {
            "training": {"learning_rate": 0.01},
            "selected_trial": {"hyperparameters": {"rank": 1}},
            "interpolation_metrics": {
                "full_psnr": 9.0,
                "missing_psnr": 5.0,
                "composite_ssim": 0.5,
            },
            "image_profile": {"actual_missing_rate": 0.4},
        },
    }
    workflow.state = {
        "workflow_id": workflow.workflow_id,
        "stage": "CREATED",
        "config": vars(workflow.config),
        "base_run_id": "base-run",
        "rounds": [],
        "feedback_history": [],
        "candidate_generation_attempts": [],
        "accepted": False,
        "accepted_rounds": [],
        "stop_reason": None,
        "best_available": None,
        "promotion": None,
        "overall_comparison": None,
        "artifacts": {"run_dir": str(workflow.run_dir)},
        "updated_at": None,
    }
    generation_contexts = []

    def make_next(context, round_index):
        generation_contexts.append(deepcopy(context))
        return str(candidates[round_index - 1])

    workflow._make_next_candidate = make_next

    class ForbiddenVisualEvaluator:
        def evaluate(self, *args, **kwargs):
            raise AssertionError("Disabled visual assessment must never run")

    workflow.visual_evaluator = ForbiddenVisualEvaluator()

    monkeypatch.setattr(
        "research_agent.workflow_day6.MODEL_CLASSES", {"tucker": FakeModel}
    )
    monkeypatch.setattr(
        "research_agent.workflow_day6.load_improver_context",
        lambda path: {"base_run_id": "base-run", "base_method": "tucker", "base_class_name": "TuckerDecomposition"},
    )
    monkeypatch.setattr(
        "research_agent.workflow_day6.load_tensor_data",
        lambda *args, **kwargs: np.zeros((4, 4, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        "research_agent.workflow_day6.load_observation_mask",
        lambda *args, **kwargs: np.ones((4, 4), dtype=bool),
    )
    monkeypatch.setattr(
        "research_agent.workflow_day6.load_validated_candidate",
        lambda directory: (
            FakeModel,
            {
                "candidate_id": Path(directory).name,
                "base_method": "tucker",
                "architecture_family": "tensor_decomposition",
                "allowed_search_space": {"rank": [1]},
                "training_budget": {
                    "max_steps": 5,
                    "validation_interval": 1,
                    "early_stopping_patience": 2,
                },
            },
        ),
    )

    tuning_calls = []
    final_calls = []

    def tuning(*args, **kwargs):
        tuning_calls.append(kwargs["model_name"])
        return {
            "trial_count": 1,
            "trials": [{"runtime_seconds": 0.1}],
            "best": {
                "best_validation_mse": 0.1,
                "best_step": 1,
                "hyperparameters": {"rank": 1},
                "learning_rate": 0.01,
                "history": [
                    {
                        "step": 1,
                        "data_train_loss": 0.1,
                        "total_train_loss": 0.1,
                        "validation_mse": 0.1,
                    }
                ],
            },
        }

    scores = {
        "tucker": (10.0, 0.8, 0.30),
        "candidate-1": (11.0, 0.81, 0.24),
        "candidate-2": (10.5, 0.80, 0.28),
        "candidate-3": (10.0, 0.79, 0.32),
    }

    def final(model_name, *args, **kwargs):
        final_calls.append(model_name)
        psnr, ssim, lpips = scores[model_name]
        output = workflow.run_dir / (model_name + ".npy")
        return {
            "metrics": {
                "full_psnr": psnr,
                "missing_psnr": psnr - 4.0,
                "composite_ssim": ssim,
                "lpips": lpips,
            },
            "runtime_seconds": 0.1,
            "parameter_count": 1,
            "final_train_mse": 0.01,
            "final_total_loss": 0.01,
            "artifacts": {"reconstruction": str(output), "preview": str(output)},
        }

    monkeypatch.setattr("research_agent.workflow_day6.tune_model_on_observed_pixels", tuning)
    monkeypatch.setattr("research_agent.workflow_day6.final_fit_and_evaluate", final)
    monkeypatch.setattr(
        "research_agent.workflow_day6.promote_candidate",
        lambda **kwargs: {
            "algorithm_name": kwargs["algorithm_name"],
            "approved_dir": str(tmp_path / "approved" / "tucker"),
        },
    )

    state = workflow.run()

    assert len(state["rounds"]) == 3
    assert state["accepted_rounds"] == [1]
    assert state["rounds"][1]["incumbent_before"] == "candidate-1"
    assert tuning_calls == ["tucker", "candidate-1", "candidate-2", "candidate-3"]
    assert final_calls == ["tucker", "candidate-1", "candidate-2", "candidate-3"]
    assert state["rounds"][0]["judgment"]["budget_audit"][
        "incumbent_training_reused"
    ] is False
    assert state["rounds"][1]["judgment"]["budget_audit"][
        "incumbent_training_reused"
    ] is True
    assert state["rounds"][2]["judgment"]["budget_audit"][
        "incumbent_training_reused"
    ] is True
    assert state["rounds"][0]["result_summary"]["incumbent_metrics"]["lpips"][
        "value"
    ] == 0.30
    assert state["rounds"][0]["result_summary"]["candidate_metrics"]["lpips"][
        "value"
    ] == 0.24
    assert state["rounds"][0]["result_summary"]["deltas"]["lpips"] == pytest.approx(
        -0.06
    )
    assert state["feedback_history"][0]["lpips_delta"] == pytest.approx(-0.06)
    assert all(
        round_record["training_budget"]["effective_shared_budget"]
        == state["rounds"][0]["training_budget"]["effective_shared_budget"]
        for round_record in state["rounds"]
    )
    assert state["promotion"]["algorithm_name"] == "tucker_evolved"
    assert state["stop_reason"] == "maximum_improvement_rounds_reached"
    assert len(
        (workflow.run_dir / "knowledge" / "practice.jsonl").read_text().splitlines()
    ) == 3
    assert len(summary_calls) == 1
    assert state["global_experience"]["status"] == "committed"
    assert state["global_experience"]["conditions"]["round_count"] == 3
    assert prior_store.context()["record_count"] == 2
    assert not list(workflow.run_dir.glob("round*/experience_record.json"))
    assert all("experience" not in record for record in state["rounds"])

    assert len(generation_contexts) == 2
    assert all(
        context["run_wide_fair_training_contract"]["training_config"]
        == state["rounds"][0]["training_budget"]["effective_shared_budget"]
        for context in generation_contexts
    )
    second_memory = generation_contexts[0]["evolution_memory"]["current_run_practice"]
    third_context = generation_contexts[1]
    third_memory = third_context["evolution_memory"]["current_run_practice"]
    assert second_memory["round_count"] == 1
    assert third_memory["round_count"] == 2
    assert third_context["evolution_memory"]["global_reusable_experience"]["record_count"] == 1
    assert all(
        set(item) == {"framework", "goal", "method", "result"}
        for item in third_memory["current_run_practice"]
    )
    assert [
        item["result"]["accepted"]
        for item in third_memory["current_run_practice"]
    ] == [True, False]
    assert len(third_context["previous_failure_feedback"]) == 2
    assert third_context["previous_round_result"]["goal"].startswith("Improve")
    latest_comparison = third_context["algorithm_comparison_reference"][
        "latest_evolution_round"
    ]
    assert latest_comparison["round"] == 2
    assert latest_comparison["incumbent_before"] == "candidate-1"
    assert latest_comparison["candidate"] == "candidate-2"
    assert latest_comparison["decision"] == "reject"
    assert latest_comparison["deltas"]["missing_psnr_db"] == pytest.approx(-0.5)
    assert third_context["incumbent_candidate"]["idea"]["candidate_id"] == "candidate-1"
    assert generation_contexts[0]["training_curve_summary"]["signals"][0][
        "code"
    ] == "best_validation_at_budget_end"
    prompt = json.loads(CandidateGenerator._messages(third_context)[1]["content"])
    assert prompt["context"]["evolution_memory"]["current_run_practice"]["round_count"] == 2
    assert len(prompt["experiment_feedback"]) == 2
    assert len(state["search_space_contract_migrations"]) == 3
    migrated_manifest = json.loads(
        (candidates[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert migrated_manifest["declared_search_space"] == {"rank": [1]}
    assert migrated_manifest["executable_search_space"] == {"rank": [1]}
    assert migrated_manifest["effective_search_space"] == {"rank": [1]}
    assert "allowed_search_space" not in migrated_manifest


def _run_summary_fixture(tmp_path):
    practice = _practice_record()
    practice["framework"]["conditions"].update({"data_type": "color_image", "mask_type": "block",
                                               "actual_missing_rate": 0.4, "training_budget": {"max_steps": 20}})
    store = RunPracticeStore(tmp_path / "run", "tucker", "day6-test")
    store.record(practice)
    state = {"workflow_id": "day6-test", "stage": "COMPLETED", "rounds": [{}],
             "best_available": {"algorithm": "tucker", "metrics": {"missing_psnr": 20.0}},
             "artifacts": {"run_practice": store.context()["documents"]}, "config": {"dataset_evaluation_steps": 30}}
    base = {"selected_model": "tucker", "config": {"mask_type": "block", "missing_rate": 0.4},
            "results": {"image_profile": {"data_type": "color_image", "actual_missing_rate": 0.4}}}
    return state, base, store


def test_run_level_extraction_uses_all_practices_and_calls_llm_only_once():
    practices = [_practice_record(), _practice_record()]
    practices[0]["result"]["accepted"] = True
    practices[0]["method"]["idea"] = "方法甲"
    practices[1]["method"]["idea"] = "方法乙"
    class RecordingLLM:
        calls = []
        def invoke(self, messages, temperature=0):
            self.calls.append(messages)
            return json.dumps({"experience": "同类任务中方法甲形成改善，方法乙仍需重新验证适用条件。", "confidence": "high"})
    llm = RecordingLLM()
    result = CandidateGenerator(llm).extract_run_experience(compact_run_practices(practices), {"algorithm": "winner"})
    assert len(llm.calls) == 1
    payload = json.loads(llm.calls[0][1]["content"])
    assert len(payload["practice_tuples"]) == 2
    assert [p["method"]["idea"] for p in payload["practice_tuples"]] == ["方法甲", "方法乙"]
    assert result["confidence"] == "medium"  # A single run is not broad empirical validation.


def test_run_summary_fallback_includes_successes_and_failures_with_one_attempt():
    practices = [_practice_record(), _practice_record()]
    practices[0]["result"]["accepted"] = True
    practices[0]["method"]["idea"] = "成功机制甲"
    practices[1]["method"]["idea"] = "失败机制乙"
    class BrokenLLM:
        calls = 0
        def invoke(self, *args, **kwargs):
            self.calls += 1
            return "not JSON"
    llm = BrokenLLM()
    result = CandidateGenerator(llm).extract_run_experience(practices, {"algorithm": "winner"})
    assert llm.calls == 1
    assert result["confidence"] == "low"
    assert "成功机制甲" in result["experience"] and "失败机制乙" in result["experience"]


def test_summary_commit_is_once_per_run_and_verifies_practice_source(tmp_path):
    state, base, practice = _run_summary_fixture(tmp_path)
    pending = prepare_run_experience(CandidateGenerator(None), state, base, str(tmp_path / "candidates"),
                                    str(tmp_path / "global"), source_run_id="complete-run")
    assert not (tmp_path / "global").exists()  # Summary alone is not a global append.
    first = commit_run_experience(pending)
    commit_run_experience(first)
    duplicate = {**pending, "experience": {"experience": "即使重新生成摘要措辞，也不能重复追加同一次运行。", "confidence": "low"}}
    commit_run_experience(duplicate)
    records = GlobalExperienceStore(tmp_path / "global", "tucker", "color_image").context()["reusable_experience"]
    assert len(records) == 1
    record = records[0]
    assert record["source_run_id"] == "complete-run"
    assert record["summary_scope"] == "whole_run_practice"
    assert record["round_count"] == 1 and record["accepted_round_count"] == 0
    assert record["requested_missing_rate"] == record["actual_missing_rate"] == 0.4
    assert record["training_budget"]["max_steps"] == 20
    practice.record(_practice_record())
    with pytest.raises(ValueError, match="practice library changed"):
        commit_run_experience(pending)


@pytest.mark.parametrize("problem", ["failed", "missing_practice"])
def test_incomplete_runs_do_not_create_global_experience(tmp_path, problem):
    state, base, _ = _run_summary_fixture(tmp_path)
    if problem == "failed":
        state["stage"] = "FAILED"
    else:
        state["rounds"].append({})
    with pytest.raises(ValueError):
        prepare_run_experience(CandidateGenerator(None), state, base, str(tmp_path / "candidates"), str(tmp_path / "global"))
    assert not (tmp_path / "global").exists()


def test_global_retrieval_prefers_matching_conditions_and_never_mixes_modalities(tmp_path):
    root = tmp_path / "global"
    for run, pattern, rate in [("same", "block", 0.4), ("near", "block", 0.5), ("far", "block", 0.8), ("different", "slices", 0.4)]:
        GlobalExperienceStore(root, "tucker", "color_image", {"mask_type": pattern,
            "requested_missing_rate": rate, "actual_missing_rate": rate, "source_run_id": run}).record(_reusable_experience())
    GlobalExperienceStore(root, "tucker", "audio", {"source_run_id": "audio"}).record(_reusable_experience())
    store = GlobalExperienceStore(root, "tucker", "Image", {"mask_type": "block", "requested_missing_rate": 0.4})
    context = store.context(3)
    assert [record["source_run_id"] for record in context["reusable_experience"]] == ["same", "near", "far"]
    assert context["record_count"] == 4
    assert all(record["data_type"] == "color_image" for record in context["reusable_experience"])


def test_corrupt_optional_global_memory_does_not_block_evolution(tmp_path):
    _, base, _ = _run_summary_fixture(tmp_path)
    directory = tmp_path / "global/Image/tucker"
    directory.mkdir(parents=True)
    (directory / "reusable_experience.jsonl").write_text("broken JSON\n")
    context = global_experience_context(base, str(tmp_path / "candidates"), str(tmp_path / "global"))
    assert context["status"] == "unavailable" and context["reusable_experience"] == []
    assert (directory / "reusable_experience.jsonl").read_text() == "broken JSON\n"


def test_concurrent_run_commits_do_not_lose_lessons(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def commit(index):
        GlobalExperienceStore(tmp_path, "tucker", "color_image", {"source_run_id": str(index)}).record(_reusable_experience())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(commit, range(8)))
    assert GlobalExperienceStore(tmp_path, "tucker", "color_image").context()["record_count"] == 8


def test_practice_change_during_summary_cannot_get_a_matching_new_hash(tmp_path):
    state, base, practice = _run_summary_fixture(tmp_path)
    class ChangingGenerator:
        def extract_run_experience(self, *args):
            practice.record(_practice_record())
            return _reusable_experience()
    with pytest.raises(ValueError, match="changed during"):
        prepare_run_experience(ChangingGenerator(), state, base, str(tmp_path / "candidates"), str(tmp_path / "global"))
    assert not (tmp_path / "global").exists()


def test_run_evidence_keeps_cohort_summary_and_failures_without_raw_curves():
    practice = _practice_record()
    practice["result"]["dataset_evaluation"] = {
        "incumbent_summary": {"complete": True, "mean_missing_psnr": 20.0},
        "candidate_summary": {"complete": False, "mean_missing_psnr": None},
        "incumbent": [{"status": "completed", "sample": "a.mat", "metrics": {"missing_psnr": 20.0}, "training": {"raw_history": "not needed"}}],
        "candidate": [{"status": "failed", "sample": "a.mat", "error": "OOM", "training": {"raw_history": "not needed"}}],
    }
    result = compact_run_practices([practice])[0]["result"]["dataset_evaluation"]
    assert result["incumbent_summary"]["mean_missing_psnr"] == 20.0
    assert result["sample_failures"] == [{"sample": "a.mat", "error": "OOM"}]
    assert result["incumbent"][0]["metrics"] == {"missing_psnr": 20.0}
    assert "training" not in json.dumps(result)


def test_cohort_run_summary_does_not_confuse_representative_and_dataset_scores(tmp_path):
    state, base, _ = _run_summary_fixture(tmp_path)
    state["best_available"]["metrics"] = {"missing_psnr": 50.0}
    state["overall_comparison"] = {"whole_modality_summary": {"mean_missing_psnr": 20.0}}
    class CheckingGenerator:
        def extract_run_experience(self, practices, outcome):
            assert "metrics" not in outcome
            assert outcome["whole_modality_summary"]["mean_missing_psnr"] == 20.0
            return _reusable_experience()
    result = prepare_run_experience(CheckingGenerator(), state, base,
        str(tmp_path / "candidates"), str(tmp_path / "global"), cases=[{}, {}])
    assert result["conditions"]["sample_count"] == 2


def test_incomplete_base_metadata_disables_optional_memory_without_guessing(tmp_path):
    context = global_experience_context({}, str(tmp_path / "candidates"), str(tmp_path / "global"))
    assert context["status"] == "unavailable"
    assert context["reusable_experience"] == []
    assert not (tmp_path / "global").exists()
