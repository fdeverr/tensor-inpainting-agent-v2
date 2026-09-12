import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research_agent.candidate.generator import CandidateGenerator, deterministic_candidate
from research_agent.candidate.approved_registry import promote_candidate
from research_agent.evolution_knowledge import (
    GlobalExperienceStore,
    RunPracticeStore,
    describe_metrics,
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
        {"missing_psnr": 21.5, "composite_ssim": 0.81, "lpips": 0.2}
    )

    assert described["missing_psnr"]["value"] == 21.5
    assert described["missing_psnr"]["unit"] == "dB"
    assert described["missing_psnr"]["direction"] == "higher_is_better"
    assert "missing region" in described["missing_psnr"]["physical_meaning"]
    assert described["lpips"]["direction"] == "lower_is_better"


def _practice_record():
    metrics = describe_metrics(
        {"missing_psnr": 20.0, "composite_ssim": 0.8}
    )
    return {
        "workflow_id": "workflow-1",
        "round": 1,
        "base_method": "tucker",
        "incumbent_before": "tucker",
        "candidate_id": "candidate-1",
        "mutation_goal": "Improve continuity in the held-out image region.",
        "idea": "Add one bounded total-variation term to the loss.",
        "mutation_target": "loss",
        "single_change": "Add total variation to loss_terms.",
        "result": {
            "incumbent_metrics": metrics,
            "candidate_metrics": metrics,
            "visual_assessment": {"status": "not_configured"},
        },
        "judgment": {"decision": "reject"},
        "incumbent_updated": False,
        "incumbent_after": "tucker",
        "conditions": {"seed": 42},
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
    assert first_practice.context()["current_run_practice"][0]["candidate_id"] == (
        "candidate-1"
    )
    assert not (global_root / "tucker" / "practice.md").exists()


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
    for name in ("candidate-1", "candidate-2"):
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
        max_steps=5,
        max_improvement_rounds=2,
        validation_ratio=0.1,
        validation_interval=1,
        patience=2,
        device="cpu",
        minimum_psnr_delta=0.2,
        ssim_tolerance=0.002,
        smoke_timeout_seconds=1.0,
        learned_metrics=False,
    )
    workflow.workflow_id = "workflow-test"
    workflow.run_dir = tmp_path / "run"
    workflow.run_dir.mkdir()
    workflow.state_path = workflow.run_dir / "state.json"
    workflow.trace = Trace()
    workflow.generator = object()
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
    workflow._make_next_candidate = lambda context, round_index: str(candidates[1])

    monkeypatch.setattr(
        "research_agent.workflow_day6.MODEL_CLASSES", {"tucker": FakeModel}
    )
    monkeypatch.setattr(
        "research_agent.workflow_day6.load_improver_context",
        lambda path: {"base_run_id": "base-run", "base_method": "tucker"},
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

    def tuning(*args, **kwargs):
        return {
            "trial_count": 1,
            "trials": [{"runtime_seconds": 0.1}],
            "best": {
                "best_validation_mse": 0.1,
                "best_step": 1,
                "hyperparameters": {"rank": 1},
            },
        }

    scores = {"tucker": (10.0, 0.8), "candidate-1": (11.0, 0.81), "candidate-2": (10.5, 0.80)}

    def final(model_name, *args, **kwargs):
        psnr, ssim = scores[model_name]
        output = workflow.run_dir / (model_name + ".npy")
        return {
            "metrics": {"missing_psnr": psnr, "composite_ssim": ssim},
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

    assert len(state["rounds"]) == 2
    assert state["accepted_rounds"] == [1]
    assert state["rounds"][1]["incumbent_before"] == "candidate-1"
    assert state["promotion"]["algorithm_name"] == "tucker_evolved"
    assert state["stop_reason"] == "maximum_improvement_rounds_reached"
    assert len(
        (workflow.run_dir / "knowledge" / "practice.jsonl").read_text().splitlines()
    ) == 2
    assert len(
        (tmp_path / "knowledge" / "tucker" / "reusable_experience.jsonl")
        .read_text()
        .splitlines()
    ) == 2
