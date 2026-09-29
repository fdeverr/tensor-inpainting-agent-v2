import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from research_agent.agent_tools import build_research_tool_registry
from research_agent.agent_tools.framework import ToolStatus
from research_agent.agent_tools.research_tools import (
    TuneTensorModelTool,
    _default_candidates,
)
from research_agent.workflow import (
    Day3Workflow,
    Day3WorkflowConfig,
    WorkflowExecutionError,
    run_day3_workflow,
)


def _write_small_image(path: Path) -> None:
    height, width = 12, 16
    y, x = np.indices((height, width), dtype=np.float32)
    image = np.stack(
        (
            x / (width - 1),
            y / (height - 1),
            0.2 + 0.6 * (x + y) / (width + height - 2),
        ),
        axis=-1,
    )
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def _candidate():
    return [
        {
            "hyperparameters": {
                "rank_h": 3,
                "rank_w": 3,
                "rank_c": 2,
                "init_scale": 0.15,
            },
            "learning_rate": 0.04,
        }
    ]


def test_tuning_expands_best_trial_when_checkpoint_is_near_limit(
    tmp_path, monkeypatch
):
    calls = []
    ground_truth_path = tmp_path / "ground_truth.npy"
    np.save(ground_truth_path, np.zeros((4, 4, 3), dtype=np.float32))

    monkeypatch.setattr(
        "research_agent.agent_tools.research_tools._load_observed_pair",
        lambda corrupted_path, mask_path: (
            np.zeros((4, 4, 3), dtype=np.float32),
            np.ones((4, 4), dtype=np.bool_),
        ),
    )

    def fake_train(**kwargs):
        max_steps = kwargs["config"].max_steps
        calls.append(max_steps)
        return SimpleNamespace(
            best_step=2 if max_steps == 4 else max_steps,
            best_validation_mse=0.1 if max_steps == 4 else 0.2,
            best_missing_psnr=10.0,
            runtime_seconds=0.5,
            parameter_count=10,
            device="cpu",
            stopped_early=False,
            reconstruction=np.zeros((4, 4, 3), dtype=np.float32),
            history=[{
                "step": 2 if max_steps == 4 else max_steps,
                "data_train_loss": 0.05,
                "total_train_loss": 0.05,
            }],
            state_dict={},
            train_mask=np.ones((4, 4), dtype=np.bool_),
        )

    monkeypatch.setattr(
        "research_agent.agent_tools.research_tools.train_tensor_model",
        fake_train,
    )
    output_path = tmp_path / "tuning.json"
    response = TuneTensorModelTool().run(
        {
            "run_id": "expand-test",
            "corrupted_path": "corrupted.npy",
            "mask_path": "mask.npy",
            "ground_truth_path": str(ground_truth_path),
            "output_path": str(output_path),
            "model_name": "tucker",
            "seed": 3,
            "candidates": _candidate(),
            "max_steps": 2,
            "auto_expand_steps": True,
            "max_steps_ceiling": 4,
            "near_limit_ratio": 0.9,
            "expansion_factor": 2.0,
            "device": "cpu",
        }
    )

    assert response.status == ToolStatus.SUCCESS
    assert calls == [2, 4]
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["trial_count"] == 1
    assert result["training_run_count"] == 2
    assert result["best"]["max_steps"] == 4
    assert result["best"]["best_step"] == 2
    assert result["expansion_history"][0]["to_max_steps"] == 4


def test_tuning_refines_learning_rate_around_coarse_winner(tmp_path, monkeypatch):
    calls = []
    ground_truth_path = tmp_path / "ground_truth.npy"
    np.save(ground_truth_path, np.zeros((4, 4, 3), dtype=np.float32))
    monkeypatch.setattr(
        "research_agent.agent_tools.research_tools._load_observed_pair",
        lambda corrupted_path, mask_path: (
            np.zeros((4, 4, 3), dtype=np.float32),
            np.ones((4, 4), dtype=np.bool_),
        ),
    )

    def fake_train(**kwargs):
        learning_rate = kwargs["config"].learning_rate
        calls.append(learning_rate)
        return SimpleNamespace(
            best_step=1,
            best_validation_mse=abs(np.log10(learning_rate) - np.log10(0.03)),
            best_missing_psnr=20.0,
            runtime_seconds=0.1,
            parameter_count=10,
            device="cpu",
            stopped_early=True,
            reconstruction=np.zeros((4, 4, 3), dtype=np.float32),
            history=[{
                "step": 1,
                "data_train_loss": 0.05,
                "total_train_loss": 0.05,
            }],
            state_dict={},
            train_mask=np.ones((4, 4), dtype=np.bool_),
        )

    monkeypatch.setattr(
        "research_agent.agent_tools.research_tools.train_tensor_model",
        fake_train,
    )
    output_path = tmp_path / "fine-tuning.json"
    candidates = [
        {"hyperparameters": _candidate()[0]["hyperparameters"], "learning_rate": rate}
        for rate in (0.001, 0.01, 0.1)
    ]
    response = TuneTensorModelTool().run(
        {
            "run_id": "fine-search-test",
            "corrupted_path": "corrupted.npy",
            "mask_path": "mask.npy",
            "ground_truth_path": str(ground_truth_path),
            "output_path": str(output_path),
            "model_name": "tucker",
            "seed": 3,
            "candidates": candidates,
            "max_steps": 2,
            "refine_learning_rate": True,
            "learning_rate_refinement_factor": 3.0,
            "device": "cpu",
        }
    )

    assert response.status == ToolStatus.SUCCESS
    assert calls == pytest.approx([0.001, 0.01, 0.1, 0.01 / 3.0, 0.03])
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["coarse_trial_count"] == 3
    assert result["fine_trial_count"] == 2
    assert result["best"]["search_stage"] == "fine"
    assert result["best"]["learning_rate"] == pytest.approx(0.03)


def test_registry_exposes_six_structured_tools():
    registry = build_research_tool_registry()
    assert registry.list_tools() == [
        "analyze_image",
        "run_interpolation",
        "tune_tensor_model",
        "train_tensor_model",
        "evaluate_reconstruction",
        "compare_experiments",
    ]
    for tool in registry.get_all_tools():
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == tool.name


@pytest.mark.parametrize(
    "model_name,required_names",
    (
        ("mode3", {"rank", "init_scale"}),
        ("nonnegative_cp", {"rank", "init_scale"}),
        ("btd", {"num_blocks", "rank_h", "rank_w", "rank_c", "init_scale"}),
        ("tsvd", {"rank", "init_scale"}),
        (
            "nonnegative_tucker",
            {"rank_h", "rank_w", "rank_c", "init_scale"},
        ),
        (
            "hierarchical_tucker",
            {"rank_h", "rank_w", "rank_c", "rank_spatial", "init_scale"},
        ),
        ("tt", {"rank_1", "rank_2", "init_scale"}),
        ("tensor_ring", {"rank", "init_scale"}),
        (
            "siren",
            {
                "hidden_features",
                "hidden_layers",
                "first_omega_0",
                "hidden_omega_0",
            },
        ),
    ),
)
def test_default_candidates_follow_each_model_search_space(
    model_name,
    required_names,
):
    candidates = _default_candidates(model_name, (8, 7, 3))

    assert len(candidates) == 2
    assert all(
        set(candidate["hyperparameters"]) == required_names
        for candidate in candidates
    )


def test_tool_registry_returns_structured_error_with_timing(tmp_path):
    registry = build_research_tool_registry()
    response = registry.execute_tool(
        "analyze_image",
        {
            "run_id": "bad-run",
            "image_path": str(tmp_path / "missing.png"),
            "run_dir": str(tmp_path / "run"),
            "mask_type": "block",
            "missing_rate": 0.4,
            "seed": 1,
        },
    )
    assert response.status == ToolStatus.ERROR
    assert response.error_info["code"] == "INVALID_PARAM"
    assert response.stats["time_ms"] >= 0
    assert response.context["tool_name"] == "analyze_image"


def test_day3_workflow_runs_all_tools_and_preserves_gt_boundary(tmp_path):
    image_path = tmp_path / "image.png"
    _write_small_image(image_path)
    state = run_day3_workflow(
        Day3WorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            image_size=None,
            missing_rate=0.3,
            seed=19,
            max_steps=20,
            validation_interval=5,
            patience=10,
            device="cpu",
            full_reference_metrics=False,
            candidates=_candidate(),
        )
    )

    assert state["stage"] == "COMPLETED"
    assert state["last_successful_stage"] == "COMPLETED"
    assert state["results"]["selected_trial"]["best_step"] >= 1
    profile = state["results"]["image_profile"]
    assert profile["missing_component_count"] >= 1
    assert 0.0 < profile["largest_missing_component_image_ratio"] < 1.0
    assert 0.0 <= profile["visible_mean_absolute_channel_correlation"] <= 1.0
    assert 0.0 <= profile["visible_local_smoothness_score"] <= 1.0
    assert profile["visible_high_frequency_energy_ratio"] >= 0.0
    assert state["results"]["training"]["observed_pixels_used"] == round(
        12 * 16 * 0.7
    )
    assert state["results"]["comparison"]["winner"] in {
        "baseline",
        "candidate",
        "tie",
    }
    for artifact_path in state["artifacts"].values():
        assert Path(artifact_path).exists()

    tuning = json.loads(Path(state["artifacts"]["tuning_result"]).read_text())
    assert tuning["ground_truth_used"] is True
    assert tuning["selection_metric"] == "missing_region_ground_truth_mse"
    training = json.loads(
        Path(state["artifacts"]["tensor_training_result"]).read_text()
    )
    assert training["ground_truth_used_for_selection"] is True
    assert training["reused_without_retraining"] is True

    trace_path = Path(state["artifacts"]["trace_jsonl"])
    trace_text = trace_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in trace_text.splitlines()]
    tool_order = [
        event["payload"]["tool_name"]
        for event in events
        if event["event"] == "tool_call"
    ]
    assert tool_order == [
        "analyze_image",
        "run_interpolation",
        "tune_tensor_model",
        "evaluate_reconstruction",
        "evaluate_reconstruction",
        "compare_experiments",
    ]
    assert "evaluation_ground_truth" not in trace_text
    run_dir = Path(state["artifacts"]["run_dir"])
    assert (run_dir / "evaluation_ground_truth.npy").is_file()
    assert not (run_dir / "evaluation_ground_truth_preview.png").exists()
    assert not (run_dir / "evaluation_ground_truth.mat").exists()


def test_failure_keeps_last_successful_stage(tmp_path):
    image_path = tmp_path / "image.png"
    _write_small_image(image_path)
    workflow = Day3Workflow(
        Day3WorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            image_size=None,
            max_steps=5,
            validation_interval=1,
            device="cpu",
            full_reference_metrics=False,
            candidates=[
                {
                    "hyperparameters": {"unknown_argument": 1},
                    "learning_rate": 0.03,
                }
            ],
        )
    )

    with pytest.raises(WorkflowExecutionError):
        workflow.run()

    saved_state = json.loads(workflow.state_path.read_text(encoding="utf-8"))
    assert saved_state["stage"] == "INTERPOLATED"
    assert saved_state["last_successful_stage"] == "INTERPOLATED"
    assert saved_state["last_error"]["tool_name"] == "tune_tensor_model"
