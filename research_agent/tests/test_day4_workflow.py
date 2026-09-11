import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research_agent.workflow_day4 import (
    Day4WorkflowConfig,
    _candidates_from_plan,
    run_day4_workflow,
)
from research_agent.workflow_day5 import (
    Day5WorkflowConfig,
    run_day5_workflow,
)
from research_agent.workflow_day6 import (
    Day6WorkflowConfig,
    run_day6_workflow,
)


def _write_small_image(path: Path) -> None:
    height, width = 12, 18
    y, x = np.indices((height, width), dtype=np.float32)
    base = (x + 0.7 * y) / ((width - 1) + 0.7 * (height - 1))
    image = np.stack((base, 0.85 * base + 0.05, 0.7 * base + 0.1), axis=-1)
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def test_new_tensor_network_plans_become_bounded_tuning_candidates():
    profile = {"image_shape": [64, 128, 3]}
    tt_candidates = _candidates_from_plan(
        {
            "method": "tt",
            "suggested_hyperparameters": {
                "rank_1_candidates": [4, 8],
                "rank_2_candidates": [2, 3],
                "init_scale": 0.1,
            },
        },
        profile,
    )
    ring_candidates = _candidates_from_plan(
        {
            "method": "tensor_ring",
            "suggested_hyperparameters": {
                "rank_candidates": [2, 4, 6],
                "init_scale": 0.1,
            },
        },
        profile,
    )
    mode3_candidates = _candidates_from_plan(
        {
            "method": "mode3",
            "suggested_hyperparameters": {
                "rank_candidates": [1, 2, 3],
                "init_scale": 0.1,
            },
        },
        profile,
    )
    nonnegative_cp_candidates = _candidates_from_plan(
        {
            "method": "nonnegative_cp",
            "suggested_hyperparameters": {
                "rank_candidates": [4, 8, 12],
                "init_scale": 0.1,
            },
        },
        profile,
    )
    btd_candidates = _candidates_from_plan(
        {
            "method": "btd",
            "suggested_hyperparameters": {
                "num_blocks_candidates": [1, 2],
                "rank_h_candidates": [4, 8],
                "rank_w_candidates": [6, 12],
                "rank_c_candidates": [2, 3],
                "init_scale": 0.1,
            },
        },
        profile,
    )
    tsvd_candidates = _candidates_from_plan(
        {
            "method": "tsvd",
            "suggested_hyperparameters": {
                "rank_candidates": [2, 4, 8],
                "init_scale": 0.1,
            },
        },
        profile,
    )
    nonnegative_candidates = _candidates_from_plan(
        {
            "method": "nonnegative_tucker",
            "suggested_hyperparameters": {
                "rank_h_candidates": [4, 8],
                "rank_w_candidates": [6, 12],
                "rank_c_candidates": [2, 3],
                "init_scale": 0.1,
            },
        },
        profile,
    )
    hierarchical_candidates = _candidates_from_plan(
        {
            "method": "hierarchical_tucker",
            "suggested_hyperparameters": {
                "rank_h_candidates": [4, 8],
                "rank_w_candidates": [6, 12],
                "rank_c_candidates": [3],
                "rank_spatial_candidates": [1, 2, 3],
                "init_scale": 0.1,
            },
        },
        profile,
    )

    assert [item["hyperparameters"]["rank_1"] for item in tt_candidates] == [4, 8]
    assert [item["hyperparameters"]["rank_2"] for item in tt_candidates] == [2, 3]
    assert [item["hyperparameters"]["rank"] for item in ring_candidates] == [2, 4, 6]
    assert [item["hyperparameters"]["rank"] for item in mode3_candidates] == [1, 2, 3]
    assert [
        item["hyperparameters"]["rank"]
        for item in nonnegative_cp_candidates
    ] == [4, 8, 12]
    assert [item["hyperparameters"]["num_blocks"] for item in btd_candidates] == [1, 2]
    assert [item["hyperparameters"]["rank"] for item in tsvd_candidates] == [2, 4, 8]
    assert all(
        set(item["hyperparameters"])
        == {"rank_h", "rank_w", "rank_c", "init_scale"}
        for item in nonnegative_candidates
    )
    assert [
        item["hyperparameters"]["rank_spatial"]
        for item in hierarchical_candidates
    ] == [1, 2, 3]


def test_day4_fallback_selects_and_trains_a_valid_method(tmp_path):
    image_path = tmp_path / "image.png"
    _write_small_image(image_path)
    state = run_day4_workflow(
        Day4WorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            image_size=None,
            missing_rate=0.3,
            seed=7,
            max_steps=12,
            validation_interval=3,
            patience=10,
            device="cpu",
            learned_metrics=False,
            llm_mode="off",
            retrieval_top_k=5,
        )
    )

    assert state["stage"] == "COMPLETED"
    plan = state["results"]["method_plan"]
    assert plan["method"] in {
        "matrix",
        "mode3",
        "cp",
        "nonnegative_cp",
        "tucker",
        "btd",
        "tsvd",
        "nonnegative_tucker",
        "hierarchical_tucker",
        "tt",
        "tensor_ring",
    }
    assert plan["selection_mode"] == "deterministic_fallback"
    assert state["selected_model"] == plan["method"]
    assert state["results"]["training"]["model_name"] == plan["method"]
    assert state["results"]["selector_diagnostics"]["llm_used"] is False

    method_plan = json.loads(Path(state["artifacts"]["method_plan"]).read_text())
    assert method_plan["ground_truth_provided_to_selector"] is False
    assert method_plan["final_metrics_provided_to_selector"] is False
    assert Path(state["artifacts"]["retrieval_result"]).is_file()

    trace_text = Path(state["artifacts"]["trace_jsonl"]).read_text()
    assert '"event": "method_selection"' in trace_text
    assert "evaluation_ground_truth" not in trace_text

    day5_state = run_day5_workflow(
        Day5WorkflowConfig(
            base_run_dir=state["artifacts"]["run_dir"],
            candidate_root=str(tmp_path / "algorithms" / "candidates"),
            output_dir=str(tmp_path / "outputs"),
            llm_mode="off",
            smoke_timeout_seconds=10,
        )
    )
    assert day5_state["stage"] == "VALIDATED"
    assert day5_state["validation"]["eligible_for_training"] is True
    manifest = json.loads(Path(day5_state["artifacts"]["manifest"]).read_text())
    assert manifest["validation_status"] == "validated"
    assert manifest["eligible_for_training"] is True

    day6_state = run_day6_workflow(
        Day6WorkflowConfig(
            base_run_dir=state["artifacts"]["run_dir"],
            initial_candidate_dir=day5_state["artifacts"]["candidate_dir"],
            candidate_root=str(tmp_path / "algorithms" / "candidates"),
            approved_root=str(tmp_path / "algorithms" / "approved"),
            output_dir=str(tmp_path / "outputs"),
            llm_mode="off",
            tuning_trials=1,
            max_steps=5,
            max_improvement_rounds=1,
            validation_interval=1,
            patience=5,
            device="cpu",
            learned_metrics=False,
        )
    )
    assert day6_state["stage"] == "COMPLETED"
    assert len(day6_state["rounds"]) == 1
    round_result = day6_state["rounds"][0]
    assert round_result["baseline_tuning"]["trial_count"] == 1
    assert round_result["candidate_tuning"]["trial_count"] == 1
    assert round_result["baseline_tuning"]["ground_truth_used"] is False
    assert round_result["candidate_tuning"]["ground_truth_used"] is False
    assert day6_state["stop_reason"] in {
        "candidate_accepted",
        "maximum_improvement_rounds_reached",
    }


@pytest.mark.parametrize(
    "model_name,expected_hyperparameters",
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
    ),
)
def test_day4_manual_selection_is_preserved_through_training(
    tmp_path,
    model_name,
    expected_hyperparameters,
):
    image_path = tmp_path / ("manual-%s.png" % model_name)
    _write_small_image(image_path)

    state = run_day4_workflow(
        Day4WorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            image_size=None,
            model_name=model_name,
            missing_rate=0.3,
            seed=17,
            max_steps=3,
            validation_interval=1,
            patience=3,
            device="cpu",
            learned_metrics=False,
            llm_mode="off",
            retrieval_top_k=5,
        )
    )

    assert state["stage"] == "COMPLETED"
    assert state["selected_model"] == model_name
    assert state["results"]["method_plan"]["selection_mode"] == "manual"
    assert state["results"]["selector_diagnostics"]["attempts"] == 0
    assert state["results"]["selector_diagnostics"]["llm_used"] is False
    assert state["results"]["training"]["model_name"] == model_name
    assert (
        set(state["results"]["training"]["hyperparameters"])
        == expected_hyperparameters
    )
