import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from research_agent.method_selector import manual_method_plan
from research_agent.workflow_day4 import (
    Day4Workflow,
    Day4WorkflowConfig,
    _candidates_from_plan,
    _siren_tuning_candidates,
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

    default_rates = {0.001, 0.01, 0.1}
    assert {
        (item["hyperparameters"]["rank_1"], item["hyperparameters"]["rank_2"])
        for item in tt_candidates
    } == {(4, 2), (8, 3)}
    assert {item["hyperparameters"]["rank"] for item in ring_candidates} == {2, 4, 6}
    assert {item["hyperparameters"]["rank"] for item in mode3_candidates} == {1, 2, 3}
    assert {
        item["hyperparameters"]["rank"] for item in nonnegative_cp_candidates
    } == {4, 8, 12}
    assert {item["hyperparameters"]["num_blocks"] for item in btd_candidates} == {1, 2}
    assert {item["hyperparameters"]["rank"] for item in tsvd_candidates} == {2, 4, 8}
    for candidates in (
        tt_candidates,
        ring_candidates,
        mode3_candidates,
        nonnegative_cp_candidates,
        btd_candidates,
        tsvd_candidates,
        nonnegative_candidates,
        hierarchical_candidates,
    ):
        assert {item["learning_rate"] for item in candidates} == default_rates
    assert all(
        set(item["hyperparameters"])
        == {"rank_h", "rank_w", "rank_c", "init_scale"}
        for item in nonnegative_candidates
    )
    assert {
        item["hyperparameters"]["rank_spatial"]
        for item in hierarchical_candidates
    } == {1, 2, 3}


def test_joint_tuning_honors_explicit_learning_rate_candidates():
    candidates = _candidates_from_plan(
        {
            "method": "cp",
            "suggested_hyperparameters": {
                "rank_candidates": [4, 8],
                "learning_rate_candidates": [0.002, 0.02],
            },
        },
        {"image_shape": [32, 32, 3]},
    )

    assert len(candidates) == 4
    assert {item["hyperparameters"]["rank"] for item in candidates} == {4, 8}
    assert {item["learning_rate"] for item in candidates} == {0.002, 0.02}


def test_siren_tuning_design_covers_width_depth_and_frequency():
    candidates = _siren_tuning_candidates(4)

    assert len(candidates) == 4
    assert {item["hyperparameters"]["hidden_features"] for item in candidates} == {
        128,
        256,
    }
    assert {item["hyperparameters"]["hidden_layers"] for item in candidates} == {
        3,
        4,
    }
    assert {item["hyperparameters"]["first_omega_0"] for item in candidates} == {
        20.0,
        30.0,
        60.0,
    }
    assert {item["learning_rate"] for item in candidates} == {5e-5, 1e-4, 3e-4}


def test_numerical_screening_can_override_the_selector_seed():
    workflow = object.__new__(Day4Workflow)
    workflow.config = SimpleNamespace(
        model_name="auto",
        method_shortlist_size=3,
        max_steps=20,
        screening_max_steps=10,
        screening_patience=2,
        screening_trials=1,
        seed=7,
        validation_interval=2,
        patience=4,
        device="cpu",
    )
    workflow.run_id = "screening-test"
    workflow.run_dir = Path("/private/tmp/screening-test")
    workflow.state = {
        "artifacts": {"corrupted": "corrupted.npy", "mask": "mask.npy"},
        "results": {
            "image_profile": {
                "image_shape": [32, 32, 3],
                "image_aspect_ratio": 1.0,
            }
        },
    }
    scores = {"tucker": 0.03, "cp": 0.01, "tsvd": 0.04}

    def fake_call(tool_name, parameters):
        assert tool_name == "tune_tensor_model"
        assert parameters["patience"] == 2
        best = {
            "best_validation_mse": scores[parameters["model_name"]],
            "hyperparameters": parameters["candidates"][0]["hyperparameters"],
            "best_step": 3,
        }
        return SimpleNamespace(data={"best": best, "trial_count": 1})

    workflow._call_tool = fake_call
    selector_plan = manual_method_plan("tucker", workflow.state["results"]["image_profile"]).model_dump()
    plan, screening = workflow._screen_method_shortlist(
        selector_plan,
        {"active_rules": [{"prefer": "cp", "weight": 2.0}], "evidence": []},
        {"status": "skipped"},
    )

    assert screening["shortlist"] == ["tucker", "cp", "tsvd"]
    assert screening["winner"] == "cp"
    assert plan["method"] == "cp"
    assert plan["selection_mode"] == "numerical_screening"


def test_llm_shortlist_is_screened_on_every_same_type_sample(tmp_path, monkeypatch):
    from research_agent import workflow_day4

    workflow = object.__new__(Day4Workflow)
    cases = [{"source": "/data/a.mat", "data_type": "Image"},
             {"source": "/data/b.mat", "data_type": "Image"}]
    workflow.config = SimpleNamespace(
        model_name="auto", method_shortlist_size=3, max_steps=20,
        screening_max_steps=8, screening_patience=2, screening_trials=1,
        seed=7, validation_interval=2, device="cpu", screening_cases=cases,
    )
    workflow.run_id = "cohort-screening"
    workflow.run_dir = tmp_path
    workflow.state = {"artifacts": {"corrupted": "corrupted.npy", "mask": "mask.npy"},
                      "results": {"image_profile": {"image_shape": [32, 32, 3],
                                                    "image_aspect_ratio": 1.0}}}
    local_losses = {"tucker": 0.01, "cp": 0.02, "hierarchical_tucker": 0.03}
    seen = []

    def fake_tune(tool_name, parameters):
        assert tool_name == "tune_tensor_model"
        assert parameters["max_steps"] == 8 and parameters["patience"] == 2
        method = parameters["model_name"]
        return SimpleNamespace(data={"best": {
            "best_validation_mse": local_losses[method],
            "hyperparameters": parameters["candidates"][0]["hyperparameters"],
            "learning_rate": parameters["candidates"][0]["learning_rate"],
            "best_step": 3}, "trial_count": 1})

    def fake_cohort(name, builder, trial, evaluated_cases, output, steps, interval, patience, device):
        seen.append((name, list(evaluated_cases), steps, interval, patience, device))
        mean = {"tucker": 18.0, "cp": 20.0, "hierarchical_tucker": 24.0}[name]
        return {"summary": {"complete": True, "expected_count": 2, "completed_count": 2,
                            "perfect_count": 0, "mean_finite_missing_psnr": mean,
                            "mean_missing_psnr": mean, "mean_composite_ssim": 0.8},
                "results": [{"source": case["source"], "status": "completed",
                             "metrics": {"missing_psnr": mean, "composite_ssim": 0.8}}
                            for case in evaluated_cases]}

    workflow._call_tool = fake_tune
    monkeypatch.setattr(workflow_day4, "evaluate_across_cases", fake_cohort)
    selector_plan = manual_method_plan("tucker", workflow.state["results"]["image_profile"]).model_dump()
    selector_plan.update(selection_mode="llm",
                         shortlist=["tucker", "cp", "hierarchical_tucker"])

    plan, screening = workflow._screen_method_shortlist(
        selector_plan, {"active_rules": [{"prefer": "tsvd", "weight": 99}], "evidence": []},
        {"status": "skipped"})

    assert screening["shortlist"] == selector_plan["shortlist"]
    assert screening["selection_scope"] == "all_valid_samples_of_modality"
    assert screening["winner"] == plan["method"] == "hierarchical_tucker"
    assert len(seen) == 3 and all(x[1:] == (cases, 8, 2, 2, "cpu") for x in seen)


def test_fallback_shortlist_keeps_strongest_measured_tensor_baseline():
    reference = {"fixed_baselines": [
        {"algorithm": "hierarchical_tucker", "summary": {
            "complete": True, "mean_missing_psnr": 31.0, "perfect_count": 0}},
        {"algorithm": "cp", "summary": {
            "complete": True, "mean_missing_psnr": 25.0, "perfect_count": 0}},
    ]}
    shortlist = Day4Workflow._shortlist_methods(
        {"method": "tucker", "selection_mode": "deterministic_fallback"},
        {"active_rules": [], "evidence": []}, {"status": "skipped"}, 3, reference)
    assert shortlist[0:2] == ["tucker", "hierarchical_tucker"]


def test_audio_shortlist_uses_lower_whole_modality_nmse(tmp_path, monkeypatch):
    from research_agent import workflow_day4

    workflow = object.__new__(Day4Workflow)
    cases = [{"source": "/data/a.wav", "data_type": "audio"},
             {"source": "/data/b.wav", "data_type": "audio"}]
    workflow.config = SimpleNamespace(
        model_name="auto", method_shortlist_size=3, max_steps=10,
        screening_max_steps=5, screening_patience=2, screening_trials=1,
        seed=1, validation_interval=1, device="cpu", screening_cases=cases)
    workflow.run_id = "audio-screening"
    workflow.run_dir = tmp_path
    workflow.state = {"artifacts": {"corrupted": "x.npy", "mask": "m.npy"},
                      "results": {"image_profile": {"image_shape": [8, 8, 1],
                                                    "image_aspect_ratio": 1.0}}}
    workflow._call_tool = lambda _name, params: SimpleNamespace(data={
        "best": {"best_validation_mse": 0.01,
                 "hyperparameters": params["candidates"][0]["hyperparameters"],
                 "learning_rate": params["candidates"][0]["learning_rate"]},
        "trial_count": 1})

    def fake_cohort(name, _builder, _trial, evaluated_cases, *_args):
        mean = {"tucker": 0.4, "cp": 0.2, "hierarchical_tucker": 0.3}[name]
        return {"summary": {"complete": True, "expected_count": 2,
                            "completed_count": 2, "mean_missing_nmse": mean},
                "results": [{"source": case["source"], "status": "completed",
                             "metrics": {"missing_nmse": mean}}
                            for case in evaluated_cases]}

    monkeypatch.setattr(workflow_day4, "evaluate_across_cases", fake_cohort)
    selector_plan = manual_method_plan("tucker", workflow.state["results"]["image_profile"]).model_dump()
    selector_plan.update(selection_mode="llm",
                         shortlist=["tucker", "cp", "hierarchical_tucker"])
    plan, screening = workflow._screen_method_shortlist(
        selector_plan, {"active_rules": [], "evidence": []}, {"status": "skipped"})

    assert plan["method"] == screening["winner"] == "cp"
    assert screening["selection_metric"] == "mean_missing_nmse"


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
            max_steps_ceiling=12,
            screening_max_steps=12,
            screening_patience=2,
            validation_interval=3,
            patience=10,
            siren_max_steps=12,
            siren_tuning_trials=1,
            siren_validation_interval=3,
            siren_patience=2,
            device="cpu",
            full_reference_metrics=False,
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
    assert plan["selection_mode"] == "numerical_screening"
    screening = state["results"]["method_screening"]
    assert screening["status"] == "completed"
    assert len(screening["shortlist"]) == 3
    assert screening["winner"] == plan["method"]
    assert screening["selection_scope"] == "missing_region_ground_truth"
    assert screening["ground_truth_used"] is True
    method_plan = json.loads(Path(state["artifacts"]["method_plan"]).read_text())
    assert method_plan["selector_recommendation"]["selection_mode"] == (
        "deterministic_fallback"
    )
    assert state["selected_model"] == plan["method"]
    assert state["results"]["training"]["model_name"] == plan["method"]
    assert state["results"]["selector_diagnostics"]["llm_used"] is False

    assert method_plan["ground_truth_provided_to_selector"] is False
    assert method_plan["final_metrics_provided_to_selector"] is False
    assert Path(state["artifacts"]["retrieval_result"]).is_file()
    assert Path(state["artifacts"]["method_screening"]).is_file()
    assert state["results"]["siren_comparison"]["status"] == "completed"
    assert state["results"]["baseline_comparison"]["winner"] in {
        "nearest_neighbor_manhattan",
        "siren",
        state["selected_model"],
    }


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
            learning_rate_candidates=(0.03,),
            refine_learning_rate=False,
            max_steps=5,
            max_improvement_rounds=1,
            validation_interval=1,
            patience=5,
            device="cpu",
            full_reference_metrics=False,
        )
    )
    assert day6_state["stage"] == "COMPLETED"
    assert len(day6_state["rounds"]) == 1
    round_result = day6_state["rounds"][0]
    assert round_result["baseline_tuning"]["trial_count"] == 1
    assert round_result["candidate_tuning"]["trial_count"] == 1
    assert round_result["baseline_tuning"]["ground_truth_used"] is True
    assert round_result["candidate_tuning"]["ground_truth_used"] is True
    assert day6_state["stop_reason"] in {
        "candidate_accepted",
        "maximum_improvement_rounds_reached",
    }


def test_siren_training_is_reused_for_the_same_data_and_config(tmp_path, monkeypatch):
    corrupted_path = tmp_path / "corrupted.npy"
    mask_path = tmp_path / "mask.npy"
    ground_truth_path = tmp_path / "ground_truth.npy"
    np.save(corrupted_path, np.zeros((4, 5, 3), dtype=np.float32))
    np.save(mask_path, np.ones((4, 5), dtype=bool))
    np.save(ground_truth_path, np.zeros((4, 5, 3), dtype=np.float32))
    config = SimpleNamespace(
        output_dir=str(tmp_path / "outputs"), siren_comparison=True, seed=7,
        siren_tuning_trials=1, siren_max_steps=12, siren_validation_interval=3,
        siren_patience=2, device="cpu", full_reference_metrics=False,
        no_reference_metrics=False, data_type="Image", siren_baseline_reference=None,
        siren_learning_rate_candidates=(5e-5, 1e-4, 3e-4),
    )
    calls = []

    def make_workflow(run_name):
        workflow = object.__new__(Day4Workflow)
        workflow.config = config
        workflow.run_id = run_name
        workflow.run_dir = tmp_path / run_name
        workflow.run_dir.mkdir()
        workflow.state = {
            "selected_model": "tucker",
            "artifacts": {"corrupted": str(corrupted_path), "mask": str(mask_path)},
            "results": {
                "image_profile": {"image_shape": [4, 5, 3]},
                "interpolation_metrics": {"missing_psnr": 10.0},
                "tensor_metrics": {"missing_psnr": 11.0},
            },
        }

        def fake_call(tool_name, parameters):
            calls.append((run_name, tool_name))
            if tool_name == "tune_tensor_model":
                Path(parameters["output_path"]).parent.mkdir(parents=True, exist_ok=True)
                Path(parameters["output_path"]).write_text("{}", encoding="utf-8")
                model_dir = Path(parameters["output_path"]).parent / "selected_model"
                model_dir.mkdir(parents=True, exist_ok=True)
                reconstruction = model_dir / "reconstruction.npy"
                preview = model_dir / "preview.png"
                np.save(reconstruction, np.zeros((4, 5, 3), dtype=np.float32))
                preview.write_bytes(b"preview")
                selected_output = {
                    "model_name": "siren",
                    "hyperparameters": {"hidden_features": 16},
                    "learning_rate": 0.001, "fitted_steps": 3,
                    "final_train_mse": 0.1, "observed_pixels_used": 20,
                    "parameter_count": 100, "runtime_seconds": 0.2,
                    "device": "cpu",
                    "artifacts": {"reconstruction": str(reconstruction),
                                  "preview": str(preview)},
                }
                return SimpleNamespace(data={
                    "best": {
                        "hyperparameters": {"hidden_features": 16},
                        "learning_rate": 0.001, "best_step": 3,
                    },
                    "selected_output": selected_output,
                })
            if tool_name == "evaluate_reconstruction":
                Path(parameters["output_path"]).parent.mkdir(parents=True, exist_ok=True)
                Path(parameters["output_path"]).write_text("{}", encoding="utf-8")
                return SimpleNamespace(data={
                    "missing_mse": 0.1, "missing_psnr": 12.0,
                    "full_psnr": 14.0, "perfect_reconstruction": False,
                    "composite_ssim": 0.8, "lpips": None, "maniqa": None,
                    "clip_iqa": None, "musiq": None,
                    "learned_metric_status": {}, "metric_group_status": {},
                })
            raise AssertionError("Unexpected tool: %s" % tool_name)

        workflow._call_tool = fake_call
        return workflow

    first = make_workflow("run-first")
    first._run_additional_baselines(str(ground_truth_path))
    second = make_workflow("run-second")
    second._run_additional_baselines(str(ground_truth_path))

    assert first.state["results"]["siren_comparison"]["training_cache"]["reused"] is False
    assert second.state["results"]["siren_comparison"]["training_cache"]["reused"] is True
    assert [tool for run, tool in calls if run == "run-first"] == [
        "tune_tensor_model", "evaluate_reconstruction",
    ]
    assert [tool for run, tool in calls if run == "run-second"] == [
        "evaluate_reconstruction"
    ]
    assert Path(second.state["artifacts"]["siren_reconstruction"]).is_file()
    original_hash = Day4Workflow._file_sha256
    monkeypatch.setattr(Day4Workflow, "_file_sha256", staticmethod(
        lambda path: "changed-trainer" if path.endswith("core/trainer.py") else original_hash(path)))
    third = make_workflow("run-third")
    third._run_additional_baselines(str(ground_truth_path))
    assert not third.state["results"]["siren_comparison"]["training_cache"]["reused"]
    assert [tool for run, tool in calls if run == "run-third"] == [
        "tune_tensor_model", "evaluate_reconstruction"]


def test_day4_can_feed_one_shot_interpolation_visual_prior_to_selector(tmp_path):
    class RecordingVisualEvaluator:
        def __init__(self):
            self.calls = []

        def assess_method_selection(self, interpolation_path, tensor_context):
            self.calls.append((interpolation_path, tensor_context))
            return {
                "status": "completed",
                "scope": "manhattan_interpolation_preview_only",
                "ground_truth_provided": False,
                "assessment": {
                    "preferred_methods": ["tucker"],
                    "rank_regime": {
                        "height": "medium",
                        "width": "high",
                        "feature": "low",
                        "overall": "medium",
                    },
                },
            }

    class RecordingSelector:
        llm = object()

        def __init__(self):
            self.visual_assessment = None

        def select(self, profile, retrieval, visual_assessment=None, shortlist_size=None,
                   comparison_reference=None):
            self.visual_assessment = visual_assessment
            return {
                "plan": manual_method_plan("tucker", profile),
                "attempts": 1,
                "fallback_reason": None,
                "validation_errors": [],
                "raw_outputs": [],
            }

    image_path = tmp_path / "selection-visual.png"
    _write_small_image(image_path)
    selector = RecordingSelector()
    visual_evaluator = RecordingVisualEvaluator()
    state = run_day4_workflow(
        Day4WorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            image_size=None,
            missing_rate=0.3,
            seed=13,
            max_steps=3,
            max_steps_ceiling=3,
            validation_interval=1,
            patience=3,
            device="cpu",
            full_reference_metrics=False,
            llm_mode="off",
            selection_visual_assessment=True,
        ),
        selector=selector,
        visual_evaluator=visual_evaluator,
    )

    assert state["stage"] == "COMPLETED"
    assert len(visual_evaluator.calls) == 1
    assert visual_evaluator.calls[0][0].endswith("interpolated_preview.png")
    assert selector.visual_assessment["status"] == "completed"
    assert state["results"]["selector_diagnostics"][
        "selection_visual_assessment_status"
    ] == "completed"
    assert Path(
        state["artifacts"]["method_selection_visual_assessment"]
    ).is_file()


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
            max_steps_ceiling=3,
            validation_interval=1,
            patience=3,
            device="cpu",
            full_reference_metrics=False,
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
