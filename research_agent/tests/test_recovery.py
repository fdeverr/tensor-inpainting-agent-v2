import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat, wavfile

from research_agent.core.data import apply_observation_mask, load_tensor_prediction
from research_agent.core.interpolation import nearest_neighbor_fill
from research_agent.core.masks import generate_tensor_mask
from research_agent.core.metrics import evaluate_reconstruction_metrics
from research_agent.core.trainer import observed_feature_mean, train_tensor_model
from research_agent.evolution_knowledge import GlobalExperienceStore
from research_agent.recovery import RecoveryConfig, run_recovery, summarize_evaluations, _report
from research_agent.recovery_data import export_audio, load_recovery_gt, prepare_recovery_case
from research_agent.recovery_registry import champion_builder, champion_records
from research_agent.schemas import TrainingConfig


@pytest.mark.parametrize("shape,axis", [((8, 9, 3), 0), ((8, 9, 6), 2), ((8, 9, 4, 3), 2)])
def test_slice_masks_hide_whole_axis_entries_and_alias(shape, axis):
    mask = generate_tensor_mask(shape, 0.4, "slices", 7, axis)
    assert np.array_equal(mask, generate_tensor_mask(shape, 0.4, "sildes", 7, axis))
    moved = np.moveaxis(mask, axis, 0).reshape(shape[axis], -1)
    assert np.all(moved.all(axis=1) | (~moved).all(axis=1))
    assert mask.any() and not mask.all()


@pytest.mark.parametrize("pattern", ["random", "block"])
def test_tensor_mask_exact_rate_and_repeatability(pattern):
    mask = generate_tensor_mask((8, 10, 3), 0.4, pattern, 12)
    assert (~mask).sum() == 96
    assert np.array_equal(mask, generate_tensor_mask(mask.shape, 0.4, pattern, 12))


def test_dataset_gt_ignores_old_missing_tensor_and_converts_video_axes(tmp_path):
    data = np.arange(8 * 9 * 3 * 4, dtype=np.float32).reshape(8, 9, 3, 4)
    data /= data.max()
    source = tmp_path / "video.mat"
    savemat(source, {"Ohsi": data, "Nhsi": np.zeros_like(data), "mask": np.zeros_like(data)})
    loaded, meta = load_recovery_gt(str(source))
    assert loaded.shape == (8, 9, 4, 3)
    assert np.allclose(loaded, data.transpose(0, 1, 3, 2))
    case = prepare_recovery_case(str(source), tmp_path / "case", 0.5, "slices", 3)
    assert case["actual_missing_rate"] == 0.5
    assert case["data_type"] == "Video"


def test_audio_tensorization_roundtrip_and_padding_not_missing(tmp_path):
    original = np.linspace(-0.8, 0.6, 37, dtype=np.float32)
    source = tmp_path / "audio.wav"
    wavfile.write(source, 16000, original)
    data, meta = load_recovery_gt(str(source), audio_frame_size=8)
    assert data.shape == (5, 8, 1)
    output = tmp_path / "output.wav"
    export_audio(data, meta, str(output))
    rate, recovered = wavfile.read(output)
    assert rate == 16000
    assert np.allclose(original, recovered, atol=1e-6)
    case = prepare_recovery_case(str(source), tmp_path / "case", 0.4, "sildes", 4, audio_frame_size=8)
    mask = np.load(case["mask_path"]).reshape(-1)
    hidden = np.flatnonzero(~mask)
    assert np.all(np.diff(hidden) == 1)
    assert hidden.max() < len(original)


def test_element_mask_training_interpolation_and_metrics(tmp_path):
    gt = np.random.default_rng(9).random((8, 9, 4), dtype=np.float32)
    mask = generate_tensor_mask(gt.shape, 0.5, "slices", 4, 2)
    observed = apply_observation_mask(gt, mask)
    mean = observed_feature_mean(observed, mask)
    assert np.isfinite(mean).all()
    prediction = nearest_neighbor_fill(observed, mask)
    assert np.array_equal(prediction[mask], gt[mask])
    metrics = evaluate_reconstruction_metrics(prediction, gt, mask)
    assert metrics["missing_mse"] == pytest.approx(np.square(prediction[~mask] - gt[~mask]).mean())
    output = train_tensor_model(
        "tucker", {"rank_h": 2, "rank_w": 2, "rank_c": 2, "init_scale": 0.1},
        observed, mask, TrainingConfig(max_steps=2, validation_interval=1, device="cpu"),
        seed=7, ground_truth=gt)
    assert output.train_mask.shape == gt.shape
    assert output.reconstruction.shape == gt.shape


def test_experience_partition_retains_pattern_rate_and_run(tmp_path):
    image = GlobalExperienceStore(tmp_path, "tucker", "color_image", {
        "mask_type": "slices", "actual_missing_rate": 0.5, "source_run_id": "run-1"})
    msi = GlobalExperienceStore(tmp_path, "tucker", "msi", {
        "mask_type": "random", "actual_missing_rate": 0.4, "source_run_id": "run-2"})
    lesson = {"experience": "A shared spectral prior may help missing bands.", "confidence": "low"}
    image.record(lesson)
    msi.record(lesson)
    record = msi.context()["reusable_experience"][0]
    assert record["mask_type"] == "random"
    assert record["actual_missing_rate"] == 0.4
    assert record["source_run_id"] == "run-2"
    assert image.directory != msi.directory
    assert (tmp_path / "MSI/tucker/reusable_experience.jsonl").is_file()


def test_failed_sample_cannot_yield_complete_average():
    result = summarize_evaluations([
        {"status": "completed", "result": {"metrics": {"missing_psnr": 20, "missing_mse": 0.01, "composite_ssim": 0.8}}},
        {"status": "failed"}], 2)
    assert result["complete"] is False
    assert result["completed_count"] == 1
    assert result["mean_missing_psnr"] is None


def test_perfect_cohort_ties_preserve_finite_scores_and_consistent_report_order():
    from research_agent.dataset_evolution import cohort_score
    from research_agent.recovery import _comparison_matrix
    perfect = {"missing_psnr": None, "missing_mse": 0, "composite_ssim": 1.0}
    def comparison(name, score, ssim):
        metrics = [perfect, {"missing_psnr": score, "missing_mse": 10 ** (-score / 10), "composite_ssim": ssim}]
        results = [{"source": "/data/%d.mat" % index, "status": "completed", "result": {"metrics": metric}}
                   for index, metric in enumerate(metrics)]
        return {"algorithm": name, "results": results, "summary": summarize_evaluations(results, 2, "Image")}
    lower = comparison("a_lower_finite", 10.0, 0.9)
    higher = comparison("z_higher_finite", 30.0, 0.8)
    assert higher["summary"]["mean_missing_psnr"] is None
    assert higher["summary"]["mean_finite_missing_psnr"] == 30.0
    assert cohort_score(higher["summary"]) > cohort_score(lower["summary"])
    report = "\n".join(_comparison_matrix("Image", {"comparisons": [lower, higher]}))
    assert report.index("z_higher_finite") < report.index("a_lower_finite")
    assert "打破平局" in report
    assert report.count("<strong>∞</strong>") == 4  # Both perfect sample cells and both means.


def test_null_psnr_without_perfect_reconstruction_is_not_complete():
    metrics = {"missing_psnr": None, "missing_mse": 0.1, "composite_ssim": 0.8}
    assert not summarize_evaluations([{"status": "completed", "result": {"metrics": metrics}}], 1, "Image")["complete"]


def test_saved_predictions_are_not_renormalized(tmp_path):
    prediction = np.linspace(-0.2, 1.3, 8 * 9 * 3, dtype=np.float32).reshape(8, 9, 3)
    path = tmp_path / "prediction.npy"
    np.save(path, prediction)
    assert np.array_equal(load_tensor_prediction(str(path)), prediction)


def test_equivalent_versions_reuse_only_identical_current_protocol(tmp_path, monkeypatch):
    from research_agent import recovery
    calls = []
    monkeypatch.setattr(recovery, "champion_builder", lambda record: None)
    def evaluate(*args):
        calls.append(args)
        return {"metrics": {"missing_psnr": 20}}
    monkeypatch.setattr(recovery, "evaluate_champion", evaluate)
    record = {"archive_id": "v1", "kind": "interpolation", "algorithm": "nearest", "config": {},
              "code_sha256": "code", "model_library_sha256": "library"}
    case = {"gt_sha256": "gt", "mask_sha256": "mask", "seed": 42}
    config = RecoveryConfig(dataset_root=str(tmp_path))
    cache = {}
    recovery._evaluate_cached(record, case, tmp_path, config, cache)
    reused = recovery._evaluate_cached({**record, "archive_id": "v2"}, case, tmp_path, config, cache)
    assert len(calls) == 1
    assert reused["equivalent_algorithm_evaluation_reused_from"] == "v1"
    recovery._evaluate_cached(record, {**case, "mask_sha256": "different"}, tmp_path, config, cache)
    assert len(calls) == 2
    recovery._evaluate_cached(record, case, tmp_path, replace(config, evaluation_patience=5), cache)
    recovery._evaluate_cached(record, case, tmp_path, replace(config, evaluation_validation_interval=3), cache)
    assert len(calls) == 4
    recovery._evaluate_cached(record, case, tmp_path, replace(config, no_reference_metrics=True), cache)
    assert len(calls) == 5


@pytest.mark.parametrize("field,value", [
    ("patience", 0), ("validation_interval", 0), ("evaluation_patience", -1),
    ("evaluation_validation_interval", 0), ("screening_patience", 0),
    ("screening_max_steps", 0), ("method_max_steps_ceiling", 100),
])
def test_invalid_recovery_training_settings(tmp_path, field, value):
    with pytest.raises(ValueError):
        RecoveryConfig(dataset_root=str(tmp_path), **{field: value}).validate()


def test_development_training_settings_reach_full_workflow(tmp_path, monkeypatch):
    from research_agent import recovery
    monkeypatch.setattr(recovery, "evaluate_fixed_baseline", lambda *args, **kwargs: {
        "metrics": {"missing_psnr": 20.0, "missing_mse": 0.01, "composite_ssim": 0.8}})
    root = tmp_path / "dataset"
    root.mkdir()
    savemat(root / "sample.mat", {"Ohsi": np.random.default_rng(1).random((8, 8, 3))})
    captured = []

    def capture(config):
        captured.append(config)
        raise RuntimeError("stop before any training")

    monkeypatch.setattr(recovery, "run_full_workflow", capture)
    config = RecoveryConfig(
        dataset_root=str(root), output_dir=str(tmp_path / "outputs"),
        history_root=str(tmp_path / "history"), data_types=("Image",),
        evolution_steps=50, validation_interval=3, patience=7,
        method_max_steps_ceiling=80, fair_max_steps=60,
        tuning_near_limit_ratio=0.8, tuning_expansion_factor=1.5,
        fair_learning_rate_candidates=(0.01,), fair_refine_learning_rate=False,
        ablation_screen_trials=2, ablation_screen_max_steps=9,
        method_shortlist_size=5, screening_trials=3, screening_max_steps=12, screening_patience=4,
        siren_max_steps=70, siren_tuning_trials=2, siren_validation_interval=6, siren_patience=8,
        retrieval_top_k=3, minimum_psnr_delta=0.1, ssim_tolerance=0.001, smoke_timeout_seconds=4,
        lpips=True, no_reference_metrics=True, selection_visual_assessment=True,
        mutation_visual_assessment=True, prompt="使用多尺度注意力",
    )
    state = run_recovery(config)
    assert state["stage"] == "COMPLETED_WITH_FAILURES"
    assert len(captured) == 1
    forwarded = captured[0]
    assert forwarded.method_max_steps == 50
    for name in ("validation_interval", "patience", "method_max_steps_ceiling", "fair_max_steps",
                 "tuning_near_limit_ratio", "tuning_expansion_factor", "fair_learning_rate_candidates",
                 "fair_refine_learning_rate", "ablation_screen_trials", "ablation_screen_max_steps",
                 "method_shortlist_size", "screening_trials", "screening_max_steps", "screening_patience",
                 "siren_max_steps", "siren_tuning_trials", "siren_validation_interval", "siren_patience",
                 "retrieval_top_k", "minimum_psnr_delta", "ssim_tolerance", "smoke_timeout_seconds"):
        assert getattr(forwarded, name) == getattr(config, name)
    assert forwarded.full_reference_metrics and forwarded.no_reference_metrics
    assert forwarded.selection_visual_assessment and forwarded.mutation_visual_assessment
    assert len(forwarded.evolution_cases) == 1
    assert forwarded.dataset_algorithm_reference["fixed_baselines"]
    assert forwarded.dataset_evaluation_steps == config.evaluation_steps
    assert forwarded.dataset_evaluation_validation_interval == config.evaluation_validation_interval
    assert forwarded.dataset_evaluation_patience == config.evaluation_patience
    assert config.prompt in forwarded.prompt


@pytest.mark.parametrize("patience,expected", [(0, 51), (7, 7)])
@pytest.mark.parametrize("data_type", ["Image", "MSI"])
def test_dataset_evaluation_training_settings_are_effective(tmp_path, monkeypatch, patience, expected, data_type):
    from research_agent import recovery
    source = tmp_path / "sample.mat"
    savemat(source, {"Ohsi": np.random.default_rng(1).random((8, 8, 3 if data_type == "Image" else 5))})
    case = prepare_recovery_case(str(source), tmp_path / "case", 0.4, "random", 1)
    captured = []
    monkeypatch.setattr(recovery, "champion_builder", lambda record: None)

    def capture(**kwargs):
        captured.append(kwargs)
        raise RuntimeError("stop before training")

    monkeypatch.setattr(recovery, "final_fit_and_evaluate", capture)
    config = RecoveryConfig(dataset_root=str(tmp_path), evaluation_steps=50,
                            evaluation_validation_interval=4, evaluation_patience=patience,
                            lpips=True, no_reference_metrics=True)
    record = {"kind": "builtin", "algorithm": "tucker",
              "config": {"hyperparameters": {}, "learning_rate": 0.02}}
    with pytest.raises(RuntimeError, match="stop before training"):
        recovery.evaluate_champion(record, case, tmp_path / "result", config)
    training = captured[0]["training_config"]
    assert training.max_steps == 50 and training.validation_interval == 4
    assert training.early_stopping_patience == expected
    assert training.learning_rate == 0.02  # Evaluation keeps the archived learning rate.
    assert captured[0]["include_full_reference_metrics"] == (data_type == "Image")
    assert captured[0]["include_no_reference_metrics"] == (data_type == "Image")


@pytest.mark.parametrize("data_type", ["Image", "MSI", "Video", "audio"])
def test_one_evolution_per_type_and_comparison_with_previous_winner(tmp_path, monkeypatch, data_type):
    from research_agent import recovery
    from research_agent.workflow_full import run_full_workflow as actual_evolution
    root = tmp_path / "dataset"
    root.mkdir()
    for index in range(2):
        if data_type == "audio":
            waveform = np.sin(np.arange(128) / (5 + index)).astype(np.float32)
            wavfile.write(root / ("audio%d.wav" % index), 16000, waveform)
        else:
            shape = (12, 16, 3, 4) if data_type == "Video" else (12, 16, 5 if data_type == "MSI" else 3)
            data = np.random.default_rng(index).random(shape, dtype=np.float32)
            savemat(root / ("sample%d.mat" % index), {"Ohsi": data})
    calls = []

    # Keep the integration test cheap while exercising the upfront comparator loop.
    events = []
    metrics = ({"missing_nmse": 0.4} if data_type == "audio" else
               {"missing_psnr": 18.0, "missing_mse": 0.02, "composite_ssim": 0.7})
    def fixed_reference(*args, **kwargs):
        events.append("fixed_reference")
        return {"metrics": metrics}
    monkeypatch.setattr(recovery, "evaluate_fixed_baseline", fixed_reference)

    def tracked(config):
        assert events and events[-1] == "fixed_reference"
        events.append("evolution")
        calls.append(config)
        return actual_evolution(replace(config, fair_learning_rate_candidates=(0.01,),
                                       fair_refine_learning_rate=False, siren_comparison=False))

    monkeypatch.setattr(recovery, "run_full_workflow", tracked)
    config = RecoveryConfig(
        dataset_root=str(root), output_dir=str(tmp_path / "outputs"),
        candidate_root=str(tmp_path / "candidates"), approved_root=str(tmp_path / "approved"),
        history_root=str(tmp_path / "history"), knowledge_root=str(tmp_path / "knowledge"),
        data_types=(data_type,), image_size=None, audio_frame_size=8, base_model="tucker", mask_type="slices",
        evolution_steps=2, evaluation_steps=2, improvement_rounds=1, tuning_trials=1,
        llm_mode="off", device="cpu", siren_comparison=data_type in {"Video", "audio"})
    first = run_recovery(config)
    second = run_recovery(config)
    assert len(calls) == 2  # Not four full workflows for two runs x two images.
    assert first["stage"] == second["stage"] == "COMPLETED"
    group = second["modalities"][data_type]
    assert len(group["comparisons"]) == 2
    assert len(group["historical_references"]) == 1
    assert len(calls[-1].historical_algorithm_reference) == 1
    assert first["modalities"][data_type]["representative"] != group["representative"]
    assert all(len(x["results"]) == 2 for x in group["comparisons"])
    assert len(group["evolution_rounds"]) == 1
    round_evaluation = group["evolution_rounds"][0]["dataset_evaluation"]
    assert len(round_evaluation["candidate"]["results"]) == 2
    assert len(round_evaluation["incumbent"]["results"]) == 2
    assert "每轮进化：整类评测与反馈" in Path(second["report"]).read_text(encoding="utf-8")
    full_state = json.loads(Path(group["development_report"]).with_name("state.json").read_text(encoding="utf-8"))
    day5_state = json.loads(Path(full_state["artifacts"]["day5_state"]).read_text(encoding="utf-8"))
    day6_state = json.loads(Path(full_state["artifacts"]["day6_state"]).read_text(encoding="utf-8"))
    global_context = json.loads(Path(day5_state["artifacts"]["global_experience_context"]).read_text(encoding="utf-8"))
    assert global_context["record_count"] == 1  # The preceding run, not this run's partial rounds.
    assert len(global_context["reusable_experience"]) == 1
    assert global_context["reusable_experience"][0]["evaluation_scope"] == "all_same_type_samples"
    assert global_context["reusable_experience"][0]["sample_count"] == 2
    assert full_state["global_experience"]["status"] == "committed"
    global_path = Path(full_state["artifacts"]["global_experience"]["experience_jsonl"])
    global_records = [json.loads(line) for line in global_path.read_text().splitlines()]
    assert len(global_records) == 2
    assert all(record["summary_scope"] == "whole_run_practice" for record in global_records)
    if data_type == "audio":
        assert all(record["metric_protocol"] == "missing_original_waveform_nmse" for record in global_records)
        assert all("PSNR" not in record["experience"] and "SSIM" not in record["experience"] for record in global_records)
    assert day5_state["algorithm_comparison_reference"]["whole_modality_algorithms"]["fixed_baselines"]
    assert day6_state["algorithm_comparison_reference"]["whole_modality_algorithms"]["fixed_baselines"]
    historical_structure = group["pre_evolution_reference"]["historical_champions"][0]["structure_reference"]
    assert historical_structure["status"] == "completed"
    assert not historical_structure["full_source_included"]  # Deferred until Day 4 selects a framework.
    assert "source_bundle" not in historical_structure
    assert "forward_and_loss_implementations" not in historical_structure
    assert calls[-1].historical_algorithm_reference[0]["structure_reference"]["structure"]
    assert "source_bundle" not in calls[-1].historical_algorithm_reference[0]["structure_reference"]
    selected_history = day5_state["algorithm_comparison_reference"]["whole_modality_algorithms"]["historical_champions"][0]["structure_reference"]
    assert selected_history["full_source_included"]
    assert selected_history["source_bundle"]
    assert selected_history["matches_current_tensor_framework"]
    assert day6_state["algorithm_comparison_reference"]["whole_modality_algorithms"]["historical_champions"][0]["structure_reference"] == selected_history
    first_round_reference = day5_state["algorithm_comparison_reference"]["whole_modality_algorithms"]
    assert len(first_round_reference["selected_incumbent_before_evolution"]["samples"]) == 2
    assert all("training" in item for item in first_round_reference["selected_incumbent_before_evolution"]["samples"])
    interpolation = "linear_interpolation_waveform" if data_type == "audio" else "nearest_neighbor_manhattan"
    assert [x["algorithm"] for x in group["baseline_comparisons"]] == [
        interpolation, *(["siren"] if data_type in {"Video", "audio"} else [])]
    assert all(x["summary"]["complete"] for x in group["baseline_comparisons"])
    assert group["pre_evolution_reference"]["fixed_baselines"]
    assert len(calls[-1].dataset_algorithm_reference["best_reference_by_sample"]) == 2
    assert len(calls[-1].dataset_algorithm_reference["fixed_baselines"][0]["samples"]) == 2
    report = Path(second["report"]).read_text(encoding="utf-8")
    sample_names = ("audio0.wav", "audio1.wav") if data_type == "audio" else ("sample0.mat", "sample1.mat")
    assert all(name in report for name in sample_names)
    assert interpolation in report and "tucker" in report
    assert "同条件全类对照最优" in report
    assert "正式固定对照仅含" in report
    if data_type == "audio":
        assert group["selection_metric"] == "mean_missing_nmse"
        best = min(group["comparisons"], key=lambda item: item["summary"]["mean_missing_nmse"])
        assert group["dataset_best_archive_id"] == best["archive_id"]
        assert all("mean_missing_psnr" not in item["summary"] for item in group["baseline_comparisons"])
        assert set(group["historical_references"][0]["metrics"]) == {"missing_nmse"}
        for comparison in group["comparisons"]:
            assert "mean_missing_psnr" not in comparison["summary"]
            for evaluated in comparison["results"]:
                metric = evaluated["result"]["metrics"]
                assert "missing_nmse" in metric and "missing_psnr" not in metric and "composite_ssim" not in metric
        assert "Missing waveform NMSE" in Path(group["development_report"]).read_text()
    records = champion_records(config.history_root, data_type)
    assert len(records) == 2
    assert Path(records[0]["archive_dir"], "development_report.md").is_file()
    assert Path(records[0]["archive_dir"], "structure.json").is_file()
    if data_type == "Image":
        evaluation = run_recovery(config, evaluate_only=True)
        assert evaluation["stage"] == "COMPLETED"
        assert len(calls) == 2
        assert len(champion_records(config.history_root, data_type)) == 2
        assert Path(config.history_root, data_type, "latest_evaluation.json").is_file()
        assert len(global_path.read_text().splitlines()) == 2  # Evaluation-only must not append.
    if records[0]["kind"] != "interpolation":
        path = Path(records[0]["archive_dir"], "model.py")
        path.write_text(path.read_text() + "\n# tampered\n")
        with pytest.raises(ValueError, match="hash mismatch"):
            champion_builder(records[0])


def test_report_lists_each_sample_and_failed_evaluation():
    results = [
        {"source": "/data/bridge.mat", "is_development": True, "status": "completed",
         "result": {"metrics": {"missing_psnr": 21.0, "missing_mse": 0.01, "composite_ssim": 0.8}}},
        {"source": "/data/traffic.mat", "is_development": False, "status": "failed",
         "error": "RuntimeError: out of memory"},
    ]
    comparison = {"archive_id": "v1", "algorithm": "tucker", "results": results,
                  "summary": summarize_evaluations(results, 2, "Video"),
                  "non_development_summary": summarize_evaluations(results[1:], 1, "Video")}
    report = _report({"modalities": {"Video": {"status": "completed_with_evaluation_failures",
        "representative": "/data/bridge.mat", "current_champion": {"algorithm": "tucker"},
        "comparisons": [comparison], "baseline_comparisons": []}}, "inventory": {"failures": []}})
    assert "bridge.mat" in report and "traffic.mat" in report
    assert "21.000" in report and "out of memory" in report


def test_paper_style_report_ranks_each_metric_and_hides_incomplete_mean():
    def comparison(name, psnrs, ssims, archive_id=None):
        results = []
        for index, (psnr, ssim) in enumerate(zip(psnrs, ssims)):
            source = "/data/sample%d.mat" % index
            results.append({"source": source, "is_development": index == 0,
                            "status": "completed" if psnr is not None else "failed",
                            **({"result": {"metrics": {"missing_psnr": psnr, "missing_mse": 0.01,
                                                      "composite_ssim": ssim}}} if psnr is not None else
                               {"error": "failed fit"})})
        return {"algorithm": name, **({"archive_id": archive_id} if archive_id else {}),
                "results": results, "summary": summarize_evaluations(results, 2, "Image")}

    incumbent = comparison("evolved", [30.0, 20.0], [0.8, 0.7], "v2")
    baseline = comparison("baseline", [28.0, 24.0], [0.9, 0.6])
    incomplete = comparison("broken", [40.0, None], [0.5, None])
    report = _report({"modalities": {"Image": {
        "status": "completed_with_evaluation_failures", "representative": "/data/sample0.mat",
        "current_champion": {"algorithm": "evolved", "archive_id": "v2"},
        "comparisons": [incumbent], "baseline_comparisons": [baseline, incomplete],
        "method_selection": {"status": "completed", "recommendation": "tucker",
                             "recommendation_mode": "llm",
                             "recommended_methods": ["tucker", "cp", "hierarchical_tucker"],
                             "shortlist": ["tucker", "cp", "hierarchical_tucker"],
                             "winner": "cp", "results": [{"method": "cp", "status": "completed",
                                 "dataset_evaluation": {"summary": {"complete": True,
                                     "mean_missing_psnr": 22.0, "mean_composite_ssim": 0.81,
                                     "perfect_count": 0}, "results": [
                                         {"source": "/data/sample0.mat", "status": "completed",
                                          "metrics": {"missing_psnr": 21.0, "composite_ssim": 0.8}},
                                         {"source": "/data/sample1.mat", "status": "completed",
                                          "metrics": {"missing_psnr": 23.0, "composite_ssim": 0.82}},
                                     ]}}]},
    }}, "inventory": {"failures": []}})
    assert '<th colspan="2">evolved<br><small>v2 · 本轮进化</small></th>' in report
    assert '<th>PSNR ↑</th>' in report and '<th>SSIM ↑</th>' in report
    assert '<strong>40.000</strong>' in report  # per-sample best, despite incomplete method
    assert '<u>30.000</u>' in report
    assert '<strong>0.900</strong>' in report  # SSIM is ranked independently
    assert '<th>全类平均</th>' in report
    assert '<strong>27.000</strong>' not in report  # partial mean is never presented
    assert '失败' in report and 'failed fit' in report
    assert 'LLM 推荐的方法：`tucker, cp, hierarchical_tucker`' in report
    assert '同预算数值预赛胜出：`cp`' in report
    assert '基础分解轻量预赛（整类样本）' in report
    assert '<small>预赛候选</small>' in report
    assert '不是最高的单样本分数' in report
    assert report.index('baseline<br><small>固定对照</small>') < report.index(
        'evolved<br><small>v2 · 本轮进化</small>')  # Sort by cohort score, not champion role.


def test_report_distinguishes_screening_winner_from_better_audio_interpolation():
    def comparison(name, values, archive_id=None):
        results = [{"source": "/data/%d.wav" % index, "status": "completed",
                    "result": {"metrics": {"missing_nmse": value}}}
                   for index, value in enumerate(values)]
        return {"algorithm": name, **({"archive_id": archive_id} if archive_id else {}),
                "results": results, "summary": summarize_evaluations(results, 2, "audio")}

    evolved = comparison("tsvd_evolved", [0.235488, 0.428454], "v1")
    interpolation = comparison("linear_interpolation_waveform", [0.120655, 0.041515])
    report = _report({"modalities": {"audio": {
        "status": "completed", "current_champion": {"algorithm": "tsvd_evolved", "archive_id": "v1"},
        "dataset_best_archive_id": "v1", "overall_best_algorithm": "linear_interpolation_waveform",
        "comparisons": [evolved], "baseline_comparisons": [interpolation],
        "method_selection": {"status": "completed", "recommendation_mode": "llm",
                             "recommended_methods": ["tsvd", "btd", "tt"],
                             "shortlist": ["tsvd", "btd", "tt"], "winner": "tsvd", "results": []},
    }}, "inventory": {"failures": []}})
    assert "同条件全类对照最优：`linear_interpolation_waveform`" in report
    assert "同类最优已归档进化版本：`v1`" in report
    assert "预赛胜出者只是进化起点" in report
    assert "<strong>0.081085</strong>" in report
    assert "<u>0.331971</u>" in report
    assert report.index('linear_interpolation_waveform<br>') < report.index('tsvd_evolved<br>')


def test_paper_style_audio_report_uses_nmse_only():
    results = [{"source": "/data/a.wav", "is_development": True, "status": "completed",
                "result": {"metrics": {"missing_nmse": 0.2}}}]
    record = {"algorithm": "siren", "results": results,
              "summary": summarize_evaluations(results, 1, "audio")}
    report = _report({"modalities": {"audio": {"status": "completed",
        "comparisons": [], "baseline_comparisons": [record]}}, "inventory": {"failures": []}})
    assert '<th>NMSE ↓</th>' in report and '<strong>0.200000</strong>' in report
    assert '<th>PSNR ↑</th>' not in report and '<th>SSIM ↑</th>' not in report


def test_fixed_video_baseline_uses_dataset_evaluation_budget(tmp_path, monkeypatch):
    from research_agent import recovery
    source = tmp_path / "video.mat"
    savemat(source, {"Ohsi": np.random.default_rng(2).random((12, 16, 3, 4))})
    case = prepare_recovery_case(str(source), tmp_path / "case", 0.4, "random", 17)
    captured = []

    def capture(**kwargs):
        captured.append(kwargs)
        return {"metrics": {"missing_psnr": 20.0}}

    monkeypatch.setattr(recovery, "final_fit_and_evaluate", capture)
    config = RecoveryConfig(dataset_root=str(tmp_path), evaluation_steps=37,
                            evaluation_validation_interval=3, evaluation_patience=0,
                            device="cpu")
    recovery.evaluate_fixed_baseline("tucker", case, tmp_path / "baseline", config)
    assert len(captured) == 1
    call = captured[0]
    assert call["seed"] == case["seed"]
    assert call["training_config"].max_steps == 37
    assert call["training_config"].validation_interval == 3
    assert call["training_config"].early_stopping_patience == 38
    assert call["observed_mask"].shape == call["ground_truth"].shape
    assert call["include_full_reference_metrics"] is False


def test_fixed_audio_baseline_keeps_nmse_context_and_exports_wav(tmp_path, monkeypatch):
    from research_agent import recovery
    from research_agent.core.audio_metrics import active_audio_metadata

    waveform = np.sin(np.arange(37) / 4).astype(np.float32)
    source = tmp_path / "audio.wav"
    wavfile.write(source, 16000, waveform)
    case = prepare_recovery_case(str(source), tmp_path / "case", 0.4, "random", 17,
                                 audio_frame_size=8)

    def capture(**kwargs):
        assert active_audio_metadata() == case
        assert kwargs["seed"] == case["seed"]
        assert kwargs["training_config"].learning_rate == 1e-4
        assert kwargs["selected_trial"]["hyperparameters"]["coordinate_mode"] == "audio"
        assert kwargs["selected_trial"]["hyperparameters"]["sample_count"] == len(waveform)
        reconstruction = tmp_path / "model_reconstruction.npy"
        np.save(reconstruction, kwargs["ground_truth"])
        return {"metrics": {"missing_nmse": 0.0},
                "artifacts": {"reconstruction": str(reconstruction)}}

    monkeypatch.setattr(recovery, "final_fit_and_evaluate", capture)
    config = RecoveryConfig(dataset_root=str(tmp_path), evaluation_steps=3, device="cpu")
    result = recovery.evaluate_fixed_baseline("siren", case, tmp_path / "baseline", config)
    rate, exported = wavfile.read(result["artifacts"]["audio"])
    assert rate == 16000 and len(exported) == len(waveform)
    assert result["metrics"] == {"missing_nmse": 0.0}
    assert active_audio_metadata() is None


def test_siren_selects_whole_modality_configuration_and_uses_final_shared_budget(tmp_path, monkeypatch):
    from research_agent import recovery
    cases = [{"source": "/data/a.mat", "data_type": "Image"}, {"source": "/data/b.mat", "data_type": "Image"}]
    calls = []
    def evaluate(name, case, output_dir, config, selected_trial):
        calls.append((case["source"], config.evaluation_steps, config.evaluation_validation_interval,
                      config.evaluation_patience, selected_trial))
        rate = selected_trial["learning_rate"]
        psnr = (40 if case["source"].endswith("a.mat") else 10) if rate == 1e-4 else 30 if rate == 5e-5 else 20
        return {"metrics": {"missing_psnr": psnr, "missing_mse": 10 ** (-psnr / 10), "composite_ssim": 0.8}}
    monkeypatch.setattr(recovery, "evaluate_fixed_baseline", evaluate)
    config = RecoveryConfig(dataset_root=str(tmp_path), siren_tuning_trials=3, siren_max_steps=7,
                            siren_validation_interval=2, siren_patience=4, evaluation_steps=11,
                            evaluation_validation_interval=3, evaluation_patience=0)
    result = recovery._siren_comparison(cases, 0, tmp_path, "Image", config)
    assert result["configuration"]["learning_rate"] == 5e-5  # Best average, despite worse representative.
    assert result["tuning"]["selected_trial"] == 2
    assert len(calls) == 8  # Three trials x two samples, then frozen final evaluation x two.
    assert all(call[1:4] == (7, 2, 4) for call in calls[:6])
    assert all(call[1:4] == (11, 3, 0) and call[4]["learning_rate"] == 5e-5 for call in calls[6:])
    feedback = recovery._comparison_feedback(result)
    assert feedback["configuration"] == result["configuration"]
    assert feedback["summary"]["mean_missing_psnr"] == 30


@pytest.mark.parametrize("kind", ["Video", "audio"])
def test_recovery_reuses_tuned_siren_for_development_and_llm_feedback(tmp_path, kind):
    root = tmp_path / "data"
    root.mkdir()
    for index in range(2):
        if kind == "audio":
            wavfile.write(root / ("sample%d.wav" % index), 16000,
                          np.sin(np.arange(37) / (4 + index)).astype(np.float32))
        else:
            savemat(root / ("sample%d.mat" % index), {"Ohsi": np.random.default_rng(index).random((8, 8, 3, 2))})
    state = run_recovery(RecoveryConfig(dataset_root=str(root), output_dir=str(tmp_path / "outputs"),
        history_root=str(tmp_path / "history"), candidate_root=str(tmp_path / "candidates"),
        knowledge_root=str(tmp_path / "knowledge"),
        approved_root=str(tmp_path / "approved"), data_types=(kind,), audio_frame_size=8,
        image_size=None, base_model="tucker", evolution_steps=2, evaluation_steps=3,
        improvement_rounds=1, tuning_trials=1, fair_learning_rate_candidates=(0.01,),
        fair_refine_learning_rate=False, siren_tuning_trials=1, siren_max_steps=2,
        siren_validation_interval=1, siren_patience=2, device="cpu", llm_mode="off"))
    group = state["modalities"][kind]
    assert state["stage"] == "COMPLETED", group.get("error")
    record = next(item for item in group["baseline_comparisons"] if item["algorithm"] == "siren")
    full = json.loads(Path(group["development_report"]).with_name("state.json").read_text())
    day4 = json.loads(Path(full["artifacts"]["day4_state"]).read_text())
    reused = day4["results"]["siren_comparison"]
    assert reused["training_cache"]["source"] == "whole_modality_evaluation"
    assert reused["training_cache"]["reused"]
    assert reused["metrics"] == record["results"][group["representative_index"]]["result"]["metrics"]
    if kind == "audio":
        formal_interpolation = next(item for item in group["baseline_comparisons"]
                                   if item["algorithm"] == "linear_interpolation_waveform")
        assert day4["results"]["interpolation_metrics"] == formal_interpolation["results"][group["representative_index"]]["result"]["metrics"]
    assert reused["training"]["hyperparameters"]["coordinate_mode"] == kind.lower()
    if kind == "audio":
        import torch
        from research_agent.core.models.registry import create_model
        result = record["results"][group["representative_index"]]["result"]
        saved = torch.load(result["artifacts"]["checkpoint"], map_location="cpu", weights_only=True)
        shape = tuple(np.load(group["cases"][group["representative_index"]]["gt_path"]).shape)
        restored = create_model("siren", shape, [0.0] * shape[-1], saved["hyperparameters"])
        restored.load_state_dict(saved["state_dict"])
        torch.testing.assert_close(restored.coordinates[:37, 0], torch.linspace(-1, 1, 37))
        torch.testing.assert_close(restored().detach(), torch.from_numpy(np.load(result["artifacts"]["raw_reconstruction"])))
    reference = next(item for item in group["pre_evolution_reference"]["fixed_baselines"] if item["algorithm"] == "siren")
    assert reference["configuration"]["learning_rate"] == 1e-4
    assert reference["summary"] == record["summary"]
    assert "SIREN 配置与训练" in Path(state["report"]).read_text()


def test_siren_identical_tuning_and_evaluation_protocol_reuses_training(tmp_path, monkeypatch):
    from research_agent import recovery
    calls = []
    def evaluate(name, case, output_dir, config, selected_trial):
        calls.append(selected_trial)
        return {"metrics": {"missing_psnr": 30, "missing_mse": 0.001, "composite_ssim": 0.8}}
    monkeypatch.setattr(recovery, "evaluate_fixed_baseline", evaluate)
    config = RecoveryConfig(dataset_root=str(tmp_path), siren_tuning_trials=1,
                            evaluation_steps=5, evaluation_validation_interval=1,
                            evaluation_patience=2, siren_validation_interval=1, siren_patience=2)
    result = recovery._siren_comparison([{"source": "/data/a.mat"}], 0, tmp_path, "Image", config)
    assert len(calls) == 1
    assert result["tuning"]["evaluation_reused"]
    assert result["protocol"]["steps"] == 5


def test_siren_perfect_cohort_ties_are_broken_by_finite_psnr_not_ssim(tmp_path, monkeypatch):
    from research_agent import recovery
    calls = []
    def evaluate(name, case, output_dir, config, selected_trial):
        trial = len(calls) // 2
        calls.append(selected_trial)
        if case["source"].endswith("perfect.mat"):
            return {"metrics": {"missing_psnr": None, "missing_mse": 0, "composite_ssim": 1.0}}
        score = 10.0 if trial == 0 else 30.0
        return {"metrics": {"missing_psnr": score, "missing_mse": 10 ** (-score / 10),
                            "composite_ssim": 0.9 if trial == 0 else 0.8}}
    monkeypatch.setattr(recovery, "evaluate_fixed_baseline", evaluate)
    config = RecoveryConfig(dataset_root=str(tmp_path), siren_tuning_trials=2,
                            evaluation_steps=5, evaluation_validation_interval=1,
                            evaluation_patience=2, siren_validation_interval=1, siren_patience=2)
    result = recovery._siren_comparison([{"source": "/data/perfect.mat"}, {"source": "/data/finite.mat"}],
                                       0, tmp_path, "Image", config)
    assert result["tuning"]["selected_trial"] == 2


def test_siren_incomplete_tuning_never_falls_back_to_untuned_results(tmp_path, monkeypatch):
    from research_agent import recovery
    def evaluate(name, case, output_dir, config, selected_trial):
        if case["source"].endswith("b.mat"):
            raise RuntimeError("failed sample")
        return {"metrics": {"missing_psnr": 30, "missing_mse": 0.001, "composite_ssim": 0.8}}
    monkeypatch.setattr(recovery, "evaluate_fixed_baseline", evaluate)
    config = RecoveryConfig(dataset_root=str(tmp_path), siren_tuning_trials=2)
    result = recovery._siren_comparison([{"source": "/data/a.mat"}, {"source": "/data/b.mat"}],
                                        1, tmp_path, "Image", config)
    assert result["tuning"]["status"] == "failed"
    assert result["configuration"] is None
    assert not result["summary"]["complete"]
    assert not result["non_development_summary"]["complete"]
    assert all(item["status"] == "failed" for item in result["results"])
    saved = json.loads(Path(result["tuning"]["path"]).read_text())
    assert saved["trials"][0]["comparison"]["results"][1]["error"] == "RuntimeError: failed sample"


def test_audio_linear_interpolation_crosses_frame_boundary_without_hidden_gt():
    from research_agent.recovery import _linear_waveform_fill

    observed = np.array([0, 0, 0, 3, 4, 5, 0, 0, 8, 9], dtype=np.float32).reshape(2, 5, 1)
    mask = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=bool).reshape(2, 5, 1)
    completed = _linear_waveform_fill(observed, mask, sample_count=10).reshape(-1)
    assert np.allclose(completed, np.arange(10))


def test_audio_interpolation_baseline_reports_only_nmse(tmp_path):
    from research_agent.recovery import evaluate_fixed_baseline

    waveform = np.sin(np.arange(37) / 4).astype(np.float32)
    source = tmp_path / "audio.wav"
    wavfile.write(source, 16000, waveform)
    case = prepare_recovery_case(str(source), tmp_path / "case", 0.4, "slices", 17,
                                 audio_frame_size=8)
    result = evaluate_fixed_baseline(
        "linear_interpolation_waveform", case, tmp_path / "baseline",
        RecoveryConfig(dataset_root=str(tmp_path), device="cpu"))
    metrics = result["metrics"]
    assert "missing_nmse" in metrics
    assert "missing_psnr" not in metrics and "composite_ssim" not in metrics
    assert Path(result["artifacts"]["audio"]).is_file()
