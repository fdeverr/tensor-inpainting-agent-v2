import json
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.io import savemat, wavfile

from research_agent import recovery, siren_cache
from research_agent.recovery import RecoveryConfig, evaluate_fixed_baseline
from research_agent.recovery_data import prepare_recovery_case


def _case(root, kind="Image", name="sample", seed=7, missing_rate=0.4, frame_size=8):
    root.mkdir(parents=True, exist_ok=True)
    if kind == "audio":
        source = root / (name + ".wav")
        wavfile.write(source, 8000, (np.sin(np.arange(37)) * 16000).astype(np.int16))
    else:
        source = root / (name + ".mat")
        shape = {"Image": (4, 5, 3), "MSI": (4, 5, 6), "Video": (4, 5, 3, 2)}[kind]
        savemat(source, {"Ohsi": np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)})
    return prepare_recovery_case(str(source), root / "prepared", missing_rate, "random", seed,
                                 image_size=None, audio_frame_size=frame_size)


def _config(root):
    return RecoveryConfig(dataset_root=str(root), output_dir=str(root / "outputs"), device="cpu",
                          evaluation_steps=3, evaluation_validation_interval=1, evaluation_patience=0,
                          siren_max_steps=2, siren_validation_interval=1, siren_patience=2,
                          siren_tuning_trials=2)


@pytest.fixture
def fake_fits(monkeypatch):
    calls = []

    def fit(**kwargs):
        calls.append(kwargs)
        directory = Path(kwargs["output_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        raw = kwargs["ground_truth"] * 0.9
        params = kwargs["selected_trial"]["hyperparameters"]
        artifacts = {}
        for key, filename in (("raw_reconstruction", "model_raw.npy"),
                              ("reconstruction", "model_completed.npy")):
            path = directory / filename
            np.save(path, raw)
            artifacts[key] = str(path)
        checkpoint = directory / "model.pt"
        checkpoint.write_bytes(b"a-test-checkpoint")
        artifacts["checkpoint"] = str(checkpoint)
        history = directory / "final_fit_history.json"
        history.write_text(json.dumps({"history": [{"step": 1, "data_train_loss": 0.1,
                                                    "total_train_loss": 0.1}],
                                       "reused_without_retraining": False}))
        artifacts["history"] = str(history)
        metrics_path = directory / "metrics.json"
        artifacts["metrics"] = str(metrics_path)
        metrics = ({"missing_nmse": 0.1} if params["coordinate_mode"] == "audio" else
                   {"missing_psnr": 30.0, "missing_mse": 0.001, "composite_ssim": 0.9})
        result = {"model_name": "siren", "hyperparameters": params,
                  "learning_rate": kwargs["selected_trial"]["learning_rate"],
                  "selected_steps": 1, "runtime_seconds": 2.0, "parameter_count": 100,
                  "final_train_mse": 0.1, "observed_pixels_used": 10,
                  "selection_validation_mse": 0.001, "metrics": metrics,
                  "artifacts": artifacts, "reused_without_retraining": False}
        metrics_path.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(recovery, "final_fit_and_evaluate", fit)
    return calls


@pytest.mark.parametrize("kind", ["Image", "MSI", "Video", "audio"])
def test_repeated_modality_run_reuses_every_trial_and_formal_fit(tmp_path, fake_fits, kind):
    config = _config(tmp_path)
    first_cases = [_case(tmp_path / "data-a", kind), _case(tmp_path / "data-b", kind, seed=8)]
    first = recovery._siren_comparison(first_cases, 0, tmp_path / "run-1", kind, config)
    assert len(fake_fits) == 6  # 2 trials + a different formal budget, each on 2 samples.
    second_cases = [_case(tmp_path / "moved-a", kind), _case(tmp_path / "moved-b", kind, seed=8)]
    second = recovery._siren_comparison(second_cases, 1, tmp_path / "run-2", kind,
                                       replace(config, improvement_rounds=1, llm_mode="off", base_model="cp",
                                               evolution_steps=9, lpips=True))
    assert len(fake_fits) == 6
    assert second["summary"] == first["summary"]
    assert second["configuration"] == first["configuration"]
    assert second["training_cache"] == {"scope": "persistent_per_sample_and_protocol",
                                        "reused_fits": 6, "trained_fits": 0, "failed_fits": 0}
    for item in second["results"]:
        result = item["result"]
        assert result["reused_without_retraining"]
        assert item["source"] in {case["source"] for case in second_cases}
        for path in result["artifacts"].values():
            assert Path(path).is_file()
            assert Path(path).is_relative_to(tmp_path / "run-2")
        saved = json.loads(Path(result["artifacts"]["metrics"]).read_text())
        assert saved == result
        assert "run-1" not in json.dumps(saved)
        history = json.loads(Path(result["artifacts"]["history"]).read_text())
        assert history["history"][0]["data_train_loss"] == 0.1
        assert history["persistent_cache_reused"]
    tuning = json.loads(Path(second["tuning"]["path"]).read_text())
    assert tuning["training_cache"] == second["training_cache"]
    assert all(item["result"]["training_cache"]["reused"]
               for trial in tuning["trials"] for item in trial["comparison"]["results"])


@pytest.mark.parametrize("change", ["gt", "mask", "seed", "steps", "interval", "patience",
                                     "learning_rate", "width", "source", "runtime", "type", "output_root"])
def test_changed_fit_conditions_invalidate_cache(tmp_path, monkeypatch, fake_fits, change):
    case = _case(tmp_path / "data")
    config = _config(tmp_path)
    selected = {"hyperparameters": {"hidden_features": 8, "hidden_layers": 1}, "learning_rate": 1e-4}
    first = evaluate_fixed_baseline("siren", case, tmp_path / "run-1", config, selected)
    if change in {"gt", "mask"}:
        path = case["gt_path" if change == "gt" else "mask_path"]
        values = np.load(path)
        values.flat[0] = (not values.flat[0]) if change == "mask" else values.flat[0] + 0.01
        np.save(path, values)  # Leave metadata digests stale to prove actual contents are hashed.
    elif change == "seed":
        case = {**case, "seed": case["seed"] + 1}
    elif change in {"steps", "interval", "patience", "output_root"}:
        changes = {"steps": {"evaluation_steps": 4}, "interval": {"evaluation_validation_interval": 2},
                   "patience": {"evaluation_patience": 1}, "output_root": {"output_dir": str(tmp_path / "other")}}
        config = replace(config, **changes[change])
    elif change == "learning_rate":
        selected = {**selected, "learning_rate": 3e-4}
    elif change == "width":
        selected = {**selected, "hyperparameters": {"hidden_features": 16, "hidden_layers": 1}}
    elif change == "source":
        monkeypatch.setattr(siren_cache, "_source_fingerprint", lambda: {"changed-code": "different"})
    elif change == "runtime":
        monkeypatch.setattr(siren_cache, "_runtime_fingerprint", lambda device: {"changed-runtime": True})
    elif change == "type":
        case = {**case, "data_type": "MSI"}
    second = evaluate_fixed_baseline("siren", case, tmp_path / "run-2", config, selected)
    assert len(fake_fits) == 2
    assert not second["training_cache"]["reused"]
    if change == "output_root":
        assert first["training_cache"]["cache_dir"] != second["training_cache"]["cache_dir"]
    else:
        assert first["training_cache"]["fingerprint"] != second["training_cache"]["fingerprint"]


@pytest.mark.parametrize("field", ["sample_count", "sample_rate", "original_min", "original_max", "frame_size"])
def test_audio_metadata_is_part_of_cache_identity(tmp_path, fake_fits, field):
    case = _case(tmp_path / "data", "audio")
    config = _config(tmp_path)
    evaluate_fixed_baseline("siren", case, tmp_path / "run-1", config)
    case = {**case, field: case.get(field, 0) + 1}
    evaluate_fixed_baseline("siren", case, tmp_path / "run-2", config)
    assert len(fake_fits) == 2


def test_adding_one_sample_reuses_existing_fits_and_recomputes_cohort(tmp_path, fake_fits):
    case = _case(tmp_path / "old")
    config = _config(tmp_path)
    first = recovery._siren_comparison([case], 0, tmp_path / "run-1", "Image", config)
    second = recovery._siren_comparison([case, _case(tmp_path / "new", seed=8)], 1,
                                       tmp_path / "run-2", "Image", config)
    assert len(fake_fits) == 6  # 3 old fits + 3 for the genuinely new seed/mask.
    assert first["summary"]["expected_count"] == 1
    assert second["summary"]["expected_count"] == 2
    assert second["training_cache"]["reused_fits"] == 3
    assert second["training_cache"]["trained_fits"] == 3


def test_increased_trial_count_reuses_previously_trained_configurations(tmp_path, fake_fits):
    case = _case(tmp_path / "data")
    config = _config(tmp_path)
    recovery._siren_comparison([case], 0, tmp_path / "run-1", "Image", replace(config, siren_tuning_trials=1))
    second = recovery._siren_comparison([case], 0, tmp_path / "run-2", "Image", config)
    assert len(fake_fits) == 3  # 1 original trial, formal fit, 1 newly requested trial.
    assert second["training_cache"]["reused_fits"] == 2
    assert second["training_cache"]["trained_fits"] == 1


@pytest.mark.parametrize("damage", ["missing_checkpoint", "modified_checkpoint", "bad_manifest",
                                     "modified_result", "escape"])
def test_invalid_cache_retrains_and_next_run_can_reuse(tmp_path, fake_fits, damage):
    case = _case(tmp_path / "data")
    config = _config(tmp_path)
    first = evaluate_fixed_baseline("siren", case, tmp_path / "run-1", config)
    directory = Path(first["training_cache"]["cache_dir"])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = directory / manifest["result"]["artifacts"]["checkpoint"]
    if damage == "missing_checkpoint":
        checkpoint.unlink()
    elif damage == "modified_checkpoint":
        checkpoint.write_bytes(b"corrupt-checkpoint")
    elif damage == "bad_manifest":
        manifest_path.write_text("not-json")
    else:
        if damage == "modified_result":
            manifest["result"]["metrics"]["missing_psnr"] = 99
        else:
            manifest["result"]["artifacts"]["checkpoint"] = str(tmp_path / "run-1" / "model.pt")
            manifest["result_sha256"] = siren_cache._json_hash(manifest["result"])
        manifest_path.write_text(json.dumps(manifest))
    second = evaluate_fixed_baseline("siren", case, tmp_path / "run-2", config)
    assert second["training_cache"]["reason"] == "invalid_cache"
    assert second["metrics"]["missing_psnr"] == 30
    third = evaluate_fixed_baseline("siren", case, tmp_path / "run-3", config)
    assert len(fake_fits) == 2
    assert third["training_cache"]["reused"]


def test_cache_is_independent_of_old_run_files(tmp_path, fake_fits):
    case = _case(tmp_path / "data")
    config = _config(tmp_path)
    first = evaluate_fixed_baseline("siren", case, tmp_path / "run-1", config)
    # Delete only our test-generated run files, leaving the persistent cache.
    shutil.rmtree(tmp_path / "run-1")
    second = evaluate_fixed_baseline("siren", case, tmp_path / "run-2", config)
    assert len(fake_fits) == 1
    assert first["metrics"] == second["metrics"]


def test_cache_disk_failure_preserves_successful_fit(tmp_path, monkeypatch, fake_fits):
    case = _case(tmp_path / "data")
    def unavailable(*args):
        raise OSError("cache disk unavailable")
    monkeypatch.setattr(siren_cache, "_publish_cache", unavailable)
    result = evaluate_fixed_baseline("siren", case, tmp_path / "run", _config(tmp_path))
    assert result["metrics"]["missing_psnr"] == 30
    assert not result["training_cache"]["written"]
    assert Path(result["artifacts"]["checkpoint"]).is_file()


def test_failed_fit_is_not_cached(tmp_path, monkeypatch, fake_fits):
    case = _case(tmp_path / "data")
    config = _config(tmp_path)
    def failed(**kwargs):
        raise RuntimeError("training failed")
    monkeypatch.setattr(recovery, "final_fit_and_evaluate", failed)
    with pytest.raises(RuntimeError, match="training failed"):
        evaluate_fixed_baseline("siren", case, tmp_path / "run", config)
    assert not list((tmp_path / "outputs").rglob("manifest.json"))


def test_metrics_audit_write_failure_cannot_destroy_successful_fit(tmp_path, monkeypatch, fake_fits):
    case = _case(tmp_path / "data")
    original_write = siren_cache._atomic_write_json
    def full_output_disk(path, payload):
        if path.name == "metrics.json":
            raise OSError("output disk full")
        original_write(path, payload)
    monkeypatch.setattr(siren_cache, "_atomic_write_json", full_output_disk)
    result = evaluate_fixed_baseline("siren", case, tmp_path / "run", _config(tmp_path))
    saved = json.loads(Path(result["artifacts"]["metrics"]).read_text())
    assert saved["metrics"] == result["metrics"]
    assert result["training_cache"]["written"]
    assert result["training_cache"]["metrics_audit_write_error"] == "output disk full"


def test_effectively_identical_tuning_and_evaluation_reuses_same_fit(tmp_path, fake_fits):
    case = _case(tmp_path / "data")
    # 0 (disabled) and steps+1 patience execute the same training protocol.
    config = replace(_config(tmp_path), siren_tuning_trials=1, siren_max_steps=3, siren_patience=4)
    result = recovery._siren_comparison([case], 0, tmp_path / "run", "Image", config)
    assert len(fake_fits) == 1
    assert result["training_cache"]["reused_fits"] == 1


def test_auto_and_cpu_use_the_same_actual_device_cache(tmp_path, fake_fits, monkeypatch):
    monkeypatch.setattr(siren_cache.torch.cuda, "is_available", lambda: False)
    case = _case(tmp_path / "data")
    config = _config(tmp_path)
    evaluate_fixed_baseline("siren", case, tmp_path / "run-1", config)
    second = evaluate_fixed_baseline("siren", case, tmp_path / "run-2", replace(config, device="auto"))
    assert len(fake_fits) == 1
    assert second["training_cache"]["reused"]


def test_repeating_complete_recovery_never_trains_siren_again(tmp_path, monkeypatch):
    from research_agent.core import fair_experiment
    _case(tmp_path / "dataset", name="a")
    # Keep only actual source files in the inventory root.
    shutil.rmtree(tmp_path / "dataset" / "prepared")
    config = replace(_config(tmp_path), data_types=("Image",), image_size=None,
                     base_model="tucker", llm_mode="off", evolution_steps=2,
                     evaluation_steps=2, evaluation_patience=2, tuning_trials=1, improvement_rounds=1,
                     siren_tuning_trials=1, fair_learning_rate_candidates=(0.01,),
                     fair_refine_learning_rate=False, candidate_root=str(tmp_path / "candidates"),
                     approved_root=str(tmp_path / "approved"), history_root=str(tmp_path / "history"),
                     knowledge_root=str(tmp_path / "knowledge"), dataset_root=str(tmp_path / "dataset"))
    first = recovery.run_recovery(config)
    assert first["stage"] == "COMPLETED"
    original_train = fair_experiment.train_tensor_model
    def prohibit_siren(**kwargs):
        assert kwargs["model_name"] != "siren", "repeat recovery must not train SIREN"
        return original_train(**kwargs)
    monkeypatch.setattr(fair_experiment, "train_tensor_model", prohibit_siren)
    second = recovery.run_recovery(config)
    assert second["stage"] == "COMPLETED", second["modalities"]["Image"].get("error")
    before = next(item for item in first["modalities"]["Image"]["baseline_comparisons"] if item["algorithm"] == "siren")
    after = next(item for item in second["modalities"]["Image"]["baseline_comparisons"] if item["algorithm"] == "siren")
    assert after["summary"] == before["summary"]
    assert after["training_cache"]["trained_fits"] == 0
    assert after["training_cache"]["reused_fits"] == 1
    assert "跨运行缓存：复用 `1` 次样本拟合，新训练 `0` 次" in Path(second["report"]).read_text()
    full = json.loads(Path(second["modalities"]["Image"]["development_report"]).with_name("state.json").read_text())
    day4 = json.loads(Path(full["artifacts"]["day4_state"]).read_text())
    assert day4["results"]["siren_comparison"]["training_cache"]["reused"]
    panel = second["modalities"]["Image"]["pre_evolution_reference"]["fixed_baselines"]
    feedback = next(item for item in panel if item["algorithm"] == "siren")
    assert feedback["summary"] == before["summary"]
    assert feedback["samples"][0]["training"]


@pytest.mark.parametrize("kind", ["Image", "MSI", "Video", "audio"])
def test_real_checkpoint_curve_and_scores_reused_without_trainer(tmp_path, monkeypatch, kind):
    from research_agent.core import fair_experiment
    case = _case(tmp_path / "data", kind)
    config = replace(_config(tmp_path), evaluation_steps=2)
    selected = {"hyperparameters": {"hidden_features": 8, "hidden_layers": 1}, "learning_rate": 1e-4}
    first = evaluate_fixed_baseline("siren", case, tmp_path / "run-1", config, selected)
    def forbidden(**kwargs):
        raise AssertionError("identical repeat must not call the trainer")
    monkeypatch.setattr(fair_experiment, "train_tensor_model", forbidden)
    second = evaluate_fixed_baseline("siren", case, tmp_path / "run-2", config, selected)
    assert second["training_cache"]["reused"]
    assert second["metrics"] == first["metrics"]
    assert siren_cache._file_hash(second["artifacts"]["checkpoint"]) == siren_cache._file_hash(first["artifacts"]["checkpoint"])
    before = json.loads(Path(first["artifacts"]["history"]).read_text())["history"]
    after = json.loads(Path(second["artifacts"]["history"]).read_text())["history"]
    assert before == after
    np.testing.assert_array_equal(np.load(first["artifacts"]["reconstruction"]),
                                  np.load(second["artifacts"]["reconstruction"]))
    if kind == "audio":
        assert "missing_nmse" in second["metrics"]
        assert not ({"missing_psnr", "composite_ssim", "psnr", "ssim"} & second["metrics"].keys())
        assert Path(second["artifacts"]["audio"]).is_file()
