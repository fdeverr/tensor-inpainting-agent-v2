import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research_agent import recovery_registry as registry
from research_agent.core.audio_metrics import audio_metric_context
from research_agent.core.models import create_model
from research_agent.core.trainer import train_tensor_model
from research_agent.schemas import TrainingConfig


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _archive(tmp_path, kind="builtin", algorithm="tucker"):
    training = {"hyperparameters": {"rank_h": 2, "rank_w": 2, "rank_c": 1}, "learning_rate": 0.01}
    if algorithm == "siren":
        training["hyperparameters"] = {"hidden_features": 8, "hidden_layers": 1}
    day4 = tmp_path / "day4.json"
    _json(day4, {"selected_model": "tucker", "results": {"training": training,
               "siren_comparison": {"training": training}}})
    report = tmp_path / "report.md"
    report.write_text("test evidence", encoding="utf-8")
    role = {"builtin": "implicit_neural_baseline" if algorithm == "siren" else "tensor_baseline",
            "candidate": "candidate", "interpolation": "interpolation_baseline"}[kind]
    state = {"run_id": "test-run", "best_available": {"role": role, "algorithm": algorithm,
             "metrics": {}}, "artifacts": {"day4_state": str(day4), "report": str(report)}}
    if kind == "candidate":
        approved = tmp_path / "approved"
        approved.mkdir()
        source = "class CandidateTensorInpaintingModel(TuckerDecomposition):\n    pass\n"
        (approved / "model.py").write_text(source)
        _json(approved / "manifest.json", {"candidate_id": "test-candidate", "validation_status": "validated",
              "eligible_for_training": True, "code_sha256": hashlib.sha256(source.encode()).hexdigest()})
        _json(approved / "validation.json", {"passed": True})
        for name in ("idea.json", "approved_manifest.json"):
            _json(approved / name, {})
        _json(approved / "best_config.json", training)
        day6 = tmp_path / "day6.json"
        _json(day6, {"promotion": {"approved_dir": str(approved)}, "rounds": []})
        state["artifacts"]["day6_state"] = str(day6)
    return registry.archive_champion(tmp_path / "history", "Image", state, "test", {})


@pytest.mark.parametrize("kind", ["builtin", "candidate"])
def test_archived_models_survive_current_library_change_and_train(tmp_path, monkeypatch, kind):
    record = _archive(tmp_path, kind)
    shape = (4, 5, 1)
    torch.manual_seed(4)
    expected = create_model("tucker", shape, [0.5], record["config"]["hyperparameters"])().detach()
    monkeypatch.setattr(registry, "model_library_fingerprint", lambda: "changed-siren-or-base")
    from research_agent.core.models import registry as current
    monkeypatch.setitem(current.MODEL_CLASSES, "tucker", None)
    builder = registry.champion_builder(record)
    torch.manual_seed(4)
    model = builder(shape, [0.5], record["config"]["hyperparameters"])
    torch.testing.assert_close(model(), expected)
    assert "_archive_models_" in type(model).__mro__[1 if kind == "candidate" else 0].__module__
    gt = np.linspace(0, 1, 20, dtype=np.float32).reshape(shape)
    mask = np.ones(shape, dtype=bool)
    mask[1] = False
    output = train_tensor_model("test", record["config"]["hyperparameters"], gt * mask, mask,
        TrainingConfig(max_steps=2, validation_interval=1, device="cpu"), 3,
        model_builder=builder, ground_truth=gt)
    assert np.isfinite(output.reconstruction).all()
    assert registry.champion_execution_protocol(record)["model_code_scope"] == "frozen_model_and_parent_dependencies"


@pytest.mark.parametrize("file", ["model.py", "model_snapshot/base.py", "model_snapshot/tucker.py"])
def test_snapshot_integrity_is_checked_even_after_first_load(tmp_path, file):
    record = _archive(tmp_path)
    registry.champion_builder(record)
    path = Path(record["archive_dir"]) / file
    path.write_text(path.read_text() + "\n# tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.champion_builder(record)


def test_snapshot_integrity_is_checked_on_evaluation_cache_hit(tmp_path, monkeypatch):
    from research_agent import recovery
    record = _archive(tmp_path)
    monkeypatch.setattr(recovery, "evaluate_champion", lambda *args: {"metrics": {"missing_psnr": 20}})
    case = {"gt_sha256": "gt", "mask_sha256": "mask", "seed": 1, "data_type": "Image"}
    config = recovery.RecoveryConfig(dataset_root=str(tmp_path), evaluation_steps=2)
    cache = {}
    recovery._evaluate_cached(record, case, tmp_path, config, cache)
    path = Path(record["archive_dir"]) / "model_snapshot/base.py"
    path.write_text(path.read_text() + "\n# tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        recovery._evaluate_cached(record, case, tmp_path, config, cache)


@pytest.mark.parametrize("shape,metadata,coordinate_mode", [
    ((3, 4, 1), {"data_type": "audio", "sample_count": 10, "channels": 1}, "audio"),
    ((3, 4, 2, 3), {"data_type": "Video"}, "video"),
])
def test_snapshot_siren_keeps_current_case_coordinate_context(tmp_path, shape, metadata, coordinate_mode):
    record = _archive(tmp_path, algorithm="siren")
    builder = registry.champion_builder(record)
    with audio_metric_context(metadata):
        model = builder(shape, np.zeros(int(np.prod(shape[2:]))), record["config"]["hyperparameters"])
        assert model.coordinate_mode == coordinate_mode
        assert tuple(model().shape) == shape


def test_legacy_builtin_uses_its_archived_source_not_current_model(tmp_path, monkeypatch):
    record = _archive(tmp_path)
    del record["model_snapshot"]
    monkeypatch.setattr(registry, "model_library_fingerprint", lambda: "changed-siren")
    builder = registry.champion_builder(record)
    model = builder((3, 4, 1), [0.5], record["config"]["hyperparameters"])
    assert "_legacy_champion_" in type(model).__module__
    assert model().shape == (3, 4, 1)
    assert "compatibility_note" in registry.champion_execution_protocol(record)


def test_legacy_candidate_does_not_silently_replace_unsaved_parents(tmp_path, monkeypatch):
    record = _archive(tmp_path, "candidate")
    del record["model_snapshot"]
    monkeypatch.setattr(registry, "model_library_fingerprint", lambda: "changed-parent")
    with pytest.raises(ValueError, match="lacks frozen parent dependencies"):
        registry.champion_builder(record)


def test_interpolation_executes_archived_version_and_checks_its_hash(tmp_path):
    record = _archive(tmp_path, "interpolation", "nearest_neighbor_manhattan")
    path = Path(record["archive_dir"]) / "algorithm.py"
    # Represent a legitimate older implementation, with its own matching hash.
    source = "def nearest_neighbor_fill(observed, mask):\n    return observed * 0 + 0.25\n"
    path.write_text(source)
    record["code_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    observed = np.ones((2, 2, 3), dtype=np.float32)
    mask = np.ones_like(observed, dtype=bool)
    builder = registry.champion_builder(record)
    np.testing.assert_array_equal(builder(observed, mask, {"data_type": "Image"}), np.full_like(observed, 0.25))
    path.write_text(source + "\n# tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.champion_builder(record)


def test_archived_audio_interpolation_uses_real_waveform_only(tmp_path):
    record = _archive(tmp_path, "interpolation", "linear_interpolation_waveform")
    observed = np.array([0, 0, 1, 999], dtype=np.float32).reshape(2, 2, 1)
    mask = np.array([True, False, True, True]).reshape(observed.shape)
    result = registry.champion_builder(record)(observed, mask, {"data_type": "audio", "sample_count": 3})
    np.testing.assert_allclose(result.reshape(-1), [0, 0.5, 1, 999])


def test_structure_reference_contains_actual_candidate_parent_and_loss_without_execution(tmp_path, monkeypatch):
    record = _archive(tmp_path, "candidate")
    directory = Path(record["archive_dir"])
    idea = {"architecture_family": "hybrid", "idea": "A proposal is a hypothesis, not a causal conclusion",
            "single_change": "test a declared mechanism", "model_code": "must not duplicate code from proposal"}
    _json(directory / "idea.json", idea)
    record["idea_sha256"] = hashlib.sha256((directory / "idea.json").read_bytes()).hexdigest()
    monkeypatch.setattr(registry, "model_library_fingerprint", lambda: "changed")
    monkeypatch.setattr(registry, "_verified_model_snapshot", lambda *args: pytest.fail("structure must not execute archived code"))
    result = registry.champion_structure_reference(record, include_source=True)
    assert result["status"] == "completed"
    assert result["entrypoint"]["class"] == "CandidateTensorInpaintingModel"
    assert result["design_proposal"]["single_change"] == idea["single_change"]
    assert "model_code" not in result["design_proposal"]
    assert result["proposal_evidence"] == "hash_verified_proposal_not_proven"
    files = {item["file"] for item in result["source_bundle"]}
    assert files == {"model.py", "model_snapshot/tucker.py", "model_snapshot/base.py"}
    methods = result["forward_and_loss_implementations"]
    assert any(item.get("class") == "TuckerDecomposition" and item["method"] == "forward" for item in methods)
    assert any(item.get("class") == "BaseTensorInpaintingModel" and item["method"] == "loss_terms"
               and "data_loss" in item["code"] for item in methods)
    assert any(item["name"] == "core" for outline in result["structure"] for node in outline["classes"]
               for item in node["declared_attributes"])
    assert Path(record["archive_dir"], "structure.json").is_file()


@pytest.mark.parametrize("kind,algorithm", [("builtin", "tucker"), ("builtin", "siren"),
                                           ("interpolation", "linear_interpolation_waveform")])
def test_builtin_and_interpolation_structures_do_not_borrow_selected_tensor_architecture(tmp_path, kind, algorithm):
    record = _archive(tmp_path, kind, algorithm)
    result = registry.champion_structure_reference(record, include_source=True)
    assert result["status"] == "completed"
    assert result["selected_configuration"] == record["config"]
    assert result["design_proposal"] == {}
    if algorithm == "siren":
        assert result["entrypoint"]["class"] == "SirenImplicitNetwork"
        assert "coordinate_mode" in json.dumps(result["structure"])
        assert any(buffer["name"] == "coordinates" for outline in result["structure"]
                   for node in outline["classes"] for buffer in node["registered_buffers"])
        assert all("tucker.py" not in item["file"] for item in result["source_bundle"])
    elif kind == "interpolation":
        assert result["entrypoint"]["function"] == "linear_waveform_fill"
        assert [item["file"] for item in result["source_bundle"]] == ["algorithm.py"]
    else:
        assert result["entrypoint"]["class"] == "TuckerDecomposition"


@pytest.mark.parametrize("file", ["model.py", "model_snapshot/base.py", "idea.json"])
def test_unverified_structure_is_explicitly_unavailable_not_guessed(tmp_path, file):
    record = _archive(tmp_path, "candidate")
    path = Path(record["archive_dir"]) / file
    path.write_text(path.read_text() + "\n# corrupted")
    result = registry.champion_structure_reference(record, include_source=True)
    assert result["status"] == "unavailable"
    assert "hash mismatch" in result["error"]
    assert "source_bundle" not in result


def test_legacy_structure_is_rebuilt_with_explicit_dependency_boundary(tmp_path, monkeypatch):
    record = _archive(tmp_path)
    del record["model_snapshot"]
    result = registry.champion_structure_reference(record, include_source=True)
    assert result["status"] == "completed"
    assert any(item["file"] == "legacy_current_dependencies/base.py" for item in result["source_bundle"])
    assert "compatibility_note" in result["execution_protocol"]
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    candidate = _archive(candidate_dir, "candidate")
    del candidate["model_snapshot"]
    monkeypatch.setattr(registry, "model_library_fingerprint", lambda: "changed")
    missing = registry.champion_structure_reference(candidate, include_source=True)
    assert missing["status"] == "unavailable"
    assert "original library" in missing["error"]


@pytest.mark.parametrize("audio", [False, True])
def test_history_source_budget_uses_current_cohort_scores_not_development_or_archive_order(monkeypatch, audio):
    from research_agent import recovery
    history = [{"archive_id": str(index), "base_method": "tucker", "kind": "candidate",
                "algorithm": "candidate-%d" % index} for index in range(5)]
    references = [{"archive_id": str(index)} for index in range(5)]
    comparisons = [{"archive_id": str(index), "summary": {
        "complete": index != 3,
        **({"mean_missing_nmse": score} if audio else {"mean_missing_psnr": score})}}
        for index, score in enumerate([0.8, 0.1, 0.2, 0.01, 0.3] if audio else [20, 40, 35, 50, 30])]
    def structure(record, include_source):
        return {"status": "completed", "base_method": record["base_method"], "kind": record["kind"],
                "structure": ["model outline"], "full_source_included": include_source,
                **({"source_bundle": [{"file": "model.py", "code": "actual archived code"}]} if include_source else {})}
    monkeypatch.setattr(recovery, "champion_structure_reference", structure)
    monkeypatch.setattr(registry, "champion_structure_reference", structure)
    policy = recovery._attach_historical_structures(history, comparisons, references)
    assert policy["full_source_archive_ids"] == []  # Current framework is not known before Day 4.
    assert all(item["structure_reference"]["structure"] for item in references)
    assert all("source_bundle" not in item["structure_reference"] for item in references)
    assert not comparisons[3]["structure_reference"]["full_source_included"]
    assert not comparisons[0]["structure_reference"]["full_source_included"]
    resolved = registry.historical_reference_for_framework({"historical_champions": comparisons}, "tucker")
    assert resolved["historical_structure_policy"]["full_source_archive_ids"] == ["1", "2", "4"]
    assert all("_source_archive" not in item for item in resolved["historical_champions"])
    assert not comparisons[1]["structure_reference"]["full_source_included"]  # No mutation.


def test_structure_and_archived_source_reach_actual_llm_prompt(tmp_path):
    from research_agent.candidate.generator import CandidateGenerator
    record = _archive(tmp_path, "candidate")
    structure = registry.champion_structure_reference(record, include_source=True)
    context = {"base_method": "tucker", "base_class_name": "TuckerDecomposition", "algorithm_comparison_reference": {"whole_modality_algorithms": {
        "historical_champions": [{"archive_id": "old-version", "structure_reference": structure}]}}}
    messages = CandidateGenerator._messages(context)
    payload = json.loads(messages[1]["content"])
    actual = payload["context"]["algorithm_comparison_reference"]["whole_modality_algorithms"]["historical_champions"][0]["structure_reference"]
    sources = payload["context"]["code_sources"]
    for expected, pooled in zip(structure["source_bundle"], actual["source_bundle"]):
        assert pooled["file"] == expected["file"]
        assert sources[pooled["code_ref"]] == expected["code"]
    assert "CandidateTensorInpaintingModel" in sources[actual["source_bundle"][0]["code_ref"]]
    assert "historical_structure_rule" in payload["evolution_protocol"]


def test_structure_refuses_builtin_algorithm_and_snapshot_disagreement(tmp_path):
    record = _archive(tmp_path)
    record["algorithm"] = "tsvd"
    assert registry.champion_structure_reference(record)["status"] == "unavailable"


def test_missing_verified_design_is_not_labeled_verified(tmp_path):
    record = _archive(tmp_path, "candidate")
    (Path(record["archive_dir"]) / "idea.json").unlink()
    reference = registry.champion_structure_reference(record)
    assert reference["status"] == "unavailable"
    assert "proposal missing" in reference["error"]


@pytest.mark.parametrize("manifest", [None, {"version": 1, "files": []}])
def test_malformed_snapshot_is_explicitly_unavailable(tmp_path, manifest):
    record = _archive(tmp_path)
    record["model_snapshot"] = manifest
    # None is also a legacy marker, so use a truthy invalid non-dictionary value.
    if manifest is None:
        record["model_snapshot"] = "invalid"
    reference = registry.champion_structure_reference(record)
    assert reference["status"] == "unavailable"
