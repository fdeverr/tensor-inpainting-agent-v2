"""Immutable champions from every complete run, including non-evolved baselines."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .candidate import candidate_builder, load_validated_candidate
from .core.models import create_model
from .recovery_data import DATA_TYPES
from .workflow import _write_json


def model_library_fingerprint():
    directory = Path(__file__).parent / "core/models"
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("._"):
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def champion_records(root, data_type):
    if data_type not in DATA_TYPES:
        raise ValueError("unsupported data_type")
    records = []
    for path in sorted((Path(root) / data_type).glob("*/champion.json")):
        record = read_json(path)
        record["archive_dir"] = str(path.parent.resolve())
        records.append(record)
    return records


def archive_champion(root, data_type, state, representative, protocol):
    winner = state["best_available"]
    day4 = read_json(state["artifacts"]["day4_state"])
    day6 = read_json(state["artifacts"]["day6_state"]) if state["artifacts"].get("day6_state") else None
    role = winner["role"]
    directory = Path(root) / data_type / state["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    model_name = winner["algorithm"]
    config = {}
    kind = "builtin"
    code_sha = None
    if role == "candidate":
        kind = "candidate"
        promotion = day6["promotion"]
        source = Path(promotion["approved_dir"])
        for name in ("model.py", "idea.json", "manifest.json", "validation.json", "approved_manifest.json"):
            shutil.copy2(source / name, directory / name)
        config = read_json(source / "best_config.json")
        code_sha = hashlib.sha256((directory / "model.py").read_bytes()).hexdigest()
    elif role == "implicit_neural_baseline":
        config = day4["results"]["siren_comparison"]["training"]
    elif role == "tensor_baseline":
        config = day4["results"]["training"]
        if day6:
            config = day6["rounds"][0]["baseline_tuning"]["best"]
    elif role == "interpolation_baseline":
        kind = "interpolation"
        source = Path(__file__).parent / "core/interpolation.py"
        shutil.copy2(source, directory / "algorithm.py")
        code_sha = hashlib.sha256((directory / "algorithm.py").read_bytes()).hexdigest()
    else:
        raise ValueError("unsupported winning role %r" % role)
    if kind == "builtin":
        from .core.models.registry import MODEL_CLASSES
        import inspect
        source = Path(inspect.getfile(MODEL_CLASSES[model_name]))
        shutil.copy2(source, directory / "model.py")
        code_sha = hashlib.sha256((directory / "model.py").read_bytes()).hexdigest()
    _write_json(directory / "best_config.json", config)
    record = {
        "archive_id": state["run_id"], "data_type": data_type, "algorithm": model_name,
        "kind": kind, "role": role, "base_method": day4["selected_model"],
        "config": {key: config[key] for key in ("hyperparameters", "learning_rate", "selected_steps", "best_step", "fitted_steps") if key in config},
        "model_library_sha256": model_library_fingerprint(), "code_sha256": code_sha,
        "representative": representative, "development_metrics": winner["metrics"],
        "protocol": protocol, "archive_dir": str(directory.resolve()),
        "source_report": state["artifacts"]["report"],
    }
    if day6 and day6.get("rounds"):
        final_round = day6["rounds"][-1]
        cohort = final_round.get("dataset_evaluation") or {}
        selected = "candidate" if final_round["judgment"]["accepted"] else "incumbent"
        if cohort.get(selected):
            record["whole_modality_evolution_summary"] = cohort[selected]["summary"]
            record["whole_modality_evolution_protocol"] = cohort[selected]["protocol"]
    # Copy the human-readable evidence so the archive does not depend on an outputs directory.
    shutil.copy2(state["artifacts"]["report"], directory / "development_report.md")
    _write_json(directory / "champion.json", record)
    return record


def champion_builder(record):
    directory = Path(record["archive_dir"])
    if record["kind"] == "interpolation":
        code = (directory / "algorithm.py").read_bytes()
        if hashlib.sha256(code).hexdigest() != record["code_sha256"]:
            raise ValueError("archived interpolation code hash mismatch")
        current_code = (Path(__file__).parent / "core/interpolation.py").read_bytes()
        if code != current_code:
            raise ValueError("interpolation implementation changed; use its original environment")
        return None
    if record["model_library_sha256"] != model_library_fingerprint():
        raise ValueError("model library changed; historical comparison requires its original environment")
    code_sha = hashlib.sha256((directory / "model.py").read_bytes()).hexdigest()
    if code_sha != record["code_sha256"]:
        raise ValueError("archived champion code hash mismatch")
    if record["kind"] == "candidate":
        candidate_class, _ = load_validated_candidate(str(directory))
        return candidate_builder(candidate_class)
    return lambda shape, mean, config: create_model(record["algorithm"], shape, mean, config)
