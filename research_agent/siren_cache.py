"""Persistent, content-addressed SIREN fits for whole-modality recovery."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import torch

from .core.trainer import resolve_device
from .workflow import _write_json


_SOURCE_FILES = (
    "siren_cache.py", "schemas.py", "core/siren_config.py",
    "core/models/siren.py", "core/models/base.py", "core/models/registry.py",
    "core/trainer.py", "core/data.py", "core/masks.py", "core/audio_metrics.py",
    "core/metrics.py", "core/fair_experiment.py", "recovery_data.py",
)
_REQUIRED_ARTIFACTS = {"checkpoint", "raw_reconstruction", "reconstruction", "history", "metrics"}


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + "-", suffix=".json",
                                         mode="w", encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_fingerprint():
    root = Path(__file__).resolve().parent
    return {name: _file_hash(root / name) for name in _SOURCE_FILES}


def _runtime_fingerprint(device):
    versions = {}
    for name in ("numpy", "torch", "scipy", "scikit-image", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    resolved = resolve_device(device)
    runtime = {"versions": versions, "device": str(resolved)}
    if resolved.type == "cuda":
        runtime.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(resolved))
    return runtime


def _cache_identity(case, config, hyperparameters, learning_rate, execution_source):
    # Hash actual prepared files, not unverified metadata or per-run paths. NPY
    # headers retain shape/dtype and contain no source path or timestamp.
    return {
        "version": 1,
        "selection_protocol": "all_observed_train_missing_gt_checkpoint_selection",
        "gt_sha256": _file_hash(case["gt_path"]),
        "mask_sha256": _file_hash(case["mask_path"]),
        "seed": case["seed"],
        "metadata": {key: case.get(key) for key in (
            "data_type", "tensor_axes", "normalization", "original_min", "original_max",
            "sample_count", "channels", "sample_rate", "frame_size", "padding_samples",
        )},
        "hyperparameters": hyperparameters,
        "learning_rate": learning_rate,
        "steps": config.evaluation_steps,
        "validation_interval": config.evaluation_validation_interval,
        "patience": config.evaluation_patience or config.evaluation_steps + 1,
        "runtime": _runtime_fingerprint(config.device),
        "source": _source_fingerprint(),
        "execution_source": hashlib.sha256(execution_source.encode("utf-8")).hexdigest(),
    }


def _read_cache(directory, fingerprint, identity, required):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["fingerprint"] != fingerprint or manifest["identity"] != identity:
        raise ValueError("cache identity mismatch")
    result = manifest["result"]
    if _json_hash(result) != manifest["result_sha256"]:
        raise ValueError("cached result integrity mismatch")
    artifacts = result["artifacts"]
    if (not isinstance(artifacts, dict) or not isinstance(manifest["files"], dict) or
            not required <= artifacts.keys() or artifacts.keys() != manifest["files"].keys()):
        raise ValueError("cache is missing required artifacts")
    sources = {}
    for key, relative in artifacts.items():
        path = (directory / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()):
            raise ValueError("cached artifact escapes its directory")
        expected = manifest["files"][key]
        if path.stat().st_size != expected["bytes"] or _file_hash(path) != expected["sha256"]:
            raise ValueError("cached artifact integrity mismatch: " + key)
        sources[key] = path
    return result, sources


def _restore_cache(result, sources, destination, audit):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for key, source in sources.items():
        target = destination / source.name
        shutil.copy2(source, target)
        artifacts[key] = str(target)
    restored = {**result, "artifacts": artifacts, "reused_without_retraining": True,
                "training_cache": audit}
    # The original metrics JSON contains old run paths. Publish current-run
    # paths and reuse status without changing the recorded training curves.
    history_path = Path(artifacts["history"])
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["persistent_cache_reused"] = True
    _write_json(history_path, history)
    _atomic_write_json(Path(artifacts["metrics"]), restored)
    return restored


def _publish_cache(directory, fingerprint, identity, result, required):
    artifacts = result["artifacts"]
    if not required <= artifacts.keys():
        raise ValueError("fit did not produce required cache artifacts")
    # Each publication has immutable objects; publish the manifest last with
    # atomic replace. Concurrent readers never observe partly copied files.
    objects = directory / "objects" / uuid.uuid4().hex
    objects.mkdir(parents=True, exist_ok=False)
    files, relative_artifacts = {}, {}
    for key, source_path in artifacts.items():
        source = Path(source_path)
        target = objects / (key + source.suffix)
        shutil.copy2(source, target)
        files[key] = {"bytes": target.stat().st_size, "sha256": _file_hash(target)}
        relative_artifacts[key] = str(target.relative_to(directory))
    stored = {key: value for key, value in result.items() if key != "training_cache"}
    stored["artifacts"] = relative_artifacts
    manifest = {"fingerprint": fingerprint, "identity": identity, "result": stored,
                "result_sha256": _json_hash(stored), "files": files}
    _atomic_write_json(directory / "manifest.json", manifest)


def cached_siren_fit(case, output_dir, config, hyperparameters, learning_rate,
                     execution_source, fit):
    """Return an intact identical fit, otherwise train and publish a new one."""
    identity = _cache_identity(case, config, hyperparameters, learning_rate, execution_source)
    fingerprint = _json_hash(identity)
    directory = Path(config.output_dir).resolve() / ".siren_cache" / "recovery-v1" / fingerprint
    required = _REQUIRED_ARTIFACTS | ({"audio"} if case["data_type"] == "audio" else set())
    audit = {"scope": "persistent_sample_fit", "fingerprint": fingerprint,
             "cache_dir": str(directory), "reused": False, "reason": "not_found"}
    if (directory / "manifest.json").exists():
        try:
            result, sources = _read_cache(directory, fingerprint, identity, required)
            restored = _restore_cache(result, sources, output_dir,
                                      {**audit, "reused": True, "reason": "identical_intact_fit"})
            print("♻️ SIREN 跨运行缓存命中：%s，跳过训练" % Path(case["source"]).name, flush=True)
            return restored
        except (OSError, ValueError, KeyError, TypeError) as error:
            audit.update(reason="invalid_cache", error=str(error))
            print("⚠️ SIREN 缓存无效，重新训练：%s" % error, flush=True)
    result = fit()
    result["training_cache"] = audit
    try:
        _publish_cache(directory, fingerprint, identity, result, required)
        audit["written"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        # A successful fit remains usable when the cache disk is unavailable.
        audit.update(written=False, write_error=str(error))
        print("⚠️ SIREN 训练成功，但缓存保存失败：%s" % error, flush=True)
    if result.get("artifacts", {}).get("metrics"):
        try:
            _atomic_write_json(Path(result["artifacts"]["metrics"]), result)
        except OSError as error:
            # Audit enrichment must not truncate or invalidate the successful
            # fit's already-written metrics if the output disk becomes full.
            audit["metrics_audit_write_error"] = str(error)
            print("⚠️ SIREN 训练结果已保留，缓存审计信息写入失败：%s" % error, flush=True)
    return result
