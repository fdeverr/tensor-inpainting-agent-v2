"""Hash-gated dynamic loading for validated or approved candidate code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from ..core.models.base import BaseTensorInpaintingModel
from ..core.models.block_term import BlockTermDecomposition
from ..core.models.cp import CPDecomposition
from ..core.models.hierarchical_tucker import HierarchicalTuckerDecomposition
from ..core.models.matrix_factorization import MatrixFactorization
from ..core.models.mode3_factorization import Mode3Factorization
from ..core.models.nonnegative_cp import NonnegativeCPDecomposition
from ..core.models.nonnegative_tucker import NonnegativeTuckerDecomposition
from ..core.models.t_svd import TSVDDecomposition
from ..core.models.tensor_ring import TensorRingDecomposition
from ..core.models.tensor_train import TensorTrainDecomposition
from ..core.models.tucker import TuckerDecomposition


def load_validated_candidate(
    candidate_dir: str, model_namespace: Dict[str, type] | None = None,
) -> Tuple[type, Dict[str, Any]]:
    """Load only when validation passed and the code hash is unchanged."""

    directory = Path(candidate_dir)
    manifest_path = directory / "manifest.json"
    validation_path = directory / "validation.json"
    model_path = directory / "model.py"
    for path in (manifest_path, validation_path, model_path):
        if not path.is_file():
            raise ValueError("candidate artifact is missing: %s" % path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("validation_status") != "validated":
        raise ValueError("candidate manifest is not validated")
    if manifest.get("eligible_for_training") is not True:
        raise ValueError("candidate is not eligible for training")
    if validation.get("passed") is not True:
        raise ValueError("candidate validation report did not pass")
    source = model_path.read_text(encoding="utf-8")
    actual_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual_hash != manifest.get("code_sha256"):
        raise ValueError("candidate code hash differs from validated manifest")

    namespace = {
        "__name__": "candidate_%s" % manifest["candidate_id"].replace("-", "_"),
        "torch": torch,
        "BaseTensorInpaintingModel": BaseTensorInpaintingModel,
        "MatrixFactorization": MatrixFactorization,
        "Mode3Factorization": Mode3Factorization,
        "CPDecomposition": CPDecomposition,
        "NonnegativeCPDecomposition": NonnegativeCPDecomposition,
        "TuckerDecomposition": TuckerDecomposition,
        "BlockTermDecomposition": BlockTermDecomposition,
        "TSVDDecomposition": TSVDDecomposition,
        "NonnegativeTuckerDecomposition": NonnegativeTuckerDecomposition,
        "HierarchicalTuckerDecomposition": HierarchicalTuckerDecomposition,
        "TensorTrainDecomposition": TensorTrainDecomposition,
        "TensorRingDecomposition": TensorRingDecomposition,
    }
    if model_namespace is not None:
        # Historical candidates must inherit their frozen parents, not today's
        # implementations. Keep all validation/hash gates above unchanged.
        required = set(namespace) - {"__name__", "torch"}
        if not required.issubset(model_namespace):
            raise ValueError("archived candidate model namespace is incomplete")
        namespace.update({name: model_namespace[name] for name in required})
    interface = namespace["BaseTensorInpaintingModel"]
    exec(compile(source, str(model_path), "exec"), namespace, namespace)
    candidate_class = namespace.get("CandidateTensorInpaintingModel")
    if not isinstance(candidate_class, type):
        raise TypeError("CandidateTensorInpaintingModel is missing")
    if not issubclass(candidate_class, interface):
        raise TypeError("candidate does not implement BaseTensorInpaintingModel")
    return candidate_class, manifest


def candidate_builder(candidate_class: type):
    """Adapt a dynamically loaded class to the fixed Trainer builder protocol."""

    def build(image_shape, initial_channel_mean, hyperparameters):
        return candidate_class(
            image_shape=image_shape,
            initial_channel_mean=initial_channel_mean,
            **hyperparameters,
        )

    return build
