"""Small deterministic registry for built-in tensor models."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from .base import BaseTensorInpaintingModel
from .block_term import BlockTermDecomposition
from .cp import CPDecomposition
from .hierarchical_tucker import HierarchicalTuckerDecomposition
from .matrix_factorization import MatrixFactorization
from .mode3_factorization import Mode3Factorization
from .nonnegative_cp import NonnegativeCPDecomposition
from .nonnegative_tucker import NonnegativeTuckerDecomposition
from .siren import SirenImplicitNetwork
from .t_svd import TSVDDecomposition
from .tensor_ring import TensorRingDecomposition
from .tensor_train import TensorTrainDecomposition
from .tucker import TuckerDecomposition


MODEL_CLASSES = {
    "matrix": MatrixFactorization,
    "mode3": Mode3Factorization,
    "cp": CPDecomposition,
    "nonnegative_cp": NonnegativeCPDecomposition,
    "tucker": TuckerDecomposition,
    "btd": BlockTermDecomposition,
    "tsvd": TSVDDecomposition,
    "nonnegative_tucker": NonnegativeTuckerDecomposition,
    "hierarchical_tucker": HierarchicalTuckerDecomposition,
    "tt": TensorTrainDecomposition,
    "tensor_ring": TensorRingDecomposition,
    "siren": SirenImplicitNetwork,
}


def get_default_hyperparameters(model_name: str) -> Dict[str, Any]:
    defaults = {
        "matrix": {"rank": 16, "init_scale": 0.1},
        "mode3": {"rank": 2, "init_scale": 0.1},
        "cp": {"rank": 12, "init_scale": 0.2},
        "nonnegative_cp": {"rank": 12, "init_scale": 0.1},
        "tucker": {"rank_h": 16, "rank_w": 16, "rank_c": 3, "init_scale": 0.15},
        "btd": {
            "num_blocks": 2,
            "rank_h": 8,
            "rank_w": 8,
            "rank_c": 2,
            "init_scale": 0.1,
        },
        "tsvd": {"rank": 8, "init_scale": 0.1},
        "nonnegative_tucker": {
            "rank_h": 8,
            "rank_w": 8,
            "rank_c": 2,
            "init_scale": 0.1,
        },
        "hierarchical_tucker": {
            "rank_h": 8,
            "rank_w": 8,
            "rank_c": 3,
            "rank_spatial": 2,
            "init_scale": 0.1,
        },
        "tt": {"rank_1": 8, "rank_2": 3, "init_scale": 0.1},
        "tensor_ring": {"rank": 4, "init_scale": 0.1},
        "siren": {
            "hidden_features": 128,
            "hidden_layers": 3,
            "first_omega_0": 30.0,
            "hidden_omega_0": 30.0,
        },
    }
    if model_name not in defaults:
        raise ValueError("unknown model_name %r" % model_name)
    return dict(defaults[model_name])


def create_model(
    model_name: str,
    image_shape: Tuple[int, ...],
    initial_channel_mean: Sequence[float],
    hyperparameters: Dict[str, Any],
) -> BaseTensorInpaintingModel:
    if model_name not in MODEL_CLASSES:
        raise ValueError("unknown model_name %r" % model_name)
    model_class = MODEL_CLASSES[model_name]
    return model_class(
        image_shape=image_shape,
        initial_channel_mean=initial_channel_mean,
        **hyperparameters,
    )
