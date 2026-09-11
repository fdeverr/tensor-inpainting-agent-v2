"""Day 4 workflow with retrieval-augmented method selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .knowledge import LocalKnowledgeRetriever
from .method_selector import (
    MethodSelector,
    _spatial_feature_shape,
    llm_from_environment,
    manual_method_plan,
)
from .schemas import SUPPORTED_MODEL_NAMES
from .workflow import Day3Workflow, Day3WorkflowConfig, _write_json


SELECTION_QUERY = (
    "Choose matrix, A mode-3 E, CP, Nonnegative CP, Tucker, BTD, t-SVD, "
    "Nonnegative Tucker, "
    "Hierarchical Tucker, Tensor Train, or Tensor Ring "
    "decomposition for color image, MSI, or video tensor inpainting "
    "using missing pattern, channel/feature correlation, local smoothness, high frequency, "
    "spatial anisotropy, rank guidance, limitations, and failure modes."
)


@dataclass(frozen=True)
class Day4WorkflowConfig(Day3WorkflowConfig):
    """Day 4 adds selector controls while retaining the Day 3 experiment budget."""

    model_name: str = "auto"
    llm_mode: str = "auto"
    retrieval_top_k: int = 8

    def validate(self) -> None:
        if not Path(self.image_path).is_file():
            raise ValueError("image_path does not point to a file: %s" % self.image_path)
        if self.mask_type not in {"random", "block"}:
            raise ValueError("mask_type must be random or block")
        if not 0.0 < self.missing_rate < 1.0:
            raise ValueError("missing_rate must be strictly between 0 and 1")
        if self.image_size is not None and self.image_size < 8:
            raise ValueError("image_size must be at least 8 or None")
        if self.mat_key is not None and not self.mat_key.strip():
            raise ValueError("mat_key must be a non-empty string or None")
        if self.model_name not in {"auto", *SUPPORTED_MODEL_NAMES}:
            raise ValueError(
                "model_name must be auto or one of %s"
                % sorted(SUPPORTED_MODEL_NAMES)
            )
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be strictly between 0 and 1")
        if self.validation_interval < 1 or self.patience < 1:
            raise ValueError("validation_interval and patience must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not isinstance(self.learned_metrics, bool):
            raise ValueError("learned_metrics must be a bool")
        if self.llm_mode not in {"auto", "off", "required"}:
            raise ValueError("llm_mode must be auto, off, or required")
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")


def _positive_ints(value: Any, maximum: int = 64) -> List[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            integer = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= integer <= maximum and integer not in result:
            result.append(integer)
    return result[:4]


def _candidates_from_plan(
    plan: Dict[str, Any],
    profile: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Convert untrusted but schema-valid rank suggestions to a bounded grid."""

    method = plan["method"]
    suggestions = plan["suggested_hyperparameters"]
    learning_rate = float(suggestions.get("learning_rate", 0.03))
    if not 1e-5 <= learning_rate <= 1.0:
        learning_rate = 0.03
    if method in {
        "matrix",
        "mode3",
        "cp",
        "nonnegative_cp",
        "tsvd",
        "tensor_ring",
    }:
        height, width, channels = _spatial_feature_shape(profile)
        max_rank = {
            "matrix": min(height, width * channels),
            "mode3": min(height * width, channels),
            "cp": 64,
            "nonnegative_cp": 64,
            "tsvd": min(height, width),
            "tensor_ring": min(16, height, width),
        }[method]
        ranks = _positive_ints(
            suggestions.get("rank_candidates"),
            maximum=max_rank,
        )
        if not ranks:
            return None
        default_init_scale = 0.2 if method == "cp" else 0.1
        init_scale = float(suggestions.get("init_scale", default_init_scale))
        if not 0.0 < init_scale <= 1.0:
            init_scale = default_init_scale
        return [
            {
                "hyperparameters": {"rank": rank, "init_scale": init_scale},
                "learning_rate": learning_rate,
            }
            for rank in ranks[:3]
        ]

    if method == "tt":
        height, width, channels = _spatial_feature_shape(profile)
        rank_1 = _positive_ints(
            suggestions.get("rank_1_candidates"),
            maximum=min(height, width * channels),
        )
        rank_2 = _positive_ints(
            suggestions.get("rank_2_candidates"),
            maximum=min(height * width, channels),
        )
        if not rank_1 or not rank_2:
            return None
        init_scale = float(suggestions.get("init_scale", 0.1))
        if not 0.0 < init_scale <= 1.0:
            init_scale = 0.1
        count = min(3, max(len(rank_1), len(rank_2)))
        return [
            {
                "hyperparameters": {
                    "rank_1": rank_1[min(index, len(rank_1) - 1)],
                    "rank_2": rank_2[min(index, len(rank_2) - 1)],
                    "init_scale": init_scale,
                },
                "learning_rate": learning_rate,
            }
            for index in range(count)
        ]

    rank_h = _positive_ints(
        suggestions.get("rank_h_candidates"),
        maximum=int(profile["image_shape"][0]),
    )
    rank_w = _positive_ints(
        suggestions.get("rank_w_candidates"),
        maximum=int(profile["image_shape"][1]),
    )
    _, _, feature_count = _spatial_feature_shape(profile)
    rank_c = _positive_ints(
        suggestions.get("rank_c_candidates"), maximum=feature_count
    )
    if not rank_h or not rank_w or not rank_c:
        return None
    init_scale = float(suggestions.get("init_scale", 0.15))
    if not 0.0 < init_scale <= 1.0:
        init_scale = 0.15
    extra_values: Dict[str, List[int]] = {}
    if method == "btd":
        num_blocks = _positive_ints(
            suggestions.get("num_blocks_candidates"), maximum=4
        )
        if not num_blocks:
            return None
        extra_values["num_blocks"] = num_blocks
    elif method == "hierarchical_tucker":
        rank_spatial = _positive_ints(
            suggestions.get("rank_spatial_candidates"),
            maximum=min(
                int(profile["image_shape"][0]) * int(profile["image_shape"][1]),
                feature_count,
            ),
        )
        if not rank_spatial:
            return None
        extra_values["rank_spatial"] = rank_spatial
    elif method not in {"tucker", "nonnegative_tucker"}:
        return None

    candidate_lengths = [len(rank_h), len(rank_w), len(rank_c)] + [
        len(values) for values in extra_values.values()
    ]
    count = min(3, max(candidate_lengths))
    candidates = []
    for index in range(count):
        hyperparameters = {
            "rank_h": rank_h[min(index, len(rank_h) - 1)],
            "rank_w": rank_w[min(index, len(rank_w) - 1)],
            "rank_c": rank_c[min(index, len(rank_c) - 1)],
            "init_scale": init_scale,
        }
        for name, values in extra_values.items():
            hyperparameters[name] = values[min(index, len(values) - 1)]
        if (
            method == "hierarchical_tucker"
            and hyperparameters["rank_spatial"]
            > min(
                hyperparameters["rank_h"] * hyperparameters["rank_w"],
                hyperparameters["rank_c"],
            )
        ):
            continue
        candidates.append(
            {
                "hyperparameters": hyperparameters,
                "learning_rate": learning_rate,
            }
        )
    return candidates or None


class Day4Workflow(Day3Workflow):
    """Replace Day 3's fixed choice with retrieval and a validated selector."""

    run_prefix = "day4"
    workflow_name = "day4_retrieval_augmented_method_selection"

    def __init__(
        self,
        config: Day4WorkflowConfig,
        selector: Optional[MethodSelector] = None,
        retriever: Optional[LocalKnowledgeRetriever] = None,
    ) -> None:
        config.validate()
        self.retriever = retriever or LocalKnowledgeRetriever()
        self.selector = selector or MethodSelector(
            llm_from_environment(config.llm_mode),
            require_valid_llm_output=config.llm_mode == "required",
        )
        self.llm_used = self.selector.llm is not None
        super().__init__(config)

    def _select_method(self) -> Dict[str, Any]:
        profile = self.state["results"]["image_profile"]
        retrieval = self.retriever.retrieve(
            profile=profile,
            query=SELECTION_QUERY,
            top_k=self.config.retrieval_top_k,
        )
        if self.config.model_name == "auto":
            selection = self.selector.select(profile=profile, retrieval=retrieval)
        else:
            selection = {
                "plan": manual_method_plan(self.config.model_name, profile),
                "attempts": 0,
                "fallback_reason": None,
                "validation_errors": [],
                "raw_outputs": [],
            }
        plan = selection["plan"]
        plan_payload = plan.model_dump()

        retrieval_path = self.run_dir / "retrieval_result.json"
        method_plan_path = self.run_dir / "method_plan.json"
        _write_json(retrieval_path, retrieval)
        _write_json(
            method_plan_path,
            {
                **plan_payload,
                "selector_attempts": selection["attempts"],
                "fallback_reason": selection["fallback_reason"],
                "validation_errors": selection["validation_errors"],
                "raw_outputs": selection["raw_outputs"],
                "ground_truth_provided_to_selector": False,
                "final_metrics_provided_to_selector": False,
            },
        )
        self.state["artifacts"]["retrieval_result"] = str(retrieval_path)
        self.state["artifacts"]["method_plan"] = str(method_plan_path)
        self.state["results"]["retrieval_summary"] = {
            "active_rule_count": len(retrieval["active_rules"]),
            "evidence_count": len(retrieval["evidence"]),
            "evidence_sources": [item["source"] for item in retrieval["evidence"]],
        }
        self.state["results"]["selector_diagnostics"] = {
            "attempts": selection["attempts"],
            "fallback_reason": selection["fallback_reason"],
            "validation_errors": selection["validation_errors"],
            "llm_used": self.llm_used and self.config.model_name == "auto",
        }
        return plan_payload

    def _tuning_candidates(
        self,
        selected_model: str,
        method_plan: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        if self.config.candidates is not None:
            return self.config.candidates
        return _candidates_from_plan(
            method_plan,
            self.state["results"]["image_profile"],
        )


def run_day4_workflow(
    config: Day4WorkflowConfig,
    selector: Optional[MethodSelector] = None,
) -> Dict[str, Any]:
    return Day4Workflow(config=config, selector=selector).run()
