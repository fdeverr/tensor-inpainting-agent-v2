"""Day 4 workflow with retrieval-augmented method selection."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .knowledge import LocalKnowledgeRetriever
from .core.audio_metrics import active_audio_metadata, audio_metric_context, audio_metadata_for_gt, metric_score, trial_selection_loss
from .core.data import load_tensor_data, load_observation_mask
from .core.siren_config import SIREN_LEARNING_RATES, siren_coordinate_mode, siren_tuning_candidates
from .dataset_evolution import evaluate_across_cases, cohort_score
from .method_selector import (
    MethodSelector,
    _default_hyperparameters,
    _spatial_feature_shape,
    _strongest_fixed_tensor_baseline,
    llm_from_environment,
    manual_method_plan,
)
from .schemas import SUPPORTED_TENSOR_MODEL_NAMES
from .visual_evaluator import (
    MultimodalQualityEvaluator,
    visual_llm_from_environment,
)
from .workflow import Day3Workflow, Day3WorkflowConfig, _write_json


SELECTION_QUERY = (
    "Choose matrix, A mode-3 E, CP, Nonnegative CP, Tucker, BTD, t-SVD, "
    "Nonnegative Tucker, "
    "Hierarchical Tucker, Tensor Train, or Tensor Ring "
    "decomposition for color image, MSI, video, or framed audio waveform tensor recovery "
    "using missing pattern, channel/feature correlation, local smoothness, high frequency, "
    "spatial anisotropy, rank guidance, limitations, and failure modes."
)


@dataclass(frozen=True)
class Day4WorkflowConfig(Day3WorkflowConfig):
    """Day 4 adds selector controls and research-grade training budgets."""

    max_steps: int = 1500
    max_steps_ceiling: int = 6000
    tuning_near_limit_ratio: float = 0.9
    tuning_expansion_factor: float = 2.0
    model_name: str = "auto"
    llm_mode: str = "auto"
    retrieval_top_k: int = 8
    selection_visual_assessment: bool = False
    method_shortlist_size: int = 3
    screening_cases: Optional[List[Dict[str, Any]]] = None
    dataset_algorithm_reference: Optional[Dict[str, Any]] = None
    screening_trials: int = 2
    screening_max_steps: int = 400
    screening_patience: int = 10
    siren_comparison: bool = True
    siren_max_steps: int = 4000
    siren_tuning_trials: int = 4
    siren_learning_rate_candidates: tuple[float, ...] = SIREN_LEARNING_RATES
    siren_baseline_reference: Optional[Dict[str, Any]] = None
    siren_validation_interval: int = 25
    siren_patience: int = 20

    def validate(self) -> None:
        if not Path(self.image_path).is_file():
            raise ValueError("image_path does not point to a file: %s" % self.image_path)
        if self.mask_type not in {"random", "block", "slices", "sildes"}:
            raise ValueError("mask_type must be random or block")
        if not 0.0 < self.missing_rate < 1.0:
            raise ValueError("missing_rate must be strictly between 0 and 1")
        if self.image_size is not None and self.image_size < 8:
            raise ValueError("image_size must be at least 8 or None")
        if self.mat_key is not None and not self.mat_key.strip():
            raise ValueError("mat_key must be a non-empty string or None")
        if self.model_name not in {"auto", *SUPPORTED_TENSOR_MODEL_NAMES}:
            raise ValueError(
                "model_name must be auto or one of %s"
                % sorted(SUPPORTED_TENSOR_MODEL_NAMES)
            )
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.max_steps_ceiling < self.max_steps:
            raise ValueError("max_steps_ceiling must be at least max_steps")
        if not 0.0 < self.tuning_near_limit_ratio <= 1.0:
            raise ValueError("tuning_near_limit_ratio must be in (0, 1]")
        if self.tuning_expansion_factor <= 1.0:
            raise ValueError("tuning_expansion_factor must be greater than 1")
        if self.validation_interval < 1 or self.patience < 1:
            raise ValueError("validation_interval and patience must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not isinstance(self.full_reference_metrics, bool):
            raise ValueError("full_reference_metrics must be a bool")
        if not isinstance(self.no_reference_metrics, bool):
            raise ValueError("no_reference_metrics must be a bool")
        if self.llm_mode not in {"auto", "off", "required"}:
            raise ValueError("llm_mode must be auto, off, or required")
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")
        if not isinstance(self.selection_visual_assessment, bool):
            raise ValueError("selection_visual_assessment must be a bool")
        if not 3 <= self.method_shortlist_size <= 5:
            raise ValueError("method_shortlist_size must be in [3, 5]")
        if not 1 <= self.screening_trials <= 3:
            raise ValueError("screening_trials must be in [1, 3]")
        if self.screening_max_steps < 1:
            raise ValueError("screening_max_steps must be positive")
        if self.screening_patience < 1:
            raise ValueError("screening_patience must be positive")
        if not isinstance(self.siren_comparison, bool):
            raise ValueError("siren_comparison must be a bool")
        if self.siren_max_steps < 1:
            raise ValueError("siren_max_steps must be positive")
        if not 1 <= self.siren_tuning_trials <= 4:
            raise ValueError("siren_tuning_trials must be in [1, 4]")
        if not self.siren_learning_rate_candidates or any(isinstance(rate, bool) or not 1e-5 <= rate <= 1 for rate in self.siren_learning_rate_candidates):
            raise ValueError("invalid SIREN learning-rate candidates")
        if self.siren_validation_interval < 1 or self.siren_patience < 1:
            raise ValueError(
                "siren_validation_interval and siren_patience must be positive"
            )


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


def _learning_rate_candidates(suggestions: Dict[str, Any]) -> List[float]:
    """Resolve a safe log-scale learning-rate grid for joint tuning."""

    raw_values = suggestions.get("learning_rate_candidates")
    if not isinstance(raw_values, list) or not raw_values:
        center = suggestions.get("learning_rate")
        if isinstance(center, bool) or not isinstance(center, (int, float)):
            raw_values = [0.001, 0.01, 0.1]
        else:
            center = float(center)
            raw_values = [center / 10.0, center, center * 10.0]
    rates: List[float] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        rate = float(value)
        if 1e-5 <= rate <= 1.0 and rate not in rates:
            rates.append(rate)
    return rates[:4] or [0.001, 0.01, 0.1]


def _joint_rank_learning_rate_candidates(
    hyperparameter_candidates: List[Dict[str, Any]],
    learning_rates: List[float],
) -> List[Dict[str, Any]]:
    """Order the bounded Cartesian grid so short screens still see diversity."""

    ordered_pairs = []
    seen = set()
    diagonal_count = max(len(hyperparameter_candidates), len(learning_rates))
    for index in range(diagonal_count):
        pair = (
            index % len(hyperparameter_candidates),
            index % len(learning_rates),
        )
        if pair not in seen:
            seen.add(pair)
            ordered_pairs.append(pair)
    for hyperparameter_index in range(len(hyperparameter_candidates)):
        for learning_rate_index in range(len(learning_rates)):
            pair = (hyperparameter_index, learning_rate_index)
            if pair not in seen:
                seen.add(pair)
                ordered_pairs.append(pair)
    return [
        {
            "hyperparameters": dict(hyperparameter_candidates[hyperparameter_index]),
            "learning_rate": learning_rates[learning_rate_index],
        }
        for hyperparameter_index, learning_rate_index in ordered_pairs
    ]


def _candidates_from_plan(
    plan: Dict[str, Any],
    profile: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Convert untrusted but schema-valid rank suggestions to a bounded grid."""

    method = plan["method"]
    suggestions = plan["suggested_hyperparameters"]
    learning_rates = _learning_rate_candidates(suggestions)
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
        return _joint_rank_learning_rate_candidates(
            [
                {"rank": rank, "init_scale": init_scale}
                for rank in ranks[:3]
            ],
            learning_rates,
        )

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
        return _joint_rank_learning_rate_candidates(
            [
                {
                    "rank_1": rank_1[min(index, len(rank_1) - 1)],
                    "rank_2": rank_2[min(index, len(rank_2) - 1)],
                    "init_scale": init_scale,
                }
                for index in range(count)
            ],
            learning_rates,
        )

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
        candidates.append(hyperparameters)
    return (
        _joint_rank_learning_rate_candidates(candidates, learning_rates)
        if candidates
        else None
    )


def _siren_tuning_candidates(trial_count: int, coordinate_mode="image", learning_rates=SIREN_LEARNING_RATES) -> List[Dict[str, Any]]:
    """Return a compact, deliberately diverse SIREN tuning design."""

    candidates = siren_tuning_candidates(trial_count, coordinate_mode, learning_rates)
    audio = active_audio_metadata()
    if coordinate_mode == "audio" and audio:
        for candidate in candidates:
            candidate["hyperparameters"]["sample_count"] = audio["sample_count"]
    return candidates


class Day4Workflow(Day3Workflow):
    """Replace Day 3's fixed choice with retrieval and a validated selector."""

    run_prefix = "day4"
    workflow_name = "day4_retrieval_augmented_method_selection"

    def __init__(
        self,
        config: Day4WorkflowConfig,
        selector: Optional[MethodSelector] = None,
        retriever: Optional[LocalKnowledgeRetriever] = None,
        visual_evaluator: Optional[MultimodalQualityEvaluator] = None,
    ) -> None:
        config.validate()
        self.retriever = retriever or LocalKnowledgeRetriever()
        self.selector = selector or MethodSelector(
            llm_from_environment(config.llm_mode),
            require_valid_llm_output=config.llm_mode == "required",
        )
        self.llm_used = self.selector.llm is not None
        self.visual_evaluator = visual_evaluator or MultimodalQualityEvaluator(
            visual_llm_from_environment(self.selector.llm)
            if config.selection_visual_assessment
            else None
        )
        super().__init__(config)

    def _observe_interpolation_for_selection(self) -> Dict[str, Any]:
        """Inspect interpolation once, before automatic method/rank selection."""

        if not self.config.selection_visual_assessment:
            return {
                "status": "skipped",
                "reason": "selection visual assessment is disabled",
                "scope": "manhattan_interpolation_preview_only",
                "ground_truth_provided": False,
                "used_for": "method_selection_and_coarse_rank_prior",
            }
        if self.config.model_name != "auto":
            return {
                "status": "skipped",
                "reason": "base decomposition was explicitly selected by the user",
                "scope": "manhattan_interpolation_preview_only",
                "ground_truth_provided": False,
                "used_for": "method_selection_and_coarse_rank_prior",
            }
        interpolation_preview = self.state["artifacts"].get(
            "interpolation_preview"
        )
        if not interpolation_preview:
            return {
                "status": "failed",
                "reason": "Manhattan interpolation preview is unavailable",
                "scope": "manhattan_interpolation_preview_only",
                "ground_truth_provided": False,
                "used_for": "method_selection_and_coarse_rank_prior",
            }
        profile = self.state["results"]["image_profile"]
        context_keys = (
            "data_type",
            "image_shape",
            "feature_shape",
            "feature_count",
            "mask_type",
            "actual_missing_rate",
            "missing_component_count",
            "largest_missing_component_image_ratio",
            "image_aspect_ratio",
            "visible_mean_absolute_channel_correlation",
            "visible_local_smoothness_score",
            "visible_high_frequency_energy_ratio",
        )
        tensor_context = {
            key: profile.get(key) for key in context_keys if key in profile
        }
        tensor_context["information_scope"] = (
            "visible-pixel statistics plus one interpolation preview; no ground truth "
            "or evaluation metrics"
        )
        return self.visual_evaluator.assess_method_selection(
            interpolation_path=interpolation_preview,
            tensor_context=tensor_context,
        )

    def _select_method(self) -> Dict[str, Any]:
        profile = self.state["results"]["image_profile"]
        visual_assessment = self._observe_interpolation_for_selection()
        visual_path = self.run_dir / "method_selection_visual.json"
        _write_json(visual_path, visual_assessment)
        self.state["artifacts"]["method_selection_visual_assessment"] = str(
            visual_path
        )
        self.state["results"]["method_selection_visual_assessment"] = (
            visual_assessment
        )
        self.trace.log_event(
            "method_selection_visual_assessment",
            {
                "status": visual_assessment.get("status"),
                "attempts": visual_assessment.get("attempts", 0),
                "scope": visual_assessment.get("scope"),
                "ground_truth_provided": False,
            },
            step=self.step,
        )
        retrieval = self.retriever.retrieve(
            profile=profile,
            query=SELECTION_QUERY,
            top_k=self.config.retrieval_top_k,
        )
        if self.config.model_name == "auto":
            selection = self.selector.select(
                profile=profile,
                retrieval=retrieval,
                visual_assessment=visual_assessment,
                shortlist_size=self.config.method_shortlist_size,
                comparison_reference=self.config.dataset_algorithm_reference,
            )
        else:
            selection = {
                "plan": manual_method_plan(self.config.model_name, profile),
                "attempts": 0,
                "fallback_reason": None,
                "validation_errors": [],
                "raw_outputs": [],
            }
        plan = selection["plan"]
        selector_plan = plan.model_dump()
        plan_payload, screening = self._screen_method_shortlist(
            selector_plan,
            retrieval,
            visual_assessment,
        )

        retrieval_path = self.run_dir / "retrieval_result.json"
        method_plan_path = self.run_dir / "method_plan.json"
        screening_path = self.run_dir / "method_screening.json"
        _write_json(retrieval_path, retrieval)
        _write_json(screening_path, screening)
        _write_json(
            method_plan_path,
            {
                **plan_payload,
                "selector_attempts": selection["attempts"],
                "fallback_reason": selection["fallback_reason"],
                "validation_errors": selection["validation_errors"],
                "raw_outputs": selection["raw_outputs"],
                "selector_recommendation": selector_plan,
                "shortlist": screening["shortlist"],
                "screening_winner": screening.get("winner"),
                "screening_metric": screening.get("selection_metric"),
                "ground_truth_provided_to_selector": False,
                "ground_truth_derived_baseline_metrics_provided_to_selector": bool(
                    self.config.dataset_algorithm_reference),
                "final_metrics_provided_to_selector": False,
                "selection_visual_assessment_status": visual_assessment.get(
                    "status"
                ),
                "selection_visual_evidence_used": (
                    visual_assessment.get("status") == "completed"
                ),
            },
        )
        self.state["artifacts"]["retrieval_result"] = str(retrieval_path)
        self.state["artifacts"]["method_plan"] = str(method_plan_path)
        self.state["artifacts"]["method_screening"] = str(screening_path)
        self.state["results"]["method_screening"] = screening
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
            "selection_visual_assessment_status": visual_assessment.get(
                "status"
            ),
        }
        return plan_payload

    @staticmethod
    def _shortlist_methods(
        selector_plan: Dict[str, Any],
        retrieval: Dict[str, Any],
        visual_assessment: Dict[str, Any],
        limit: int,
        comparison_reference: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Use the LLM's complete ranked list, or fill a fallback from local evidence."""

        if selector_plan.get("selection_mode") in {"llm", "llm_repaired"}:
            llm_shortlist = selector_plan.get("shortlist", [])
            if len(llm_shortlist) == limit:
                return list(llm_shortlist)

        ranked: List[str] = []

        def add(method: Any) -> None:
            if method in SUPPORTED_TENSOR_MODEL_NAMES and method not in ranked:
                ranked.append(str(method))

        add(selector_plan["method"])
        add(_strongest_fixed_tensor_baseline(comparison_reference))
        if visual_assessment.get("status") == "completed":
            for method in visual_assessment.get("assessment", {}).get(
                "preferred_methods", []
            ):
                add(method)
        for rule in sorted(
            retrieval.get("active_rules", []),
            key=lambda item: -float(item.get("weight", 1.0)),
        ):
            add(rule.get("prefer"))
        for evidence in sorted(
            retrieval.get("evidence", []),
            key=lambda item: -float(item.get("score", 0.0)),
        ):
            add(evidence.get("method"))
        for fallback in ("tucker", "tsvd", "cp", "matrix", "tt"):
            add(fallback)
        return ranked[:limit]

    def _screen_method_shortlist(
        self,
        selector_plan: Dict[str, Any],
        retrieval: Dict[str, Any],
        visual_assessment: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Numerically screen several families before the framework is locked."""

        if self.config.model_name != "auto":
            return selector_plan, {
                "status": "skipped",
                "reason": "base decomposition was explicitly selected by the user",
                "shortlist": [selector_plan["method"]],
                "winner": selector_plan["method"],
                "ground_truth_used": True,
            }

        profile = self.state["results"]["image_profile"]
        shortlist = self._shortlist_methods(
            selector_plan,
            retrieval,
            visual_assessment,
            self.config.method_shortlist_size,
            getattr(self.config, "dataset_algorithm_reference", None),
        )
        method_plans: Dict[str, Dict[str, Any]] = {}
        screening_results = []
        shared_steps = min(self.config.max_steps, self.config.screening_max_steps)
        shared_interval = min(self.config.validation_interval, shared_steps)
        cases = getattr(self.config, "screening_cases", None)
        for method in shortlist:
            if method == selector_plan["method"]:
                method_plan = dict(selector_plan)
            else:
                method_plan = {
                    "method": method,
                    "reason": "Included in the numerical shortlist by visual or local evidence.",
                    "evidence": [],
                    "confidence": 0.0,
                    "suggested_hyperparameters": _default_hyperparameters(
                        method, profile
                    ),
                    "risks": [
                        "The family remains provisional until mask-matched numerical screening."
                    ],
                    "selection_mode": "deterministic_fallback",
                }
            method_plans[method] = method_plan
            candidates = _candidates_from_plan(method_plan, profile)
            if not candidates:
                raise RuntimeError("no bounded screening candidates for %s" % method)
            output_path = self.run_dir / "screening" / method / "tuning.json"
            record = {"method": method, "status": "failed", "artifact": str(output_path)}
            try:
                response = self._call_tool(
                    "tune_tensor_model",
                    {
                    "run_id": self.run_id,
                    "corrupted_path": self.state["artifacts"]["corrupted"],
                    "mask_path": self.state["artifacts"]["mask"],
                    "ground_truth_path": str(
                        self.run_dir / "evaluation_ground_truth.npy"
                    ),
                    "output_path": str(output_path),
                    "model_name": method,
                    "seed": self.config.seed,
                    "candidates": candidates[: self.config.screening_trials],
                    "max_steps": shared_steps,
                    "validation_interval": shared_interval,
                    "patience": self.config.screening_patience,
                    "device": self.config.device,
                    },
                )
                best = response.data["best"]
                record.update(best_validation_mse=best["best_validation_mse"],
                              best_trial=best, trial_count=response.data["trial_count"])
                if cases:
                    evaluation = evaluate_across_cases(
                        method, None, best, cases,
                        self.run_dir / "screening" / method / "whole_modality",
                        shared_steps, shared_interval, self.config.screening_patience,
                        self.config.device,
                    )
                    record["dataset_evaluation"] = evaluation
                    record["status"] = ("completed" if evaluation["summary"]["complete"]
                                        else "incomplete")
                else:
                    record["status"] = "completed"
            except Exception as error:
                record["error"] = "%s: %s" % (type(error).__name__, error)
            screening_results.append(record)

        eligible = [item for item in screening_results if item["status"] == "completed"]
        if not eligible:
            raise RuntimeError("no shortlist method completed the numerical screening on every sample")
        if cases:
            audio = cases[0]["data_type"] == "audio"
            def rank(item):
                summary = item["dataset_evaluation"]["summary"]
                return (tuple(-value for value in cohort_score(summary)), item["method"])
            winner = min(eligible, key=rank)
            winner_summary = winner["dataset_evaluation"]["summary"]
            shortlist_source = ("The LLM" if selector_plan.get("selection_mode") in
                                {"llm", "llm_repaired"} else "Local fallback rules")
            reason = ("%s won the equal-budget full-modality screening: %s. "
                      "%s chose the shortlist; measured whole-modality results chose the winner."
                      % (winner["method"], winner_summary, shortlist_source))
        else:
            winner = min(eligible, key=lambda item: (
                trial_selection_loss(item["best_trial"]), item["method"]))
            reason = ("%s won the equal-budget representative-sample screening "
                      "with missing-region GT MSE %.8f. The selector only formed the shortlist."
                      % (winner["method"], winner["best_validation_mse"]))
        winner_plan = dict(method_plans[winner["method"]])
        winner_plan.update(
            {
                "reason": reason,
                "evidence": [
                    {
                        "source": "numerical_screening:missing_region_ground_truth",
                        "claim": "%s ranked first in the shared-budget numerical screen of %s."
                                 % (winner["method"], shortlist),
                    }
                ],
                "confidence": 1.0,
                "selection_mode": "numerical_screening",
            }
        )
        return winner_plan, {
            "status": "completed",
            "shortlist": shortlist,
            "selector_recommendation": selector_plan["method"],
            "selector_recommended_methods": selector_plan.get("shortlist", []),
            "winner": winner["method"],
            "selection_metric": ("mean_missing_nmse" if cases and cases[0]["data_type"] == "audio" else
                                 "mean_missing_psnr_then_ssim" if cases else
                                 "missing_original_waveform_nmse" if active_audio_metadata() is not None else
                                 "missing_region_ground_truth_mse"),
            "selection_scope": "all_valid_samples_of_modality" if cases else "missing_region_ground_truth",
            "shared_max_steps": shared_steps,
            "trials_per_method": self.config.screening_trials,
            "validation_interval": shared_interval,
            "early_stopping_patience": self.config.screening_patience,
            "ground_truth_used": True,
            "results": screening_results,
        }

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _siren_training_cache(self, ground_truth_path: str) -> tuple[str, Path]:
        audio = active_audio_metadata()
        payload = {
            "cache_version": 4,
            "selection_protocol": "all_observed_train_missing_gt_checkpoint_selection",
            "model": "siren",
            "model_source_sha256": self._file_sha256(
                str(Path(__file__).resolve().parent / "core/models/siren.py")
            ),
            "protocol_source_sha256": {name: self._file_sha256(
                str(Path(__file__).resolve().parent / name)) for name in
                ("core/trainer.py", "core/models/base.py", "core/audio_metrics.py", "core/metrics.py")},
            "corrupted_sha256": self._file_sha256(
                self.state["artifacts"]["corrupted"]
            ),
            "mask_sha256": self._file_sha256(self.state["artifacts"]["mask"]),
            "ground_truth_sha256": self._file_sha256(
                ground_truth_path
            ),
            "audio_metadata": ({key: audio.get(key) for key in
                ("sample_count", "channels", "normalization", "original_min", "original_max", "sample_rate")}
                if audio else None),
            "seed": self.config.seed,
            "candidates": _siren_tuning_candidates(self.config.siren_tuning_trials,
                siren_coordinate_mode(self.config.data_type, self.state["results"]["image_profile"]["image_shape"]),
                self.config.siren_learning_rate_candidates),
            "max_steps": self.config.siren_max_steps,
            "validation_interval": self.config.siren_validation_interval,
            "patience": self.config.siren_patience,
            "device": self.config.device,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return fingerprint, Path(self.config.output_dir) / ".siren_cache" / fingerprint

    def _reuse_siren_reference(self, ground_truth_path, baseline_dir):
        """Reuse the representative's exact whole-modality result, never retune it."""
        reference = self.config.siren_baseline_reference
        case, result = reference["case"], reference["result"]
        audio = active_audio_metadata()
        if (case["seed"] != self.config.seed or
                (audio is not None and case["sample_count"] != audio["sample_count"]) or
                not np.array_equal(load_tensor_data(case["gt_path"]), load_tensor_data(ground_truth_path)) or
                not np.array_equal(load_observation_mask(case["mask_path"]), load_observation_mask(self.state["artifacts"]["mask"]))):
            raise ValueError("precomputed SIREN baseline does not match this GT/mask/seed")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for key, value in result["artifacts"].items():
            source = Path(value)
            target = baseline_dir / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            artifacts[key] = str(target)
        training = {key: result[key] for key in ("model_name", "hyperparameters", "learning_rate",
                    "final_train_mse", "observed_pixels_used", "parameter_count", "runtime_seconds")}
        training.update(fitted_steps=result["selected_steps"], device=self.config.device, artifacts=artifacts)
        best = {"hyperparameters": result["hyperparameters"], "learning_rate": result["learning_rate"],
                "best_step": result["selected_steps"], "best_validation_mse": result["selection_validation_mse"]}
        _write_json(baseline_dir / "tuning.json", {"status": "reused_whole_modality_selection",
                    "best": best, "configuration": reference["configuration"], "protocol": reference["protocol"]})
        return best, training

    def _run_additional_baselines(self, ground_truth_path: str) -> None:
        """Train SIREN once per data/config fingerprint and reuse it thereafter."""

        if not self.config.siren_comparison:
            self.state["results"]["siren_comparison"] = {
                "status": "skipped",
                "reason": "SIREN comparison is disabled",
            }
            return
        if self.config.siren_baseline_reference and self.config.siren_baseline_reference.get("status") == "failed":
            self.state["results"]["siren_comparison"] = dict(self.config.siren_baseline_reference)
            return

        baseline_dir = self.run_dir / "siren_baseline"
        tuning_path = baseline_dir / "tuning.json"
        fingerprint, cache_dir = self._siren_training_cache(ground_truth_path)
        cache_manifest_path = cache_dir / "manifest.json"
        cached_baseline_dir = cache_dir / "siren_baseline"
        cache_reused = bool(
            cache_manifest_path.is_file() and cached_baseline_dir.is_dir()
        )
        if self.config.siren_baseline_reference:
            best, training_data = self._reuse_siren_reference(ground_truth_path, baseline_dir)
            cache_reused = True
            print("♻️ SIREN 复用整类调参后同条件评测结果，跳过代表样本重复训练", flush=True)
        elif cache_reused:
            cache_record = json.loads(
                cache_manifest_path.read_text(encoding="utf-8")
            )
            shutil.copytree(cached_baseline_dir, baseline_dir, dirs_exist_ok=True)
            best = cache_record["selected_trial"]
            training_data = dict(cache_record["training"])
            training_data["artifacts"] = {
                key: str(baseline_dir / relative_path)
                for key, relative_path in cache_record[
                    "training_artifact_relpaths"
                ].items()
            }
            print("♻️ SIREN 已有相同数据与配置的训练缓存，跳过调参与训练", flush=True)
        else:
            tuning = self._call_tool(
                "tune_tensor_model",
                {
                    "run_id": self.run_id,
                    "corrupted_path": self.state["artifacts"]["corrupted"],
                    "mask_path": self.state["artifacts"]["mask"],
                    "ground_truth_path": ground_truth_path,
                    "output_path": str(tuning_path),
                    "model_name": "siren",
                    "seed": self.config.seed,
                    "candidates": _siren_tuning_candidates(
                        self.config.siren_tuning_trials,
                        siren_coordinate_mode(self.config.data_type, self.state["results"]["image_profile"]["image_shape"]),
                        self.config.siren_learning_rate_candidates,
                    ),
                    "max_steps": self.config.siren_max_steps,
                    "validation_interval": self.config.siren_validation_interval,
                    "patience": self.config.siren_patience,
                    "device": self.config.device,
                },
            )
            best = tuning.data["best"]
            training_data = dict(tuning.data["selected_output"])
            print(
                "🎯 SIREN 调参完成，直接使用 GT 评分最优 checkpoint：%d 步"
                % int(best["best_step"]),
                flush=True,
            )
            artifact_relpaths = {
                key: str(Path(value).resolve().relative_to(baseline_dir.resolve()))
                for key, value in training_data["artifacts"].items()
            }
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(baseline_dir, cached_baseline_dir, dirs_exist_ok=True)
            _write_json(
                cache_manifest_path,
                {
                    "fingerprint": fingerprint,
                    "selected_trial": best,
                    "training": {
                        key: value
                        for key, value in training_data.items()
                        if key != "artifacts"
                    },
                    "training_artifact_relpaths": artifact_relpaths,
                },
            )
        metrics_path = self.run_dir / "siren_baseline" / "metrics.json"
        evaluation = self._call_tool(
            "evaluate_reconstruction",
            {
                "run_id": self.run_id,
                "algorithm_name": "siren_implicit_neural_representation",
                "reconstruction_path": training_data["artifacts"]["reconstruction"],
                "ground_truth_path": ground_truth_path,
                "mask_path": self.state["artifacts"]["mask"],
                "output_path": str(metrics_path),
                "device": self.config.device,
                "full_reference_metrics": self.config.full_reference_metrics,
                "no_reference_metrics": self.config.no_reference_metrics,
            },
        )
        metric_keys = (
            "missing_nmse", "evaluation_metric", "audio_metric_status",
            "missing_mse",
            "missing_psnr",
            "full_psnr",
            "perfect_reconstruction",
            "composite_ssim",
            "lpips",
            "maniqa",
            "clip_iqa",
            "musiq",
            "learned_metric_status",
            "metric_group_status",
        )
        self.state["artifacts"].update(
            {
                "siren_tuning_result": str(tuning_path),
                "siren_metrics": str(metrics_path),
                **{
                    "siren_%s" % key: value
                    for key, value in training_data["artifacts"].items()
                },
            }
        )
        self.state["results"]["siren_comparison"] = {
            "status": "completed",
            "selected_trial": best,
            "training": {
                key: training_data[key]
                for key in (
                    "model_name",
                    "hyperparameters",
                    "learning_rate",
                    "fitted_steps",
                    "final_train_mse",
                    "observed_pixels_used",
                    "parameter_count",
                    "runtime_seconds",
                    "device",
                )
            },
            "metrics": {key: evaluation.data[key] for key in metric_keys if key in evaluation.data},
            "training_cache": {
                "reused": cache_reused,
                "source": "whole_modality_evaluation" if self.config.siren_baseline_reference else "standalone_tuning",
                "fingerprint": fingerprint,
                "cache_dir": str(cache_dir),
            },
        }
        baseline_rows = [
            {
                "algorithm": "linear_interpolation_waveform" if active_audio_metadata() else "nearest_neighbor_manhattan",
                "metrics": self.state["results"]["interpolation_metrics"],
            },
            {
                "algorithm": self.state["selected_model"],
                "metrics": self.state["results"]["tensor_metrics"],
            },
            {
                "algorithm": "siren",
                "metrics": self.state["results"]["siren_comparison"]["metrics"],
            },
        ]
        baseline_winner = max(
            baseline_rows,
            key=lambda item: metric_score(item["metrics"]),
        )
        baseline_comparison = {
            "selection_rule": (
                "lowest missing original-waveform NMSE among linear waveform interpolation, the selected tensor family, and SIREN"
                if active_audio_metadata() else
                "highest missing-region PSNR among Manhattan interpolation, "
                "the selected tensor family, and SIREN"
            ),
            "winner": baseline_winner["algorithm"],
            "results": baseline_rows,
        }
        if "missing_nmse" in baseline_winner["metrics"]:
            baseline_comparison["selection_rule"] = "lowest missing original-waveform NMSE"
        baseline_comparison_path = self.run_dir / "baseline_comparison.json"
        _write_json(baseline_comparison_path, baseline_comparison)
        self.state["artifacts"]["baseline_comparison"] = str(
            baseline_comparison_path
        )
        self.state["results"]["baseline_comparison"] = baseline_comparison

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

    def _tuning_parameter_overrides(self) -> Dict[str, Any]:
        return {
            "refine_learning_rate": True,
            "learning_rate_refinement_factor": 3.0,
            "auto_expand_steps": True,
            "max_steps_ceiling": self.config.max_steps_ceiling,
            "near_limit_ratio": self.config.tuning_near_limit_ratio,
            "expansion_factor": self.config.tuning_expansion_factor,
        }


def run_day4_workflow(
    config: Day4WorkflowConfig,
    selector: Optional[MethodSelector] = None,
    visual_evaluator: Optional[MultimodalQualityEvaluator] = None,
) -> Dict[str, Any]:
    with audio_metric_context(audio_metadata_for_gt(config.image_path)):
        return Day4Workflow(config=config, selector=selector, visual_evaluator=visual_evaluator).run()
