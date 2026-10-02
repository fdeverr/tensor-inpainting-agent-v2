"""Schema-constrained tensor-decomposition method selection."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .agent_tools.framework import TensorInpaintingLLM


LLM_SAFE_SOURCE_METADATA_FIELDS = {
    "source_format",
    "original_dtype",
    "mat_key",
    "mat_backend",
    "original_shape",
    "loaded_shape",
    "data_type",
    "feature_shape",
    "feature_count",
}

METHOD_SELECTION_VISUAL_SOURCE = "visual:manhattan_interpolation_structure"


def _comparison_evidence(reference: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expose measured scores, never tensors, filesystem paths, or checkpoints."""
    rows = []
    if not isinstance(reference, dict):
        return rows
    for key in ("fixed_baselines", "historical_champions"):
        for index, record in enumerate(reference.get(key, [])):
            if not isinstance(record, dict):
                continue
            rows.append({
                "source": "cohort:%s:%d" % (key, index),
                "algorithm": record.get("algorithm"),
                "archive_id": record.get("archive_id"),
                "summary": {name: value for name, value in
                            (record.get("summary") or {}).items() if name in {
                                "complete", "expected_count", "completed_count",
                                "mean_missing_psnr", "mean_composite_ssim",
                                "mean_missing_nmse", "perfect_count"}},
                "samples": [{"sample_index": item.get("sample_index"),
                             "status": item.get("status"),
                             "metrics": {name: value for name, value in
                                         (item.get("metrics") or {}).items() if name in {
                                             "missing_psnr", "composite_ssim", "missing_nmse"}}}
                            for item in record.get("samples", [])],
            })
    return rows


def _strongest_fixed_tensor_baseline(reference: Optional[Dict[str, Any]]) -> Optional[str]:
    """Protect the strongest measured tensor baseline from an N=3 omission."""
    candidates = []
    for row in _comparison_evidence(reference):
        if not row["source"].startswith("cohort:fixed_baselines:"):
            continue
        if row["algorithm"] not in HYPERPARAMETER_CONTRACTS:
            continue
        summary = row["summary"]
        if not summary.get("complete"):
            continue
        if summary.get("mean_missing_nmse") is not None:
            key = (0, float(summary["mean_missing_nmse"]), row["algorithm"])
        else:
            psnr = summary.get("mean_missing_psnr")
            if psnr is None and not summary.get("perfect_count"):
                continue
            key = (1, -int(summary.get("perfect_count", 0)),
                   -float(psnr) if psnr is not None else -math.inf,
                   -float(summary.get("mean_composite_ssim") or 0.0),
                   row["algorithm"])
        candidates.append((key, row["algorithm"]))
    return min(candidates)[1] if candidates else None


def llm_safe_image_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Remove raw-source paths and full-input value statistics from LLM context."""

    safe_profile = dict(profile)
    source_metadata = profile.get("source_metadata")
    if isinstance(source_metadata, dict):
        safe_profile["source_metadata"] = {
            key: value
            for key, value in source_metadata.items()
            if key in LLM_SAFE_SOURCE_METADATA_FIELDS
        }
    safe_profile["llm_information_scope"] = (
        "visible-pixel statistics and non-content tensor metadata only"
    )
    return safe_profile


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=3)
    claim: str = Field(min_length=8)


class MethodPlan(BaseModel):
    """Only validated plans may influence the training workflow."""

    model_config = ConfigDict(extra="forbid")

    method: Literal[
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
    ]
    shortlist: List[Literal[
        "matrix", "mode3", "cp", "nonnegative_cp", "tucker", "btd",
        "tsvd", "nonnegative_tucker", "hierarchical_tucker", "tt",
        "tensor_ring",
    ]] = Field(default_factory=list, max_length=5)
    reason: str = Field(min_length=20, max_length=1200)
    evidence: List[EvidenceReference] = Field(min_length=1, max_length=6)
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_hyperparameters: Dict[str, Any]
    risks: List[str] = Field(min_length=1, max_length=6)
    selection_mode: Literal[
        "llm", "llm_repaired", "deterministic_fallback", "manual"
    ] = "llm"

    @field_validator("suggested_hyperparameters") #用classmethod验证suggested_hyperparameters字段
    @classmethod
    def require_nonempty_hyperparameters(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("suggested_hyperparameters cannot be empty")
        return value

    @field_validator("shortlist")
    @classmethod
    def require_distinct_shortlist(cls, value: List[str]) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("shortlist methods must be distinct")
        return value


def _json_from_text(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("MethodPlan output must be a JSON object")
    return payload


def _validate_sources(
    plan: MethodPlan,
    retrieval: Dict[str, Any],
    visual_assessment: Optional[Dict[str, Any]] = None,
    comparison_reference: Optional[Dict[str, Any]] = None,
) -> None:
    allowed = {
        item["source"] for item in retrieval["evidence"]
    } | {rule["source"] for rule in retrieval["active_rules"]}
    if visual_assessment and visual_assessment.get("status") == "completed":
        allowed.add(METHOD_SELECTION_VISUAL_SOURCE)
    allowed.update(row["source"] for row in _comparison_evidence(comparison_reference))
    unknown = [item.source for item in plan.evidence if item.source not in allowed]
    if unknown:
        raise ValueError("MethodPlan cited unknown evidence sources: %s" % unknown) #| 是集合的并集运算符，表示合并两个集合中的所有元素。


HYPERPARAMETER_CONTRACTS = {
    "matrix": {
        "required": ["rank_candidates"],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "mode3": {
        "required": ["rank_candidates"],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "cp": {
        "required": ["rank_candidates"],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "nonnegative_cp": {
        "required": ["rank_candidates"],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "tucker": {
        "required": [
            "rank_h_candidates",
            "rank_w_candidates",
            "rank_c_candidates",
        ],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "btd": {
        "required": [
            "num_blocks_candidates",
            "rank_h_candidates",
            "rank_w_candidates",
            "rank_c_candidates",
        ],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "tsvd": {
        "required": ["rank_candidates"],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "nonnegative_tucker": {
        "required": [
            "rank_h_candidates",
            "rank_w_candidates",
            "rank_c_candidates",
        ],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "hierarchical_tucker": {
        "required": [
            "rank_h_candidates",
            "rank_w_candidates",
            "rank_c_candidates",
            "rank_spatial_candidates",
        ],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "tt": {
        "required": ["rank_1_candidates", "rank_2_candidates"],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
    "tensor_ring": {
        "required": ["rank_candidates"],
        "optional": ["learning_rate", "learning_rate_candidates", "init_scale"],
    },
}


def _spatial_feature_shape(profile: Dict[str, Any]) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in profile["image_shape"])
    if len(shape) not in (3, 4):
        raise ValueError("profile image_shape must be [H,W,C] or [H,W,T,C]")
    return shape[0], shape[1], int(math.prod(shape[2:]))


def _validate_hyperparameter_contract(
    plan: MethodPlan,
    profile: Dict[str, Any],
) -> None:
    """Reject schema-valid suggestions that the bounded tuner cannot consume."""

    contract = HYPERPARAMETER_CONTRACTS[plan.method]
    suggestions = plan.suggested_hyperparameters
    required = set(contract["required"])
    allowed = required | set(contract["optional"])
    missing = sorted(required - set(suggestions))
    unexpected = sorted(set(suggestions) - allowed)
    if missing:
        raise ValueError(
            "%s suggested_hyperparameters is missing required fields: %s"
            % (plan.method, missing)
        )
    if unexpected:
        raise ValueError(
            "%s suggested_hyperparameters contains unsupported fields: %s"
            % (plan.method, unexpected)
        )

    height, width, channels = _spatial_feature_shape(profile)
    single_rank_maximum = {
        "matrix": min(height, width * channels),
        "mode3": min(height * width, channels),
        "cp": 64,
        "nonnegative_cp": 64,
        "tsvd": min(height, width),
        "tensor_ring": min(16, height, width),
    }.get(plan.method, 64)
    maxima = {
        "rank_candidates": single_rank_maximum,
        "rank_h_candidates": int(profile["image_shape"][0]),
        "rank_w_candidates": int(profile["image_shape"][1]),
        "rank_c_candidates": channels,
        "rank_1_candidates": min(height, width * channels),
        "rank_2_candidates": min(height * width, channels),
        "num_blocks_candidates": 4,
        "rank_spatial_candidates": min(height * width, channels),
    }
    for field_name in required:
        values = suggestions[field_name]
        if not isinstance(values, list) or not values:
            raise ValueError("%s must be a non-empty list" % field_name)
        normalized = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("%s must contain integers only" % field_name)
            if value < 1 or value > maxima[field_name]:
                raise ValueError(
                    "%s values must be between 1 and %d"
                    % (field_name, maxima[field_name])
                )
            if value not in normalized:
                normalized.append(value)
        suggestions[field_name] = normalized[:4]

    if "learning_rate" in suggestions:
        learning_rate = float(suggestions["learning_rate"])
        if not 1e-5 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 1e-5 and 1.0")
    if "learning_rate_candidates" in suggestions:
        values = suggestions["learning_rate_candidates"]
        if not isinstance(values, list) or not values:
            raise ValueError("learning_rate_candidates must be a non-empty list")
        normalized_rates = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "learning_rate_candidates must contain numbers only"
                )
            rate = float(value)
            if not 1e-5 <= rate <= 1.0:
                raise ValueError(
                    "learning_rate_candidates values must be between 1e-5 and 1.0"
                )
            if rate not in normalized_rates:
                normalized_rates.append(rate)
        suggestions["learning_rate_candidates"] = normalized_rates[:4]
    if "init_scale" in suggestions:
        init_scale = float(suggestions["init_scale"])
        if not 0.0 < init_scale <= 1.0:
            raise ValueError("init_scale must be between 0 and 1.0")


def _default_hyperparameters(method: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    if method == "matrix":
        height, width, channels = _spatial_feature_shape(profile)
        max_rank = min(height, width * channels)
        return {
            "rank_candidates": sorted(
                {max(1, min(max_rank, value)) for value in (8, 16)}
            ),
            "init_scale": 0.1,
        }
    if method == "mode3":
        _, _, channels = _spatial_feature_shape(profile)
        return {
            "rank_candidates": sorted(
                {max(1, min(channels, value)) for value in (1, 2, 4, 8, 16, 32)}
            ),
            "init_scale": 0.1,
        }
    if method == "cp":
        return {"rank_candidates": [8, 12], "init_scale": 0.2}
    if method == "nonnegative_cp":
        return {"rank_candidates": [8, 12], "init_scale": 0.1}
    if method == "tsvd":
        height, width, _ = _spatial_feature_shape(profile)
        max_rank = min(height, width)
        return {
            "rank_candidates": sorted(
                {max(1, min(max_rank, value)) for value in (2, 4, 8, 16)}
            ),
            "init_scale": 0.1,
        }
    if method == "tt":
        height, width, channels = _spatial_feature_shape(profile)
        max_rank_1 = min(height, width * channels)
        max_rank_2 = min(height * width, channels)
        return {
            "rank_1_candidates": sorted(
                {max(1, min(max_rank_1, value)) for value in (4, 8, 16)}
            ),
            "rank_2_candidates": sorted(
                {max(1, min(max_rank_2, value)) for value in (1, 2, 3, 4, 8, 16)}
            ),
            "init_scale": 0.1,
        }
    if method == "tensor_ring":
        height, width, _ = _spatial_feature_shape(profile)
        max_rank = min(16, height, width)
        return {
            "rank_candidates": sorted(
                {max(1, min(max_rank, value)) for value in (2, 4, 6)}
            ),
            "init_scale": 0.1,
        }
    height, width, channels = _spatial_feature_shape(profile)
    if method == "btd":
        return {
            "num_blocks_candidates": [1, 2, 3],
            "rank_h_candidates": sorted(
                {max(1, min(height, value)) for value in (4, 8, 16)}
            ),
            "rank_w_candidates": sorted(
                {max(1, min(width, value)) for value in (4, 8, 16)}
            ),
            "rank_c_candidates": sorted(
                {max(1, min(channels, value)) for value in (1, 2, 3, 4, 8, 16)}
            ),
            "init_scale": 0.1,
        }
    if method == "hierarchical_tucker":
        return {
            "rank_h_candidates": sorted(
                {max(1, min(height, value)) for value in (4, 8, 16)}
            ),
            "rank_w_candidates": sorted(
                {max(1, min(width, value)) for value in (4, 8, 16)}
            ),
            "rank_c_candidates": [channels],
            "rank_spatial_candidates": sorted(
                {max(1, min(channels, value)) for value in (1, 2, 4, 8)}
            ),
            "init_scale": 0.1,
        }
    aspect_ratio = float(profile["image_aspect_ratio"])
    if aspect_ratio >= 1.6:
        return {
            "rank_h_candidates": sorted(
                {max(1, min(height, value)) for value in (8, 12)}
            ),
            "rank_w_candidates": sorted(
                {max(1, min(width, value)) for value in (12, 16)}
            ),
            "rank_c_candidates": sorted(
                {max(1, min(channels, value)) for value in (2, 3)}
            ),
            "init_scale": 0.15,
        }
    return {
        "rank_h_candidates": sorted(
            {max(1, min(height, value)) for value in (8, 16)}
        ),
        "rank_w_candidates": sorted(
            {max(1, min(width, value)) for value in (8, 16)}
        ),
        "rank_c_candidates": sorted(
            {max(1, min(channels, value)) for value in (2, 3)}
        ),
        "init_scale": 0.15,
    }


def deterministic_method_plan(
    profile: Dict[str, Any],
    retrieval: Dict[str, Any],
) -> MethodPlan:
    """Create a reproducible fallback from activated, weighted rules."""

    method_names = tuple(HYPERPARAMETER_CONTRACTS)
    scores = {method: 0.0 for method in method_names}
    for rule in retrieval["active_rules"]:
        scores[str(rule["prefer"])] += float(rule.get("weight", 1.0))
    method = max(method_names, key=lambda name: (scores[name], name))
    supporting_rules = [
        rule for rule in retrieval["active_rules"] if rule["prefer"] == method
    ]
    if supporting_rules:
        evidence = [
            EvidenceReference(source=rule["source"], claim=str(rule["reason"]))
            for rule in supporting_rules[:3]
        ]
        reason = " ".join(str(rule["reason"]) for rule in supporting_rules[:3])
    else:
        first_evidence = next(
            item for item in retrieval["evidence"] if item["method"] == method
        )
        evidence = [
            EvidenceReference(
                source=first_evidence["source"],
                claim="This is the highest-scoring available method evidence.",
            )
        ]
        reason = "No profile rule dominated, so the highest-scoring local evidence was used."
    total_score = sum(scores.values())
    confidence = (
        scores[method] / total_score
        if total_score > 0
        else 1.0 / len(method_names)
    )
    return MethodPlan(
        method=method,
        reason=(
            "%s Profile evidence: mask=%s, channel_correlation=%.3f, "
            "largest_hole_ratio=%.3f."
            % (
                reason,
                profile["mask_type"],
                profile["visible_mean_absolute_channel_correlation"],
                profile["largest_missing_component_image_ratio"],
            )
        ),
        evidence=evidence,
        confidence=float(min(1.0, max(0.0, confidence))),
        suggested_hyperparameters=_default_hyperparameters(method, profile),
        risks=[
            "Held-out observed pixels may not represent the artificial missing pattern.",
            "A global low-rank model may blur or stripe local structure.",
        ],
        selection_mode="deterministic_fallback",
    )


def manual_method_plan(method: str, profile: Dict[str, Any]) -> MethodPlan:
    """Create a validated plan when the caller explicitly fixes the base method."""

    if method not in HYPERPARAMETER_CONTRACTS:
        raise ValueError("unsupported manual tensor method: %s" % method)
    plan = MethodPlan(
        method=method,
        reason=(
            "The base tensor decomposition was explicitly fixed by the user, so "
            "automatic method ranking was bypassed while bounded tuning remains "
            "enabled."
        ),
        evidence=[
            EvidenceReference(
                source="user_cli:--base-model",
                claim=(
                    "The user explicitly selected %s as the base decomposition."
                    % method
                ),
            )
        ],
        confidence=1.0,
        suggested_hyperparameters=_default_hyperparameters(method, profile),
        risks=[
            "A manually fixed decomposition may be less suitable than another "
            "method for this image."
        ],
        selection_mode="manual",
    )
    _validate_hyperparameter_contract(plan, profile)
    return plan


class MethodSelector:
    """Use an optional LLM, with one repair attempt and deterministic fallback."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        require_valid_llm_output: bool = False,
    ) -> None:
        self.llm = llm
        self.require_valid_llm_output = bool(require_valid_llm_output)

    @staticmethod
    def _prompt(
        profile: Dict[str, Any],
        retrieval: Dict[str, Any],
        visual_assessment: Optional[Dict[str, Any]] = None,
        shortlist_size: int = 3,
        comparison_reference: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        allowed_sources = [
            item["source"] for item in retrieval["evidence"]
        ] + [rule["source"] for rule in retrieval["active_rules"]]
        completed_visual = bool(
            visual_assessment
            and visual_assessment.get("status") == "completed"
        )
        if completed_visual:
            allowed_sources.append(METHOD_SELECTION_VISUAL_SOURCE)
        comparison_rows = _comparison_evidence(comparison_reference)
        empirical_anchor = _strongest_fixed_tensor_baseline(comparison_reference)
        allowed_sources.extend(row["source"] for row in comparison_rows)
        schema = MethodPlan.model_json_schema()
        user_payload = {
            "fixed_prompt": (
                "请根据当前图像的缺失模式、统计特征、插值结果和张量分解经验文档，"
                "按推荐顺序提供恰好 %d 个不同的张量分解方法作为候选短名单。"
                "method 必须等于 shortlist 第一项；最终家族由同预算数值预赛决定。"
                % shortlist_size
            ),
            "shortlist_size": shortlist_size,
            "image_profile": llm_safe_image_profile(profile),
            "interpolation": {
                "available": True,
                "metrics_available_during_selection": any(
                    "interpolation" in str(row.get("algorithm", "")) or
                    "nearest_neighbor" in str(row.get("algorithm", ""))
                    for row in comparison_rows),
                "visual_description_available": completed_visual,
                "visual_evidence_source": (
                    METHOD_SELECTION_VISUAL_SOURCE if completed_visual else None
                ),
            },
            "interpolation_visual_structure_assessment": (
                visual_assessment.get("assessment")
                if completed_visual and visual_assessment is not None
                else {"status": "unavailable"}
            ),
            "active_rules": retrieval["active_rules"],
            "evidence_chunks": retrieval["evidence"],
            "measured_whole_modality_comparisons": comparison_rows,
            "required_empirical_anchor": empirical_anchor,
            "allowed_evidence_sources": allowed_sources,
            "hyperparameter_contract": {
                "rule": (
                    "Use exactly the required keys for the selected method. Optional "
                    "keys may be omitted. Do not invent aliases such as rank, "
                    "spatial_rank_candidates, or channel_rank_candidates. Prefer "
                    "learning_rate_candidates with two or three log-spaced values; "
                    "a legacy scalar learning_rate will be expanded around its center."
                ),
                **HYPERPARAMETER_CONTRACTS,
            },
            "output_schema": schema,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a tensor-decomposition method selector. Return one JSON "
                    "object only. Recommend exactly %d distinct tensor methods in ranked "
                    "shortlist, and set method to its first entry. Choose from matrix, mode3, cp, "
                    "nonnegative_cp, tucker, btd, tsvd, nonnegative_tucker, "
                    "hierarchical_tucker, tt, or tensor_ring. Here mode3 means "
                    "X = A ×₃ E; btd is Block-Term "
                    "Decomposition; tsvd is a low-tubal-rank t-product model. Every "
                    "reason must cite supplied profile values or allowed evidence sources. "
                    "If interpolation_visual_structure_assessment is available, use it only "
                    "as a coarse prior for the decomposition family and low/medium/high rank "
                    "regime. Translate that regime into two or three bounded candidates; "
                    "do not assume the interpolation is ground truth. "
                    "Do not request raw ground truth or invent unprovided metrics."
                    " If measured_whole_modality_comparisons is nonempty, use its "
                    "scores as empirical guidance and cite its source when relevant. "
                    "Prefer including strong complete methods unless there is a "
                    "concrete diversity or applicability reason not to."
                    " If required_empirical_anchor is present, include it in shortlist "
                    "so the strongest measured tensor baseline cannot be omitted."
                    " Your recommendation is not the final winner; a deterministic "
                    "equal-budget numerical screening stage makes that decision."
                ) % shortlist_size,
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                #字典转换为json字符串，ensure_ascii=False表示不转义非ASCII字符，indent=2表示缩进为2个空格
            },
        ]

    @staticmethod
    def _content(response: Any) -> str:
        if isinstance(response, str):
            content = response
        else:
            content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise ValueError("LLM response does not contain string content")
        if not content.strip():
            raise ValueError("LLM response content is empty")
        return content

    def select(
        self,
        profile: Dict[str, Any],
        retrieval: Dict[str, Any],
        visual_assessment: Optional[Dict[str, Any]] = None,
        shortlist_size: Optional[int] = None,
        comparison_reference: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if shortlist_size is not None and not 3 <= shortlist_size <= 5:
            raise ValueError("shortlist_size must be in [3, 5]")
        messages = self._prompt(profile, retrieval, visual_assessment,
                                shortlist_size or 3, comparison_reference)
        raw_outputs = []
        validation_errors = []
        attempt_count = 0
        if self.llm is not None:
            for attempt in range(2):
                current_messages = list(messages)
                if attempt == 1:
                    if raw_outputs:
                        current_messages.append(
                            {"role": "assistant", "content": raw_outputs[-1]}
                        )
                        retry_instruction = (
                            "The previous JSON was invalid: %s. Return a corrected "
                            "JSON object only; do not add new evidence sources."
                            % validation_errors[-1]
                        )
                    else:
                        retry_instruction = (
                            "The previous LLM call failed before returning output: %s. "
                            "Retry once and return a JSON object only; do not add new "
                            "evidence sources."
                            % validation_errors[-1]
                        )
                    current_messages.append(
                        {
                            "role": "user",
                            "content": retry_instruction,
                        }
                    )
                try:
                    attempt_count += 1
                    raw = self._content(
                        self.llm.invoke(current_messages, temperature=0.0)
                    )
                    raw_outputs.append(raw)
                    plan = MethodPlan.model_validate(_json_from_text(raw))
                    _validate_sources(plan, retrieval, visual_assessment,
                                      comparison_reference)
                    _validate_hyperparameter_contract(plan, profile)
                    if shortlist_size is not None and (
                        len(plan.shortlist) != shortlist_size
                        or plan.shortlist[0] != plan.method
                    ):
                        raise ValueError(
                            "LLM shortlist must contain exactly %d distinct methods "
                            "and start with method" % shortlist_size
                        )
                    empirical_anchor = _strongest_fixed_tensor_baseline(
                        comparison_reference)
                    if (shortlist_size is not None and empirical_anchor
                            and empirical_anchor not in plan.shortlist):
                        raise ValueError("LLM shortlist omitted strongest measured tensor baseline: %s"
                                         % empirical_anchor)
                    plan.selection_mode = "llm" if attempt == 0 else "llm_repaired"
                    return {
                        "plan": plan,
                        "attempts": attempt + 1,
                        "raw_outputs": raw_outputs,
                        "validation_errors": validation_errors,
                        "fallback_reason": None,
                        "messages": messages,
                    }
                except Exception as error:
                    validation_errors.append(
                        "%s: %s" % (type(error).__name__, str(error))
                    )

        if self.require_valid_llm_output and self.llm is not None:
            raise RuntimeError(
                "required LLM method selection failed after %d attempts: %s"
                % (attempt_count, " | ".join(validation_errors))
            )

        fallback_reason = (
            "LLM is not configured"
            if self.llm is None
            else "LLM output remained invalid after one repair attempt"
        )
        return {
            "plan": deterministic_method_plan(profile, retrieval),
            "attempts": attempt_count,
            "raw_outputs": raw_outputs,
            "validation_errors": validation_errors,
            "fallback_reason": fallback_reason,
            "messages": messages,
        }


def llm_from_environment(mode: str = "auto") -> Optional[TensorInpaintingLLM]:
    """Build the Tensor Inpainting Agent Framework LLM only when configuration is complete."""

    if mode not in {"auto", "off", "required"}:
        raise ValueError("llm mode must be auto, off, or required")
    if mode == "off":
        return None
    required_names = ("LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL")
    missing = [name for name in required_names if not os.getenv(name)]
    if missing:
        if mode == "required":
            raise ValueError("missing required LLM environment variables: %s" % missing)
        return None
    return TensorInpaintingLLM(temperature=0.0)
