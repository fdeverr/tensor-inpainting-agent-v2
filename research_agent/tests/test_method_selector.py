import json

import pytest

from research_agent.knowledge import LocalKnowledgeRetriever
from research_agent.method_selector import (
    METHOD_SELECTION_VISUAL_SOURCE,
    MethodSelector,
    manual_method_plan,
)


def _profile():
    return {
        "image_shape": [64, 128, 3],
        "mask_type": "block",
        "actual_missing_rate": 0.4,
        "missing_component_count": 1,
        "largest_missing_component_image_ratio": 0.4,
        "image_aspect_ratio": 2.0,
        "visible_mean_absolute_channel_correlation": 0.82,
        "visible_local_smoothness_score": 0.76,
        "visible_high_frequency_energy_ratio": 0.04,
    }


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


def _valid_output(source):
    return json.dumps(
        {
            "method": "tucker",
            "reason": (
                "The block mask, high channel correlation, and anisotropic image "
                "support separate Tucker spatial and channel ranks."
            ),
            "evidence": [
                {
                    "source": source,
                    "claim": "The retrieved evidence supports separate spatial ranks.",
                }
            ],
            "confidence": 0.81,
            "suggested_hyperparameters": {
                "rank_h_candidates": [8, 12],
                "rank_w_candidates": [12, 16],
                "rank_c_candidates": [2, 3],
                "init_scale": 0.15,
            },
            "risks": ["The contiguous hole may still show global banding."],
            "selection_mode": "llm",
        }
    )


def _invalid_tucker_hyperparameters(source):
    payload = json.loads(_valid_output(source))
    payload["suggested_hyperparameters"] = {
        "rank": [8, 16, 3],
        "spatial_rank_candidates": [4, 8, 16],
        "channel_rank_candidates": [1, 2, 3],
    }
    return json.dumps(payload)


def _retrieval():
    return LocalKnowledgeRetriever().retrieve(
        profile=_profile(),
        query="block channel correlation spatial rank failure modes",
        top_k=6,
    )


def _visual_assessment():
    return {
        "status": "completed",
        "assessment": {
            "visible_structure_summary": "存在明显的横向重复纹理。",
            "spatial_complexity": "medium",
            "spatial_anisotropy": "width_dominant",
            "texture_complexity": "high",
            "repetition_or_periodicity": "high",
            "channel_coupling": "strong",
            "interpolation_artifacts": ["局部过度平滑"],
            "preferred_methods": ["tucker", "btd"],
            "rank_regime": {
                "height": "medium",
                "width": "high",
                "feature": "low",
                "overall": "medium",
            },
            "rationale": ["方向性明显"],
            "confidence": "medium",
            "limitations": ["插值可能低估高频结构"],
        },
    }


def test_visual_structure_prior_is_exposed_and_may_be_cited():
    retrieval = _retrieval()
    output = json.loads(_valid_output(retrieval["evidence"][0]["source"]))
    output["evidence"] = [
        {
            "source": METHOD_SELECTION_VISUAL_SOURCE,
            "claim": "The interpolation preview suggests anisotropic spatial rank needs.",
        }
    ]
    llm = FakeLLM([json.dumps(output)])

    result = MethodSelector(llm).select(
        _profile(), retrieval, visual_assessment=_visual_assessment()
    )

    assert result["plan"].method == "tucker"
    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert payload["interpolation"]["visual_description_available"] is True
    assert payload["interpolation_visual_structure_assessment"][
        "rank_regime"
    ]["width"] == "high"
    assert METHOD_SELECTION_VISUAL_SOURCE in payload["allowed_evidence_sources"]


def test_local_retrieval_returns_rules_and_sourced_chunks():
    retrieval = _retrieval()
    assert retrieval["active_rules"]
    assert retrieval["evidence"]
    assert all("#" in item["source"] for item in retrieval["evidence"])
    assert any(rule["prefer"] == "tucker" for rule in retrieval["active_rules"])
    assert any(rule["prefer"] == "mode3" for rule in retrieval["active_rules"])
    assert any(
        rule["prefer"] == "hierarchical_tucker"
        for rule in retrieval["active_rules"]
    )

    expanded = LocalKnowledgeRetriever().retrieve(
        profile=_profile(),
        query="A mode-3 E channel subspace factorization",
        top_k=30,
    )
    assert any(item["method"] == "mode3" for item in expanded["evidence"])


def test_valid_llm_method_plan_is_accepted_stably():
    retrieval = _retrieval()
    source = retrieval["evidence"][0]["source"]
    llm = FakeLLM([_valid_output(source)])
    result = MethodSelector(llm).select(_profile(), retrieval)
    assert result["plan"].method == "tucker"
    assert result["plan"].selection_mode == "llm"
    assert result["attempts"] == 1
    assert llm.calls[0]["kwargs"]["temperature"] == 0.0


def test_valid_tt_method_plan_is_accepted():
    retrieval = LocalKnowledgeRetriever().retrieve(
        profile=_profile(),
        query="Tensor Train TT rank anisotropic mode ordering",
        top_k=20,
    )
    source = next(
        item["source"] for item in retrieval["evidence"] if item["method"] == "tt"
    )
    output = json.dumps(
        {
            "method": "tt",
            "reason": (
                "The anisotropic image motivates a parameter-efficient Tensor Train "
                "whose two unfolding ranks remain explicitly bounded."
            ),
            "evidence": [
                {
                    "source": source,
                    "claim": "The retrieved evidence supports TT for anisotropic modes.",
                }
            ],
            "confidence": 0.72,
            "suggested_hyperparameters": {
                "rank_1_candidates": [4, 8, 16],
                "rank_2_candidates": [2, 3],
                "init_scale": 0.1,
            },
            "risks": ["The chain mode ordering can introduce directional bias."],
            "selection_mode": "llm",
        }
    )

    result = MethodSelector(FakeLLM([output])).select(_profile(), retrieval)

    assert result["plan"].method == "tt"
    assert result["plan"].suggested_hyperparameters["rank_2_candidates"] == [2, 3]


def test_manual_tensor_ring_plan_uses_bounded_defaults():
    plan = manual_method_plan("tensor_ring", _profile())

    assert plan.method == "tensor_ring"
    assert plan.selection_mode == "manual"
    assert plan.confidence == 1.0
    assert plan.evidence[0].source == "user_cli:--base-model"
    assert max(plan.suggested_hyperparameters["rank_candidates"]) <= 16


def test_manual_mode3_plan_uses_rgb_channel_rank_candidates():
    plan = manual_method_plan("mode3", _profile())

    assert plan.method == "mode3"
    assert plan.selection_mode == "manual"
    assert plan.suggested_hyperparameters == {
        "rank_candidates": [1, 2, 3],
        "init_scale": 0.1,
    }


@pytest.mark.parametrize(
    "method,required_keys",
    (
        (
            "nonnegative_cp",
            {"rank_candidates", "init_scale"},
        ),
        (
            "btd",
            {
                "num_blocks_candidates",
                "rank_h_candidates",
                "rank_w_candidates",
                "rank_c_candidates",
                "init_scale",
            },
        ),
        ("tsvd", {"rank_candidates", "init_scale"}),
        (
            "nonnegative_tucker",
            {
                "rank_h_candidates",
                "rank_w_candidates",
                "rank_c_candidates",
                "init_scale",
            },
        ),
        (
            "hierarchical_tucker",
            {
                "rank_h_candidates",
                "rank_w_candidates",
                "rank_c_candidates",
                "rank_spatial_candidates",
                "init_scale",
            },
        ),
    ),
)
def test_manual_new_method_plans_use_method_specific_defaults(method, required_keys):
    plan = manual_method_plan(method, _profile())

    assert plan.method == method
    assert set(plan.suggested_hyperparameters) == required_keys


def test_invalid_json_is_repaired_once():
    retrieval = _retrieval()
    source = retrieval["evidence"][0]["source"]
    llm = FakeLLM(["not JSON", _valid_output(source)])
    result = MethodSelector(llm).select(_profile(), retrieval)
    assert result["plan"].selection_mode == "llm_repaired"
    assert result["attempts"] == 2
    assert len(result["validation_errors"]) == 1
    assert "previous JSON was invalid" in llm.calls[1]["messages"][-1]["content"]


def test_unconsumable_hyperparameter_aliases_are_repaired_once():
    retrieval = _retrieval()
    source = retrieval["evidence"][0]["source"]
    llm = FakeLLM(
        [_invalid_tucker_hyperparameters(source), _valid_output(source)]
    )
    result = MethodSelector(llm).select(_profile(), retrieval)

    assert result["plan"].selection_mode == "llm_repaired"
    assert result["attempts"] == 2
    assert "missing required fields" in result["validation_errors"][0]
    assert result["plan"].suggested_hyperparameters["rank_h_candidates"] == [8, 12]


def test_prompt_exposes_method_specific_hyperparameter_contract():
    retrieval = _retrieval()
    messages = MethodSelector._prompt(_profile(), retrieval)
    payload = json.loads(messages[1]["content"])

    contract = payload["hyperparameter_contract"]["tucker"]
    assert contract["required"] == [
        "rank_h_candidates",
        "rank_w_candidates",
        "rank_c_candidates",
    ]
    assert payload["hyperparameter_contract"]["tt"]["required"] == [
        "rank_1_candidates",
        "rank_2_candidates",
    ]
    assert payload["hyperparameter_contract"]["tensor_ring"]["required"] == [
        "rank_candidates"
    ]
    assert payload["hyperparameter_contract"]["mode3"]["required"] == [
        "rank_candidates"
    ]
    assert payload["hyperparameter_contract"]["nonnegative_cp"]["required"] == [
        "rank_candidates"
    ]
    assert payload["hyperparameter_contract"]["btd"]["required"] == [
        "num_blocks_candidates",
        "rank_h_candidates",
        "rank_w_candidates",
        "rank_c_candidates",
    ]
    assert payload["hyperparameter_contract"]["tsvd"]["required"] == [
        "rank_candidates"
    ]
    assert payload["hyperparameter_contract"]["hierarchical_tucker"][
        "required"
    ] == [
        "rank_h_candidates",
        "rank_w_candidates",
        "rank_c_candidates",
        "rank_spatial_candidates",
    ]


def test_prompt_filters_full_input_metadata_from_llm_context():
    profile = {
        **_profile(),
        "source_metadata": {
            "source_path": "/private/data/complete_ground_truth.png",
            "source_format": ".png",
            "original_dtype": "uint8",
            "original_min": 0.0,
            "original_max": 255.0,
            "original_shape": [64, 128, 3],
            "loaded_shape": [64, 128, 3],
        },
    }

    payload = json.loads(MethodSelector._prompt(profile, _retrieval())[1]["content"])
    llm_metadata = payload["image_profile"]["source_metadata"]

    assert llm_metadata["source_format"] == ".png"
    assert llm_metadata["original_shape"] == [64, 128, 3]
    assert "source_path" not in llm_metadata
    assert "original_min" not in llm_metadata
    assert "original_max" not in llm_metadata


def test_two_invalid_outputs_use_deterministic_fallback():
    retrieval = _retrieval()
    llm = FakeLLM(["bad", '{"method": "not-supported"}'])
    result = MethodSelector(llm).select(_profile(), retrieval)
    assert result["plan"].method in {
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
    }
    assert result["plan"].selection_mode == "deterministic_fallback"
    assert result["fallback_reason"] is not None
    assert result["attempts"] == 2


def test_transport_failure_retries_without_index_error():
    retrieval = _retrieval()
    llm = FakeLLM([RuntimeError("service unavailable"), RuntimeError("timeout")])
    result = MethodSelector(llm).select(_profile(), retrieval)

    assert result["plan"].selection_mode == "deterministic_fallback"
    assert result["attempts"] == 2
    assert result["raw_outputs"] == []
    assert result["validation_errors"] == [
        "RuntimeError: service unavailable",
        "RuntimeError: timeout",
    ]
    assert "failed before returning output" in llm.calls[1]["messages"][-1]["content"]


def test_required_mode_reports_underlying_llm_failures():
    retrieval = _retrieval()
    llm = FakeLLM([RuntimeError("service unavailable"), RuntimeError("timeout")])

    with pytest.raises(
        RuntimeError,
        match="required LLM method selection failed.*service unavailable.*timeout",
    ):
        MethodSelector(llm, require_valid_llm_output=True).select(
            _profile(), retrieval
        )


def test_same_profile_has_stable_deterministic_plan():
    retrieval = _retrieval()
    first = MethodSelector(None).select(_profile(), retrieval)["plan"]
    second = MethodSelector(None).select(_profile(), retrieval)["plan"]
    assert first.model_dump() == second.model_dump()
