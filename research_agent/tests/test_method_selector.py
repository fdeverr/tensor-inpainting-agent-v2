import json

import pytest

from research_agent.knowledge import LocalKnowledgeRetriever
from research_agent.method_selector import MethodSelector


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


def test_local_retrieval_returns_rules_and_sourced_chunks():
    retrieval = _retrieval()
    assert retrieval["active_rules"]
    assert retrieval["evidence"]
    assert all("#" in item["source"] for item in retrieval["evidence"])
    assert any(rule["prefer"] == "tucker" for rule in retrieval["active_rules"])


def test_valid_llm_method_plan_is_accepted_stably():
    retrieval = _retrieval()
    source = retrieval["evidence"][0]["source"]
    llm = FakeLLM([_valid_output(source)])
    result = MethodSelector(llm).select(_profile(), retrieval)
    assert result["plan"].method == "tucker"
    assert result["plan"].selection_mode == "llm"
    assert result["attempts"] == 1
    assert llm.calls[0]["kwargs"]["temperature"] == 0.0


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


def test_two_invalid_outputs_use_deterministic_fallback():
    retrieval = _retrieval()
    llm = FakeLLM(["bad", '{"method": "not-supported"}'])
    result = MethodSelector(llm).select(_profile(), retrieval)
    assert result["plan"].method in {"matrix", "cp", "tucker"}
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
