import json
from pathlib import Path

import pytest

from research_agent.knowledge import LocalKnowledgeRetriever
from research_agent.knowledge.retriever import METHOD_FILE_NAMES, canonical_data_type
from research_agent.method_selector import MethodSelector, manual_method_plan, _validate_sources, deterministic_method_plan
from research_agent.prompt_context import ContextBudget, check_text_messages


def _profile(kind, shape, mask="random"):
    return {"data_type": kind, "image_shape": shape, "mask_type": mask,
            "actual_missing_rate": .8, "image_aspect_ratio": 6.,
            "visible_mean_absolute_channel_correlation": .95,
            "visible_local_smoothness_score": .99,
            "visible_high_frequency_energy_ratio": .2,
            "largest_missing_component_image_ratio": .8, "missing_component_count": 10,
            "structure_statistics_imputation": True}


@pytest.mark.parametrize("kind,shape,canonical", [
    ("Image", [16, 20, 3], "Image"), ("color_image", [16, 20, 3], "Image"),
    ("MSI", [16, 20, 31], "MSI"), ("msi", [16, 20, 31], "MSI"),
    ("Video", [16, 20, 5, 3], "Video"), ("video", [16, 20, 5, 3], "Video"),
    ("audio", [10, 256, 1], "audio"), ("audio", [10, 256, 2], "audio")])
def test_every_type_receives_all_eleven_current_type_cards_without_cross_type_chunks(kind, shape, canonical):
    retrieval = LocalKnowledgeRetriever().retrieve(_profile(kind, shape), "rank CP Tucker TT matrix FFT", top_k=3)
    cards = retrieval["method_catalogue"]
    assert len(cards) == len(METHOD_FILE_NAMES) == 11
    assert {row["method"] for row in cards} == set(METHOD_FILE_NAMES)
    assert all(row["heading"] == canonical + " applicability" for row in cards)
    assert len(retrieval["evidence"]) == 3
    assert len({row["method"] for row in retrieval["evidence"]}) == 3
    assert not {row["source"] for row in cards} & {row["source"] for row in retrieval["evidence"]}
    assert retrieval["selection_context"]["feature_count"] == (shape[2] * shape[3] if len(shape) == 4 else shape[2])
    assert all(canonical in row["data_types"] for row in retrieval["active_rules"])


def test_audio_never_receives_rgb_aspect_smoothness_frequency_rules_even_with_high_proxies():
    retrieval = LocalKnowledgeRetriever().retrieve(_profile("audio", [30, 256, 1]), "choose tensor method", 8)
    conditions = {row["condition"] for row in retrieval["active_rules"]}
    assert "audio_mono" in conditions and "data_type_audio" in conditions
    assert not conditions & {"image_aspect_ratio_high", "local_smoothness_high", "high_frequency_high",
                             "channel_correlation_high", "channel_correlation_low", "large_contiguous_hole", "many_small_holes"}
    assert not {row["prefer"] for row in retrieval["active_rules"]} & {"nonnegative_cp", "nonnegative_tucker"}
    warnings = " ".join(retrieval["selection_context"]["warnings"])
    assert "not temporal FFT/STFT" in warnings and "signed waveform" in warnings
    assert "imputation" in warnings and "High missingness" in warnings


def test_tsvd_temporal_prior_differentiates_grayscale_video_from_joint_rgb_fft():
    retriever = LocalKnowledgeRetriever()
    gray = retriever.retrieve(_profile("video", [16, 20, 5, 1]), "FFT", 3)
    rgb = retriever.retrieve(_profile("video", [16, 20, 5, 3]), "FFT", 3)
    assert any(row["condition"] == "ordered_feature_mode" for row in gray["active_rules"])
    assert not any(row["condition"] == "ordered_feature_mode" for row in rgb["active_rules"])
    assert "mixes time and color" in " ".join(rgb["selection_context"]["warnings"])


def test_slice_alias_has_the_same_conditions_and_explicit_identifiability_warning():
    retriever = LocalKnowledgeRetriever()
    slices = retriever.retrieve(_profile("MSI", [16, 20, 31], "slices"), "slice rank", 3)
    alias = retriever.retrieve(_profile("MSI", [16, 20, 31], "sildes"), "slice rank", 3)
    assert slices["active_rules"] == alias["active_rules"]
    assert alias["selection_context"]["mask_type"] == "slices"
    assert "free-index factors" in " ".join(alias["selection_context"]["warnings"])


def test_modality_metadata_takes_precedence_over_shape_and_legacy_shape_inference_works():
    assert canonical_data_type({"image_shape": [10, 256, 1], "source_metadata": {"data_type": "audio"}}) == "audio"
    assert canonical_data_type({"image_shape": [16, 20, 3]}) == "Image"
    assert canonical_data_type({"image_shape": [16, 20, 8]}) == "MSI"
    assert canonical_data_type({"image_shape": [16, 20, 5, 3]}) == "Video"
    with pytest.raises(ValueError, match="unknown knowledge data type"):
        canonical_data_type({"data_type": "unknown"})


@pytest.mark.parametrize("kind,shape", [("Image", [16, 20, 3]), ("MSI", [16, 20, 31]),
    ("Video", [16, 20, 5, 3]), ("audio", [30, 256, 1])])
def test_all_cards_and_semantic_warnings_reach_selector_and_are_valid_citation_sources(kind, shape):
    profile = _profile(kind, shape)
    retrieval = LocalKnowledgeRetriever().retrieve(profile, "choose CP Tucker TT matrix ranks", 8)
    messages = MethodSelector._prompt(profile, retrieval)
    payload = json.loads(messages[1]["content"])
    assert payload["knowledge_selection_context"] == retrieval["selection_context"]
    assert payload["method_catalogue"] == retrieval["method_catalogue"]
    # The all-family concise catalogue does not blow the default model budget.
    check_text_messages(messages, ContextBudget())
    for card in retrieval["method_catalogue"]:
        assert card["source"] in payload["allowed_evidence_sources"]
        plan = manual_method_plan(card["method"], profile)
        plan.evidence[0].source = card["source"]
        _validate_sources(plan, retrieval)


def test_all_documents_have_modality_rank_missingness_and_evidence_sections():
    directory = Path(__file__).resolve().parents[1] / "knowledge/methods"
    for filename in METHOD_FILE_NAMES.values():
        content = (directory / filename).read_text()
        for section in ["Model and implementation", "Image applicability", "MSI applicability",
                        "Video applicability", "audio applicability", "Rank and budget",
                        "Missingness and failure modes", "Evidence boundary"]:
            assert "## " + section in content
        assert "GT 缺失区" in content and "NMSE" in content
        assert "slices" in content and "工程" in content
        assert "held-out observed" not in content.lower()
        assert "https://" in content


def test_unknown_rule_type_scope_is_rejected_instead_of_silently_recommending(tmp_path):
    (tmp_path / "selection_rules.yaml").write_text('rules:\n  - condition: random_missing\n    prefer: cp\n    data_types: [unsupported]\n')
    with pytest.raises(ValueError, match="data_types"):
        LocalKnowledgeRetriever(tmp_path).active_rules(_profile("audio", [3, 8, 1]))


def test_no_active_rule_fallback_uses_actual_available_method_evidence(monkeypatch):
    retriever = LocalKnowledgeRetriever()
    monkeypatch.setattr(retriever, "active_rules", lambda _: [])
    profile = _profile("Image", [16, 20, 3])
    retrieval = retriever.retrieve(profile, "matrix", top_k=1)
    assert retrieval["evidence"][0]["method"] == "matrix"
    plan = deterministic_method_plan(profile, retrieval)
    assert plan.method == "matrix"  # Not the lexicographically last unrelated method.
    _validate_sources(plan, retrieval)
