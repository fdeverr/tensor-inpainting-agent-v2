import pytest
from pydantic import ValidationError
from types import SimpleNamespace

from research_agent.ablation import (
    build_ablation_arms,
    build_final_removal_audit,
    build_pruning_audit,
    classify_ablation,
)
from research_agent.candidate.generator import CandidateGenerator, deterministic_candidate
from research_agent.candidate.schemas import CandidateProposal
from research_agent.schemas import TrainingConfig
from research_agent.workflow_day6 import Day6Workflow


def _three_component_payload():
    payload = deterministic_candidate("tucker").model_dump()
    payload.update(
        {
            "pain_point": "The incumbent loses both local edges and multi-scale texture.",
            "core_difficulty": "The evidence does not isolate representation capacity from supervision.",
            "simplified_problem": "Test separable capacity, supervision, and gating mechanisms.",
            "mutation_mode": "combination",
            "mutation_target": "mixed",
            "single_change": "Test three separable mechanisms and their full factorial ablation.",
            "proposed_changes": ["add A", "add B", "add C"],
            "components": [
                {
                    "id": "A",
                    "target": "algorithm",
                    "change": "Add a multi-scale convolutional residual branch.",
                    "expected_role": "Provide local features at several receptive fields.",
                    "implementation_switch": "enable_component_a",
                },
                {
                    "id": "B",
                    "target": "loss",
                    "change": "Add a bounded edge-consistency loss term.",
                    "expected_role": "Provide a direct optimization signal for edges.",
                    "implementation_switch": "enable_component_b",
                },
                {
                    "id": "C",
                    "target": "algorithm",
                    "change": "Add a bounded residual gate for the new branch.",
                    "expected_role": "Control coupling strength and stabilize fitting.",
                    "implementation_switch": "enable_component_c",
                },
            ],
            "interaction_hypothesis": (
                "The multi-scale branch, edge signal, and residual gate may jointly provide "
                "capacity, supervision, and stability that no component provides alone."
            ),
            "search_space": {
                "tv_weight": [0.0, 0.0001],
                "enable_component_a": [False, True],
                "enable_component_b": [False, True],
                "enable_component_c": [False, True],
            },
        }
    )
    return payload


def test_three_components_create_six_proper_ablations_plus_base_and_full():
    proposal = CandidateProposal.model_validate(_three_component_payload())
    arms = build_ablation_arms([item.model_dump() for item in proposal.components])

    assert [arm["name"] for arm in arms] == [
        "Base",
        "A",
        "B",
        "C",
        "A+B",
        "A+C",
        "B+C",
        "A+B+C",
    ]
    assert len(arms[1:-1]) == 6


def test_combination_requires_every_component_switch():
    payload = _three_component_payload()
    payload["search_space"].pop("enable_component_c")
    with pytest.raises(ValidationError, match="component switches"):
        CandidateProposal.model_validate(payload)


def test_remove_operation_requires_named_target_evidence_and_kind():
    payload = _three_component_payload()
    payload["components"][0]["operation"] = "remove"
    with pytest.raises(ValidationError, match="affected_modules"):
        CandidateProposal.model_validate(payload)

    payload["components"][0].update(
        {
            "affected_modules": ["texture_refiner"],
            "evidence": (
                "Matched ablation arms show that the texture refiner increases validation error."
            ),
            "removal_kind": "parameterized_module",
        }
    )
    proposal = CandidateProposal.model_validate(payload)
    assert proposal.components[0].operation == "remove"


def test_prompt_requires_diagnosis_simplification_and_bounded_combination():
    payload = CandidateGenerator._messages(
        {
            "base_method": "tucker",
            "base_class_name": "TuckerDecomposition",
            "mutation_visual_assessment_enabled": False,
        }
    )[1]["content"]
    assert "pain_point" in payload
    assert "simplify the difficulty" in payload
    assert "two or three independently switchable components" in payload
    assert "complete 2^N ablation" in payload


def test_ablation_classifies_positive_interaction():
    screening = {
        "arms": [
            {"name": "Base", "best_validation_mse": 1.0},
            {"name": "A", "best_validation_mse": 0.99},
            {"name": "B", "best_validation_mse": 0.99},
            {"name": "A+B", "best_validation_mse": 0.8},
        ]
    }
    assert classify_ablation(screening)["classification"] == "synergistic"


def test_pruning_audit_distinguishes_structural_prune_from_switch_off():
    screening = {
        "selected_non_base_arm": "A+B",
        "arms": [
            {"name": "Base", "enabled_components": [], "parameter_count": 100},
            {"name": "A", "enabled_components": ["A"], "parameter_count": 70},
            {"name": "B", "enabled_components": ["B"], "parameter_count": 120},
            {
                "name": "A+B",
                "enabled_components": ["A", "B"],
                "parameter_count": 90,
            },
        ],
    }
    components = [
        {
            "id": "A",
            "target": "algorithm",
            "operation": "remove",
            "removal_kind": "parameterized_module",
            "affected_modules": ["old_refiner"],
        },
        {"id": "B", "target": "algorithm", "operation": "add"},
    ]
    audit = build_pruning_audit(screening, components)
    assert audit["removals"][0]["verification"] == "confirmed_structural_prune"
    assert audit["selected_parameterized_removals_all_confirmed"] is True

    final_audit = build_final_removal_audit(
        components,
        {"parameter_count": 100},
        {"parameter_count": 90},
        selected_component_ids=["A", "B"],
    )
    assert final_audit["parameterized_removal_net_reduction_confirmed"] is True


def test_day6_screen_runs_all_eight_arms_and_locks_the_winner(monkeypatch, tmp_path):
    scores = {
        "Base": 1.0,
        "A": 0.96,
        "B": 0.97,
        "C": 0.98,
        "A+B": 0.88,
        "A+C": 0.90,
        "B+C": 0.91,
        "A+B+C": 0.75,
    }
    calls = []

    def fake_tune(*, model_name, configurations, **kwargs):
        arm = model_name.rsplit("-ablation-", 1)[-1]
        calls.append(arm)
        return {
            "best": {
                "best_validation_mse": scores[arm],
                "hyperparameters": configurations[0],
                "best_step": 10,
                "stopped_early": False,
            },
            "trial_count": 1,
            "trials": [],
        }

    monkeypatch.setattr(
        "research_agent.workflow_day6.tune_model_on_observed_pixels",
        fake_tune,
    )
    workflow = object.__new__(Day6Workflow)
    workflow.config = SimpleNamespace(
        ablation_screen_trials=1,
        ablation_screen_max_steps=30,
    )
    proposal = CandidateProposal.model_validate(_three_component_payload())
    components = [item.model_dump() for item in proposal.components]
    search_space = {
        "rank_1": [2],
        "enable_component_a": [False, True],
        "enable_component_b": [False, True],
        "enable_component_c": [False, True],
    }

    selected_space, screening = workflow._screen_combination_ablation(
        candidate_manifest={"candidate_id": "candidate-test", "components": components},
        candidate_class=object,
        candidate_search_space=search_space,
        incumbent={"name": "tucker", "builder": None, "anchor": {"rank_1": 2}},
        incumbent_search_space={"rank_1": [2]},
        observed=[[0.0]],
        observed_mask=[[True]],
        ground_truth=[[0.0]],
        training=TrainingConfig(max_steps=100),
        seed=42,
        round_dir=tmp_path,
    )

    assert calls == ["Base", "A", "B", "C", "A+B", "A+C", "B+C", "A+B+C"]
    assert screening["arm_count"] == 8
    assert screening["ground_truth_used"] is True
    assert screening["selected_non_base_arm"] == "A+B+C"
    assert selected_space["enable_component_a"] == [True]
    assert selected_space["enable_component_b"] == [True]
    assert selected_space["enable_component_c"] == [True]
