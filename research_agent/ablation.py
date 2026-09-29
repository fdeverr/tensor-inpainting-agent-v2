"""Deterministic bounded ablation plans and validation-based attribution."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Dict, List, Optional


def build_ablation_arms(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the complete 2^N plan for two or three ordered components."""

    if len(components) not in {2, 3}:
        raise ValueError("combination ablation requires two or three components")
    expected_ids = ["A", "B", "C"][: len(components)]
    if [item.get("id") for item in components] != expected_ids:
        raise ValueError("components must use ordered IDs A, B, C")
    switches = {
        item["id"]: item["implementation_switch"] for item in components
    }
    arms: List[Dict[str, Any]] = []
    for size in range(0, len(components) + 1):
        for subset in combinations(expected_ids, size):
            enabled = list(subset)
            name = "Base" if not enabled else "+".join(enabled)
            arms.append(
                {
                    "name": name,
                    "enabled_components": enabled,
                    "switch_values": {
                        switches[component_id]: component_id in enabled
                        for component_id in expected_ids
                    },
                }
            )
    return arms


def classify_ablation(screening: Dict[str, Any]) -> Dict[str, Any]:
    """Describe component and interaction effects using held-out validation MSE."""

    arms = screening.get("arms") or []
    values = {
        arm["name"]: float(arm["best_validation_mse"])
        for arm in arms
        if arm.get("best_validation_mse") is not None
    }
    if "Base" not in values:
        return {"classification": "unavailable", "reason": "Base arm is missing"}
    component_names = sorted(name for name in values if len(name) == 1)
    full_name = "+".join(component_names)
    if not component_names or full_name not in values:
        return {"classification": "unavailable", "reason": "full factorial is incomplete"}

    base = values["Base"]
    effects = {name: base - values[name] for name in component_names}
    full_effect = base - values[full_name]
    additive_expectation = sum(effects.values())
    interaction = full_effect - additive_expectation
    scale = max(abs(base), 1e-12)
    material = 0.001 * scale
    positive_components = [name for name, effect in effects.items() if effect > material]

    if full_effect <= material and not positive_components:
        classification = "ineffective"
    elif interaction > material:
        classification = "synergistic"
    elif interaction < -material:
        classification = "antagonistic"
    elif len(positive_components) == 1:
        classification = "single_component_dominant"
    else:
        classification = "additive"
    return {
        "classification": classification,
        "metric": "held_out_observed_validation_mse",
        "direction": "lower_is_better",
        "component_effects": effects,
        "full_combination_effect": full_effect,
        "additive_expectation": additive_expectation,
        "interaction_effect": interaction,
    }


def build_pruning_audit(
    screening: Dict[str, Any], components: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Audit declared removals with matched full-factorial parameter counts."""

    removals = [item for item in components if item.get("operation") == "remove"]
    if not removals:
        return {"status": "not_requested", "removals": []}
    arms = {
        frozenset(arm.get("enabled_components", [])): arm
        for arm in screening.get("arms", [])
    }
    selected = set(
        next(
            (
                arm.get("enabled_components", [])
                for arm in screening.get("arms", [])
                if arm.get("name") == screening.get("selected_non_base_arm")
            ),
            [],
        )
    )
    records = []

    def parameter_map(arm: Dict[str, Any]) -> Dict[str, Any]:
        return {
            json.dumps(
                item.get("hyperparameters", {}),
                sort_keys=True,
                ensure_ascii=True,
            ): item.get("parameter_count")
            for item in arm.get("parameter_count_by_configuration", [])
        }

    for component in removals:
        component_id = component["id"]
        comparisons = []
        for enabled, without_arm in arms.items():
            if component_id in enabled:
                continue
            with_arm = arms.get(frozenset(set(enabled) | {component_id}))
            if with_arm is None:
                continue
            before_by_config = parameter_map(without_arm)
            after_by_config = parameter_map(with_arm)
            shared_signatures = sorted(set(before_by_config) & set(after_by_config))
            if not shared_signatures:
                shared_signatures = [None]
            for signature in shared_signatures:
                before = (
                    without_arm.get("parameter_count")
                    if signature is None
                    else before_by_config[signature]
                )
                after = (
                    with_arm.get("parameter_count")
                    if signature is None
                    else after_by_config[signature]
                )
                comparisons.append(
                    {
                        "without_removal_arm": without_arm.get("name"),
                        "with_removal_arm": with_arm.get("name"),
                        "matched_hyperparameters": (
                            None if signature is None else json.loads(signature)
                        ),
                        "parameter_count_before": before,
                        "parameter_count_after": after,
                        "parameter_delta": (
                            None
                            if before is None or after is None
                            else int(after) - int(before)
                        ),
                    }
                )
        deltas = [
            item["parameter_delta"]
            for item in comparisons
            if item["parameter_delta"] is not None
        ]
        removal_kind = component.get("removal_kind")
        if removal_kind == "parameterized_module":
            verification = (
                "confirmed_structural_prune"
                if deltas and all(delta < 0 for delta in deltas)
                else "logical_or_unverified_removal"
            )
        elif deltas and all(delta <= 0 for delta in deltas):
            verification = "parameter_free_removal_consistent"
        else:
            verification = "logical_or_unverified_removal"
        records.append(
            {
                "component_id": component_id,
                "target": component.get("target"),
                "removal_kind": removal_kind,
                "affected_modules": component.get("affected_modules", []),
                "selected_arm_applies_removal": component_id in selected,
                "verification": verification,
                "matched_parameter_comparisons": comparisons,
            }
        )
    selected_records = [
        item for item in records if item["selected_arm_applies_removal"]
    ]
    selected_parameterized = [
        item
        for item in selected_records
        if item["removal_kind"] == "parameterized_module"
    ]
    return {
        "status": "completed",
        "metric": "trainable_parameter_count",
        "limitation": (
            "Parameter counts can confirm parameterized pruning but cannot prove that a "
            "parameter-free path or loss expression was removed from source code."
        ),
        "selected_removal_count": len(selected_records),
        "selected_parameterized_removals_all_confirmed": (
            None
            if not selected_parameterized
            else all(
                item["verification"] == "confirmed_structural_prune"
                for item in selected_parameterized
            )
        ),
        "removals": records,
    }


def build_final_removal_audit(
    components: List[Dict[str, Any]],
    baseline_final: Dict[str, Any],
    candidate_final: Dict[str, Any],
    selected_component_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Report whether the formally evaluated candidate actually became smaller."""

    selected = set(selected_component_ids or [item.get("id") for item in components])
    removals = [
        item
        for item in components
        if item.get("operation") == "remove" and item.get("id") in selected
    ]
    before = baseline_final.get("parameter_count")
    after = candidate_final.get("parameter_count")
    delta = None if before is None or after is None else int(after) - int(before)
    parameterized = [
        item for item in removals if item.get("removal_kind") == "parameterized_module"
    ]
    return {
        "status": "completed" if removals else "not_requested",
        "selected_removal_components": [item.get("id") for item in removals],
        "parameter_count_before": before,
        "parameter_count_after": after,
        "parameter_delta": delta,
        "parameterized_removal_net_reduction_confirmed": (
            None if not parameterized or delta is None else delta < 0
        ),
        "warning": (
            "A non-negative net parameter delta does not prove removal failed when the same "
            "candidate also adds or enlarges another component; use matched ablation evidence."
            if parameterized and delta is not None and delta >= 0
            else None
        ),
    }
