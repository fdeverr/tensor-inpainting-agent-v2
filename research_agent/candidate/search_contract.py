"""Consistency checks for declared and executable candidate search spaces."""

from __future__ import annotations

from typing import Any, Dict


def _copy_search_space(search_space: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Return a JSON-friendly copy after checking the executable contract shape."""

    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("%s must be a non-empty dict" % label)
    copied: Dict[str, Any] = {}
    for name, values in search_space.items():
        if not isinstance(name, str) or not name:
            raise ValueError("%s keys must be non-empty strings" % label)
        if not isinstance(values, list) or not values:
            raise ValueError(
                "%s[%s] must be a non-empty list" % (label, name)
            )
        copied[name] = list(values)
    return copied


def declared_search_space_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Read the new declaration field with backward compatibility for old artifacts."""

    declared = manifest.get("declared_search_space")
    if declared is None:
        declared = manifest.get(
            "original_allowed_search_space",
            manifest.get("allowed_search_space", {}),
        )
    return declared


def evaluate_search_space_contract(
    base_search_space: Dict[str, Any],
    candidate_search_space: Dict[str, Any],
    declared_search_space: Dict[str, Any],
) -> Dict[str, Any]:
    """Return an actionable report without silently changing either search space."""

    base_search_space = _copy_search_space(base_search_space, "base_search_space")
    candidate_search_space = _copy_search_space(
        candidate_search_space, "candidate_search_space"
    )
    declared_search_space = _copy_search_space(
        declared_search_space, "declared_search_space"
    )

    executable_changes = {
        key: values
        for key, values in candidate_search_space.items()
        if key not in base_search_space or values != base_search_space[key]
    }
    declared_changes = {
        key: values
        for key, values in declared_search_space.items()
        if key not in base_search_space or values != base_search_space[key]
    }
    passed = executable_changes == declared_changes
    feedback = None
    if not passed:
        feedback = (
            "SEARCH_SPACE_CONTRACT_MISMATCH: CandidateProposal.search_space and "
            "CandidateTensorInpaintingModel.search_space() describe different changes "
            "relative to the base model. executable_changes=%s; declared_changes=%s. "
            "Regenerate one internally consistent candidate by revising either the "
            "proposal declaration or the executable search_space(); do not silently "
            "drop, narrow, or add hyperparameters."
            % (executable_changes, declared_changes)
        )
    return {
        "passed": passed,
        "declared_search_space": declared_search_space,
        "executable_search_space": candidate_search_space,
        "effective_search_space": candidate_search_space if passed else None,
        "declared_changes": declared_changes,
        "executable_changes": executable_changes,
        "feedback": feedback,
    }


def validate_search_space_contract(
    base_search_space: Dict[str, Any],
    candidate_search_space: Dict[str, Any],
    declared_search_space: Dict[str, Any],
) -> None:
    """Reject code whose effective changes differ from the proposal declaration."""

    report = evaluate_search_space_contract(
        base_search_space,
        candidate_search_space,
        declared_search_space,
    )
    if not report["passed"]:
        raise ValueError(report["feedback"])
