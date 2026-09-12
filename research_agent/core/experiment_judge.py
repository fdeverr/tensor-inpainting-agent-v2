"""Deterministic acceptance rules and structured research feedback."""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _total_runtime(tuning: Dict[str, Any], final: Dict[str, Any]) -> float:
    return float(sum(item["runtime_seconds"] for item in tuning["trials"])) + float(
        final["runtime_seconds"]
    )


def judge_candidate(
    baseline_tuning: Dict[str, Any],
    baseline_final: Dict[str, Any],
    candidate_tuning: Dict[str, Any],
    candidate_final: Dict[str, Any],
    minimum_psnr_delta: float = 0.2,
    ssim_tolerance: float = 0.002,
) -> Dict[str, Any]:
    """Apply the fixed promotion gate; the LLM never decides acceptance."""

    baseline_psnr = baseline_final["metrics"]["missing_psnr"]
    candidate_psnr = candidate_final["metrics"]["missing_psnr"]
    psnr_delta = (
        None
        if baseline_psnr is None or candidate_psnr is None
        else float(candidate_psnr - baseline_psnr)
    )
    ssim_delta = float(
        candidate_final["metrics"]["composite_ssim"]
        - baseline_final["metrics"]["composite_ssim"]
    )
    baseline_runtime = _total_runtime(baseline_tuning, baseline_final)
    candidate_runtime = _total_runtime(candidate_tuning, candidate_final)
    runtime_ratio = candidate_runtime / baseline_runtime if baseline_runtime > 0.0 else None

    failures: List[str] = []
    if psnr_delta is None or psnr_delta < minimum_psnr_delta:
        failures.append("missing-region PSNR improvement is below the promotion threshold")
    if ssim_delta < -ssim_tolerance:
        failures.append("composite SSIM regression exceeds the allowed tolerance")
    if baseline_tuning["trial_count"] != candidate_tuning["trial_count"]:
        failures.append("baseline and candidate received unequal tuning trial counts")

    selected_candidate = candidate_tuning["best"]["hyperparameters"]
    selected_weight = selected_candidate.get("tv_weight")
    baseline_best = baseline_tuning["best"]
    candidate_best = candidate_tuning["best"]
    baseline_last_step = (
        baseline_best["history"][-1]["step"] if baseline_best.get("history") else None
    )
    candidate_last_step = (
        candidate_best["history"][-1]["step"] if candidate_best.get("history") else None
    )
    baseline_final_total_loss = baseline_final.get("final_total_loss")
    candidate_final_total_loss = candidate_final.get("final_total_loss")
    candidate_final_train_mse = candidate_final.get("final_train_mse")
    final_fit_diverged = (
        isinstance(candidate_final_total_loss, (int, float))
        and math.isfinite(float(candidate_final_total_loss))
        and (
            float(candidate_final_total_loss) > 1.0
            or (
                isinstance(baseline_final_total_loss, (int, float))
                and math.isfinite(float(baseline_final_total_loss))
                and float(candidate_final_total_loss)
                > max(0.1, 100.0 * float(baseline_final_total_loss))
            )
        )
    )
    if final_fit_diverged:
        failures.append(
            "candidate final all-observation refit became numerically unstable"
        )
    accepted = not failures
    suspected_causes = []
    constraints = []
    if not accepted:
        constraints.extend(
            [
                (
                    "Keep the same trial count, seed, split, device, optimizer, and "
                    "judge thresholds; any revised LLM-requested training budget must "
                    "still be shared by baseline and candidate and stay under the user ceiling."
                ),
                (
                    "Do not expose ground-truth tensor contents to generation, tuning, or "
                    "checkpoint selection; later evolution rounds may use only persisted "
                    "aggregate evaluation results."
                ),
            ]
        )
    if not accepted and selected_weight == 0.0:
        suspected_causes.append(
            "Validation selected tv_weight=0, so the proposed regularizer added no useful signal."
        )
        constraints.append("Try a smaller non-zero regularization range or a different bounded prior.")
    elif not accepted and psnr_delta is not None and psnr_delta < 0.0:
        suspected_causes.append(
            "The selected spatial prior may oversmooth edges or optimize visible pixels without extrapolating into the hole."
        )
        constraints.append(
            "Reduce excessive smoothing or revise the architecture while preserving the fixed evaluation contract."
        )
    if not accepted and ssim_delta < 0.0:
        suspected_causes.append(
            "The candidate reduced structural similarity, consistent with excessive smoothing or rank mismatch."
        )
    if not accepted and final_fit_diverged:
        suspected_causes.append(
            "The candidate's final all-observation refit became unstable: its total "
            "training loss was far above the baseline even though validation-time "
            "data loss had appeared reasonable. A residual branch, regularizer, "
            "normalization, initialization, or learning-rate interaction likely diverged."
        )
        constraints.append(
            "Stabilize the next architecture: use bounded residual gates and outputs, "
            "conservative initialization, normalized coordinates/features, and "
            "regularization terms whose scale cannot dominate data loss."
        )
    if (
        not accepted
        and candidate_last_step is not None
        and candidate_best["best_step"] == candidate_last_step
        and not candidate_best.get("stopped_early", False)
    ):
        suspected_causes.append(
            "The candidate's best checkpoint was the final budgeted step, so optimization may be unfinished."
        )
        constraints.append(
            "Request a longer shared training budget in the next proposal if the user ceiling permits it."
        )
    if not accepted and not suspected_causes:
        suspected_causes.append(
            "The gain was positive but too small to distinguish it from a practically negligible change."
        )

    return {
        "decision": "accept" if accepted else "reject",
        "accepted": accepted,
        "psnr_delta": psnr_delta,
        "ssim_delta": ssim_delta,
        "runtime_ratio": runtime_ratio,
        "thresholds": {
            "minimum_psnr_delta": minimum_psnr_delta,
            "maximum_ssim_drop": ssim_tolerance,
        },
        "gate_failures": failures,
        "training_behavior": {
            "baseline_best_validation_mse": baseline_tuning["best"]["best_validation_mse"],
            "candidate_best_validation_mse": candidate_tuning["best"]["best_validation_mse"],
            "baseline_best_step": baseline_tuning["best"]["best_step"],
            "candidate_best_step": candidate_tuning["best"]["best_step"],
            "candidate_selected_hyperparameters": selected_candidate,
            "baseline_stopped_early": baseline_best.get("stopped_early"),
            "candidate_stopped_early": candidate_best.get("stopped_early"),
            "baseline_last_validation_step": baseline_last_step,
            "candidate_last_validation_step": candidate_last_step,
            "candidate_best_was_final_step": (
                candidate_last_step is not None
                and candidate_best["best_step"] == candidate_last_step
            ),
            "baseline_final_train_mse": baseline_final.get("final_train_mse"),
            "candidate_final_train_mse": candidate_final_train_mse,
            "baseline_final_total_loss": baseline_final_total_loss,
            "candidate_final_total_loss": candidate_final_total_loss,
            "candidate_final_fit_diverged": final_fit_diverged,
        },
        "suspected_causes": suspected_causes,
        "next_round_constraints": constraints,
        "budget_audit": {
            "baseline_trial_count": baseline_tuning["trial_count"],
            "candidate_trial_count": candidate_tuning["trial_count"],
            "baseline_total_runtime_seconds": baseline_runtime,
            "candidate_total_runtime_seconds": candidate_runtime,
        },
    }
