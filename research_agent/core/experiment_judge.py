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
    minimum_nmse_delta: float = 0.0,
) -> Dict[str, Any]:
    """Apply the fixed promotion gate; the LLM never decides acceptance."""

    baseline_metrics = baseline_final["metrics"]
    candidate_metrics = candidate_final["metrics"]
    if "missing_nmse" in baseline_metrics or "missing_nmse" in candidate_metrics:
        baseline_nmse = baseline_metrics.get("missing_nmse")
        candidate_nmse = candidate_metrics.get("missing_nmse")
        defined = all(isinstance(value, (int, float)) and math.isfinite(value)
                      for value in (baseline_nmse, candidate_nmse))
        reduction = baseline_nmse - candidate_nmse if defined else None
        failures = []
        if not defined:
            failures.append("missing waveform NMSE is undefined; no promotion")
        elif reduction <= minimum_nmse_delta:
            failures.append("missing waveform NMSE reduction must exceed the promotion threshold")
        total_loss = candidate_final.get("final_total_loss")
        unstable = isinstance(total_loss, (int, float)) and (not math.isfinite(total_loss) or total_loss > 1.0)
        if unstable:
            failures.append("candidate selected checkpoint has a numerically unstable training loss")
        baseline_runtime = _total_runtime(baseline_tuning, baseline_final)
        candidate_runtime = _total_runtime(candidate_tuning, candidate_final)
        return {
            "decision": "reject" if failures else "accept", "accepted": not failures,
            "primary_metric": "missing_nmse", "nmse_delta": reduction,
            "nmse_delta_interpretation": "incumbent_minus_candidate; positive_is_improvement",
            "psnr_delta": None, "ssim_delta": None, "lpips_delta": None,
            "runtime_ratio": candidate_runtime / baseline_runtime if baseline_runtime > 0 else None,
            "thresholds": {"minimum_nmse_delta": minimum_nmse_delta}, "gate_failures": failures,
            "training_behavior": {
                "baseline_best_step": baseline_tuning["best"]["best_step"],
                "candidate_best_step": candidate_tuning["best"]["best_step"],
                "candidate_selected_checkpoint_unstable": unstable,
            },
            "suspected_causes": failures,
            "next_round_constraints": ["Reduce missing original-waveform NMSE under the same frozen mask, seed and training budget; do not use PSNR/SSIM for audio."],
            "budget_audit": {
                "trial_count_equality_required": False,
                "baseline_trial_count": baseline_tuning["trial_count"],
                "candidate_trial_count": candidate_tuning["trial_count"],
                "baseline_total_runtime_seconds": baseline_runtime,
                "candidate_total_runtime_seconds": candidate_runtime,
            },
        }
    baseline_psnr = baseline_metrics.get("missing_psnr")
    candidate_psnr = candidate_metrics.get("missing_psnr")
    psnr_delta = (
        None
        if baseline_psnr is None or candidate_psnr is None
        else float(candidate_psnr - baseline_psnr)
    )
    ssim_delta = float(
        candidate_final["metrics"]["composite_ssim"]
        - baseline_final["metrics"]["composite_ssim"]
    )
    baseline_lpips = baseline_metrics.get("lpips")
    candidate_lpips = candidate_metrics.get("lpips")
    lpips_delta = (
        float(candidate_lpips - baseline_lpips)
        if isinstance(baseline_lpips, (int, float))
        and isinstance(candidate_lpips, (int, float))
        and math.isfinite(float(baseline_lpips))
        and math.isfinite(float(candidate_lpips))
        else None
    )
    baseline_runtime = _total_runtime(baseline_tuning, baseline_final)
    candidate_runtime = _total_runtime(candidate_tuning, candidate_final)
    runtime_ratio = candidate_runtime / baseline_runtime if baseline_runtime > 0.0 else None

    failures: List[str] = []
    if psnr_delta is None or psnr_delta < minimum_psnr_delta:
        failures.append("missing-region PSNR improvement is below the promotion threshold")
    if ssim_delta < -ssim_tolerance:
        failures.append("composite SSIM regression exceeds the allowed tolerance")
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
    baseline_selected_total_loss = baseline_final.get("final_total_loss")
    candidate_selected_total_loss = candidate_final.get("final_total_loss")
    candidate_selected_train_mse = candidate_final.get("final_train_mse")
    selected_checkpoint_unstable = (
        isinstance(candidate_selected_total_loss, (int, float))
        and math.isfinite(float(candidate_selected_total_loss))
        and (
            float(candidate_selected_total_loss) > 1.0
            or (
                isinstance(baseline_selected_total_loss, (int, float))
                and math.isfinite(float(baseline_selected_total_loss))
                and float(candidate_selected_total_loss)
                > max(0.1, 100.0 * float(baseline_selected_total_loss))
            )
        )
    )
    if selected_checkpoint_unstable:
        failures.append(
            "candidate selected checkpoint has a numerically unstable training loss"
        )
    accepted = not failures
    suspected_causes = []
    constraints = []
    if not accepted:
        constraints.extend(
            [
                (
                    "Keep the same GT-selection scope, seed, per-trial training protocol, "
                    "learning-rate search policy, device, optimizer, and judge thresholds. "
                    "Each model may independently exhaust its own bounded structure "
                    "space. The run-wide fair training budget is frozen after round one "
                    "so an already trained incumbent can be reused."
                ),
                (
                    "Do not expose raw ground-truth tensor contents to candidate generation; "
                    "training tools may use missing-region GT scores for checkpoint selection, "
                    "and evolution rounds may use persisted aggregate comparison results."
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
    if not accepted and selected_checkpoint_unstable:
        suspected_causes.append(
            "The candidate's GT-selected checkpoint has an unstable total training "
            "loss relative to the baseline. A residual branch, regularizer, "
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
            "Treat the unfinished optimization as evidence for a future run-wide budget change; "
            "do not change the frozen protocol inside the current evolution run."
        )
    if not accepted and not suspected_causes:
        suspected_causes.append(
            "The gain was positive but too small to distinguish it from a practically negligible change."
        )

    return {
        "decision": "accept" if accepted else "reject",
        "accepted": accepted,
        "psnr_delta": psnr_delta,
        "psnr_metric": "missing_psnr",
        "ssim_delta": ssim_delta,
        "lpips_delta": lpips_delta,
        "lpips_delta_interpretation": (
            "candidate_minus_incumbent; lower_is_better"
            if lpips_delta is not None
            else "unavailable"
        ),
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
            "baseline_selected_train_mse": baseline_final.get("final_train_mse"),
            "candidate_selected_train_mse": candidate_selected_train_mse,
            "baseline_selected_total_loss": baseline_selected_total_loss,
            "candidate_selected_total_loss": candidate_selected_total_loss,
            "candidate_selected_checkpoint_unstable": selected_checkpoint_unstable,
        },
        "suspected_causes": suspected_causes,
        "next_round_constraints": constraints,
        "budget_audit": {
            "trial_count_policy": (
                "independent model-specific tuning up to the configured per-model limit"
            ),
            "trial_count_equality_required": False,
            "baseline_trial_count": baseline_tuning["trial_count"],
            "candidate_trial_count": candidate_tuning["trial_count"],
            "baseline_total_runtime_seconds": baseline_runtime,
            "candidate_total_runtime_seconds": candidate_runtime,
        },
    }
