"""Whole-modality evidence used to select each evolution-round incumbent."""

from __future__ import annotations

import math
import json
import statistics
from pathlib import Path

from .baseline_diagnostics import summarize_baseline_history
from .core.audio_metrics import audio_metric_context
from .core.data import apply_observation_mask, load_observation_mask, load_tensor_data
from .core.fair_experiment import final_fit_and_evaluate
from .schemas import TrainingConfig


def evaluate_across_cases(name, builder, selected_trial, cases, output_dir,
                          steps, interval, patience, device):
    """Fit one fixed architecture/configuration independently on every frozen mask."""
    results = []
    for index, case in enumerate(cases):
        item = {"source": case["source"], "status": "failed"}
        try:
            gt = load_tensor_data(case["gt_path"], max_size=None)
            mask = load_observation_mask(case["mask_path"])
            observed = apply_observation_mask(gt, mask)
            # Never reuse the development checkpoint for a different tensor or budget.
            trial = {key: selected_trial[key] for key in ("hyperparameters", "learning_rate")}
            trial["runtime_seconds"] = 0.0  # No per-sample hyperparameter search.
            with audio_metric_context(case):
                result = final_fit_and_evaluate(
                    name, builder, trial, observed, mask, gt,
                    TrainingConfig(
                        learning_rate=float(trial["learning_rate"]),
                        max_steps=steps, validation_interval=interval,
                        early_stopping_patience=patience or steps + 1,
                        device=device,
                    ), case["seed"], str(Path(output_dir) / str(index)),
                    False, False,
                )
            json.dumps(result["metrics"], allow_nan=False)
            item.update(status="completed", metrics=result["metrics"],
                        artifacts=result["artifacts"],
                        training=training_feedback(result))
        except Exception as error:
            item["error"] = "%s: %s" % (type(error).__name__, error)
        results.append(item)
    audio = cases[0]["data_type"] == "audio"
    valid = [item["metrics"] for item in results if item["status"] == "completed"]
    summary = {"expected_count": len(cases), "completed_count": len(valid),
               "complete": len(valid) == len(cases)}
    if audio:
        values = [metric.get("missing_nmse") for metric in valid]
        summary["complete"] &= all(_finite(value) for value in values)
        summary["mean_missing_nmse"] = statistics.mean(values) if summary["complete"] else None
    else:
        psnr = [metric.get("missing_psnr") for metric in valid]
        ssim = [metric.get("composite_ssim") for metric in valid]
        mse = [metric.get("missing_mse") for metric in valid]
        # A perfect reconstruction has null PSNR in the existing metric schema.
        perfect = [value == 0 for value in mse]
        summary["complete"] &= all(_finite(value) or is_perfect for value, is_perfect in zip(psnr, perfect))
        summary["complete"] &= all(_finite(value) for value in ssim)
        summary["perfect_count"] = sum(perfect)
        finite_psnr = [value for value in psnr if _finite(value)]
        summary["mean_finite_missing_psnr"] = (statistics.mean(finite_psnr) if summary["complete"] and finite_psnr else None)
        summary["mean_missing_psnr"] = (summary["mean_finite_missing_psnr"] if not any(perfect) else None)
        summary["mean_composite_ssim"] = statistics.mean(ssim) if summary["complete"] else None
        summary["mean_missing_mse"] = statistics.mean(mse) if summary["complete"] else None
    return {"algorithm": name, "results": results, "summary": summary,
            "protocol": {"steps": steps, "validation_interval": interval,
                         "patience": patience, "device": device,
                         "configuration_source": "development-sample tuning, frozen for every case"}}


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def cohort_score(summary):
    """Shared higher-is-better ranking, including explicit perfect-PSNR ties."""
    if "mean_missing_nmse" in summary:
        value = summary["mean_missing_nmse"]
        return (-value if _finite(value) else -math.inf,)
    finite_psnr = summary.get("mean_finite_missing_psnr", summary.get("mean_missing_psnr"))
    ssim = summary.get("mean_composite_ssim")
    return (summary.get("perfect_count", 0),
            finite_psnr if _finite(finite_psnr) else -math.inf,
            ssim if _finite(ssim) else -math.inf)


def training_feedback(result):
    """Bounded per-sample curve evidence for LLM context; full history stays on disk."""
    path = (result.get("artifacts") or {}).get("history")
    if not path:
        return {"status": "no_training", "reason": "interpolation or unavailable history"}
    try:
        history = json.loads(Path(path).read_text(encoding="utf-8"))["history"]
        summary = summarize_baseline_history(history)
        compact_summary = {key: summary.get(key) for key in
                           ("record_count", "first", "last", "best_validation",
                            "relative_train_improvement", "relative_validation_improvement",
                            "relative_validation_regression_after_best", "training_loss_protocol")}
        compact_summary["signals"] = [item.get("code") for item in summary.get("signals", [])]
        if history:
            positions = sorted({0, len(history) // 4, len(history) // 2,
                                3 * len(history) // 4, len(history) - 1} |
                               {index for index, item in enumerate(history) if item["step"] == result.get("selected_steps")})
            points = [{key: history[index].get(key) for key in
                       ("step", "data_train_loss", "total_train_loss", "validation_mse",
                        "missing_gt_psnr", "missing_gt_nmse")
                       if key in history[index]} for index in positions]
        else:
            points = []
        return {"status": "completed", "selected_step": result.get("selected_steps"),
                "runtime_seconds": result.get("runtime_seconds"),
                "parameter_count": result.get("parameter_count"),
                "curve_summary": compact_summary, "curve_points": points}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"status": "history_unavailable", "error": "%s: %s" % (type(error).__name__, error)}


def judge_across_cases(local_judgment, incumbent, candidate, minimum_psnr_delta,
                       ssim_tolerance, minimum_nmse_delta):
    """Replace the development-only gate with a complete-cohort gate."""
    judgment = {**local_judgment, "scope": "all_valid_samples_of_modality",
                "aggregation": "equal-weight arithmetic mean across all valid samples",
                "development_sample_judgment": local_judgment,
                "incumbent_dataset_summary": incumbent["summary"],
                "candidate_dataset_summary": candidate["summary"]}
    judgment["lpips_delta"] = None  # Optional development LPIPS is not the cohort gate.
    before, after = incumbent["summary"], candidate["summary"]
    failures = []
    if not after["complete"]:
        failures.append("candidate must finish every valid sample with defined evaluation metrics")
    rescued_incomplete_incumbent = not before["complete"] and after["complete"]
    judgment["rescued_incomplete_incumbent"] = rescued_incomplete_incumbent
    if "mean_missing_nmse" in before:
        reduction = (before["mean_missing_nmse"] - after["mean_missing_nmse"]
                     if before["complete"] and after["complete"] else None)
        judgment.update(nmse_delta=reduction, psnr_delta=None, ssim_delta=None)
        if reduction is not None and reduction <= minimum_nmse_delta:
            failures.append("mean missing waveform NMSE reduction does not exceed threshold")
        constraints = ["Improve mean missing waveform NMSE over all same-type samples under the frozen masks and evaluation budget."]
    else:
        # Standard arithmetic mean PSNR, with perfect cases sorted ahead of finite cases.
        if before["complete"] and after["complete"]:
            if after["perfect_count"] != before["perfect_count"]:
                delta = None
                if after["perfect_count"] < before["perfect_count"]:
                    failures.append("candidate has fewer perfectly recovered samples")
            else:
                before_psnr = before.get("mean_finite_missing_psnr", before.get("mean_missing_psnr"))
                after_psnr = after.get("mean_finite_missing_psnr", after.get("mean_missing_psnr"))
                delta = after_psnr - before_psnr if after_psnr is not None and before_psnr is not None else None
            ssim_delta = after["mean_composite_ssim"] - before["mean_composite_ssim"]
        else:
            delta = ssim_delta = None
        judgment.update(psnr_delta=delta, ssim_delta=ssim_delta)
        if not failures and delta is not None and delta < minimum_psnr_delta:
            failures.append("mean missing-region PSNR gain is below threshold")
        if not failures and delta is None and not rescued_incomplete_incumbent and after.get("perfect_count", 0) <= before.get("perfect_count", 0):
            failures.append("mean missing-region PSNR gain is undefined")
        if ssim_delta is not None and ssim_delta < -ssim_tolerance:
            failures.append("mean composite SSIM regression exceeds tolerance")
        constraints = ["Improve mean missing-region PSNR without regressing mean SSIM across all same-type samples."]
    if local_judgment["training_behavior"].get("candidate_selected_checkpoint_unstable"):
        failures.append("candidate development checkpoint is numerically unstable")
    judgment.update(decision="reject" if failures else "accept", accepted=not failures,
                    gate_failures=failures, suspected_causes=failures or [],
                    next_round_constraints=constraints)
    return judgment


def compact_case_feedback(evaluation):
    """Give the next round per-sample scores, never ground-truth tensor values."""
    return [{"sample_index": index, "sample": Path(item["source"]).name,
             "status": item["status"],
             "metrics": {key: item.get("metrics", {}).get(key) for key in
                         ("missing_nmse",) if key in item.get("metrics", {})} or
                        {key: item.get("metrics", {}).get(key) for key in
                         ("missing_psnr", "composite_ssim") if key in item.get("metrics", {})},
             "training": item.get("training"), "error": item.get("error")}
            for index, item in enumerate(evaluation["results"])]
