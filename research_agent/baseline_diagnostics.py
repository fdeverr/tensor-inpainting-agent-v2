"""Small, deterministic summaries of baseline training behaviour.

The diagnostics are observations only: they describe the recorded training and
validation curves without prescribing the next mutation or claiming a cause.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize_baseline_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return compact curve facts and a few conservative diagnostic signals."""

    records = []
    for item in history:
        step = item.get("step")
        train = _finite_number(item.get("data_train_loss"))
        total = _finite_number(item.get("total_train_loss"))
        validation = _finite_number(item.get("validation_mse"))
        if isinstance(step, int) and train is not None:
            records.append(
                {
                    "step": step,
                    "data_train_loss": train,
                    "total_train_loss": total,
                    "validation_mse": validation,
                }
            )

    if not records:
        return {
            "record_count": 0,
            "first": None,
            "last": None,
            "best_validation": None,
            "signals": [
                {
                    "code": "diagnostics_unavailable",
                    "observation": "No finite training-history records were available.",
                }
            ],
        }

    first = records[0]
    last = records[-1]
    validation_records = [
        item for item in records if item["validation_mse"] is not None
    ]
    best_validation = (
        min(validation_records, key=lambda item: item["validation_mse"])
        if validation_records
        else None
    )
    train_improvement = (
        first["data_train_loss"] - last["data_train_loss"]
    ) / max(abs(first["data_train_loss"]), 1e-12)
    validation_improvement = None
    validation_regression = None
    if best_validation is not None and first["validation_mse"] is not None:
        validation_improvement = (
            first["validation_mse"] - best_validation["validation_mse"]
        ) / max(abs(first["validation_mse"]), 1e-12)
        validation_regression = (
            last["validation_mse"] - best_validation["validation_mse"]
        ) / max(abs(best_validation["validation_mse"]), 1e-12)

    signals: List[Dict[str, Any]] = []
    if best_validation is not None and best_validation["step"] == last["step"]:
        signals.append(
            {
                "code": "best_validation_at_budget_end",
                "observation": (
                    "The lowest recorded validation MSE occurred at the last recorded step."
                ),
                "evidence": {
                    "step": last["step"],
                    "validation_mse": best_validation["validation_mse"],
                },
            }
        )
    if validation_regression is not None and validation_regression >= 0.05:
        signals.append(
            {
                "code": "validation_regressed_after_best",
                "observation": (
                    "Validation MSE ended at least 5% above its best recorded value while training continued."
                ),
                "evidence": {
                    "best_step": best_validation["step"],
                    "best_validation_mse": best_validation["validation_mse"],
                    "last_validation_mse": last["validation_mse"],
                    "relative_regression": validation_regression,
                },
            }
        )
    recent = records[-min(5, len(records)) :]
    if len(recent) >= 3:
        recent_start = recent[0]["data_train_loss"]
        recent_end = recent[-1]["data_train_loss"]
        recent_gain = (recent_start - recent_end) / max(abs(recent_start), 1e-12)
        if abs(recent_gain) < 0.01:
            signals.append(
                {
                    "code": "training_loss_plateau",
                    "observation": (
                        "Data training loss changed by less than 1% across the last recorded window."
                    ),
                    "evidence": {
                        "window_records": len(recent),
                        "relative_improvement": recent_gain,
                    },
                }
            )
    if not signals:
        signals.append(
            {
                "code": "no_simple_curve_pathology",
                "observation": (
                    "The recorded aggregate training and validation curves show no deterministic plateau or post-best regression."
                ),
            }
        )

    return {
        "record_count": len(records),
        "training_loss_protocol": {
            "timing": sorted({item.get("training_loss_timing", "legacy_unspecified") for item in history}),
            "prediction_scope": sorted({item.get("training_prediction_scope", "legacy_unspecified") for item in history}),
            "total_loss_includes_regularization": True,
        },
        "first": first,
        "last": last,
        "minimum_recorded_train_loss": min(
            item["data_train_loss"] for item in records
        ),
        "best_validation": best_validation,
        "relative_train_improvement": train_improvement,
        "relative_validation_improvement": validation_improvement,
        "relative_validation_regression_after_best": validation_regression,
        "signals": signals,
    }
