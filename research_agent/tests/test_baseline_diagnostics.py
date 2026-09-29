from research_agent.baseline_diagnostics import summarize_baseline_history


def test_empty_history_is_reported_without_inventing_a_cause():
    summary = summarize_baseline_history([])

    assert summary["record_count"] == 0
    assert summary["signals"][0]["code"] == "diagnostics_unavailable"


def test_diagnostics_detect_post_best_validation_regression():
    history = [
        {"step": 1, "data_train_loss": 1.0, "total_train_loss": 1.0, "validation_mse": 1.0},
        {"step": 10, "data_train_loss": 0.5, "total_train_loss": 0.5, "validation_mse": 0.4},
        {"step": 20, "data_train_loss": 0.3, "total_train_loss": 0.3, "validation_mse": 0.5},
    ]

    summary = summarize_baseline_history(history)

    assert summary["best_validation"]["step"] == 10
    assert "validation_regressed_after_best" in {
        signal["code"] for signal in summary["signals"]
    }


def test_diagnostics_detect_budget_end_improvement_and_plateau():
    improving = [
        {"step": 1, "data_train_loss": 1.0, "total_train_loss": 1.0, "validation_mse": 1.0},
        {"step": 10, "data_train_loss": 0.7, "total_train_loss": 0.7, "validation_mse": 0.8},
        {"step": 20, "data_train_loss": 0.5, "total_train_loss": 0.5, "validation_mse": 0.6},
    ]
    plateau = [
        {"step": 1, "data_train_loss": 0.5, "total_train_loss": 0.5, "validation_mse": 0.6},
        {"step": 10, "data_train_loss": 0.499, "total_train_loss": 0.499, "validation_mse": 0.59},
        {"step": 20, "data_train_loss": 0.498, "total_train_loss": 0.498, "validation_mse": 0.58},
    ]

    improving_codes = {
        item["code"] for item in summarize_baseline_history(improving)["signals"]
    }
    plateau_codes = {
        item["code"] for item in summarize_baseline_history(plateau)["signals"]
    }

    assert "best_validation_at_budget_end" in improving_codes
    assert "training_loss_plateau" in plateau_codes
