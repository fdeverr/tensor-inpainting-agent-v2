"""Public evaluation switches expose their intended defaults and overrides."""

import importlib

import numpy as np
import pytest

from research_agent.core.metrics import evaluate_reconstruction_metrics


@pytest.mark.parametrize("module,required,options", [
    (
        "run",
        ["--image", "input.png"],
        [
            "full-reference-metrics",
            "no-reference-metrics",
            "selection-visual-assessment",
            "mutation-visual-assessment",
            "siren-comparison",
        ],
    ),
    ("run_day1", ["--image", "input.png"], ["full-reference-metrics", "no-reference-metrics"]),
    ("run_day2", ["--image", "input.png", "--model", "matrix"], ["full-reference-metrics", "no-reference-metrics"]),
    ("run_day3", ["--image", "input.png"], ["full-reference-metrics", "no-reference-metrics"]),
    (
        "run_day4",
        ["--image", "input.png"],
        [
            "full-reference-metrics",
            "no-reference-metrics",
            "selection-visual-assessment",
            "siren-comparison",
        ],
    ),
    (
        "run_day5",
        ["--base-run-dir", "base"],
        ["mutation-visual-assessment"],
    ),
    (
        "run_day6",
        ["--base-run-dir", "base", "--candidate-dir", "candidate"],
        ["full-reference-metrics", "no-reference-metrics", "mutation-visual-assessment"],
    ),
    (
        "run_benchmark",
        ["--images", "input.png"],
        [
            "full-reference-metrics",
            "no-reference-metrics",
            "selection-visual-assessment",
            "mutation-visual-assessment",
            "siren-comparison",
        ],
    ),
])
def test_cli_evaluation_switches_use_intended_defaults_and_allow_overrides(module, required, options):
    parser = importlib.import_module("research_agent." + module).build_parser()
    defaults = parser.parse_args(required)
    for option in options:
        destination = option.replace("-", "_")
        assert getattr(defaults, destination) is (
            option == "siren-comparison"
        )
        assert getattr(parser.parse_args(required + ["--" + option]), destination) is True
        assert getattr(parser.parse_args(required + ["--skip-" + option]), destination) is False
        with pytest.raises(SystemExit):
            parser.parse_args(required + ["--" + option, "--skip-" + option])


def test_explicit_lpips_metrics_include_learned_models(monkeypatch):
    def learned_scores(*args, **kwargs):
        assert kwargs["include_full_reference_metrics"] is True
        assert kwargs["include_no_reference_metrics"] is False
        return (
            {"lpips": 0.0, "maniqa": None, "clip_iqa": None, "musiq": None},
            {
                "requested": True,
                "requested_metrics": ["lpips"],
                "skipped_reason": None,
                "errors": {},
            },
        )

    monkeypatch.setattr("research_agent.core.metrics.learned_image_quality_metrics", learned_scores)
    reference = np.full((16, 16, 3), 0.5, dtype=np.float32)
    mask = np.ones((16, 16), dtype=bool)
    mask[4:12, 4:12] = False
    result = evaluate_reconstruction_metrics(
        reference, reference, mask, include_full_reference_metrics=True
    )
    assert result["composite_ssim"] == 1.0
    assert result["learned_metric_status"]["requested"] is True
    assert result["lpips"] == 0.0
    assert all(result[key] is None for key in ("maniqa", "clip_iqa", "musiq"))
    assert result["metric_group_status"]["full_reference"]["enabled"] is True
    assert result["metric_group_status"]["no_reference"]["enabled"] is False
