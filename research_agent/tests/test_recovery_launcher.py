"""Verify the Bash entry point without launching training or external LLM calls."""

import json
import os
import re
from pathlib import Path
import subprocess

import pytest

from research_agent.run_recovery import build_parser

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "research_agent/scripts/run_recovery.sh"


@pytest.fixture
def fake_python(tmp_path):
    executable = tmp_path / "fake python"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
        "sys.exit(int(os.environ.get('RECOVERY_TEST_EXIT_CODE', '0')))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


@pytest.mark.parametrize("selected", [("Image",), ("MSI",), ("Video",), ("audio",), ("Image", "audio")])
def test_launcher_forwards_selected_types_and_paths(tmp_path, fake_python, selected):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--data-types", *selected,
         "--dataset-root", str(tmp_path / "dataset with spaces"),
         "--output-dir", str(tmp_path / "results with spaces"), "--llm-mode", "off"],
        cwd=tmp_path, env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True, text=True, check=True,
    )
    invocation = json.loads(result.stdout)
    assert invocation["cwd"] == str(ROOT)
    assert invocation["args"][:2] == ["-m", "research_agent.run_recovery"]
    args = build_parser().parse_args(invocation["args"][2:])
    assert args.data_types == list(selected)
    assert args.dataset_root == str(tmp_path / "dataset with spaces")
    assert args.output_dir == str(tmp_path / "results with spaces")
    assert args.llm_mode == "off"
    assert args.evolution_steps == 1500 and args.evaluation_steps == 1000
    assert args.patience == 20 and args.evaluation_patience == 0
    assert args.siren_learning_rate_candidates == (5e-5, 1e-4, 3e-4)


def test_launcher_preserves_exit_status(tmp_path, fake_python):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--data-types", "MSI", "--llm-mode", "off"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": str(fake_python), "RECOVERY_TEST_EXIT_CODE": "7"},
        capture_output=True, text=True,
    )
    assert result.returncode == 7


def test_launcher_help_does_not_require_python(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": "/nonexistent/python"},
        capture_output=True, text=True, check=True,
    )
    assert "--data-types" in result.stdout


def test_launcher_training_and_boolean_overrides(tmp_path, fake_python):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--data-types", "MSI", "--evolution-steps=60",
         "--evaluation-steps", "90", "--validation-interval", "3", "--patience", "7",
         "--evaluation-validation-interval", "5", "--evaluation-patience", "11",
         "--screening-max-steps", "20", "--siren-max-steps", "70", "--lpips",
         "--skip-siren-comparison", "--no-fair-learning-rate-refinement",
         "--fair-learning-rates", "0.01,0.03", "--siren-learning-rates", "0.0001,0.0002",
         "--llm-context-tokens", "65536", "--llm-output-reserve-tokens", "8192"],
        cwd=tmp_path, env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True, text=True, check=True,
    )
    args = build_parser().parse_args(json.loads(result.stdout)["args"][2:])
    assert (args.evolution_steps, args.evaluation_steps, args.validation_interval, args.patience) == (60, 90, 3, 7)
    assert (args.evaluation_validation_interval, args.evaluation_patience) == (5, 11)
    assert args.screening_max_steps == 20 and args.siren_max_steps == 70
    assert args.lpips and not args.siren_comparison and not args.fair_refine_learning_rate
    assert args.fair_learning_rate_candidates == (0.01, 0.03)
    assert args.siren_learning_rate_candidates == (1e-4, 2e-4)
    assert args.llm_context_tokens == 65536 and args.llm_output_reserve_tokens == 8192


def test_directly_edited_script_defaults_are_effective(tmp_path, fake_python):
    script_copy = tmp_path / "project/research_agent/scripts/run_recovery.sh"
    script_copy.parent.mkdir(parents=True)
    content = SCRIPT.read_text()
    for old, new in (
        ('DATA_TYPES=("Image" "MSI" "Video" "audio")', 'DATA_TYPES=("MSI")'),
        ('EVOLUTION_STEPS="1500"', 'EVOLUTION_STEPS="80"'),
        ('PATIENCE="20"', 'PATIENCE="9"'),
        ('EVALUATION_PATIENCE="0"', 'EVALUATION_PATIENCE="12"'),
        ('FAIR_MAX_STEPS=""', 'FAIR_MAX_STEPS="100"'),
        ('LPIPS="off"', 'LPIPS="on"'),
        ('SIREN_COMPARISON="on"', 'SIREN_COMPARISON="off"'),
        ('LLM_CONTEXT_TOKENS="131072"', 'LLM_CONTEXT_TOKENS="65536"'),
        ('LLM_OUTPUT_RESERVE_TOKENS="16384"', 'LLM_OUTPUT_RESERVE_TOKENS="8192"'),
    ):
        content = content.replace(old, new, 1)
    script_copy.write_text(content)
    result = subprocess.run(
        ["bash", str(script_copy)], cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True, text=True, check=True,
    )
    args = build_parser().parse_args(json.loads(result.stdout)["args"][2:])
    assert args.llm_context_tokens == 65536 and args.llm_output_reserve_tokens == 8192
    assert args.data_types == ["MSI"]
    assert args.evolution_steps == 80 and args.patience == 9
    assert args.evaluation_patience == 12 and args.fair_max_steps == 100
    assert args.lpips and not args.siren_comparison


def test_cli_builds_config_with_all_training_options(tmp_path, monkeypatch):
    import research_agent.run_recovery as cli
    captured = []

    def capture(config, **kwargs):
        config.validate()
        captured.append(config)
        return {"stage": "INVENTORY", "run_dir": str(tmp_path),
                "inventory": {"groups": {}, "failures": []}}

    monkeypatch.setattr(cli, "run_recovery", capture)
    monkeypatch.setattr("sys.argv", [
        "run_recovery", "--dataset-root", str(tmp_path), "--data-types", "MSI",
        "--evolution-steps", "50", "--evaluation-steps", "80",
        "--patience", "7", "--validation-interval", "3",
        "--evaluation-patience", "9", "--evaluation-validation-interval", "4",
        "--method-max-steps-ceiling", "100", "--fair-max-steps", "60",
        "--screening-max-steps", "15", "--screening-patience", "2",
        "--siren-max-steps", "70", "--siren-patience", "6",
        "--siren-learning-rates", "0.00005,0.0002",
        "--fair-learning-rates", "0.02,0.04", "--no-fair-learning-rate-refinement",
        "--smoke-timeout", "5", "--knowledge-root", str(tmp_path / "knowledge"),
        "--skip-siren-comparison", "--lpips", "--no-lpips", "--inventory-only",
        "--no-reference-metrics", "on", "--selection-visual-assessment", "on",
        "--mutation-visual-assessment", "on", "--prompt", "使用多尺度注意力",
    ])
    cli.main()
    config = captured[0]
    assert config.data_types == ("MSI",)
    assert config.evolution_steps == 50 and config.evaluation_steps == 80
    assert config.patience == 7 and config.validation_interval == 3
    assert config.evaluation_patience == 9 and config.evaluation_validation_interval == 4
    assert config.method_max_steps_ceiling == 100 and config.fair_max_steps == 60
    assert config.screening_max_steps == 15 and config.screening_patience == 2
    assert config.siren_max_steps == 70 and config.siren_patience == 6
    assert config.siren_learning_rate_candidates == (5e-5, 2e-4)
    assert config.fair_learning_rate_candidates == (0.02, 0.04)
    assert not config.fair_refine_learning_rate and not config.siren_comparison and not config.lpips
    assert config.smoke_timeout_seconds == 5
    assert config.knowledge_root == str(tmp_path / "knowledge")
    assert config.no_reference_metrics and config.selection_visual_assessment and config.mutation_visual_assessment
    assert config.prompt == "使用多尺度注意力"


def test_all_run_ai_experiment_options_have_recovery_equivalents():
    ai_script = (ROOT / "research_agent/scripts/run_ai.sh").read_text()
    ai_options = set(re.findall(r"^\s*(--[a-z-]+)\)", ai_script, re.MULTILINE))
    supported = {flag for action in build_parser()._actions for flag in action.option_strings}
    # These are single-input controls, not experiment hyperparameters. Recovery
    # uses a modality representative and always reads the explicit Ohsi GT.
    assert ai_options - supported == {"--image", "--mat-key"}


def test_run_ai_on_off_syntax_and_aliases(tmp_path, fake_python):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--method-max-steps=60", "--max-improvement-rounds", "2",
         "--image-size", "original", "--full-reference-metrics", "on",
         "--no-reference-metrics=on", "--selection-visual-assessment", "on",
         "--mutation-visual-assessment", "off", "--siren-comparison", "off",
         "--fair-learning-rate-refinement", "off", "--prompt", "考虑交叉注意力"],
        cwd=tmp_path, env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True, text=True, check=True,
    )
    args = build_parser().parse_args(json.loads(result.stdout)["args"][2:])
    assert args.evolution_steps == 60 and args.improvement_rounds == 2 and args.image_size == 0
    assert args.lpips and args.no_reference_metrics and args.selection_visual_assessment
    assert not args.mutation_visual_assessment and not args.siren_comparison and not args.fair_refine_learning_rate
    assert args.prompt == "考虑交叉注意力"


@pytest.mark.parametrize("option", ["--full-reference-metrics", "--no-reference-metrics",
                                   "--selection-visual-assessment", "--mutation-visual-assessment",
                                   "--siren-comparison", "--fair-learning-rate-refinement"])
def test_invalid_on_off_values_are_rejected(option):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dataset-root", ".", option, "invalid"])


def test_run_ai_named_script_variables_are_effective(tmp_path, fake_python):
    script_copy = tmp_path / "project/research_agent/scripts/run_recovery.sh"
    script_copy.parent.mkdir(parents=True)
    content = SCRIPT.read_text()
    for name, value in (("METHOD_MAX_STEPS", "80"), ("MAX_IMPROVEMENT_ROUNDS", "2"),
                        ("FULL_REFERENCE_METRICS", "on"), ("NO_REFERENCE_METRICS", "on"),
                        ("SELECTION_VISUAL_ASSESSMENT", "on"), ("MUTATION_VISUAL_ASSESSMENT", "on"),
                        ("PROMPT", "多尺度空洞卷积")):
        content = re.sub(rf'^{name}="[^"]*"', f'{name}="{value}"', content, count=1, flags=re.MULTILINE)
    script_copy.write_text(content)
    result = subprocess.run(
        ["bash", str(script_copy)], cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True, text=True, check=True,
    )
    args = build_parser().parse_args(json.loads(result.stdout)["args"][2:])
    assert args.evolution_steps == 80 and args.improvement_rounds == 2
    assert args.lpips and args.no_reference_metrics and args.selection_visual_assessment and args.mutation_visual_assessment
    assert args.prompt == "多尺度空洞卷积"
