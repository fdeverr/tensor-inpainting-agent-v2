from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research_agent.benchmark import aggregate_cases
from research_agent.core.data import load_tensor_data
from research_agent.core.masks import generate_observation_mask
from research_agent.reporting import _best_algorithm_lines, _comparison_image_lines
from research_agent.workflow_full import (
    FullWorkflowConfig,
    _export_comparison_images,
    _score,
    run_full_workflow,
)


def _write_image(path: Path) -> None:
    height, width = 12, 16
    y, x = np.indices((height, width), dtype=np.float32)
    base = (x + y) / ((width - 1) + (height - 1))
    image = np.stack((base, 0.8 * base + 0.1, 0.6 * base + 0.2), axis=-1)
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def test_final_algorithm_score_uses_missing_region_psnr():
    better_missing = {"missing_psnr": 20.0, "full_psnr": 10.0}
    better_full = {"missing_psnr": 15.0, "full_psnr": 30.0}

    assert _score(better_missing) > _score(better_full)


def test_dataset_evolution_exposes_first_round_references_and_sample_curves(tmp_path):
    cases = []
    for index in range(2):
        source = tmp_path / ("sample%d.png" % index)
        _write_image(source)
        gt = load_tensor_data(str(source), max_size=None)
        gt_path = tmp_path / ("gt%d.npy" % index)
        mask_path = tmp_path / ("mask%d.npy" % index)
        np.save(gt_path, gt)
        np.save(mask_path, generate_observation_mask(gt.shape[0], gt.shape[1], 0.3, "block", index + 11))
        cases.append({"source": str(source), "gt_path": str(gt_path),
                      "mask_path": str(mask_path), "seed": index + 11,
                      "data_type": "Image"})
    reference = {"fixed_baselines": [{"algorithm": "nearest_neighbor_manhattan",
                                       "samples": [{"sample": "sample0.png", "metrics": {"missing_psnr": 10.0}}]}]}
    state = run_full_workflow(FullWorkflowConfig(
        image_path=cases[0]["source"], observation_mask_path=cases[0]["mask_path"],
        data_type="color_image", image_size=None, seed=cases[0]["seed"],
        output_dir=str(tmp_path / "outputs"), candidate_root=str(tmp_path / "candidates"),
        approved_root=str(tmp_path / "approved"), base_model="tucker",
        evolution_cases=cases, dataset_algorithm_reference=reference,
        dataset_evaluation_steps=5, dataset_evaluation_validation_interval=1,
        method_max_steps=5, method_max_steps_ceiling=5, fair_max_steps=5,
        tuning_trials=1, fair_learning_rate_candidates=(0.03,),
        fair_refine_learning_rate=False, max_improvement_rounds=1,
        screening_max_steps=5, screening_patience=2, validation_interval=1,
        patience=5, siren_comparison=False, device="cpu", llm_mode="off",
    ))
    import json
    day5 = json.loads(Path(state["artifacts"]["day5_state"]).read_text(encoding="utf-8"))
    day6 = json.loads(Path(state["artifacts"]["day6_state"]).read_text(encoding="utf-8"))
    first_reference = day5["algorithm_comparison_reference"]["whole_modality_algorithms"]
    assert first_reference["fixed_baselines"]
    assert len(first_reference["selected_incumbent_before_evolution"]["samples"]) == 2
    cohort = day6["rounds"][0]["result_summary"]["dataset_evaluation"]
    assert len(cohort["candidate"]) == len(cohort["incumbent"]) == 2
    assert all(item["training"]["curve_summary"]["record_count"] > 0 for item in cohort["candidate"])


def test_one_command_workflow_writes_report_and_best_image(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("optional evaluators must stay off by default")

    monkeypatch.setattr("research_agent.visual_evaluator.MultimodalQualityEvaluator.evaluate", forbidden)
    monkeypatch.setattr("research_agent.core.metrics.learned_image_quality_metrics", forbidden)
    image_path = tmp_path / "input.png"
    _write_image(image_path)
    state = run_full_workflow(
        FullWorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            candidate_root=str(tmp_path / "candidates"),
            approved_root=str(tmp_path / "approved"),
            mask_type="block",
            missing_rate=0.3,
            seed=11,
            image_size=None,
            method_max_steps=5,
            method_max_steps_ceiling=5,
            fair_max_steps=5,
            tuning_trials=1,
            fair_learning_rate_candidates=(0.03,),
            fair_refine_learning_rate=False,
            max_improvement_rounds=1,
            screening_max_steps=5,
            screening_patience=2,
            validation_interval=1,
            patience=5,
            siren_max_steps=5,
            siren_tuning_trials=1,
            siren_validation_interval=1,
            siren_patience=2,
            device="cpu",
            llm_mode="off",
        )
    )

    assert state["stage"] == "COMPLETED"
    assert set(state["child_runs"]) == {"day4", "day5", "day6"}
    assert "global_experience" not in state["artifacts"]
    assert Path(state["artifacts"]["best_completion"]).is_file()
    report_path = Path(state["artifacts"]["report"])
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "最终指标" in report
    assert "最佳算法与模型框架" in report
    assert "最佳算法：`%s`" % state["best_available"]["algorithm"] in report
    assert "LPIPS ↓" not in report
    assert "CLIP-IQA ↑" not in report
    assert "多模态恢复质量观察" not in report
    assert "效果图对比" in report
    assert "证据边界" in report
    assert len(state["method_results"]) == 4
    assert set(state["artifacts"]["comparison_images"]) == {
        "corrupted_input",
        "interpolation_baseline",
        "implicit_neural_baseline",
        "tensor_baseline",
        "candidate",
    }
    assert all(
        Path(path).is_file()
        for path in state["artifacts"]["comparison_images"].values()
    )


def test_best_algorithm_description_uses_champion_not_last_rejected_candidate():
    state = {"best_available": {"algorithm": "tucker_evolved", "role": "candidate"}}
    day4 = {"selected_model": "tucker"}
    day6 = {
        "best_evolved": {"algorithm": "tucker_evolved", "model_description": {
            "proposal": {"idea": "冠军采用交叉注意力", "architecture_family": "hybrid",
                         "single_change": "注意力融合空间与通道特征"},
            "source_path": "/champion/model.py",
            "source_code": "class Champion:\n    def forward(self):\n        return self.attention(self.features)\n",
            "selected_config": {"hyperparameters": {"use_attention": True}, "best_step": 42},
        }},
        "rounds": [{"idea": "被拒绝的空洞卷积", "judgment": {"accepted": False}}],
    }
    report = "\n".join(_best_algorithm_lines(state, day4, day6))
    assert "冠军采用交叉注意力" in report
    assert "class Champion" in report
    assert "前向重建框架" in report
    assert "return self.attention(self.features)" in report
    assert '"use_attention": true' in report
    assert "被拒绝的空洞卷积" not in report


@pytest.mark.parametrize("algorithm", ["nearest_neighbor_manhattan", "tucker", "siren"])
def test_baseline_winner_has_its_own_framework_not_candidate(algorithm):
    report = "\n".join(_best_algorithm_lines(
        {"best_available": {"algorithm": algorithm, "role": "tensor_baseline"}},
        {"selected_model": "tucker"},
        {"best_evolved": {"algorithm": "tucker_evolved"}},
    ))
    assert "最佳算法：`%s`" % algorithm in report
    assert "未记录" not in report
    assert "tucker_evolved" not in report


def test_comparison_images_use_predictable_names_and_relative_report_links(tmp_path):
    sources = {}
    for name in ("corrupted", "interpolation", "tensor", "candidate"):
        path = tmp_path / (name + ".png")
        path.write_bytes((name + "-image").encode("utf-8"))
        sources[name] = str(path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exported = _export_comparison_images(
        run_dir,
        {"artifacts": {"corrupted": sources["corrupted"]}},
        [
            {
                "role": "interpolation_baseline",
                "reconstruction": sources["interpolation"],
            },
            {"role": "tensor_baseline", "reconstruction": sources["tensor"]},
            {"role": "candidate", "reconstruction": sources["candidate"]},
        ],
    )

    assert Path(exported["interpolation_baseline"]).name == (
        "01_manhattan_interpolation.png"
    )
    assert Path(exported["tensor_baseline"]).name == "03_tensor_baseline.png"
    assert all(Path(path).is_file() for path in exported.values())
    report = "\n".join(
        _comparison_image_lines(
            {
                "candidate_accepted": False,
                "artifacts": {
                    "run_dir": str(run_dir),
                    "comparison_images": exported,
                },
            }
        )
    )
    assert "Manhattan 插值" in report
    assert "张量基线" in report
    assert "候选（未接受）" in report
    assert "comparison_images/03_tensor_baseline.png" in report
    assert '<table width="100%" style="table-layout: fixed; width: 100%;">' in report
    assert report.count('width="25.000000%"') == 8
    assert report.count('width="100%" />') == 4


def test_full_workflow_config_accepts_new_manual_base_models(tmp_path):
    image_path = tmp_path / "input.png"
    _write_image(image_path)

    FullWorkflowConfig(image_path=str(image_path), base_model="tt").validate()
    FullWorkflowConfig(image_path=str(image_path), base_model="mode3").validate()
    FullWorkflowConfig(
        image_path=str(image_path), base_model="tensor_ring"
    ).validate()
    for model_name in (
        "nonnegative_cp",
        "btd",
        "tsvd",
        "nonnegative_tucker",
        "hierarchical_tucker",
    ):
        FullWorkflowConfig(
            image_path=str(image_path), base_model=model_name
        ).validate()

    with pytest.raises(ValueError, match="base_model"):
        FullWorkflowConfig(
            image_path=str(image_path), base_model="unsupported"
        ).validate()


def test_full_workflow_uses_balanced_training_budget_defaults(tmp_path):
    image_path = tmp_path / "input.png"
    _write_image(image_path)

    config = FullWorkflowConfig(image_path=str(image_path))

    assert config.method_max_steps == 1500
    assert config.method_max_steps_ceiling == 6000
    assert config.tuning_near_limit_ratio == 0.9
    assert config.tuning_expansion_factor == 2.0
    assert config.screening_max_steps == 400
    assert config.screening_patience == 10
    assert config.siren_max_steps == 4000
    assert config.siren_tuning_trials == 4
    assert config.siren_validation_interval == 25
    assert config.siren_patience == 20
    assert config.fair_max_steps == 3000
    assert config.fair_learning_rate_candidates == (0.001, 0.01, 0.1)
    assert config.fair_refine_learning_rate is True
    assert config.fair_learning_rate_refinement_factor == 3.0
    assert config.patience == 20
    assert config.max_improvement_rounds == 5


def test_benchmark_aggregation_reports_mean_std_and_failures():
    method = {
        "role": "tensor_baseline",
        "metrics": {
            "full_psnr": 13.0,
            "missing_psnr": 10.0,
            "composite_ssim": 0.5,
        },
        "runtime_seconds": 2.0,
        "parameter_count": 100,
    }
    cases = [
        {
            "status": "completed",
            "candidate_accepted": True,
            "method_results": [method],
        },
        {
            "status": "completed",
            "candidate_accepted": False,
            "method_results": [
                {
                    **method,
                    "metrics": {
                        "full_psnr": 15.0,
                        "missing_psnr": 12.0,
                        "composite_ssim": 0.7,
                    },
                }
            ],
        },
        {"status": "failed"},
    ]
    result = aggregate_cases(cases)
    tensor = result["methods_by_role"]["tensor_baseline"]
    assert result["failed_cases"] == 1
    assert result["candidate_acceptance_count"] == 1
    assert tensor["mean_missing_psnr"] == 11.0
    assert tensor["std_missing_psnr"] == 1.0
    assert tensor["mean_full_psnr"] == 14.0
    assert tensor["std_full_psnr"] == 1.0
    assert tensor["mean_lpips"] is None
