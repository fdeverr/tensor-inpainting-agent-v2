"""One generic Tensor Inpainting Agent Framework tool for every promoted tensor candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..candidate import candidate_builder, load_validated_candidate
from ..core.data import load_observation_mask, load_tensor_data
from ..core.fair_experiment import final_fit_and_evaluate
from ..schemas import TrainingConfig
from .framework import ToolParameter, ToolResponse
from .research_tools import ResearchTool, _required_string


class AlgorithmRunnerTool(ResearchTool):
    """Load a hash-gated approved model and run the fixed final-fit protocol."""

    def __init__(self) -> None:
        super().__init__(
            name="run_approved_tensor_algorithm",
            description=(
                "按 approved manifest 加载已晋升的张量算法，使用固定 Trainer "
                "在全部观测像素上重新拟合，并用缺失区 GT 选择最佳输出。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="approved_dir", type="string", description="已晋升算法版本目录"),
            ToolParameter(name="corrupted_path", type="string", description="缺失图像路径"),
            ToolParameter(name="mask_path", type="string", description="观测 mask 路径"),
            ToolParameter(name="ground_truth_path", type="string", description="用于缺失区选择与评估的真值路径"),
            ToolParameter(name="output_dir", type="string", description="本次运行输出目录"),
            ToolParameter(name="seed", type="integer", description="随机种子", required=False, default=42),
            ToolParameter(name="device", type="string", description="auto、cpu 或 cuda", required=False, default="auto"),
            ToolParameter(
                name="full_reference_metrics",
                type="boolean",
                description="是否计算全参考 LPIPS；PSNR/SSIM 始终计算",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="no_reference_metrics",
                type="boolean",
                description="是否计算 MANIQA/CLIP-IQA/MUSIQ",
                required=False,
                default=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            directory = Path(_required_string(parameters, "approved_dir"))
            approved_manifest_path = directory / "approved_manifest.json"
            best_config_path = directory / "best_config.json"
            if not approved_manifest_path.is_file() or not best_config_path.is_file():
                raise ValueError("approved algorithm metadata is incomplete")
            approved_manifest = json.loads(
                approved_manifest_path.read_text(encoding="utf-8")
            )
            best = json.loads(best_config_path.read_text(encoding="utf-8"))
            candidate_class, source_manifest = load_validated_candidate(str(directory))
            if approved_manifest.get("code_sha256") != source_manifest.get("code_sha256"):
                raise ValueError("approved manifest code hash is inconsistent")

            observed = load_tensor_data(_required_string(parameters, "corrupted_path"), None)
            mask = load_observation_mask(_required_string(parameters, "mask_path"))
            ground_truth = load_tensor_data(
                _required_string(parameters, "ground_truth_path"), None
            )
            training = TrainingConfig(
                learning_rate=float(best["learning_rate"]),
                max_steps=max(1, int(best["selected_steps"])),
                validation_interval=max(1, int(best["selected_steps"])),
                early_stopping_patience=1,
                device=str(parameters.get("device", "auto")),
            )
            selected_trial = {
                "hyperparameters": best["hyperparameters"],
                "learning_rate": float(best["learning_rate"]),
                "best_step": int(best["selected_steps"]),
                "best_validation_mse": 0.0,
                "runtime_seconds": 0.0,
            }
            result = final_fit_and_evaluate(
                model_name=approved_manifest["algorithm_name"],
                model_builder=candidate_builder(candidate_class),
                selected_trial=selected_trial,
                observed_image=observed,
                observed_mask=mask,
                ground_truth=ground_truth,
                training_config=training,
                seed=int(parameters.get("seed", 42)),
                output_dir=_required_string(parameters, "output_dir"),
                include_full_reference_metrics=bool(
                    parameters.get("full_reference_metrics", True)
                ),
                include_no_reference_metrics=bool(
                    parameters.get("no_reference_metrics", False)
                ),
            )
            return ToolResponse.success(
                text="已晋升算法运行完成。",
                data={
                    "algorithm_name": approved_manifest["algorithm_name"],
                    "version": approved_manifest["version"],
                    **result,
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)
