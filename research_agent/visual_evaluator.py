"""Multimodal observations for method selection and optional evolution feedback."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .agent_tools.framework import TensorInpaintingLLM
from .method_selector import _json_from_text


ComparisonMode = Literal["evolution_candidate", "initial_interpolation"]

TensorMethod = Literal[
    "matrix",
    "mode3",
    "cp",
    "nonnegative_cp",
    "tucker",
    "btd",
    "tsvd",
    "nonnegative_tucker",
    "hierarchical_tucker",
    "tt",
    "tensor_ring",
]


class RankRegime(BaseModel):
    """Coarse rank prior; exact candidates remain a numerical-tuning decision."""

    model_config = ConfigDict(extra="forbid")

    height: Literal["low", "medium", "high"]
    width: Literal["low", "medium", "high"]
    feature: Literal["low", "medium", "high"]
    overall: Literal["low", "medium", "high"]


class MethodSelectionVisualAssessment(BaseModel):
    """Compact structural prior extracted from the interpolation preview."""

    model_config = ConfigDict(extra="forbid")

    visible_structure_summary: str = Field(min_length=8, max_length=400)
    spatial_complexity: Literal["low", "medium", "high"]
    spatial_anisotropy: Literal[
        "isotropic", "height_dominant", "width_dominant", "unclear"
    ]
    texture_complexity: Literal["low", "medium", "high"]
    repetition_or_periodicity: Literal["low", "medium", "high", "unclear"]
    channel_coupling: Literal["weak", "moderate", "strong", "unclear"]
    interpolation_artifacts: List[str] = Field(max_length=4)
    preferred_methods: List[TensorMethod] = Field(min_length=1, max_length=4)
    rank_regime: RankRegime
    rationale: List[str] = Field(min_length=1, max_length=4)
    confidence: Literal["low", "medium", "high"]
    limitations: List[str] = Field(min_length=1, max_length=3)


class ReconstructionObservation(BaseModel):
    """Detailed visual diagnosis for one reconstruction."""

    model_config = ConfigDict(extra="forbid")

    overall_sharpness: str = Field(min_length=4, max_length=240)
    blur_and_over_smoothing: str = Field(min_length=4, max_length=240)
    object_detail_clarity: str = Field(min_length=4, max_length=240)
    edge_and_structure_continuity: str = Field(min_length=4, max_length=240)
    texture_recovery: str = Field(min_length=4, max_length=240)
    color_and_tone_consistency: str = Field(min_length=4, max_length=240)
    artifacts: List[str] = Field(max_length=4)
    poorly_recovered_regions: List[str] = Field(max_length=4)


class VisualQualityAssessment(BaseModel):
    """Structured comparison whose mutation guidance is fed to the next round."""

    model_config = ConfigDict(extra="forbid")

    visible_image_content: str = Field(min_length=8, max_length=400)
    salient_objects_and_details: List[str] = Field(min_length=1, max_length=6)
    incumbent: ReconstructionObservation
    candidate: ReconstructionObservation
    comparison: str = Field(min_length=8, max_length=600)
    candidate_improvements: List[str] = Field(max_length=4)
    candidate_regressions: List[str] = Field(max_length=4)
    mutation_guidance: List[str] = Field(min_length=1, max_length=3)
    confidence: Literal["low", "medium", "high"]
    limitations: List[str] = Field(min_length=1, max_length=3)


def _response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("multimodal model returned empty content")
    return content


def _image_data_url(
    path: str,
    max_edge: int = 1024,
) -> str:
    source = Path(path)
    if not source.is_file():
        raise ValueError("visual assessment image does not exist: %s" % source)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64,%s" % encoded


def visual_llm_from_environment(base_llm: Optional[Any]) -> Optional[Any]:
    """Optionally use a dedicated vision model with the existing API endpoint."""

    model = os.getenv("VISION_MODEL_ID")
    if not model:
        return base_llm
    if base_llm is not None and getattr(base_llm, "model", None) == model:
        return base_llm
    api_key = os.getenv("VISION_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("VISION_BASE_URL") or os.getenv("LLM_BASE_URL")
    if not api_key or not base_url:
        return None
    return TensorInpaintingLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        api_style=os.getenv("VISION_API_STYLE") or os.getenv("LLM_API_STYLE"),
    )


class MultimodalQualityEvaluator:
    """Ask a vision-capable LLM for evidence-focused restoration observations."""

    def __init__(self, llm: Optional[Any]) -> None:
        self.llm = llm

    @staticmethod
    def _method_selection_messages(
        interpolation_path: str,
        tensor_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "下面只有一张 Manhattan 插值恢复图。它是算法输出，不是 "
                    "Ground Truth，也不能证明隐藏区域内容正确。请只把它当作图像整体"
                    "结构、方向性、纹理复杂度、重复性和通道耦合的粗略代理。"
                ),
            },
            {"type": "text", "text": "Manhattan 插值恢复图："},
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(interpolation_path),
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": (
                    "非内容张量统计：%s\n"
                    "请为后续张量分解方法选择提供紧凑的结构先验。preferred_methods "
                    "只能从 schema 枚举中选择；rank_regime 只能给出 low/medium/high，"
                    "不要输出精确秩。插值通常会过度平滑，因此不要把插值造成的低频"
                    "外观误判为真实低秩。对于 MSI 或视频，RGB 预览无法可靠展示全部"
                    "波段或时间关系，应将 channel_coupling 标为 unclear 或降低置信度。"
                    "只返回完整合法 JSON，不使用 Markdown，也不添加额外说明。\n"
                    "输出 schema：%s"
                    % (
                        json.dumps(tensor_context, ensure_ascii=False),
                        json.dumps(
                            MethodSelectionVisualAssessment.model_json_schema(),
                            ensure_ascii=False,
                        ),
                    )
                ),
            },
        ]
        return [
            {
                "role": "system",
                "content": (
                    "你是张量图像结构分析员。视觉观察只用于选择分解家族和粗粒度"
                    "秩范围，不能代替可见像素验证集调参。不要虚构物体，不要把插值"
                    "结果当成真值。只输出中文 JSON。"
                ),
            },
            {"role": "user", "content": content},
        ]

    def assess_method_selection(
        self,
        interpolation_path: str,
        tensor_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract a one-shot structural prior before decomposition selection."""

        if self.llm is None:
            return {
                "status": "skipped",
                "reason": "multimodal LLM is not configured",
                "scope": "manhattan_interpolation_preview_only",
                "ground_truth_provided": False,
                "used_for": "method_selection_and_coarse_rank_prior",
            }
        messages = self._method_selection_messages(
            interpolation_path, tensor_context
        )
        raw_outputs: List[str] = []
        errors: List[str] = []
        for attempt in range(2):
            current_messages = list(messages)
            if attempt == 1:
                if raw_outputs:
                    current_messages.append(
                        {"role": "assistant", "content": raw_outputs[-1]}
                    )
                current_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "上一次结构分析不是完整合法的 JSON，错误为：%s。"
                            "请按 schema 补全所有字段，只返回 JSON。" % errors[-1]
                        ),
                    }
                )
            try:
                raw = _response_content(
                    self.llm.invoke(
                        current_messages,
                        temperature=0.0,
                        max_tokens=3072,
                    )
                )
                raw_outputs.append(raw)
                assessment = MethodSelectionVisualAssessment.model_validate(
                    _json_from_text(raw)
                ).model_dump()
                return {
                    "status": "completed",
                    "model": getattr(self.llm, "model", "unknown"),
                    "scope": "manhattan_interpolation_preview_only",
                    "ground_truth_provided": False,
                    "used_for": "method_selection_and_coarse_rank_prior",
                    "attempts": attempt + 1,
                    "repair_attempted": attempt > 0,
                    "assessment": assessment,
                }
            except Exception as error:
                errors.append(
                    "%s: %s" % (type(error).__name__, str(error)[:800])
                )
        return {
            "status": "failed",
            "reason": errors[-1],
            "attempts": 2,
            "repair_attempted": True,
            "validation_errors": errors,
            "model": getattr(self.llm, "model", "unknown"),
            "scope": "manhattan_interpolation_preview_only",
            "ground_truth_provided": False,
            "used_for": "method_selection_and_coarse_rank_prior",
        }

    @staticmethod
    def _messages(
        image_paths: Dict[str, str],
        tensor_context: Dict[str, Any],
        comparison_mode: ComparisonMode = "evolution_candidate",
    ) -> List[Dict[str, Any]]:
        initial_interpolation = comparison_mode == "initial_interpolation"
        labels = (
            {
                "incumbent": "图1：当前张量分解基线的恢复结果",
                "candidate": "图2：Manhattan 插值参考（算法输出，不是真值）",
            }
            if initial_interpolation
            else {
                "incumbent": "图1：当前最优算法的恢复结果",
                "candidate": "图2：本轮候选算法的恢复结果",
            }
        )
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    (
                        "请严格按顺序比较张量分解基线与 Manhattan 插值参考。"
                        "插值结果不是 Ground Truth，不能当作缺失区域的正确答案；"
                        "它只用于观察可能有价值的颜色、低频结构和空间连续性信息。"
                    )
                    if initial_interpolation
                    else (
                        "请严格按顺序比较当前最优与本轮候选两张恢复图。"
                    )
                )
                + (
                    "你只获得这两张图，没有任何其他图像输入或真值指标。"
                    "请进行可定位的无参考视觉观察。"
                ),
            }
        ]
        for role in ("incumbent", "candidate"):
            content.append({"type": "text", "text": labels[role]})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(image_paths[role]),
                        "detail": "high",
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "张量/预览上下文：%s\n"
                    "请输出与 schema 完全一致的 JSON。重点描述：内容是什么；"
                    "不要把任一恢复结果中的内容当成缺失区域的真实答案；"
                    "判断两张图是否模糊、过度平滑、纹理丢失、"
                    "边缘断裂、颜色漂移或出现伪影；具体哪些位置恢复不好；"
                    "候选相比当前最优好在哪里、差在哪里。mutation_guidance "
                    "必须是能服务下一轮单点算法或 loss 变异的具体启示，"
                    "但不要一次绑定多个改动。请用词精炼，避免在不同字段重复描述；"
                    "每个普通说明字段尽量控制在 180 个汉字以内，comparison 控制在 "
                    "300 个汉字以内，各列表最多给出 3 项。必须返回完整、合法的 JSON，"
                    "不要使用 Markdown 代码块或在 JSON 前后添加文字。\n输出 schema：%s"
                    % (
                        json.dumps(tensor_context, ensure_ascii=False),
                        json.dumps(
                            VisualQualityAssessment.model_json_schema(),
                            ensure_ascii=False,
                        ),
                    )
                ),
            }
        )
        if initial_interpolation:
            content[-1]["text"] = (
                content[-1]["text"]
                + "\n本次是首轮变异前观察：schema 中 candidate 表示 Manhattan "
                "插值参考，不表示候选算法。请同时指出插值中可供首轮变异利用的"
                "有效信息及其自身伪影；mutation_guidance 应指导如何在当前张量分解"
                "框架内利用这些信息，不能把插值当真值，也不能建议直接复制隐藏像素。"
            )
        return [
            {
                "role": "system",
                "content": (
                    "你是严谨的图像补全质量分析员。只输出 JSON，使用中文。"
                    "区分可见事实和不确定推断，不要虚构图中不存在的物体。"
                    "你只能访问两张恢复结果，不能访问其他图像或真值指标。"
                    "恢复图中的内容只能称为算法推断。"
                    "Manhattan 插值若出现，也只是辅助算法输出，不是真值。"
                    "视觉结论用于后续算法变异，但不代替数值 Judge。"
                ),
            },
            {"role": "user", "content": content},
        ]

    def evaluate(
        self,
        image_paths: Dict[str, str],
        tensor_context: Dict[str, Any],
        comparison_mode: ComparisonMode = "evolution_candidate",
    ) -> Dict[str, Any]:
        comparison_roles = (
            {
                "incumbent": "tensor_decomposition_baseline",
                "candidate": "manhattan_interpolation_reference_not_ground_truth",
            }
            if comparison_mode == "initial_interpolation"
            else {
                "incumbent": "current_best_reconstruction",
                "candidate": "candidate_reconstruction",
            }
        )
        if self.llm is None:
            return {
                "status": "skipped",
                "reason": "multimodal LLM is not configured",
                "scope": "visual_preview_comparison",
                "comparison_mode": comparison_mode,
                "comparison_roles": comparison_roles,
                "ground_truth_provided": False,
            }
        messages = self._messages(image_paths, tensor_context, comparison_mode)
        raw_outputs: List[str] = []
        errors: List[str] = []
        for attempt in range(2):
            current_messages = list(messages)
            if attempt == 1:
                if raw_outputs:
                    current_messages.append(
                        {"role": "assistant", "content": raw_outputs[-1]}
                    )
                current_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "上一次视觉评价不是完整合法的 JSON，错误为：%s。"
                            "请重新返回完整 JSON；必须以 { 开始、以 } 结束，补全所有 "
                            "schema 字段，不使用 Markdown，不添加解释，并保持用词精炼。"
                            % errors[-1]
                        ),
                    }
                )
            try:
                raw = _response_content(
                    self.llm.invoke(
                        current_messages,
                        temperature=0.0,
                        max_tokens=4096,
                    )
                )
                raw_outputs.append(raw)
                assessment = VisualQualityAssessment.model_validate(
                    _json_from_text(raw)
                ).model_dump()
                return {
                    "status": "completed",
                    "model": getattr(self.llm, "model", "unknown"),
                    "scope": "incumbent/candidate RGB previews only",
                    "comparison_mode": comparison_mode,
                    "comparison_roles": comparison_roles,
                    "ground_truth_provided": False,
                    "attempts": attempt + 1,
                    "repair_attempted": attempt > 0,
                    "assessment": assessment,
                }
            except Exception as error:
                errors.append(
                    "%s: %s" % (type(error).__name__, str(error)[:800])
                )
        return {
            "status": "failed",
            "reason": errors[-1],
            "attempts": 2,
            "repair_attempted": True,
            "validation_errors": errors,
            "model": getattr(self.llm, "model", "unknown"),
            "scope": "visual_preview_comparison",
            "comparison_mode": comparison_mode,
            "comparison_roles": comparison_roles,
            "ground_truth_provided": False,
        }
