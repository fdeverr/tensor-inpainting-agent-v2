import json
from types import SimpleNamespace

from PIL import Image

from research_agent.visual_evaluator import MultimodalQualityEvaluator


def _assessment_payload():
    observation = {
        "overall_sharpness": "主体边缘基本清晰，细小区域偏软。",
        "blur_and_over_smoothing": "缺失区域中心有轻微过度平滑。",
        "object_detail_clarity": "主物体轮廓可辨，高频纹理不足。",
        "edge_and_structure_continuity": "大部分边缘连续，右下方有轻微断裂。",
        "texture_recovery": "低频结构恢复较好，细颗粒纹理偏少。",
        "color_and_tone_consistency": "色调与参考图基本一致。",
        "artifacts": ["右下方有轻微块状伪影"],
        "poorly_recovered_regions": ["缺失区域右下角的斜向边缘"],
    }
    return {
        "visible_image_content": "可见区域包含一个主体物体与有纹理的背景。",
        "salient_objects_and_details": ["中央主体", "右下方斜向边缘"],
        "incumbent": observation,
        "candidate": observation,
        "comparison": "候选的主体边缘更连续，但中心纹理更平滑。",
        "candidate_improvements": ["主体边缘连续性提升"],
        "candidate_regressions": ["中心纹理被过度平滑"],
        "mutation_guidance": ["下一轮优先单独增强高频纹理保真"],
        "confidence": "high",
        "limitations": ["只根据 RGB 预览图进行观察"],
    }


def _method_selection_payload():
    return {
        "visible_structure_summary": "图像包含平滑背景和方向明显的重复边缘结构。",
        "spatial_complexity": "medium",
        "spatial_anisotropy": "width_dominant",
        "texture_complexity": "medium",
        "repetition_or_periodicity": "high",
        "channel_coupling": "strong",
        "interpolation_artifacts": ["局部边缘呈阶梯状"],
        "preferred_methods": ["tucker", "tsvd"],
        "rank_regime": {
            "height": "medium",
            "width": "high",
            "feature": "low",
            "overall": "medium",
        },
        "rationale": ["空间方向复杂度不对称", "颜色通道变化一致"],
        "confidence": "medium",
        "limitations": ["插值会压低高频复杂度"],
    }


def _images(tmp_path):
    paths = {}
    roles = ("incumbent", "candidate")
    for index, role in enumerate(roles):
        path = tmp_path / (role + ".png")
        Image.new("RGB", (12, 10), (index * 20, 30, 40)).save(path)
        paths[role] = str(path)
    return paths


def test_method_selection_assessment_uses_only_interpolation_preview(tmp_path):
    class FakeVisionLLM:
        model = "selection-vision-test"

        def __init__(self):
            self.messages = None

        def invoke(self, messages, **kwargs):
            self.messages = messages
            return SimpleNamespace(content=json.dumps(_method_selection_payload()))

    interpolation = tmp_path / "interpolation.png"
    Image.new("RGB", (12, 10), (20, 30, 40)).save(interpolation)
    llm = FakeVisionLLM()
    result = MultimodalQualityEvaluator(llm).assess_method_selection(
        str(interpolation),
        {"data_type": "color_image", "image_shape": [10, 12, 3]},
    )

    assert result["status"] == "completed"
    assert result["scope"] == "manhattan_interpolation_preview_only"
    assert result["ground_truth_provided"] is False
    assert result["assessment"]["preferred_methods"] == ["tucker", "tsvd"]
    assert result["assessment"]["rank_regime"]["width"] == "high"
    serialized = json.dumps(llm.messages, ensure_ascii=False).lower()
    assert serialized.count("data:image/jpeg;base64,") == 1
    assert "ground truth" in serialized
    assert "精确秩" in serialized


def test_multimodal_evaluator_sends_only_incumbent_and_candidate(
    tmp_path,
):
    class FakeVisionLLM:
        model = "vision-test"

        def __init__(self):
            self.messages = None

        def invoke(self, messages, **kwargs):
            self.messages = messages
            return SimpleNamespace(content=json.dumps(_assessment_payload()))

    llm = FakeVisionLLM()
    result = MultimodalQualityEvaluator(llm).evaluate(
        _images(tmp_path),
        {"data_type": "color_image", "tensor_shape": [10, 12, 3]},
    )

    assert result["status"] == "completed"
    assert result["attempts"] == 1
    assert result["repair_attempted"] is False
    assert result["ground_truth_provided"] is False
    assert result["assessment"]["confidence"] == "high"
    assert "过度平滑" in result["assessment"]["candidate"]["blur_and_over_smoothing"]
    user_content = llm.messages[1]["content"]
    assert sum(block["type"] == "image_url" for block in user_content) == 2
    assert "corrupted" not in json.dumps(llm.messages).lower()
    assert "mask" not in json.dumps(llm.messages).lower()
    assert "reference" not in json.dumps(llm.messages).lower()
    assert "ground_truth" not in json.dumps(llm.messages).lower()
    assert "full_psnr" not in json.dumps(llm.messages).lower()
    assert "composite_ssim" not in json.dumps(llm.messages).lower()
    image_urls = [
        block["image_url"]["url"]
        for block in user_content
        if block["type"] == "image_url"
    ]
    assert sum(url.startswith("data:image/jpeg;base64,") for url in image_urls) == 2


def test_initial_visual_bootstrap_compares_tensor_with_interpolation_only(tmp_path):
    class FakeVisionLLM:
        model = "vision-bootstrap-test"

        def __init__(self):
            self.messages = None

        def invoke(self, messages, **kwargs):
            self.messages = messages
            return SimpleNamespace(content=json.dumps(_assessment_payload()))

    llm = FakeVisionLLM()
    result = MultimodalQualityEvaluator(llm).evaluate(
        _images(tmp_path),
        {"data_type": "color_image", "tensor_shape": [10, 12, 3]},
        comparison_mode="initial_interpolation",
    )

    assert result["status"] == "completed"
    assert result["comparison_mode"] == "initial_interpolation"
    assert result["comparison_roles"]["incumbent"] == (
        "tensor_decomposition_baseline"
    )
    assert "interpolation_reference_not_ground_truth" in result[
        "comparison_roles"
    ]["candidate"]
    serialized = json.dumps(llm.messages, ensure_ascii=False)
    assert "当前张量分解基线" in serialized
    assert "Manhattan 插值参考" in serialized
    assert "不是真值" in serialized
    user_content = llm.messages[1]["content"]
    assert sum(block["type"] == "image_url" for block in user_content) == 2


def test_multimodal_evaluator_skips_cleanly_without_llm():
    result = MultimodalQualityEvaluator(None).evaluate({}, {})

    assert result["status"] == "skipped"
    assert result["ground_truth_provided"] is False


def test_multimodal_evaluator_failure_does_not_escape(tmp_path):
    class BrokenVisionLLM:
        model = "broken-vision"

        def invoke(self, messages, **kwargs):
            raise RuntimeError("vision endpoint unavailable")

    result = MultimodalQualityEvaluator(BrokenVisionLLM()).evaluate(
        _images(tmp_path), {}
    )

    assert result["status"] == "failed"
    assert "vision endpoint unavailable" in result["reason"]
    assert result["attempts"] == 2


def test_multimodal_evaluator_repairs_malformed_json_once(tmp_path):
    class RepairingVisionLLM:
        model = "repairing-vision"

        def __init__(self):
            self.calls = []

        def invoke(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                return SimpleNamespace(content='{"visible_image_content": "truncated"')
            return SimpleNamespace(content=json.dumps(_assessment_payload()))

    llm = RepairingVisionLLM()
    result = MultimodalQualityEvaluator(llm).evaluate(_images(tmp_path), {})

    assert result["status"] == "completed"
    assert result["attempts"] == 2
    assert result["repair_attempted"] is True
    assert len(llm.calls) == 2
    assert llm.calls[0][1]["max_tokens"] == 4096
    assert "完整合法的 JSON" in llm.calls[1][0][-1]["content"]


def test_multimodal_evaluator_retries_empty_content(tmp_path):
    class EmptyThenValidVisionLLM:
        model = "empty-then-valid"

        def __init__(self):
            self.calls = 0

        def invoke(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content="")
            return SimpleNamespace(content=json.dumps(_assessment_payload()))

    llm = EmptyThenValidVisionLLM()
    result = MultimodalQualityEvaluator(llm).evaluate(_images(tmp_path), {})

    assert result["status"] == "completed"
    assert result["attempts"] == 2
    assert llm.calls == 2
