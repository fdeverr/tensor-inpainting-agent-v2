import math
import sys
from types import SimpleNamespace

import numpy as np

from research_agent.core.metrics import (
    _LEARNED_MODEL_CACHE,
    composite_ssim,
    evaluate_reconstruction_metrics,
    full_image_mse,
    full_image_psnr,
    missing_region_mse,
    missing_region_psnr,
    structural_similarity,
)


def test_perfect_reconstruction_metrics():
    ground_truth = np.full((16, 16, 3), 0.5, dtype=np.float32)
    mask = np.ones((16, 16), dtype=np.bool_)
    mask[4:12, 4:12] = False

    assert missing_region_mse(ground_truth, ground_truth, mask) == 0.0
    assert math.isinf(missing_region_psnr(ground_truth, ground_truth, mask))
    assert full_image_mse(ground_truth, ground_truth, mask) == 0.0
    assert math.isinf(full_image_psnr(ground_truth, ground_truth, mask))
    assert np.isclose(composite_ssim(ground_truth, ground_truth, mask), 1.0)


def test_missing_region_error_is_not_diluted_by_observed_pixels():
    ground_truth = np.zeros((20, 20, 3), dtype=np.float32)
    prediction = ground_truth.copy()
    mask = np.ones((20, 20), dtype=np.bool_)
    mask[8:12, 8:12] = False
    prediction[~mask] = 1.0

    assert np.isclose(missing_region_mse(prediction, ground_truth, mask), 1.0)
    assert np.isclose(missing_region_psnr(prediction, ground_truth, mask), 0.0)
    assert np.isclose(full_image_mse(prediction, ground_truth, mask), 0.04)
    assert np.isclose(full_image_psnr(prediction, ground_truth, mask), 10 * np.log10(25))
    assert composite_ssim(prediction, ground_truth, mask) < 1.0


def test_structural_similarity_detects_change():
    first = np.zeros((16, 16, 3), dtype=np.float32)
    second = first.copy()
    second[4:12, 4:12] = 1.0

    assert structural_similarity(first, first) > structural_similarity(first, second)


def test_learned_iqa_metrics_use_pyiqa_names_and_composite_image(monkeypatch):
    ground_truth = np.full((16, 20, 3), 0.5, dtype=np.float32)
    prediction = np.zeros_like(ground_truth)
    observed_mask = np.ones((16, 20), dtype=np.bool_)
    observed_mask[4:12, 5:15] = False
    created = []

    class FakeMetric:
        def __init__(self, name):
            self.name = name

        def __call__(self, image, reference=None):
            if self.name == "lpips":
                assert reference is not None
                return (image - reference).abs().mean()
            assert reference is None
            return image.mean()

    def create_metric(name, device):
        created.append((name, device))
        return FakeMetric(name)

    monkeypatch.setitem(
        sys.modules,
        "pyiqa",
        SimpleNamespace(create_metric=create_metric),
    )
    _LEARNED_MODEL_CACHE.clear()
    metrics = evaluate_reconstruction_metrics(
        prediction,
        ground_truth,
        observed_mask,
        device="cpu",
        include_full_reference_metrics=True,
        include_no_reference_metrics=True,
    )

    assert [name for name, _ in created] == [
        "lpips",
        "maniqa",
        "clipiqa",
        "musiq",
    ]
    assert np.isclose(metrics["lpips"], 0.125)
    assert np.isclose(metrics["maniqa"], 0.375)
    assert np.isclose(metrics["clip_iqa"], 0.375)
    assert np.isclose(metrics["musiq"], 0.375)
    assert metrics["learned_metric_status"]["errors"] == {}


def test_default_metric_group_does_not_load_lpips(monkeypatch):
    ground_truth = np.full((8, 8, 3), 0.5, dtype=np.float32)
    observed_mask = np.ones((8, 8), dtype=np.bool_)
    observed_mask[2:6, 2:6] = False
    created = []

    class ConstantMetric:
        def __call__(self, *args):
            return 0.25

    def create_metric(name, device):
        created.append(name)
        return ConstantMetric()

    monkeypatch.setitem(
        sys.modules,
        "pyiqa",
        SimpleNamespace(create_metric=create_metric),
    )
    _LEARNED_MODEL_CACHE.clear()
    metrics = evaluate_reconstruction_metrics(
        ground_truth,
        ground_truth,
        observed_mask,
        device="cpu",
    )

    assert created == []
    assert metrics["lpips"] is None
    assert metrics["learned_metric_status"]["requested"] is False
    assert all(
        metrics[name] is None for name in ("maniqa", "clip_iqa", "musiq")
    )
    assert metrics["metric_group_status"]["no_reference"]["enabled"] is False


def test_one_learned_iqa_failure_does_not_discard_other_metrics(
    monkeypatch,
):
    ground_truth = np.full((8, 8, 3), 0.5, dtype=np.float32)
    prediction = ground_truth.copy()
    observed_mask = np.ones((8, 8), dtype=np.bool_)
    observed_mask[2:6, 2:6] = False

    class ConstantMetric:
        def __init__(self, value):
            self.value = value

        def __call__(self, *args):
            return self.value

    def create_metric(name, device):
        assert device == "cpu"
        if name == "maniqa":
            raise RuntimeError("checkpoint unavailable")
        return ConstantMetric(0.25)

    monkeypatch.setitem(
        sys.modules,
        "pyiqa",
        SimpleNamespace(create_metric=create_metric),
    )
    _LEARNED_MODEL_CACHE.clear()
    metrics = evaluate_reconstruction_metrics(
        prediction,
        ground_truth,
        observed_mask,
        device="cpu",
        include_full_reference_metrics=True,
        include_no_reference_metrics=True,
    )

    assert metrics["missing_mse"] == 0.0
    assert metrics["missing_psnr"] is None
    assert metrics["full_psnr"] is None
    assert metrics["composite_ssim"] == 1.0
    assert metrics["lpips"] == 0.25
    assert metrics["maniqa"] is None
    assert metrics["clip_iqa"] == 0.25
    assert metrics["musiq"] == 0.25
    assert set(metrics["learned_metric_status"]["errors"]) == {"maniqa"}
