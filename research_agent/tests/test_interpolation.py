import numpy as np
import pytest

from research_agent.core.interpolation import nearest_neighbor_fill


def test_nearest_neighbor_fill_preserves_observations_and_fills_hole():
    observed = np.zeros((5, 5, 3), dtype=np.float32)
    mask = np.ones((5, 5), dtype=np.bool_)
    mask[1:4, 1:4] = False
    observed[mask] = np.array([0.2, 0.5, 0.8], dtype=np.float32)

    reconstruction = nearest_neighbor_fill(observed, mask)

    assert np.array_equal(reconstruction[mask], observed[mask])
    assert np.allclose(reconstruction[~mask], [0.2, 0.5, 0.8])
    assert np.isfinite(reconstruction).all()


def test_nearest_neighbor_fill_rejects_fully_missing_image():
    observed = np.zeros((4, 4, 3), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.bool_)

    with pytest.raises(ValueError, match="at least one observed"):
        nearest_neighbor_fill(observed, mask)


@pytest.mark.parametrize("shape", [(5, 5, 3), (5, 5, 2, 3)])
def test_interpolation_never_copies_between_color_channels(shape):
    target = np.zeros(shape, dtype=np.float32)
    target[..., 1] = 1
    target[..., 2] = 0.7
    mask = np.ones(shape, dtype=bool)
    mask[1:4, 1:4, ..., 0] = False
    observed = np.where(mask, target, 0)
    result = nearest_neighbor_fill(observed, mask)
    assert np.array_equal(result, target)
    altered = observed.copy()
    altered[~mask] = 999  # Hidden values never influence the reconstruction.
    assert np.array_equal(nearest_neighbor_fill(altered, mask), result)


def test_video_interpolation_uses_time_for_a_whole_missing_frame():
    target = np.empty((3, 4, 3, 3), dtype=np.float32)
    target[:] = [0.2, 0.5, 0.8]
    mask = np.ones_like(target, dtype=bool)
    mask[:, :, 1, :] = False
    result = nearest_neighbor_fill(np.where(mask, target, 0), mask)
    assert np.array_equal(result, target)


def test_msi_interpolation_can_use_ordered_bands_when_a_band_is_missing():
    target = np.full((3, 4, 4), 0.5, dtype=np.float32)
    mask = np.ones_like(target, dtype=bool)
    mask[..., 1] = False
    assert np.array_equal(nearest_neighbor_fill(np.where(mask, target, 0), mask, "MSI"), target)


def test_entire_missing_rgb_channel_fails_instead_of_copying_other_colors():
    observed = np.zeros((3, 4, 3), dtype=np.float32)
    mask = np.ones_like(observed, dtype=bool)
    mask[..., 0] = False
    with pytest.raises(ValueError, match="each color channel"):
        nearest_neighbor_fill(observed, mask)


def test_audio_interpolation_dispatch_matches_continuous_linear_reference():
    from research_agent.core.audio_metrics import audio_metric_context
    from research_agent.core.interpolation import linear_waveform_fill
    target = np.arange(10, dtype=np.float32).reshape(2, 5, 1) / 10
    mask = np.ones_like(target, dtype=bool)
    mask.reshape(-1)[3:7] = False
    observed = np.where(mask, target, 0)
    with audio_metric_context({"data_type": "audio", "sample_count": 10}):
        result = nearest_neighbor_fill(observed, mask)
    np.testing.assert_allclose(result, target)
    assert np.array_equal(result, linear_waveform_fill(observed, mask, 10))
@pytest.mark.parametrize("data_type,success", [("msi", True), ("color_image", False)])
def test_interpolation_tool_respects_explicit_semantics_for_three_channels(tmp_path, data_type, success):
    from research_agent.agent_tools.research_tools import RunInterpolationTool
    tensor = np.full((4, 5, 3), 0.5, dtype=np.float32)
    mask = np.ones_like(tensor, dtype=bool)
    mask[..., 1] = False
    np.save(tmp_path / "observed.npy", np.where(mask, tensor, 0))
    np.save(tmp_path / "mask.npy", mask)
    response = RunInterpolationTool().run({"run_id": "test", "corrupted_path": str(tmp_path / "observed.npy"),
        "mask_path": str(tmp_path / "mask.npy"), "output_path": str(tmp_path / "filled.npy"), "data_type": data_type})
    if success:
        np.testing.assert_array_equal(np.load(response.data["artifacts"]["reconstruction"]), tensor)
    else:
        assert response.error_info["code"] == "INVALID_PARAM"
