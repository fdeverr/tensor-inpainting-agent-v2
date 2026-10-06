import numpy as np
import pytest
import torch
from torch import nn

from research_agent.core.models import create_model
from research_agent.core.models.registry import MODEL_CLASSES, get_default_hyperparameters
from research_agent.schemas import SUPPORTED_MODEL_NAMES


MODEL_CASES = (
    (
        "matrix",
        {"rank": 4, "init_scale": 0.1},
        {"left_factor", "right_factor", "channel_bias"},
    ),
    (
        "mode3",
        {"rank": 2, "init_scale": 0.1},
        {"coefficient_tensor", "channel_factor", "channel_bias"},
    ),
    (
        "cp",
        {"rank": 4, "init_scale": 0.2},
        {"height_factor", "width_factor", "channel_factor", "channel_bias"},
    ),
    (
        "nonnegative_cp",
        {"rank": 4, "init_scale": 0.1},
        {
            "raw_height_factor",
            "raw_width_factor",
            "raw_channel_factor",
            "channel_bias",
        },
    ),
    (
        "tucker",
        {"rank_h": 4, "rank_w": 3, "rank_c": 2, "init_scale": 0.15},
        {"core", "height_factor", "width_factor", "channel_factor", "channel_bias"},
    ),
    (
        "btd",
        {
            "num_blocks": 2,
            "rank_h": 4,
            "rank_w": 3,
            "rank_c": 2,
            "init_scale": 0.1,
        },
        {
            "cores",
            "height_factors",
            "width_factors",
            "channel_factors",
            "channel_bias",
        },
    ),
    (
        "tsvd",
        {"rank": 4, "init_scale": 0.1},
        {"left_factor", "singular_tubes", "right_factor", "channel_bias"},
    ),
    (
        "nonnegative_tucker",
        {"rank_h": 4, "rank_w": 3, "rank_c": 2, "init_scale": 0.1},
        {
            "raw_core",
            "raw_height_factor",
            "raw_width_factor",
            "raw_channel_factor",
            "channel_bias",
        },
    ),
    (
        "hierarchical_tucker",
        {
            "rank_h": 4,
            "rank_w": 3,
            "rank_c": 2,
            "rank_spatial": 2,
            "init_scale": 0.1,
        },
        {
            "height_factor",
            "width_factor",
            "channel_factor",
            "spatial_transfer",
            "root_transfer",
            "channel_bias",
        },
    ),
    (
        "tt",
        {"rank_1": 4, "rank_2": 2, "init_scale": 0.1},
        {"height_core", "width_core", "channel_core", "channel_bias"},
    ),
    (
        "tensor_ring",
        {"rank": 4, "init_scale": 0.1},
        {"height_core", "width_core", "channel_core", "channel_bias"},
    ),
)


def test_registry_schema_and_defaults_cover_the_same_model_names():
    assert set(MODEL_CLASSES) == SUPPORTED_MODEL_NAMES
    assert all(get_default_hyperparameters(name) for name in SUPPORTED_MODEL_NAMES)


def test_siren_forward_backward_matches_image_shape():
    model = create_model(
        model_name="siren",
        image_shape=(8, 7, 3),
        initial_channel_mean=[0.4, 0.5, 0.6],
        hyperparameters={
            "hidden_features": 16,
            "hidden_layers": 2,
            "first_omega_0": 30.0,
            "hidden_omega_0": 30.0,
        },
    )
    prediction = model()
    loss = prediction.square().mean()
    loss.backward()
    assert tuple(prediction.shape) == (8, 7, 3)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_siren_audio_uses_original_time_across_frames_and_padding():
    from research_agent.core.audio_metrics import audio_metric_context
    with audio_metric_context({"data_type": "audio", "sample_count": 10}):
        model = create_model("siren", (3, 4, 2), [0.4, 0.5], {"hidden_features": 16})
    assert model.coordinates.shape == (12, 1)
    torch.testing.assert_close(model.coordinates[:10, 0], torch.linspace(-1, 1, 10))
    assert model.coordinates[10, 0] > 1  # Padding does not stretch valid time.
    assert model.network[0].linear.in_features == 1
    assert model.output_layer.out_features == 2
    output = model()
    assert output.shape == (3, 4, 2)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_siren_video_uses_time_as_input_and_only_channels_as_output():
    model = create_model("siren", (2, 3, 4, 2), [0.4, 0.5] * 4, {"hidden_features": 16})
    assert model.network[0].linear.in_features == 3
    assert model.output_layer.out_features == 2
    grid = model.coordinates.reshape(2, 3, 4, 3)
    torch.testing.assert_close(grid[0, 0, :, 2], torch.linspace(-1, 1, 4))
    assert (grid[0, 0, :, :2] == -1).all()
    output = model()
    assert output.shape == (2, 3, 4, 2)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_siren_audio_is_invariant_to_storage_frame_size():
    from research_agent.core.audio_metrics import audio_metric_context
    params = {"hidden_features": 16, "hidden_layers": 2}
    with audio_metric_context({"data_type": "audio", "sample_count": 10}):
        small_frames = create_model("siren", (3, 4, 2), [0.4, 0.5], params)
        large_frames = create_model("siren", (2, 8, 2), [0.4, 0.5], params)
    large_frames.load_state_dict(small_frames.state_dict())
    torch.testing.assert_close(small_frames.coordinates[:10], large_frames.coordinates[:10])
    torch.testing.assert_close(small_frames().reshape(-1, 2)[:10], large_frames().reshape(-1, 2)[:10])


@pytest.mark.parametrize("shape,mode,mean", [((3, 4, 2), "audio", [0.4, 0.5]),
    ((3, 4, 2), "image", [0.4, 0.5]), ((3, 4, 2, 2), "video", [0.4, 0.5] * 2)])
def test_siren_coordinate_chunks_preserve_outputs_and_gradients(shape, mode, mean):
    params = {"hidden_features": 16, "hidden_layers": 2, "coordinate_mode": mode}
    whole = create_model("siren", shape, mean, {**params, "coordinate_batch_size": 1000})
    chunks = create_model("siren", shape, mean, {**params, "coordinate_batch_size": 3})
    chunks.load_state_dict(whole.state_dict())
    expected, actual = whole(), chunks()
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for left, right in zip(whole.parameters(), chunks.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("model_name,hyperparameters,expected_parameters", MODEL_CASES)
def test_tensor_model_forward_backward_and_parameters(
    model_name,
    hyperparameters,
    expected_parameters,
):
    torch.manual_seed(3)
    image_shape = (8, 7, 3)
    model = create_model(
        model_name=model_name,
        image_shape=image_shape,
        initial_channel_mean=[0.4, 0.5, 0.6],
        hyperparameters=hyperparameters,
    )
    prediction = model()
    observed = torch.rand(image_shape)
    train_mask = torch.ones(image_shape[:2], dtype=torch.bool)
    train_mask[2:4, 3:5] = False
    loss = sum(model.loss_terms(prediction, observed, train_mask).values())
    loss.backward()

    named_parameters = dict(model.named_parameters())
    assert tuple(prediction.shape) == image_shape
    assert expected_parameters == set(named_parameters)
    assert all(isinstance(parameter, nn.Parameter) for parameter in named_parameters.values())
    assert all(parameter.grad is not None for parameter in named_parameters.values())
    assert all(torch.isfinite(parameter.grad).all() for parameter in named_parameters.values())


def test_tucker_rejects_rank_larger_than_image_dimension():
    with pytest.raises(ValueError, match="Tucker ranks"):
        create_model(
            model_name="tucker",
            image_shape=(8, 7, 3),
            initial_channel_mean=np.full(3, 0.5),
            hyperparameters={"rank_h": 9, "rank_w": 3, "rank_c": 2},
        )


def test_mode3_forward_is_exact_a_times_3_e_product():
    model = create_model(
        model_name="mode3",
        image_shape=(2, 3, 3),
        initial_channel_mean=np.zeros(3),
        hyperparameters={"rank": 2, "init_scale": 0.1},
    )
    with torch.no_grad():
        model.coefficient_tensor.copy_(torch.arange(12).reshape(2, 3, 2))
        model.channel_factor.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        )
        model.channel_bias.zero_()

    expected = torch.einsum(
        "ijr,cr->ijc",
        model.coefficient_tensor,
        model.channel_factor,
    )
    assert torch.equal(model(), expected)


def test_mode3_rejects_rank_larger_than_channel_dimension():
    with pytest.raises(ValueError, match="Mode-3 rank"):
        create_model(
            model_name="mode3",
            image_shape=(8, 7, 3),
            initial_channel_mean=np.full(3, 0.5),
            hyperparameters={"rank": 4},
        )


def test_nonnegative_tucker_output_is_nonnegative():
    model = create_model(
        model_name="nonnegative_tucker",
        image_shape=(8, 7, 3),
        initial_channel_mean=np.full(3, 0.5),
        hyperparameters={"rank_h": 4, "rank_w": 3, "rank_c": 2},
    )
    assert bool((model() >= 0.0).all())


def test_nonnegative_cp_output_is_nonnegative_and_mean_initialized():
    channel_mean = np.asarray([0.3, 0.5, 0.7], dtype=np.float32)
    model = create_model(
        model_name="nonnegative_cp",
        image_shape=(8, 7, 3),
        initial_channel_mean=channel_mean,
        hyperparameters={"rank": 8, "init_scale": 0.01},
    )
    prediction = model()

    assert bool((prediction >= 0.0).all())
    assert torch.allclose(
        prediction.mean(dim=(0, 1)),
        torch.as_tensor(channel_mean),
        atol=0.08,
        rtol=0.08,
    )


def test_tsvd_rejects_rank_larger_than_spatial_matrix_dimensions():
    with pytest.raises(ValueError, match="t-SVD tubal rank"):
        create_model(
            model_name="tsvd",
            image_shape=(8, 7, 3),
            initial_channel_mean=np.full(3, 0.5),
            hyperparameters={"rank": 8},
        )


def test_hierarchical_tucker_rejects_incompatible_spatial_rank():
    with pytest.raises(ValueError, match="rank_spatial"):
        create_model(
            model_name="hierarchical_tucker",
            image_shape=(8, 7, 3),
            initial_channel_mean=np.full(3, 0.5),
            hyperparameters={
                "rank_h": 4,
                "rank_w": 3,
                "rank_c": 2,
                "rank_spatial": 3,
            },
        )


def test_tt_rejects_second_rank_larger_than_channel_unfolding():
    with pytest.raises(ValueError, match="TT rank_2"):
        create_model(
            model_name="tt",
            image_shape=(8, 7, 3),
            initial_channel_mean=np.full(3, 0.5),
            hyperparameters={"rank_1": 4, "rank_2": 4},
        )


def test_tensor_ring_rejects_excessive_ring_rank():
    with pytest.raises(ValueError, match="Tensor Ring rank"):
        create_model(
            model_name="tensor_ring",
            image_shape=(8, 7, 3),
            initial_channel_mean=np.full(3, 0.5),
            hyperparameters={"rank": 8},
        )
