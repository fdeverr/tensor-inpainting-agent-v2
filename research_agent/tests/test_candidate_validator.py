from pathlib import Path

from research_agent.candidate.generator import deterministic_candidate
from research_agent.candidate.validator import CandidateValidator


def _validate(tmp_path: Path, source: str):
    model_path = tmp_path / "model.py"
    model_path.write_text(source, encoding="utf-8")
    return CandidateValidator(timeout_seconds=10).validate(str(model_path))


def test_deterministic_tv_candidate_passes_all_checks(tmp_path):
    proposal = deterministic_candidate("tucker")
    result = _validate(tmp_path, proposal.model_code)
    assert result["passed"] is True
    assert result["static_validation"]["passed"] is True
    assert result["smoke_test"]["passed"] is True
    assert "tv_regularization" in result["smoke_test"]["loss_terms"]


def test_forbidden_import_is_rejected_before_execution(tmp_path):
    source = '''import os
import torch

class CandidateTensorInpaintingModel(TuckerDecomposition):
    @classmethod
    def search_space(cls, image_shape):
        return {"rank_h": [4]}
'''
    result = _validate(tmp_path, source)
    assert result["passed"] is False
    assert result["smoke_test"]["skipped"] is True
    assert any(
        error["code"] == "FORBIDDEN_IMPORT"
        for error in result["static_validation"]["errors"]
    )


def test_registering_base_owned_channel_bias_is_rejected_statically(tmp_path):
    source = '''import torch

class CandidateTensorInpaintingModel(TuckerDecomposition):
    def __init__(self, image_shape, initial_channel_mean, **kwargs):
        super().__init__(image_shape, initial_channel_mean, **kwargs)
        self.register_buffer("channel_bias", torch.zeros(3))

    @classmethod
    def search_space(cls, image_shape):
        return {"rank_h": [4]}
'''
    result = _validate(tmp_path, source)

    assert result["passed"] is False
    assert result["smoke_test"]["skipped"] is True
    assert any(
        error["code"] == "PROTECTED_BASE_ATTRIBUTE"
        for error in result["static_validation"]["errors"]
    )


def test_keyword_only_loss_terms_signature_is_rejected_statically(tmp_path):
    source = '''import torch

class CandidateTensorInpaintingModel(TuckerDecomposition):
    def loss_terms(self, *, prediction, observed, train_mask):
        return super().loss_terms(prediction, observed, train_mask)

    @classmethod
    def search_space(cls, image_shape):
        return {"rank_h": [4]}
'''
    result = _validate(tmp_path, source)

    assert result["passed"] is False
    assert result["smoke_test"]["skipped"] is True
    assert any(
        error["code"] == "INVALID_LOSS_TERMS_SIGNATURE"
        for error in result["static_validation"]["errors"]
    )


def test_wrong_forward_shape_fails_smoke_test(tmp_path):
    source = '''import torch

class CandidateTensorInpaintingModel(TuckerDecomposition):
    def forward(self):
        return super().forward()[:, :, 0]

    @classmethod
    def search_space(cls, image_shape):
        return {"rank_h": [4]}
'''
    result = _validate(tmp_path, source)
    assert result["static_validation"]["passed"] is True
    assert result["passed"] is False
    assert "forward shape mismatch" in result["smoke_test"]["message"]


def test_candidate_without_gradient_fails_smoke_test(tmp_path):
    source = '''import torch

class CandidateTensorInpaintingModel(BaseTensorInpaintingModel):
    def __init__(self, image_shape, initial_channel_mean):
        super().__init__(image_shape, initial_channel_mean)
        self.factor = torch.nn.Parameter(torch.ones(1))

    def forward(self):
        return torch.zeros(self.image_shape, dtype=torch.float32)

    @classmethod
    def search_space(cls, image_shape):
        return {"rank": [1]}
'''
    result = _validate(tmp_path, source)
    assert result["static_validation"]["passed"] is True
    assert result["passed"] is False
    assert result["smoke_test"]["error_type"] == "RuntimeError"


def test_coordinate_mlp_candidate_passes_deep_model_contract(tmp_path):
    source = '''import torch

class CandidateTensorInpaintingModel(BaseTensorInpaintingModel):
    def __init__(
        self,
        image_shape,
        initial_channel_mean,
        hidden_dim=32,
        num_frequencies=4,
    ):
        super().__init__(image_shape, initial_channel_mean)
        height, width, _ = self.image_shape
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        features = [yy, xx]
        for frequency in range(1, int(num_frequencies) + 1):
            scale = float(frequency) * torch.pi
            features.extend([
                torch.sin(scale * yy),
                torch.cos(scale * yy),
                torch.sin(scale * xx),
                torch.cos(scale * xx),
            ])
        coordinates = torch.stack(features, dim=-1).reshape(-1, len(features))
        self.register_buffer("coordinates", coordinates)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(len(features), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(hidden_dim), 3),
        )

    def forward(self):
        height, width, channels = self.image_shape
        residual = self.network(self.coordinates).reshape(height, width, channels)
        return residual + self.channel_bias

    @classmethod
    def search_space(cls, image_shape):
        del image_shape
        return {"hidden_dim": [32, 64], "num_frequencies": [4, 8]}
'''
    result = _validate(tmp_path, source)

    assert result["passed"] is True
    assert result["smoke_test"]["forward_shape"] == [17, 23, 3]
    assert result["smoke_test"]["parameter_count"] > 0
