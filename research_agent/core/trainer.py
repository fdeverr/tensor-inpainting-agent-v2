"""Fixed Trainer shared by every tensor-decomposition model."""

from __future__ import annotations

import copy
import math
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..schemas import TrainingConfig
from .masks import split_observed_mask
from .models import create_model


ModelBuilder = Callable[
    [Tuple[int, ...], Sequence[float], Dict[str, Any]],
    torch.nn.Module,
]


def _duration_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return "%dh%02dm%02ds" % (hours, minutes, seconds)
    if minutes:
        return "%dm%02ds" % (minutes, seconds)
    return "%ds" % seconds


class _ConsoleProgress:
    """Small dependency-free progress bar for interactive and captured output."""

    def __init__(self, label: str, total_steps: int) -> None:
        self.label = label
        self.total_steps = max(1, int(total_steps))
        self.started_at = time.perf_counter()
        self.interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.last_bucket = -1
        self.last_noninteractive_print = self.started_at
        self.last_noninteractive_step = -1
        self.last_message_length = 0
        print("\n▶ %s（共 %d 步）" % (self.label, self.total_steps), flush=True)

    def update(
        self,
        step: int,
        train_loss: float,
        validation_loss: Optional[float] = None,
        status: str = "训练中",
        force: bool = False,
    ) -> None:
        step = max(0, min(int(step), self.total_steps))
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        eta = (elapsed / step) * (self.total_steps - step) if step else 0.0
        ratio = step / self.total_steps
        filled = min(24, int(round(24 * ratio)))
        bar = "█" * filled + "░" * (24 - filled)
        validation_text = (
            " val=%.6g" % validation_loss
            if validation_loss is not None
            else ""
        )
        message = (
            "[%s] %d/%d %6.2f%% loss=%.6g%s ETA %s  %s"
            % (
                bar,
                step,
                self.total_steps,
                ratio * 100.0,
                train_loss,
                validation_text,
                _duration_text(eta),
                status,
            )
        )
        if self.interactive:
            padding = " " * max(0, self.last_message_length - len(message))
            sys.stdout.write("\r" + message + padding)
            sys.stdout.flush()
            self.last_message_length = len(message)
            return

        bucket = int(ratio * 10)
        now = time.perf_counter()
        if force and step == self.last_noninteractive_step and status.startswith("完成"):
            return
        if (
            force
            or step in {1, self.total_steps}
            or bucket > self.last_bucket
            or now - self.last_noninteractive_print >= 30.0
        ):
            print(message, flush=True)
            self.last_bucket = bucket
            self.last_noninteractive_print = now
            self.last_noninteractive_step = step

    def finish(
        self,
        step: int,
        train_loss: float,
        validation_loss: Optional[float] = None,
        status: str = "完成",
    ) -> None:
        self.update(
            step=step,
            train_loss=train_loss,
            validation_loss=validation_loss,
            status=status,
            force=True,
        )
        if self.interactive:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _build_model(
    model_name: str,
    image_shape: Tuple[int, ...],
    initial_channel_mean: Sequence[float],
    model_hyperparameters: Dict[str, Any],
    model_builder: Optional[ModelBuilder],
) -> torch.nn.Module:
    if model_builder is not None:
        return model_builder(
            image_shape,
            initial_channel_mean,
            model_hyperparameters,
        )
    return create_model(
        model_name=model_name,
        image_shape=image_shape,
        initial_channel_mean=initial_channel_mean,
        hyperparameters=model_hyperparameters,
    )


@dataclass
class TrainingOutput:
    """Internal training result; reconstruction arrays are saved by a pipeline."""

    reconstruction: np.ndarray
    history: List[Dict[str, Any]]
    best_step: int
    best_validation_mse: float
    best_missing_psnr: Optional[float]
    selection_metric: str
    ground_truth_used_for_selection: bool
    runtime_seconds: float
    parameter_count: int
    device: str
    stopped_early: bool
    train_mask: np.ndarray
    validation_mask: np.ndarray
    state_dict: Dict[str, torch.Tensor]


@dataclass
class FinalFitOutput:
    """Result of refitting a selected configuration on every observed pixel."""

    reconstruction: np.ndarray
    history: List[Dict[str, Any]]
    fitted_steps: int
    final_train_mse: float
    final_total_loss: float
    runtime_seconds: float
    parameter_count: int
    device: str
    fit_mask: np.ndarray
    state_dict: Dict[str, torch.Tensor]


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` to CUDA on Linux servers and CPU otherwise."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested_device)


def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def _validate_training_arrays(
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
) -> None:
    if observed_image.ndim not in (3, 4):
        raise ValueError("observed_image must have shape [H,W,C] or [H,W,T,C]")
    if not np.issubdtype(observed_image.dtype, np.floating):
        raise ValueError("observed_image must be floating point")
    if observed_mask.shape != observed_image.shape[:2] or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be bool and match image height and width")
    if not observed_mask.any():
        raise ValueError("at least one observed pixel is required")
    if not np.isfinite(observed_image).all():
        raise ValueError("observed_image contains NaN or Inf")
    if ground_truth is not None:
        if ground_truth.shape != observed_image.shape:
            raise ValueError("ground_truth must match observed_image shape")
        if not np.issubdtype(ground_truth.dtype, np.floating):
            raise ValueError("ground_truth must be floating point")
        if not np.isfinite(ground_truth).all():
            raise ValueError("ground_truth contains NaN or Inf")
        if observed_mask.all():
            raise ValueError("ground-truth selection requires at least one missing pixel")


def _validation_mse(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    validation_mask: torch.Tensor,
) -> torch.Tensor:
    expanded_mask = validation_mask[
        (...,) + (None,) * (prediction.ndim - 2)
    ].expand_as(prediction)
    return torch.square(prediction - observed)[expanded_mask].mean()


def train_tensor_model(
    model_name: str,
    model_hyperparameters: Dict[str, Any],
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    config: TrainingConfig,
    seed: int,
    model_builder: Optional[ModelBuilder] = None,
    progress_label: Optional[str] = None,
    ground_truth: Optional[np.ndarray] = None,
) -> TrainingOutput:
    """Fit one model and select its checkpoint.

    When ``ground_truth`` is provided, every observed pixel is used for gradient
    training and the missing-region GT MSE selects checkpoints.  The GT tensor is
    never included in the optimization loss.  Omitting it preserves the legacy
    held-out-observed diagnostic path for low-level callers.
    """

    config.validate()
    _validate_training_arrays(observed_image, observed_mask, ground_truth)
    set_reproducibility(seed, config.deterministic)
    device = resolve_device(config.device)
    if ground_truth is not None:
        train_mask_np = observed_mask.copy()
        validation_mask_np = ~observed_mask
        selection_reference_np = ground_truth
        selection_metric = "missing_region_ground_truth_mse"
    else:
        train_mask_np, validation_mask_np = split_observed_mask(
            observed_mask,
            validation_ratio=config.validation_observed_ratio,
            seed=seed + 10_003,
            strategy=config.validation_strategy,
        )
        selection_reference_np = observed_image
        selection_metric = "%s_held_out_observed_mse" % config.validation_strategy

    # Initialization statistics use training pixels only, not validation pixels.
    initial_channel_mean = observed_image[train_mask_np].mean(axis=0)
    model = _build_model(
        model_name=model_name,
        image_shape=tuple(observed_image.shape),
        initial_channel_mean=initial_channel_mean,
        model_hyperparameters=model_hyperparameters,
        model_builder=model_builder,
    ).to(device)

    observed = torch.as_tensor(observed_image, dtype=torch.float32, device=device)
    selection_reference = torch.as_tensor(
        selection_reference_np, dtype=torch.float32, device=device
    )
    train_mask = torch.as_tensor(train_mask_np, dtype=torch.bool, device=device)
    validation_mask = torch.as_tensor(
        validation_mask_np,
        dtype=torch.bool,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_validation_mse = float("inf")
    best_step = 0
    best_state = None
    checks_without_improvement = 0
    stopped_early = False
    history = []
    started_at = time.perf_counter()
    progress = _ConsoleProgress(
        progress_label or ("%s 参数选择" % model_name),
        config.max_steps,
    )
    progress_interval = max(1, min(config.validation_interval, config.max_steps // 100 or 1))

    for step in range(1, config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model()
        loss_terms = model.loss_terms(prediction, observed, train_mask)
        total_loss = sum(loss_terms.values())
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("training loss became NaN or Inf at step %d" % step)
        total_loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("model gradient became NaN or Inf at step %d" % step)
        optimizer.step()

        should_validate = (
            step == 1
            or step % config.validation_interval == 0
            or step == config.max_steps
        )
        if not should_validate:
            if step % progress_interval == 0:
                progress.update(
                    step=step,
                    train_loss=float(total_loss.detach().item()),
                )
            continue

        model.eval()
        with torch.no_grad():
            current_prediction = model()
            validation_mse = float(
                _validation_mse(
                    current_prediction, selection_reference, validation_mask
                ).item()
            )
        missing_psnr = (
            None
            if ground_truth is None or validation_mse <= 0.0
            else float(10.0 * math.log10(1.0 / validation_mse))
        )
        history.append(
            {
                "step": step,
                "total_train_loss": float(total_loss.detach().item()),
                "data_train_loss": float(loss_terms["data_loss"].detach().item()),
                "validation_mse": validation_mse,
                "selection_metric": selection_metric,
                "missing_gt_mse": validation_mse if ground_truth is not None else None,
                "missing_gt_psnr": missing_psnr,
            }
        )

        if validation_mse < best_validation_mse - config.early_stopping_min_delta:
            best_validation_mse = validation_mse
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        progress.update(
            step=step,
            train_loss=float(total_loss.detach().item()),
            validation_loss=validation_mse,
        )

        if checks_without_improvement >= config.early_stopping_patience:
            stopped_early = True
            break

    progress.finish(
        step=step,
        train_loss=float(total_loss.detach().item()),
        validation_loss=validation_mse,
        status=("早停，best=%d" % best_step) if stopped_early else ("完成，best=%d" % best_step),
    )

    if best_state is None or not math.isfinite(best_validation_mse):
        raise RuntimeError("training did not produce a finite validation checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    checkpoint_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    with torch.no_grad():
        reconstruction = model().clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)

    return TrainingOutput(
        reconstruction=reconstruction,
        history=history,
        best_step=best_step,
        best_validation_mse=best_validation_mse,
        best_missing_psnr=(
            None
            if ground_truth is None or best_validation_mse <= 0.0
            else float(10.0 * math.log10(1.0 / best_validation_mse))
        ),
        selection_metric=selection_metric,
        ground_truth_used_for_selection=ground_truth is not None,
        runtime_seconds=float(time.perf_counter() - started_at),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        device=str(device),
        stopped_early=stopped_early,
        train_mask=train_mask_np,
        validation_mask=validation_mask_np,
        state_dict=checkpoint_state,
    )


def fit_tensor_model_on_all_observations(
    model_name: str,
    model_hyperparameters: Dict[str, Any],
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    config: TrainingConfig,
    selected_steps: int,
    seed: int,
    model_builder: Optional[ModelBuilder] = None,
    progress_label: Optional[str] = None,
) -> FinalFitOutput:
    """Refit a selected model using 100% of the genuinely observed pixels.

    This function performs no validation and receives no missing-region ground
    truth. ``selected_steps`` must be chosen before this call, normally by
    ``train_tensor_model`` on a temporary train/validation split.
    """

    config.validate()
    _validate_training_arrays(observed_image, observed_mask)
    if selected_steps < 1:
        raise ValueError("selected_steps must be positive")

    set_reproducibility(seed, config.deterministic)
    device = resolve_device(config.device)
    initial_channel_mean = observed_image[observed_mask].mean(axis=0)
    model = _build_model(
        model_name=model_name,
        image_shape=tuple(observed_image.shape),
        initial_channel_mean=initial_channel_mean,
        model_hyperparameters=model_hyperparameters,
        model_builder=model_builder,
    ).to(device)
    observed = torch.as_tensor(observed_image, dtype=torch.float32, device=device)
    fit_mask = torch.as_tensor(observed_mask, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    started_at = time.perf_counter()
    progress = _ConsoleProgress(
        progress_label or ("%s 全观测像素重训" % model_name),
        selected_steps,
    )
    progress_interval = max(1, min(config.validation_interval, selected_steps // 100 or 1))

    for step in range(1, selected_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model()
        loss_terms = model.loss_terms(prediction, observed, fit_mask)
        total_loss = sum(loss_terms.values())
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("final-fit loss became NaN or Inf at step %d" % step)
        total_loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(
                    "final-fit gradient became NaN or Inf at step %d" % step
                )
        optimizer.step()

        should_record = (
            step == 1
            or step % config.validation_interval == 0
            or step == selected_steps
        )
        if not should_record:
            if step % progress_interval == 0:
                progress.update(
                    step=step,
                    train_loss=float(total_loss.detach().item()),
                )
            continue
        model.eval()
        with torch.no_grad():
            current_prediction = model()
            current_terms = model.loss_terms(current_prediction, observed, fit_mask)
        history.append(
            {
                "step": step,
                "total_train_loss": float(sum(current_terms.values()).item()),
                "data_train_loss": float(current_terms["data_loss"].item()),
            }
        )
        progress.update(
            step=step,
            train_loss=float(sum(current_terms.values()).item()),
        )

    progress.finish(
        step=selected_steps,
        train_loss=float(sum(current_terms.values()).item()),
        status="完成",
    )

    model.eval()
    with torch.no_grad():
        final_prediction = model()
        final_terms = model.loss_terms(final_prediction, observed, fit_mask)
        reconstruction = (
            final_prediction.clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        )
    checkpoint_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

    return FinalFitOutput(
        reconstruction=reconstruction,
        history=history,
        fitted_steps=selected_steps,
        final_train_mse=float(final_terms["data_loss"].item()),
        final_total_loss=float(sum(final_terms.values()).item()),
        runtime_seconds=float(time.perf_counter() - started_at),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        device=str(device),
        fit_mask=observed_mask.copy(),
        state_dict=checkpoint_state,
    )
