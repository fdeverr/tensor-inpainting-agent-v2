"""Prompt construction and constrained fallback candidate generation."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..baseline_diagnostics import summarize_baseline_history
from ..method_selector import (
    METHOD_SELECTION_VISUAL_SOURCE,
    _json_from_text,
    llm_safe_image_profile,
)
from .schemas import CandidateProposal, ExperienceExtraction


PROMPT_VERSION = "reference-guided-attention-dilation-v23"
ALLOWED_CHANGES = [
    "tensor mechanisms: asymmetric or hierarchical ranks, factor sharing, local tensor blocks, factor initialization, and spatial/channel/time-specific parameterizations",
    "implicit representations: coordinate MLPs, Fourier features, sinusoidal activations, multi-resolution grids, and low-frequency backbones with high-frequency residuals",
    "convolutional mechanisms: Conv2d/Conv3d decoders, residual refiners, depthwise-separable, dilated, large-kernel, spatial-channel separable, and axis-separable spatiotemporal convolutions",
    "multi-scale mechanisms: feature pyramids, U-Net-like encoder-decoders, skip connections, parallel receptive fields, Laplacian/residual pyramids, and learned residuals at several scales",
    "efficient Transformer/attention: axial, patch, windowed, channel, temporal, spatial, or latent-token attention and gated local-global fusion",
    "hybrids that retain the selected tensor contraction while augmenting it with an MLP, convolutional, multi-scale, or efficient-attention branch",
    "loss mechanisms: smoothness, total variation, gradients, edges, frequency/wavelet or multi-scale consistency, local variance, anisotropic, spectral/channel/temporal consistency, and mask-aware differentiable priors",
    "optimization-compatible model mechanisms: learnable loss weights, residual gates, initialization scales, normalization, and bounded internal stabilization without changing the shared Trainer",
    "evidence-backed simplification: modify or structurally remove an incumbent module, branch, path, loss term, or interaction that ablation or repeated failure shows is ineffective, harmful, or unnecessarily costly",
]
FORBIDDEN_CHANGES = [
    "pretrained weights, external datasets, foundation models, or downloaded assets",
    "access to complete ground truth, per-pixel hidden targets, residual maps, or evaluation code during training; candidate design may use only controller-provided aggregate reference metrics",
    "network, subprocess, shell, or external-program calls",
    "changes to evaluation code",
    "candidate-specific training loops, optimizers, schedulers, or coarse-to-fine training",
    "full-resolution global attention over all H*W pixels; use axial, windowed, patch, or latent attention",
    "architectures whose memory is quadratic in the number of image pixels",
]


REPORT_ONLY_PSNR_KEYS = {
    "full_psnr",
    "full_psnr_db",
    "full_image_psnr_delta_db",
}
MUTATION_EXCLUDED_METRIC_KEYS = {
    "maniqa",
    "clip_iqa",
    "musiq",
    "learned_metric_status",
}
INTERPOLATION_OBJECTIVE_KEYS = {
    "interpolation_metrics",
    "interpolation_relative_objective",
    "incumbent_relative_to_interpolation",
    "candidate_relative_to_interpolation",
    "initial_interpolation_visual_assessment",
}
REFERENCE_METRIC_KEYS = {"missing_psnr", "composite_ssim", "lpips", "missing_nmse"}

EXPLANATORY_TEXT_LIMITS = {
    "mutation_goal": 800,
    "idea": 1200,
    "single_change": 800,
    "hypothesis": 1500,
    "expected_effect": 1000,
}
TRAINING_BUDGET_TEXT_LIMITS = {"rationale": 800}


def _compact_explanatory_text(value: Any, limit: int) -> Any:
    """Bound prose fields while preserving both their premise and conclusion."""

    if not isinstance(value, str) or len(value) <= limit:
        return value
    marker = " ...[middle omitted]... "
    available = limit - len(marker)
    head_length = (available * 2) // 3
    tail_length = available - head_length
    return value[:head_length].rstrip() + marker + value[-tail_length:].lstrip()


def _normalize_candidate_payload(payload: Any) -> Any:
    """Safely compact only explanatory prose; never alter generated model code."""

    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    for field_name, limit in EXPLANATORY_TEXT_LIMITS.items():
        if field_name in normalized:
            normalized[field_name] = _compact_explanatory_text(
                normalized[field_name], limit
            )
    training_budget = normalized.get("training_budget")
    if isinstance(training_budget, dict):
        normalized_budget = dict(training_budget)
        for field_name, limit in TRAINING_BUDGET_TEXT_LIMITS.items():
            if field_name in normalized_budget:
                normalized_budget[field_name] = _compact_explanatory_text(
                    normalized_budget[field_name], limit
                )
        normalized["training_budget"] = normalized_budget
    return normalized


def _evolution_context_without_full_psnr(value: Any) -> Any:
    """Keep only the metrics intentionally exposed to mutation reasoning."""

    if isinstance(value, dict):
        return {
            key: _evolution_context_without_full_psnr(item)
            for key, item in value.items()
            if key
            not in (
                REPORT_ONLY_PSNR_KEYS
                | MUTATION_EXCLUDED_METRIC_KEYS
                | INTERPOLATION_OBJECTIVE_KEYS
            )
        }
    if isinstance(value, list):
        return [_evolution_context_without_full_psnr(item) for item in value]
    return value


def _reference_metrics(metrics: Any) -> Dict[str, Any]:
    """Return the comparable final metrics intentionally exposed to evolution."""

    if not isinstance(metrics, dict):
        return {}
    return {
        key: value
        for key, value in metrics.items()
        if key in REFERENCE_METRIC_KEYS and (key != "lpips" or value is not None)
    }


def _algorithm_comparison_reference(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build one audited reference panel from the completed Day 4 run.

    These final metrics may use the artificial hidden-region ground truth. The panel
    therefore declares that it is evolution feedback rather than an untouched test.
    Raw tensors and ground-truth contents are never included.
    """

    results = state.get("results", {})
    base_method = state.get("selected_model")
    rows: List[Dict[str, Any]] = []

    interpolation_metrics = _reference_metrics(results.get("interpolation_metrics"))
    if interpolation_metrics:
        rows.append(
            {
                "algorithm": "nearest_neighbor_manhattan",
                "role": "interpolation_baseline",
                "metrics": interpolation_metrics,
            }
        )

    tensor_metrics = _reference_metrics(results.get("tensor_metrics"))
    if tensor_metrics:
        rows.append(
            {
                "algorithm": base_method,
                "role": "selected_tensor_baseline",
                "metrics": tensor_metrics,
            }
        )

    siren = results.get("siren_comparison") or {}
    siren_metrics = _reference_metrics(siren.get("metrics"))
    if siren.get("status") == "completed" and siren_metrics:
        rows.append(
            {
                "algorithm": "siren",
                "role": "implicit_neural_baseline",
                "metrics": siren_metrics,
            }
        )

    ranked_rows = sorted(
        rows,
        key=lambda item: (
            item["metrics"].get("missing_psnr") is not None,
            float(item["metrics"]["missing_psnr"])
            if item["metrics"].get("missing_psnr") is not None
            else float("-inf"),
        ),
        reverse=True,
    )
    audio = any("missing_nmse" in row["metrics"] for row in rows)
    if audio:
        from ..core.audio_metrics import metric_score
        ranked_rows = sorted(rows, key=lambda item: metric_score(item["metrics"]), reverse=True)
    winner = ranked_rows[0]["algorithm"] if ranked_rows else None
    base_psnr = tensor_metrics.get("missing_psnr")
    best_psnr = (
        ranked_rows[0]["metrics"].get("missing_psnr") if ranked_rows else None
    )
    base_gap = (
        float(base_psnr) - float(best_psnr)
        if base_psnr is not None and best_psnr is not None
        else None
    )

    # The complete 3–5-method screen stays in Day 4 artifacts and the report.
    # Mutation context needs only the selected tensor family's screen evidence;
    # independent fixed comparison baselines remain available in the cohort panel.
    screening = results.get("method_screening") or {}
    screening_rows = []
    selected_method = screening.get("winner")
    for item in screening.get("results", []):
        if not isinstance(item, dict) or item.get("method") != selected_method:
            continue
        selected_row = {
            "method": item["method"],
            "best_validation_mse": item.get("best_validation_mse"),
            "trial_count": item.get("trial_count"),
        }
        dataset_evaluation = item.get("dataset_evaluation") or {}
        if dataset_evaluation:
            selected_row["whole_modality_summary"] = dataset_evaluation.get("summary")
            selected_row["samples"] = [
                {"sample_index": index, "status": sample.get("status"),
                 "metrics": _reference_metrics(sample.get("metrics")),
                 "training": sample.get("training"), "error": sample.get("error")}
                for index, sample in enumerate(dataset_evaluation.get("results", []))
            ]
        screening_rows.append(selected_row)

    reference = {
        "purpose": (
            "Provide explicit performance anchors and gap signals for candidate design; "
            "the fixed incumbent-versus-candidate Judge remains the promotion authority."
        ),
        "evaluation_feedback_reused_for_evolution": bool(rows),
        "evidence_boundary": (
            "Final reference metrics are aggregate scores computed with the artificial "
            "hidden-region ground truth. Because they are exposed to evolution, they are "
            "development feedback rather than an untouched final-test claim. No ground-truth "
            "tensor values or residual maps are exposed."
        ),
        "selection_metric": "missing_psnr",
        "winner": winner,
        "selected_tensor_gap_to_best_missing_psnr_db": base_gap,
        "ranked_results": ranked_rows,
        "tensor_family_screening": {
            "ground_truth_used": bool(screening.get("ground_truth_used", False)),
            "selection_metric": screening.get("selection_metric"),
            "winner": selected_method,
            "results": screening_rows,
        },
    }
    if audio:
        reference["selection_metric"] = "missing_nmse"
        reference["metric_direction"] = "lower_is_better"
        reference.pop("selected_tensor_gap_to_best_missing_psnr_db")
        base_nmse = tensor_metrics.get("missing_nmse")
        best_nmse = ranked_rows[0]["metrics"].get("missing_nmse") if ranked_rows else None
        reference["selected_tensor_nmse_above_best"] = base_nmse - best_nmse if base_nmse is not None and best_nmse is not None else None
    return reference


BASE_CLASS_NAMES = {
    "matrix": "MatrixFactorization",
    "mode3": "Mode3Factorization",
    "cp": "CPDecomposition",
    "nonnegative_cp": "NonnegativeCPDecomposition",
    "tucker": "TuckerDecomposition",
    "btd": "BlockTermDecomposition",
    "tsvd": "TSVDDecomposition",
    "nonnegative_tucker": "NonnegativeTuckerDecomposition",
    "hierarchical_tucker": "HierarchicalTuckerDecomposition",
    "tt": "TensorTrainDecomposition",
    "tensor_ring": "TensorRingDecomposition",
}

BASE_SOURCE_FILES = {
    "matrix": "matrix_factorization.py",
    "mode3": "mode3_factorization.py",
    "cp": "cp.py",
    "nonnegative_cp": "nonnegative_cp.py",
    "tucker": "tucker.py",
    "btd": "block_term.py",
    "tsvd": "t_svd.py",
    "nonnegative_tucker": "nonnegative_tucker.py",
    "hierarchical_tucker": "hierarchical_tucker.py",
    "tt": "tensor_train.py",
    "tensor_ring": "tensor_ring.py",
}


@dataclass
class CandidateGenerationResult:
    proposal: CandidateProposal
    attempts: int
    raw_outputs: List[str]
    validation_errors: List[str]
    fallback_reason: Optional[str]
    prompt_version: str


def _tv_candidate_code(base_method: str, tv_weights: Optional[List[float]] = None) -> str:
    base_class = BASE_CLASS_NAMES[base_method]
    weights = tv_weights or [0.0, 0.0001, 0.0005, 0.001]
    return f'''import torch


class CandidateTensorInpaintingModel({base_class}):
    """Add an image-space TV prior while retaining the learned tensor factors."""

    def __init__(self, image_shape, initial_channel_mean, tv_weight=0.0001, **kwargs):
        super().__init__(
            image_shape=image_shape,
            initial_channel_mean=initial_channel_mean,
            **kwargs,
        )
        if tv_weight < 0.0:
            raise ValueError("tv_weight must be non-negative")
        self.tv_weight = float(tv_weight)

    def loss_terms(self, prediction, observed, train_mask):
        terms = super().loss_terms(prediction, observed, train_mask)
        vertical_tv = torch.abs(prediction[1:, ...] - prediction[:-1, ...]).mean()
        horizontal_tv = torch.abs(prediction[:, 1:, ...] - prediction[:, :-1, ...]).mean()
        terms["tv_regularization"] = self.tv_weight * (vertical_tv + horizontal_tv)
        return terms

    @classmethod
    def search_space(cls, image_shape):
        space = dict({base_class}.search_space(image_shape))
        space["tv_weight"] = {weights!r}
        return space
'''


def deterministic_candidate(
    base_method: str,
    previous_feedback: Optional[List[Dict[str, Any]]] = None,
) -> CandidateProposal:
    """Safe learning-project fallback used when no code LLM is configured."""

    previous_feedback = previous_feedback or []
    tv_weights = (
        [0.0, 0.00001, 0.00005, 0.0001]
        if previous_feedback
        else [0.0, 0.0001, 0.0005, 0.001]
    )
    return CandidateProposal(
        base_method=base_method,
        architecture_family="tensor_decomposition",
        mutation_goal=(
            "Improve spatial continuity inside the missing region without changing "
            "the selected tensor-decomposition framework."
        ),
        mutation_target="loss",
        idea=(
            "Add one bounded image-space total-variation term to the existing masked "
            "reconstruction objective."
        ),
        single_change="Add a tunable total-variation regularization term to loss_terms().",
        hypothesis=(
            "Adding a small differentiable total-variation penalty to the selected "
            "tensor decomposition may reduce the banding and abrupt spatial changes "
            "observed inside a contiguous missing region."
        ),
        proposed_changes=[
            "Add horizontal and vertical image-space total variation to loss_terms()."
        ],
        expected_effect=(
            "The candidate should favour spatially coherent reconstructions while "
            "retaining the global low-rank structure of the base method."
        ),
        risks=[
            "An excessive TV weight can oversmooth edges and textures.",
            "Observed-pixel validation may still be mismatched with a block hole.",
            "The extra forward regularity does not add semantic information.",
        ],
        training_budget={
            "max_steps": 2000,
            "validation_interval": 20,
            "early_stopping_patience": 20,
            "rationale": (
                "Use a longer shared optimization horizon while retaining validation-based "
                "checkpoint selection and early stopping."
            ),
        },
        search_space={"tv_weight": tv_weights},
        model_code=_tv_candidate_code(base_method, tv_weights=tv_weights),
        generation_mode="deterministic_template",
    )


def load_improver_context(base_run_dir: str) -> Dict[str, Any]:
    """Load only the evidence needed by the improver, never GT image contents."""

    run_dir = Path(base_run_dir)
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise ValueError("base run state does not exist: %s" % state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("stage") != "COMPLETED":
        raise ValueError("base run must be COMPLETED")
    base_method = state.get("selected_model")
    if base_method not in BASE_CLASS_NAMES:
        raise ValueError("base run has unsupported selected_model")

    models_dir = Path(__file__).resolve().parents[1] / "core/models"
    source_paths = {
        method: models_dir / filename
        for method, filename in BASE_SOURCE_FILES.items()
    }
    history_path = Path(state["artifacts"]["tensor_history"])
    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = history_payload.get("history", [])
    history_summary = summarize_baseline_history(history)
    base_metrics = state["results"]["tensor_metrics"]
    method_plan = state["results"].get("method_plan")
    visual_influenced_selection = False
    if isinstance(method_plan, dict):
        method_plan = dict(method_plan)
        evidence = method_plan.get("evidence", [])
        if isinstance(evidence, list):
            visual_influenced_selection = any(
                isinstance(item, dict)
                and item.get("source") == METHOD_SELECTION_VISUAL_SOURCE
                for item in evidence
            )
            method_plan["evidence"] = [
                item
                for item in evidence
                if not (
                    isinstance(item, dict)
                    and item.get("source") == METHOD_SELECTION_VISUAL_SOURCE
                )
            ]
        if visual_influenced_selection:
            method_plan["reason"] = (
                "The Day 4 selector chose this decomposition and bounded rank search "
                "space. Raw selection-stage visual reasoning is intentionally withheld "
                "from mutation context."
            )
    return {
        "base_run_id": state["run_id"],
        "base_method": base_method,
        "image_profile": llm_safe_image_profile(
            state["results"]["image_profile"]
        ),
        "method_plan": method_plan,
        "selection_visual_influence_withheld_from_mutation": (
            visual_influenced_selection
        ),
        "base_best_config": state["results"]["training"],
        "training_curve_summary": history_summary,
        "base_metrics": {
            key: value
            for key, value in base_metrics.items()
            if key in REFERENCE_METRIC_KEYS
            and (key != "lpips" or value is not None)
        },
        "algorithm_comparison_reference": {
            **_algorithm_comparison_reference(state),
            "historical_algorithms": state.get("config", {}).get("historical_algorithm_reference") or [],
        },
        "model_interface_source": (
            Path(__file__).resolve().parents[1] / "core/models/base.py"
        ).read_text(encoding="utf-8"),
        "base_model_source": source_paths[base_method].read_text(encoding="utf-8"),
        "base_class_name": BASE_CLASS_NAMES[base_method],
        "previous_failure_feedback": [],
    }


class CandidateGenerator:
    """Generate one schema-valid proposal with one optional repair attempt."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        require_valid_llm_output: bool = False,
    ) -> None:
        self.llm = llm
        self.require_valid_llm_output = bool(require_valid_llm_output)

    @staticmethod
    def _messages(context: Dict[str, Any]) -> List[Dict[str, str]]:
        previous_feedback = _evolution_context_without_full_psnr(
            context.get("previous_failure_feedback") or []
        )
        previous_candidate = context.get("previous_candidate")
        validator_repair = bool(previous_feedback and previous_candidate)
        experiment_revision = bool(previous_feedback and not previous_candidate)
        base_context = _evolution_context_without_full_psnr(
            {
                key: value
                for key, value in context.items()
                if key
                not in {
                    "previous_failure_feedback",
                    "previous_candidate",
                    "comparison",
                }
            }
        )
        prompt_payload = {
            "task": (
                (
                    "Repair the previous candidate using the validator feedback and return "
                    "a complete replacement CandidateProposal JSON."
                )
                if validator_repair
                else (
                    "Revise the candidate hypothesis using the fixed Judge's experiment "
                    "feedback and return a complete replacement CandidateProposal JSON."
                )
                if experiment_revision
                else (
                    "Propose one testable atomic or bounded combination mutation for the "
                    "current tensor-based inpainting framework and return the complete "
                    "CandidateProposal JSON."
                )
            ),
            "context": base_context,
            "repair_context": (
                {
                    "validator_feedback": previous_feedback,
                    "previous_candidate": previous_candidate,
                    "rule": (
                        "Fix every validator error without weakening or bypassing the "
                        "validator. Preserve the research hypothesis unless the feedback "
                        "shows it cannot be implemented by the allowed model interface."
                    ),
                }
                if validator_repair
                else None
            ),
            "experiment_feedback": previous_feedback if experiment_revision else None,
            "response_style": {
                "instruction": (
                    "Use concise, information-dense wording. Do not repeat the image "
                    "profile, evidence, motivation, or implementation details across "
                    "multiple explanatory fields. Put executable detail in model_code."
                ),
                "maximum_characters": {
                    **EXPLANATORY_TEXT_LIMITS,
                    "training_budget.rationale": TRAINING_BUDGET_TEXT_LIMITS[
                        "rationale"
                    ],
                },
                "priority": (
                    "A complete, executable model_code is more important than verbose "
                    "prose. Stay comfortably below every maximum."
                ),
            },
            "evolution_protocol": {
                "required_reasoning_order": [
                    "identify the dominant observable pain_point",
                    "state the core_difficulty while separating evidence from hypothesis",
                    "simplify the difficulty into the smallest testable subproblem",
                    "state one mutation_goal for that simplified problem",
                    "propose the minimum sufficient idea",
                    "assign each component one operation: add, modify, remove, or retain",
                    "choose atomic mode by default, or combination mode only for a concrete interaction hypothesis",
                ],
                "diagnose_before_idea_rule": (
                    "Do not jump directly from a low score to a fashionable architecture. "
                    "First identify where the incumbent fails, why the framework may find it "
                    "difficult, and how that difficulty can be reduced to a small falsifiable "
                    "problem. Curve and visual signals are evidence, never proof of cause."
                ),
                "mutation_scope_rule": (
                    "Atomic mode is the default and contains exactly component A. Combination "
                    "mode may contain two or three independently switchable components A/B/C, "
                    "but only when a concrete interaction_hypothesis explains why the components "
                    "may be complementary. proposed_changes has one entry per component. "
                    "single_change is retained as a concise overall mutation summary."
                ),
                "module_operation_rule": (
                    "Every component declares operation add, modify, remove, or retain. A remove "
                    "operation is a first-class mutation, not a zero-weight workaround: identify "
                    "affected_modules, cite concrete ablation or repeated-failure evidence, and set "
                    "removal_kind to parameterized_module, parameter_free_path, or loss_term. Use "
                    "retain only to make an interaction plan explicit; at least one component must "
                    "add, modify, or remove behavior. Prefer removing a negative or redundant "
                    "incumbent mechanism over stacking another compensating module on top."
                ),
                "ablation_rule": (
                    "For combination mode, implement every component behind the exact boolean "
                    "constructor switch enable_component_a, enable_component_b, and optionally "
                    "enable_component_c. Declare each switch in search_space with [false, true]. "
                    "The system, not the LLM, constructs the complete 2^N ablation: Base, every "
                    "single component, every pair, and the full combination. Never omit an arm. "
                    "A true switch applies the declared operation. Therefore for remove, false "
                    "retains the incumbent mechanism and true omits it from construction and the "
                    "forward/loss path; multiplying an active branch by zero is only logical disable, "
                    "not structural removal."
                ),
                "incumbent_rule": (
                    "When incumbent_candidate is supplied, return a complete replacement "
                    "implementation based on its model_code and active_hyperparameters. Preserve "
                    "all active incumbent behavior except the explicitly declared add/modify/remove "
                    "operations. Do not resurrect components disabled by active_ablation_arm unless "
                    "the new hypothesis explicitly modifies or restores them."
                ),
                "knowledge_rule": (
                    "Use only the current run's (framework, goal, method, result) "
                    "practice tuples and the explicit algorithm comparison evidence. "
                    "Never treat another run's raw practice log as context. Do not repeat a failed idea unless the new "
                    "idea explicitly addresses its documented applicability or failure condition."
                ),
                "algorithm_comparison_rule": (
                    "Use algorithm_comparison_reference as an explicit directional benchmark. "
                    "In recovery runs, whole_modality_algorithms contains fixed baselines, "
                    "historical champions, and the Day 4 selected incumbent evaluated before "
                    "candidate generation on every same-type sample; "
                    "latest_evolution_round contains the current incumbent/candidate scores and "
                    "per-sample training-curve digests. Compare cohort means, hard samples, "
                    "optimization plateaus, instability, and reference gaps before proposing "
                    "the next change; explain which sample-level evidence motivates it. "
                    "Identify which reference mechanism appears to close the largest quality gap, "
                    "then propose the smallest testable change that could transfer that advantage "
                    "into the locked tensor framework. Aggregate reference scores are development "
                    "feedback, not proof of mechanism or an untouched test result. Never request or "
                    "infer raw hidden-region ground-truth values. The deterministic "
                    "whole-modality Judge, when enabled, decides promotion."
                ),
                "attribution_rule": (
                    "Do not claim that a component caused an improvement before ablation. Later "
                    "rounds must use the recorded per-component and interaction evidence to "
                    "distinguish dominant, additive, synergistic, antagonistic, and ineffective "
                    "combinations."
                ),
                "pruning_audit_rule": (
                    "The controller compares matched ablation arms by parameter count. A removed "
                    "parameterized_module is structurally confirmed only when enabling its remove "
                    "operation reduces parameters. Parameter-free paths and loss terms are reported "
                    "separately. Never describe a logical switch-off as confirmed structural pruning."
                ),
                "baseline_diagnostics_rule": (
                    "Treat training_curve_summary as observations about the current "
                    "framework's training and validation curves. Use its signals to form "
                    "a testable goal, but do not mistake a curve pattern for a proven cause."
                ),
                "visual_feedback_rule": (
                    "When a later-round incumbent-versus-candidate visual_assessment is "
                    "available, use its localized blur, over-smoothing, detail, "
                    "edge, texture, color, artifact, and poorly recovered-region observations. "
                    "Use mutation_guidance as evidence, not as an instruction to bundle changes. "
                    "The deterministic numerical Judge remains the acceptance authority."
                ),
                "psnr_feedback_rule": (
                    "Use missing-region PSNR, not full-image PSNR, as the PSNR signal for "
                    "mutation reasoning. Full-image PSNR is report-only because restored "
                    "observations can dilute errors in the missing region. Treat historical "
                    "experience that says only 'PSNR' as advisory; the current round's "
                    "missing-region PSNR has priority."
                ),
                "optimization_objective_rule": (
                    "Optimize one primary objective: improve candidate Missing-region PSNR "
                    "relative to the current incumbent. Composite SSIM is only a guardrail "
                    "with the fixed Judge tolerance, not a second objective that must improve. "
                    "When LPIPS was enabled and successfully computed, inspect its incumbent "
                    "value, candidate value, and signed delta as a secondary perceptual diagnostic; "
                    "lower LPIPS is better and a negative candidate-minus-incumbent delta is an "
                    "improvement. Use it to diagnose perceptual tradeoffs, but do not change the "
                    "fixed promotion gate or make LPIPS the primary objective. "
                    "Use interpolation, SIREN, the selected tensor-family screening result, and other exposed references "
                    "to diagnose concrete performance gaps and promising inductive biases. Do not "
                    "copy a reference blindly or let it replace the current-incumbent promotion "
                    "target. Do not construct a Pareto front or optimize no-reference metrics. "
                    "Historical experience must not override the fixed current-round Judge."
                ),
            },
            "allowed_changes": ALLOWED_CHANGES,
            "forbidden_changes": FORBIDDEN_CHANGES,
            "architecture_guidance": {
                "allowed_families": [
                    "tensor_decomposition",
                    "coordinate_mlp",
                    "convolutional_decoder",
                    "transformer",
                    "hybrid",
                ],
                "continuous_factor_examples": [
                    (
                        "For matrix factorization, generate the row factor or the "
                        "width-channel factor from normalized coordinates using an MLP."
                    ),
                    (
                        "For A-mode3-E/CP/Nonnegative CP/Tucker/BTD/t-SVD/Nonnegative Tucker/"
                        "Hierarchical Tucker/TT/Tensor Ring, replace one or more spatial "
                        "factor/core tables with coordinate MLP outputs and retain the "
                        "corresponding contraction."
                    ),
                ],
                "deep_examples": [
                    "learned low-resolution feature grid followed by Conv2d upsampling",
                    "residual convolutional refinement on top of a decomposition",
                    "Fourier coordinate encoding followed by a residual MLP",
                    "memory-bounded self-attention over axial, windowed, patch, channel, temporal, or latent tokens with a residual output head",
                    "cross-attention from tensor-decomposition factors or coordinates as queries to compact convolutional, spectral, channel, or temporal feature tokens",
                    "parallel multi-scale dilated Conv2d/Conv3d branches with bounded dilation rates, explicit padding, feature fusion, and an optional residual gate",
                    "hybrid local convolution plus efficient self/cross-attention for complementary short- and long-range structure",
                ],
                "attention_guidance": [
                    "Self-attention must operate on bounded axial, windowed, patch, channel, temporal, or latent-token sequences; never allocate full H*W by H*W attention.",
                    "Cross-attention must name the query source and key/value source, keep their dimensions explicit, and explain which missing-region dependency it is intended to transfer.",
                    "Prefer a residual or gated attention contribution initialized conservatively so the selected tensor contraction remains a stable starting point.",
                ],
                "dilated_convolution_guidance": [
                    "Use two or more bounded dilation scales only when the observed failure suggests missing local-to-mid-range context.",
                    "Preserve spatial size with explicit padding and fuse branches by concatenation plus projection, summation, or a bounded learned gate.",
                    "Include a standard local branch or anti-gridding fusion path so sparse dilation sampling does not create checkerboard or gridding artifacts.",
                    "For video tensors, state whether dilation is spatial-only or spatiotemporal and keep memory practical for arbitrary H, W, T, and C.",
                ],
                "resource_rules": [
                    "All modules are initialized from scratch; the base masked data term and shared Trainer remain fixed, while a proposed loss component may add a differentiable term through loss_terms().",
                    "Tensor axes are semantic: audio uses [time_frame,sample_in_frame,channel], MSI uses bands, and video uses time. For entirely missing rows/bands/frames, an independent unconstrained factor has no observed supervision; consider cross-slice coupling or differentiable priors rather than claiming rank alone identifies hidden slices.",
                    (
                        "The implementation must support arbitrary H and W and both "
                        "[H,W,C] and [H,W,T,C] image_shape values."
                    ),
                    "Keep parameter count and activation memory practical for original-resolution images.",
                    "Never construct an [H*W, H*W] attention matrix.",
                    "Multi-scale structure is allowed, but candidate-specific staged or coarse-to-fine training loops are forbidden.",
                ],
                "search_space_rules": [
                    (
                        "A hybrid inheriting the selected decomposition may copy its "
                        "base search-space entries unchanged, or propose a bounded "
                        "architecture-specific range for the same rank names, and add "
                        "candidate-only entries. Same-name ranges that differ are tuned "
                        "inside each model's own bounded search, up to the same per-model "
                        "structure-trial limit."
                    ),
                    (
                        "A direct BaseTensorInpaintingModel architecture should expose "
                        "only its own constructor hyperparameters; do not include "
                        "irrelevant decomposition rank/init keys."
                    ),
                    "Keep the total Cartesian search space compact because tuning_trials is at most five.",
                ],
            },
            "training_budget_policy": {
                "instruction": (
                    "Choose max_steps, validation_interval, and early_stopping_patience "
                    "in training_budget based on architecture depth and the supplied "
                    "training curve. The first fair-evaluation round establishes a run-wide "
                    "protocol and Day 6 clamps it to the user's CLI ceiling. In later rounds, "
                    "run_wide_fair_training_contract overrides new requests so the already "
                    "trained incumbent can be reused without breaking fairness."
                ),
                "fairness": (
                    "The resolved per-trial budget, seed, observed split, optimizer, and "
                    "learning-rate search policy are identical across the entire evolution run. "
                    "An incumbent is tuned and finally fitted only once, then its persisted result "
                    "is reused in later rounds. Each new candidate "
                    "model independently searches up to the same structure-trial limit; "
                    "a smaller finite space may be exhausted in fewer trials."
                ),
                "recommended_ranges_for_non_smoke_runs": {
                    "max_steps": [1000, 3000],
                    "validation_interval": [10, 50],
                    "early_stopping_patience": [10, 40],
                },
            },
            "fixed_contract": {
                "class_name": "CandidateTensorInpaintingModel",
                "base_class_available": context["base_class_name"],
                "preloaded_symbols": [
                    "torch",
                    "BaseTensorInpaintingModel",
                    "MatrixFactorization",
                    "Mode3Factorization",
                    "CPDecomposition",
                    "NonnegativeCPDecomposition",
                    "TuckerDecomposition",
                    "BlockTermDecomposition",
                    "TSVDDecomposition",
                    "NonnegativeTuckerDecomposition",
                    "HierarchicalTuckerDecomposition",
                    "TensorTrainDecomposition",
                    "TensorRingDecomposition",
                ],
                "forward_output": (
                    "finite floating tensor matching image_shape: [H,W,C] for "
                    "color/MSI or [H,W,T,C] for video"
                ),
                "exact_method_signatures": {
                    "forward": "forward(self)",
                    "loss_terms": (
                        "loss_terms(self, prediction, observed, train_mask); "
                        "prediction, observed, and train_mask must accept positional arguments"
                    ),
                    "regularization_terms": "regularization_terms(self)",
                    "search_space": "search_space(cls, image_shape) decorated with @classmethod",
                },
                "inheritance_rules": [
                    (
                        "The candidate must remain based on the selected decomposition "
                        "framework. Inherit the selected decomposition class and preserve "
                        "its factorization/contraction; deep modules may only augment it."
                    ),
                    (
                        "Every base class owns channel_bias and image_shape. Never "
                        "register, assign, or replace either protected attribute."
                    ),
                    (
                        "Built-in models flatten every trailing dimension into a joint "
                        "feature mode and restore image_shape at the output. Direct "
                        "architectures must likewise use self.feature_count and "
                        "self._restore_shape; never hard-code three output channels."
                    ),
                    (
                        "Call the chosen superclass __init__ exactly once before adding "
                        "candidate modules. For direct BaseTensorInpaintingModel "
                        "inheritance call super().__init__(image_shape, initial_channel_mean)."
                    ),
                    (
                        "If loss_terms is overridden, preserve its exact positional "
                        "signature and begin with terms = super().loss_terms(prediction, "
                        "observed, train_mask). train_mask can be spatial [H,W] or "
                        "elementwise with prediction.shape; always use super().loss_terms "
                        "for the masked data loss and never assume a spatial-only mask."
                    ),
                ],
                "implementation_surface": [
                    "CandidateTensorInpaintingModel.__init__",
                    "CandidateTensorInpaintingModel.forward",
                    "CandidateTensorInpaintingModel.regularization_terms",
                    "CandidateTensorInpaintingModel.loss_terms",
                    "CandidateTensorInpaintingModel.search_space",
                ],
                "combination_switch_contract": {
                    "maximum_components": 3,
                    "switch_names": [
                        "enable_component_a",
                        "enable_component_b",
                        "enable_component_c"
                    ],
                    "rule": (
                        "Combination components must be separable using these constructor "
                        "booleans. Disabling all switches must preserve the supplied active "
                        "incumbent behavior. For remove operations, enabling the switch applies "
                        "the deletion."
                    ),
                },
                "import_rules": {
                    "allowed_modules": [
                        "torch",
                        "torch.nn",
                        "torch.nn.functional",
                        "math",
                    ],
                    "forbidden_examples": [
                        "from __future__ import annotations",
                        "from typing import ...",
                        "from base import ...",
                        "relative imports",
                    ],
                    "instruction": (
                        "Prefer only `import torch`. The approved base class is already "
                        "available by name. The supplied base-model source is reference "
                        "material only: do not copy its module header or imports."
                    ),
                },
                "trainer_editable": False,
                "optimizer": "fixed Adam owned by the shared Trainer",
                "ground_truth_available": False,
                "evaluation_code_editable": False,
            },
            "output_schema": CandidateProposal.model_json_schema(),
            "prompt_version": PROMPT_VERSION,
        }
        if not context.get("mutation_visual_assessment_enabled", False):
            prompt_payload["evolution_protocol"].pop("visual_feedback_rule", None)
        if "missing_nmse" in base_context.get("base_metrics", {}):
            protocol = prompt_payload["evolution_protocol"]
            protocol.pop("psnr_feedback_rule", None)
            protocol["optimization_objective_rule"] = (
                "Audio uses only missing original-waveform NMSE (lower is better). "
                "Reduce candidate NMSE relative to the current incumbent under the "
                "frozen protocol and NMSE reduction threshold. Do not compute, invent "
                "or optimize PSNR, SSIM, LPIPS or image quality scores for audio. "
                "Use interpolation, tensor, SIREN and historical NMSE references "
                "to identify gaps; only the deterministic Judge may promote."
            )
        return [
            {
                "role": "system",
                "content": (
                    "You design per-image PyTorch inpainting models under a strict research "
                    "contract. Return one JSON object only. You may use continuous factor "
                    "MLPs, convolutions, efficient Transformers, residual connections, or "
                    "tensor/deep hybrids. Do not perform I/O, networking, subprocess or "
                    "dynamic code execution, use external data, or change evaluation."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            },
        ]

    @staticmethod
    def _content(response: Any) -> str:
        if isinstance(response, str):
            content = response
        else:
            content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise ValueError("LLM response does not contain string content")
        if not content.strip():
            raise ValueError("LLM response content is empty")
        return content

    @staticmethod
    def _validate_framework_lock(proposal: CandidateProposal) -> None:
        """Require every mutation to retain the selected decomposition superclass."""

        expected_base = BASE_CLASS_NAMES[proposal.base_method]
        tree = ast.parse(proposal.model_code)
        candidates = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CandidateTensorInpaintingModel"
        ]
        if len(candidates) != 1:
            raise ValueError("model_code must define exactly one candidate class")
        base_names = {
            base.id for base in candidates[0].bases if isinstance(base, ast.Name)
        }
        if expected_base not in base_names:
            raise ValueError(
                "candidate must inherit %s to preserve the selected tensor framework"
                % expected_base
            )

    def generate(self, context: Dict[str, Any]) -> CandidateGenerationResult:
        messages = self._messages(context)
        raw_outputs = []
        errors = []
        attempt_count = 0
        if self.llm is not None:
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
                                "The previous LLM attempt failed: %s. Retry once and "
                                "return only a complete JSON object."
                                % errors[-1]
                            ),
                        }
                    )
                try:
                    attempt_count += 1
                    raw = self._content(
                        self.llm.invoke(current_messages, temperature=0.0)
                    )
                    raw_outputs.append(raw)
                    proposal_payload = _normalize_candidate_payload(
                        _json_from_text(raw)
                    )
                    proposal = CandidateProposal.model_validate(proposal_payload)
                    if proposal.base_method != context["base_method"]:
                        raise ValueError("proposal base_method differs from selected model")
                    self._validate_framework_lock(proposal)
                    proposal.generation_mode = (
                        "llm" if attempt == 0 else "llm_repaired"
                    )
                    return CandidateGenerationResult(
                        proposal=proposal,
                        attempts=attempt + 1,
                        raw_outputs=raw_outputs,
                        validation_errors=errors,
                        fallback_reason=None,
                        prompt_version=PROMPT_VERSION,
                    )
                except Exception as error:
                    errors.append("%s: %s" % (type(error).__name__, str(error)))

        if self.require_valid_llm_output and self.llm is not None:
            raise RuntimeError(
                "required LLM candidate generation failed after %d attempts: %s"
                % (attempt_count, " | ".join(errors))
            )

        fallback_reason = (
            "LLM is not configured"
            if self.llm is None
            else "LLM proposal remained invalid after one repair attempt"
        )
        fallback = deterministic_candidate(
            context["base_method"],
            previous_feedback=context.get("previous_failure_feedback"),
        )
        self._validate_framework_lock(fallback)
        return CandidateGenerationResult(
            proposal=fallback,
            attempts=attempt_count,
            raw_outputs=raw_outputs,
            validation_errors=errors,
            fallback_reason=fallback_reason,
            prompt_version=PROMPT_VERSION,
        )

    def extract_experience(self, practice: Dict[str, Any]) -> Dict[str, Any]:
        """Distill one simple cross-round lesson from a complete practice tuple."""

        required = {"framework", "goal", "method", "result"}
        missing = required - set(practice)
        if missing:
            raise ValueError(
                "practice tuple is missing: %s" % ", ".join(sorted(missing))
            )
        compact_evidence = {
            "framework": practice["framework"],
            "goal": practice["goal"],
            "method": practice["method"],
            "result": dict(practice["result"]),
        }
        result = practice["result"]
        visual = result.get("visual_assessment") or {}
        if visual.get("status") == "completed":
            assessment = visual.get("assessment") or {}
            candidate_observation = assessment.get("candidate") or {}
            compact_evidence["result"]["visual_assessment"] = {
                "comparison": assessment.get("comparison"),
                "candidate_improvements": assessment.get("candidate_improvements", []),
                "candidate_regressions": assessment.get("candidate_regressions", []),
                "poorly_recovered_regions": candidate_observation.get(
                    "poorly_recovered_regions", []
                ),
                "mutation_guidance": assessment.get("mutation_guidance", []),
                "confidence": assessment.get("confidence"),
            }
        if self.llm is not None:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You distill one compact, cross-run algorithm-design principle from "
                        "a (framework, goal, method, result) practice tuple. Return one JSON object "
                        "with exactly two fields: experience and confidence. The experience "
                        "must say under what kind of framework the method tends to help or fail, "
                        "and abstract its effect on the task's declared primary metric; audio "
                        "uses only missing original-waveform NMSE (lower is better). Include "
                        "perceptual or structural effects only when actually evaluated, "
                        "using ablation evidence to distinguish individual component contribution "
                        "from additive, synergistic, antagonistic, or ineffective interaction, "
                        "and distinguish confirmed parameterized pruning from a logical switch-off, "
                        "including concrete visual side effects when visual evidence is present, "
                        "into one or two generally reusable sentences. Do not include workflow "
                        "IDs, round numbers, candidate IDs, file paths, condition lists, evidence "
                        "sections, or a next-round task. Write the experience in concise Chinese."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": (
                                "Infer a concise cross-round lesson from "
                                "(framework, goal, method, result). "
                                "Describe effects on the declared evaluation metric: audio uses "
                                "missing original-waveform NMSE only (lower is better); other types "
                                "use their image/tensor metrics. Do not invent unavailable scores. When "
                                "ablation is present, state which component or interaction mattered; "
                                "do not attribute the full result to every component. When removal "
                                "is present, state whether pruning was structurally confirmed."
                            ),
                            "practice_tuple": compact_evidence,
                            "output_schema": ExperienceExtraction.model_json_schema(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ]
            try:
                raw = self._content(self.llm.invoke(messages, temperature=0.0))
                return ExperienceExtraction.model_validate(
                    _json_from_text(raw)
                ).model_dump()
            except Exception:
                pass

        accepted = bool(result.get("accepted"))
        deltas = result.get("deltas") or {}
        psnr_delta = deltas.get("missing_psnr_db")
        ssim_delta = deltas.get("composite_ssim")
        lpips_delta = deltas.get("lpips")

        def directional_effect(value: Any, metric: str) -> str:
            if value is None:
                return "%s影响尚不明确" % metric
            numeric = float(value)
            tolerance = 0.05 if metric.endswith("PSNR") else 0.001
            if numeric > tolerance:
                return "有助于提高%s" % metric
            if numeric < -tolerance:
                return "可能降低%s" % metric
            return "对%s影响较小" % metric

        method_record = practice["method"]
        method = method_record.get("idea") or method_record["single_change"]
        ablation = result.get("ablation") or {}
        attribution = ablation.get("attribution") or {}
        attribution_text = ""
        if attribution.get("classification") not in {None, "unavailable"}:
            attribution_text = "消融归因属于%s；" % attribution["classification"]
        removal_audit = result.get("removal_audit") or {}
        removal_text = ""
        if removal_audit.get("status") == "completed":
            confirmed = removal_audit.get(
                "parameterized_removal_net_reduction_confirmed"
            )
            removal_text = (
                "参数化剪枝已由参数量下降确认；"
                if confirmed is True
                else "删除已执行，但尚不能由净参数量确认结构剪枝；"
            )
        if "missing_nmse_reduction" in deltas:
            reduction = deltas["missing_nmse_reduction"]
            effect = "尚无可定义的NMSE改善" if reduction is None else "降低缺失波形NMSE" if reduction > 0 else "未降低缺失波形NMSE"
            return ExperienceExtraction(
                experience="在音频波形恢复中，%s；%s%s实验证据：%s；该经验仅由NMSE评价，不包含图像结构或感知质量结论。" % (method.rstrip("。."), attribution_text, removal_text, effect),
                confidence="medium" if accepted else "low",
            ).model_dump()
        psnr_effect = directional_effect(psnr_delta, "缺失区域PSNR")
        structure_effect = directional_effect(ssim_delta, "图像结构相似度")
        perceptual_effect = ""
        if lpips_delta is not None:
            numeric_lpips_delta = float(lpips_delta)
            if numeric_lpips_delta < -0.001:
                perceptual_effect = "，并降低LPIPS、改善感知距离"
            elif numeric_lpips_delta > 0.001:
                perceptual_effect = "，但提高LPIPS、恶化感知距离"
            else:
                perceptual_effect = "，对LPIPS感知距离影响较小"
        conclusion = (
            "该机制可作为同类任务的有效优化方向，同时需关注它对图像结构的影响"
            if accepted
            else "该机制暂未形成可靠收益，不宜在类似任务中直接假定它能改善补全质量"
        )
        magnitude = max(
            abs(float(psnr_delta)) if psnr_delta is not None else 0.0,
            abs(float(ssim_delta)) if ssim_delta is not None else 0.0,
            abs(float(lpips_delta)) if lpips_delta is not None else 0.0,
        )
        return ExperienceExtraction(
            experience=(
                "在张量补全中，%s；%s%s实验表明这种方法%s，且%s%s。%s。"
                % (
                    method.rstrip("。."),
                    attribution_text,
                    removal_text,
                    psnr_effect,
                    structure_effect,
                    perceptual_effect,
                    conclusion,
                )
            ),
            confidence="medium" if accepted and magnitude >= 0.2 else "low",
        ).model_dump()
