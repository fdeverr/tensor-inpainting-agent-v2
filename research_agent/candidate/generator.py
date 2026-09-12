"""Prompt construction and constrained fallback candidate generation."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..method_selector import _json_from_text
from .schemas import CandidateProposal, ExperienceExtraction


PROMPT_VERSION = "iterative-single-change-v9"
ALLOWED_CHANGES = [
    "coordinate MLPs or Fourier-feature MLPs that continuously generate factors for any supported tensor baseline",
    "replace one or more learned factor tables with MLP-generated continuous factors",
    "convolutional decoders or convolutional residual refiners trained from scratch on the current image",
    "lightweight axial, patch, windowed, or latent-token self-attention/Transformer blocks",
    "residual connections, gated residual branches, normalization, and learned multi-scale features",
    "hybrids that combine tensor decomposition with MLP, convolution, or efficient attention",
    "factor initialization, asymmetric rank, smoothness, total variation, and other differentiable priors",
    "a completely new per-image PyTorch parameterization that implements BaseTensorInpaintingModel",
]
FORBIDDEN_CHANGES = [
    "pretrained weights, external datasets, foundation models, or downloaded assets",
    "access to complete ground truth or hidden-region metrics during selection/training",
    "network, subprocess, shell, or external-program calls",
    "changes to evaluation code",
    "candidate-specific training loops, optimizers, schedulers, or coarse-to-fine training",
    "full-resolution global attention over all H*W pixels; use axial, windowed, patch, or latent attention",
    "architectures whose memory is quadratic in the number of image pixels",
]


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
            "max_steps": 1000,
            "validation_interval": 20,
            "early_stopping_patience": 40,
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
    history_summary = {
        "record_count": len(history),
        "first": history[0] if history else None,
        "last": history[-1] if history else None,
        "minimum_recorded_train_loss": (
            min(item["total_train_loss"] for item in history) if history else None
        ),
    }
    return {
        "base_run_id": state["run_id"],
        "base_method": base_method,
        "image_profile": state["results"]["image_profile"],
        "method_plan": state["results"].get("method_plan"),
        "base_best_config": state["results"]["training"],
        "training_curve_summary": history_summary,
        "base_metrics": state["results"]["tensor_metrics"],
        "interpolation_metrics": state["results"]["interpolation_metrics"],
        "comparison": state["results"]["comparison"],
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
        previous_feedback = context.get("previous_failure_feedback") or []
        previous_candidate = context.get("previous_candidate")
        validator_repair = bool(previous_feedback and previous_candidate)
        experiment_revision = bool(previous_feedback and not previous_candidate)
        base_context = {
            key: value
            for key, value in context.items()
            if key not in {"previous_failure_feedback", "previous_candidate"}
        }
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
                    "Propose exactly one testable deep or tensor-based inpainting "
                    "architecture and return the complete CandidateProposal JSON."
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
            "evolution_protocol": {
                "required_reasoning_order": [
                    "state one mutation_goal",
                    "state one corresponding idea",
                    "choose exactly one mutation_target: algorithm or loss",
                    "implement exactly one controlled change",
                ],
                "single_variable_rule": (
                    "Change exactly one algorithmic mechanism or one loss mechanism. "
                    "proposed_changes must contain exactly one item and single_change "
                    "must describe the same change. Do not bundle architecture and loss changes."
                ),
                "incumbent_rule": (
                    "When incumbent_candidate is supplied, return a complete replacement "
                    "implementation that preserves all incumbent behavior except the one "
                    "declared change."
                ),
                "knowledge_rule": (
                    "Use global reusable experience from previous runs together with only "
                    "the current run's practice trajectory. Never treat another run's raw "
                    "practice log as context. Do not repeat a failed idea unless the new "
                    "idea explicitly addresses its documented applicability or failure condition."
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
                    "axial/windowed/latent-token Transformer with a residual output head",
                ],
                "resource_rules": [
                    "All modules are initialized from scratch and optimized only through the fixed masked loss.",
                    (
                        "The implementation must support arbitrary H and W and both "
                        "[H,W,C] and [H,W,T,C] image_shape values."
                    ),
                    "Keep parameter count and activation memory practical for original-resolution images.",
                    "Never construct an [H*W, H*W] attention matrix.",
                ],
                "search_space_rules": [
                    (
                        "A hybrid inheriting the selected decomposition may copy its "
                        "base search-space entries unchanged, or propose a bounded "
                        "architecture-specific range for the same rank names, and add "
                        "candidate-only entries. Same-name ranges that differ are tuned "
                        "independently with the same trial count."
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
                    "training curve. Day 6 clamps max_steps to the user's CLI ceiling."
                ),
                "fairness": (
                    "The resolved budget, seed, observed split, optimizer, and learning "
                    "rate are identical for baseline and candidate within each round."
                ),
                "recommended_ranges_for_non_smoke_runs": {
                    "max_steps": [500, 2000],
                    "validation_interval": [10, 50],
                    "early_stopping_patience": [20, 80],
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
                        "observed, train_mask)."
                    ),
                ],
                "implementation_surface": [
                    "CandidateTensorInpaintingModel.__init__",
                    "CandidateTensorInpaintingModel.forward",
                    "CandidateTensorInpaintingModel.regularization_terms",
                    "CandidateTensorInpaintingModel.loss_terms",
                    "CandidateTensorInpaintingModel.search_space",
                ],
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
                    proposal = CandidateProposal.model_validate(_json_from_text(raw))
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

    def extract_experience(self, round_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Distill one cross-run reusable lesson with a stable offline fallback."""

        judgment = round_evidence["judgment"]
        compact_evidence = {
            "goal": round_evidence.get("mutation_goal"),
            "method": {
                "idea": round_evidence.get("idea"),
                "single_change": round_evidence.get("single_change"),
                "target": round_evidence.get("mutation_target"),
            },
            "result": {
                "accepted": bool(judgment["accepted"]),
                "decision": judgment.get("decision"),
                "missing_psnr_delta_db": judgment.get("psnr_delta"),
                "composite_ssim_delta": judgment.get("ssim_delta"),
            },
        }
        if self.llm is not None:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You distill one compact, cross-run algorithm-design principle from "
                        "a mutation goal, method, and measured result. Return one JSON object "
                        "with exactly two fields: experience and confidence. The experience "
                        "must abstract the method's directional effect on PSNR and image quality "
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
                                "Infer a concise general lesson from (goal, method, result). "
                                "Describe what the mechanism tends to do to reconstruction PSNR "
                                "and visual/structural quality without copying run metadata."
                            ),
                            "goal_method_result": compact_evidence,
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
                ).model_dump() | {"extraction_mode": "llm"}
            except Exception:
                pass

        accepted = bool(judgment["accepted"])
        psnr_delta = judgment.get("psnr_delta")
        ssim_delta = judgment.get("ssim_delta")

        def directional_effect(value: Any, metric: str) -> str:
            if value is None:
                return "%s影响尚不明确" % metric
            numeric = float(value)
            tolerance = 0.05 if metric == "PSNR" else 0.001
            if numeric > tolerance:
                return "有助于提高%s" % metric
            if numeric < -tolerance:
                return "可能降低%s" % metric
            return "对%s影响较小" % metric

        method = round_evidence.get("idea") or round_evidence["single_change"]
        psnr_effect = directional_effect(psnr_delta, "PSNR")
        structure_effect = directional_effect(ssim_delta, "图像结构相似度")
        conclusion = (
            "该机制可作为同类任务的有效优化方向，同时需关注它对图像结构的影响"
            if accepted
            else "该机制暂未形成可靠收益，不宜在类似任务中直接假定它能改善补全质量"
        )
        magnitude = max(
            abs(float(psnr_delta)) if psnr_delta is not None else 0.0,
            abs(float(ssim_delta)) if ssim_delta is not None else 0.0,
        )
        return ExperienceExtraction(
            experience=(
                "在张量补全中，%s；实验表明这种方法%s，且%s。%s。"
                % (method.rstrip("。."), psnr_effect, structure_effect, conclusion)
            ),
            confidence="medium" if accepted and magnitude >= 0.2 else "low",
        ).model_dump() | {"extraction_mode": "deterministic_fallback"}
