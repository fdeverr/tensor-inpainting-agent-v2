"""Strict schemas for an LLM-proposed tensor-model improvement."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrainingBudgetProposal(BaseModel):
    """LLM-requested budget; Day 6 applies a user-controlled hard ceiling."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(ge=1, le=20_000)
    validation_interval: int = Field(ge=1, le=500)
    early_stopping_patience: int = Field(ge=1, le=500)
    rationale: str = Field(min_length=20, max_length=800)


class CandidateProposal(BaseModel):
    """Research hypothesis and complete candidate source code."""

    model_config = ConfigDict(extra="forbid")

    base_method: Literal[
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
    architecture_family: Literal[
        "tensor_decomposition",
        "coordinate_mlp",
        "convolutional_decoder",
        "transformer",
        "hybrid",
    ]
    mutation_goal: str = Field(min_length=20, max_length=800)
    mutation_target: Literal["algorithm", "loss"]
    idea: str = Field(min_length=20, max_length=1200)
    single_change: str = Field(min_length=10, max_length=800)
    hypothesis: str = Field(min_length=30, max_length=1500)
    proposed_changes: List[str] = Field(min_length=1, max_length=1)
    expected_effect: str = Field(min_length=20, max_length=1000)
    risks: List[str] = Field(min_length=1, max_length=8)
    training_budget: TrainingBudgetProposal
    search_space: Dict[str, List[Any]]
    model_code: str = Field(min_length=100, max_length=50_000)
    generation_mode: Literal[
        "llm",
        "llm_repaired",
        "deterministic_template",
    ] = "llm"

    @field_validator("search_space")
    @classmethod
    def validate_search_space(
        cls,
        value: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        if not value:
            raise ValueError("search_space cannot be empty")
        if any(not isinstance(options, list) or not options for options in value.values()):
            raise ValueError("every search_space entry must be a non-empty list")
        return value


class ExperienceExtraction(BaseModel):
    """One compact, cross-run reusable lesson extracted from a round."""

    model_config = ConfigDict(extra="forbid")

    experience: str = Field(min_length=10, max_length=800)
    confidence: Literal["low", "medium", "high"]
