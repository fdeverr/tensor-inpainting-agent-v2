"""Strict schemas for an LLM-proposed tensor-model improvement."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TrainingBudgetProposal(BaseModel):
    """LLM-requested budget; Day 6 applies a user-controlled hard ceiling."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(ge=1, le=20_000)
    validation_interval: int = Field(ge=1, le=500)
    early_stopping_patience: int = Field(ge=1, le=500)
    rationale: str = Field(min_length=20, max_length=800)


class MutationComponent(BaseModel):
    """One independently switchable mechanism in an evolution proposal."""

    model_config = ConfigDict(extra="forbid")

    id: Literal["A", "B", "C"]
    target: Literal["algorithm", "loss"]
    operation: Literal["add", "modify", "remove", "retain"] = "add"
    change: str = Field(min_length=10, max_length=800)
    expected_role: str = Field(min_length=10, max_length=800)
    affected_modules: List[str] = Field(default_factory=list, max_length=8)
    evidence: Optional[str] = Field(default=None, max_length=1200)
    removal_kind: Optional[
        Literal["parameterized_module", "parameter_free_path", "loss_term"]
    ] = None
    implementation_switch: str = Field(
        pattern=r"^enable_component_[abc]$",
    )


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
    pain_point: str = Field(default="Current reconstruction quality needs improvement.", min_length=10, max_length=800)
    core_difficulty: str = Field(default="The limiting mechanism has not yet been isolated.", min_length=10, max_length=1000)
    simplified_problem: str = Field(default="Test one minimal mechanism under the fixed protocol.", min_length=10, max_length=800)
    mutation_goal: str = Field(min_length=20, max_length=800)
    mutation_mode: Literal["atomic", "combination"] = "atomic"
    mutation_target: Literal["algorithm", "loss", "mixed"]
    idea: str = Field(min_length=20, max_length=1200)
    single_change: str = Field(min_length=10, max_length=800)
    hypothesis: str = Field(min_length=30, max_length=1500)
    proposed_changes: List[str] = Field(min_length=1, max_length=3)
    components: List[MutationComponent] = Field(default_factory=list, max_length=3)
    interaction_hypothesis: Optional[str] = Field(default=None, max_length=1200)
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

    @model_validator(mode="after")
    def validate_mutation_design(self) -> "CandidateProposal":
        if not self.components:
            switch = "enable_component_a"
            self.components = [
                MutationComponent(
                    id="A",
                    target=(
                        self.mutation_target
                        if self.mutation_target in {"algorithm", "loss"}
                        else "algorithm"
                    ),
                    change=self.single_change,
                    expected_role=self.expected_effect,
                    implementation_switch=switch,
                )
            ]
        expected_ids = ["A", "B", "C"][: len(self.components)]
        if [component.id for component in self.components] != expected_ids:
            raise ValueError("components must be ordered, contiguous IDs A, B, C")
        expected_switches = [
            "enable_component_%s" % component_id.lower()
            for component_id in expected_ids
        ]
        if [component.implementation_switch for component in self.components] != expected_switches:
            raise ValueError("component switches must be enable_component_a/b/c")
        if len(self.proposed_changes) != len(self.components):
            raise ValueError("proposed_changes must contain one entry per component")
        if all(component.operation == "retain" for component in self.components):
            raise ValueError("at least one component must add, modify, or remove a mechanism")
        for component in self.components:
            if component.operation == "remove":
                if not component.affected_modules:
                    raise ValueError("remove operations must identify affected_modules")
                if not component.evidence or len(component.evidence) < 20:
                    raise ValueError("remove operations require ablation or failure evidence")
                if component.removal_kind is None:
                    raise ValueError("remove operations require removal_kind")
            elif component.removal_kind is not None:
                raise ValueError("removal_kind is valid only for remove operations")
        if self.mutation_mode == "atomic" and len(self.components) != 1:
            raise ValueError("atomic proposals must contain exactly one component")
        if self.mutation_mode == "combination":
            if len(self.components) not in {2, 3}:
                raise ValueError("combination proposals must contain two or three components")
            if not self.interaction_hypothesis or len(self.interaction_hypothesis) < 20:
                raise ValueError("combination proposals require an interaction_hypothesis")
            for component in self.components:
                options = self.search_space.get(component.implementation_switch)
                if options is None or set(options) != {False, True}:
                    raise ValueError(
                        "combination component switches must have search-space values [false, true]"
                    )
        targets = {component.target for component in self.components}
        expected_target = next(iter(targets)) if len(targets) == 1 else "mixed"
        if self.mutation_target != expected_target:
            raise ValueError("mutation_target must summarize the component targets")
        return self

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
    """One compact, cross-run reusable lesson (workflows summarize a whole run)."""

    model_config = ConfigDict(extra="forbid")

    experience: str = Field(min_length=10, max_length=800)
    confidence: Literal["low", "medium", "high"]
