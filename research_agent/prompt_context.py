"""Deterministic context compaction and conservative, tokenizer-free input guards."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from dataclasses import dataclass


class ContextBudgetError(ValueError):
    """Required evidence cannot fit; never silently cut executable model code."""


@dataclass(frozen=True)
class ContextBudget:
    context_tokens: int = 131072
    output_reserve_tokens: int = 16384

    def __post_init__(self):
        if self.context_tokens <= 0 or not 0 < self.output_reserve_tokens < self.context_tokens:
            raise ValueError("LLM context budget must exceed the positive output reserve")

    @classmethod
    def from_environment(cls):
        return cls(int(os.getenv("LLM_CONTEXT_TOKENS", "131072")),
                   int(os.getenv("LLM_OUTPUT_RESERVE_TOKENS", "16384")))

    @property
    def input_limit(self):
        return self.context_tokens - self.output_reserve_tokens


def input_token_upper_bound(messages):
    # One UTF-8 byte per token is deliberately pessimistic for byte-based text
    # tokenizers. This is a safety estimate, NOT measured provider token usage.
    return sum(len(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
               for message in messages) + 32 * len(messages) + 256


def invoke_with_output_budget(llm, messages, budget):
    parameters = inspect.signature(llm.invoke).parameters
    kwargs = {"temperature": 0.0}
    if "max_tokens" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        configured = getattr(llm, "max_tokens", None)
        kwargs["max_tokens"] = min(configured or budget.output_reserve_tokens, budget.output_reserve_tokens)
    return llm.invoke(messages, **kwargs)


def _slim_round(practice, index):
    result = practice.get("result", {})
    method = practice.get("method", {})
    cohort = result.get("dataset_evaluation") or {}
    return {"round": index, "framework": practice.get("framework", {}).get("name"),
            "goal": practice.get("goal"), "method": {key: method[key] for key in
                ("idea", "single_change", "target", "components") if key in method},
            "result": {key: result[key] for key in
                ("accepted", "decision", "incumbent_after", "incumbent_metrics", "candidate_metrics", "deltas") if key in result},
            "cohort_summary": {key: cohort[key] for key in ("incumbent_summary", "candidate_summary") if key in cohort},
            "sample_failures": [{"sample": item.get("sample"), "error": item.get("error")}
                                for role in ("incumbent", "candidate") for item in cohort.get(role, [])
                                if item.get("status") != "completed"],
            "training_behavior": result.get("training_behavior"),
            "ablation_attribution": (result.get("ablation") or {}).get("attribution"),
            "removal_audit": result.get("removal_audit")}


def _clean_idea(idea):
    return {key: value for key, value in (idea or {}).items() if key not in {
        "model_code", "raw_outputs", "previous_failure_feedback", "schema_validation_errors",
        "validation", "attempt_history"}}


def compact_evolution_context(context):
    work = copy.deepcopy(context)
    comparison = work.get("algorithm_comparison_reference") or {}
    whole = comparison.get("whole_modality_algorithms") or {}
    if whole.get("historical_champions"):
        comparison.pop("historical_algorithms", None)  # Same versions, not two reference panels.
    for item in whole.get("historical_champions", []) + comparison.get("historical_algorithms", []):
        item.pop("_source_archive", None)  # Filesystem lookup is controller-only.
        structure = item.get("structure_reference") or {}
        same = (structure.get("base_method") == work.get("base_method") and
                (structure.get("kind") == "candidate" or structure.get("kind") == "builtin"
                 and item.get("algorithm") == work.get("base_method")))
        if not same:
            structure.pop("source_bundle", None)
            structure.pop("forward_and_loss_implementations", None)
            structure["full_source_included"] = False
        structure["matches_current_tensor_framework"] = same
    incumbent = work.get("incumbent_candidate") or {}
    if incumbent:
        incumbent["idea"] = _clean_idea(incumbent.get("idea"))
    previous = work.get("previous_candidate")
    if isinstance(previous, dict):
        work["previous_candidate"] = {**_clean_idea(previous),
                                     **({"model_code": previous["model_code"]} if "model_code" in previous else {})}
    memory = work.get("evolution_memory") or {}
    practice = memory.get("current_run_practice") or {}
    records = practice.get("current_run_practice", []) if isinstance(practice, dict) else practice
    if records:
        latest = copy.deepcopy(records[-1])
        ablation = latest.get("result", {}).get("ablation")
        if ablation:
            latest["result"]["ablation"] = {key: ablation[key] for key in
                ("attribution", "pruning_audit", "selected_non_base_arm", "status") if key in ablation}
        memory["current_run_practice"] = {
            "round_count": len(records), "latest_round": latest,
            "earlier_rounds": [_slim_round(record, index + 1) for index, record in enumerate(records[:-1])],
            "policy": "latest round keeps all per-sample metrics and curve digests; earlier rounds keep changes, outcomes, diagnostics and failure evidence"}
        work.pop("previous_round_result", None)
        feedback = work.get("previous_failure_feedback") or []
        if feedback and isinstance(feedback[-1], dict):
            # Same cohort panel occurs in practice, Judge feedback and comparison.
            panel = latest.get("result", {}).get("dataset_evaluation")
            if panel and feedback[-1].get("dataset_evaluation") == panel:
                feedback[-1].pop("dataset_evaluation")
                feedback[-1]["dataset_evidence_ref"] = "evolution_memory.current_run_practice.latest_round.result.dataset_evaluation"
        if len(feedback) > 1 and all(isinstance(item, dict) for item in feedback):
            work["previous_failure_feedback"] = [feedback[-1], {
                "earlier_feedback_count": len(feedback) - 1,
                "evidence_ref": "evolution_memory.current_run_practice.earlier_rounds"}]
        recent = comparison.get("latest_evolution_round")
        if recent:
            comparison["latest_evolution_round"] = {key: recent[key] for key in
                ("round", "incumbent_before", "candidate", "incumbent_after", "decision") if key in recent}
            comparison["latest_evolution_round"]["evidence_ref"] = "evolution_memory.current_run_practice.latest_round"
    global_memory = memory.get("global_reusable_experience") or {}
    global_memory.pop("documents", None)
    for lesson in global_memory.get("reusable_experience", []):
        lesson.pop("source_practice_jsonl", None)
    return work


def _pool_sources(context):
    work = copy.deepcopy(context)
    pool = {}
    def reference(code):
        if not isinstance(code, str):
            return code
        identity = hashlib.sha256(code.encode("utf-8")).hexdigest()
        pool[identity] = code
        return {"code_ref": identity}
    for key in ("base_model_source", "model_interface_source"):
        if key in work:
            work[key] = reference(work[key])
    for key in ("incumbent_candidate", "previous_candidate"):
        if isinstance(work.get(key), dict) and "model_code" in work[key]:
            work[key]["model_code"] = reference(work[key]["model_code"])
    comparison = work.get("algorithm_comparison_reference") or {}
    historical = (comparison.get("whole_modality_algorithms") or {}).get("historical_champions", [])
    for item in historical + comparison.get("historical_algorithms", []):
        structure = item.get("structure_reference") or {}
        for source in structure.get("source_bundle", []):
            if "code" in source:
                source.update(reference(source.pop("code")))
        # Full files already contain the method excerpts. Avoid sending them twice.
        if structure.get("source_bundle"):
            structure.pop("forward_and_loss_implementations", None)
    if pool:
        work["code_sources"] = pool
        work["code_reference_rule"] = "Every code_ref resolves to the EXACT full source in code_sources by SHA256. Shared files occur once. Return actual executable model_code, never a reference."
    return work


def fit_evolution_messages(context, builder, budget=None, suffix=None):
    budget = budget or ContextBudget.from_environment()
    work = compact_evolution_context(context)
    actions = ["deduplicate reference panels, latest-round evidence and source files; compact earlier practices"]
    whole = (work.get("algorithm_comparison_reference") or {}).get("whole_modality_algorithms") or {}
    history = whole.get("historical_champions", [])
    ranked_sources = list((whole.get("historical_structure_policy") or {}).get("full_source_archive_ids", []))
    if not ranked_sources:
        ranked_sources = [item.get("archive_id") for item in history
                          if (item.get("structure_reference") or {}).get("full_source_included")]
    memory = work.get("evolution_memory") or {}
    global_lessons = (memory.get("global_reusable_experience") or {}).get("reusable_experience", [])
    def render():
        policy = whole.get("historical_structure_policy")
        if policy:
            available = {item.get("archive_id") for item in history
                         if (item.get("structure_reference") or {}).get("full_source_included")}
            policy["full_source_archive_ids"] = [identity for identity in ranked_sources if identity in available]
        if memory.get("global_reusable_experience"):
            memory["global_reusable_experience"]["retrieved_count"] = len(global_lessons)
        return builder(_pool_sources(work)) + list(suffix or [])
    before = input_token_upper_bound(builder(context) + list(suffix or []))
    messages = render()
    # Keep the highest-ranked historical source. Drop optional source packs first.
    detailed = ranked_sources
    for identity in reversed(detailed):
        if input_token_upper_bound(messages) <= budget.input_limit:
            break
        for item in history:
            if item.get("archive_id") == identity:
                structure = item.get("structure_reference") or {}
                structure.pop("source_bundle", None)
                structure["full_source_included"] = False
                structure["source_omitted_reason"] = "input budget; structure and measured results retained"
                actions.append("omit optional historical source: " + str(identity))
        messages = render()
    while global_lessons and input_token_upper_bound(messages) > budget.input_limit:
        global_lessons.pop()
        actions.append("omit lower-priority global lesson")
        messages = render()
    if input_token_upper_bound(messages) > budget.input_limit:
        for item in history:
            structure = item.get("structure_reference") or {}
            structure.pop("forward_and_loss_implementations", None)
            structure.pop("structure", None)
            structure["structure_omitted_reason"] = "input budget; design/configuration and measured scores retained"
        actions.append("compact historical structure detail")
        messages = render()
    if input_token_upper_bound(messages) > budget.input_limit:
        for item in history:
            for sample in item.get("samples", []):
                training = sample.get("training") or {}
                training.pop("curve_points", None)
                training.pop("curve_summary", None)
        actions.append("compact old historical curve detail; keep sample metrics and training facts")
        messages = render()
    practice = memory.get("current_run_practice") or {}
    earlier = practice.get("earlier_rounds", []) if isinstance(practice, dict) else []
    if earlier and input_token_upper_bound(messages) > budget.input_limit:
        for row in earlier:
            row.pop("training_behavior", None)
            row.pop("removal_audit", None)
            row["method"].pop("components", None)
        actions.append("compact earlier-round detail; keep every round's changes, scores and failures")
        messages = render()
    after = input_token_upper_bound(messages)
    audit = {"estimator": "conservative UTF-8 byte upper estimate plus framing; not provider usage",
             "context_tokens": budget.context_tokens, "output_reserve_tokens": budget.output_reserve_tokens,
             "input_limit": budget.input_limit, "before_compaction_upper_estimate": before,
             "final_input_upper_estimate": after, "fits": after <= budget.input_limit, "actions": actions}
    if not audit["fits"]:
        error = ContextBudgetError("Required champion code/latest per-sample feedback cannot fit the LLM input budget (%d > %d); increase the correctly configured model budget or reduce experiment size. No model code or current sample evidence was truncated." % (after, budget.input_limit))
        error.audit = audit
        raise error
    return messages, audit


def check_text_messages(messages, budget=None):
    budget = budget or ContextBudget.from_environment()
    size = input_token_upper_bound(messages)
    if size > budget.input_limit:
        error = ContextBudgetError("LLM text input exceeds conservative budget (%d > %d); request was not sent" % (size, budget.input_limit))
        error.audit = {"context_tokens": budget.context_tokens, "output_reserve_tokens": budget.output_reserve_tokens,
                       "input_limit": budget.input_limit, "final_input_upper_estimate": size, "fits": False}
        raise error
    return budget
