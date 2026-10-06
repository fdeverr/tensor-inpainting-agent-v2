"""Context limits must not silently remove executable code or current sample evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from research_agent.candidate.generator import CandidateGenerator, deterministic_candidate
from research_agent.prompt_context import (
    ContextBudget, ContextBudgetError, compact_evolution_context,
    fit_evolution_messages, input_token_upper_bound,
)
from research_agent import recovery_registry


def _builder(context):
    return [{"role": "user", "content": json.dumps(context, ensure_ascii=False)}]


def _history(identity, code, base="tucker", kind="candidate", algorithm=None):
    return {"archive_id": identity, "algorithm": algorithm or identity,
            "summary": {"complete": True, "mean_missing_psnr": 30},
            "structure_reference": {"base_method": base, "kind": kind,
                "full_source_included": True, "structure": ["outline"],
                "forward_and_loss_implementations": ["duplicate excerpt"],
                "source_bundle": [{"file": "model.py", "code": code}]}}


def _context():
    code = "class Champion:\n    pass\n" + "# exact source\n" * 60
    return {"base_method": "tucker", "base_class_name": "TuckerDecomposition",
        "base_model_source": code, "model_interface_source": "class SharedBase: pass",
        "incumbent_candidate": {"model_code": code, "idea": {"idea": "retain champion",
            "model_code": code, "raw_outputs": [json.dumps({"model_code": code})]}},
        "algorithm_comparison_reference": {"whole_modality_algorithms": {
            "fixed_baselines": [{"algorithm": "siren", "samples": [{"sample": "one", "metrics": {"missing_psnr": 25}}]}],
            "historical_champions": [_history("old", code)],
            "historical_structure_policy": {"full_source_archive_ids": ["old"]}}}}


def _practice(index):
    samples = [{"sample": "sample-%d" % sample, "status": "completed",
                "metrics": {"missing_psnr": 20 + index + sample},
                "training": {"best_step": index * 10, "curve_points": [{"step": 1, "loss": 1}, {"step": 2, "loss": .5}],
                             "curve_summary": "all sample facts retained"}}
               for sample in range(5)]
    return {"framework": {"name": "candidate-%d" % index}, "goal": "round goal",
        "method": {"idea": "round-%d" % index, "single_change": "one change"},
        "result": {"decision": "accept", "accepted": True, "deltas": {"psnr": 1},
                   "training_behavior": {"stable": True}, "dataset_evaluation": {
                       "candidate": samples, "incumbent": samples,
                       "candidate_summary": {"mean_missing_psnr": 20 + index}}}}


def test_full_code_is_pooled_exactly_once_and_source_context_is_not_mutated():
    context = _context()
    original = copy.deepcopy(context)
    messages, audit = fit_evolution_messages(context, _builder)
    payload = json.loads(messages[0]["content"])
    champion = context["incumbent_candidate"]["model_code"]
    identity = hashlib.sha256(champion.encode()).hexdigest()
    assert payload["code_sources"][identity] == champion
    assert list(payload["code_sources"].values()).count(champion) == 1
    assert payload["base_model_source"] == {"code_ref": identity}
    assert payload["incumbent_candidate"]["model_code"] == {"code_ref": identity}
    assert "raw_outputs" not in payload["incumbent_candidate"]["idea"]
    assert audit["final_input_upper_estimate"] < audit["before_compaction_upper_estimate"]
    assert context == original


def test_actual_llm_message_keeps_latest_sample_evidence_and_resolves_champion_code():
    context = _context()
    practice = _practice(1)
    context["evolution_memory"] = {"current_run_practice": [practice]}
    payload = json.loads(CandidateGenerator._messages(context)[1]["content"])
    actual = payload["context"]
    assert actual["evolution_memory"]["current_run_practice"]["latest_round"]["result"]["dataset_evaluation"] == practice["result"]["dataset_evaluation"]
    identity = actual["incumbent_candidate"]["model_code"]["code_ref"]
    assert actual["code_sources"][identity] == context["incumbent_candidate"]["model_code"]


def test_latest_round_all_samples_and_curves_survive_older_round_compaction():
    context = _context()
    practices = [_practice(index) for index in range(1, 8)]
    context["evolution_memory"] = {"current_run_practice": {"current_run_practice": practices}}
    panel = practices[-1]["result"]["dataset_evaluation"]
    context["previous_round_result"] = practices[-1]
    context["previous_failure_feedback"] = [{"round": 7, "dataset_evaluation": panel, "next_round_constraints": ["keep stable"]}]
    context["algorithm_comparison_reference"]["latest_evolution_round"] = {"round": 7, "dataset_evaluation": panel}
    original = copy.deepcopy(context)
    messages, _ = fit_evolution_messages(context, _builder)
    payload = json.loads(messages[0]["content"])
    memory = payload["evolution_memory"]["current_run_practice"]
    assert memory["latest_round"] == practices[-1]
    assert len(memory["latest_round"]["result"]["dataset_evaluation"]["candidate"]) == 5
    assert len(memory["earlier_rounds"]) == 6
    assert "dataset_evaluation" not in memory["earlier_rounds"][0]["result"]
    assert "previous_round_result" not in payload
    assert "dataset_evaluation" not in payload["previous_failure_feedback"][0]
    assert payload["previous_failure_feedback"][0]["next_round_constraints"] == ["keep stable"]
    assert payload["algorithm_comparison_reference"]["whole_modality_algorithms"]["fixed_baselines"] == original["algorithm_comparison_reference"]["whole_modality_algorithms"]["fixed_baselines"]
    assert context == original


@pytest.mark.parametrize("base,kind,algorithm", [("tt", "candidate", "tt-candidate"), ("tucker", "builtin", "siren"), ("tucker", "interpolation", "interpolation")])
def test_foreign_framework_source_does_not_reach_evolution_prompt(base, kind, algorithm):
    context = _context()
    context["algorithm_comparison_reference"]["whole_modality_algorithms"]["historical_champions"].append(
        _history("foreign", "foreign source must disappear", base, kind, algorithm))
    compact = compact_evolution_context(context)
    foreign = compact["algorithm_comparison_reference"]["whole_modality_algorithms"]["historical_champions"][-1]
    assert foreign["summary"]["mean_missing_psnr"] == 30
    assert "source_bundle" not in foreign["structure_reference"]
    assert "forward_and_loss_implementations" not in foreign["structure_reference"]
    assert not foreign["structure_reference"]["matches_current_tensor_framework"]


def test_source_resolver_ranks_only_selected_framework_not_other_high_scores(monkeypatch):
    records = []
    for index, (base, kind, algorithm, score) in enumerate([
        ("tt", "candidate", "best-tt", 50), ("tucker", "builtin", "siren", 49),
        ("tucker", "candidate", "best-tucker", 40), ("tucker", "builtin", "tucker", 30),
        ("tucker", "candidate", "older-tucker", 20), ("tucker", "candidate", "lowest-tucker", 10)]):
        item = _history(str(index), "source", base, kind, algorithm)
        item["summary"]["mean_missing_psnr"] = score
        item["_source_archive"] = {"base_method": base, "kind": kind, "algorithm": algorithm, "archive_id": str(index)}
        records.append(item)
    def archived(record, include_source):
        structure = _history(record["archive_id"], "exact frozen code", record["base_method"], record["kind"])["structure_reference"]
        structure["full_source_included"] = include_source
        if not include_source:
            structure.pop("source_bundle")
        return structure
    monkeypatch.setattr(recovery_registry, "champion_structure_reference", archived)
    original = copy.deepcopy(records)
    result = recovery_registry.historical_reference_for_framework({"historical_champions": records}, "tucker")
    assert result["historical_structure_policy"]["full_source_archive_ids"] == ["2", "3", "4"]
    assert all("_source_archive" not in item for item in result["historical_champions"])
    assert records == original


def test_optional_historical_sources_are_pruned_by_score_order_not_archive_order():
    context = _context()
    whole = context["algorithm_comparison_reference"]["whole_modality_algorithms"]
    whole["historical_champions"] = [_history("low", "# low\n" * 5000), _history("high", "class BestHistory: pass")]
    whole["historical_structure_policy"]["full_source_archive_ids"] = ["high", "low"]
    budget = ContextBudget(12000, 1000)
    messages, audit = fit_evolution_messages(context, _builder, budget)
    payload = json.loads(messages[0]["content"])
    policy = payload["algorithm_comparison_reference"]["whole_modality_algorithms"]["historical_structure_policy"]
    assert policy["full_source_archive_ids"] == ["high"]
    assert "omit optional historical source: low" in audit["actions"]
    assert input_token_upper_bound(messages) <= budget.input_limit
    assert context["incumbent_candidate"]["model_code"] in payload["code_sources"].values()


class RecordingLLM:
    max_tokens = 32768

    def __init__(self, responses=()):
        self.calls = []
        self.responses = iter(responses)

    def invoke(self, messages, temperature=0.0, max_tokens=None):
        self.calls.append((messages, max_tokens))
        return next(self.responses)


def test_mandatory_oversized_context_blocks_before_llm_without_truncation(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_TOKENS", "18000")
    monkeypatch.setenv("LLM_OUTPUT_RESERVE_TOKENS", "2000")
    context = _context()
    context["incumbent_candidate"]["model_code"] = "# complete champion\n" * 20000
    context["evolution_memory"] = {"current_run_practice": [_practice(1)]}
    original = copy.deepcopy(context)
    llm = RecordingLLM()
    generator = CandidateGenerator(llm)
    with pytest.raises(ContextBudgetError) as error:
        generator.generate(context)
    assert not llm.calls
    assert not error.value.audit["fits"]
    assert generator.last_context_audit[-1]["fits"] is False
    assert context == original


def test_generation_output_reserve_and_invalid_retry_text_are_bounded(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_TOKENS", "100000")
    monkeypatch.setenv("LLM_OUTPUT_RESERVE_TOKENS", "4096")
    response = json.dumps(deterministic_candidate("tucker").model_dump())
    llm = RecordingLLM(["not json " * 30000, response])
    result = CandidateGenerator(llm).generate(_context())
    assert result.proposal.generation_mode == "llm_repaired"
    assert len(llm.calls) == 2
    assert all(limit == 4096 for _, limit in llm.calls)
    assert len(llm.calls[-1][0][-2]["content"]) < 2200
    assert len(result.context_audit) == 2
    assert all(audit["fits"] for audit in result.context_audit)


def test_valid_json_repair_code_is_never_truncated_to_fit(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_TOKENS", "70000")
    monkeypatch.setenv("LLM_OUTPUT_RESERVE_TOKENS", "4096")
    proposal = deterministic_candidate("tucker").model_dump()
    proposal["base_method"] = "tt"  # Parseable but wrong framework: repair needed.
    proposal["model_code"] = "# exact repair candidate\n" * 20000
    llm = RecordingLLM([json.dumps(proposal)])
    with pytest.raises(ContextBudgetError):
        CandidateGenerator(llm).generate(_context())
    assert len(llm.calls) == 1  # No second API call with a truncated candidate.


def test_oversized_once_run_summary_uses_audited_fallback_without_api(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_TOKENS", "1000")
    monkeypatch.setenv("LLM_OUTPUT_RESERVE_TOKENS", "200")
    practices = [_practice(1), _practice(2)]
    practices[1]["result"]["accepted"] = False
    llm = RecordingLLM()
    generator = CandidateGenerator(llm)
    result = generator.extract_run_experience(practices, {"algorithm": "winner"})
    assert not llm.calls
    assert "round-1" in result["experience"] and "round-2" in result["experience"]
    assert generator.last_summary_context_audit["mode"] == "deterministic_fallback"
    assert not generator.last_summary_context_audit["llm_request_sent"]
    assert not generator.last_summary_context_audit["fits"]


@pytest.mark.parametrize("day", [5, 6])
def test_workflow_budget_failure_is_audited_without_safety_fallback(tmp_path, monkeypatch, day):
    from research_agent.workflow_day5 import Day5Workflow, Day5WorkflowConfig
    from research_agent.workflow_day6 import Day6Workflow, Day6WorkflowConfig
    monkeypatch.setenv("LLM_CONTEXT_TOKENS", "18000")
    monkeypatch.setenv("LLM_OUTPUT_RESERVE_TOKENS", "2000")
    context = _context()
    context.update({"base_run_id": "base", "image_profile": {"image_shape": [4, 4, 3]},
                    "method_plan": {}, "base_best_config": {}, "training_curve_summary": {}, "base_metrics": {}})
    context["incumbent_candidate"]["model_code"] = "# required champion\n" * 20000
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "state.json").write_text(json.dumps({"stage": "COMPLETED", "run_id": "base"}))
    llm = RecordingLLM()
    generator = CandidateGenerator(llm)
    if day == 5:
        monkeypatch.setattr("research_agent.workflow_day5.load_improver_context", lambda _: context)
        workflow = Day5Workflow(Day5WorkflowConfig(base_run_dir=str(base_dir), output_dir=str(tmp_path / "outputs"),
                                   candidate_root=str(tmp_path / "candidates"), llm_mode="off"), generator=generator)
        with pytest.raises(ContextBudgetError):
            workflow.run()
        assert workflow.state["stage"] == "FAILED"
        assert workflow.state["candidate_id"] is None
        assert workflow.state["attempt_history"] == []
    else:
        initial = tmp_path / "initial"
        initial.mkdir()
        (initial / "manifest.json").write_text("{}")
        workflow = Day6Workflow(Day6WorkflowConfig(base_run_dir=str(base_dir), initial_candidate_dir=str(initial),
                                   output_dir=str(tmp_path / "outputs"), candidate_root=str(tmp_path / "candidates"),
                                   llm_mode="off"), generator=generator)
        with pytest.raises(ContextBudgetError):
            workflow._make_next_candidate(context, 2)
        assert workflow.state["candidate_generation_attempts"] == []
    audit = json.loads(Path(workflow.state["artifacts"]["context_budget"]).read_text())
    assert not (audit["calls"][-1] if day == 5 else audit)["fits"]
    assert not llm.calls


@pytest.mark.parametrize("entry,required", [("run", ["--image", "input.mat"]),
    ("run_recovery", ["--dataset-root", "dataset"]), ("run_day4", ["--image", "input.mat"]),
    ("run_day5", ["--base-run-dir", "base"]), ("run_day6", ["--base-run-dir", "base", "--candidate-dir", "candidate"])])
def test_context_budget_flags_are_available_for_every_llm_entry_point(entry, required):
    import importlib
    parser = importlib.import_module("research_agent." + entry).build_parser()
    args = parser.parse_args(required + ["--llm-context-tokens", "32000", "--llm-output-reserve-tokens", "4000"])
    assert args.llm_context_tokens == 32000 and args.llm_output_reserve_tokens == 4000


@pytest.mark.parametrize("total,reserve", [(0, 1), (100, 0), (100, 100), (100, 101)])
def test_invalid_context_budgets_are_rejected(total, reserve):
    with pytest.raises(ValueError):
        ContextBudget(total, reserve)
