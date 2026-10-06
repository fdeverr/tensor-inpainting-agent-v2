"""Run-scoped practice memory and cross-run reusable evolution experience."""

from __future__ import annotations

import json
import hashlib
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_SEMANTICS: Dict[str, Dict[str, str]] = {
    "missing_nmse": {
        "name": "Missing-waveform NMSE", "unit": "unitless", "direction": "lower_is_better",
        "scope": "missing original-amplitude waveform samples, all channels, excluding padding",
        "physical_meaning": "Squared waveform reconstruction error divided by reference waveform energy; no mean subtraction.",
    },
    "full_psnr": {
        "name": "Full-image PSNR",
        "unit": "dB",
        "direction": "higher_is_better",
        "scope": "full completed tensor after observed samples are restored",
        "physical_meaning": (
            "Logarithmic inverse reconstruction error over the entire completed tensor; "
            "observed samples contribute zero error after restoration."
        ),
    },
    "missing_psnr": {
        "name": "Missing-region PSNR",
        "unit": "dB",
        "direction": "higher_is_better",
        "scope": "artificially hidden pixels across all trailing tensor features",
        "physical_meaning": (
            "Logarithmic inverse reconstruction error in the missing region; an increase "
            "means lower missing-region mean-squared error."
        ),
    },
    "composite_ssim": {
        "name": "Composite SSIM",
        "unit": "unitless",
        "direction": "higher_is_better",
        "scope": "full tensor after known pixels are restored from ground truth",
        "physical_meaning": (
            "Structural similarity of luminance, contrast, and local structure. Known "
            "pixels can dilute errors, so this is not a standardized masked SSIM."
        ),
    },
    "lpips": {
        "name": "LPIPS",
        "unit": "perceptual distance",
        "direction": "lower_is_better",
        "scope": "composite RGB image only",
        "physical_meaning": "Learned full-reference perceptual feature distance.",
    },
    "maniqa": {
        "name": "MANIQA",
        "unit": "model score",
        "direction": "higher_is_better",
        "scope": "composite RGB image only",
        "physical_meaning": "No-reference learned estimate of perceived image quality.",
    },
    "clip_iqa": {
        "name": "CLIP-IQA",
        "unit": "model score",
        "direction": "higher_is_better",
        "scope": "composite RGB image only",
        "physical_meaning": "No-reference semantic image-quality score derived from CLIP features.",
    },
    "musiq": {
        "name": "MUSIQ",
        "unit": "model score",
        "direction": "higher_is_better",
        "scope": "composite RGB image only",
        "physical_meaning": "No-reference multi-scale learned estimate of perceived quality.",
    },
}


def describe_metrics(
    metrics: Dict[str, Any],
    include_full_psnr: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Attach stable semantics to every metric value persisted in practice memory."""

    described: Dict[str, Dict[str, Any]] = {}
    for name, semantics in METRIC_SEMANTICS.items():
        if "missing_nmse" in metrics and name != "missing_nmse":
            continue
        if name == "missing_nmse" and name not in metrics:
            continue
        if name == "full_psnr" and not include_full_psnr:
            continue
        described[name] = {"value": metrics.get(name), **semantics}
    for name, value in metrics.items():
        if name == "full_psnr" and not include_full_psnr:
            continue
        if name not in described:
            described[name] = {
                "value": value,
                "name": name,
                "unit": "unspecified",
                "direction": "reported_only",
                "scope": "see evaluator output",
                "physical_meaning": "Evaluator-provided auxiliary metric.",
            }
    return described


def resolve_knowledge_root(candidate_root: str, configured_root: Optional[str]) -> Path:
    """Resolve the persistent root shared by independent workflow runs."""

    if configured_root:
        return Path(configured_root)
    return Path(candidate_root).parent / "evolution_knowledge"


def experience_conditions(base_state, cases=None):
    """Keep experimental applicability explicit; a representative is not a cohort."""
    profile = base_state["results"]["image_profile"]
    config = base_state["config"]
    rates = [case["actual_missing_rate"] for case in (cases or []) if "actual_missing_rate" in case]
    conditions = {
        "mask_type": "slices" if config["mask_type"] == "sildes" else config["mask_type"],
        "requested_missing_rate": config.get("missing_rate", profile.get("requested_missing_rate", profile["actual_missing_rate"])),
        "actual_missing_rate": profile["actual_missing_rate"],
        "evaluation_scope": "all_same_type_samples" if cases else "single_sample",
        "sample_count": len(cases) if cases else 1,
    }
    if rates:
        conditions["actual_missing_rate_range"] = [min(rates), max(rates)]
    return conditions


def global_experience_context(base_state, candidate_root, configured_root, cases=None):
    profile = (base_state.get("results") or {}).get("image_profile") or {}
    base_method = base_state.get("selected_model")
    try:
        if not base_method:
            raise ValueError("global experience requires an explicitly selected base method")
        store = GlobalExperienceStore(resolve_knowledge_root(candidate_root, configured_root),
            base_method, profile.get("data_type", "color_image"),
            experience_conditions(base_state, cases))
        return {"status": "completed", **store.context(limit=20)}
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("⚠️ 历史经验暂不可读，本轮继续使用当前实践与对照：%s" % error, flush=True)
        return {"status": "unavailable", "base_method": base_method,
                "data_type": profile.get("data_type", "color_image"), "record_count": 0,
                "retrieved_count": 0, "reusable_experience": [], "error": str(error)}


def compact_run_practices(practices):
    """Represent every round, excluding full curves, tensors and redundant trials."""
    evidence = []
    for practice in practices:
        if set(practice) != {"framework", "goal", "method", "result"}:
            raise ValueError("run summary requires complete practice tuples")
        result = practice["result"]
        compact = {key: result[key] for key in (
            "incumbent_metrics", "candidate_metrics", "deltas", "decision", "accepted",
            "incumbent_after", "metric_scope", "training_behavior", "removal_audit",
        ) if key in result}
        cohort = result.get("dataset_evaluation")
        if cohort:
            compact["dataset_evaluation"] = {key: cohort.get(key) for key in ("incumbent_summary", "candidate_summary")}
            for role in ("incumbent", "candidate"):
                compact["dataset_evaluation"][role] = [
                    {key: sample[key] for key in ("sample", "status", "metrics", "error") if key in sample}
                    for sample in cohort.get(role, [])]
            compact["dataset_evaluation"]["sample_failures"] = [
                {"sample": sample.get("sample"), "error": sample.get("error")}
                for role in ("incumbent", "candidate") for sample in cohort.get(role, []) if sample.get("status") != "completed"]
        ablation = result.get("ablation") or {}
        if ablation:
            compact["ablation_attribution"] = ablation.get("attribution", {"classification": "unavailable"})
        visual = result.get("visual_assessment") or {}
        if visual.get("status") == "completed":
            assessment = visual.get("assessment", {})
            compact["visual_assessment"] = {key: assessment.get(key) for key in
                ("comparison", "candidate_improvements", "candidate_regressions", "mutation_guidance", "confidence")}
        evidence.append({"framework": practice["framework"], "goal": practice["goal"],
                         "method": practice["method"], "result": compact})
    return evidence


def prepare_run_experience(generator, state, base_state, candidate_root, configured_root,
                           cases=None, source_run_id=None):
    """Summarize the persisted practice library exactly once after a complete run."""
    if state.get("stage") != "COMPLETED":
        raise ValueError("only completed evolution runs may produce global experience")
    path = Path(state["artifacts"]["run_practice"]["practice_jsonl"])
    source_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    practices = _parse_jsonl(source_bytes.decode("utf-8"))
    if not practices:
        return {"status": "skipped", "reason": "no completed practice tuples"}
    if len(practices) != len(state["rounds"]):
        raise ValueError("practice library does not cover every completed evolution round")
    extractor = getattr(generator, "extract_run_experience", None)
    if extractor is None:
        from .candidate.generator import CandidateGenerator
        extractor = CandidateGenerator(None).extract_run_experience
    outcome = dict(state["best_available"])
    if cases:
        # The representative's reconstruction metrics are not the cohort score.
        outcome.pop("metrics", None)
        outcome["whole_modality_summary"] = state.get("overall_comparison", {}).get("whole_modality_summary")
    summary = extractor(compact_run_practices(practices), outcome)
    from .candidate.schemas import ExperienceExtraction
    summary = ExperienceExtraction.model_validate(summary).model_dump()
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("practice library changed during run-level summarization")
    conditions = experience_conditions(base_state, cases)
    conditions.update({
        "source_run_id": source_run_id or state["workflow_id"],
        "summary_scope": "whole_run_practice", "round_count": len(practices),
        "accepted_round_count": sum(bool(p["result"].get("accepted")) for p in practices),
        "source_practice_jsonl": str(path.resolve()),
        "source_practice_sha256": source_sha256,
        "final_algorithm": state["best_available"]["algorithm"],
        "training_budget": practices[0]["framework"].get("conditions", {}).get("training_budget"),
        "evaluation_steps": state.get("config", {}).get("dataset_evaluation_steps") if cases else None,
        "metric_protocol": "missing_original_waveform_nmse" if base_state["results"]["image_profile"].get("data_type") == "audio" else "missing_region_psnr_and_composite_ssim",
    })
    profile = base_state["results"]["image_profile"]
    return {"status": "prepared", "experience": summary,
            "context_audit": getattr(generator, "last_summary_context_audit", {}),
            "root": str(resolve_knowledge_root(candidate_root, configured_root)),
            "base_method": base_state["selected_model"],
            "data_type": profile.get("data_type", "color_image"), "conditions": conditions}


def commit_run_experience(summary):
    """Idempotent commit; the summary must still match its original practice log."""
    if summary.get("status") == "skipped":
        return summary
    if summary.get("status") not in {"prepared", "committed"}:
        raise ValueError("run experience is not prepared")
    conditions = summary["conditions"]
    path = Path(conditions["source_practice_jsonl"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != conditions["source_practice_sha256"]:
        raise ValueError("practice library changed after run-level summarization")
    store = GlobalExperienceStore(summary["root"], summary["base_method"], summary["data_type"], conditions)
    artifacts = store.record(summary["experience"])
    return {**summary, "status": "committed", "documents": artifacts}


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return _parse_jsonl(path.read_text(encoding="utf-8"))


def _parse_jsonl(contents):
    records = []
    for line in contents.splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("knowledge JSONL records must be JSON objects")
            records.append(record)
    return records


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def _experience_lock(directory):
    """Serialize read/modify/replace across concurrent Linux/macOS runs."""
    import fcntl
    with (directory / ".experience.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class GlobalExperienceStore:
    """Persistent compact principles reused across independent runs."""

    def __init__(self, root: str | Path, base_method: str,
                 data_type: Optional[str] = None, conditions: Optional[Dict[str, Any]] = None) -> None:
        normalization = {"Image": "color_image", "MSI": "msi", "Video": "video"}
        data_type = normalization.get(data_type, data_type)
        aliases = {"color_image": "Image", "msi": "MSI", "video": "Video", "audio": "audio"}
        if data_type is not None and data_type not in aliases:
            raise ValueError("unsupported experience data_type")
        if not base_method or base_method in {".", ".."} or Path(base_method).name != base_method:
            raise ValueError("invalid base_method")
        self.conditions = conditions or {}
        self.data_type = data_type
        self.base_method = base_method
        self.directory = Path(root) / aliases[data_type] / base_method if data_type else Path(root) / base_method
        self.directory.mkdir(parents=True, exist_ok=True)
        self.experience_jsonl = self.directory / "reusable_experience.jsonl"
        self.experience_markdown = self.directory / "reusable_experience.md"
        with _experience_lock(self.directory):
            records = self._compact_records(_read_jsonl(self.experience_jsonl))
            if records or self.experience_jsonl.exists():
                _write_jsonl(self.experience_jsonl, records)
            self._write_markdown(records)

    @staticmethod
    def _compact_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize legacy lessons while retaining tagged applicability/evidence."""

        compact: List[Dict[str, Any]] = []
        seen = set()
        for record in records:
            lesson = record.get("experience") or record.get("reusable_principle")
            if not isinstance(lesson, str) or not lesson.strip():
                continue
            lesson = lesson.strip()
            confidence = record.get("confidence", "low")
            if confidence not in {"low", "medium", "high"}:
                confidence = "low"
            metadata = {key: record[key] for key in (
                "data_type", "base_method", "mask_type", "requested_missing_rate",
                "actual_missing_rate", "source_run_id", "summary_scope", "round_count",
                "accepted_round_count", "source_practice_jsonl", "source_practice_sha256",
                "final_algorithm", "evaluation_scope", "sample_count", "actual_missing_rate_range",
                "training_budget", "evaluation_steps", "metric_protocol",
            ) if key in record} if record.get("data_type") else {}
            if metadata:
                metadata["data_type"] = {"Image": "color_image", "MSI": "msi", "Video": "video"}.get(metadata["data_type"], metadata["data_type"])
                if metadata.get("mask_type") == "sildes":
                    metadata["mask_type"] = "slices"
            identity = (lesson, metadata.get("data_type"), metadata.get("mask_type"),
                        metadata.get("actual_missing_rate"), metadata.get("source_run_id"))
            if identity in seen:
                continue
            seen.add(identity)
            compact.append({"experience": lesson, "confidence": confidence, **metadata})
        return compact

    def _write_markdown(self, records: List[Dict[str, Any]]) -> None:
        lines = [
            "# %s 可复用算法变异经验" % self.directory.name,
            "",
            "本文件跨运行累积；有类型标注的经验仅适用于记录的数据类型、缺失模式和缺失率。旧版无标注经验不会自动混入分类库。",
            "",
        ]
        for record in records:
            lines.extend(
                [
                    "- 经验：%s" % record["experience"],
                    "  - 置信度：`%s`" % record["confidence"],
                ]
            )
            if record.get("data_type"):
                lines.append("  - 适用条件：`%s / %s / 缺失率 %s`；来源：`%s`" % (
                    record["data_type"], record.get("mask_type"), record.get("actual_missing_rate"),
                    record.get("source_run_id", "unknown")))
                if record.get("summary_scope") == "whole_run_practice":
                    lines.append("  - 运行级总结：%s 轮实践，%s 轮晋级；最终输出：`%s`；评测范围：`%s`（%s 个样本）。" % (
                        record.get("round_count"), record.get("accepted_round_count"), record.get("final_algorithm"),
                        record.get("evaluation_scope"), record.get("sample_count")))
                    lines.append("  - 原始实践：`%s`；校验值：`%s`。" % (
                        record.get("source_practice_jsonl"), record.get("source_practice_sha256")))
        self.experience_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def context(self, limit: int = 50) -> Dict[str, Any]:
        if limit < 1:
            raise ValueError("experience retrieval limit must be positive")
        records = _read_jsonl(self.experience_jsonl)
        eligible = [(index, record) for index, record in enumerate(records)
                    if not self.data_type or record.get("data_type") == self.data_type and record.get("base_method") == self.base_method]
        rate = self.conditions.get("requested_missing_rate", self.conditions.get("actual_missing_rate"))
        pattern = self.conditions.get("mask_type")
        def relevance(index_record):
            index, record = index_record
            historical_rate = record.get("requested_missing_rate", record.get("actual_missing_rate"))
            distance = abs(historical_rate - rate) if isinstance(historical_rate, (int, float)) and isinstance(rate, (int, float)) else math.inf
            return (record.get("mask_type") == pattern, -distance, index)
        selected = [dict(record) for _, record in sorted(eligible, key=relevance, reverse=True)[:limit]]
        if self.data_type:
            for record in selected:
                record.setdefault("summary_scope", "legacy_unspecified")
        return {
            "base_method": self.base_method,
            "data_type": self.data_type,
            "current_conditions": self.conditions,
            "record_count": len(records),
            "retrieved_count": len(selected),
            "retrieval_rule": "same modality/base method; matching mask type, nearest requested missing rate, newest first; advisory only",
            "reusable_experience": selected,
            "documents": {
                "experience_jsonl": str(self.experience_jsonl),
                "experience_markdown": str(self.experience_markdown),
            },
        }

    def record(
        self,
        experience: Dict[str, Any],
    ) -> Dict[str, str]:
        """Append one run-level lesson, retaining applicability and source evidence."""

        with _experience_lock(self.directory):
            return self._record_locked(experience)

    def _record_locked(self, experience):

        payload = dict(experience)
        if self.data_type:
            payload.update({"data_type": self.data_type, "base_method": self.base_method,
                            **self.conditions})
        record = self._compact_records([payload])
        if not record:
            raise ValueError("experience must contain a non-empty general lesson")
        new_record = record[0]
        records = self._compact_records(_read_jsonl(self.experience_jsonl))
        if new_record.get("summary_scope") == "whole_run_practice":
            previous = [record for record in records if record.get("summary_scope") == "whole_run_practice"
                        and record.get("source_run_id") == new_record.get("source_run_id")]
            if previous:
                if previous[0].get("source_practice_sha256") != new_record.get("source_practice_sha256"):
                    raise ValueError("run already summarized with different practice evidence")
                return {"experience_jsonl": str(self.experience_jsonl), "experience_markdown": str(self.experience_markdown)}
        if new_record not in records:
            records.append(new_record)
            _write_jsonl(self.experience_jsonl, records)
            self._write_markdown(records)
        return {
            "experience_jsonl": str(self.experience_jsonl),
            "experience_markdown": str(self.experience_markdown),
        }


class RunPracticeStore:
    """Run-local trajectory for later rounds and one end-of-run distillation."""

    def __init__(self, run_dir: str | Path, base_method: str, workflow_id: str) -> None:
        self.directory = Path(run_dir) / "knowledge"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.practice_jsonl = self.directory / "practice.jsonl"
        self.practice_markdown = self.directory / "practice.md"
        if not self.practice_markdown.exists():
            self.practice_markdown.write_text(
                "# 当前运行算法进化实践\n\n"
                "- Workflow：`%s`\n- 基础分解：`%s`\n\n"
                "每条实践严格保存为（当前框架与条件、目标、方法、结果）四元组。"
                "原始实践只服务于当前运行；完整运行结束后统一提炼一次全局经验，其他运行仅读取该摘要。\n"
                % (workflow_id, base_method),
                encoding="utf-8",
            )

    def context(self) -> Dict[str, Any]:
        records = _read_jsonl(self.practice_jsonl)
        return {
            "round_count": len(records),
            "current_run_practice": records,
            "documents": {
                "practice_jsonl": str(self.practice_jsonl),
                "practice_markdown": str(self.practice_markdown),
            },
        }

    def record(self, practice: Dict[str, Any]) -> Dict[str, str]:
        required = {"framework", "goal", "method", "result"}
        missing = required - set(practice)
        extras = set(practice) - required
        if missing or extras:
            raise ValueError(
                "practice must be exactly (framework, goal, method, result); "
                "missing=%s; extras=%s"
                % (sorted(missing), sorted(extras))
            )
        round_index = len(_read_jsonl(self.practice_jsonl)) + 1
        _append_jsonl(self.practice_jsonl, practice)
        framework = practice["framework"]
        method = practice["method"]
        result = practice["result"]
        signal_codes = [
            item.get("code", "unknown")
            for item in framework.get("training_diagnostics", {}).get("signals", [])
        ]
        with self.practice_markdown.open("a", encoding="utf-8") as stream:
            stream.write(
                "\n## 第 %s 轮\n\n"
                "- 当前框架：`%s`（基础分解：`%s`）\n"
                "- 条件：`%s`\n"
                "- 训练诊断：`%s`\n"
                "- 目标：%s\n- 方法：%s\n- 变异对象：`%s`\n- 唯一改动：%s\n"
                "- 结果：`%s`；是否接受：`%s`\n"
                % (
                    round_index,
                    framework.get("name", "unknown"),
                    framework.get("base_method", "unknown"),
                    json.dumps(
                        framework.get("conditions", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    ", ".join(signal_codes) if signal_codes else "none",
                    practice["goal"],
                    method["idea"],
                    method["target"],
                    method["single_change"],
                    result["decision"],
                    result["accepted"],
                )
            )
            for role in ("incumbent_metrics", "candidate_metrics"):
                stream.write("\n### %s\n\n" % role)
                for metric in result[role].values():
                    stream.write(
                        "- **%s**：`%s %s`，%s；范围：%s。%s\n"
                        % (
                            metric["name"],
                            metric["value"],
                            metric["unit"],
                            metric["direction"],
                            metric["scope"],
                            metric["physical_meaning"],
                        )
                    )
            visual = result.get("visual_assessment") or {
                "status": "skipped",
                "reason": "visual assessment was not recorded",
            }
            stream.write("\n### 多模态视觉观察\n\n- 状态：`%s`\n" % visual["status"])
            if visual["status"] == "completed":
                assessment = visual["assessment"]
                stream.write(
                    "- 可见图像内容：%s\n"
                    % assessment["visible_image_content"]
                )
                stream.write("- 候选对比结论：%s\n" % assessment["comparison"])
                poorly_recovered = assessment["candidate"].get(
                    "poorly_recovered_regions", []
                )
                stream.write(
                    "- 候选未良好恢复区域：%s\n"
                    % ("；".join(poorly_recovered) if poorly_recovered else "未观察到")
                )
                stream.write(
                    "- 下轮变异启示：%s\n"
                    % "；".join(assessment.get("mutation_guidance", []))
                )
                stream.write("- 视觉置信度：`%s`\n" % assessment["confidence"])
            else:
                stream.write("- 原因：%s\n" % visual.get("reason", "未提供"))
        return {
            "practice_jsonl": str(self.practice_jsonl),
            "practice_markdown": str(self.practice_markdown),
        }
