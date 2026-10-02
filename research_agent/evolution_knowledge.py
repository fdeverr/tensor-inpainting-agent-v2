"""Run-scoped practice memory and cross-run reusable evolution experience."""

from __future__ import annotations

import json
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


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


class GlobalExperienceStore:
    """Persistent compact principles reused across independent runs."""

    def __init__(self, root: str | Path, base_method: str,
                 data_type: Optional[str] = None, conditions: Optional[Dict[str, Any]] = None) -> None:
        aliases = {"color_image": "Image", "msi": "MSI", "video": "Video", "audio": "audio"}
        if data_type is not None and data_type not in aliases:
            raise ValueError("unsupported experience data_type")
        if Path(base_method).name != base_method:
            raise ValueError("invalid base_method")
        self.conditions = conditions or {}
        self.data_type = data_type
        self.base_method = base_method
        self.directory = Path(root) / aliases[data_type] / base_method if data_type else Path(root) / base_method
        self.directory.mkdir(parents=True, exist_ok=True)
        self.experience_jsonl = self.directory / "reusable_experience.jsonl"
        self.experience_markdown = self.directory / "reusable_experience.md"
        records = self._compact_records(_read_jsonl(self.experience_jsonl))
        if records or self.experience_jsonl.exists():
            _write_jsonl(self.experience_jsonl, records)
        self._write_markdown(records)

    @staticmethod
    def _compact_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize legacy verbose records into the two-field compact schema."""

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
                "actual_missing_rate", "source_run_id"
            ) if key in record} if record.get("data_type") else {}
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
        self.experience_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def context(self, limit: int = 50) -> Dict[str, Any]:
        records = _read_jsonl(self.experience_jsonl)
        return {
            "base_method": self.base_method,
            "data_type": self.data_type,
            "current_conditions": self.conditions,
            "record_count": len(records),
            "reusable_experience": records[-limit:],
            "documents": {
                "experience_jsonl": str(self.experience_jsonl),
                "experience_markdown": str(self.experience_markdown),
            },
        }

    def record(
        self,
        experience: Dict[str, Any],
    ) -> Dict[str, str]:
        """Append one unique two-field lesson and discard run-specific metadata."""

        payload = dict(experience)
        if self.data_type:
            payload.update({"data_type": self.data_type, "base_method": self.base_method,
                            **self.conditions})
        record = self._compact_records([payload])
        if not record:
            raise ValueError("experience must contain a non-empty general lesson")
        new_record = record[0]
        records = self._compact_records(_read_jsonl(self.experience_jsonl))
        if new_record not in records:
            records.append(new_record)
            _write_jsonl(self.experience_jsonl, records)
            self._write_markdown(records)
        return {
            "experience_jsonl": str(self.experience_jsonl),
            "experience_markdown": str(self.experience_markdown),
        }


class RunPracticeStore:
    """Practice trajectory visible only to later rounds of the current Day 6 run."""

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
                "本实践库只服务于当前运行，不会被其他运行读取。\n"
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
