"""Run-scoped practice memory and cross-run reusable evolution experience."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_SEMANTICS: Dict[str, Dict[str, str]] = {
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


def describe_metrics(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Attach stable semantics to every metric value persisted in practice memory."""

    described: Dict[str, Dict[str, Any]] = {}
    for name, semantics in METRIC_SEMANTICS.items():
        described[name] = {"value": metrics.get(name), **semantics}
    for name, value in metrics.items():
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

    def __init__(self, root: str | Path, base_method: str) -> None:
        self.directory = Path(root) / base_method
        self.directory.mkdir(parents=True, exist_ok=True)
        self.experience_jsonl = self.directory / "reusable_experience.jsonl"
        self.experience_markdown = self.directory / "reusable_experience.md"
        records = self._compact_records(_read_jsonl(self.experience_jsonl))
        if records or self.experience_jsonl.exists():
            _write_jsonl(self.experience_jsonl, records)
        self._write_markdown(records)

    @staticmethod
    def _compact_records(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Normalize legacy verbose records into the two-field compact schema."""

        compact: List[Dict[str, str]] = []
        seen = set()
        for record in records:
            lesson = record.get("experience") or record.get("reusable_principle")
            if not isinstance(lesson, str) or not lesson.strip():
                continue
            lesson = lesson.strip()
            confidence = record.get("confidence", "low")
            if confidence not in {"low", "medium", "high"}:
                confidence = "low"
            if lesson in seen:
                continue
            seen.add(lesson)
            compact.append({"experience": lesson, "confidence": confidence})
        return compact

    def _write_markdown(self, records: List[Dict[str, str]]) -> None:
        lines = [
            "# %s 可复用算法变异经验" % self.directory.name,
            "",
            "本文件跨运行累积，只保存一般性经验和置信度。",
            "",
        ]
        for record in records:
            lines.extend(
                [
                    "- 经验：%s" % record["experience"],
                    "  - 置信度：`%s`" % record["confidence"],
                ]
            )
        self.experience_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def context(self, limit: int = 50) -> Dict[str, Any]:
        records = _read_jsonl(self.experience_jsonl)
        return {
            "base_method": self.directory.name,
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

        record = self._compact_records([experience])
        if not record:
            raise ValueError("experience must contain a non-empty general lesson")
        new_record = record[0]
        records = self._compact_records(_read_jsonl(self.experience_jsonl))
        if all(item["experience"] != new_record["experience"] for item in records):
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
        practice_record = {"recorded_at": datetime.now().isoformat(), **practice}
        _append_jsonl(self.practice_jsonl, practice_record)
        with self.practice_markdown.open("a", encoding="utf-8") as stream:
            stream.write(
                "\n## 第 %s 轮\n\n"
                "- 目标：%s\n- Idea：%s\n- 变异对象：`%s`\n- 唯一改动：%s\n"
                "- 结果：`%s`；是否替换当前最优：`%s`\n"
                % (
                    practice["round"],
                    practice["mutation_goal"],
                    practice["idea"],
                    practice["mutation_target"],
                    practice["single_change"],
                    practice["judgment"]["decision"],
                    practice["incumbent_updated"],
                )
            )
            for role in ("incumbent_metrics", "candidate_metrics"):
                stream.write("\n### %s\n\n" % role)
                for metric in practice["result"][role].values():
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
            stream.write(
                "\n- 多模态视觉评价：`%s`（后续可接入）\n"
                % practice["result"]["visual_assessment"]["status"]
            )
        return {
            "practice_jsonl": str(self.practice_jsonl),
            "practice_markdown": str(self.practice_markdown),
        }
