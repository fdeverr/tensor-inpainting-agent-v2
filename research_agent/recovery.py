"""Whole-modality evolution rounds followed by fixed-algorithm comparisons."""

from __future__ import annotations

import hashlib
import html
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .core.data import apply_observation_mask, load_observation_mask, load_tensor_data, save_tensor_data
from .core.fair_experiment import final_fit_and_evaluate
from .core.interpolation import nearest_neighbor_fill
from .core.audio_metrics import audio_metric_context, metric_score
from .core.metrics import evaluate_reconstruction_metrics
from .core.models.registry import MODEL_CLASSES, create_model, get_default_hyperparameters
from .dataset_evolution import training_feedback
from .recovery_data import DATA_TYPES, discover_recovery_data, export_audio, prepare_recovery_case
from .recovery_registry import archive_champion, champion_builder, champion_records
from .schemas import TrainingConfig
from .workflow import _make_run_id, _write_json
from .workflow_full import FullWorkflowConfig, run_full_workflow


FIXED_TENSOR_BASELINES = tuple(name for name in MODEL_CLASSES if name != "siren")


def _linear_waveform_fill(observed, mask, sample_count):
    """Interpolate each channel along original sample time, never reading hidden GT."""
    channels = observed.shape[-1]
    flat_observed = observed.reshape(-1, channels)
    flat_mask = mask.reshape(-1, channels)
    completed = flat_observed.copy()
    positions = np.arange(sample_count)
    for channel in range(channels):
        known = positions[flat_mask[:sample_count, channel]]
        if not len(known):
            raise ValueError("audio interpolation needs an observed sample per channel")
        completed[:sample_count, channel] = np.where(
            flat_mask[:sample_count, channel], flat_observed[:sample_count, channel],
            np.interp(positions, known, flat_observed[known, channel]))
    return completed.reshape(observed.shape)


def _fixed_baseline_hyperparameters(model_name, shape):
    """Choose a deterministic, dimension-valid configuration without GT tuning."""
    defaults = get_default_hyperparameters(model_name)
    search_space = MODEL_CLASSES[model_name].search_space(shape)
    selected = {}
    for name, preferred in defaults.items():
        options = search_space.get(name)
        if not options or preferred in options:
            selected[name] = preferred
        elif isinstance(preferred, (int, float)) and all(
                isinstance(option, (int, float)) for option in options):
            selected[name] = min(options, key=lambda option: (abs(option - preferred), option))
        else:
            selected[name] = options[0]
    return selected


def evaluate_fixed_baseline(model_name, case, output_dir, config):
    """Evaluate a fixed same-modality reference under the current metric protocol."""
    with audio_metric_context(case):
        return _evaluate_fixed_baseline(model_name, case, output_dir, config)


def _evaluate_fixed_baseline(model_name, case, output_dir, config):
    gt = load_tensor_data(case["gt_path"])
    mask = load_observation_mask(case["mask_path"])
    observed = apply_observation_mask(gt, mask)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if model_name in {"nearest_neighbor_manhattan", "linear_interpolation_waveform"}:
        completion = (_linear_waveform_fill(observed, mask, case["sample_count"])
                      if model_name == "linear_interpolation_waveform" else
                      nearest_neighbor_fill(observed, mask))
        path = directory / "reconstruction.npy"
        save_tensor_data(str(path), completion)
        result = {"metrics": evaluate_reconstruction_metrics(completion, gt, mask, device=config.device),
                  "artifacts": {"reconstruction": str(path)}, "parameter_count": 0}
    else:
        hyperparameters = _fixed_baseline_hyperparameters(model_name, gt.shape)
        learning_rate = TrainingConfig().learning_rate
        result = final_fit_and_evaluate(
            model_name=model_name,
            model_builder=lambda shape, mean, selected: create_model(model_name, shape, mean, selected),
            selected_trial={"hyperparameters": hyperparameters, "learning_rate": learning_rate,
                            "best_step": config.evaluation_steps, "runtime_seconds": 0.0},
            observed_image=observed, observed_mask=mask, ground_truth=gt,
            training_config=TrainingConfig(
                learning_rate=learning_rate, max_steps=config.evaluation_steps,
                validation_interval=config.evaluation_validation_interval,
                early_stopping_patience=config.evaluation_patience or config.evaluation_steps + 1,
                device=config.device),
            seed=case["seed"], output_dir=str(directory),
            include_full_reference_metrics=False, include_no_reference_metrics=False,
        )
        if case["data_type"] == "audio":
            completion = np.load(result["artifacts"]["reconstruction"], allow_pickle=False)
    if case["data_type"] == "audio":
        wav_path = directory / "reconstruction.wav"
        export_audio(completion, case, str(wav_path))
        result["artifacts"]["audio"] = str(wav_path)
    return result


@dataclass(frozen=True)
class RecoveryConfig:
    dataset_root: str
    output_dir: str = "research_agent/outputs/recovery"
    history_root: str = "research_agent/algorithms/history"
    candidate_root: str = "research_agent/algorithms/candidates"
    approved_root: str = "research_agent/algorithms/approved"
    knowledge_root: str = "research_agent/algorithms/evolution_knowledge"
    data_types: tuple[str, ...] = ("Image", "MSI", "Video", "audio")
    representatives: dict[str, str] = field(default_factory=dict)
    mask_type: str = "random"
    missing_rate: float = 0.4
    seed: int = 42
    image_size: int | None = 128
    audio_frame_size: int = 256
    evolution_steps: int = 1500
    evaluation_steps: int = 1000
    validation_interval: int = 10
    patience: int = 20
    evaluation_validation_interval: int = 10
    evaluation_patience: int = 0
    method_max_steps_ceiling: int | None = None
    fair_max_steps: int | None = None
    tuning_near_limit_ratio: float = 0.9
    tuning_expansion_factor: float = 2.0
    fair_learning_rate_candidates: tuple[float, ...] = (0.001, 0.01, 0.1)
    fair_refine_learning_rate: bool = True
    fair_learning_rate_refinement_factor: float = 3.0
    ablation_screen_trials: int = 1
    ablation_screen_max_steps: int = 300
    method_shortlist_size: int = 3
    screening_trials: int = 2
    screening_max_steps: int | None = None
    screening_patience: int = 10
    siren_max_steps: int | None = None
    siren_tuning_trials: int = 4
    siren_validation_interval: int = 25
    siren_patience: int = 20
    retrieval_top_k: int = 8
    minimum_psnr_delta: float = 0.2
    minimum_nmse_delta: float = 0.0
    ssim_tolerance: float = 0.002
    smoke_timeout_seconds: float = 10.0
    improvement_rounds: int = 5
    tuning_trials: int = 4
    base_model: str = "auto"
    device: str = "auto"
    llm_mode: str = "auto"
    lpips: bool = False
    no_reference_metrics: bool = False
    selection_visual_assessment: bool = False
    mutation_visual_assessment: bool = False
    prompt: str = ""
    siren_comparison: bool = True

    def validate(self):
        if not Path(self.dataset_root).is_dir():
            raise ValueError("dataset_root is not a directory")
        if not self.data_types or len(set(self.data_types)) != len(self.data_types) or any(x not in DATA_TYPES for x in self.data_types):
            raise ValueError("data_types must be a nonempty unique selection of Image/MSI/Video/audio")
        if self.mask_type not in {"random", "block", "slices", "sildes"} or not 0 < self.missing_rate < 1:
            raise ValueError("invalid mask_type or missing_rate")
        if self.image_size is not None and self.image_size < 8:
            raise ValueError("image_size must be >= 8 or None")
        if self.audio_frame_size < 2 or min(self.evolution_steps, self.evaluation_steps) < 1:
            raise ValueError("budgets and frame size must be positive")
        if not 1 <= self.improvement_rounds <= 100 or not 1 <= self.tuning_trials <= 5:
            raise ValueError("invalid rounds or tuning_trials")
        if min(self.validation_interval, self.patience, self.evaluation_validation_interval,
               self.screening_patience, self.siren_validation_interval, self.siren_patience,
               self.ablation_screen_max_steps, self.retrieval_top_k) < 1:
            raise ValueError("intervals, patience and screening/retrieval budgets must be positive")
        if self.evaluation_patience < 0:
            raise ValueError("evaluation_patience must be >= 0 (0 disables early stopping)")
        if self.method_max_steps_ceiling is not None and self.method_max_steps_ceiling < self.evolution_steps:
            raise ValueError("method_max_steps_ceiling must be >= evolution_steps")
        if any(x is not None and x < 1 for x in (self.fair_max_steps, self.screening_max_steps, self.siren_max_steps)):
            raise ValueError("optional training budgets must be positive")
        if not 3 <= self.method_shortlist_size <= 5 or not 1 <= self.screening_trials <= 3:
            raise ValueError("invalid method_shortlist_size or screening_trials")
        if not 1 <= self.siren_tuning_trials <= 4 or not 1 <= self.ablation_screen_trials <= 3:
            raise ValueError("invalid siren/ablation trial counts")
        if not 0 < self.tuning_near_limit_ratio <= 1 or self.tuning_expansion_factor <= 1:
            raise ValueError("invalid tuning expansion settings")
        if self.fair_learning_rate_refinement_factor <= 1 or not self.fair_learning_rate_candidates or any(
                not 1e-5 <= rate <= 1 for rate in self.fair_learning_rate_candidates):
            raise ValueError("invalid fair learning-rate search settings")
        if self.fair_refine_learning_rate and any(
                rate / self.fair_learning_rate_refinement_factor < 1e-5 or
                rate * self.fair_learning_rate_refinement_factor > 1
                for rate in self.fair_learning_rate_candidates):
            raise ValueError("refined learning rates must stay in [1e-5, 1]")
        if self.minimum_psnr_delta < 0 or self.minimum_nmse_delta < 0 or self.ssim_tolerance < 0 or self.smoke_timeout_seconds <= 0:
            raise ValueError("invalid promotion thresholds or smoke timeout")
        if any(key not in self.data_types for key in self.representatives):
            raise ValueError("representative modality not selected")
        if not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string")
        if any(not isinstance(value, bool) for value in (
                self.lpips, self.no_reference_metrics, self.selection_visual_assessment,
                self.mutation_visual_assessment, self.siren_comparison, self.fair_refine_learning_rate)):
            raise ValueError("feature switches must be booleans")


def evaluate_champion(record, case, output_dir, config):
    with audio_metric_context(case):
        return _evaluate_champion(record, case, output_dir, config)


def _evaluate_champion(record, case, output_dir, config):
    gt = load_tensor_data(case["gt_path"])
    mask = load_observation_mask(case["mask_path"])
    observed = apply_observation_mask(gt, mask)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    builder = champion_builder(record)
    if record["kind"] == "interpolation":
        completion = nearest_neighbor_fill(observed, mask)
        path = directory / "reconstruction.npy"
        save_tensor_data(str(path), completion)
        result = {"metrics": evaluate_reconstruction_metrics(
            completion, gt, mask, device=config.device,
            include_full_reference_metrics=config.lpips and case["data_type"] == "Image",
            include_no_reference_metrics=config.no_reference_metrics and case["data_type"] == "Image"),
            "artifacts": {"reconstruction": str(path)}, "parameter_count": 0}
    else:
        selected = record["config"]
        result = final_fit_and_evaluate(
            model_name=record["algorithm"], model_builder=builder,
            selected_trial={"hyperparameters": selected["hyperparameters"],
                            "learning_rate": selected["learning_rate"],
                            "best_step": config.evaluation_steps, "runtime_seconds": 0.0},
            observed_image=observed, observed_mask=mask, ground_truth=gt,
            training_config=TrainingConfig(
                learning_rate=selected["learning_rate"], max_steps=config.evaluation_steps,
                validation_interval=config.evaluation_validation_interval,
                early_stopping_patience=config.evaluation_patience or config.evaluation_steps + 1,
                device=config.device),
            seed=case["seed"], output_dir=str(directory),
            include_full_reference_metrics=config.lpips and case["data_type"] == "Image",
            include_no_reference_metrics=config.no_reference_metrics and case["data_type"] == "Image",
        )
        completion = np.load(result["artifacts"]["reconstruction"], allow_pickle=False)
    if case["data_type"] == "audio":
        wav_path = directory / "reconstruction.wav"
        export_audio(completion, case, str(wav_path))
        result["artifacts"]["audio"] = str(wav_path)
    return result


def summarize_evaluations(results, expected_count, data_type=None):
    valid = [x for x in results if x["status"] == "completed"]
    metrics = [x["result"]["metrics"] for x in valid]
    if data_type == "audio" or any("missing_nmse" in x for x in metrics):
        values = [x["missing_nmse"] for x in metrics if x.get("missing_nmse") is not None]
        return {
            "complete": len(valid) == len(values) == expected_count and expected_count > 0,
            "expected_count": expected_count, "completed_count": len(valid), "defined_nmse_count": len(values),
            "primary_metric": "mean_missing_nmse", "direction": "lower_is_better",
            "mean_missing_nmse": statistics.mean(values) if values else None,
            "perfect_count": sum(value == 0 for value in values),
        }
    psnr = [x["missing_psnr"] for x in metrics if x.get("missing_psnr") is not None]
    perfect = sum(x.get("missing_mse") == 0 for x in metrics)
    return {
        "complete": len(valid) == expected_count and expected_count > 0,
        "expected_count": expected_count, "completed_count": len(valid),
        "perfect_count": perfect,
        "mean_missing_psnr": statistics.mean(psnr) if psnr and not perfect else None,
        "mean_composite_ssim": statistics.mean(x["composite_ssim"] for x in metrics) if metrics else None,
        "mean_missing_mse": statistics.mean(x["missing_mse"] for x in metrics) if metrics else None,
    }


def _evaluate_cached(record, case, output_dir, config, cache):
    """Equivalent archived versions share one fit, never scores from other protocols."""
    champion_builder(record)  # Hash gates also apply on a cache hit.
    genome = {"kind": record["kind"], "algorithm": record["algorithm"],
              "code_sha256": record["code_sha256"], "library": record["model_library_sha256"],
              "hyperparameters": record["config"].get("hyperparameters"),
              "learning_rate": record["config"].get("learning_rate")}
    key = (json.dumps(genome, sort_keys=True), case["gt_sha256"], case["mask_sha256"],
           case["seed"], config.evaluation_steps, config.evaluation_validation_interval,
           config.evaluation_patience, config.device, config.lpips, config.no_reference_metrics,
           json.dumps({key: case.get(key) for key in ("data_type", "normalization", "original_min", "original_max", "sample_count", "channels")}, sort_keys=True))
    if key in cache:
        source_id, result = cache[key]
        return {**result, "equivalent_algorithm_evaluation_reused_from": source_id}
    result = evaluate_champion(record, case, output_dir, config)
    cache[key] = (record["archive_id"], result)
    return result


def _historical_comparison(record, cases, dev_index, run_dir, kind, config, cache):
    results = []
    for index, case in enumerate(cases):
        try:
            result = _evaluate_cached(
                record, case, run_dir / kind / "historical_reference" / record["archive_id"] / str(index),
                config, cache,
            )
            results.append({"source": case["source"], "is_development": index == dev_index,
                            "status": "completed", "result": result})
        except Exception as error:
            results.append({"source": case["source"], "is_development": index == dev_index,
                            "status": "failed", "error": "%s: %s" % (type(error).__name__, error)})
    return {"archive_id": record["archive_id"], "algorithm": record["algorithm"],
            "results": results, "summary": summarize_evaluations(results, len(cases), kind),
            "non_development_summary": summarize_evaluations(
                [item for item in results if not item["is_development"]], len(cases) - 1, kind)}


def _fixed_comparison(name, cases, dev_index, run_dir, kind, config):
    results = []
    for index, case in enumerate(cases):
        try:
            result = evaluate_fixed_baseline(
                name, case, run_dir / kind / "fixed_baselines" / name / str(index), config)
            results.append({"source": case["source"], "is_development": index == dev_index,
                            "status": "completed", "result": result})
        except Exception as error:
            results.append({"source": case["source"], "is_development": index == dev_index,
                            "status": "failed", "error": "%s: %s" % (type(error).__name__, error)})
    return {"algorithm": name, "role": "fixed_baseline", "results": results,
            "summary": summarize_evaluations(results, len(cases), kind),
            "non_development_summary": summarize_evaluations(
                [item for item in results if not item["is_development"]], len(cases) - 1, kind)}


def _comparison_feedback(record):
    """Keep scores and useful curve facts, not raw GT or full training histories."""
    return {"algorithm": record["algorithm"], "archive_id": record.get("archive_id"),
            "summary": record["summary"],
            "samples": [{"sample_index": index, "sample": Path(item["source"]).name,
                         "status": item["status"],
                         "metrics": {key: value for key, value in item.get("result", {}).get("metrics", {}).items()
                                     if key in {"missing_psnr", "composite_ssim", "missing_nmse"}},
                         "training": (training_feedback(item["result"])
                                      if item["status"] == "completed" else None),
                         "error": item.get("error")}
                        for index, item in enumerate(record["results"])]}


def _best_reference_by_sample(comparisons, cases):
    winners = []
    for index, case in enumerate(cases):
        eligible = [(record, record["results"][index]["result"]["metrics"])
                    for record in comparisons
                    if record["results"][index]["status"] == "completed"]
        if eligible:
            record, metrics = max(eligible, key=lambda item: metric_score(item[1]))
            winners.append({"sample_index": index, "sample": Path(case["source"]).name,
                            "algorithm": record["algorithm"],
                            "archive_id": record.get("archive_id"),
                            "metrics": {key: metrics.get(key) for key in
                                        (("missing_nmse",) if case["data_type"] == "audio" else
                                         ("missing_psnr", "composite_ssim"))}})
        else:
            winners.append({"sample_index": index, "sample": Path(case["source"]).name,
                            "status": "no_valid_reference"})
    return winners


def _comparison_matrix(kind, group, title="逐数据集算法对比"):
    """Render paper-style, per-sample comparisons with global per-row ranks."""
    records = [*group.get("comparisons", []), *group.get("baseline_comparisons", [])]
    if not records:
        return ["### " + title, "", "暂无可比较的算法结果。", ""]

    metric_keys = (("missing_nmse", "NMSE ↓", False),) if kind == "audio" else (
        ("missing_psnr", "PSNR ↑", True), ("composite_ssim", "SSIM ↑", True))
    current_id = group.get("current_champion", {}).get("archive_id")
    current = [record for record in records if record.get("archive_id") == current_id] if current_id else []
    current_objects = {id(record) for record in current}
    remainder = [record for record in records if id(record) not in current_objects]
    sort_key, _, higher_is_better = metric_keys[0]
    summary_key = "mean_" + sort_key

    def ordering(record):
        summary = record.get("summary", {})
        value = summary.get(summary_key) if summary.get("complete") else None
        if sort_key == "missing_psnr" and summary.get("complete") and summary.get("perfect_count"):
            value = math.inf
        return (value is None, -value if higher_is_better and value is not None else
                value if value is not None else math.inf, record["algorithm"])

    ordered = current + sorted(remainder, key=ordering)
    sources = [case["source"] for case in group.get("cases", [])]
    for record in records:
        for item in record["results"]:
            if item["source"] not in sources:
                sources.append(item["source"])
    by_record = [{item["source"]: item for item in record["results"]} for record in ordered]

    def value_at(record_index, source, key):
        if source is None:
            summary = ordered[record_index].get("summary", {})
            if not summary.get("complete"):
                return None
            if key == "missing_psnr" and summary.get("perfect_count"):
                return math.inf
            return summary.get("mean_" + key)
        item = by_record[record_index].get(source)
        if not item or item["status"] != "completed":
            return None
        return item.get("result", {}).get("metrics", {}).get(key)

    def ranks(source, key, higher):
        values = sorted({value_at(index, source, key) for index in range(len(ordered))
                         if value_at(index, source, key) is not None}, reverse=higher)
        return values[:2]

    def cell(value, rank, key):
        if value is None:
            return "—"
        digits = 6 if key == "missing_nmse" else 3
        rendered = "∞" if math.isinf(value) and value > 0 else ("%%.%df" % digits) % value
        if rank and value == rank[0]:
            return "<strong>" + rendered + "</strong>"
        if len(rank) > 1 and value == rank[1]:
            return "<u>" + rendered + "</u>"
        return rendered

    lines = ["### " + title, "",
             "同一行比较相同数据与 mask；<strong>粗体</strong>为最优，<u>下划线</u>为次优（并列按同一名次标记）。"
             "每个指标独立排名；PSNR/SSIM 越高越好，NMSE 越低越好。",
             "“全类平均”只显示全部样本成功的算法；失败不计作零分，也不以部分样本平均冒充完整结果。",
             "表格按平均主指标排序%s，每块最多 7 种算法。" %
             ("（本轮进化算法固定在第一列）" if current_id else ""), ""]
    for start in range(0, len(ordered), 7):
        stop = min(start + 7, len(ordered))
        lines += ["<div style=\"overflow-x:auto\">", "<table>", "<thead>",
                  "<tr><th rowspan=\"2\">数据</th>"]
        for record in ordered[start:stop]:
            role = "本轮进化" if record.get("archive_id") == current_id and current_id else (
                "历史版本" if record.get("archive_id") else "固定基线")
            title = html.escape(record["algorithm"])
            version = html.escape(str(record.get("archive_id", "")))
            if version:
                title += "<br><small>" + version + " · " + role + "</small>"
            else:
                title += "<br><small>" + role + "</small>"
            lines.append('<th colspan="%d">%s</th>' % (len(metric_keys), title))
        lines += ["</tr>", "<tr>"]
        for _ in ordered[start:stop]:
            for _, label, _ in metric_keys:
                lines.append("<th>%s</th>" % label)
        lines += ["</tr>", "</thead>", "<tbody>"]
        for source in [*sources, None]:
            row_label = "全类平均" if source is None else Path(source).name
            if source == group.get("representative"):
                row_label += " *"
            lines.append("<tr><th>" + html.escape(row_label) + "</th>")
            row_ranks = {key: ranks(source, key, higher) for key, _, higher in metric_keys}
            for index in range(start, stop):
                for key, _, _ in metric_keys:
                    lines.append("<td>" + cell(value_at(index, source, key), row_ranks[key], key) + "</td>")
            lines.append("</tr>")
        lines += ["</tbody>", "</table>", "</div>", ""]
    if group.get("representative"):
        lines += ["* 表示用于结构与超参数搜索的代表样本；其他样本也参与每轮整类进化判定。", ""]
    failures = [(record, item) for record in ordered for item in record["results"]
                if item["status"] != "completed"]
    if failures:
        lines += ["评测失败（对应单元格显示“—”）：", ""]
        for record, item in failures:
            lines.append("- `%s` / `%s`：%s" % (
                record["algorithm"], Path(item["source"]).name,
                str(item.get("error", item["status"])).replace("\n", " ")))
        lines.append("")
    return lines


def _report(state):
    def fmt(value, digits=4):
        return "N/A" if value is None else ("%%.%df" % digits) % value

    lines = ["# Multi-Dimensional Data Recovery 报告", "",
             "每类只在一个样本上搜索候选结构与超参数；每轮候选和 incumbent 均在该类全部有效样本上用冻结配置独立拟合与评测，整类平均结果决定是否接受，并反馈到下一轮。",
             "新旧冠军均使用本次同一批 GT、mask、随机种子和评测步数重新拟合；历史分数不直接混排。",
             "GT 仍用于各样本的 checkpoint 选择，属于 oracle 开发评测，不是盲测泛化证据。",
             "标 * 的代表样本用于结构/超参数搜索；整类全部样本已参与每轮算法选择，不是独立测试集。",
             "音频仅评价缺失原始波形 NMSE（越低越好），不计算 PSNR/SSIM；padding 不计入误差。", ""]
    lines += ["LPIPS、无参考图像指标、选择/变异视觉观察开关仅应用于 Image；其他类型不使用 RGB 图像指标或图像观察来替代本类型评测。", ""]
    for kind, group in state["modalities"].items():
        lines += ["## %s" % kind, "", "- 状态：`%s`" % group["status"]]
        if group.get("error"):
            lines += ["- 错误：%s" % group["error"], ""]
            continue
        lines += ["- 研发样本：`%s`" % group.get("representative", "无"),
                  "- 本轮冠军：`%s`" % group.get("current_champion", {}).get("algorithm", "无"),
                  "- 算法框架与进化报告：`%s`" % group.get("development_report", "无"),
                  "- 进化前整类对比证据：`%s`" % group.get("pre_evolution_reference_path", "无"),
                  "- 首轮所见实际 incumbent 整类证据：`%s`" % group.get("selected_incumbent_reference_path", "无"),
                  "- 同类全量最优版本：`%s`" % group.get("dataset_best_archive_id", "无完整结果"), ""]
        selection = group.get("method_selection", {})
        if selection:
            lines += ["### 基础模型如何选出", ""]
            if selection.get("status") == "skipped":
                lines += ["本次手工指定基础方法 `%s`，没有运行自动短名单预赛。" %
                          selection.get("winner", "未知"), ""]
            else:
                recommendation_source = ("LLM 推荐" if selection.get("recommendation_mode") in
                                         {"llm", "llm_repaired"} else "自动回退建议")
                lines += ["- %s的方法：`%s`" %
                          (recommendation_source,
                           ", ".join(selection.get("recommended_methods") or
                                     [selection.get("recommendation") or "未记录"])),
                          "- 实际入围预赛：`%s`" % ", ".join(selection.get("shortlist", [])),
                          "- 同预算数值预赛胜出：`%s`" % selection.get("winner", "未记录"),
                          "- 预赛范围：同类全部样本，使用冻结配置、相同 mask 和轻量评测预算；只有全量成功的方法参与排名。",
                          "- LLM 推荐时可参考进化前固定基线的整类及逐样本指标，但不接收原始 GT。",
                          "- 固定基线不会自动挤占 LLM 的推荐名额；完整对比结果还会供后续算法进化参考。", ""]
                screening_records = []
                for item in selection.get("results", []):
                    cohort = item.get("dataset_evaluation") or {}
                    if not cohort:
                        continue
                    screening_records.append({
                        "algorithm": item["method"], "summary": cohort["summary"],
                        "results": [{"source": sample["source"], "status": sample["status"],
                                     "result": {"metrics": sample.get("metrics", {})},
                                     "error": sample.get("error")}
                                    for sample in cohort["results"]],
                    })
                if screening_records:
                    lines.extend(_comparison_matrix(kind, {
                        "comparisons": screening_records, "baseline_comparisons": [],
                        "cases": group.get("cases", []),
                        "representative": group.get("representative"),
                    }, title="基础分解轻量预赛（整类样本）"))
        lines.extend(_comparison_matrix(kind, group))
        rounds = group.get("evolution_rounds", [])
        if rounds:
            lines += ["### 每轮进化：整类评测与反馈", "",
                      "所有样本使用该轮同一候选架构/超参数、评测步数和 checkpoint 规则；每个样本独立拟合。候选必须完整覆盖全类，并达到整类提升阈值；若旧 incumbent 不完整，完整候选可接替。", ""]
            if kind == "audio":
                lines += ["| 轮次 | incumbent | candidate | 全量成功数 | incumbent 平均 NMSE ↓ | candidate 平均 NMSE ↓ | 判定 |",
                          "|---:|---|---|---:|---:|---:|---|"]
            else:
                lines += ["| 轮次 | incumbent | candidate | 全量成功数 | incumbent 平均 PSNR ↑ | candidate 平均 PSNR ↑ | incumbent / candidate 平均 SSIM ↑ | 判定 |",
                          "|---:|---|---|---:|---:|---:|---|---|"]
            for round_item in rounds:
                cohort = round_item.get("dataset_evaluation") or {}
                before = cohort.get("incumbent", {}).get("summary", {})
                after = cohort.get("candidate", {}).get("summary", {})
                common = (round_item["round"], round_item["incumbent_before"],
                          round_item["candidate_id"],
                          "%s/%s" % (after.get("completed_count", 0), after.get("expected_count", 0)))
                if kind == "audio":
                    lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                        *common, fmt(before.get("mean_missing_nmse"), 6),
                        fmt(after.get("mean_missing_nmse"), 6), round_item["judgment"]["decision"]))
                else:
                    before_psnr = "∞" if before.get("perfect_count") else fmt(before.get("mean_missing_psnr"))
                    after_psnr = "∞" if after.get("perfect_count") else fmt(after.get("mean_missing_psnr"))
                    lines.append("| %s | %s | %s | %s | %s | %s | %s / %s | %s |" % (
                        *common, before_psnr, after_psnr,
                        fmt(before.get("mean_composite_ssim")), fmt(after.get("mean_composite_ssim")),
                        round_item["judgment"]["decision"]))
            lines += ["", "#### 每轮逐样本结果", ""]
            if kind == "audio":
                lines += ["| 轮次 | 数据文件 | incumbent NMSE ↓ | candidate NMSE ↓ | candidate 状态 |",
                          "|---:|---|---:|---:|---|"]
            else:
                lines += ["| 轮次 | 数据文件 | incumbent PSNR ↑ | candidate PSNR ↑ | incumbent / candidate SSIM ↑ | candidate 状态 |",
                          "|---:|---|---:|---:|---|---|"]
            for round_item in rounds:
                cohort = round_item.get("dataset_evaluation") or {}
                old_results = cohort.get("incumbent", {}).get("results", [])
                new_results = cohort.get("candidate", {}).get("results", [])
                for old, new in zip(old_results, new_results):
                    name = Path(new["source"]).name.replace("|", "\\|")
                    status = new["status"] + ((": " + str(new["error"]).replace("|", "\\|")) if new.get("error") else "")
                    old_metrics, new_metrics = old.get("metrics", {}), new.get("metrics", {})
                    if kind == "audio":
                        lines.append("| %s | %s | %s | %s | %s |" % (
                            round_item["round"], name,
                            fmt(old_metrics.get("missing_nmse"), 6),
                            fmt(new_metrics.get("missing_nmse"), 6), status))
                    else:
                        old_psnr = "∞" if old_metrics.get("missing_mse") == 0 else fmt(old_metrics.get("missing_psnr"))
                        new_psnr = "∞" if new_metrics.get("missing_mse") == 0 else fmt(new_metrics.get("missing_psnr"))
                        lines.append("| %s | %s | %s | %s | %s / %s | %s |" % (
                            round_item["round"], name, old_psnr, new_psnr,
                            fmt(old_metrics.get("composite_ssim")),
                            fmt(new_metrics.get("composite_ssim")), status))
            lines += ["", "完整轮次证据保存在各轮的 `dataset_evaluation.json` 和 Day 6 `state.json`。", ""]
        lines += ["固定基线在进化前运行：与冠军共用 GT、mask、种子、评测步数和 checkpoint 规则；"
                  "插值不训练，可训练基线使用固定且合法的超参数。其逐样本效果与训练摘要会反馈给进化 LLM。"
                  "固定基线只作参照，不参与冠军历史版本排序；不同数据类型不混合求平均。", ""]
    if state["inventory"]["failures"]:
        lines += ["## 数据完整性问题", "", "以下源文件未纳入有效数据集，不会默默按成功样本算入均值：", ""]
        lines += ["- `%s`：%s" % (x["source"], x["error"]) for x in state["inventory"]["failures"]]
    return "\n".join(lines) + "\n"


def run_recovery(config: RecoveryConfig, inventory_only=False, evaluate_only=False):
    config.validate()
    run_id = _make_run_id("recovery")
    run_dir = Path(config.output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    inventory = discover_recovery_data(config.dataset_root, config.image_size, config.audio_frame_size)
    state: dict[str, Any] = {"run_id": run_id, "config": asdict(config), "inventory": inventory,
                             "modalities": {}, "stage": "INVENTORY", "run_dir": str(run_dir),
                             "mode": "evaluation_only" if evaluate_only else "evolve_and_evaluate"}
    _write_json(run_dir / "state.json", state)
    if inventory_only:
        return state
    for kind in config.data_types:
        group: dict[str, Any] = {"status": "preparing", "comparisons": [], "baseline_comparisons": []}
        state["modalities"][kind] = group
        try:
            samples = inventory["groups"][kind]
            if not samples:
                raise ValueError("no intact GT sample for %s" % kind)
            cases = []
            for index, sample in enumerate(samples):
                sample_seed = config.seed + int(hashlib.sha256(Path(sample["source"]).name.encode()).hexdigest()[:7], 16)
                cases.append(prepare_recovery_case(
                    sample["source"], run_dir / kind / "cases" / str(index), config.missing_rate,
                    config.mask_type, sample_seed, config.image_size, config.audio_frame_size))
            history = champion_records(config.history_root, kind)
            if kind in config.representatives:
                requested = Path(config.representatives[kind])
                matches = [i for i, x in enumerate(cases) if Path(x["source"]) == requested.resolve() or Path(x["source"]).name == str(requested)]
                if len(matches) != 1:
                    raise ValueError("representative must identify exactly one intact %s sample" % kind)
                dev_index = matches[0]
            else:
                # Rotate representatives across complete runs, not always the first file.
                dev_index = len(history) % len(cases)
            dev = cases[dev_index]
            group.update(cases=cases, representative=dev["source"], representative_index=dev_index)
            group["status"] = "pre_evolution_comparisons"
            _write_json(run_dir / "state.json", state)
            evaluation_cache = {}
            references = []
            for old in history:
                comparison = _historical_comparison(
                    old, cases, dev_index, run_dir, kind, config, evaluation_cache)
                group["comparisons"].append(comparison)
                dev_result = comparison["results"][dev_index]
                reference = {"archive_id": old["archive_id"], "algorithm": old["algorithm"],
                             "base_method": old["base_method"], "config": old["config"],
                             "whole_modality_summary": comparison["summary"]}
                if dev_result["status"] == "completed":
                    reference["metrics"] = {key: dev_result["result"]["metrics"].get(key)
                                            for key in (("missing_nmse",) if kind == "audio" else
                                                        ("missing_psnr", "composite_ssim", "lpips"))}
                else:
                    reference.update(status="failed", error=dev_result["error"])
                references.append(reference)
                _write_json(run_dir / "state.json", state)
            group["historical_references"] = references
            interpolation = ("linear_interpolation_waveform" if kind == "audio" else
                             "nearest_neighbor_manhattan")
            baseline_names = [interpolation, *FIXED_TENSOR_BASELINES]
            if config.siren_comparison:
                baseline_names.append("siren")
            for name in baseline_names:
                group["baseline_comparisons"].append(
                    _fixed_comparison(name, cases, dev_index, run_dir, kind, config))
                _write_json(run_dir / "state.json", state)
            dataset_reference = {
                "data_type": kind,
                "selection_policy": "all same-type samples are development feedback, not an untouched test set",
                "protocol": {"evaluation_steps": config.evaluation_steps,
                             "validation_interval": config.evaluation_validation_interval,
                             "patience": config.evaluation_patience,
                             "mask_type": config.mask_type, "missing_rate": config.missing_rate},
                "fixed_baselines": [_comparison_feedback(item) for item in group["baseline_comparisons"]],
                "historical_champions": [_comparison_feedback(item) for item in group["comparisons"]],
                "best_reference_by_sample": _best_reference_by_sample(
                    [*group["baseline_comparisons"], *group["comparisons"]], cases),
            }
            group["pre_evolution_reference"] = dataset_reference
            reference_path = run_dir / kind / "pre_evolution_reference.json"
            _write_json(reference_path, dataset_reference)
            group["pre_evolution_reference_path"] = str(reference_path)
            group["status"] = "evaluating" if evaluate_only else "evolving"
            _write_json(run_dir / "state.json", state)
            if evaluate_only:
                if not history:
                    raise ValueError("evaluation-only requires an archived %s champion" % kind)
                current = history[-1]
                records = history
                development_report = str(Path(current["archive_dir"]) / "development_report.md")
            else:
                evolution = run_full_workflow(FullWorkflowConfig(
                    image_path=dev["gt_path"], observation_mask_path=dev["mask_path"], image_size=None,
                    data_type=DATA_TYPES[kind], historical_algorithm_reference=references,
                    dataset_algorithm_reference=dataset_reference,
                    valid_element_count=dev["sample_count"] * dev["channels"] if kind == "audio" else None,
                    output_dir=str(run_dir / kind / "development"), candidate_root=config.candidate_root,
                    approved_root=config.approved_root, knowledge_root=config.knowledge_root,
                    mask_type="slices" if config.mask_type == "sildes" else config.mask_type,
                    missing_rate=config.missing_rate, seed=dev["seed"], device=config.device,
                    llm_mode=config.llm_mode, base_model=config.base_model,
                    method_max_steps=config.evolution_steps,
                    method_max_steps_ceiling=config.method_max_steps_ceiling or config.evolution_steps,
                    tuning_near_limit_ratio=config.tuning_near_limit_ratio,
                    tuning_expansion_factor=config.tuning_expansion_factor,
                    fair_max_steps=config.fair_max_steps or config.evolution_steps,
                    max_improvement_rounds=config.improvement_rounds,
                    tuning_trials=config.tuning_trials, validation_interval=config.validation_interval,
                    patience=config.patience,
                    fair_learning_rate_candidates=config.fair_learning_rate_candidates,
                    fair_refine_learning_rate=config.fair_refine_learning_rate,
                    fair_learning_rate_refinement_factor=config.fair_learning_rate_refinement_factor,
                    ablation_screen_trials=config.ablation_screen_trials,
                    ablation_screen_max_steps=config.ablation_screen_max_steps,
                    method_shortlist_size=config.method_shortlist_size,
                    screening_trials=config.screening_trials,
                    screening_max_steps=config.screening_max_steps or min(200, config.evolution_steps),
                    screening_patience=config.screening_patience,
                    siren_max_steps=config.siren_max_steps or config.evolution_steps,
                    siren_comparison=config.siren_comparison,
                    siren_tuning_trials=config.siren_tuning_trials,
                    siren_validation_interval=config.siren_validation_interval,
                    siren_patience=config.siren_patience,
                    retrieval_top_k=config.retrieval_top_k,
                    minimum_psnr_delta=config.minimum_psnr_delta, ssim_tolerance=config.ssim_tolerance,
                    minimum_nmse_delta=config.minimum_nmse_delta,
                    evolution_cases=cases,
                    dataset_evaluation_steps=config.evaluation_steps,
                    dataset_evaluation_validation_interval=config.evaluation_validation_interval,
                    dataset_evaluation_patience=config.evaluation_patience,
                    smoke_timeout_seconds=config.smoke_timeout_seconds,
                    full_reference_metrics=config.lpips and kind == "Image",
                    no_reference_metrics=config.no_reference_metrics and kind == "Image",
                    selection_visual_assessment=config.selection_visual_assessment and kind == "Image",
                    mutation_visual_assessment=config.mutation_visual_assessment and kind == "Image",
                    prompt="针对 %s 张量恢复提出并验证算法改进；可参考本轮同条件实测的历史算法。\n\n%s" % (kind, config.prompt),
                ))
                current = archive_champion(config.history_root, kind, evolution, dev, asdict(config))
                records = [*history, current]
                development_report = evolution["artifacts"]["report"]
                day4_path = evolution["artifacts"].get("day4_state")
                if day4_path:
                    day4_state = json.loads(Path(day4_path).read_text(encoding="utf-8"))
                    screening = day4_state.get("results", {}).get("method_screening", {})
                    plan_path = day4_state.get("artifacts", {}).get("method_plan")
                    method_plan = (json.loads(Path(plan_path).read_text(encoding="utf-8"))
                                   if plan_path else {})
                    group["method_selection"] = {
                        "status": screening.get("status", "not_recorded"),
                        "recommendation": screening.get("selector_recommendation"),
                        "recommended_methods": screening.get("selector_recommended_methods", []),
                        "results": screening.get("results", []),
                        "recommendation_mode": method_plan.get("selector_recommendation", {}).get("selection_mode"),
                        "shortlist": screening.get("shortlist", []),
                        "winner": screening.get("winner", day4_state.get("selected_model")),
                        "llm_used": day4_state.get("results", {}).get("selector_diagnostics", {}).get("llm_used"),
                        "selection_scope": screening.get("selection_scope"),
                        "results": [{"method": item["method"], "status": item["status"],
                                     "dataset_evaluation": {
                                         "summary": item["dataset_evaluation"]["summary"],
                                         "results": [{key: sample.get(key) for key in
                                                      ("source", "status", "metrics", "error")}
                                                     for sample in item["dataset_evaluation"]["results"]],
                                     }}
                                    for item in screening.get("results", [])
                                    if item.get("dataset_evaluation")],
                    }
                group["selected_incumbent_reference_path"] = evolution["artifacts"].get(
                    "pre_candidate_incumbent_dataset")
                day6_path = evolution["artifacts"].get("day6_state")
                group["evolution_rounds"] = (json.loads(Path(day6_path).read_text(encoding="utf-8"))["rounds"]
                                             if day6_path else [])
            group.update(current_champion=current, development_report=development_report, status="evaluating")
            _write_json(run_dir / "state.json", state)
            for record in records:
                if any(item["archive_id"] == record["archive_id"] for item in group["comparisons"]):
                    continue  # Historical versions were evaluated before evolution.
                results = []
                for index, case in enumerate(cases):
                    try:
                        result = _evaluate_cached(
                            record, case, run_dir / kind / "evaluation" / record["archive_id"] / str(index), config, evaluation_cache)
                        results.append({"source": case["source"], "is_development": index == dev_index,
                                        "status": "completed", "result": result})
                    except Exception as error:
                        results.append({"source": case["source"], "is_development": index == dev_index,
                                        "status": "failed", "error": "%s: %s" % (type(error).__name__, error)})
                comparison = {"archive_id": record["archive_id"], "algorithm": record["algorithm"],
                              "results": results, "summary": summarize_evaluations(results, len(cases), kind),
                              "non_development_summary": summarize_evaluations([x for x in results if not x["is_development"]], len(cases) - 1, kind)}
                group["comparisons"].append(comparison)
                _write_json(run_dir / "state.json", state)
            complete = [x for x in group["comparisons"] if x["summary"]["complete"]]
            if complete:
                best = min(complete, key=lambda x: x["summary"]["mean_missing_nmse"]) if kind == "audio" else max(complete, key=lambda x: (
                    bool(x["summary"]["perfect_count"]), x["summary"]["mean_missing_psnr"] or 0,
                    x["summary"]["mean_composite_ssim"]))
                group["dataset_best_archive_id"] = best["archive_id"]
            group["selection_metric"] = "mean_missing_nmse" if kind == "audio" else "mean_missing_psnr"
            all_comparisons = [*group["comparisons"], *group["baseline_comparisons"]]
            group["status"] = "completed" if all(x["summary"]["complete"] for x in all_comparisons) else "completed_with_evaluation_failures"
            evaluation_record = {"run_id": run_id, "data_type": kind, "protocol": asdict(config),
                                 "dataset_best_archive_id": group.get("dataset_best_archive_id"),
                                 "cases": cases, "comparisons": group["comparisons"],
                                 "baseline_comparisons": group["baseline_comparisons"]}
            _write_json(Path(config.history_root) / kind / "evaluations" / (run_id + ".json"), evaluation_record)
            _write_json(Path(config.history_root) / kind / "latest_evaluation.json", evaluation_record)
        except Exception as error:
            group.update(status="failed", error="%s: %s" % (type(error).__name__, error))
        _write_json(run_dir / "state.json", state)
    input_failures = [x for x in inventory["failures"] if x.get("data_type") in (*config.data_types, None)]
    state["stage"] = "COMPLETED" if all(x["status"] == "completed" for x in state["modalities"].values()) and not input_failures else "COMPLETED_WITH_FAILURES"
    (run_dir / "report.md").write_text(_report(state), encoding="utf-8")
    state["report"] = str(run_dir / "report.md")
    _write_json(run_dir / "state.json", state)
    return state
