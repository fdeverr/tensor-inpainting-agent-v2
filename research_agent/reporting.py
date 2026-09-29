"""Markdown reporting for a complete, auditable research run."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional


def _number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return ("%%.%df" % digits) % float(value)


def build_method_results(
    day4: Dict[str, Any],
    day6: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize all evaluated methods into one report-friendly table."""

    results = [
        {
            "algorithm": "nearest_neighbor_manhattan",
            "role": "interpolation_baseline",
            "metrics": day4["results"]["interpolation_metrics"],
            "runtime_seconds": None,
            "parameter_count": 0,
            "eligible_for_final_output": True,
            "reconstruction": day4["artifacts"]["interpolation"],
            "reconstruction_mat": day4["artifacts"].get("interpolation_mat"),
            "preview": day4["artifacts"].get(
                "interpolation_preview", day4["artifacts"]["interpolation"]
            ),
        }
    ]
    siren = day4.get("results", {}).get("siren_comparison", {})
    if siren.get("status") == "completed":
        training = siren["training"]
        results.append(
            {
                "algorithm": "siren",
                "role": "implicit_neural_baseline",
                "metrics": siren["metrics"],
                "runtime_seconds": training["runtime_seconds"],
                "parameter_count": training["parameter_count"],
                "eligible_for_final_output": True,
                "reconstruction": day4["artifacts"]["siren_reconstruction"],
                "reconstruction_mat": day4["artifacts"].get(
                    "siren_reconstruction_mat"
                ),
                "preview": day4["artifacts"].get(
                    "siren_preview",
                    day4["artifacts"]["siren_reconstruction"],
                ),
            }
        )
    if day6 is None:
        results.append(
            {
                "algorithm": day4["selected_model"],
                "role": "tensor_baseline",
                "metrics": day4["results"]["tensor_metrics"],
                "runtime_seconds": day4["results"]["training"]["runtime_seconds"],
                "parameter_count": day4["results"]["training"]["parameter_count"],
                "eligible_for_final_output": True,
                "reconstruction": day4["artifacts"]["tensor_reconstruction"],
                "reconstruction_mat": day4["artifacts"].get(
                    "tensor_reconstruction_mat"
                ),
                "preview": day4["artifacts"].get(
                    "tensor_preview", day4["artifacts"]["tensor_reconstruction"]
                ),
            }
        )
        return results

    last_round = day6["rounds"][-1]
    baseline = day6["rounds"][0]["baseline_final"]
    evolved = day6.get("best_evolved")
    candidate = (
        evolved["final"] if evolved is not None else last_round["candidate_final"]
    )
    accepted = evolved is not None
    results.extend(
        [
            {
                "algorithm": day4["selected_model"],
                "role": "tensor_baseline",
                "metrics": baseline["metrics"],
                "runtime_seconds": baseline["runtime_seconds"],
                "parameter_count": baseline["parameter_count"],
                "eligible_for_final_output": True,
                "reconstruction": baseline["artifacts"]["reconstruction"],
                "reconstruction_mat": baseline["artifacts"].get(
                    "reconstruction_mat"
                ),
                "preview": baseline["artifacts"].get(
                    "preview", baseline["artifacts"]["reconstruction"]
                ),
            },
            {
                "algorithm": (
                    evolved["algorithm"]
                    if accepted
                    else last_round["candidate_id"]
                ),
                "role": "candidate",
                "metrics": candidate["metrics"],
                "runtime_seconds": candidate["runtime_seconds"],
                "parameter_count": candidate["parameter_count"],
                "eligible_for_final_output": accepted,
                "reconstruction": candidate["artifacts"]["reconstruction"],
                "reconstruction_mat": candidate["artifacts"].get(
                    "reconstruction_mat"
                ),
                "preview": candidate["artifacts"].get(
                    "preview", candidate["artifacts"]["reconstruction"]
                ),
            },
        ]
    )
    return results


def _comparison_image_lines(final_state: Dict[str, Any]) -> List[str]:
    """Render a side-by-side gallery with equal-width image columns."""

    images = final_state.get("artifacts", {}).get("comparison_images", {})
    if not images:
        return []
    run_dir = Path(final_state["artifacts"]["run_dir"])
    ordered = [
        ("corrupted_input", "破损输入"),
        ("interpolation_baseline", "Manhattan 插值"),
        ("implicit_neural_baseline", "SIREN"),
        ("tensor_baseline", "张量基线"),
        (
            "candidate",
            "候选（%s）"
            % ("已接受" if final_state["candidate_accepted"] else "未接受"),
        ),
    ]
    available = [(key, label) for key, label in ordered if key in images]
    paths = []
    for key, _ in available:
        image_path = Path(images[key])
        try:
            image_path = image_path.relative_to(run_dir)
        except ValueError:
            pass
        paths.append(image_path.as_posix())
    column_width = "%.6f%%" % (100.0 / len(available))
    header_cells = "".join(
        '<th width="%s" align="center">%s</th>'
        % (column_width, escape(label))
        for _, label in available
    )
    image_cells = "".join(
        (
            '<td width="%s" align="center" valign="top">'
            '<img src="%s" alt="%s" width="100%%" />'
            "</td>"
        )
        % (column_width, escape(path, quote=True), escape(label, quote=True))
        for (_, label), path in zip(available, paths)
    )
    return [
        "## 效果图对比",
        "",
        '<table width="100%" style="table-layout: fixed; width: 100%;">',
        "  <thead><tr>%s</tr></thead>" % header_cells,
        "  <tbody><tr>%s</tr></tbody>" % image_cells,
        "</table>",
        "",
        "候选图即使未通过 Judge 也会保留，但不会被当作最终可用结果。",
        "",
    ]


def _visual_assessment_lines(
    day5: Dict[str, Any],
    day6: Optional[Dict[str, Any]],
) -> List[str]:
    """Render optional incumbent-versus-candidate visual evidence."""

    if (day5.get("config", {}).get("visual_assessment") is False
            and (not day6 or day6.get("config", {}).get("visual_assessment") is False)):
        return []

    rounds = day6.get("rounds", []) if day6 else []
    visual_rounds = [
        round_record
        for round_record in rounds
        if round_record.get("result_summary", {}).get("visual_assessment")
    ]
    if not visual_rounds:
        return []

    lines = [
        "## 多模态恢复质量观察",
        "",
        "首轮只使用张量基线的数值信息，不进行插值视觉对比。产生候选结果后，视觉模型"
        "仅比较当前最优与候选，并把局部模糊、边界和纹理观察用于下一轮变异；候选是否"
        "晋级仍由固定数值 Judge 决定。",
        "",
    ]
    for round_record in visual_rounds:
        visual = round_record.get("result_summary", {}).get("visual_assessment") or {}
        status = visual.get("status", "not_recorded")
        lines.extend(["### 第 %s 轮" % round_record["round"], "", "- 状态：`%s`" % status])
        if status != "completed":
            lines.extend(["- 原因：%s" % visual.get("reason", "未提供"), ""])
            continue
        assessment = visual["assessment"]
        candidate = assessment["candidate"]
        lines.extend(
            [
                "- 可见图像内容：%s" % assessment["visible_image_content"],
                "- 候选锐度：%s" % candidate["overall_sharpness"],
                "- 模糊/过度平滑：%s" % candidate["blur_and_over_smoothing"],
                "- 物体细节：%s" % candidate["object_detail_clarity"],
                "- 边缘与结构：%s" % candidate["edge_and_structure_continuity"],
                "- 候选对比结论：%s" % assessment["comparison"],
                "- 恢复不佳区域：%s"
                % (
                    "；".join(candidate.get("poorly_recovered_regions", []))
                    or "未观察到"
                ),
                "- 下轮单点变异启示：%s"
                % "；".join(assessment.get("mutation_guidance", [])),
                "- 置信度：`%s`" % assessment["confidence"],
                "",
            ]
        )
    return lines


def render_research_report(
    final_state: Dict[str, Any],
    day4: Dict[str, Any],
    day5: Dict[str, Any],
    day6: Optional[Dict[str, Any]],
) -> str:
    """Render facts from persisted states; do not invent experimental claims."""

    profile = day4["results"]["image_profile"]
    plan = day4["results"]["method_plan"]
    selection_visual = day4["results"].get(
        "method_selection_visual_assessment", {}
    )
    screening = day4["results"].get("method_screening", {})
    siren = day4["results"].get("siren_comparison", {})
    lines = [
        "# Tensor Inpainting Agent 实验报告",
        "",
        "## 结论摘要",
        "",
        "- Run ID：`%s`" % final_state["run_id"],
        "- 输入任务：%s" % final_state["prompt"],
        "- 选择的张量分解：`%s`" % day4["selected_model"],
        "- 分解短名单 / 数值预赛胜者：`%s / %s`"
        % (screening.get("shortlist", []), screening.get("winner", "N/A")),
        "- SIREN 对比：`%s`" % siren.get("status", "not_recorded"),
        "- 选择前视觉 / 变异视觉：`%s / %s`"
        % (
            "on" if final_state["config"].get(
                "selection_visual_assessment", False
            ) else "off",
            "on" if final_state["config"].get(
                "mutation_visual_assessment", False
            ) else "off",
        ),
        "- 候选代码验证：`%s`" % day5["validation"]["status"],
        "- 进化轮数：`%s`" % (len(day6["rounds"]) if day6 else 0),
        "- 接受并替换当前最优的轮次：`%s`"
        % (day6.get("accepted_rounds", []) if day6 else []),
        "- 最终输出算法：`%s`" % final_state["best_available"]["algorithm"],
        "- 最终补全数据：`%s`"
        % final_state["artifacts"].get(
            "best_completion_data", final_state["artifacts"]["best_completion"]
        ),
        "- 最终补全 MAT：`%s`"
        % final_state["artifacts"].get("best_completion_mat", "N/A"),
        "- 最终补全预览：`%s`" % final_state["artifacts"]["best_completion"],
        "",
        "## 数据与缺失模式",
        "",
        "| 项目 | 值 |",
        "|---|---:|",
        "| 数据类型 | `%s` |" % profile.get("data_type", "color_image"),
        "| 张量尺寸 | `%s` |" % profile["image_shape"],
        "| Mask 类型 | `%s` |" % profile["mask_type"],
        "| 实际缺失率 | %s |" % _number(profile["actual_missing_rate"]),
        "| 缺失连通区域数 | %d |" % profile["missing_component_count"],
        "| 可见像素局部平滑度 | %s |"
        % _number(profile["visible_local_smoothness_score"]),
        "",
        "## 方法选择",
        "",
        plan["reason"],
        "",
        "证据：",
        "",
    ]
    lines.extend(
        "- `%s`：%s" % (evidence["source"], evidence["claim"])
        for evidence in plan["evidence"]
    )
    lines.append(
        "- 选择前视觉状态：`%s`" % selection_visual.get("status", "not_recorded")
    )
    if selection_visual.get("status") == "completed":
        visual_prior = selection_visual.get("assessment", {})
        lines.extend(
            [
                "- 视觉建议分解：`%s`"
                % visual_prior.get("preferred_methods", []),
                "- 视觉粗粒度秩先验：`%s`"
                % visual_prior.get("rank_regime", {}),
                "- 视觉置信度：`%s`"
                % visual_prior.get("confidence", "unknown"),
            ]
        )
    if screening.get("status") == "completed":
        lines.extend(
            [
                "- 数值预赛候选：`%s`" % screening.get("shortlist", []),
                "- 预赛选择范围：`%s`"
                % screening.get("selection_scope", "unknown"),
                "- 预赛共享步数 / 每方法 trials：`%s / %s`"
                % (
                    screening.get("shared_max_steps", "N/A"),
                    screening.get("trials_per_method", "N/A"),
                ),
            ]
        )
    lines.extend([""] + _comparison_image_lines(final_state))
    lines.extend(
        [
            "",
            "## 候选研究假设",
            "",
            day5["generation"]["hypothesis"],
            "",
            "- 架构族：`%s`"
            % day5["generation"].get("architecture_family", "tensor_decomposition"),
            "- LLM 请求训练预算：`%s`"
            % day5["generation"].get("training_budget", "legacy/default"),
            "",
            "代码必须先通过 AST 策略检查，以及独立进程中的 forward、backward 和一步优化检查。",
            "",
            "## 最终指标",
            "",
        ]
    )
    full_reference_optional = [("lpips", "LPIPS ↓")]
    no_reference_optional = [
        ("maniqa", "MANIQA ↑"),
        ("clip_iqa", "CLIP-IQA ↑"),
        ("musiq", "MUSIQ ↑"),
    ]
    show_lpips = any(
        "lpips"
        in item["metrics"].get("learned_metric_status", {}).get(
            "requested_metrics", []
        )
        or item["metrics"].get("lpips") is not None
        for item in final_state["method_results"]
    )
    show_no_reference = any(
        item["metrics"].get("metric_group_status", {})
        .get("no_reference", {})
        .get("enabled", False)
        or any(
            item["metrics"].get(key) is not None
            for key, _ in no_reference_optional
        )
        for item in final_state["method_results"]
    )
    columns = [("missing_psnr", "Missing-region PSNR ↑"),
               ("full_psnr", "Full-image PSNR ↑"),
               ("composite_ssim", "Composite SSIM ↑")]
    if show_lpips:
        columns += full_reference_optional
    if show_no_reference:
        columns += no_reference_optional
    headers = ["方法", "角色"] + [label for _, label in columns] + ["最终拟合时间", "参数量", "可作为最终输出"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---", "---"] + ["---:"] * (len(columns) + 2) + ["---"]) + "|")
    for item in final_state["method_results"]:
        runtime = "N/A" if item["runtime_seconds"] is None else _number(item["runtime_seconds"]) + " s"
        cells = [item["algorithm"], item["role"]]
        cells += [_number(item["metrics"].get(key)) for key, _ in columns]
        cells += [runtime, str(item["parameter_count"]), "是" if item["eligible_for_final_output"] else "否"]
        lines.append("| " + " | ".join(cells) + " |")
    if show_lpips or show_no_reference:
        lines.extend([
            "",
            "全参考组包含 MSE、PSNR、SSIM 与 LPIPS；无参考组包含 MANIQA、CLIP-IQA 与 MUSIQ。神经指标仅适用于 RGB `[H,W,3]`，MSI/视频会明确跳过；`N/A` 的具体原因记录在 state JSON 的 `learned_metric_status` 中。",
        ])
    if day6:
        judgment = day6["rounds"][-1]["judgment"]
        lines.extend(
            [
                "",
                "## 多轮算法进化与 Judge",
                "",
                "- 实际轮数 / 请求轮数：`%d / %d`"
                % (len(day6["rounds"]), day6["config"]["max_improvement_rounds"]),
                "- 接受轮次：`%s`" % day6.get("accepted_rounds", []),
                "- 最终进化算法：`%s`"
                % (
                    day6.get("promotion", {}).get("algorithm_name", "未产生")
                    if day6.get("promotion")
                    else "未产生"
                ),
                "- incumbent 调参训练次数：%d"
                % judgment["budget_audit"]["baseline_trial_count"],
                "- candidate 调参训练次数：%d"
                % judgment["budget_audit"]["candidate_trial_count"],
                "- 调优口径：`%s`"
                % judgment["budget_audit"].get(
                    "trial_count_policy", "legacy equal-trial protocol"
                ),
                "- 本轮共享训练配置：`%s`"
                % judgment["budget_audit"].get("shared_training_config", "legacy/default"),
                "- LLM 请求步数 / 用户上限：`%s / %s`"
                % (
                    judgment["budget_audit"].get("llm_requested_max_steps", "N/A"),
                    judgment["budget_audit"].get("user_max_steps_ceiling", "N/A"),
                ),
                "- Missing-region PSNR 差值：%s dB"
                % _number(judgment["psnr_delta"]),
                "- Composite SSIM 差值：%s" % _number(judgment["ssim_delta"]),
                "- 总运行时间比：%s" % _number(judgment["runtime_ratio"]),
                "- 决策：`%s`" % judgment["decision"],
                "- 停止原因：`%s`" % day6["stop_reason"],
                "",
                "模型梯度始终只由全部可见像素计算；缺失区 Ground Truth MSE 直接选择结构、学习率、早停点和最终输出。该结果属于单图 oracle 搜索，不代表未见数据泛化性。",
                "每轮只允许一个算法或 loss 变异点；接受后更新当前最优并继续，而不是提前终止。",
            ]
        )
    visual_lines = _visual_assessment_lines(day5, day6)
    if visual_lines:
        lines.extend([""] + visual_lines)
    lines.extend(
        [
            "",
            "## 可审计产物",
            "",
            "- 方法选择状态：`%s`" % final_state["artifacts"]["day4_state"],
            "- 候选生成状态：`%s`" % final_state["artifacts"]["day5_state"],
            "- 公平实验状态：`%s`"
            % final_state["artifacts"].get("day6_state", "not generated"),
            "- 当前运行实践文档：`%s`"
            % (
                day6.get("artifacts", {}).get("run_practice", {}).get(
                    "practice_markdown", "not generated"
                )
                if day6
                else "not generated"
            ),
            "- 跨运行可复用经验文档：`%s`"
            % (
                day6.get("artifacts", {}).get("global_experience", {}).get(
                    "experience_markdown", "not generated"
                )
                if day6
                else "not generated"
            ),
            "- 顶层 Trace：`%s`" % final_state["artifacts"]["trace_jsonl"],
            "",
            "## 证据边界",
            "",
            "本报告只证明该工作流在当前图片、mask、缺失率、随机种子和预算下得到上述结果。单张图片上的晋升不能推出跨数据集优势，也不构成 SOTA 声明。",
            "",
        ]
    )
    return "\n".join(lines)


def write_research_report(path: str, *args: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_research_report(*args), encoding="utf-8")
