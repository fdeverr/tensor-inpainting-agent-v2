"""Markdown reporting for a complete, auditable research run."""

from __future__ import annotations

import ast
from html import escape
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return ("%%.%df" % digits) % float(value)


BUILTIN_FRAMEWORKS = {
    "nearest_neighbor_manhattan": "Manhattan 最近邻插值：对每个缺失位置寻找曼哈顿距离最近的可见像素，复制其特征值；无需训练。",
    "matrix": "低秩矩阵分解：将张量展开为 H × (W·F)，用两个可学习因子 U、V 的矩阵乘积重建，再恢复原始形状并加特征偏置。",
    "mode3": "Mode-3 分解：学习空间系数 A[H,W,R] 与特征因子 E[F,R]，沿秩维收缩得到完整张量，再加特征偏置。",
    "cp": "CP 分解：学习高度、宽度、特征三个因子，将 R 个秩一张量相加得到重建，再加特征偏置。",
    "nonnegative_cp": "非负 CP 分解：用非负参数化约束因子，以秩一张量的求和构造非负重建。",
    "tucker": "Tucker 分解：学习核心张量 G 和高度、宽度、特征因子 U、V、E；通过 G ×₁ U ×₂ V ×₃ E 重建，再加特征偏置。",
    "nonnegative_tucker": "非负 Tucker 分解：对核心和各模态因子采用 softplus 非负参数化，通过核心与三组因子的收缩生成重建。",
    "hierarchical_tucker": "层次 Tucker 分解：以高度、宽度叶因子和空间转移张量组织空间低秩表示，再通过根核心与特征因子组合重建。",
    "btd": "块项分解：每个块使用核心张量和高度、宽度、特征因子，重建各块后相加并加入特征偏置。",
    "tsvd": "t-SVD 风格低秩模型：在特征轴的频域学习低秩因子，频域矩阵乘积后通过逆实数 FFT 恢复张量，再加特征偏置。",
    "tt": "Tensor Train 分解：学习三个链式核心，沿相邻核心之间的秩维收缩，得到完整张量并加入特征偏置。",
    "tensor_ring": "Tensor Ring 分解：学习高度、宽度、特征三个环状核心，闭合收缩环上的秩维，得到重建并加入特征偏置。",
    "siren": "SIREN 隐式神经表示：归一化二维坐标 → 正弦激活的全连接隐藏层 → 线性特征输出 → 恢复图像/张量形状。",
}


def _best_algorithm_lines(
    final_state: Dict[str, Any], day4: Dict[str, Any],
    day6: Optional[Dict[str, Any]],
) -> List[str]:
    """Describe the actual winner, not the last proposed (possibly rejected) model."""
    winner = final_state["best_available"]
    algorithm = winner["algorithm"]
    audio = "missing_nmse" in winner.get("metrics", {})
    dataset_mode = bool(final_state.get("config", {}).get("evolution_cases"))
    lines = ["## 最佳算法与模型框架", "", "- 最佳算法：`%s`" % algorithm,
             "- 选择依据：每轮在该类全部样本上比较 incumbent 与 candidate，按平均缺失波形 NMSE 和完整性门槛决定是否接受；输出最后的进化 incumbent。" if dataset_mode and audio else
             "- 选择依据：每轮在该类全部样本上比较 incumbent 与 candidate，按平均 Missing-region PSNR、平均 SSIM 和完整性门槛决定是否接受；输出最后的进化 incumbent。" if dataset_mode else
             "- 选择依据：选择缺失原始波形 NMSE 最低者；进化候选须通过 NMSE Judge。" if audio else
             "- 选择依据：在允许作为最终输出的方法中，选择 Missing-region PSNR 最高者；进化候选须先通过固定 Judge。",
             "- 参数量：`%s`" % winner.get("parameter_count", "N/A"), ""]
    evolved = (day6 or {}).get("best_evolved") or {}
    if algorithm == evolved.get("algorithm") and winner.get("role") == "candidate":
        description = evolved.get("model_description", {})
        accepted = [r for r in (day6 or {}).get("rounds", [])
                    if r.get("judgment", {}).get("accepted")]
        record = accepted[-1] if accepted else {}
        proposal = description.get("proposal") or record
        lines += ["### 模型设计", "",
                  "- 架构族：`%s`" % proposal.get("architecture_family", "未记录"),
                  "- 基础方法：`%s`（进化后的具体实现以冠军代码为准）" % day4["selected_model"],
                  "- 设计思路：%s" % proposal.get("idea", "未记录"),
                  "- 变异说明：%s" % proposal.get("single_change", "未记录")]
        for component in proposal.get("components", []):
            lines.append("- 组件 `%s`：%s；预期作用：%s"
                         % (component.get("id", "?"), component.get("change", "未记录"),
                            component.get("expected_role", "未记录")))
        lines += ["", "以上设计说明来自候选提案，不等同于消融验证结论；实际启用的组件由最佳配置中的开关决定。"]
        config = description.get("selected_config", {})
        source = description.get("source_path")
        code = description.get("source_code")
        if source:
            lines += ["", "冠军模型实现：`%s`" % source]
        if code:
            try:
                tree = ast.parse(code)
                forward_sections = [
                    (node.name, ast.get_source_segment(code, method))
                    for node in tree.body if isinstance(node, ast.ClassDef)
                    for method in node.body
                    if isinstance(method, ast.FunctionDef) and method.name == "forward"
                ]
            except SyntaxError:
                forward_sections = []
            if forward_sections:
                lines += ["", "### 前向重建框架", "",
                          "以下摘录冠军代码中的实际 forward；内部模块的初始化与具体实现见下方完整代码。"]
                for name, segment in forward_sections:
                    lines += ["", "`%s`：" % name, "", "```python", segment or "", "```"]
            lines += ["", "<details>", "<summary>冠军模型完整实现</summary>",
                      "", "```python", code.rstrip(), "```", "", "</details>"]
    else:
        lines += [BUILTIN_FRAMEWORKS.get(algorithm, "该算法的框架说明未记录，请查阅模型实现。")]
        if algorithm == "siren":
            config = day4.get("results", {}).get("siren_comparison", {}).get("training", {})
        elif winner.get("role") == "tensor_baseline":
            config = day4.get("results", {}).get("training", {})
            if day6 and day6.get("rounds"):
                config = day6["rounds"][0].get("baseline_tuning", {}).get("best", config)
        else:
            config = {}
    selected = {key: config[key] for key in
                ("hyperparameters", "learning_rate", "best_step", "fitted_steps") if key in config}
    if selected:
        lines += ["", "### 最佳配置", "", "```json",
                  json.dumps(selected, ensure_ascii=False, indent=2), "```"]
    if algorithm != "nearest_neighbor_manhattan":
        lines += ["", "训练与输出流程：波形张量预测 → 可见采样点损失反向传播 → 缺失原始波形 GT NMSE 选择 checkpoint → 直接复用 → 恢复已观测采样点。" if audio else
                  "训练与输出流程：完整张量预测 → 全部可见像素上的损失反向传播 → 缺失区 GT MSE 选择最佳 checkpoint → 直接复用该 checkpoint 的预测 → 恢复已观测像素，输出完整补全结果。",
                  "GT 用于参数/检查点选择，不进入梯度损失；音频效果仅用缺失原始波形 NMSE 评价。" if audio else
                  "GT 用于参数/检查点选择，不进入梯度损失；PSNR、SSIM 用于效果评估。"]
    return lines + [""]


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
                ("- LLM 推荐方法：`%s`" % screening["selector_recommended_methods"] if
                 screening.get("selector_recommended_methods") else
                 "- 规则回退首选：`%s`" % screening.get("selector_recommendation", "N/A")),
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
        if screening.get("selection_scope") == "all_valid_samples_of_modality":
            audio_screen = screening.get("selection_metric") == "mean_missing_nmse"
            lines.extend(["", "### 基础方法整类轻量预赛", "",
                          "代表样本用于等量调参；各方法最佳配置随后冻结，在同类全部样本上独立拟合。"
                          "只有全部样本成功的方法才能胜出。", ""])
            if audio_screen:
                lines.extend(["| 方法 | 成功/总数 | 平均 NMSE ↓ | 状态 |",
                              "|---|---:|---:|---|"])
            else:
                lines.extend(["| 方法 | 成功/总数 | 平均 PSNR ↑ | 平均 SSIM ↑ | 状态 |",
                              "|---|---:|---:|---:|---|"])
            for result in screening.get("results", []):
                summary = (result.get("dataset_evaluation") or {}).get("summary", {})
                name = result["method"]
                if name == screening.get("winner"):
                    name = "**%s**" % name
                count = "%s/%s" % (summary.get("completed_count", 0),
                                    summary.get("expected_count", "?"))
                if audio_screen:
                    lines.append("| %s | %s | %s | %s |" % (
                        name, count, _number(summary.get("mean_missing_nmse"), 6),
                        result.get("status", "unknown")))
                else:
                    psnr = ("∞" if summary.get("complete") and summary.get("perfect_count") else
                            _number(summary.get("mean_missing_psnr")))
                    lines.append("| %s | %s | %s | %s | %s |" % (
                        name, count, psnr, _number(summary.get("mean_composite_ssim")),
                        result.get("status", "unknown")))
            lines.append("")
    lines.extend([""] + _best_algorithm_lines(final_state, day4, day6))
    lines.extend(_comparison_image_lines(final_state))
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
    audio = any("missing_nmse" in item["metrics"] for item in final_state["method_results"])
    if audio:
        columns = [("missing_nmse", "Missing waveform NMSE ↓")]
    if final_state.get("config", {}).get("evolution_cases"):
        lines.extend(["", "下表为结构搜索样本上的诊断指标；每轮整类评分、逐样本结果和最终整类版本比较见外层 recovery 报告。", ""])
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
        cells += [("N/A" if item["metrics"].get(key) is None else "%.6g" % item["metrics"][key]) if audio else _number(item["metrics"].get(key)) for key, _ in columns]
        cells += [runtime, str(item["parameter_count"]), "是" if item["eligible_for_final_output"] else "否"]
        lines.append("| " + " | ".join(cells) + " |")
    if show_lpips or show_no_reference:
        lines.extend([
            "",
            "MSE、PSNR、SSIM 始终计算；LPIPS 属于可选全参考指标，默认关闭。无参考组包含 MANIQA、CLIP-IQA 与 MUSIQ。神经指标仅适用于 RGB `[H,W,3]`，MSI/视频会明确跳过；`N/A` 的具体原因记录在 state JSON 的 `learned_metric_status` 中。",
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
                ("- Missing waveform NMSE 降低量：%s" % _number(judgment["nmse_delta"])) if audio else
                "- Missing-region PSNR 差值：%s dB" % _number(judgment["psnr_delta"]),
                "音频仅使用原始幅值波形的缺失区 NMSE 评价，不计算 PSNR/SSIM。" if audio else
                "- Composite SSIM 差值：%s" % _number(judgment["ssim_delta"]),
                "- 总运行时间比：%s" % _number(judgment["runtime_ratio"]),
                "- 决策：`%s`" % judgment["decision"],
                "- 停止原因：`%s`" % day6["stop_reason"],
                "",
                ("模型梯度始终只由全部可见像素计算；研发样本 GT 用于结构/超参数与 checkpoint 搜索，各样本缺失区 GT 指标用于每轮整类算法选择。这是 oracle 开发评测，不代表未见数据泛化性。"
                 if final_state.get("config", {}).get("evolution_cases") else
                 "模型梯度始终只由全部可见像素计算；缺失区 Ground Truth MSE 直接选择结构、学习率、早停点和最终输出。该结果属于单图 oracle 搜索，不代表未见数据泛化性。"),
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
