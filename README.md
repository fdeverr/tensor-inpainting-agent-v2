# Tensor Inpainting Agent

一个支持彩图、MSI 与视频多维数据的张量补全研究 Agent。它集成 Matrix、A×₃E、CP、Nonnegative CP、Tucker、BTD、t-SVD、Nonnegative Tucker、Hierarchical Tucker、Tensor Train 和 Tensor Ring，并能自主生成连续 MLP、卷积、高效 Transformer 与混合 PyTorch 候选。

## 目录结构

```text
tensor_inpainting_agent/
├── agents/、core/、tools/、context/  # Agent 框架
├── observability/、skills/           # 追踪与技能支持
└── research_agent/                   # 图像补全研究应用
```

## 快速开始

在本目录中执行：

```bash
python -m pip install -r research_agent/requirements.txt
cp research_agent/.env.example research_agent/.env
```

填写 `research_agent/.env` 后，可以选择三种运行模式：

```bash
# 快速链路检查，最长边 64
./research_agent/scripts/run_ai.sh smoke

# 标准实验，最长边 128
./research_agent/scripts/run_ai.sh full

# 手动指定 TT 分解（也可选其他已支持分解）
./research_agent/scripts/run_ai.sh full --base-model tt

# 手动指定 X = A×₃E 通道模分解
./research_agent/scripts/run_ai.sh full --base-model mode3

# 手动指定 Block-Term Decomposition
./research_agent/scripts/run_ai.sh full --base-model btd

# 标准实验预算，保留原始分辨率
./research_agent/scripts/run_ai.sh original
```

自定义图片：

```bash
./research_agent/scripts/run_ai.sh original \
  --image path/to/image.png
```

MAT 数据可直接输入；MSI/彩图形状为 `[H,W,C]`，视频形状为 `[H,W,T,C]`：

```bash
./research_agent/scripts/run_ai.sh original \
  --image /data/video.mat \
  --mat-key video
```

省略 `--mat-key` 时会优先查找 `data/tensor/image/msi/video/X/x`，否则自动选择最大的合法三维或四维数值变量。空间 mask 为 `[H,W]`，会同时作用于全部波段、帧和通道。

完整参数说明：

```bash
./research_agent/scripts/run_ai.sh --help
```

每次完整运行都会同时保存无损多维 `.npy` 和 `.mat` 补全数据（MAT 变量名为 `data`），并在 `comparison_images/` 中导出 RGB 预览。评价默认采用全参考组 MSE、PSNR、SSIM 和 LPIPS；MANIQA、CLIP-IQA、MUSIQ 属于可选的无参考组。四个神经指标仅适用于 RGB，MSI/视频会明确标记为跳过。SIREN 默认作为独立 INR 基线：同一份损坏数据、mask、模型代码和训练配置只调参并训练一次，后续运行直接复用指纹缓存，再与 Manhattan 插值、张量基线和进化候选进入最终对比。

胜出的张量家族会先将秩/结构参数与 `0.001、0.01、0.1` 作联合粗搜，再固定粗搜胜出的秩/结构，对其学习率追加 `÷3`、`×3` 精搜。标准预设每个 trial 先训练 1500 步；若当前全局最佳 checkpoint 位于上限的后 10% 且没有早停，只对该配置从头重跑并将预算翻倍，最多扩展到 6000 步。最终重训仍使用选中的秩、学习率和 `best_step`，在全部已观测像素上从头训练。

算法进化默认执行 5 轮，可用 `--max-improvement-rounds N` 调整。多模态观察拆成两个独立的消融开关：`--selection-visual-assessment` 在张量分解选择前只观察 Manhattan 插值恢复图，用于生成候选短名单；`--mutation-visual-assessment` 在已有候选结果后比较当前最优与候选。无论是否开启视觉观察，候选生成都会收到一份结构化算法参考面板，包含 Manhattan 插值、SIREN、选中张量基线、张量家族预赛及最新 incumbent/candidate 的效果与差距。这些聚合指标作为开发反馈指导方向，但不替代固定 Judge，也不暴露 Ground Truth 张量内容。每轮先诊断痛点、核心难点并把难点简化为可验证问题；默认做一个原子改动，也允许最多三个带独立开关的组件。每个组件明确声明 `add/modify/remove/retain`，删除操作必须给出消融或失败证据、目标模块和删除类型。组合候选会在同一候选类上以开关全关作为 Base，自动运行完整 `2^N` 消融，再将验证集胜出的非基线组合与 incumbent 做完整公平实验。候选方向显式包含内存受控的自注意力、交叉注意力、多尺度空洞卷积以及它们与张量分解的混合结构。

Day 6 会在首轮锁定全程公平协议，对初始 incumbent 和首个候选分别使用相同的 `0.001、0.01、0.1` 学习率粗搜网格，再围绕各自胜出值做 `÷3`、`×3` 精搜。一个算法一旦在该协议下完成调参和最终重训，若它下轮仍是 incumbent，则直接复用已有结果，不会再训练。后续候选的预算建议只作记录，不改变已锁定的协议。LLM 变异以“相对当前 incumbent 提升 Missing-region PSNR”为唯一主目标，Composite SSIM 只是 Judge 的容差约束。参考面板中的插值、SIREN 和张量家族结果用于诊断差距和借鉴归纳偏置，但不取代当前 incumbent 目标。LPIPS 作为越低越好的感知质量诊断；无参考指标仍只用于最终报告。

更多设计、实验协议与测试说明见 [research_agent/README.md](research_agent/README.md)。
