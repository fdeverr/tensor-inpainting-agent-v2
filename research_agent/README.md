# Tensor Inpainting Agent

一个基于 Tensor Inpainting Agent Framework 的多维张量补全研究 Agent：它支持彩图、MSI 与视频，分析缺失数据、检索方法经验、从 11 种张量分解基线中自动或手动选择，再由 LLM 设计连续因子 MLP、卷积解码器、高效 Transformer、残差结构或混合 PyTorch 候选，并通过无 Ground Truth 泄漏的公平实验决定接受、拒绝或继续修改候选。

项目目标不是宣称 SOTA，而是展示一条**可验证、可证伪、可复现、可审计**的自主算法研究闭环。

## 30 秒架构

```text
固定研究 Prompt + 图像
          │
          ▼
Tensor Inpainting Agent Framework 编排层
  ├─ Image Profiler ────────────── 只看 corrupted image + mask
  ├─ Knowledge Retriever ───────── 检索本地 11 种分解的经验与限制
  ├─ Method Selector ───────────── LLM 或确定性 fallback
  ├─ Model Improver ────────────── 输出结构化 idea + 候选模型代码
  └─ Experiment Controller ─────── 默认五轮单点变异，记录完整 Trace
          │
          ▼
固定实验内核
  ├─ AST + 独立进程 smoke validation
  ├─ 同预算 paired tuning：M_train 训练，M_val 选优
  ├─ 全部观测像素 final refit
  ├─ 最终 PSNR / SSIM / LPIPS / MANIQA / CLIP-IQA / MUSIQ
  └─ 固定 Judge + approved algorithm registry
```

核心依赖方向始终是 `Agent → 实验内核`。训练器、指标和 Judge 不依赖 LLM，LLM 也不能修改评估规则或自行宣布成功。

## 为什么使用 Tensor Inpainting Agent Framework

Tensor Inpainting Agent Framework 在这里提供 Tool、ToolRegistry、LLM 适配和 TraceLogger。项目保留它的编排与可观测性能力，但不修改框架源码；所有图像补全算法都位于独立目录中。这样可以展示 Agent 工程能力，同时保持实验内核可独立测试，也方便以后替换模型服务或 Agent 框架。

## Quick start（Linux）

在 `tensor_inpainting_agent` 仓库根目录安装依赖并准备 `.env` 后，可使用统一脚本运行：

```bash
chmod +x research_agent/scripts/run_ai.sh

# 查看模式、图像尺寸和参数说明
./research_agent/scripts/run_ai.sh --help

# 快速验证 API、CUDA 和完整 Agent 链路
./research_agent/scripts/run_ai.sh smoke

# 使用标准实验预设
./research_agent/scripts/run_ai.sh full

# 跳过自动方法排名，手动指定 TT 基线
./research_agent/scripts/run_ai.sh full --base-model tt

# 手动指定 X = A×₃E 通道模分解
./research_agent/scripts/run_ai.sh full --base-model mode3

# 使用标准实验预算并保留原始分辨率
./research_agent/scripts/run_ai.sh original \
  --image path/to/image.png
```

`smoke` 将图像最长边限制为 64 像素，并减少训练、调参和改进轮数，只用于快速检查系统是否跑通。`full` 默认将最长边限制为 128 像素，方法选择最多训练 1000 步，Day 6 允许 LLM 在 2000 步硬上限内提议训练步数、验证间隔和早停耐心值。`original` 使用与 `full` 相同的实验预算，但不缩放输入图像。三种模式都保留原始宽高比，不会强制图像变为正方形。

### 多维数据与 MAT 输入

支持的内部形状如下：

- 彩图和 MSI：`[H,W,C]`
- 视频：`[H,W,T,C]`
- 空间观测 mask：`[H,W]`，自动广播到所有波段、帧和通道

普通图片会转为 RGB。MATLAB `.mat` 文件应包含实数数值三维或四维数组，可显式指定变量：

```bash
./research_agent/scripts/run_ai.sh original \
  --image /data/msi.mat \
  --mat-key cube

./research_agent/scripts/run_ai.sh original \
  --image /data/video.mat \
  --mat-key video
```

省略 `--mat-key` 时，系统依次尝试 `data`、`tensor`、`image`、`msi`、`video`、`X`、`x`，再确定性选择元素最多的合法变量。整数及超出 `[0,1]` 的浮点数据会按全局最小值/最大值归一化，原始类型、范围、变量名和归一化方式写入 `image_profile.json`。MATLAB v7.3 文件通过 `h5py` 读取。

所有 11 种现有三阶分解将尾部维度展平为联合特征模：MSI 为 `C`，视频为 `T×C`；训练后再恢复原始形状。这样同一算法接口可以处理两种数据，而空间掩码语义保持不变。完整重建同时保存为 `.npy` 和 `.mat`（MAT 变量名统一为 `data`），PNG 仅用于预览：视频取中间帧，MSI 超过 3 个波段时使用 `[末波段, 中间波段, 首波段]` 假彩色映射。

基础分解默认为 `--base-model auto`，由检索增强选择器在 Matrix、A×₃E、CP、Nonnegative CP、Tucker、BTD、t-SVD、Nonnegative Tucker、Hierarchical Tucker、TT 和 Tensor Ring 中决定。要做可控对比实验，可显式传入对应名称：`matrix|mode3|cp|nonnegative_cp|tucker|btd|tsvd|nonnegative_tucker|hierarchical_tucker|tt|tensor_ring`；仍会执行该方法的有界调参，只跳过方法排名。

可以针对不同图像显式设置最长边：

```bash
./research_agent/scripts/run_ai.sh full \
  --image research_agent/assets/example.png \
  --image-size 256
```

要保留原始分辨率，可以直接选择 `original` 模式，也可以在其他模式后使用 `--image-size original`。高分辨率图像会显著增加张量分解的参数量、显存占用和训练时间，因此建议先使用 `smoke`，再根据 GPU 情况逐步增大到 128、256 或原图。

所有常用实验参数都可在命令行中覆盖，脚本启动时会打印最终实际使用的值：

```bash
./research_agent/scripts/run_ai.sh full \
  --image path/to/image.png \
  --image-size original \
  --mask-type random \
  --missing-rate 0.3 \
  --method-max-steps 1500 \
  --fair-max-steps 3000 \
  --tuning-trials 3 \
  --device cuda
```

统一脚本默认使用 `--llm-mode required --device cuda`，并从 `research_agent/.env` 加载 LLM 配置。传入 `--llm-mode off` 时不再依赖 `.env`，并使用确定性选择器和 TV 候选模板；`auto` 会在环境变量完整时创建 LLM 客户端，否则回退到确定性规则。

默认同时计算 LPIPS、MANIQA、CLIP-IQA 和 MUSIQ。这些指标通过 PyIQA 在已知像素恢复后的完整复合图像上计算：LPIPS 是全参考感知距离（越低越好），MANIQA、CLIP-IQA、MUSIQ 是无参考质量分数（越高越好）。它们的预训练输入仅支持 RGB，因此 MSI/视频会在 `learned_metric_status.skipped_reason` 中明确说明跳过；PSNR 与 SSIM 仍在全部波段/帧/通道上计算。首次 RGB 运行会下载预训练权重；无网络或单个权重不可用时，该项记为 `N/A`、错误写入 `learned_metric_status.errors`。可用 `--learned-metrics off` 显式跳过这些较重的指标。

### 指标协议

| 指标 | 类型 | 范围 | 方向 | 作用 |
|---|---|---|---|---|
| Missing PSNR | 全参考 | 仅缺失区域 | ↑ | 像素级保真度与主排序 |
| Composite SSIM | 全参考 | 复合整图 | ↑ | 局部结构一致性 |
| LPIPS | 全参考 | 复合整图 | ↓ | 深度特征感知距离 |
| MANIQA | 无参考 | 复合整图 | ↑ | Transformer 感知质量 |
| CLIP-IQA | 无参考 | 复合整图 | ↑ | 视觉-语言特征质量 |
| MUSIQ | 无参考 | 复合整图 | ↑ | 多尺度感知质量 |

新增的四项指标只用于最终诊断和 benchmark 汇总，不进入训练、调参、Judge 或冠军选择；Ground Truth 仍只在训练和超参选择结束后进入评估函数。

运行时会显示当前工作流阶段、LLM 候选生成/修复轮次、自动调参 trial 以及最终重训状态。交互式终端中的训练进度条会在同一行刷新，包含 `当前步/总步数`、百分比、train/validation loss 和 ETA；输出被重定向到日志时则每 10% 保留一条记录。

```text
▶ 公平实验 candidate trial 2/4（共 1500 步）
[██████░░░░░░░░░░░░░░░░░░] 375/1500 25.00% loss=0.0121 val=0.0134 ETA 1m42s  训练中
```

## 支持的方法

| 类型 | 方法 | 可学习参数 |
|---|---|---|
| 无训练基线 | Manhattan 最近邻插值 | 无 |
| 张量基线 | Matrix Factorization | 两个矩阵因子与通道偏置 |
| 张量基线 | A×₃E (Mode-3) | 空间系数张量 A、通道因子 E 与通道偏置 |
| 张量基线 | CP Decomposition | 三个模态因子与通道偏置 |
| 张量基线 | Nonnegative CP | Softplus 约束的三个非负模态因子 |
| 张量基线 | Tucker Decomposition | 核张量、三个因子与通道偏置 |
| 张量基线 | Block-Term Decomposition (BTD) | 多个 Tucker 块的核、因子与通道偏置 |
| 张量基线 | t-SVD | Fourier 域左右因子、奇异管与通道偏置 |
| 张量基线 | Nonnegative Tucker | Softplus 约束的非负核与因子 |
| 张量基线 | Hierarchical Tucker | 叶因子、空间转移核和根转移核 |
| 张量基线 | Tensor Train (TT) | 三个链式核、两个内部 TT rank 与通道偏置 |
| 张量基线 | Tensor Ring (TR) | 三个环式核、环秩与通道偏置 |
| LLM 候选 | 连续因子 / 坐标 MLP | MLP 根据坐标生成矩阵或张量因子 |
| LLM 候选 | Convolutional decoder / refiner | 从头训练的卷积解码或残差精修分支 |
| LLM 候选 | Efficient Transformer | 轴向、窗口、patch 或 latent-token 注意力，禁止全像素二次方注意力 |
| LLM 候选 | Hybrid / residual | 张量分解与 MLP、卷积、注意力或门控残差结合 |
| 当前 fallback 候选 | Selected decomposition + Total Variation | 保留已选择分解的参数，增加固定形式的 TV 损失 |

候选可继承已选分解并添加深度分支，也可直接继承 `BaseTensorInpaintingModel` 实现全新的每图像 PyTorch 参数化。它们不能生成或替换训练流程，不能使用预训练权重或外部数据。

单独运行 Day 2 时可直接选择新模型：

```bash
python -m research_agent.run_day2 --image path/to/image.png --model mode3 --rank 2
python -m research_agent.run_day2 --image path/to/image.png --model nonnegative_cp --rank 12
python -m research_agent.run_day2 --image path/to/image.png --model btd --num-blocks 2 --rank-h 8 --rank-w 8 --rank-c 2
python -m research_agent.run_day2 --image path/to/image.png --model tsvd --rank 8
python -m research_agent.run_day2 --image path/to/image.png --model nonnegative_tucker --rank-h 8 --rank-w 8 --rank-c 2
python -m research_agent.run_day2 --image path/to/image.png --model hierarchical_tucker --rank-h 8 --rank-w 8 --rank-c 3 --rank-spatial 2
python -m research_agent.run_day2 --image path/to/image.png --model tt --rank-1 8 --rank-2 3
python -m research_agent.run_day2 --image path/to/image.png --model tensor_ring --rank 4
```

## 无 Ground Truth 泄漏协议

设 `M` 为真实可见像素。调参时再将它确定性划分为：

```text
M = M_train ∪ M_val，M_train ∩ M_val = ∅
```

- 梯度更新只使用 `M_train`；
- 超参数与训练步数只根据 `M_val` 的 MSE 选择；
- 选定配置后重置模型，并在整个 `M` 上重新拟合；
- 人工遮挡区域的 Ground Truth 只在最终评估函数中出现；
- 调参函数的接口中根本不存在 `ground_truth` 参数。

LLM 可在 `--fair-max-steps` 用户硬上限内请求每轮的最大步数、验证间隔和早停耐心值。解析后的训练预算会同时用于基础模型和候选模型；trial 数、随机种子、数据划分、学习率、优化器和设备也保持相同。值域完全相同的超参数按 trial 直接配对；hybrid 候选若为同名 rank 提出不同合法值域，则将该参数视为架构专属参数独立调优。因此候选不能单纯靠“多训几步”获得优势。

## 候选代码验证与晋升

候选进入训练前必须依次通过：

1. Pydantic `CandidateProposal` 结构验证；
2. AST 语法、导入、危险调用、父类与接口检查；
3. 独立进程中的非方形动态尺寸构造、forward、有限值、参数量上限、backward、梯度与一步优化检查；
4. manifest 状态和 SHA-256 完整性检查；
5. 与基础张量模型的同预算公平实验。

如果首个 LLM 候选在 AST 或独立进程检查中失败，Day 5 和 Day 6 的改进轮都会保留原候选与验证报告，再将精确的工具反馈与原代码交给 LLM 修复一次。若两轮 LLM 候选仍未通过，系统会验证一次内置的确定性安全候选。Day 6 中即使安全候选也失败，也会保留已完成轮次、正常生成报告，不会因候选代码问题丢失整次运行。验证器规则本身不会交给 LLM 修改。

固定 Judge 的默认准入条件：

```text
missing PSNR delta >= +0.2 dB
composite SSIM delta >= -0.002
trial 数相同，训练过程无 NaN / Inf / 异常
最终全观测像素重训未发生明显数值发散
```

完成指定轮数后，最终 incumbent 及 idea、验证报告、最佳配置、指标、适用条件、来源 run 和代码哈希会进入 `algorithms/approved/<base-method>/<name>/<version>/`。所有已晋升算法共用一个 `AlgorithmRunnerTool`，不会为每个候选复制训练代码。

## 进化记忆的生命周期

- 当前运行实践：`outputs/<day6-run-id>/knowledge/practice.md` 和 `practice.jsonl`。只供同一次运行中的后续轮次使用。
- 全局可复用经验：`algorithms/evolution_knowledge/<base-method>/reusable_experience.md` 和 `reusable_experience.jsonl`。跨运行持续追加，新的独立运行会读取它。
- 全局经验每条只保存从“目标—方法—结果”中提炼的一般性 `experience` 和 `confidence`；不保存运行号、轮次、候选 ID、证据分段或下一轮指令。完全重复的经验不会再次追加。
- 旧版可能产生的全局 `practice.*`/`experience.*` 不再读取或追加，避免把历史运行轨迹混入新运行上下文。

## 一次真实研究轨迹

固定 Prompt：

> 请分析图像和缺失模式，选择合适的张量分解，并在公平实验下提出、验证和改进一个补全算法。

在 `example.png`、block mask、40% 缺失、seed 42、CPU、每方 4 trials × 200 steps 下：

```text
可见特征分析
→ 本地知识检索选择 Tucker
→ 生成 Tucker + TV 假设
→ AST 与训练 smoke test 通过
→ 基础 Tucker / 候选各执行 4 个配对 trials
→ 候选相对 Tucker：PSNR +0.7230 dB，SSIM +0.0818
→ Judge 接受并晋升候选
→ 全体方法比较后，最终仍输出指标更高的最近邻插值
```

“候选被晋升”和“候选是全体冠军”是两个不同问题。Agent 不会因为自己提出了改进，就隐藏更强的简单基线。

## 真实结果

| 方法 | Missing PSNR ↑ | Composite SSIM ↑ | 最终拟合时间 | 参数量 | 结论 |
|---|---:|---:|---:|---:|---|
| 最近邻插值 | 15.0618 | 0.7271 | N/A | 0 | 全体冠军 |
| 基础 Tucker | 12.7903 | 0.6269 | 0.0985 s | 8,201 | 候选比较基线 |
| Tucker + TV | 13.5133 | 0.7087 | 0.2882 s | 8,201 | 相对 Tucker 通过晋升 |

| Corrupted image | 插值结果 | Tucker + TV |
|---|---|---|
| ![corrupted](docs/demo/corrupted.png) | ![interpolation](docs/demo/interpolation.png) | ![tucker-tv](docs/demo/tucker_tv.png) |

完整机器生成结果见 [`docs/demo/results.json`](docs/demo/results.json)。

## Benchmark

Benchmark 命令支持多图片、多个 mask 和多个缺失率：

```bash
python -m research_agent.run_benchmark \
  --images /data/image1.png /data/image2.png /data/image3.png \
  --mask-types random block \
  --missing-rates 0.3 0.5 \
  --device cuda
```

当前仓库只有用户提供的一张合法图片，因此只执行了 `1 image × 2 masks = 2 cases` 的 smoke benchmark：2/2 case 成功，候选晋升 1 次。该结果只验证端到端稳定性，不作为跨图像泛化证据。正式求职展示前，建议补充至少 3～5 张有明确使用权的测试图片。

本次聚合表见 [`docs/demo/benchmark.md`](docs/demo/benchmark.md)。

## 输出与可审计性

每次完整运行至少生成：

```text
outputs/research-agent-<run-id>/
├── state.json
├── report.md
├── best_completion.png
├── best_completion.npy           # 完整 HWC/HWTC 张量
├── best_completion.mat           # 同形状 MAT，变量名 data
├── comparison_images/
│   ├── 00_corrupted_input.png
│   ├── 01_manhattan_interpolation.png
│   ├── 02_tensor_baseline.png
│   └── 03_candidate.png          # 候选被评估时生成
└── traces/
    ├── trace-*.jsonl
    └── trace-*.html
```

Day 4/5/6 子流程分别保存自己的状态、图片、配置、训练曲线、checkpoint 和 Trace；顶层状态只保存结构化摘要与子流程引用。

## 测试

```bash
pytest -q
```

测试集覆盖 mask、PSNR/SSIM 与学习式 IQA 指标、11 种张量模型、两阶段训练、终端进度显示、Tensor Inpainting Agent Framework Tools、方法选择、深度候选接口、LLM 训练预算限幅、hybrid 同名 rank 独立调参、Day 5/6 候选代码验证、反馈修复与安全回退、最终重训发散诊断、效果图对比导出、公平配对、Judge、停止条件、统一入口和 benchmark 聚合。

## 已知限制

- 当前真实结果只来自一张图片，不能证明跨数据集优势。
- `composite_ssim` 是项目定义的复合 SSIM，不是标准 masked SSIM。
- AST 与独立进程 smoke test 是学习项目级防护，不是生产安全沙箱。
- 当前无 LLM 时的 fallback improver 仍只实现 TV 正则；真实 LLM 可生成 MLP/CNN/Transformer/混合架构，但仍必须通过安全与可训练性验证。
- 超参数搜索是最多 5 次的轻量网格抽样，不是成熟的 Bayesian optimization。
- 图像语义分析尚未接入 DepictQA；当前 profiler 只使用可见像素统计特征。
- 当前同步串行执行，未实现 GPU 并行 benchmark 或远程任务队列。

## Future Work

- 接入 vLLM 或 DepictQA，但继续隔离完整图像 Ground Truth；
- 增加 tensor completion 专用优化器、可证明秩自适应策略和更丰富的受限生命周期 hook；
- 在多数据集、多 mask、多随机种子上建立稳定 benchmark；
- 将完整训练放入带 CPU/GPU、内存和时间限制的容器沙箱；
- 对候选做多案例晋升，避免单图过拟合；
- 接入 MLflow/W&B 与可视化前端；
- 加入人工审批节点和可恢复的异步执行。

## 学习记录

- [Day 1：可信插值基线](notes/DAY1_LEARNING.md)
- [Day 2：张量分解模型与两阶段训练](notes/DAY2_LEARNING.md)
- [Day 3：Tensor Inpainting Agent Framework Tools 与状态机](notes/DAY3_LEARNING.md)
- [Day 4：知识检索与方法选择](notes/DAY4_LEARNING.md)
- [Day 5：候选代码生成与验证](notes/DAY5_LEARNING.md)
- [Day 6：公平实验、反馈迭代与晋升](notes/DAY6_LEARNING.md)
- [Day 7：端到端交付、Benchmark 与求职展示](notes/DAY7_LEARNING.md)

2～3 分钟录屏提纲见 [`DEMO_SCRIPT.md`](notes/DEMO_SCRIPT.md)。
