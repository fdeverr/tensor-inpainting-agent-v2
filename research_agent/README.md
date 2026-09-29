# Tensor Inpainting Agent

一个基于 Tensor Inpainting Agent Framework 的多维张量补全研究 Agent：它支持彩图、MSI 与视频，分析缺失数据、检索方法经验、从 11 种张量分解基线中自动或手动选择，再由 LLM 设计新候选，并在单张图像上用缺失区 Ground Truth 直接选择结构、超参数和最佳 checkpoint。

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
  ├─ Multimodal Observer ───────── 只对比 incumbent/candidate（不看真值）
  └─ Experiment Controller ─────── 默认五轮诊断式变异，可选三组件完整消融，记录完整 Trace
          │
          ▼
固定实验内核
  ├─ AST + 独立进程 smoke validation
  ├─ 同预算 paired tuning：全部可见像素训练，缺失区 GT 选优
  ├─ 复现 GT 选中的最佳 checkpoint
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

`smoke` 将图像最长边限制为 64 像素，并减少训练、调参和改进轮数，只用于快速检查系统是否跑通。`full` 默认将最长边限制为 128 像素：分解家族预赛每个 trial 最多 400 步，胜出张量方法每个 trial 先训练 1500 步，必要时对当前最佳配置自动扩展到 3000/6000 步，SIREN 每个 trial 最多 4000 步。Day 6 允许 LLM 在 3000 步硬上限内为首个候选提议训练步数、验证间隔和早停耐心值，并将解析结果锁定为整个进化过程的公平协议；确定性后备候选默认请求 2000 步。`original` 使用与 `full` 相同的实验预算，但不缩放输入图像。三种模式都保留原始宽高比，不会强制图像变为正方形。

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
  --method-max-steps-ceiling 6000 \
  --fair-max-steps 3000 \
  --tuning-trials 3 \
  --device cuda
```

统一脚本默认使用 `--llm-mode required --device cuda`，并从 `research_agent/.env` 加载 LLM 配置。传入 `--llm-mode off` 时不再依赖 `.env`，并使用确定性选择器和 TV 候选模板；`auto` 会在环境变量完整时创建 LLM 客户端，否则回退到确定性规则。

多模态视觉观察默认关闭，并拆分为两个可独立组合的消融选项。脚本参数 `--selection-visual-assessment on`（Python 入口使用 `--selection-visual-assessment`）会在 Day 4 的 Manhattan 插值完成后只把插值恢复图交给视觉模型。视觉结果只用于候选家族召回和 low/medium/high 粗粒度秩先验，不再直接决定唯一分解。候选短名单会先进行同预算数值预赛，胜出家族再做更完整的秩调参。该信息不会自动进入算法变异历史。

默认短名单大小为 3，每个家族预赛 2 个 trial、最多 400 步，预赛在连续 10 次 GT 选择分数无改善后早停。当前主工作流不再划分可见像素训练/验证集，因此 CLI 不再提供验证比例和验证形状参数。

家族胜出后不再把学习率当作一个固定值。第一阶段对结构配置与学习率做有界粗搜，第二阶段固定粗搜胜出结构，在胜出学习率两侧追加 `÷3` 和 `×3` 精搜。每个组合都在全部可见像素上训练，并持续跟踪缺失区 GT MSE 上的最佳值和 `best_step`。GT 只用于选择，不进入模型的梯度损失。

SIREN 默认作为独立的坐标 INR 基线训练与调参：4 个覆盖宽度、深度和频率的配置，每个最多 4000 步，每 25 步验证，连续 20 次验证无改善后早停。可用 `--siren-max-steps`、`--siren-tuning-trials`、`--siren-validation-interval` 和 `--siren-patience` 独立调整。完成的 SIREN 训练会按损坏数据、mask、SIREN 源码和全部训练配置生成指纹，保存在输出根目录的 `.siren_cache/`。同指纹再次运行时跳过调参和训练，只复用完整模型产物并重新生成本次评估记录；数据、mask、代码或配置任一改变都会产生新指纹。SIREN 与 Manhattan 插值、数值预赛胜出的张量分解、进化候选共同出现在报告和 `comparison_images/`，并进入最终 Missing-region PSNR 冠军池。可用 `--siren-comparison off`（Python 入口为 `--skip-siren-comparison`）关闭。SIREN 不作为张量变异父类。

脚本参数 `--mutation-visual-assessment on`（Python 入口使用 `--mutation-visual-assessment`）用于测试视觉反馈：首轮不与插值比较，后续每轮只比较当前 incumbent 与 candidate，并将局部模糊、过度平滑、边界和纹理观察写入当前运行实践。候选是否晋级仍只由固定数值 Judge 决定。`selection-visual-assessment` 仍只用于 Day 4 前置分解选择，其观察不进入算法变异。

Day 6 不再强制新架构沿用基础张量模型的单一学习率。模型首次进入公平实验时，与对手使用相同的学习率搜索规则，但在自己的结构空间内独立调优；默认粗搜 `0.001、0.01、0.1`，再围绕胜出点追加 `÷3`、`×3` 精搜。成为下轮 incumbent 后直接复用这份结果。脚本入口可用 `--fair-learning-rates`、`--fair-learning-rate-refinement on/off` 和 `--fair-learning-rate-refinement-factor` 调整；Python 入口用 `--skip-fair-learning-rate-refinement` 关闭精搜。`--tuning-trials` 是每个模型各自的结构搜索上限：若 incumbent 只有 1 个不同组合而 candidate 有 3 个，则分别运行 1 与 3 个结构 trial，不会因一方空间较小而限制另一方。实际粗搜运行数为“该模型的有效结构配置数 × 学习率数”，搜索覆盖率、trial 数和总运行时间均单独记录。

LLM 的进化以“相对当前 incumbent 提升 Missing-region PSNR”为主目标；Composite SSIM 是固定 Judge 的容差约束。同时，首轮和后续每轮候选生成都可读取结构化 `algorithm_comparison_reference`：它排列 Manhattan 插值、SIREN、选中张量基线和张量家族预赛结果，并在后续轮次追加最新 incumbent/candidate 指标及差值。LLM 可用这些参考诊断差距、识别值得迁移的归纳偏置，但不能用它替代当前 incumbent 的晋升目标或固定 Judge。参考面板仅暴露聚合指标，不暴露 Ground Truth 张量内容；因为人工隐藏区域的评估分数被反复用于进化，它们属于开发反馈，不再是完全未见的最终测试证据。LPIPS 仍作为感知质量诊断，无参考指标仍只用于最终展示。

每轮生成候选前都会诊断当前框架的训练/验证曲线。诊断只输出训练末端仍在改善、验证集在最佳点后回退、训练损失进入平台期等可核查信号，不直接推断原因，也不替 LLM 决定下一种变异。首轮读取 Day 4 张量基线的历史；后续轮次读取实际成为 incumbent 的模型历史。

候选生成遵循“痛点 → 核心难点 → 简化后的可验证问题 → idea”的顺序。默认 `atomic` 模式只改变一个机制；`combination` 模式可组合两个或三个独立组件。系统自动构造完整 `2^N` 消融，并用缺失区 GT MSE 进行小预算筛选；胜出的非基线组才进入完整预算的正式 Judge。

`.env` 中可用 `VISION_MODEL_ID` 指定专用多模态模型；留空时复用 `LLM_MODEL_ID`。如果视觉模型位于另一个服务，还可设置 `VISION_API_KEY` 和 `VISION_BASE_URL`；二者留空时复用普通 LLM 的连接配置。模型服务需支持 OpenAI 兼容的图像输入。如果模型不支持图像、接口失败或未配置 LLM，对应视觉阶段会记为 `failed` 或 `skipped`，其余评测和进化不会中断。

OpenAI 系接口默认走 `/v1/chat/completions`，兼容 DeepSeek、Qwen、Kimi、智谱、Ollama 等第三方服务。改为 OpenAI 官方新的 Responses 接口时设置 `LLM_API_STYLE=responses`，此时改走 `/v1/responses`；多模态模型如需单独指定，用 `VISION_API_STYLE` 覆盖。注意 `/v1/responses` 目前仅 OpenAI 官方及少数厂商提供，第三方兼容端点只有 `/v1/chat/completions`，设置错误会直接返回 404。该选项对 Anthropic 与 Gemini 服务无效。

评价指标分为两组。全参考组包含 MSE、Missing-region/Full-image PSNR、Composite SSIM 和 LPIPS，默认开启；其中 PSNR/SSIM 是 Judge 与报告所需的核心指标，始终计算，`--full-reference-metrics off`（Python 入口为 `--skip-full-reference-metrics`）只关闭较重的 LPIPS。无参考组包含 MANIQA、CLIP-IQA 和 MUSIQ，默认关闭，可用 `--no-reference-metrics on`（Python 入口为 `--no-reference-metrics`）开启。四个神经指标均通过 PyIQA 对恢复已观测像素后的完整复合图像计算，并且仅支持 RGB；MSI/视频会记录为跳过。首次运行可能下载预训练权重，单项失败记为 `N/A`，不会中断其他指标。

### 指标协议

| 指标 | 类型 | 范围 | 方向 | 作用 |
|---|---|---|---|---|
| Missing-region PSNR | 全参考 | 人工隐藏区域 | ↑ | 进化反馈、Judge 与最终算法选择 |
| Full-image PSNR | 全参考 | 恢复已观测像素后的完整张量 | ↑ | 最终结果展示与完整图像质量诊断 |
| Composite SSIM | 全参考 | 复合整图 | ↑ | 局部结构一致性 |
| LPIPS | 全参考 | 复合整图 | ↓ | 深度特征感知距离 |
| MANIQA | 无参考 | 复合整图 | ↑ | Transformer 感知质量 |
| CLIP-IQA | 无参考 | 复合整图 | ↑ | 视觉-语言特征质量 |
| MUSIQ | 无参考 | 复合整图 | ↑ | 多尺度感知质量 |

缺失区域 PSNR 只衡量真正需要补全的位置，因此用于指导算法变异、Judge 和冠军选择。全图 PSNR 先将已观测位置恢复为真实值，再对完整张量计算；由于大量已知像素误差为零，它可能稀释缺失区域差异，所以只作为最终输出指标之一展示，不进入进化上下文和选择逻辑。最终报告同时给出两种 PSNR。

新增的四项感知指标只用于最终诊断和 benchmark 汇总，不进入训练、调参、Judge 或冠军选择。缺失区 GT MSE/PSNR 则是当前单图 oracle 搜索的选择信号。

运行时会显示当前工作流阶段、LLM 候选生成/修复轮次、自动调参 trial 以及最佳 checkpoint 直接复用状态。交互式终端中的训练进度条会在同一行刷新，包含 `当前步/总步数`、百分比、训练损失、GT 选择分数和 ETA。

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
| LLM 候选 | Convolutional decoder / refiner | 从头训练的卷积解码、残差精修或带防网格融合的多尺度空洞 Conv2d/Conv3d 分支 |
| LLM 候选 | Efficient Transformer | 轴向、窗口、patch、channel、temporal 或 latent-token 自注意力，以及张量因子/坐标 query 到紧凑特征 token 的交叉注意力；禁止全像素二次方注意力 |
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

## 单图 GT 直接选择协议

设 `M` 为真实可见像素，`~M` 为人工遮挡区域：

```text
M_train = M，M_select = ~M
```

- 梯度更新只使用全部 `M`，GT 不进入优化损失；
- 结构、学习率、早停与 `best_step` 根据 `~M` 上的 GT MSE 选择；
- Agent 可读取 Missing-region PSNR 等聚合对比指标，但不读取 GT 张量内容；
- 最终输出是已搜索配置中 GT 分数最好的结果；
- 这是单图专用的 oracle/development 结果，不是未见测试集上的无偏泛化证据。

LLM 可在 `--fair-max-steps` 用户硬上限内为首轮请求最大步数、GT 评分间隔和早停耐心值。首轮解析后的单次训练预算、学习率搜索规则、随机种子、优化器和设备被锁定为本次进化的全程公平协议。每个 trial 训练结束后保留 GT 评分最优 checkpoint，不再进行“最终重训”；incumbent 在下轮继续直接复用已有结果。

## 候选代码验证与晋升

候选进入训练前必须依次通过：

1. Pydantic `CandidateProposal` 结构验证；
2. AST 语法、导入、危险调用、父类与接口检查；
3. 独立进程中的非方形动态尺寸构造、forward、有限值、参数量上限、backward、梯度与一步优化检查；
4. manifest 状态和 SHA-256 完整性检查；
5. 与基础张量模型的同预算公平实验。

如果首个 LLM 候选在 AST 或独立进程检查中失败，Day 5 和 Day 6 的改进轮都会保留原候选与验证报告，再将精确的工具反馈与原代码交给 LLM 修复一次。若两轮 LLM 候选仍未通过，系统会验证一次内置的确定性安全候选。Day 6 中即使安全候选也失败，也会保留已完成轮次、正常生成报告，不会因候选代码问题丢失整次运行。验证器规则本身不会交给 LLM 修改。

搜索空间元数据明确分为三层：`declared_search_space` 保留 LLM 在结构化 proposal 中的原始声明，`executable_search_space` 记录候选代码经验证后 `search_space()` 的实际返回值，`effective_search_space` 是 Day 6 真正交给调参器的完整空间。新候选不再写入含义模糊的 `allowed_search_space`；旧候选进入 Day 6 时会自动迁移为三层字段。

如果候选已通过 AST、forward 和 backward 检查，但 `declared_search_space` 与 `executable_search_space` 不一致，本次候选会被判为验证失败，但不会终止 Day 5/Day 6。系统把两侧相对基础模型的精确差异反馈给 LLM，允许它修改结构化声明或代码中的 `search_space()` 后重新生成完整候选；不会静默删减、扩张或用预定义空间覆盖研究假设。连续两个 LLM 候选仍未自洽时才使用确定性安全 fallback。只有契约一致的候选才会设置 `effective_search_space` 并进入调参。

固定 Judge 的默认准入条件：

```text
missing-region PSNR delta >= +0.2 dB
composite SSIM delta >= -0.002
双方均完成各自有界调优，训练过程无 NaN / Inf / 异常
最终全观测像素重训未发生明显数值发散
```

完成指定轮数后，最终 incumbent 及 idea、验证报告、最佳配置、指标、适用条件、来源 run 和代码哈希会进入 `algorithms/approved/<base-method>/<name>/<version>/`。所有已晋升算法共用一个 `AlgorithmRunnerTool`，不会为每个候选复制训练代码。

## 进化记忆的生命周期

- 当前运行实践：`outputs/<day6-run-id>/knowledge/practice.md` 和 `practice.jsonl`。每条严格保存为（当前框架与条件、目标、方法、结果）四元组，只供同一次运行中的后续轮次使用。
- 可选前置方法选择观察（开启 `selection-visual-assessment` 时）：`outputs/<day4-run-id>/method_selection_visual.json`；只观察 Manhattan 插值恢复图，为分解家族和粗粒度秩范围提供一次性先验。
- 分解家族数值预赛：`outputs/<day4-run-id>/method_screening.json`；记录短名单、同预算 trial、mask-matched 验证分数与最终胜者。
- SIREN 基线：`outputs/<day4-run-id>/siren_baseline/`；包含调参、checkpoint、补全数据、预览和指标。
- 首轮变异不额外做插值视觉对比，但会读取插值、SIREN、选中张量基线和张量家族预赛的结构化数值参考。`outputs/<day5-run-id>/initial_interpolation_visual.json` 记录“使用结构化算法参考、跳过首轮插值视觉对比”的审计状态。
- 可选每轮变异视觉观察（开启 `mutation-visual-assessment` 时）：`outputs/<day6-run-id>/round-XX/visual_assessment.json`；关键结论同时写入当前运行实践，并参与下一轮变异目标与 idea 的生成。
- 全局可复用经验：`algorithms/evolution_knowledge/<base-method>/reusable_experience.md` 和 `reusable_experience.jsonl`。跨运行持续追加，新的独立运行会读取它。
- 全局经验由 LLM 根据“当前框架、目标、方法、结果”四元组提炼。每条仍只保存一段可跨轮复用的 `experience` 和 `confidence`；不保存运行号、轮次、候选 ID、证据分段或下一轮指令。完全重复的经验不会再次追加。
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

| 方法 | Missing-region PSNR ↑ | Composite SSIM ↑ | 最终拟合时间 | 参数量 | 结论 |
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

测试集覆盖 mask、PSNR/SSIM 与学习式 IQA 指标、11 种张量模型、GT checkpoint 选择与直接复用、终端进度显示、Tensor Inpainting Agent Framework Tools、方法选择、深度候选接口、LLM 训练预算限幅、Day 5/6 候选代码验证、反馈修复与安全回退、效果图对比导出、Judge、停止条件、统一入口和 benchmark 聚合。

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

每轮算法进化会将当前运行已完成的实践四元组传给下一轮，并附带当前框架的训练诊断、当前最优候选代码以及同一基础分解方法的跨轮经验。成功和失败结果都会保留；关闭视觉观察和学习式指标不影响这些记忆。当前运行记录保存在 `knowledge/practice.jsonl` 与 `knowledge/practice.md`，全局经验按基础分解方法独立累计。
