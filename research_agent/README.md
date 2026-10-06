# Tensor Inpainting Agent

一个基于 Tensor Inpainting Agent Framework 的多维张量恢复研究 Agent：它支持彩图、MSI、视频与分帧音频波形，分析缺失数据、检索方法经验、从 11 种张量分解基线中自动或手动选择，再由 LLM 设计新候选，并用缺失区 Ground Truth 直接选择结构、超参数和最佳 checkpoint。

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

### Multi-Dimensional Data Recovery

新入口 `python -m research_agent.run_recovery` 与原单样本入口并存。**正式进化前**，每种所选数据类型的固定插值、可选 SIREN 与历史冠军先在全部有效样本上评测；SIREN 先按整类平均指标选择配置，再按共享预算评测。LLM 可见对照算法的逐样本信息和整类指标，并推荐 3–5 个张量分解方法。入围方法先在代表样本上做等量轻量调参，再冻结配置跑同类全部样本，按整类平均选出基础方法；Day 4 确定实际 incumbent 后，还会用其已选配置跑完整类。首轮及后续轮次的 LLM 能看到对比算法和真实 incumbent 的逐样本指标、训练曲线摘要。每种类型只选一个样本搜索进化候选结构和超参数；但**每一轮进化**都会在该类型全部有效样本上，用冻结的同一候选结构/超参数和 `--evaluation-steps` 预算逐样本独立拟合。候选必须在全部样本上成功，并以整类平均指标通过 Judge，才能成为下一轮 incumbent；每个样本的指标、训练曲线摘要和整类均值都会反馈到下一轮，完整曲线保存在各样本的 `final_fit_history.json`。最终输出的是最后胜出的进化算法，而非研发样本上单独得分最高的算法。不会把一张图的权重直接迁移到其他样本，也不会在每个评测样本上重新生成代码或搜索超参数。计算量会随基线数、调参次数、有效样本数和进化轮数增长。

推荐用专用 Bash 入口，默认数据目录是项目同一级的 `Multi_dimensional_data`，读取 `research_agent/.env`。每次可以只选一类，也可以选多类；不传 `--data-types` 才会运行全部四类。类型名称区分大小写，未选中的类型不会训练、进化或更新其算法档案与经验库（启动时仍会扫描数据目录生成完整清单）。

```bash
# 本次只进行 MSI 的完整进化与同类数据集评测
bash research_agent/scripts/run_recovery.sh --data-types MSI --llm-mode required

# 本次只进行 Video，整帧缺失
bash research_agent/scripts/run_recovery.sh --data-types Video --mask-type slices --missing-rate 0.4

# 也可以同时选两类
bash research_agent/scripts/run_recovery.sh --data-types Image audio

# 只重新评测 MSI 历史算法，不进化
bash research_agent/scripts/run_recovery.sh --data-types MSI --evaluate-only

# 查看脚本选项；PYTHON_BIN 可指定已安装依赖的虚拟环境解释器
bash research_agent/scripts/run_recovery.sh --help
```

脚本从任意工作目录启动都可以，选项与下方 Python 入口一致；相对输出/输入路径按项目根目录解析。Image/MSI/Video 默认 PSNR/SSIM，音频仅 NMSE；只有显式传入 `--lpips` 才额外计算 Image 的 LPIPS。

`run_recovery.sh` 顶部的“直接修改这里的实验参数”区可以直接编辑，不需要每次输入长命令。比如只跑 MSI，研发训练最多 3000 步、每 20 步验证一次、连续 30 次验证未改善后早停：

```bash
DATA_TYPES=("MSI")
EVOLUTION_STEPS="3000"
VALIDATION_INTERVAL="20"
PATIENCE="30"
EVALUATION_STEPS="2000"
EVALUATION_VALIDATION_INTERVAL="20"
EVALUATION_PATIENCE="0"
```

编辑后执行 `bash research_agent/scripts/run_recovery.sh` 即可。命令行仍可逐项覆盖（同时支持 `--name value` 与 `--name=value`）：

```bash
bash research_agent/scripts/run_recovery.sh --data-types MSI \
  --evolution-steps 3000 --validation-interval 20 --patience 30 \
  --evaluation-steps 2000 --evaluation-validation-interval 20 --evaluation-patience 15
```

`PATIENCE` 的单位是**验证次数**，不是训练步数；基于 GT 缺失区误差选择 checkpoint，不重新切分观测训练/验证集。研发、基线预赛和 SIREN 各有独立 patience。数据集评测的 `EVALUATION_PATIENCE=0` 默认关闭早停，正整数启用早停；新旧算法始终使用同一评测上限、验证间隔与早停规则，具体最佳 checkpoint 步数可能不同。评测学习率和模型结构保留归档冠军配置，不再次搜索。

`METHOD_MAX_STEPS_CEILING`、`FAIR_MAX_STEPS` 为空时跟随 `EVOLUTION_STEPS`；`SCREENING_MAX_STEPS` 为空时为 `min(200, EVOLUTION_STEPS)`。`SIREN_MAX_STEPS` 控制每个 SIREN 调参 trial 的步数，空值跟随 `EVALUATION_STEPS`；`SIREN_VALIDATION_INTERVAL` 和 `SIREN_PATIENCE` 控制调参阶段。选出的 SIREN 配置随后用统一的 `EVALUATION_STEPS`、`EVALUATION_VALIDATION_INTERVAL`、`EVALUATION_PATIENCE` 跑全部样本。`SIREN_LEARNING_RATES="0.00005,0.0001,0.0003"` 控制小学习率搜索。顶部还开放了学习率粗搜/精搜、消融预赛、基线预赛、晋级阈值、知识检索和持久化路径。布尔变量使用 `on/off`；命令行开关使用 `--lpips/--no-lpips`、`--siren-comparison/--no-siren-comparison` 和 `--fair-learning-rate-refinement/--no-fair-learning-rate-refinement`，旧 `--skip-siren-comparison` 仍有效。

已对齐 `run_ai.sh` 的全部实验超参数/开关，支持原 `--name on|off` 语法、无值开关和 `--name=value`。新增的 `PROMPT` 可追加各类型的算法研发要求。兼容名称对应如下（两种命令行名称共用同一配置，最后传入者优先）：

| `run_ai.sh` 名称 | Recovery 对应 |
|---|---|
| `--method-max-steps` / 顶部 `METHOD_MAX_STEPS` | `--evolution-steps` / `EVOLUTION_STEPS` |
| `--max-improvement-rounds` / `MAX_IMPROVEMENT_ROUNDS` | `--improvement-rounds` / `IMPROVEMENT_ROUNDS` |
| `--full-reference-metrics` / `FULL_REFERENCE_METRICS` | `--lpips` / `LPIPS` |
| `--image-size original` | 保留原尺寸，同 `--image-size 0` |
| `--image`（单样本输入，不是超参数） | `--dataset-root` + `--representative TYPE=FILE` |
| `--mat-key`（单样本变量选择，不是超参数） | Recovery 固定读取 `Ohsi` GT，不自动选择旧缺失数据 |

顶部的 `METHOD_MAX_STEPS`、`MAX_IMPROVEMENT_ROUNDS`、`FULL_REFERENCE_METRICS` 默认为空，此时跟随右列配置；填值后覆盖右列脚本默认值。`NO_REFERENCE_METRICS`（MANIQA/CLIP-IQA/MUSIQ）、`SELECTION_VISUAL_ASSESSMENT`、`MUTATION_VISUAL_ASSESSMENT` 也已接通。LPIPS、无参考图像指标及两类视觉观察**仅应用于 Image**，不把 MSI、Video、audio 投影成任意 RGB 图像进行误导性评价；PSNR/SSIM 仅对 Image/MSI/Video 计算，音频仅 NMSE。高级可选指标/视觉模型仍需要相应依赖及模型配置，默认关闭。Recovery 没有 `smoke/full/original` 模式参数，其预算直接由顶部参数区控制。

```bash
# 只检查 GT、分类和文件完整性，不训练、不调用 LLM
python -m research_agent.run_recovery \
  --dataset-root ../Multi_dimensional_data --inventory-only

# 四类各一个研发样本，切片缺失率 40%，再评测同类所有有效样本
python -m research_agent.run_recovery \
  --dataset-root ../Multi_dimensional_data \
  --mask-type slices --missing-rate 0.4 --llm-mode required

# 只重新评测已经归档的算法，不进化、不调用 LLM
python -m research_agent.run_recovery \
  --dataset-root ../Multi_dimensional_data \
  --data-types MSI --mask-type random --missing-rate 0.6 --evaluate-only

# 快速调试；显式指定研发样本时可重复 --representative
python -m research_agent.run_recovery \
  --dataset-root ../Multi_dimensional_data --data-types Image \
  --representative Image=F16_random_missing_0.10.mat \
  --image-size 64 --evolution-steps 100 --evaluation-steps 100 \
  --improvement-rounds 1 --tuning-trials 1 --llm-mode off --skip-siren-comparison
```

输入数据不会被修改。MAT 仅读取 `Ohsi`（GT），忽略旧 `Nhsi`、mask 与文件名中的缺失率。当前数据集的视频轴序 `[H,W,C,T]` 会显式转换为模型的 `[H,W,T,C]`。RGB/MSI/Video 默认把空间最长边调整为 128，`--image-size 0` 保留原始空间分辨率；这项设置会记录在评测协议中。音频不会被空间缩放。

| 类型 | 模型张量 | `slices` 默认缺失对象 |
|---|---|---|
| Image | `[H,W,3]` | 完整图像行 |
| MSI | `[H,W,B]` | 完整波段 |
| Video | `[H,W,T,C]` | 完整帧（全部空间位置与通道） |
| audio | `[时间帧,帧内采样,声道]` | 原始波形的一段连续时间，所有声道共同缺失 |

`sildes` 是 `slices` 的兼容别名。非音频的切片随机选取完整轴切片，缺失率按整数切片数取整，报告记录实际缺失率；音频为连续时间段。`random` 对非音频逐元素遮挡，对音频按时间采样点、共同遮挡各声道；`block` 对非音频生成紧凑 N 维块，对音频生成连续时间段。支持完整张量布尔 mask，梯度与缺失区指标都按实际元素计算，而不把波段/帧缺失投影成二维空间缺块。

WAV 保留采样率和声道；波形分帧是可逆 reshape，不使用 GT 的隐藏相位进行恢复。最后一帧补零位置不是真实音频观测，不参与训练损失、可见均值初始化、数据分析统计或缺失区误差，输出 WAV 裁掉 padding。**音频不计算 PSNR/SSIM，只使用缺失波形 NMSE，越低越好**：先撤销幅值归一化和分帧，再在人工缺失采样点（所有声道）计算 `sum((prediction-GT)^2) / sum(GT^2)`，不减均值、不使用偏移到 `[0,1]` 后的能量。它衡量波形恢复误差，不声称等同于主观听觉质量。

SIREN 的坐标按数据类型定义：Image/MSI 使用归一化二维 `(x,y)` 并输出图像/光谱通道；Video 使用三维 `(x,y,t)` 并输出当前位置的通道值；audio 使用原始采样顺序的一维连续时间 `t`，跨分帧边界，输出声道值。有效坐标统一归一化到 `[-1,1]`；audio 分帧仅决定保存形状，不成为额外输入坐标。网络分块计算坐标，并在训练时重计算隐藏激活，控制长视频/音频的内存；优化损失仍覆盖全部观测点。

Recovery 默认给 SIREN 搜索 4 个组合，覆盖学习率 `5e-5/1e-4/3e-4`、宽度 `128/256`、隐藏层 `3/4`、首层频率 `20/30/60`。这是有界组合设计，不是笛卡尔穷举；`SIREN_TUNING_TRIALS` 缩小时只运行前几个组合。每个配置在同类全部样本上独立拟合，只有全量成功者可按平均 PSNR（audio 为平均 NMSE）入选。完整证据在 `<recovery-run>/<类型>/siren_tuning/tuning.json`；正式分数在 `fixed_baselines/siren/`，协议完全一致时直接复用调参产物。Day 4 复用正式代表样本的同一结果，后续 LLM 的整类面板与最终对比表一致。报告增加 SIREN 配置、各 trial 均值、每个样本的最佳 checkpoint 步数和训练曲线路径。全量调参失败时明确记录失败。

Recovery 的 SIREN 调参 trial 和正式评测均使用跨运行持久缓存，位于 `OUTPUT_DIR/.siren_cache/recovery-v1/`，而非每次新建的 run 目录。按单个样本的实际 GT/mask 内容、种子、数据类型及音频恢复元数据、网络配置、学习率、步数、GT 检查间隔、有效早停耐心、相关实现和运行环境指纹匹配；复用前校验全部产物的大小及 SHA-256。命中后复制 checkpoint、重建数据、完整训练曲线与指标到本次 run，并更新路径，不再调用训练。只改 LLM 配置、进化轮数或代表样本不会使缓存失效；改变某个样本或增加候选配置只重跑未命中的拟合。每次仍按当前整类样本重新汇总并选出最佳 SIREN 配置，不直接沿用旧冠军或旧均值。失败拟合不缓存，缓存损坏时重新训练；缓存不可写不影响已成功的训练结果。报告与 `tuning.json` 显示复用、新训练及失败的样本拟合次数。

调参与正式评测协议不同（默认检查间隔和早停规则不同）时，首次仍须分别拟合；之后分别命中各自缓存。保持相同 `OUTPUT_DIR` 且保留隐藏缓存目录才能跨运行复用。旧版本没有本缓存清单的运行结果不能安全自动复用，升级后首次会建立缓存，后续相同条件跳过训练。分发 ZIP 不包含运行输出或缓存。

音频 checkpoint、候选比较、LLM 参考面板、当前运行实践、最终报告和整类冠军均按 NMSE 协议处理。`MINIMUM_NMSE_DELTA` / `--minimum-nmse-delta` 控制候选晋级所需的绝对 NMSE 降低量，默认 0 且须严格改善；音频不使用 PSNR 增益或 SSIM 容差门槛。数据集先计算每条音频的 NMSE，再取算术均值，所有有效样本均有可定义结果的版本才参与冠军排序。训练的可见样本 MSE/loss 及内部归一化误差仍可作为优化诊断保存，但不是额外的音频评价指标。

#### 训练与输出的一致性

GT checkpoint 选择与最终评分均使用裁剪到 `[0,1]` 的预测；训练梯度仍只来自真实观测位置的未裁剪预测。`early_stopping_min_delta` 仅控制 patience，微小但真实的指标改善也会保留为最佳 checkpoint。训练曲线中的 `data_train_loss`、`total_train_loss`、GT 分数均对应同一次参数更新后的模型；总 loss 包含正则项，额外保存 `regularization_losses`。LLM 曲线摘要注明损失时序和输出范围，并包含实际选中 checkpoint 的曲线点。裁剪前的 NaN/Inf 会立即报错，不会被裁剪掩盖。SIREN 缓存同时校验模型、Trainer 和指标代码，协议变化不会复用旧分数。

彩图插值逐颜色通道按二维空间执行，Video 逐颜色通道按 `(y,x,t)` 时空执行，不借用另一个颜色的数值；整条颜色通道完全无观测时明确失败。MSI 仍允许沿有序光谱轴恢复完全缺失波段。音频的研发入口和正式对照均使用连续时间轴逐声道线性插值，跨分帧边界且排除 padding。

Image/MSI/Video 出现完全恢复时，PSNR 以 JSON `null`、报告 `∞` 表示；全类算术平均也是 `∞`。这类平局统一按完全恢复样本数、其余样本平均有限 PSNR、平均 SSIM 打破，预赛、SIREN 调参、进化晋级与正式冠军排序保持一致。

缺失区 GT 能量为 0 时，完全恢复约定 NMSE=0；非零误差记为 `null`（未定义），明确记录原因，不能靠排除该样本获得有效均值或冠军。LPIPS 默认关闭，`--lpips` 仅对 Image 开启。

包含数据的分发 ZIP 解压后保留两个同级目录：`tensor_inpainting_agent/`（项目）和 `Multi_dimensional_data/`（原始数据），这样 `run_recovery.sh` 的默认数据路径可直接使用。数据原样保存，包括旧 `Nhsi`/mask；实际运行仍只从 GT 重新造 mask。包内 `DATASET_MANIFEST.json` 记录逐文件 SHA256、大小及数据完整性问题。当前损坏的 `kodim_random_missing_0.10.mat` 和 `gt_counting.wav` 也原样保留，但运行时排除，不修补或默默计入有效样本。

每次无显式选择时，研发样本根据同类已归档运行次数轮换。历史冠军与固定对照（插值、启用时的 SIREN）在进化前用当前全部有效样本的 GT、mask 和种子重新评测；可训练算法共用 `--evaluation-steps`，插值无需训练。逐样本指标与训练摘要会提供给每轮 LLM，但不提供隐藏 GT 内容。每轮候选的整类评测固定算法和配置、统一预生成的 mask、各样本固定随机种子与评测预算，只使用 GT 选 checkpoint；逐样本指标、训练曲线摘要和整类平均值决定进化方向。不沿用历史运行分数冒充同条件比较。可能不适配新尺寸的算法会明确记录失败，不会悄悄调整结构参数。

如果不同历史版本的代码、配置和当前评测条件完全相同，本轮会共享一次拟合，并标注复用来源，避免重复训练同一个算法；这不是跨缺失率、mask 或预算复用旧分数。

历史算法不再只提供名称和成绩。每个历史版本有 `structure_reference`：基础方法、入口类/函数、继承关系、声明的参数/模块及构造参数默认值、设计提案与最佳配置。基础分解推荐阶段只接收结构概要；完整历史源码要等 Day 4 选定基础框架后才加载。**只加载与当前基础张量分解框架一致的历史源码**：例如选中 Tucker，只参考以 Tucker 为基础的历史内置/进化版本，不加载 TT、SIREN 或插值源码。符合条件且本次同条件整类评测完整的前 3 个历史版本（按当前指标排名）可提供完整 `source_bundle` 及父类依赖，受输入预算约束；其他框架保留成绩和结构概要，不附实现代码。`pre_evolution_reference.json` 记录待选状态，Day 5/Day 6 的 `algorithm_comparison_reference.json` 记录框架选定后的完整源码名单。

#### LLM 上下文预算

最新轮的所有样本指标、训练曲线摘要、失败证据，以及当前冠军和当前基础模型的完整源码优先保留。历史轮次压缩为变更、接受/拒绝、指标趋势、失败与诊断；重复的最新轮评测面板改为引用。源码按 SHA256 去重放入提示词的 `context.code_sources`，各位置用 `code_ref` 指向**同一请求内的完整源码**，不是磁盘路径，也不截断模型代码。完整实践 JSONL 与报告不受提示词压缩影响。

预算紧张时依次移除低排名可选历史源码、低优先级全局经验、历史结构/曲线细节和较早轮次的次要诊断，保留固定插值/SIREN 对照与最新逐样本证据。必要信息仍超预算时，在 API 调用前明确报错，不以截断代码或删除最新样本反馈来继续。修复请求也计入预算；无效非 JSON 响应只保留诊断摘录，可解析候选 JSON 的源码不裁剪。全局经验仍每次完整运行只总结一次；总结超预算则使用全部实践生成确定性总结，不丢弃已完成模型。

`run_ai.sh` 与 `run_recovery.sh` 顶部新增 `LLM_CONTEXT_TOKENS="131072"` 和 `LLM_OUTPUT_RESERVE_TOKENS="16384"`，也可用 `--llm-context-tokens` / `--llm-output-reserve-tokens` 覆盖。前者是输入+输出的总窗口，后者包含在总窗口内，并限制实际生成输出；应按所用模型与服务端限制设置，不能任意增大。输入采用 UTF-8 字节加消息开销的保守估算（**不是服务端精确 token 统计**），可能提前触发压缩/拒绝。每次候选调用记录 `context_budget*.json`，包括预算、压缩前后估算、删减项和是否可发送；运行总结的审计记录在全局经验产物的 `context_audit` 中。

结构参考通过 AST 静态读取并校验归档源码哈希生成，不额外调用 LLM，也不执行代码来猜结构。提案描述只代表设计意图，不能充当已验证收益；实际启用分支应结合最佳超参数开关、构造默认值和实现判断，静态声明不等于运行时激活。摘录若截断会明确标注。新候选档案也记录设计文件哈希；旧提案缺少校验信息时标为未校验，旧依赖采用兼容加载时明确注明。源码缺失、校验失败或无法恢复的结构标为 `unavailable`，不会用当前实现冒充历史算法；当前评测与 Judge 规则不变。

持久化目录：

```text
research_agent/algorithms/history/<Image|MSI|Video|audio>/<run_id>/
  champion.json             # 每次完整研发运行的实际冠军（基线也保存）
  model.py                  # 可训练冠军的代码快照；插值无需模型文件
  model_snapshot/*.py       # 可训练冠军的模型、父类、注册表源码快照
  algorithm.py              # 冠军是插值时保存其实现快照
  best_config.json          # 当次最佳结构/学习率配置
  structure.json            # 冠军构成、设计说明及 forward/损失实现摘要
  development_report.md     # 最佳算法与模型框架报告快照
research_agent/algorithms/history/<类型>/evaluations/<评测run_id>.json
research_agent/algorithms/history/<类型>/latest_evaluation.json
```

冠军按版本归档，不覆盖旧代码。新归档包含模型及父类依赖的独立源码快照和逐文件哈希；加载归档源码，不依赖当前整库哈希，即使只更新 SIREN 也不会影响历史 Tucker/候选的调用。每次加载（包括评测缓存命中）均验证源码，改动或缺失依赖会明确失败。历史插值也执行其归档实现，而非替换为当前版本；所有算法共享当前统一 Trainer、mask 与指标协议。

旧版内置模型归档只保存了模型文件：可使用其归档源码和当前共享 Base 接口，评测元数据与报告会注明兼容边界。旧候选若未保存父类依赖，则只有原模型库仍匹配时才可安全加载；无法恢复的父类不会被悄悄替换为当前实现。可以用 `--history-root` 指向另一个档案库。跨运行全局经验库已恢复：原始实践供本次后续轮次使用，完整运行结束后统一总结一次再供未来运行参考。

最终 recovery 报告在每个数据类型开头用论文式对比表展示结果：数据集为行，算法为分组列，Image/MSI/Video 每个算法下分别列 PSNR、SSIM，audio 只列 NMSE；全类平均单独成行，是该类所有完整样本的算术平均，而非最高单样本分数。各指标最优值加粗、次优值加下划线；失败项显示空缺并列出原因，未完整评测的算法不展示部分样本均值。报告区分两阶段：LLM 推荐的 3–5 个张量分解方法仅在轻量预赛中选进化起点；正式同条件对照只比较本轮/历史进化算法与插值、SIREN（可关闭），不再额外跑全部内置张量分解基线。预赛与正式评测预算/配置可能不同，分数不能直接混排。报告分别标出最优已归档进化版本和含固定对照的全类最优算法。Video 用 Manhattan 最近邻插值，audio 用沿原始波形时间轴的线性插值（跨分帧边界）。固定对照不归档为进化冠军。报告不混合四类数据计算总平均。全部样本已参与算法选择，因此这里是 oracle 开发评测，不是独立盲测。

数据完整性问题会列入报告，并以 `COMPLETED_WITH_FAILURES` 返回非零退出码（仅针对所选数据类型的损坏文件或评测失败）。目前目录中的 `kodim_random_missing_0.10.mat` 无法读出完整 GT，`gt_counting.wav` 有截断警告，默认排除而不是补造 GT；需要修复源文件后才能纳入全量评测。

### 原单样本流程

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

静态[张量分解适用参考库](knowledge/README.md)已按 Image/MSI/Video/audio 扩展，所有 11 个方法都区分实际实现、四类条件、缺失模式/缺失率风险、秩与预算及证据边界。LLM 推荐时先接收全部家族的当前类型简短卡片，再接收 `retrieval_top_k` 条详细证据（优先不同方法），不会把 RGB 的秩限制或空间统计机械用于音频/视频。规则仅作弱先验，整类同预算数值预赛仍决定胜者；这些静态假设不等于跨运行实践总结的全局经验。

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

多模态视觉观察默认关闭，并拆分为两个可独立组合的消融选项。脚本参数 `--selection-visual-assessment on`（Python 入口使用 `--selection-visual-assessment`）会在 Day 4 的 Manhattan 插值完成后只把插值恢复图交给视觉模型。视觉结果作为 LLM 推荐短名单时的粗粒度结构和秩先验；LLM 不可用时也可辅助规则回退，不直接决定唯一分解。候选短名单会先进行同预算数值预赛，胜出家族再做更完整的秩调参。该信息不会自动进入算法变异历史。

默认短名单大小为 3（可设为 3–5），每个家族预赛 2 个 trial；单样本入口默认最多 400 步，recovery 入口默认最多 `min(200, EVOLUTION_STEPS)` 步。预赛在连续 10 次 GT 选择分数无改善后早停。当前主工作流不再划分可见像素训练/验证集，因此 CLI 不再提供验证比例和验证形状参数。

家族胜出后不再把学习率当作一个固定值。第一阶段对结构配置与学习率做有界粗搜，第二阶段固定粗搜胜出结构，在胜出学习率两侧追加 `÷3` 和 `×3` 精搜。每个组合都在全部可见像素上训练，并持续跟踪缺失区 GT MSE 上的最佳值和 `best_step`。GT 只用于选择，不进入模型的梯度损失。

原单样本入口中，SIREN 默认作为独立的坐标 INR 基线训练与调参：4 个覆盖小学习率、宽度、深度和频率的组合，每个最多 4000 步，每 25 步验证，连续 20 次验证无改善后早停。可用 `--siren-max-steps`、`--siren-tuning-trials`、`--siren-learning-rates`、`--siren-validation-interval` 和 `--siren-patience` 独立调整。完成的 SIREN 训练会按损坏数据、mask、音频有效长度/归一化元数据、SIREN 源码和全部训练配置生成指纹，保存在输出根目录的 `.siren_cache/`。同指纹再次运行时跳过调参和训练，只复用完整模型产物并重新生成本次评估记录；数据、mask、代码或配置任一改变都会产生新指纹。Recovery 则复用整类选配置后的代表样本正式结果，不另行调参。SIREN 与 Manhattan 插值、数值预赛胜出的张量分解、进化候选共同出现在单样本报告和 `comparison_images/`，并进入最终 Missing-region PSNR 冠军池。可用 `--siren-comparison off`（Python 入口为 `--skip-siren-comparison`）关闭。SIREN 不作为张量变异父类。

脚本参数 `--mutation-visual-assessment on`（Python 入口使用 `--mutation-visual-assessment`）用于测试视觉反馈：首轮不与插值比较，后续每轮只比较当前 incumbent 与 candidate，并将局部模糊、过度平滑、边界和纹理观察写入当前运行实践。候选是否晋级仍只由固定数值 Judge 决定。`selection-visual-assessment` 仍只用于 Day 4 前置分解选择，其观察不进入算法变异。

Day 6 不再强制新架构沿用基础张量模型的单一学习率。模型首次进入公平实验时，与对手使用相同的学习率搜索规则，但在自己的结构空间内独立调优；默认粗搜 `0.001、0.01、0.1`，再围绕胜出点追加 `÷3`、`×3` 精搜。成为下轮 incumbent 后直接复用这份结果。脚本入口可用 `--fair-learning-rates`、`--fair-learning-rate-refinement on/off` 和 `--fair-learning-rate-refinement-factor` 调整；Python 入口用 `--skip-fair-learning-rate-refinement` 关闭精搜。`--tuning-trials` 是每个模型各自的结构搜索上限：若 incumbent 只有 1 个不同组合而 candidate 有 3 个，则分别运行 1 与 3 个结构 trial，不会因一方空间较小而限制另一方。实际粗搜运行数为“该模型的有效结构配置数 × 学习率数”，搜索覆盖率、trial 数和总运行时间均单独记录。

LLM 的进化以“相对当前 incumbent 提升 Missing-region PSNR”为主目标；Composite SSIM 是固定 Judge 的容差约束，音频则只按缺失波形 NMSE 判定。同时，首轮和后续每轮候选生成都可读取结构化 `algorithm_comparison_reference`：它提供固定插值、SIREN 和胜出张量基线的信息，不把其他入围张量方法的训练记录继续传给进化 LLM；后续轮次追加最新 incumbent/candidate 指标及差值。Recovery 每轮提供逐样本指标、训练摘要与整类统计，但不暴露 Ground Truth 张量内容。LLM 可用这些参考诊断差距、识别值得迁移的归纳偏置，但不能用它替代当前 incumbent 的晋升目标或固定 Judge。因为人工隐藏区域的评估分数被反复用于进化，它们属于开发反馈，不再是完全未见的最终测试证据。LPIPS 仍作为感知质量诊断，无参考指标仍只用于最终展示。

每轮生成候选前都会诊断当前框架的训练/验证曲线。诊断只输出训练末端仍在改善、验证集在最佳点后回退、训练损失进入平台期等可核查信号，不直接推断原因，也不替 LLM 决定下一种变异。首轮读取 Day 4 张量基线的历史；后续轮次读取实际成为 incumbent 的模型历史。

候选生成遵循“痛点 → 核心难点 → 简化后的可验证问题 → idea”的顺序。默认 `atomic` 模式只改变一个机制；`combination` 模式可组合两个或三个独立组件。系统自动构造完整 `2^N` 消融，并用缺失区 GT MSE 进行小预算筛选；胜出的非基线组才进入完整预算的正式 Judge。

`.env` 中可用 `VISION_MODEL_ID` 指定专用多模态模型；留空时复用 `LLM_MODEL_ID`。如果视觉模型位于另一个服务，还可设置 `VISION_API_KEY` 和 `VISION_BASE_URL`；二者留空时复用普通 LLM 的连接配置。模型服务需支持 OpenAI 兼容的图像输入。如果模型不支持图像、接口失败或未配置 LLM，对应视觉阶段会记为 `failed` 或 `skipped`，其余评测和进化不会中断。

OpenAI 系接口默认走 `/v1/chat/completions`，兼容 DeepSeek、Qwen、Kimi、智谱、Ollama 等第三方服务。改为 OpenAI 官方新的 Responses 接口时设置 `LLM_API_STYLE=responses`，此时改走 `/v1/responses`；多模态模型如需单独指定，用 `VISION_API_STYLE` 覆盖。注意 `/v1/responses` 目前仅 OpenAI 官方及少数厂商提供，第三方兼容端点只有 `/v1/chat/completions`，设置错误会直接返回 404。该选项对 Anthropic 与 Gemini 服务无效。

评价指标分为两组。默认只计算核心全参考指标 MSE、Missing-region/Full-image PSNR、Composite SSIM；PSNR/SSIM 是 Judge 与报告所需的核心指标，始终计算。LPIPS 默认关闭，可用 Bash 的 `--full-reference-metrics on`（Python 入口为 `--full-reference-metrics`）开启，用 `--full-reference-metrics off`（Python 入口为 `--skip-full-reference-metrics`）关闭。无参考组包含 MANIQA、CLIP-IQA 和 MUSIQ，默认关闭，可用 `--no-reference-metrics on`（Python 入口为 `--no-reference-metrics`）开启。四个神经指标均通过 PyIQA 对恢复已观测像素后的完整复合图像计算，并且仅支持 RGB；MSI/视频会记录为跳过。首次运行可能下载预训练权重，单项失败记为 `N/A`，不会中断其他指标。

最终报告的“最佳算法与模型框架”介绍实际冠军的模型结构、参数量和最佳配置。进化冠军使用对应候选的设计说明、组件记录及保存的模型代码快照；最后一轮未晋升的候选不会冒充冠军。插值或内置基线胜出时，报告介绍该基线本身的框架。

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
- 分解家族数值预赛：`outputs/<day4-run-id>/method_screening.json`；记录短名单、同预算 trial、缺失区 GT 评分与最终胜者。
- 完整的 3–5 方法预赛仍留在报告和 `method_screening.json`；传给后续进化 LLM 的 `tensor_family_screening` 只保留胜出张量方法的分数和逐样本训练摘要。独立的插值、SIREN（若启用）与历史进化算法对比信息照常保留。其他张量分解方法不作为正式固定对照额外运行。
- `BASE_MODEL=auto` 时，LLM 按顺序推荐 3–5 个不同的张量分解方法，组成完整短名单；可参考插值、SIREN（若启用）和历史进化算法的同条件结果，但这些对照不占推荐名额。只有 LLM 不可用或两次输出无效时，才由知识检索规则和固定回退补齐。recovery 中每个入围方法先在代表样本上以相同试验次数和步数做轻量调参，再冻结各自最佳配置，在同类全部有效样本上以相同 mask、种子和轻量预算独立拟合；仅全量成功的方法参与排序，图像/MSI/视频按平均缺失区 PSNR（SSIM 辅助）、音频按平均 NMSE 选出进化起点。单样本入口仍按代表样本预赛。`BASE_MODEL` 手工指定时跳过自动预赛。recovery 报告区分 LLM 推荐、入围名单和整类预赛胜出者。
- SIREN 基线：`outputs/<day4-run-id>/siren_baseline/`；包含调参、checkpoint、补全数据、预览和指标。
- 首轮变异不额外做插值视觉对比，但会读取插值、SIREN 和胜出张量基线的结构化数值参考，未胜出的入围张量方法只保留在筛选产物中。`outputs/<day5-run-id>/initial_interpolation_visual.json` 记录“使用结构化算法参考、跳过首轮插值视觉对比”的审计状态。
- 可选每轮变异视觉观察（开启 `mutation-visual-assessment` 时）：`outputs/<day6-run-id>/round-XX/visual_assessment.json`；关键结论同时写入当前运行实践，并参与下一轮变异目标与 idea 的生成。
- 跨运行全局经验库：`algorithms/evolution_knowledge/<Image|MSI|Video|audio>/<基础分解>/reusable_experience.jsonl` 与 `.md`。每次完整运行结束后从全部实践总结一次；逐轮不再调用经验提炼 LLM、不再生成 `experience_record.json`，但原始实践和 Judge 反馈逐轮保留。

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

每轮算法进化会将当前运行已完成的实践四元组传给下一轮，并附带当前框架的训练诊断和当前最优候选代码。成功和失败结果都会保留；关闭视觉观察和学习式指标不影响当前运行实践。当前运行记录保存在 `knowledge/practice.jsonl` 与 `knowledge/practice.md`。

全局经验由本次**全部已完成实践**提炼：所有进化轮次结束后只调用一次总结 LLM，覆盖晋级与拒绝的方向、适用条件和证据限制，不把最后一轮当作整个运行，也不把拒绝直接等同于机制无效。LLM 未启用、输出无效或调用失败时，用相同实践作保守规则摘要，不反复重试。摘要保存于 Day 6 的 `run_experience.json`；完整主流程成功生成产物与报告后才提交全局库。直接运行 Day 6 时在该流程完整结束后提交。失败/中断的进化流程、无有效实践的运行、`--evaluate-only` 和 `--inventory-only` 不追加全局经验；Recovery 后续历史归档或附加评测失败，不撤销此前已完成进化的有效摘要。重复提交同一运行不会重复追加，实践源文件发生变化会拒绝提交；并发写入使用文件锁保护。

每次完整进化贡献一条全局摘要；Recovery 若选择多个数据类型，每个类型的完整进化分别贡献一条，不将四类任务混合成通用结论。库按数据类型和进化起点的基础张量方法分目录，保存缺失类型、请求/实际缺失率、整类缺失率范围、评测范围/样本数、训练预算、轮数、最终输出算法、实践源路径及 SHA256。旧版未标类型的经验不自动混入分类库。

首轮候选和后续每轮均读取对应类型/基础方法的历史摘要，优先匹配缺失类型、接近的请求缺失率及新近记录，最多 20 条；本次尚未结束的实践不会提前成为全局经验。跨缺失模式、缺失率、预算的经验只是参考，不能替代当前同条件对照与固定 Judge。读到损坏的全局文件时明确提示并继续当前实验，不删除原文件；总结或写库失败也不会丢弃已完成的算法产物。

`run_ai.sh` 与 `run_recovery.sh` 均可设置 `KNOWLEDGE_ROOT`，或用 `--knowledge-root PATH` 指定；单样本入口默认跟随 `candidate-root` 的同级 `evolution_knowledge`，Recovery 默认 `research_agent/algorithms/evolution_knowledge`。报告显示本次摘要、状态与全局文档位置。
