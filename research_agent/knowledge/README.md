# 多维恢复：张量分解适用条件参考库

这是供 LLM 推荐 3–5 个基础家族的静态参考，不是成绩库。不从单一彩图推出“最强算法”，
也不按数据类型固定冠军。理论模型、当前实现与待测增强分开描述；当前同预算整类预赛决定基础方法。

## 当前实现的张量模

所有 11 个张量基线都在 `D1×D2×F` 三模视图上拟合；SIREN 不在这些张量家族中。

| 类型 | 保存张量 | 基线视图 | 必须注意的实际语义 |
| --- | --- | --- | --- |
| Image | `H×W×C`，通常 RGB | `H×W×C` | 通道秩≤3 仅对实际 C=3 成立 |
| MSI | `H×W×B` | `H×W×B` | 光谱模不固定为三个颜色 |
| Video | `H×W×T×C` | `H×W×(T·C)` | 不是四独立模；彩色视频第三模 FFT 混合时间颜色 |
| audio | `N_frames×frame_size×C` | 同左 | 前两模是时间表示，第三模是声道，不是时间频率 |

audio 分帧是可逆 reshape，目标是有符号原始波形，不是 STFT/幅度谱。单声道 F=1：
mode3 几乎逐采样自由；t-SVD 的第三模 FFT 退化；TT 末秩与层次 Tucker 的 Rc/Rs 都为 1。
这不禁止数值试验，但不能用未实现的高阶、频率或时间机制解释胜出。

## 方法索引

每个单页都有实现、四类适用条件、秩/预算、缺失风险与证据边界。下表是需检验的假设，不是推荐排名。

| 家族 | 检验的结构假设 | 关键限制 |
| --- | --- | --- |
| [Matrix](methods/matrix_factorization.md) | 固定展开具有小矩阵秩 | 不自动换展开方向；整行/列缺失无约束 |
| [mode3](methods/mode3_factorization.md) | 低维特征子空间，位置还有部分特征观测 | 独立位置系数不能自动填空间/时间缺口 |
| [CP](methods/cp_decomposition.md) | 少量可分离成分、紧凑控制组 | 共用秩、尺度漂移；相关性不等于小 CP 秩 |
| [非负 CP](methods/nonnegative_cp_decomposition.md) | 潜在加性非负模式合理 | 波形归一化非负不是物理依据 |
| [Tucker](methods/tucker_decomposition.md) | 独立模容量与密集核交互 | 核乘法成本，无默认局部/时间先验 |
| [非负 Tucker](methods/nonnegative_tucker_decomposition.md) | 独立模秩与非负交互均合理 | 约束可能欠拟合，不是现成光谱解混 |
| [BTD](methods/block_term_decomposition.md) | 多个全局成分优于单核 | block 指分解项，不是局部 patch |
| [t-SVD](methods/t_svd_decomposition.md) | 当前第三模上的频域低秩合理 | 不是 TNN 求解器，FFT 轴/循环边界重要 |
| [层次 Tucker](methods/hierarchical_tucker_decomposition.md) | 联合前两模对特征模的瓶颈可压缩 | 固定浅树，默认特征叶 F²，未必省参数 |
| [TT](methods/tensor_train_decomposition.md) | 固定顺序的两道展开瓶颈可压缩 | 三模链，依赖顺序，不自动高阶张量化 |
| [TR](methods/tensor_ring_decomposition.md) | 小环秩耦合值得比较 | 核成本随 R² 增大，环闭合不是时间插值 |

## 缺失类型、缺失率与可识别性

random 对非音频逐元素遮挡，audio 按真实时间点共同遮挡所有声道。
block 对非音频遮 N 维紧凑块，audio 遮连续真实时间段。
Recovery 的 slices（兼容 sildes）默认是 Image 完整行、MSI 完整波段、Video 完整帧；
audio 同样是连续波形缺段。其他入口使用不同切片轴时，应按实际 mask 解释。

自由索引因子在某索引完全无观测时，可能没有数据损失梯度：如 CP/Tucker/TT/TR/HT 的未知行、
波段、帧系数，或 mode3 在全特征缺失位置的潜向量。低秩与非负不自动识别这些参数，
需额外跨索引参数化、邻接约束或其他先验。t-SVD 的 FFT 耦合不同，也不保证任意切片恢复。
偏置/初始化给出的值不能当作从观测唯一识别的证据。

缺失率需结合模式、覆盖、观测量、参数量与欠拟合/不稳定解释。实际缺失率≥0.7仅为高缺失的
启发式提醒，不是恢复阈值、秩公式或家族排序。相同缺失率的随机散点与完整行/波段/帧缺失不同。
不能假定大缺块一定选 Tucker、高频一定增秩，或层次 Tucker 因浅树而不值得预赛。

## 统计与训练协议

profile 相关性只采样部分联合特征；元素级 mask 下部分统计使用可见特征均值填补。
缺失连通块是二维投影，平滑/高频分数来自空间邻接或预览，不是时间频谱、完整光谱统计或真实张量秩。
audio 不启用图像纵横比、空间平滑、RGB 相关性推荐；MSI/Video 也只将这些统计作为弱线索。

训练梯度只用真实观测值，audio padding 不参与。GT 缺失区结果可用于选 checkpoint/参数和进化反馈；
这不是训练/验证/最终测试三分，也不是旧的留出观测像素选秩。Recovery 以同类全部有效样本的
算术平均决定预赛/晋级：audio 仅缺失原始波形 NMSE（低优），非音频按当前 PSNR/SSIM 协议。
这是当前数据的开发成绩，不自动证明独立数据泛化或 SOTA。

## 检索与证据

按真实 data_type（兼容 color_image/msi/video）提供所有 11 个方法的当前类型简短适用卡片，来源可引用。
retrieval_top_k 限制额外详细段落：先不同方法，再补第二段。各卡片不重复作为详细证据发送。
规则 data_types 限定类型，权重只是弱短名单先验，不是准确率、理论胜率或实测成绩。
手动选择、同预算数值预赛、指标、进化主干和冠军归档机制不变。

静态参考库与全局经验库不同：后者来自完整运行实践总结，并标注模态/基础方法、缺失率、模式、范围、预算。
静态假设、跨运行条件性经验与本次实测对照不能互相冒充。

## 理论参考与实现边界

以下原始来源支持模型背景；四类适用建议是结合本项目代码的工程推断，不是论文承诺的固定排名。

- [Kolda、Bader：CP/Tucker 与非负模型综述](https://www.kolda.net/publication/koba09/)
- [De Lathauwer：BTD 定义](https://epubs.siam.org/doi/10.1137/070690729)
- [Grasedyck：HT 格式](https://epubs.siam.org/doi/10.1137/090764189)
- [Oseledets：TT 格式](https://epubs.siam.org/doi/10.1137/090752286)
- [Zhao 等：TR 格式](https://arxiv.org/abs/1606.05535)
- [Zhang、Aeron：t-SVD 随机采样补全](https://arxiv.org/abs/1502.04689)
- [Kim、Choi：非负 Tucker](https://www.cs.cmu.edu/~ftorre/ca/ca_final_version/p16.pdf)

实现位于 `../core/models/`。自/交叉注意力、多尺度空洞卷积、连续时间/波段生成器是待测候选机制，
不是基础模型默认能力；应明确增强哪个因子/核心，并用整类、逐样本与消融证据验证。
