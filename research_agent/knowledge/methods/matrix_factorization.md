# Matrix Factorization / 矩阵低秩分解

## Model and implementation

当前把 `X ∈ R^(D1×D2×F)` 展开为 `D1×(D2·F)`，学习 `UV+bias`。分解参数量 `R(D1+D2F)`，另有 F 个偏置。Image/MSI 的前两模为高、宽；Video 的 F=`T·C`；audio 为帧数×帧长×声道。这是固定展开的低秩拟合，不自动选择展开方向。

## Image applicability

Image：行与其余维度的展开近似低秩、重复全局结构明显，或需要简单低成本控制组时值得入选。random 且每行/展开列保留观测时可试验；细纹理和大 block 不能仅靠低秩保证恢复。非正方形不是自动优势，关键是展开方向是否匹配结构。

## MSI applicability

MSI：可比较空间展开的低秩假设，但宽度和波段合并，没有独立光谱秩。若主要结构在低维光谱子空间，需与 mode3/Tucker 比较。完整波段 slices 会使对应展开列无观测，自由列因子不可由数据项识别。

## Video applicability

Video：使用 `H×(W·T·C)`，不是 `HW×T` 的逐帧矩阵。重复空间模式支持试验；运动、遮挡、场景切换可能提高有效秩。整帧 slices 会产生无观测列，没有显式时间参数化/平滑，不能声称已有自动插帧能力。

## audio applicability

audio：单声道实际是帧数×帧长矩阵，相似、对齐或近周期片段支持小秩试验。瞬态、非平稳波形和帧长不合适时可能欠拟合。长 block/slices 覆盖整帧时，其行因子无数据约束；帧长是表示超参数，不是采样率。仅按真实缺失波形 NMSE 判优。

## Rank and budget

`rank≤min(D1,D2F)`，当前网格 `{4,8,16,32}` 按尺寸截取去重。不同家族的同名 rank 不代表同容量。小秩可能欠拟合，大秩可能加重不适定，查看参数量与逐样本收敛。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

random、block、slices 的可识别性不同，整行/整列缺失比相同缺失率的散点更困难。高缺失率不是固定选小秩或大秩的依据。可能的进化方向是空间/真实时间因子的连续参数化或邻接正则，均属待验证假设，不是已有能力。

## Evidence boundary

实现依据 `core/models/matrix_factorization.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [理论背景](https://www.kolda.net/publication/koba09/)。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
