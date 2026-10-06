# Tensor Ring Decomposition / TR 环式分解

## Model and implementation

`X[i,j,f]=trace(G1[:,i,:]G2[:,j,:]G3[:,f,:])+bias[f]`，三个核共用环秩 R。分解参数量 `R²(D1+D2+F)`，另有 F 个偏置。优化非凸、有规范自由度；环耦合不意味着任意维度置换等价，也不自动高阶重排或时间邻接。

## Image applicability

Image：小 CP/TT 对跨模交互欠拟合时，可尝试小环秩的不同容量。重复纹理只是试验假设，大环秩在短预算内可能难收敛。需要结合参数量比较 TT/CP，而不是相同数字 rank。

## MSI applicability

MSI：可以探索多种空间-光谱耦合，F 为真实波段数。没有光谱连续/物理解混约束，完整波段 slices 对应自由核切片无观测，不能凭环闭合宣称可恢复未知波段。

## Video applicability

Video：特征核为 `T·C`，可比较比小 TT 更丰富的耦合，但仍只有三个核。没有独立时间核，不能由一般高阶 TR 理论保证整帧缺失/运动补全。长视频核内存随 F 和 R² 增大。

## audio applicability

audio：帧数×帧内采样×声道三核环。单声道可以表达矩阵结构，环秩不是时间频率数。重复帧波形可支持试验，瞬态/相位漂移需更多容量；完全缺失帧的自由核切片无数据约束，环闭合不是跨帧插值规则。

## Rank and budget

`1≤R≤min(16,D1,D2)`，当前网格 `{2,4,6,8}` 按尺寸截取。参数量随 R² 增大，rank 相关计算也增长，不能由紧凑表示直接断言速度优势。结合实际 runtime、参数量与收敛比较。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

random、block、slices 分开验证，可能出现条纹、细节丢失、尺度漂移。高缺失率/无观测索引需要结构先验，不是简单加 rank。因子注意力、真实时间生成器、多尺度修正是待测增强。

## Evidence boundary

实现依据 `core/models/tensor_ring.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [TR 原文](https://arxiv.org/abs/1606.05535)讨论环格式与循环排列性质，不保证固定三核的全类型优胜。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
