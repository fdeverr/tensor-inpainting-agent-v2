# Tucker Decomposition / 核张量与独立模秩

## Model and implementation

`X≈G×₁A×₂B×₃C+bias`，分解参数量 `D1Rh+D2Rw+FRc+RhRwRc`，另有 F 个偏置。当前只有三个分解模：Image/MSI 是高×宽×颜色/波段，Video 为高×宽×`(时间·颜色)`，audio 是帧数×帧长×声道。独立模秩不意味着内置局部或时间平滑。

## Image applicability

Image：空间两个方向复杂度不同、颜色子空间较低维时值得尝试，用不等空间秩分配容量。大 block 可测试全局耦合，但不能凭“核灵活”保证胜过 CP、层次 Tucker 或插值。高频纹理需比较容量、成本与训练曲线。

## MSI applicability

MSI：独立光谱秩 Rc 可表达波段共同子空间，不固定为 RGB 的 1–3。部分波段观测支持跨谱共享，但材质变化需核容量。整条波段 slices 完全无观测时，其光谱因子行不可由数据项识别，低秩不能替代波长连续生成。

## Video applicability

Video：共享空间基和较低维联合时间-颜色模式是可测试假设。Rc 上限 `T·C`，不是颜色数；当前不是四阶 Tucker。场景切换和快速运动可能提高联合秩，整帧缺失仍需额外时间耦合来约束未知帧系数。

## audio applicability

audio：Rh 控制跨帧变化，Rw 控制帧内波形，Rc 控制声道；单声道 Rc=1。可测试重复片段的低秩结构，但实际是矩阵结构，不享有 RGB 光谱压缩优势。非平稳波形、整帧长缺段需谨慎，相邻帧平滑不是默认能力。

## Rank and budget

`Rh≤D1,Rw≤D2,Rc≤F`。当前空间网格 `{4,8,16,32}`，特征网格 `{1,2,3,4,8,16}` 按尺寸截取。核大小乘法增长，秩不由缺失率单独确定，需同时看参数量、欠拟合和优化稳定性。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

随机散点成绩不能推广到完整波段/帧 slices。可能出现核过大、细节欠拟合、无观测因子行和收敛不足。因子连续化、交叉注意力、多尺度空洞卷积可作后续假设，须明确增强位置并保留分解主干。

## Evidence boundary

实现依据 `core/models/tucker.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [Tucker/CP 理论背景](https://www.kolda.net/publication/koba09/)。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
