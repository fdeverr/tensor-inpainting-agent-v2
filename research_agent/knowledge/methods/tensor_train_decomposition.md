# Tensor Train Decomposition / TT 链式分解

## Model and implementation

`X[i,j,f]=Σ_ab G1[i,a]G2[a,j,b]G3[b,f]+bias[f]`，固定三模链 `D1→D2→F`。分解参数量 `D1R1+R1D2R2+R2F`，另有 F 个偏置。没有高阶 tensorization、模顺序搜索或直接 TT-SVD 求解。

## Image applicability

Image：顺序展开可用较小秩表示、预算受限时，是 CP/Tucker 之外的重要候选。R1 控制第一模对其余模，R2 控制前两模对 RGB。模式顺序与各向异性一起检查，不因颜色少就排除，也不能保证复杂图像上最优。

## MSI applicability

MSI：F 为波段，R2 可以超过 3，可测试空间-光谱瓶颈与链参数效率。很多波段不意味着更多独立模，当前仍三阶。整波段 slices 时末核对应列无数据约束，没有自动生成未知光谱系数的邻接机制。

## Video applicability

Video：末核表示联合时间-颜色 `T·C`，可探索空间到跨帧模式的低秩连接。不是四阶 TT，没有独立时间核、光流或时间平滑。运动/场景切换可能提高展开秩，整帧缺失需额外时间约束末核列。

## audio applicability

audio：链为时间帧→帧内采样→声道。单声道 R2=1，是低秩帧矩阵的另一种参数化。帧长影响周期对齐和矩阵秩，不能直接宣传一般高阶 TT 压缩优势；整帧缺失时首核行无数据约束，需真实时间耦合。

## Rank and budget

`R1≤min(D1,D2F),R2≤min(D1D2,F)`。当前 R1 网格 `{4,8,16,32}`，R2 `{1,2,3,4,8,16}` 按尺寸截取。RGB R2≤3，MSI/Video 不固定≤3；不同家族 rank 不等容量。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

小秩可能沿链方向平滑，大秩可能不稳。完整缺失索引不是散点 random，缺失率不能单独确定 R1/R2。末核时间/光谱连续参数化、因子注意力、尺度正则都是后续待测方向。

## Evidence boundary

实现依据 `core/models/tensor_train.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [TT 原文](https://epubs.siam.org/doi/10.1137/090752286)。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
