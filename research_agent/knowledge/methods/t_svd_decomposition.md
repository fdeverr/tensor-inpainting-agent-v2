# t-SVD / Low-Tubal-Rank Decomposition

## Model and implementation

当前在第三个联合特征模 F 上 rFFT，各频率学习 `U_f diag(S_f)V_f^H`，irFFT 后加偏置。实值分解参数量 `R·F·(D1+D2+1)`，另有 F 个偏置。因子无显式正交约束，不是张量核范数/TNN 凸优化求解器。

## Image applicability

Image：RGB 为 F=3，仅两个实 FFT 频点，可作另一种耦合参数化候选。颜色次序不是连续光谱/时间，不能描述成已获物理频率先验。与 Tucker/CP 的胜负需当前同预算实测，不因 t-SVD 名称而保证保边补块更强。

## MSI applicability

MSI：F 为波段，有序波段和共同空间模式支持测试 tubal-rank 假设。FFT 循环耦合首尾波段，实际光谱未必周期或低 tubal rank。完整波段 slices 必须实测，频域低秩不能保证任意波段缺失可恢复。

## Video applicability

Video：灰度 C=1 时 F=T，FFT 真正沿时间；RGB 时 F=`T·C`，时间-颜色交织，不等于逐颜色时间轴 t-SVD。重复/近周期序列值得试验，运动、场景切换、循环边界和完整帧 slices 需检查，不能直接移植论文视频优势。

## audio applicability

audio：FFT 轴是声道，不是时间。单声道 F=1 时退化成帧矩阵低秩参数化，没有时间 FFT/STFT，也没有隐藏相位恢复。它可因参数化数值胜出，但只能按缺失波形 NMSE 解释；不能把收益归因于未实现的时间频率机制。

## Rank and budget

`rank≤min(D1,D2)`，当前网格 `{2,4,8,16}` 按尺寸截取。成本随 F 线性增加；长视频/多波段不一定比 Tucker 紧凑。rank 为频域矩阵秩上界，不是 Tucker 通道秩或时间频率数。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

随机采样的理论条件不能推广到 block/slices，缺失率不是充分判据。注意 FFT 轴语义、循环边界、频域欠拟合和未收敛。真实时间/波段连续化或更合适的变换可作待测增强，当前不自动轴置换或学习变换。

## Evidence boundary

实现依据 `core/models/t_svd.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [t-SVD 补全原文](https://arxiv.org/abs/1502.04689)讨论满足条件的随机采样与核范数方法，不为本项目任意缺失模式背书。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
