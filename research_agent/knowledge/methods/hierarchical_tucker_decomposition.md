# Hierarchical Tucker Decomposition / 层次 Tucker

## Model and implementation

固定树 `((D1,D2),F)`：两个叶经 `Rh×Rw×Rs` transfer 与 `Rs×Rc` 根 transfer 耦合特征叶。分解参数量 `D1Rh+D2Rw+FRc+RhRwRs+RsRc`，另有 F 个偏置。Rs=`rank_spatial` 是跨组瓶颈；没有自动树搜索或视频四独立叶。

## Image applicability

Image：联合空间-颜色耦合可压缩、要比较密集核与层次 transfer 时应列入候选。RGB 的 Rs 通常不超过 3，但浅树不代表一定弱于普通 Tucker；不能仅凭三个模就排除层次方法，实际胜负看同预算预赛。

## MSI applicability

MSI：波段多时 Rc/Rs 范围比 RGB 大，可测试低维联合空间-光谱瓶颈。默认特征叶 Rc=F 不表示没有跨组压缩，Rs 仍是瓶颈；但特征叶成本可为 F²。完整波段 slices 的未知叶行仍无数据约束。

## Video applicability

Video：固定树是空间 `(H,W)` 对联合 `(T·C)`，不是多层空间-时间-颜色树。可以比较分层与密集耦合的容量/优化差别，但没有显式时间生成器。长序列需看特征叶成本，快速运动可能需要更大 Rs。

## audio applicability

audio：分组为 `(时间帧,帧内采样)` 对声道。单声道 `Rc=Rs=1`，树浅，不能宣称自动多尺度时间建模。仍可按不同参数化/优化行为与矩阵、CP、Tucker 比较，而非凭名字排除。整帧缺失需真实时间耦合。

## Rank and budget

`Rh≤D1,Rw≤D2,Rc≤F,Rs≤min(RhRw,Rc)`。当前 Rh/Rw 网格 `{4,8,16}`、Rc 固定 `[F]`、Rs `{1,2,4,8}` 按 `min(D1D2,F)` 截取。F 大时特征叶 F² 很贵，不能笼统说层次模型必然省参数。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

Rs 太小会压扁跨组细节，叶秩大可能加重内存和优化。random 有覆盖不代表 slices 可识别，无观测叶索引仍不受数据项约束。transfer/因子注意力和连续特征叶是待测增强，不是基础模型已有能力。

## Evidence boundary

实现依据 `core/models/hierarchical_tucker.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [HT 理论原文](https://epubs.siam.org/doi/10.1137/090764189)。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
