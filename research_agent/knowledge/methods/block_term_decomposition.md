# Block-Term Decomposition / 多 Tucker 项分解

## Model and implementation

当前 BTD 为 `Σ_n G_n×₁A_n×₂B_n×₃C_n+bias`，每项有独立因子和核。分解参数量 `N(D1Rh+D2Rw+FRc+RhRwRc)`，另有 F 个偏置，N≤4。block 指分解项，不是缺失 block，不自动切图/切音频，也不包含所有 BTD 变体。

## Image applicability

Image：若多个空间-颜色模式比单个小核更合适，可测试 N=1 与 N>1。异质/重复纹理只是线索，冗余项可能浪费容量。不能因为出现大缺块就选 BTD，它不是局部 patch 模型。

## MSI applicability

MSI：多个空间-光谱成分、单一小核欠拟合时可入候选。没有端元非负、丰度和为一等物理约束，不是现成的光谱解混。波段多时每项特征因子增加成本，完整波段 slices 也不会因多项而自动可识别。

## Video applicability

Video：可探索多组共享空间-联合帧颜色成分，没有自动前景分离、运动跟踪或时间段定位。F=`T·C`，不是四阶 BTD；快速变化会提高项数/秩需求，整帧缺失仍需额外时间约束。

## audio applicability

audio：可以测试多组重复波形模式；单声道实际是多个低秩帧矩阵之和，常常只是增加总矩阵容量。不能据此宣称多声源分离或多尺度建模。瞬态、帧相位变化及长缺段仍难，最终只用真实缺失波形 NMSE。

## Rank and budget

当前 N 网格 `{1,2,3}`，Rh/Rw `{4,8,16}`，Rc `{1,2,3,4,8,16}` 按维度截取。N≤4，Rh≤D1、Rw≤D2、Rc≤F。项数与核大小同时增加可能超预算；N=1 可作为容量对照。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

每项仍是全局自由索引因子，random、block、slices 均需验证，无观测索引不会因多项而获得信息。常见冗余组件、尺度漂移和参数爆炸。跨项交叉注意力、多尺度因子与连续模参数化是待测方向。

## Evidence boundary

实现依据 `core/models/block_term.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [BTD 定义原文](https://epubs.siam.org/doi/10.1137/070690729)。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
