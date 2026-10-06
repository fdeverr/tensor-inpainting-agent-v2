# Nonnegative Tucker Decomposition / 非负 Tucker

## Model and implementation

当前 core 和三个因子均经 softplus，正特征 scale 从观测均值初始化。参数量 `D1Rh+D2Rw+FRc+RhRwRc+F`。独立模秩与非负加性交互，没有稀疏、丰度和为一或时间平滑约束，不等于论文完整优化算法。

## Image applicability

Image：非负加性成分合理且需要独立空间/颜色秩时，可与普通 Tucker、非负 CP 比较。非负亮度不能证明最优潜表示非负，不能抵消可能损失复杂纹理灵活性；RGB Rc≤3 仅为这一尺寸限制。

## MSI applicability

MSI：非负强度/加性光谱模式支持试验，Rc 依波段数和拟合设置。非负约束不保证物理端元可识别，也不提供波长连续性；完整波段 slices 的自由因子行仍需跨波段结构约束。

## Video applicability

Video：非负共享空间-联合帧颜色模式的受约束候选，Rc 上限 `T·C`。不是四阶非负 Tucker，没有时间正则/光流/前景分离；复杂运动可能不适合少量正成分，整帧未知系数不能靠非负直接识别。

## audio applicability

audio：当前是有符号波形，非负缩放值不构成非负潜成分的物理依据。单声道 Rc=1，实际是受约束帧矩阵参数化，不是非负谱分解。可按同预算 NMSE 试验，但不应凭归一化非负优先推荐，需检查瞬态、相位和长缺段。

## Rank and budget

`Rh≤D1,Rw≤D2,Rc≤F`，当前空间网格 `{4,8,16}`、特征网格 `{1,2,3,4,8,16}` 按尺寸截取。核大小乘法增长，softplus 与不能抵消的约束可能加重欠拟合或延缓优化。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

对 random、block、slices 分别验证，完全缺失索引仍无数据支持。高缺失率与非负约束可能限制容量也可能欠拟合，不预判净收益。可测试空间/真实时间耦合，非负收益需单独消融。

## Evidence boundary

实现依据 `core/models/nonnegative_tucker.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [非负 Tucker 原文](https://www.cs.cmu.edu/~ftorre/ca/ca_final_version/p16.pdf)。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
