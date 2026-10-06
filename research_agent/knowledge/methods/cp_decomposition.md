# CP Decomposition / 典范多元分解

## Model and implementation

`X[i,j,f]=Σ_r A[i,r]B[j,r]C[f,r]+bias[f]`，分解参数量 `R(D1+D2+F)`，另有 F 个偏置。三个模共享 rank-one 组件，尺度不唯一、优化非凸。当前学习自由因子，不是 ALS 求解器；Video 合并时间颜色，audio 前两模是帧数、帧内采样。

## Image applicability

Image：可分离的重复全局模式、有限参数预算时值得列入短名单，与 Tucker 独立模秩形成容量对照。random 且各因子索引有观测支持时可试验；复杂局部边缘不一定有小 CP 秩。颜色相关性低不直接证明 CP 比 Tucker 好。

## MSI applicability

MSI：空间-光谱模式可用少量可分离成分近似时是低成本候选。光谱相关不等于小 CP 秩或可解释端元。复杂材质变化下共享 rank 可能过于刚性，完整波段 slices 的 C 行仍无数据约束。

## Video applicability

Video：可探索重复空间组件和联合帧-颜色系数。复杂运动可能需要大量可分离成分；F=`T·C`，没有独立时间核、光流或时间邻接先验。整帧缺失不能仅靠无观测的 C 行恢复。

## audio applicability

audio：重复/近周期帧波形支持小 rank 试验。单声道是矩阵低秩的另一种参数化，不享有一般高阶 CP 的额外优势。帧间相位漂移、瞬态和非平稳信号可能需更多组件；整帧缺失时对应 A 行无数据约束。仅用缺失波形 NMSE。

## Rank and budget

当前网格 `{4,8,12,16,24}`；构造函数接受正 rank，选择器推荐限制为 1–64。CP rank 不能简单截断到最小模长度，也不能把相同数字当作和 Tucker/TT 等容量。检查因子尺度和收敛，不靠观测 loss 单独判优。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

random 通常提供更均匀的索引支持，但不是恢复保证。完全无观测的空间行、波段、帧系数不受数据项识别。高缺失率加 rank 可能加重不适定。连续因子、自注意力或多尺度正则都是待测进化方向。

## Evidence boundary

实现依据 `core/models/cp.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [CP/Tucker 理论背景](https://www.kolda.net/publication/koba09/)。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
