# A Mode-3 E Factorization / 联合特征模分解

## Model and implementation

`X[i,j,f]=Σ_r A[i,j,r]E[f,r]+bias[f]`，即 `A ×₃ E`。A 是每个 `(i,j)` 独立的自由潜向量，E 为联合特征因子。分解参数量 `D1D2R+FR`，另有 F 个偏置。没有内置空间/时间正则。

## Image applicability

Image：颜色子空间确实低维时可作消融控制组；RGB rank 1/2 会压缩颜色耦合。所有颜色在同一像素共同缺失时，该位置 A 没有数据梯度，不能靠颜色相关性完成空间外推。不应仅因 RGB 高相关而置于空间耦合的 CP/Tucker 前。

## MSI applicability

MSI：很多波段共用低维光谱子空间、每像素仍有足够部分波段观测时值得试验。共同空间缺块需额外空间先验。完整波段 slices 时对应 E 行与偏置没有数据约束；低光谱秩本身不是沿波长连续生成的替代品。

## Video applicability

Video：E 表示联合时间-颜色模式，可测试近静态/重复场景的子空间假设，但不是独立时间因子模型。空间点在部分帧可见有利于约束 A；整帧缺失会使 E 行无观测。光流、运动轨迹和相邻帧平滑都未显式建模。

## audio applicability

audio：F 为声道，单声道 rank=1 时 A 几乎逐采样自由，不压缩帧内/跨帧结构。当前 mask 同时遮挡所有声道，缺失采样点 A 无数据约束，多声道相关性也不能直接补它。通常作为控制组或带真实时间生成器的待测进化起点。

## Rank and budget

`rank≤min(D1D2,F)`，当前网格 `{1,2,4,8,16,32}` 按尺寸截取。RGB 上限 3，MSI/Video 不能硬套 1–3，单声道仅 1。提高 rank 不会为完全无观测的自由参数产生数据约束。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

“每位置全特征缺失”与“完整特征切片缺失”是不同退化。random 元素缺失可能允许跨特征估计，但不保证潜向量可识别。可测试连续坐标生成 A/E、邻接正则或注意力，不能把增强能力记在基础模型名下。

## Evidence boundary

实现依据 `core/models/mode3_factorization.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [张量模型背景](https://www.kolda.net/publication/koba09/)。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
