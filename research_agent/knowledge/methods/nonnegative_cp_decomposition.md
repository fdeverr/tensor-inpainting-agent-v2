# Nonnegative CP Decomposition / 非负 CP

## Model and implementation

softplus 生成非负 A/B/C，输出为非负 rank-one 项之和；正特征 scale 由观测均值初始化。参数量 `R(D1+D2+F)+F`。组件不能抵消，不是普通 CP 结果截断，也没有稀疏、丰度和为一等物理混合约束。

## Image applicability

Image：非负加性成分是合理先验时可作普通 CP 的受约束对照。像素非负仅是兼容条件，不证明最优潜因子非负；不能抵消可能限制复杂颜色/纹理。是否有益看当前缺失区实测。

## MSI applicability

MSI：非负辐射/反射强度、加性光谱成分支持试验，但不是自动端元/丰度解混。归一化到非负也不能证明物理因子非负；中心化/有符号数据需谨慎。完整波段 slices 的未知因子仍不可由非负性唯一识别。

## Video applicability

Video：可对非负重复亮度模式作受约束试验，F=`T·C`。运动差异/带符号变化未必适合加性成分；没有自动背景前景分离。整帧无观测因子不因非负而获得时间插值约束。

## audio applicability

audio：目标是有符号原始波形，不是非负幅度谱。归一化后数值非负只是表示变换，不能作为优先推荐非负 CP 的物理理由。可参与同预算试验，但要撤销归一化以缺失波形 NMSE 比较，不使用 PSNR/SSIM 论证波形适用性。

## Rank and budget

当前网格 `{4,8,12,16,24}`，选择器推荐正 rank≤64。softplus 饱和、冗余正成分可能慢收敛，提高 rank 也不能替代抵消能力。与普通 CP 相同预算比较参数与质量。

训练只用真实观测值；GT 缺失区反馈选配置/checkpoint。Recovery 以当前同类全部有效样本的平均指标确定基础家族与进化晋级；audio 仅 NMSE，其余按当前 PSNR/SSIM 协议。不是旧的留出观测像素选秩流程。

## Missingness and failure modes

random 仍需索引数据支持，block/slices 的无观测自由因子不因非负而可识别。高缺失率下过强约束或小 rank 可能欠拟合。连续因子/局部先验可作增强，是否保留非负需消融，不能归因于整套候选成绩。

## Evidence boundary

实现依据 `core/models/nonnegative_cp.py`。四类适用建议是条件性工程推断，不是已验证成绩或固定排序；参考 [非负张量模型背景](https://www.kolda.net/publication/koba09/)。理论背景不能替代本次同条件整类预赛；不保证任一缺失率或类型上的最优。
