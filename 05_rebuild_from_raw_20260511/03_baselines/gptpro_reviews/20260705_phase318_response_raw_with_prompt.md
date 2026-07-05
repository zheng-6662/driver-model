# 给本地高级模型的追问：第317版门控失败后，第318版应如何修正

请你继续作为机器学习实验顾问，基于第317版最新失败结果，给出下一版第318版的可执行修正方案。请用中文回答，重点给出能在当前代码库里快速实现的方案，不要泛泛建议。

## 任务边界

我们做的是车辆方向盘变化曲线预测。模型要预测锚点后 0 到 2 秒内的方向盘变化曲线。用户强调：样本必须由当前窗口方向盘快速转动引起，不能把后续 2 到 6 秒才发生的大动作错当成当前任务。

固定约束：

1. 不使用测试集误差做特征、调参、选模或阈值选择。
2. 不把锚点后真实曲线作为部署输入。
3. 可以在训练阶段用真实 0 到 2 秒曲线构造监督目标，例如最优候选、峰值、相位误差，但预测时不能输入这些未来信息。
4. 第315版已经固定当前窗口保留清单，训练、验证、测试保留事件分别为 650、211、222，84 个隔离事件不参与训练、验证选模或主测试统计。
5. 第316版是基础预测模型。
6. 第317版已经跑完，但验证失败，没有报告测试集。

## 第317版方法

第317版固定第316版基础预测，构造 20 条候选曲线：

1. 原预测不改。
2. 幅值缩放：0.85、1.15、1.30、1.50、1.75。
3. 时间平移：提前 0.40 秒、提前 0.25 秒、提前 0.10 秒、延后 0.10 秒、延后 0.25 秒、延后 0.40 秒。
4. 幅值加时间组合：1.30 配提前 0.25 秒、1.30 配延后 0.25 秒、1.50 配提前 0.25 秒、1.50 配延后 0.25 秒。
5. 4 个训练集残差原型候选。

门控输入只包含：

1. 锚点前车辆信号统计特征。
2. 第316版预测曲线摘要。
3. 可部署的粗场景标签。

门控模型训练后比较候选加权输出和候选单选输出。

## 第317版验证结果

第317版守卫通过，但验证门槛全部未通过，固定方案为“随机森林加候选单选”，因此没有报告测试集。

关键验证结果：

| 方法 | 分组 | 数量 | 平均误差 | 相对第316版变化 | 大退化比例 | 严重低估比例 | 原预测保持率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 第316版基础预测 | 全部样本 | 211 | 0.531658 | 0 | 0 | 0.157635 | 空 |
| 第316版基础预测 | 普通样本 | 89 | 0.405874 | 0 | 0 | 0.132530 | 空 |
| 第316版基础预测 | 强方向盘样本 | 112 | 0.624528 | 0 | 0 | 0.169643 | 空 |
| 第316版基础预测 | 困难前20 | 45 | 0.810938 | 0 | 0 | 0.232558 | 空 |
| 第316版基础预测 | 困难前10 | 24 | 1.002263 | 0 | 0 | 0.363636 | 空 |
| 候选最优上限 | 全部样本 | 211 | 0.375611 | -0.156047 | 0 | 0.064039 | 0.033175 |
| 候选最优上限 | 普通样本 | 89 | 0.276733 | -0.129140 | 0 | 0.036145 | 0.044944 |
| 候选最优上限 | 强方向盘样本 | 112 | 0.446074 | -0.178454 | 0 | 0.080357 | 0.026786 |
| 候选最优上限 | 困难前20 | 45 | 0.625778 | -0.185160 | 0 | 0.116279 | 0 |
| 候选最优上限 | 困难前10 | 24 | 0.766937 | -0.235327 | 0 | 0.181818 | 0 |
| 实际门控方案 | 全部样本 | 211 | 0.586667 | +0.055009 | 0.511848 | 0.113300 | 0.028436 |
| 实际门控方案 | 普通样本 | 89 | 0.490111 | +0.084238 | 0.629213 | 0.096386 | 0.011236 |
| 实际门控方案 | 强方向盘样本 | 112 | 0.652488 | +0.027960 | 0.410714 | 0.125000 | 0.044643 |
| 实际门控方案 | 困难前20 | 45 | 0.831892 | +0.020954 | 0.333333 | 0.139535 | 0.022222 |
| 实际门控方案 | 困难前10 | 24 | 1.033731 | +0.031468 | 0.375000 | 0.227273 | 0.041667 |

验证门槛情况：

1. 全部样本退化不超过 0.5%：失败。
2. 普通样本退化不超过 1.0%：失败。
3. 强方向盘改善至少 3.0%：失败。
4. 困难前20改善至少 5.0%：失败。
5. 困难前10改善至少 8.0%：失败。
6. 幅值严重低估比例下降至少 20%：只有部分单选模型通过。
7. 全部样本大退化比例不超过 15%：失败。
8. 普通样本大退化比例不超过 10%：失败。

候选使用情况说明：

- 候选最优上限明显优于第316版，说明候选库本身有价值。
- 实际门控方案原预测保持率极低，全部样本只有 2.84%，普通样本只有 1.12%。
- 门控过度修改普通样本，造成普通样本大退化比例达到 62.9%。
- 实际门控选择和候选最优明显不一致。例如残差原型4在验证集最优次数为 25，但实际只选 1 次；原预测不改最优次数为 7，实际只选 6 次，且普通样本保持率很低。

## 当前失败分型

我们认为第317版属于：

候选库有上限，但门控选不准；同时门控过度修改普通样本。

也就是说，下一步不应该优先扩大候选库，也不应该报告测试集，而应该先修门控机制。

## 想请你回答的问题

请给出第318版可执行方案，重点回答：

1. 第318版应该如何把“是否需要校正”和“选哪个候选”拆成两段式？
2. 如何设计一个保守门控，让普通样本默认保持第316版原预测不改？
3. 在只看训练和验证、不碰测试集的前提下，如何选择校正阈值？
4. 是否应该把第317版的候选最优上限转成“可校正性判别器”，先预测某个样本是否值得校正？如果应该，输入、标签和损失怎么定义？
5. 如何避免门控把每个样本都强行分到某个候选，导致普通样本大面积退化？
6. 第318版应该保留候选单选、候选加权，还是采用“原预测加小幅残差”的保守融合？
7. 请给出一个最小可执行第318版实验方案，包括训练目标、验证门槛、候选使用规则、失败分流和必须输出的表格。

请注意：第318版最好是第317版脚本的增量修改，不要建议大规模重写主模型，也不要建议人工逐个复核。
我会按“第317版脚本的增量修改”来设计第318版，不动第316主模型，也不扩候选库。核心是把门控从“强制选候选”改成“先判定是否值得校正，再在小集合内保守应用”。
重点会落到可直接写成训练标签、阈值搜索、候选规则和输出表格；默认保持原预测，只在高置信、低退化风险样本上校正。
结论
缺少拒绝机制
第318版不要再做“20候选强制分类”。第317版失败的核心不是候选库无效，而是
，导致普通样本被大面积强行修改。
第318版应固定为：
第316版原预测作为默认输出；只有样本被判定为“高置信可校正”时，才进入候选选择；候选选择后也不直接全量替换，而是采用“原预测 + 小幅残差融合”。
建议命名为：
v318_conservative_two_stage_gate
核心结构：
有问题，尽管问
第316原预测
   ↓
阶段1：是否值得校正？ correctability gate
   ↓ 否
保持第316原预测
   ↓ 是
阶段2：哪个候选最安全、预期收益最大？
   ↓
原预测 + α × (候选曲线 - 原预测)
1. 第318版如何拆成“是否校正”和“选哪个候选”
第317版相当于直接做：
输入特征 → 20候选之一
这会天然导致每个样本都被分到某个候选，普通样本也被修改。
第318版改成两段：
阶段A：可校正性判别器
只回答一个问题：
这个样本是否值得离开第316原预测？
输出：
p_corr = P(该样本值得校正)
g_hat  = 预测可获得的误差下降量
其中 
 是分类概率，
 是预测收益。
p_corr
g_hat
阶段B：候选收益/风险选择器
只在阶段A通过时运行。
对每个非原始候选 
 预测：
k=1...19
d_hat[k]   = 预测候选k相对第316原预测的误差下降量
bad_hat[k] = 候选k导致大退化的概率
然后选择：
k* = argmax d_hat[k]
但必须满足：
d_hat[k*] 足够大
bad_hat[k*] 足够低
d_hat[k*] 相比原预测和第二候选有足够 margin
否则仍然保持原预测。
关键点是：
阶段B没有资格强制输出非原始候选。原预测是吸收态。
2. 如何设计保守门控，让普通样本默认保持第316版
第318版的默认规则应非常明确：
Python
final_pred
=
pred_v316
只有满足所有条件时才允许修改。
推荐门控规则
对每个样本 
：
i
Python
allow_correct
=
 (
p_corr
[
i
] 
>=
theta_p
[
group
]
and
g_hat
[
i
] 
>=
theta_g
[
group
]
and
max_d_hat
[
i
] 
>=
theta_d
[
group
]
and
bad_hat
[
i
, 
best_k
] 
<=
theta_bad
[
group
]
and
margin_to_original
[
i
] 
>=
theta_margin
[
group
]
)
其中 
 只能使用部署时可获得的粗场景标签，例如：
group
ordinary
strong_steering
不能使用“困难前20”“困难前10”作为部署输入，因为它们通常依赖真实误差，只能用于验证统计。
普通样本使用更严格阈值
建议初始网格：
同时增加硬约束：
普通样本原预测保持率 ≥ 80%，建议优先尝试 ≥ 85%
普通样本校正率 ≤ 15%，最多不超过 20%
全部样本校正率 ≤ 45%
这一步直接针对第317版失败点：
第317普通样本原预测保持率只有 1.12%
第317普通样本大退化比例达到 62.9%
第318版必须把普通样本的默认行为改成：
没有强证据 → 不动
而不是：
一定选一个候选
3. 不碰测试集时，如何选择校正阈值
推荐使用：
训练集 OOF 选阈值，验证集只做一次最终判定，测试集不碰。
不要直接在验证集上反复调阈值，否则验证集会被“调参污染”。
具体流程
训练集有 650 个保留事件。
对这 650 个训练事件做 event-wise 5-fold OOF：
fold1: 训练4折 → 预测剩余1折
fold2: 训练4折 → 预测剩余1折
...
得到训练集每个样本的 OOF 输出：
p_corr_oof
g_hat_oof
d_hat_oof[k]
bad_hat_oof[k]
然后只在这些 OOF 结果上搜索阈值。
阈值选择目标
先设硬约束：
全部样本平均误差相对第316版退化 ≤ 0.5%
普通样本平均误差相对第316版退化 ≤ 1.0%
全部样本大退化比例 ≤ 15%
普通样本大退化比例 ≤ 10%
普通样本原预测保持率 ≥ 80% 或 85%
普通样本校正率 ≤ 15% 或 20%
在满足硬约束的阈值中，最大化：
强方向盘样本改善
困难前20改善
困难前10改善
严重低估比例下降
可以定义一个简单评分：
Python
score
=
 (
1.0
*
improve_strong
+
1.0
*
improve_hard20
+
1.0
*
improve_hard10
+
0.5
*
severe_underestimate_drop
)
阈值选定后，冻结：
theta_p
theta_g
theta_d
theta_bad
theta_margin
alpha
correction_rate_cap
然后重新在完整训练集上训练阶段A和阶段B模型，或者直接使用 5-fold ensemble，对验证集做一次评估。
验证集规则
验证集 211 个事件只允许做：
pass / fail 判定
如果验证失败：
不报告测试集
不根据测试集改阈值
不碰84个隔离事件
4. 是否应该把候选最优上限转成“可校正性判别器”
应该。
第317版结果已经说明：
候选最优上限明显优于第316版
实际门控明显劣于第316版
所以问题不是“有没有候选能救”，而是“能不能提前判断这个样本是否值得救”。
训练标签定义
设：
Python
e0
[
i
] 
=
第316原预测误差
ek
[
i
, 
k
] 
=
第k个候选误差，k
=
1
...
19
注意，这里的误差只允许使用锚点后 0 到 2 秒真实曲线，不能使用 2 到 6 秒。
定义非原始候选最优误差：
Python
best_non0_err
[
i
] 
=
min
(
ek
[
i
, 
1
:
20
])
best_non0_k
[
i
]   
=
argmin
(
ek
[
i
, 
1
:
20
])
oracle_gain
[
i
]   
=
e0
[
i
] 
-
best_non0_err
[
i
]
定义可校正标签：
Python
y_corr
[
i
] 
=
1
if
oracle_gain
[
i
] 
>=
max
(
delta_abs
, 
delta_rel
*
e0
[
i
]) 
else
0
建议初始：
delta_abs = 0.03
delta_rel = 0.05
对普通样本可以更严格：
ordinary_delta_abs = 0.05
ordinary_delta_rel = 0.08
也可以设置灰区，减少标签噪声：
Python
positive
: 
oracle_gain
>=
max
(
0.03
, 
0.05
*
e0
)
negative
: 
oracle_gain
<=
max
(
0.01
, 
0.02
*
e0
)
gray
zone
: 
降低权重或不参与分类器训练
阶段A输入
只能用部署时可获得的信息：
1. 锚点前车辆信号统计特征
2. 第316版预测曲线摘要
3. 可部署粗场景标签
4. 候选曲线相对第316预测的确定性摘要
第4类不是未来信息，因为候选曲线由第316预测确定变换得到。
可以快速加入这些摘要：
候选最大幅值与原预测最大幅值差
候选峰值时刻与原预测峰值时刻差
候选曲线面积差
候选最大斜率差
候选与原预测的L2距离
候选类型：缩放 / 平移 / 缩放+平移 / 残差原型
但阶段A最好先用样本级摘要，不一定展开成候选级。
阶段A损失
如果当前代码库用 sklearn，推荐先用随机森林或梯度提升，不需要大改。
逻辑目标为：
分类：预测 y_corr
回归：预测 oracle_gain
等价损失：
L_A = weighted_BCE(y_corr, p_corr)
    + λ * Huber(oracle_gain, g_hat)
实际代码中可以拆成两个模型：
Python
corr_clf
=
RandomForestClassifier
(...)
gain_reg
=
RandomForestRegressor
(...)
普通样本负例要加大权重：
Python
sample_weight
=
1.0
if
group
==
"ordinary"
and
y_corr
==
0
:
sample_weight
=
3.0
if
group
==
"strong_steering"
and
y_corr
==
1
:
sample_weight
=
2.0
if
base_is_severe_underestimate
:
sample_weight
*=
1.5
目的很明确：
宁可少改普通样本，也不要把普通样本误判为需要校正。
5. 如何避免每个样本被强行分到某个候选
第318版必须加入三个机制。
机制1：显式 no-action 类
原预测不再只是候选之一，而是默认动作：
action = keep_original
非原始候选只有在通过门控时才能覆盖默认动作。
机制2：候选收益必须大于原预测
第317版的问题是候选分类器会在 20 个候选中选一个，即使所有候选都不可靠。
第318版应该改成：
Python
if
max
(
d_hat
) 
<=
0
:
keep
original
更保守：
Python
if
max
(
d_hat
) 
<
theta_d
:
keep
original
其中：
Python
d_hat
[
k
] 
=
预测候选k相对原预测的误差下降量
机制3：大退化风险过滤
对每个候选训练一个风险判断：
Python
bad_label
[
i
, 
k
] 
=
1
if
candidate_k
causes
large_degrade
relative
to
v316
else
0
使用第317版已有的大退化定义，不要重新造一个指标。
候选必须满足：
Python
bad_hat
[
k
] 
<=
theta_bad
否则不允许选择。
6. 第318版应该保留候选单选、候选加权，还是保守残差融合
固定方案建议：
候选单选决定方向；
原预测加小幅残差决定最终输出；
候选加权只作为消融，不作为第318固定方案。
即：
Python
selected_candidate
=
candidate
[
k
]
final_pred
=
pred_v316
+
alpha
*
 (
selected_candidate
-
pred_v316
)
其中：
alpha = 0.25 / 0.50 / 0.75
由训练集 OOF 选，不用测试集。
为什么不建议直接候选加权作为主方案
第317版已经暴露出一个问题：
门控不准时，加权输出会让大量样本偏离原预测
候选加权如果没有强约束，本质上还是“每个样本都被修改”。
如果一定保留加权，只能作为对照实验，并且要加原预测权重下限：
普通样本：w_original ≥ 0.80
强方向盘样本：w_original ≥ 0.50
但第318版主方案不建议用它。
推荐主方案
v318_fixed = 两段式门控 + 候选收益选择 + 大退化风险过滤 + 小幅残差融合
其中：
不是 pure candidate single-select
也不是 free candidate weighted
而是 selected-candidate residual shrinkage
7. 最小可执行第318版实验方案
下面是可以直接在第317版脚本上增量修改的方案。
7.1 保留第317版候选库
不扩候选。
继续使用 20 条候选：
0. 原预测不改
1-5. 幅值缩放
6-11. 时间平移
12-15. 幅值+时间组合
16-19. 残差原型候选
第318版重点不在候选库，而在：
是否允许离开候选0
7.2 新增训练标签
在训练集上计算：
Python
err
[:, 
0
]      
# 第316原预测误差
err
[:, 
1
:
20
]   
# 19个非原始候选误差
生成：
Python
best_non0_k
best_non0_err
oracle_gain
y_corr
candidate_gain
[
k
]
candidate_bad
[
k
]
伪代码：
Python
base_err
=
err
[:, 
0
]
cand_err
=
err
[:, 
1
:]
best_non0_idx
=
np
.
argmin(
cand_err
, 
axis
=
1
) 
+
1
best_non0_err
=
cand_err
[
np
.
arange(
len
(
err
)), 
best_non0_idx
-
1
]
oracle_gain
=
base_err
-
best_non0_err
delta_abs
=
0.03
delta_rel
=
0.05
y_corr
=
 (
oracle_gain
>=
np
.
maximum(
delta_abs
, 
delta_rel
*
base_err
)
)
.
astype(
int
)
candidate_gain
=
base_err
[:, 
None
] 
-
err
[:, 
1
:]
candidate_bad
=
np
.
zeros_like(
candidate_gain
, 
dtype
=
int
)
for
k
in
range
(
1
, 
20
):
candidate_bad
[:, 
k
-
1
] 
=
is_large_degrade
(
candidate_err
=
err
[:, 
k
],
base_err
=
base_err
    )
其中 
 直接复用第317版的大退化定义。
is_large_degrade
7.3 训练阶段A：可校正性模型
输入：
Python
X_sample
包含第317版门控输入：
锚点前车辆信号统计特征
第316预测曲线摘要
可部署粗场景标签
训练两个模型：
Python
corr_clf
.
fit(
X_sample
, 
y_corr
, 
sample_weight
=
w
)
gain_reg
.
fit(
X_sample
, 
oracle_gain
, 
sample_weight
=
w
)
建议参数先保守：
Python
RandomForestClassifier
(
n_estimators
=
500
,
max_depth
=
4
,
min_samples_leaf
=
20
,
class_weight
=
None
,
random_state
=
seed
,
)
RandomForestRegressor
(
n_estimators
=
500
,
max_depth
=
4
,
min_samples_leaf
=
20
,
random_state
=
seed
,
)
再加概率校准：
Python
CalibratedClassifierCV
(
corr_clf
,
method
=
"sigmoid"
,
cv
=
5
)
小样本下不建议先用 isotonic，sigmoid 更稳。
7.4 训练阶段B：候选收益/风险模型
把训练数据展开成长表：
一行 = 一个样本 i + 一个候选 k
输入：
Python
X_long
=
 [
X_sample_i
,
candidate_type_k
,
candidate_scale_k
,
candidate_shift_k
,
candidate_proto_id_k
,
candidate_summary_delta_i_k
]
目标：
Python
target_gain_i_k
=
base_err_i
-
err_i_k
target_bad_i_k
=
1
if
candidate
k
causes
large
degradation
else
0
训练：
Python
cand_gain_reg
.
fit(
X_long
, 
target_gain
)
cand_bad_clf
.
fit(
X_long
, 
target_bad
)
建议：
Python
cand_gain_reg
=
RandomForestRegressor
(
n_estimators
=
500
,
max_depth
=
5
,
min_samples_leaf
=
30
,
random_state
=
seed
,
)
cand_bad_clf
=
RandomForestClassifier
(
n_estimators
=
500
,
max_depth
=
5
,
min_samples_leaf
=
30
,
class_weight
=
"balanced_subsample"
,
random_state
=
seed
,
)
大退化风险模型里，大退化样本权重要更高：
Python
if
target_bad
==
1
:
sample_weight
*=
5.0
因为第318版的首要任务是避免第317版那种普通样本大面积退化。
7.5 第318版候选使用规则
伪代码如下：
Python
def
apply_v318_policy
(
pred_v316
,
candidates
,
X_sample
,
scene_group
,
p_corr
,
g_hat
,
d_hat
,
bad_hat
,
theta
,
):
final_pred
=
pred_v316
.
copy()
selected_k
=
np
.
zeros(
len
(
pred_v316
), 
dtype
=
int
)
alpha_used
=
np
.
zeros(
len
(
pred_v316
))
for
i
in
range
(
len
(
pred_v316
)):
group
=
scene_group
[
i
]  
# ordinary or strong_steering
# 默认保持原预测
k_final
=
0
# 阶段A：是否允许校正
if
p_corr
[
i
] 
<
theta
[
f"p_
{
group
}
"
]:
continue
if
g_hat
[
i
] 
<
theta
[
f"g_
{
group
}
"
]:
continue
# 阶段B：候选选择
valid
=
 []
for
k
in
range
(
1
, 
20
):
if
d_hat
[
i
, 
k
] 
<
theta
[
f"d_
{
group
}
"
]:
continue
if
bad_hat
[
i
, 
k
] 
>
theta
[
f"bad_
{
group
}
"
]:
continue
valid
.
append(
k
)
if
len
(
valid
) 
==
0
:
continue
best_k
=
max
(
valid
, 
key
=lambda
k
: 
d_hat
[
i
, 
k
])
# 相比原预测必须有 margin
if
d_hat
[
i
, 
best_k
] 
<
theta
[
f"margin_
{
group
}
"
]:
continue
# 通过所有门控，才允许修改
alpha
=
theta
[
f"alpha_
{
group
}
"
]
final_pred
[
i
] 
=
pred_v316
[
i
] 
+
alpha
*
 (
candidates
[
i
, 
best_k
] 
-
pred_v316
[
i
]
        )
selected_k
[
i
] 
=
best_k
alpha_used
[
i
] 
=
alpha
return
final_pred
, 
selected_k
, 
alpha_used
注意：
selected_k = 0 表示保持第316原预测。
第318版必须统计 
 的比例，尤其是普通样本。
selected_k=0
8. 阈值网格建议
建议先用小网格，不要搞太大。
Python
grid
=
 {
"p_ordinary"
:      [
0.80
, 
0.85
, 
0.90
],
"p_strong"
:        [
0.55
, 
0.65
, 
0.75
],
"g_ordinary"
:      [
0.05
, 
0.07
, 
0.09
],
"g_strong"
:        [
0.03
, 
0.05
, 
0.07
],
"d_ordinary"
:      [
0.04
, 
0.06
, 
0.08
],
"d_strong"
:        [
0.02
, 
0.04
, 
0.06
],
"bad_ordinary"
:    [
0.05
, 
0.08
, 
0.10
],
"bad_strong"
:      [
0.10
, 
0.15
],
"margin_ordinary"
: [
0.02
, 
0.03
],
"margin_strong"
:   [
0.01
, 
0.02
],
"alpha_ordinary"
:  [
0.25
, 
0.50
],
"alpha_strong"
:    [
0.25
, 
0.50
, 
0.75
],
}
再加两个校正率硬约束：
Python
ordinary_correction_rate
<=
0.15
all_correction_rate
<=
0.45
如果没有任何阈值组合通过，可以放宽为：
ordinary_correction_rate ≤ 0.20
ordinary_keep_rate ≥ 0.80
但不建议再放宽普通样本大退化比例。
9. 第318版验证门槛
第318版继续沿用第317版验证门槛，不通过就不报告测试集。
第318版额外增加两个门槛：
这两个新增门槛是针对第317版失败原因设计的。
10. 必须输出的表格
第318版脚本至少输出以下表格。
表1：候选最优上限与可校正标签分布
目的：确认候选库仍然有价值，并检查普通样本中到底有多少是真正值得校正的。
表2：阶段A可校正性判别器表现
重点看：
普通负例误触发率
如果这个值高，第318版仍会重蹈第317版覆辙。
表3：阶段B候选选择表现
其中：
top1一致率 = 预测候选是否等于 oracle best candidate
top3一致率 = oracle best candidate 是否在预测收益前三候选中
如果 top1 很差但 top3 尚可，可以考虑下一版做 family-level selector。
表4：最终验证主结果
格式保持第317版一致：
至少包括：
第316版基础预测
候选最优上限
第317版实际门控方案
第318版两段式保守门控
表5：候选使用分布
重点看残差原型4。
第317版中：
残差原型4验证集最优25次，但实际只选1次
第318版需要确认它是否被合理恢复使用。
表6：失败分流表
11. 第318版的最小实验矩阵
不要一次跑太多。建议固定三组：
v318-A：只加可校正门控
阶段A通过 → 使用第317版原来的候选单选
阶段A不通过 → 保持第316
目的：判断第317失败主要是不是“普通样本误触发”。
v318-B：可校正门控 + 候选收益选择
阶段A通过 → 用 d_hat 选择候选
阶段A不通过 → 保持第316
目的：验证候选选择器是否比第317的20类分类更稳。
v318-C：可校正门控 + 候选收益选择 + 残差融合
final = pred_v316 + alpha * (candidate - pred_v316)
这是建议的第318版固定候选。
优先顺序：
先看 v318-C
再用 v318-A / v318-B 做归因
不要从验证集里挑一个看起来最好的版本作为最终版本。版本选择规则应在训练 OOF 上确定。
12. 第318版通过/失败后的处理
如果第318-C通过全部验证门槛
才允许进入测试集报告。
并且测试集只报告一次，不允许根据测试结果回改阈值。
如果第318-C失败但普通样本安全
例如：
普通样本不退化
普通大退化合格
但强方向盘/困难样本改善不足
说明保守门控有效，但召回不足。
下一版应该只放宽强方向盘样本阈值，不要放宽普通样本阈值。
如果第318-C普通样本仍失败
说明阶段A的可校正性判别器还没学会“不要动普通样本”。
下一版不应该继续改候选库，而应该：
提高普通负例权重
提高 p_ordinary
降低 ordinary correction cap
降低 alpha_ordinary
如果阶段A通过但阶段B选错
看表3和表5。
如果：
top1一致率低
top3一致率尚可
下一版改成候选族选择：
原预测
幅值缩放类
时间提前类
时间延后类
缩放+时间类
残差原型类
先选族，再在族内选候选。这样比直接20类分类稳。
13. 第318版最关键的实现原则
第318版只需要在第317版候选评估脚本后面加一层策略，不需要重写第316主模型。
必须坚持以下原则：
1. 第316原预测是默认输出。
2. 候选库不扩展。
3. 先判定是否值得校正，再选候选。
4. 候选选择必须有 reject option。
5. 普通样本使用更严格阈值和校正率上限。
6. 最终输出采用原预测到候选之间的小幅残差融合。
7. 阈值用训练集 OOF 选择，验证集只做 pass/fail。
8. 验证失败不报告测试集。
第318版的目标不是追求候选最优上限，而是先把第317版的灾难性问题消掉：
普通样本过度修改
大退化比例过高
原预测保持率过低
只要第318版能做到：
普通样本基本不退化
大退化比例降回门槛内
强方向盘和困难样本有稳定改善
它就是比第317版正确的方向。
到目前为止，这段对话对你有帮助吗？
