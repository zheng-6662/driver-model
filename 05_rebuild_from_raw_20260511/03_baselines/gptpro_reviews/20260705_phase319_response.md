## 结论

第318版失败不是“候选库没有价值”，而是**门控搜索目标把“完全不改”当成了安全最优解**。第319版第一优先级不是重训主预测模型，也不是扩候选库，而是把阈值搜索改成：

> **有最低激活约束的分组风险预算搜索**
> 即：普通样本必须严保护，强方向盘/困难样本必须被激活一部分；全不改直接判为不可行方案。

同时，第319版不要继续只靠绝对阈值，例如 `p_min >= 0.90`、`gain_min >= 0.08`。绝对阈值容易在验证集分布上清空候选，导致第318版这种“训练集看似安全，验证集一个都不改”的情况。第319版应引入**固定比例 Top-K 配额门控**，尤其在强方向盘和困难代理样本上强制选择一部分最高分样本。

---

# 1. 第319版怎么改训练集内部阈值搜索目标

## 1.1 不再用“通过项数”作为主目标

第318版的问题是：

```text
零校正 = 普通样本不坏 + 大退化为0 + 守卫通过
```

所以搜索器自然选择全不改。

第319版中，以下情况必须直接判为不可行：

```python
if corr_rate_all < 0.02:
    reject_cfg()

if corr_rate_strong < 0.05:
    reject_cfg()

if corr_rate_hard20 < 0.08:
    reject_cfg()

if gain_strong <= 0:
    reject_cfg()

if gain_hard20 <= 0:
    reject_cfg()
```

也就是说，**只保护但不激活，不能算通过**。

---

## 1.2 用“硬约束 + 加权收益目标”替代单纯安全目标

建议在训练集 5 折 OOF 上搜索配置时，定义：

```python
gain_g = mean(err_v316[g] - err_new[g])
bad_g = mean((err_new[g] - err_v316[g]) > BAD_DELTA)
corr_g = mean(is_corrected[g])
```

其中：

```python
BAD_DELTA = 0.05
```

如果你当前脚本已有“大退化”定义，继续沿用原定义即可，不必另起定义。

第319版推荐硬约束如下：

```python
# 收益约束
gain_all      >= 0.002
gain_normal   >= -0.002
gain_strong   >= 0.006
gain_hard20   >= 0.010
gain_hard10   >= -0.002   # hard10 样本少，可先设为不明显劣化

# 大退化约束
bad_all       <= 0.015
bad_normal    <= 0.005
bad_strong    <= 0.030
bad_hard20    <= 0.040

# 校正覆盖率约束
corr_all      >= 0.030
corr_all      <= 0.120
corr_normal   <= 0.030
corr_strong   >= 0.080
corr_hard20   >= 0.100
```

然后在满足硬约束的配置里，最大化：

```python
score = (
    1.0 * gain_all
  + 1.5 * gain_strong
  + 2.0 * gain_hard20
  + 0.8 * gain_hard10
  - 3.0 * bad_all
  - 8.0 * bad_normal
)
```

解释：

* `gain_all` 防止整体被改坏；
* `gain_strong` 和 `gain_hard20` 是主目标；
* `gain_hard20` 权重最高，因为当前主线就是修严重方向/意图错误；
* `bad_normal` 惩罚最重，因为第317版主要问题就是普通样本被大面积改坏。

---

## 1.3 加一个“宽松备用层”，但仍然禁止全不改

如果严格约束找不到任何可行配置，不要退回全不改，而是进入二级约束：

```python
gain_all      >= 0.000
gain_normal   >= -0.003
gain_strong   >= 0.004
gain_hard20   >= 0.006

bad_all       <= 0.020
bad_normal    <= 0.008

corr_all      >= 0.020
corr_all      <= 0.150
corr_normal   <= 0.040
corr_strong   >= 0.050
corr_hard20   >= 0.080
```

仍然保留：

```python
if corr_all == 0:
    reject_cfg()

if corr_strong == 0:
    reject_cfg()

if corr_hard20 == 0:
    reject_cfg()
```

第319版必须把“完全不改”从搜索空间里删除。

---

# 2. 是否设置最低校正覆盖率

应该设置，而且这是第319版的关键。

## 推荐覆盖率范围

| 分组     | 训练 OOF 最低校正率 |       验证集最低校正率 |     最大校正率 | 说明                     |
| ------ | -----------: | -------------: | --------: | ---------------------- |
| 全部样本   |           3% |             2% |       12% | 防止全不改，也防止第317版式过改      |
| 普通样本   |         不设最低 |           不设最低 |   3% 到 4% | 普通样本只保护，不主动追求改善        |
| 强方向盘样本 |           8% |             5% | 20% 到 25% | 当前任务核心样本之一             |
| 困难前20  |          10% |             8% | 25% 到 30% | 用于训练搜索和验证评价，不可直接作为部署输入 |
| 困难前10  |          仅诊断 | 建议 ≥ 5%，或至少不劣化 |       35% | 样本数可能较少，不建议第一版设过硬      |

需要特别注意：

> “困难前20 / 困难前10”通常是由真实误差定义的，不能在验证集或部署时直接用于门控决策。
> 它们只能作为训练 OOF 搜索约束和验证评价指标。

因此第319版不能写成：

```python
if sample in hard20:
    use_loose_gate()
```

这属于标签泄漏。

正确做法是训练或构造一个**困难代理分数**，例如 `hard_proxy_score`，用它近似捕捉困难样本。

---

# 3. 是否拆成“普通样本保护”和“困难样本激活”双通道

应该拆。

第318版把所有样本放进同一个保守门槛，结果就是：

```text
门槛足够严 → 普通样本安全 → 强方向盘/困难样本也全被挡掉
```

第319版建议使用双通道：

## 通道 A：普通样本保护通道

目标：普通样本基本不动。

建议规则：

```python
normal_channel = (
    ~strong_steer_mask
    & (hard_proxy_score < hard_proxy_q80)
)
```

普通通道使用严格门槛：

```python
p_correctable      >= 0.95
stage1_gain_pred   >= 0.10
candidate_gain_pred >= 0.10
bad_prob           <= 0.35
margin             >= 0.05
```

并且设置硬上限：

```python
corr_rate_normal <= 0.03
```

普通样本没有最低校正率。

---

## 通道 B：强方向盘 / 困难代理激活通道

目标：必须激活一部分潜在严重错误样本。

建议规则：

```python
hard_channel = (
    strong_steer_mask
    | (hard_proxy_score >= hard_proxy_q80)
)
```

这里的 `strong_steer_mask` 必须来自锚点前或当前窗口可观测信息，例如当前窗口方向盘转角速度、方向盘角变化量等。

`hard_proxy_score` 不能用未来真实曲线，建议用以下特征训练或合成：

```python
hard_proxy_score =
    0.25 * rank01(p_correctable)
  + 0.25 * rank01(stage1_gain_pred)
  + 0.25 * rank01(best_candidate_gain_pred)
  + 0.15 * rank01(candidate_disagreement)
  + 0.10 * rank01(current_window_steer_rate_abs)
  - 0.20 * rank01(best_bad_prob)
```

如果脚本里已经方便训练模型，更推荐训练一个轻量 `hard20_proxy_classifier`：

```python
target_hard20 = err_v316_train >= percentile(err_v316_train, 80)
```

输入只允许使用部署时可用特征：

```text
当前窗口方向盘特征
锚点前车辆信号
第316版预测曲线摘要
第317版候选曲线摘要
第318版第一段 p_correctable / gain_pred
第318版第二段 candidate_gain_pred / bad_prob / margin
```

不能输入：

```text
真实 0 到 2 秒曲线
真实误差
真实 hard20 标签
真实候选最优编号
```

---

# 4. 第一段模型和第二段模型，下一版先修哪个？

第319版先不要优先修模型。

优先顺序应该是：

```text
第1优先级：改阈值搜索目标和配额门控
第2优先级：加 hard_proxy 或困难代理分数
第3优先级：再诊断第一段/第二段模型谁失效
```

原因很直接：第318版验证集一个样本都没改，说明还没进入“候选选得准不准”的阶段。现在最先要解决的是**门控不激活**。

第319版跑完后，用下面规则决定下一步修哪个模型。

## 4.1 如果困难样本进不来，修第一段 / hard_proxy

检查训练 OOF：

```python
oracle_good = (err_v316 - err_oracle_best) > 0.05
```

看第一段或 hard_proxy 的前 20% 高分样本里，是否富集 `oracle_good`。

建议阈值：

```python
recall_oracle_good_strong_at20 < 0.35
recall_oracle_good_hard20_at20 < 0.35
```

如果低于这个水平，说明第一段“可校正性 / 困难代理”没有找到该改的样本，应修第一段或 hard_proxy。

---

## 4.2 如果样本进来了但候选选错，修第二段

检查：

```python
top1_candidate_gain / oracle_best_gain
```

建议诊断标准：

```python
mean(top1_candidate_gain / oracle_best_gain) < 0.35
```

或者：

```python
bad_rate_selected_candidates > 0.25
```

如果进入门控的样本很多，但选中的候选经常不是收益候选，则修第二段候选收益模型和候选风险模型。

---

# 5. 是否引入验证集前固定的候选风险预算、收益预算或分组配额

应该引入。

第319版建议固定三类预算。

## 5.1 风险预算

训练 OOF 搜索时：

```python
bad_all    <= 0.015
bad_normal <= 0.005
bad_strong <= 0.030
bad_hard20 <= 0.040
```

同时增加一个“正损失预算”：

```python
pos_loss_g = mean(max(err_new[g] - err_v316[g], 0))
```

建议：

```python
pos_loss_normal <= 0.003
pos_loss_all    <= 0.006
```

这样可以防止虽然“大退化比例”不高，但很多样本被轻微改坏。

---

## 5.2 收益预算

训练 OOF 上要求：

```python
gain_strong >= 0.006
gain_hard20 >= 0.010
```

验证集上要求略低或相同：

```python
gain_strong >= 0.008
gain_hard20 >= 0.012
```

这里验证集略高，是因为第319版的目标本来就是修强方向盘和困难样本；如果这两项仍然没有明显改善，第319版没有存在价值。

---

## 5.3 分组配额

建议固定：

```python
corr_rate_all     in [0.03, 0.12]
corr_rate_normal <= 0.03
corr_rate_strong >= 0.08   # OOF
corr_rate_hard20 >= 0.10   # OOF评价，不作为部署输入
```

验证集：

```python
corr_rate_all     in [0.02, 0.12]
corr_rate_normal <= 0.04
corr_rate_strong >= 0.05
corr_rate_hard20 >= 0.08
```

这样可以防止第319版再次回到原预测保持率为 1 的状态。

---

# 6. 第319版保留残差融合，还是做困难样本候选单选？

建议采用：

> **困难 / 强方向盘通道允许候选单选，普通通道只允许残差融合或不改。**

不要所有样本都单选，也不要所有样本都残差融合。

第319版推荐规则：

```python
if hard_channel and high_confidence:
    alpha = 1.00      # 候选单选
elif hard_channel:
    alpha = 0.75      # 较强残差融合
elif normal_channel:
    alpha = 0.50      # 普通样本最多半融合
else:
    alpha = 0.00      # 不改
```

`high_confidence` 建议定义为：

```python
high_confidence = (
    candidate_gain_pred >= 0.08
    and bad_prob <= 0.40
    and margin >= 0.05
)
```

对于普通样本，不建议 `alpha=1.0`。

普通通道建议：

```python
alpha_normal in {0.25, 0.50}
```

困难/强方向盘通道建议：

```python
alpha_hard in {0.75, 1.00}
```

第319版的主线应该让困难样本有真实改变幅度，否则很可能继续“安全但没改善”。

---

# 7. 可直接落地的第319版方案

## 7.1 固定实验名

建议脚本名：

```text
stage03_v319_dual_channel_quota_gate_20260705.py
```

输出目录：

```text
v319_dual_channel_quota_gate_20260705
```

实验中文名：

```text
第319版-双通道配额激活门控
```

---

## 7.2 第319版整体流程

```text
1. 复用第316版原预测。
2. 复用第317版候选库。
3. 复用第318版第一段可校正性模型输出：
   - p_correctable
   - stage1_gain_pred

4. 复用第318版第二段候选收益/风险输出：
   - candidate_gain_pred
   - bad_prob
   - margin

5. 新增 hard_proxy_score：
   - 优先训练 hard20_proxy_classifier
   - 或用现有分数合成 hard_proxy_score

6. 用训练集 5 折 OOF 输出搜索：
   - hard_proxy 分位阈值
   - 强方向盘校正配额
   - 困难代理校正配额
   - 普通样本最大校正率
   - alpha 融合幅度

7. 固定搜索出的配置。
8. 在验证集一次性评估。
9. 验证通过才报告测试集。
```

---

## 7.3 候选选择分数

对每个样本，先排除原预测候选，选择最优非原始候选：

```python
candidate_score = (
    0.35 * rank01(candidate_gain_pred)
  + 0.25 * rank01(stage1_gain_pred)
  + 0.20 * rank01(p_correctable)
  + 0.15 * rank01(margin)
  - 0.35 * rank01(bad_prob)
)
```

然后：

```python
best_candidate = argmax(candidate_score over non_original_candidates)
```

保留原预测只作为 fallback，不参与“最佳候选”竞争。

---

## 7.4 hard_proxy_score

推荐先训练一个轻量 hard20 代理分类器：

```python
y_hard20 = err_v316_train >= np.percentile(err_v316_train, 80)
```

模型可以用：

```python
HistGradientBoostingClassifier(
    max_iter=200,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    learning_rate=0.05,
    random_state=seed
)
```

输入特征：

```text
p_correctable
stage1_gain_pred
best_candidate_gain_pred
best_bad_prob
best_margin
candidate_gain_pred 的 max / mean / std
bad_prob 的 min / mean
候选曲线峰值分布
候选曲线终点分布
第316版预测峰值
第316版预测终点
当前窗口方向盘角速度峰值
当前窗口方向盘角变化量
当前窗口方向盘角加速度峰值
```

训练方式必须是 5 折 OOF：

```python
for fold in folds:
    train hard20_proxy on train_fold
    predict p_hard20_proxy on valid_fold
```

验证集则用全训练集训练后的模型预测。

如果暂时不想新增模型，可以用合成分数：

```python
hard_proxy_score = (
    0.25 * rank01(p_correctable)
  + 0.25 * rank01(stage1_gain_pred)
  + 0.25 * rank01(best_candidate_gain_pred)
  + 0.15 * rank01(candidate_disagreement)
  + 0.10 * rank01(current_window_steer_rate_abs)
  - 0.20 * rank01(best_bad_prob)
)
```

但优先建议训练 `hard20_proxy_classifier`，因为第319版需要明确提高困难前20覆盖率。

---

## 7.5 双通道门控规则

### 强/困难通道

```python
hard_channel = (
    strong_steer_mask
    | (p_hard20_proxy >= hard_proxy_threshold)
)
```

其中：

```python
hard_proxy_threshold in {
    q70, q75, q80, q85
}
```

训练 OOF 搜索时用训练集分位数，验证集应用固定分位阈值对应的数值。

强/困难通道的宽松安全底线：

```python
eligible_hard = (
    hard_channel
    & (best_bad_prob <= bad_floor_hard)
    & (best_candidate_gain_pred >= gain_floor_hard)
    & (margin >= margin_floor_hard)
)
```

搜索范围：

```python
bad_floor_hard in {0.70, 0.80, 0.90}
gain_floor_hard in {-0.02, 0.00, 0.02}
margin_floor_hard in {-0.03, -0.01, 0.00}
```

然后不是继续用绝对阈值清空样本，而是做 Top-K：

```python
selected_strong = top_k_by_score(
    eligible_hard & strong_steer_mask,
    candidate_score,
    k = ceil(r_strong * n_strong)
)

selected_proxy_hard = top_k_by_score(
    eligible_hard & ~strong_steer_mask & (p_hard20_proxy >= hard_proxy_threshold),
    candidate_score,
    k = ceil(r_proxy_hard * n_total)
)
```

搜索范围：

```python
r_strong in {0.05, 0.08, 0.10, 0.12, 0.15}
r_proxy_hard in {0.02, 0.04, 0.06, 0.08}
```

---

### 普通通道

```python
normal_channel = (
    ~strong_steer_mask
    & (p_hard20_proxy < hard_proxy_threshold)
)
```

普通通道必须严格：

```python
eligible_normal = (
    normal_channel
    & (p_correctable >= p_normal_min)
    & (stage1_gain_pred >= gain_normal_min)
    & (best_candidate_gain_pred >= candidate_gain_normal_min)
    & (best_bad_prob <= bad_prob_normal_max)
    & (margin >= margin_normal_min)
)
```

搜索范围：

```python
p_normal_min              in {0.90, 0.95, 0.98}
gain_normal_min           in {0.08, 0.10, 0.12}
candidate_gain_normal_min in {0.08, 0.10, 0.12}
bad_prob_normal_max       in {0.30, 0.40, 0.50}
margin_normal_min         in {0.03, 0.05, 0.08}
normal_corr_cap           in {0.00, 0.01, 0.02, 0.03}
```

普通通道只取最高分，且不强制填满：

```python
selected_normal = top_k_by_score(
    eligible_normal,
    candidate_score,
    k = floor(normal_corr_cap * n_normal)
)
```

---

## 7.6 融合规则

```python
selected = selected_strong | selected_proxy_hard | selected_normal
```

对强/困难通道：

```python
high_confidence = (
    best_candidate_gain_pred >= 0.08
    and best_bad_prob <= 0.40
    and margin >= 0.05
)

if high_confidence:
    alpha = 1.00
else:
    alpha = 0.75
```

对普通通道：

```python
alpha = 0.50
```

最终预测：

```python
pred_new = pred_v316.copy()

pred_new[selected] = (
    pred_v316[selected]
    + alpha[selected] * (
        pred_candidate_best[selected] - pred_v316[selected]
    )
)
```

---

## 7.7 OOF 配置选择伪代码

```python
best_cfg = None
best_score = -inf

for cfg in grid:

    pred_new, selected, selected_channel = apply_gate_on_oof(cfg)

    metrics = compute_metrics(
        pred_new=pred_new,
        pred_base=pred_v316,
        y_true=y_train,
        groups={
            "all": mask_all,
            "normal": mask_normal,
            "strong": mask_strong,
            "hard20": mask_hard20_train,
            "hard10": mask_hard10_train,
        }
    )

    # ---------- 硬拒绝 ----------
    if metrics.corr_all < 0.030:
        continue
    if metrics.corr_all > 0.120:
        continue
    if metrics.corr_normal > 0.030:
        continue
    if metrics.corr_strong < 0.080:
        continue
    if metrics.corr_hard20 < 0.100:
        continue

    if metrics.gain_all < 0.002:
        continue
    if metrics.gain_normal < -0.002:
        continue
    if metrics.gain_strong < 0.006:
        continue
    if metrics.gain_hard20 < 0.010:
        continue
    if metrics.gain_hard10 < -0.002:
        continue

    if metrics.bad_all > 0.015:
        continue
    if metrics.bad_normal > 0.005:
        continue
    if metrics.bad_strong > 0.030:
        continue
    if metrics.bad_hard20 > 0.040:
        continue

    # ---------- 主目标 ----------
    score = (
        1.0 * metrics.gain_all
      + 1.5 * metrics.gain_strong
      + 2.0 * metrics.gain_hard20
      + 0.8 * metrics.gain_hard10
      - 3.0 * metrics.bad_all
      - 8.0 * metrics.bad_normal
    )

    if score > best_score:
        best_score = score
        best_cfg = cfg
```

如果找不到可行配置，进入二级约束，但仍然禁止全不改：

```python
if best_cfg is None:
    run_relaxed_search_but_forbid_zero_correction()
```

---

# 8. 第319版输出表

至少输出以下 7 张表。

## 表1：主结果表

同第318版，但增加校正率：

```text
方法
分组
平均误差
相对第316变化
大退化比例
校正率
原预测保持率
```

分组：

```text
全部样本
普通样本
强方向盘样本
困难前20
困难前10
幅值低估样本
```

---

## 表2：通道贡献表

```text
通道
样本数
校正数
校正率
平均收益
大退化比例
正损失均值
候选单选比例
残差融合比例
```

通道：

```text
strong_channel
proxy_hard_channel
normal_channel
all_corrected
```

---

## 表3：候选选择分布表

```text
候选类型
被选次数
被选比例
平均收益
大退化比例
强方向盘占比
困难前20占比
```

候选类型包括：

```text
原预测
幅值缩放
时间平移
幅值+时间组合
残差原型
```

这张表用于判断到底是哪类候选在起作用。

---

## 表4：覆盖率预算表

```text
分组
最低校正率
实际校正率
最大校正率
是否通过
```

分组：

```text
全部样本
普通样本
强方向盘样本
困难前20
困难前10
```

---

## 表5：风险预算表

```text
分组
bad_rate_limit
bad_rate_actual
pos_loss_limit
pos_loss_actual
是否通过
```

---

## 表6：OOF 与验证集对照表

```text
指标
训练OOF
验证集
差值
是否稳定
```

指标：

```text
gain_all
gain_normal
gain_strong
gain_hard20
corr_all
corr_normal
corr_strong
corr_hard20
bad_all
bad_normal
```

如果 OOF 改善明显、验证集完全不改，说明仍然存在阈值迁移问题。

---

## 表7：hard_proxy 富集表

按 `p_hard20_proxy` 分位数分桶：

```text
hard_proxy分位桶
样本数
第316平均误差
候选最优上限改善
实际校正率
实际改善
大退化比例
```

分桶：

```text
0-50%
50-70%
70-80%
80-90%
90-100%
```

这张表非常关键。它能判断困难代理分数是否真的找到了第316版容易错、候选库又有潜力的样本。

---

# 9. 第319版验证通过标准

建议固定如下，不要根据验证集临时改。

```python
valid_pass = (
    gain_all      >= 0.002
    and gain_normal >= -0.002
    and gain_strong >= 0.008
    and gain_hard20 >= 0.012
    and gain_hard10 >= 0.000

    and bad_all    <= 0.015
    and bad_normal <= 0.005

    and corr_all    >= 0.020
    and corr_all    <= 0.120
    and corr_normal <= 0.040
    and corr_strong >= 0.050
    and corr_hard20 >= 0.080
)
```

如果你认为 `hard10` 样本数太少，可以改为：

```python
gain_hard10 >= -0.002
```

但不建议取消 `hard20` 要求。第319版就是为困难样本激活设计的。

---

# 10. 第319版如果仍不通过，下一步怎么分流

## 情况 A：第319版还是验证集全不改

说明 Top-K 配额没有真正生效，优先查代码：

```text
1. strong_steer_mask 在验证集是否为空？
2. hard_proxy_threshold 是否用错了训练/验证分位？
3. eligible_hard 是否被 bad_prob / gain_floor 清空？
4. selected_strong 是否真的按比例 top-k 选了？
5. 原预测候选是否错误地参与了 best_candidate 并吞掉了非原始候选？
```

这种情况不是模型问题，先修门控实现。

---

## 情况 B：校正率达标，但强方向盘和困难前20没有改善

说明激活到了样本，但候选选错。

下一版应修第二段候选收益模型：

```text
v320_candidate_ranker_fix
```

方向：

```python
target_best_candidate = argmin(candidate_error)
target_gain = err_v316 - err_candidate
target_bad = candidate_error - err_v316 > 0.05
```

用 pairwise ranking 或 multiclass ranking，而不是只做候选收益回归。

---

## 情况 C：强方向盘改善，但普通样本被改坏

说明普通通道仍然太宽。

下一版直接关闭普通通道：

```python
normal_corr_cap = 0.00
```

实验名：

```text
v320_hard_only_gate
```

即只允许：

```python
strong_steer_mask | hard_proxy_top20
```

中的样本被改。

---

## 情况 D：普通样本安全，但困难前20覆盖率不足

说明 hard_proxy 找不到真正困难样本。

下一版修第一段 / hard_proxy：

```text
v320_hard_proxy_retrain
```

增加特征：

```text
当前窗口方向盘角速度峰值
方向盘角速度持续时间
方向盘角加速度
方向盘转向符号变化
第316版预测峰值/终点/斜率
候选之间的峰值分歧
候选之间的相位分歧
候选之间的终点分歧
```

目标不只训练 hard20，还可以训练：

```python
oracle_improvable = (err_v316 - err_oracle_best) > 0.05
severe_direction_error = sign_error_or_phase_error_label
```

这些只作为训练监督，不作为部署输入。

---

## 情况 E：困难样本覆盖率足够，但大退化超标

说明风险模型低估风险。

下一版修第二段风险模型：

```text
v320_candidate_bad_risk_calibration
```

做法：

```python
bad_label = (err_candidate - err_v316) > 0.05
```

并对强方向盘/困难样本单独校准：

```python
bad_prob_calibrated = calibrate_by_group(
    bad_prob,
    group = strong_or_hard_proxy
)
```

---

# 第319版一句话方案

第319版不要再做“更保守的阈值门控”。应改成：

```text
双通道 + hard_proxy + Top-K 配额 + OOF硬约束收益搜索
```

固定目标是：

```text
普通样本最多改 3% 到 4%，强方向盘至少改 5% 到 8%，困难前20至少覆盖 8% 到 10%；
同时要求强方向盘和困难前20在验证集上有正收益。
```

这样第319版不会再出现第318版的“守卫通过、普通样本安全、但全部不改”的无效成功。
