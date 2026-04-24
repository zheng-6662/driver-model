# Gate Source vs Activation Diagnosis

## 结论先行

`2026-04-17 100716` 最有价值的新证据不是“又一个 no-go”，而是它把当前问题正式拆成了两层：

1. `gate source` 已明显变准
2. 当前 `activation` 仍然错误

所以后续如果还继续 gate 路线，优先级最高的不是再换 source，而是改 activation / routing。

## 直接证据

基于以下两个 `test_state_dump.csv` 直接计算：

- `evidence/20260416_mainline220918_test_state_dump.csv`
- `evidence/20260417_strongposgate100716_test_state_dump.csv`

在相同测试集上：

- 样本数：`528`
- `rev_gt_strong=1` 的样本数：`19`

### 旧 source：`220918` 的 `rev_prob`

- score：`rev_prob`
- label：`rev_gt_strong`
- AUC：`0.4725`
- 正样本均值：`0.2256`
- 负样本均值：`0.2186`

这说明旧 `rev_prob` 对 strong-pos / hard reversal 的区分几乎不可用。

### 新 source：`100716` 的 `strong_pos_gate_prob`

- score：`strong_pos_gate_prob`
- label：`rev_gt_strong`
- AUC：`0.7044`
- 正样本均值：`0.4101`
- 负样本均值：`0.2654`

这说明 dedicated strong-pos gate source 已经明显优于旧 raw `rev_logit` 路线。

## 但为什么 `100716` 仍然是 no-go

尽管 source separability 提升明显，`100716` 相比 `220918` 仍然全面失守：

- `rmse_steer: 0.5697 -> 0.6491`
- `mae_steer: 0.4001 -> 0.4431`
- `head_amp_ratio_pred_over_gt: 1.5739 -> 2.2357`
- `tail_rmse_steer: 0.6826 -> 0.8068`
- `late_peak_recall: 0.5940 -> 0.5128`
- `first_reversal_time_mae_sec: 0.6206 -> 0.8016`
- `strong_pos.tail_amp_ratio_pred_over_gt: 0.3732 -> 0.3601`
- `strong_pos.tail_flatness_rate: 0.7368 -> 0.8947`

也就是说：

- 新 source 成功识别了“哪些样本更像 strong-pos”
- 但当前 centered multiplicative late fine gate 仍然把能量错误地注入了整盘
- 结果是大盘被破坏，目标 bucket 也没有真正救回来

## 与 `011755` 的关系

`011755` 已经证明：

- raw `rev_logit` gate source 不够定向
- 即便给 late fine 加 gate，也没法稳定把能量送到 `strong_pos` bucket

`100716` 再进一步证明：

- 即使 source 做准了
- 当前这类前向乘法式 full late fine gain 仍然不对

所以当前更强的判断是：

- 问题已不主要在 `source`
- 问题更像是 `activation / routing / residual target` 的设计不对

## 对 GPT Pro 最重要的提问

如果继续沿 gate 路线推进，什么比当前方案更值得优先尝试？

候选方向应优先考虑：

- tail-only gate
- additive late residual，而不是 multiplicative gain
- training-only gate，用于 loss / target / sample routing，而不是推理时直接放大
- gate 只作用于 fine residual 的一部分子空间，避免整盘受扰
- 更结构化的 phase-specific late residual 建模，直接替代当前 gate activation
