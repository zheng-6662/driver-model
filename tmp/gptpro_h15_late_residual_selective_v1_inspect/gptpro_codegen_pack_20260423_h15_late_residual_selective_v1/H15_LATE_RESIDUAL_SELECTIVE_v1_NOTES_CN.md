# H15_LATE_RESIDUAL_SELECTIVE_v1 实施说明

## 这次为什么要改

上一版 `H15_LATE_RESIDUAL_HEAD_v1` 已经证明：

- late residual 分支不是 dead branch；
- 它确实能缓解旧 H15 在 `strong_pos` 上的 late-tail collapse；
- 但它对目标 failure bucket 的 **condition selectivity 还不够强**，因此：
  - `best_by_loss` 仍可能收敛到“平均指标好看、危险桶塌陷”的 collapse solution；
  - `best_by_structured` 虽然明显优于旧 H15 的 `strong_pos` collapse，但仍未把 `strong_pos.tail_amp_ratio_pred_over_gt` 推过安全下限。

所以这次不是继续做 optimizer / width / 普通 loss 微调，而是把改动集中在：
**让 late residual 更 selective 地作用到 strong_pos / late-tail under-amplitude case，并把 anti-collapse 约束更早放进训练与 selection。**

## 这次实际新增了什么

### 1) selective late residual gate

在 `code/v58_modular/modeling.py` 里，late residual 现在不再默认对所有样本等强度打开。

新增机制：

- 当 `ENABLE_LATE_RESIDUAL_SELECTIVE_GATE=1` 时，模型会额外计算一个 `strong_pos` gate probability；
- 这个 gate 不只看 encoder pooled context 和 late decoder feature，还叠加了一组 detached late-tail 统计量：
  - `late_abs_mean`
  - `late_peak`
  - `late_amp`
  - `late_slope`
  - `late_mean`
  - `rev_prob`
  - `coarse late abs mean`
  - `fine late abs mean`
- gate 输出经过 `prob_center + floor + boost + late_ramp` 形成 **按样本、按时间** 的 `late_residual_selective_scale`；
- 最终只有 late residual 分支被 selective scaling，旧主干默认行为保持不变。

这样做的目的，是把 correction 更集中地打到 `strong_pos` / under-amplitude 样本，而不是对全部样本平均加幅值。

### 2) stronger late residual focus loss

在 `code/v58_modular/losses.py` 里，`compute_late_residual_head_loss()` 现在除了原来的 late mask / hard late weighting 之外，还会额外考虑：

- `rev_gt_strong`
- detached base tail amplitude deficit
- detached base tail flatness deficit

对应的新权重控制项：

- `LATE_RESIDUAL_STRONG_BOOST`
- `LATE_RESIDUAL_UNDERAMP_BOOST`
- `LATE_RESIDUAL_FOCUS_MAX`

这样做的目的，是阻止 residual head 把容量花在普通样本上，而是优先去修 base path 在 `strong_pos` late-tail 上的 under-amplitude / flatness 问题。

### 3) explicit strong_pos tail guard loss

新增 `compute_strong_pos_tail_guard_loss()`：

- 只对 `rev_gt_strong == 1` 的样本生效；
- 针对 two-sided failure：
  - tail amplitude 低于 floor
  - tail flatness 过高
- 通过 `W_STRONG_POS_TAIL_GUARD` 接入总 loss。

这一步是为了更早阻断 `best_by_loss` 再次收敛到 collapse solution，而不是等 evaluation 才发现。

### 4) structured selection 增加 strong_pos 保护

在 `code/v58_modular/metrics.py` 里，`compute_structured_score()` 现在可以在 `ENABLE_STRONG_POS_STRUCTURED_GUARD=1` 时额外惩罚：

- `strong_pos_tail_amp_ratio_pred_over_gt < STRUCT_STRONG_POS_AMP_FLOOR`
- `strong_pos_tail_flatness_rate > STRUCT_STRONG_POS_FLATNESS_MAX`

这会让 `best_by_structured` 更接近真正想保留的 checkpoint，而不是只看平均 fit/tail。

### 5) 更细粒度的 late residual diagnostics

在 `code/v58_modular/evaluation.py` 里，`test_late_residual_metrics.json` 现在会继续导出并增强：

- `mean_abs_by_bucket`
- `peak_abs_by_bucket`
- `tail_amp_by_bucket`
- `strong_pos_vs_non_strong_ratio`
- `tail_amp_gain_on_strong_pos`
- `correlation_with_tail_under_amp`
- `gate_prob_by_bucket`
- `gate_mean_by_bucket`
- `gate_peak_by_bucket`

`test_state_dump.csv` 里也会追加：

- `late_residual_gate_mean`
- `late_residual_gate_peak`

### 6) 评估导出路径更稳

评估阶段新增了对 `is_curve / rev_gt / idx / gate_prob` 等事件向量的一维展平处理，减少不同 DataLoader 形状下的导出风险。这不改变指标定义，只是让导出链路更稳。

## 这次改动覆盖的文件

限定在 modular 路径内：

- `code/v58_modular/config.py`
- `code/v58_modular/modeling.py`
- `code/v58_modular/losses.py`
- `code/v58_modular/metrics.py`
- `code/v58_modular/evaluation.py`
- `code/v58_modular/train.py`
- `code/recalc_v58_checkpoint_with_current_metrics.py`

## 推荐首轮 full run 配置

见：

- `configs/h15_late_residual_selective_v1_env_block.txt`

推荐策略：

- 继续以 `H15_AC_CF_HLF_v1` 为基础；
- 保持 `FUTURE_SEC=1.5`；
- 保持 coarse-fine / hard late fine；
- 打开 selective late residual gate；
- 打开 strong_pos tail guard；
- 打开 structured strong_pos guard；
- 不同时再叠 optimizer / width sweep。

## 我做过的代码级验证

本次没有替你在真实数据上完整重跑 full train，但已经做了这些最小连通性验证：

- `py_compile` 通过
- selective gate 开启时的随机前向通过
- `compute_total_task_loss()` 新返回路径通过
- `collect_structured_metrics_from_loader()` smoke test 通过
- `evaluate_and_plot()` smoke test 通过，新的 residual diagnostics / state dump 正常落盘
- `recalc instantiate_model()` 对新 config 字段兼容

## 建议你在真实 full run 上优先盯的指标

第一优先级：

- `strong_pos.tail_amp_ratio_pred_over_gt`
- `strong_pos.tail_flatness_rate`

第二优先级：

- `rmse_steer`
- `abs_tail_last_0p5s.rmse_steer`
- `late_peak_recall`

诊断确认项：

- `test_late_residual_metrics.json`
  - `strong_pos_vs_non_strong_ratio`
  - `tail_amp_gain_on_strong_pos`
  - `correlation_with_tail_under_amp`
  - `gate_mean_by_bucket`
  - `gate_peak_by_bucket`

## 这版最可能的风险

### 风险 1：gate 过于保守

如果 `late_residual_gate_prob_center` 太高、或者 `gate_prob` 训练不起来，可能出现：

- residual 大部分时间只停在 floor 附近；
- `strong_pos` 修复幅度不够。

### 风险 2：tail guard 太强，换来整体 fit/tail 回退

如果 `W_STRONG_POS_TAIL_GUARD` 太大，可能会出现：

- `strong_pos` 指标上来；
- 但 `rmse_steer` / `tail_rmse_steer` 明显回退。

### 风险 3：gate 学成“强样本粗标签”，但与 under-amplitude 对齐不够

这时需要重点看：

- `gate_prob_by_bucket`
- `correlation_with_tail_under_amp`

如果 `strong_pos` 桶内 gate 明显更高，但和 `tail_under_amp` 相关性仍弱，说明还需要进一步让 gate 与 failure mechanism 对齐，而不只是与 bucket label 对齐。
