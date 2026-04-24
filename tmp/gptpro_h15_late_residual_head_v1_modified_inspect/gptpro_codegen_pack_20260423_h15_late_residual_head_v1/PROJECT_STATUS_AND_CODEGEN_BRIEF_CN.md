# 当前状态与代码生成任务简报

## 1. 一句话结论

到 `2026-04-23` 为止，`H15_AC_CF_HLF_v1` 已经完整 full run + same-tool recalc + 人工 strong-pos 复核闭环，但仍然失败。
它证明了 `H15` 的 anti-collapse 方向有真实信号，却没有把 `H15` 修到可晋级。
因此，下一步如果要让 GPT Pro 直接产代码，目标不该再是 optimizer / width / gate / loss 小修小补，而应直接切到：

- `H15_LATE_RESIDUAL_HEAD_v1`

## 2. 当前已锁定的项目状态

- 当前 fit / tail keeper:
  - `baseline_fixed_input` full `best_by_structured`
- 当前 response-structure anchor:
  - Run A full `best_by_structured`
- 当前 active 输入版本:
  - `fixed_v20260421`
- 当前协议口径:
  - `protocol-safe`
  - `subject-level fixed split`
  - stable GPU path locked
  - pooled training only
  - no scene-specific models
- 当前 active 实现位置:
  - wrapper entrypoint:
    - `02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
  - real implementation:
    - `02_code/final_code/model/training/v58_modular/`

## 3. 为什么 `H15` 没有被放弃，但也不能直接晋级

旧 `H15` 的收益是真收益，不是 fraction tail 假象。

- `rmse_steer: 0.5559 -> 0.4930`
- `abs_tail_last_0p5s.rmse_steer: 0.7171 -> 0.6022`

但它的问题也很明确：

- `strong_pos.tail_amp_ratio_pred_over_gt = 0.2687`
- `strong_pos.tail_flatness_rate = 1.0000`

也就是说：

- `1.5s` 线是有真实潜力的
- 问题不是 “没信号”
- 问题是模型为了拿更好看的整体 / tail 指标，把最危险的 `strong_pos` late tail 压塌了

## 4. 为什么 `H15_AC_CF_HLF_v1` 仍然算失败

本次 full run:

- run:
  - `H15_AC_CF_HLF_v1`
- path:
  - `03_results/tmp/effectiveness_followup_20260423/h15_ac_cf_hlf_v1/TRAIN_V5_4_STATECOND_REV_20260423_131956`

本次 anti-collapse bundle:

- 保持 `FUTURE_SEC=1.5`
- 开 `STEER_COARSE_FINE`
- 开 `HARD_LATE_FINE`
- 把 hard-late / hard-tail 窗口对齐到真实失败段：
  - `1.0s -> 1.5s`
- 不开：
  - `PHASE_ADAPTIVE_TREND`
  - `STRONG_POS_GATE`
  - `W_FIRSTREV_LOCAL`

结果：

- `best_by_loss`
  - `rmse_steer=0.5063`
  - `abs_tail_last_0p5s.rmse_steer=0.6323`
  - `late_peak_recall=0.5786`
  - `strong_pos.tail_amp_ratio_pred_over_gt=0.5141`
  - `strong_pos.tail_flatness_rate=0.3750`
- `best_by_structured`
  - `rmse_steer=0.5231`
  - `abs_tail_last_0p5s.rmse_steer=0.6472`
  - `late_peak_recall=0.5251`
  - `strong_pos.tail_amp_ratio_pred_over_gt=0.3304`
  - `strong_pos.tail_flatness_rate=0.7500`

这说明：

- anti-collapse 方向是有效的
- old `H15` 的极端 flatness 确实被部分修住了
- 但 repair 不够稳定，也不够强
- `best_by_structured` 甚至还在明显失败区

## 5. 人工复核为什么把它从 borderline 打回 no-go

因为 `best_by_loss` 看起来像“差一点点”，所以又人工复核了 `8` 个 `strong_pos` representative plots。

复核结果：

- `3/8` 仍然是明显 severe under-amplitude
  - `0.161`
  - `0.319`
  - `0.354`
- `2/8` 只是勉强中等修复
  - `0.602`
  - `0.602`
- `2/8` 接近但仍然 capped / biased
  - `0.618`
  - `0.618`
- `1/8` 才接近真正修住
  - `0.837`

所以结论不是“几乎成功”，而是：

- 这条 anti-collapse 路线有信号
- 但还不够把 `H15` 变成 promotable branch
- 不值得继续在 optimizer / width / generic loss 上烧预算

## 6. 对下一步代码生成的直接含义

这一步最像的问题已经不是：

- optimizer 不够好
- mild width 不够大
- 还差几个 generic loss scalar

而更像是：

- 当前 decoder 在 `t >= 1.0s` 的 late slice 表达不够
- 单靠 coarse-fine + hard-late objective 还不足以稳定地保住 `strong_pos` late tail
- 需要一个最小结构化 late residual slice，直接服务于最后 `0.5s`

因此最合适的 codegen 目标就是：

- `H15_LATE_RESIDUAL_HEAD_v1`

## 7. 希望 GPT Pro 直接生成什么

不是只做高层分析，也不是只提 10 个想法。
而是希望 GPT Pro 基于包内当前源码，给出可落地的最小代码实现方案，最好能直接产出 patch 级修改。

目标版本应满足：

- 仍然基于当前 `v58_modular` 代码树
- wrapper 继续保持薄
- 不改 split
- 不改 `fixed_v20260421`
- 不改 stable GPU path
- 不做 scene-specific 模型
- 不把 bridge / gate / loss 再拉回主线
- 优先只改：
  - `v58_modular/config.py`
  - `v58_modular/modeling.py`
  - `v58_modular/losses.py`
  - `v58_modular/train.py`
  - `v58_modular/evaluation.py`

## 8. 希望 GPT Pro 产出的“最小可验证版本”应长什么样

推荐它围绕以下问题直接给代码级回答：

1. 如何在当前 coarse-fine 主干上，加一个只负责 `t >= 1.0s` 的 late residual head。
2. 这个 head 的输出如何和现有主输出相加或路由，而不破坏前段 `0 ~ 1.0s`。
3. 训练时如何让这个 head 主要为 `strong_pos` late-tail failure 服务，而不是把整体 objective 又搞回大而全的 loss matrix。
4. 如何把新增路径的开关、权重、start sec 清楚接到 config / train / eval / recalc 兼容路径里。
5. 第一个 full run 的 env 应该怎么最小配置。

## 9. 红线

- 不要建议重新扫 optimizer
- 不要建议重新扫 width
- 不要建议重开 bridge / gate / loss matrix
- 不要建议切回 random split
- 不要建议只看 overall RMSE
- 不要建议把 `H10` 直接升为默认主线
- 不要输出一个需要大规模重写训练系统的方案

## 10. 这份包里最关键的证据

- 当前项目锚点：
  - `context/current-state.md`
  - `context/daily_2026-04-23.md`
  - `context/decision_log.md`
  - `context/experiment_registry.md`
- 最新复盘与结论：
  - `evidence/gptpro_effectiveness_review_20260423.md`
  - `evidence/h15_ac_cf_hlf_v1_summary.md`
- 三个最关键对照 summary：
  - `evidence/baseline_fixed_input_recalc_best_by_structured_summary.json`
  - `evidence/h15_recalc_best_by_structured_summary.json`
  - `evidence/h15_ac_cf_hlf_v1_recalc_best_by_loss_summary.json`
  - `evidence/h15_ac_cf_hlf_v1_recalc_best_by_structured_summary.json`
- 人工 strong-pos 复核：
  - `evidence/strong_pos_review/strong_pos_review_index.csv`
  - `evidence/strong_pos_review/*.png`

## 11. 这次 codegen 的真实目标

不是让 GPT Pro 再告诉我们 “也许可以试试更多组合”。
而是让它直接给出：

- 当前最值得做的一刀
- 最小可落地的代码修改位置
- 最小可验证的实现版本
- 第一个 full run 的明确配置入口
