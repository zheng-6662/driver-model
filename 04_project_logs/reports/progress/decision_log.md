# 项目决策日志

这里只记录真正改变方向、标准或交付边界的判断。

## 决策表

| 日期 | 决策 | 白话解释 | 触发原因 | 影响范围 | 证据 / 入口 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-04-23 | V5.8 active code line switches to modular editing under `02_code/final_code/model/training/v58_modular/`; the old script remains a thin wrapper entrypoint | 以后继续运行老脚本可以，但新增逻辑不再往那个大文件里堆，而是优先写进对应模块。 | active training code had become a multi-thousand-line monolith with duplicated / unreachable sections, and the user explicitly required a literature-style multi-module structure for future work. | all future code edits on the V5.8 training line; wrapper maintenance, training snapshots, and future refactors should all preserve the modular package as the source of truth | [2026-04-23 日志](daily/2026-04-23.md), [current-state](../../references/current-state.md), [v58_modular README](../../../02_code/final_code/model/training/v58_modular/README.md) | active |
| 2026-04-21 | 本轮主线切换到 `D:/下载/codex_next_steps_plan.md`，旧 `current-state` 降为背景锚点 | 本轮不再把“继续追 220918 复现差距”当作唯一下一步，而是先查清输入管线、任务定义、checkpoint 选择和 spike 位置这些更直接的问题。 | 外部计划指出 lane 与 speed 风险直击当前 active script；本地代码与数据抽样已确认这两个风险真实存在。 | 2026-04-21 全部执行顺序、日志落盘规则、第一批分析任务与后续 D/E 路线 | [2026-04-21 日志](daily/2026-04-21.md), [当前状态 handoff](../../references/current-state.md) | active |
| 2026-04-21 | Batch 1 + Batch 2 已完成，下一优先级正式切到 D 输入分组消融，再切到 E bridge 训练 | A/B/F/G 诊断和 active script 输入修复已经闭环，后续不应再停留在“先补工具”阶段，而要进入按计划锁死的 D/E 全矩阵执行。 | 诊断报告与 smoke 验证都已落盘：lane / speed bug 已修，`input_qc` 写出成功，same-tool recalc 兼容新接口，Pareto 与 spike 诊断也已完成。 | 从本条开始，训练侧默认先跑 D，再跑 E；每个 full run 完成后必须 same-tool recalc 并写 daily / registry。 | [2026-04-21 日志](daily/2026-04-21.md), [当前状态 handoff](../../references/current-state.md), [项目中枢](../project_progress_hub.md) | active |
| 2026-04-20 | 稳定 manual-upsample control 继续作为固定基线 | active-script 全跑版本仍没有替换掉这个旧稳定控制线，因此任何新结论都必须先和它比。 | 2026-04-20 的 old-baseline equivalence 检查没有追回旧稳定控制。 | 全部 D/E 训练与后续答辩表述 | [2026-04-20 日志](daily/2026-04-20.md) | active |
| 2026-04-20 | Run A / Run B 保持双 keeper 解读，Run C / Run D 不提升为默认主线 | Run A 更强在 response-structure，Run B 更强在 overall / tail fit；Run C / D 目前只能作为辅助证据。 | 2026-04-20 全量 run + same-tool recalc 的比较结果。 | keeper 选择、答辩叙事、D/E 对照锚点 | [2026-04-20 日志](daily/2026-04-20.md) | active |

## 记录模板

### YYYY-MM-DD 决策标题

- 决策：
- 白话解释：
- 触发原因：
- 影响范围：
- 证据链接：
- 后续动作：
- 状态：`active` / `superseded` / `obsolete`
## 2026-04-21 Final D Closure

- Decision:
  - D input-ablation is closed.
  - `baseline_fixed_input` replaces old Run B as the maintained-line fit / tail keeper.
  - Run A remains the response-structure keeper.
  - `plus_pedals` stays as a late-peak trade-off branch, not a clean replacement.
  - `plus_lat_dyn`, `plus_road_cond`, and `minus_z` are not promoted.
- Plain-language meaning:
  - The biggest D gain came from repairing lane/speed reading, not from adding more optional input families.
  - Turning off `z` is not justified; `USE_Z=1` remains the default.
- Trigger:
  - All five D groups completed with same-tool recalc.
  - `baseline_fixed_input` gives `rmse_steer=0.5559`, `tail_rmse_steer=0.7171`, `late_peak_recall=0.6496`.
  - `minus_z` gives `rmse_steer=0.5760`, `tail_rmse_steer=0.7385`, so it does not beat the repaired baseline on fit/tail.
- Impact scope:
  - The next priority moves to E bridge training.
  - E should bridge Run A response-structure strength against `baseline_fixed_input` fit/tail strength, not the older pre-fix Run B line.
- Evidence:
  - [2026-04-21 daily](daily/2026-04-21.md)
  - [D comparison table](../input_group_ablation_20260421/input_ablation_comparison_table.csv)
  - [D summary](../input_group_ablation_20260421/input_ablation_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-22 E Interim Update

- Decision:
  - `bridge_50_50` is now the provisional E fit / tail bridge candidate.
  - Run A remains the response-structure keeper.
  - `baseline_fixed_input` stays as the post-D fit/tail anchor and comparator until E closes.
- Plain-language meaning:
  - A stronger static bridge can beat the repaired D baseline on fit/tail, but it still does not recover the Run A structure axis.
- Trigger:
  - `bridge_55_45` completed first and did not change the keeper split.
  - `bridge_50_50` `best_by_structured` gives `rmse_steer=0.5385`, `tail_rmse_steer=0.6846`, `late_peak_recall=0.6197`.
- Impact scope:
  - The last E run, `bridge_schedule_B_to_A`, should now be judged against Run A on structure and `bridge_50_50` on fit/tail.
  - No thesis claim is promoted yet; E is still in progress.
- Evidence:
  - [2026-04-21 daily](daily/2026-04-21.md)
  - [bridge_50_50 structured recalc](../bridge_training_20260421/bridge_50_50/TRAIN_V5_4_STATECOND_REV_20260422_004147/figures/recalc_best_by_structured_summary.json)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-22 E Final Closure

- Decision:
  - E bridge matrix is closed.
  - Run A remains the response-structure keeper.
  - `baseline_fixed_input` remains the fit / tail keeper after the full E guardrail review.
  - `bridge_50_50` is not promoted even though it posts the best fit/tail headline numbers.
  - `bridge_schedule_B_to_A` is mixed and not promoted.
  - `bridge_55_45` is a no-go.
- Plain-language meaning:
  - Weighting / supervision bridge choices do move the trade-off frontier.
  - They still do not collapse fit/tail and response-structure into one promoted checkpoint that survives the full guardrail set.
- Trigger:
  - All three E groups completed with same-tool recalc.
  - `bridge_50_50` `best_by_structured` gives `rmse_steer=0.5385` and `tail_rmse_steer=0.6846`, but `strong_pos.tail_amp_ratio_pred_over_gt=0.4987` and `strong_pos.tail_flatness_rate=0.7368`.
  - `bridge_schedule_B_to_A` `best_by_structured` lifts `late_peak_recall` to `0.6581` and `first_reversal_time_mae_sec` to `0.4923`, but `rmse_steer=0.5819` and `tail_rmse_steer=0.7749` regress too much.
- Impact scope:
  - The live keeper split remains Run A versus `baseline_fixed_input`.
  - `bridge_50_50` becomes fit/tail frontier evidence rather than a keeper; `bridge_schedule_B_to_A` becomes mixed bridge evidence rather than a keeper.
- Evidence:
  - [2026-04-21 daily](daily/2026-04-21.md)
  - [bridge_50_50 structured recalc](../bridge_training_20260421/bridge_50_50/TRAIN_V5_4_STATECOND_REV_20260422_004147/figures/recalc_best_by_structured_summary.json)
  - [bridge_schedule_B_to_A structured recalc](../bridge_training_20260421/bridge_schedule_B_to_A/TRAIN_V5_4_STATECOND_REV_20260422_010228/figures/recalc_best_by_structured_summary.json)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-22 Effectiveness Mainline Switch

- Decision:
  - The active round switches from bridge / gate / loss exploration to the effectiveness plan.
  - The execution order is locked as D0 absolute-window diagnosis, `H15`, `OPT_A_20`, then conditional `H10`, `OPT_C_BEST`, `CAP_192_BEST`, and winner confirmation as budget allows.
  - `H10` is conditional only, not a default mainline run.
  - `OPT_C_BEST` is conditional regularization, not a fixed mandatory `wd=5e-4` slot unless overfit is observed.
- Plain-language meaning:
  - This round asks whether the current ceiling is mainly horizon length, optimization, or mild capacity.
  - It does not reopen bridge / gate / loss as the main frontier.
- Trigger:
  - The external 2026-04-22 revised plan explicitly reprioritized absolute time windows, `1.5 s` viability, and `2.0 s` optimization rescue over more bridge experiments.
- Impact scope:
  - New training must preserve `fixed_v20260421`, the stable GPU path, protocol-safe split, and the existing live keeper split.
- Evidence:
  - [2026-04-22 daily](daily/2026-04-22.md)
  - [effectiveness summary](../effectiveness_followup_20260422/effectiveness_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-22 Effectiveness Round Closure

- Decision:
  - `baseline_fixed_input` remains the current best non-collapse `2.0 s` fit / tail base.
  - Run A remains the response-structure anchor.
  - `H15` is not eligible for `OPT_A_H15 / OPT_B_H15` because it hard-collapses on `strong_pos`.
  - `OPT_A_20` does not rescue the original `2.0 s` task.
  - `H10` remains diagnostic ceiling evidence only.
  - `OPT_C_BEST`, `CAP_192_BEST`, and `WINNER_CONFIRM` do not replace the original baseline anchor.
- Plain-language meaning:
  - Shortening the horizon can produce large overall / tail gains, but the current model loses the strong positive tail behavior.
  - Simple optimizer, regularization, and mild-width changes do not raise the safe `2.0 s` ceiling enough to replace the current baseline.
- Trigger:
  - `H15` `best_by_structured` gives `rmse_steer=0.4930` and `abs_tail_last_0p5s.rmse_steer=0.6022`, but `strong_pos.tail_amp_ratio_pred_over_gt=0.2687` and `strong_pos.tail_flatness_rate=1.0000`.
  - `OPT_A_20` `best_by_structured` gives `rmse_steer=0.5887`, `abs_tail_last_0p5s.rmse_steer=0.7698`, and `late_peak_recall=0.5726`, all worse than the baseline anchor except the first-second prefix.
  - `OPT_C_BEST` and `CAP_192_BEST` are non-collapse but do not beat the baseline anchor on the selection order.
- Impact scope:
  - Future work should keep the live keeper split and only open a new branch if it directly targets `H15` strong-pos collapse or a justified architecture change.
- Evidence:
  - [2026-04-22 daily](daily/2026-04-22.md)
  - [effectiveness comparison table](../effectiveness_followup_20260422/effectiveness_comparison_table.csv)
  - [effectiveness summary](../effectiveness_followup_20260422/effectiveness_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-23 GPT Pro Follow-up Recommendation

- Decision:
  - If a new effectiveness follow-up is approved, prioritize `H15_AC_CF_HLF_v1`.
  - Define it as the `1.5 s` line plus coarse-fine steer decomposition and hard-late fine residual supervision.
  - Align the anti-collapse supervision window to the actual failing slice:
    - `HARD_LATE_START_SEC = 1.00`
    - `HARD_TAIL_START_SEC = 1.00`
  - Do not spend the next budget on plain optimizer or width sweeps.
  - If this anti-collapse attempt fails, escalate directly to `H15_LATE_RESIDUAL_HEAD_v1` instead of continuing loss micro-sweeps.
- Plain-language meaning:
  - `H15` is not bad because it lacks signal.
  - It is bad because it gets better average numbers by flattening the dangerous strong-pos late tail.
  - So the next step should protect late-tail shape directly, not polish optimizer settings.
- Trigger:
  - GPT Pro review judged that `H15` gains are real under the D0 absolute-window metric:
    - `rmse_steer: 0.5559 -> 0.4930`
    - `abs_tail_last_0p5s.rmse_steer: 0.7171 -> 0.6022`
  - The same review also judged that the gains are "bought" by collapse on:
    - `strong_pos.tail_amp_ratio_pred_over_gt = 0.2687`
    - `strong_pos.tail_flatness_rate = 1.0000`
  - `OPT_A_20` and `CAP_192_BEST` are taken as enough evidence that optimizer / width sweeps are no longer the highest-EV next move.
- Impact scope:
  - If approved, the next implementation should land in the modular V5.8 package:
    - `v58_modular/config.py`
    - `v58_modular/modeling.py`
    - `v58_modular/losses.py`
    - `v58_modular/train.py`
    - `v58_modular/evaluation.py`
  - The old monolithic wrapper should remain a launcher only.
- Evidence:
  - [2026-04-23 daily](daily/2026-04-23.md)
  - [GPT Pro review](../gptpro_effectiveness_review_20260423.md)
  - [effectiveness summary](../effectiveness_followup_20260422/effectiveness_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-23 `H15_AC_CF_HLF_v1` Closure

- Decision:
  - `H15_AC_CF_HLF_v1` is closed as a no-go.
  - `H15` is still not promotable.
  - Do not spend more budget on optimizer, width, or generic loss micro-sweeps on this branch.
  - If another run is approved, go directly to `H15_LATE_RESIDUAL_HEAD_v1`.
- Plain-language meaning:
  - The anti-collapse bundle helped, but not enough.
  - It reduced the old `H15` strong-pos flatness failure without restoring enough late-tail amplitude or late-peak recall.
  - The bottleneck now looks architectural in the late slice, not like a tuning-only problem.
- Trigger:
  - `best_by_loss` reaches:
    - `rmse_steer=0.5063`
    - `abs_tail_last_0p5s.rmse_steer=0.6323`
    - `late_peak_recall=0.5786`
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.5141`
    - `strong_pos.tail_flatness_rate=0.3750`
  - `best_by_structured` fails the explicit strong-pos floor:
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.3304 < 0.50`
  - manual review of eight representative `strong_pos` plots still finds `3/8` severe final-tail under-amplitude cases
- Impact scope:
  - Run A remains the response-structure anchor.
  - `baseline_fixed_input` remains the fit / tail keeper.
  - The next approved branch, if any, should be a minimal late residual head for `t >= 1.0 s`.
- Evidence:
  - [2026-04-23 daily](daily/2026-04-23.md)
  - [H15_AC_CF_HLF_v1 summary](../effectiveness_followup_20260423/h15_ac_cf_hlf_v1_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-23 `H15_LATE_RESIDUAL_HEAD_v1` Closure

- Decision:
  - `H15_LATE_RESIDUAL_HEAD_v1` is not promoted as a new keeper.
  - Run A remains the response-structure anchor.
  - `baseline_fixed_input` remains the fit / tail keeper.
  - If the late-residual direction is continued later, only `best_by_structured` from this run should be treated as the control checkpoint.
  - Do not reopen optimizer or width sweeps as the next explanation for this branch.
- Plain-language meaning:
  - Adding extra late-slice capacity helps more than the previous anti-collapse bundle alone, but not enough to make the branch safe to promote.
  - One checkpoint keeps the attractive average metrics and still collapses the dangerous bucket.
  - The other checkpoint repairs the dangerous bucket partially, but gives back too much fit / tail and still misses the amplitude floor.
- Trigger:
  - `best_by_loss` reaches:
    - `rmse_steer=0.4954`
    - `abs_tail_last_0p5s.rmse_steer=0.6284`
    - `late_peak_recall=0.6522`
    - but `strong_pos.tail_amp_ratio_pred_over_gt=0.3163`
    - and `strong_pos.tail_flatness_rate=1.0000`
  - `best_by_structured` improves the old `H15` collapse materially:
    - `strong_pos.tail_amp_ratio_pred_over_gt: 0.2687 -> 0.4904`
    - `strong_pos.tail_flatness_rate: 1.0000 -> 0.5000`
    - `late_peak_recall: 0.6355 -> 0.6656`
  - but `best_by_structured` still misses the amplitude target:
    - `strong_pos.tail_amp_ratio_pred_over_gt = 0.4904 < 0.60`
  - built-in late-residual diagnostics show the new head is active, but only mildly more active on `strong_pos` than on non-strong cases.
- Impact scope:
  - The branch is informative as a mechanism probe, not as a keeper replacement.
  - Future work on this line, if approved, should focus on making the late residual path more selective to the target failure bucket rather than reopening broad hyperparameter searches.
  - Same-tool recalc on the modular path still relies on the temporary shim:
    - `tmp/recalc_v58_metrics_shim_20260423.py`
- Evidence:
  - [2026-04-23 daily](daily/2026-04-23.md)
  - [H15_LATE_RESIDUAL_HEAD_v1 summary](../effectiveness_followup_20260423/h15_late_residual_head_v1_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-23 `H15_LATE_RESIDUAL_SELECTIVE_v1` Closure

- Decision:
  - `H15_LATE_RESIDUAL_SELECTIVE_v1` is not promoted as a new keeper.
  - Run A remains the response-structure anchor.
  - `baseline_fixed_input` remains the fit / tail keeper.
  - Do not reopen optimizer, width, bridge, or generic loss sweeps as the next explanation for this branch.
  - If this line is continued later, stay on the selective late-residual path and aim for stronger under-amplitude alignment with fit/head protection.
- Plain-language meaning:
  - The new selective gate is real.
  - It makes the late residual path much more `strong_pos`-aware.
  - But the branch still splits into two bad endpoints:
    - one checkpoint keeps reasonable fit and still under-repairs the dangerous bucket
    - the other checkpoint repairs the dangerous bucket, but damages the main task too much
  - So the next bottleneck is no longer "does late residual work?".
  - The next bottleneck is whether the correction can be aligned tightly enough to the true failure mechanism without hurting fit / tail / prefix / onset.
- Trigger:
  - `best_by_loss` reaches:
    - `rmse_steer=0.5356`
    - `abs_tail_last_0p5s.rmse_steer=0.6745`
    - `late_peak_recall=0.6756`
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.4947`
    - `strong_pos.tail_flatness_rate=0.7500`
  - `best_by_structured` proves the branch can push strong-pos repair much harder:
    - `strong_pos.tail_amp_ratio_pred_over_gt=1.4833`
    - `strong_pos.tail_flatness_rate=0.3750`
  - but `best_by_structured` does so in a globally damaged regime:
    - `rmse_steer=0.6379`
    - `abs_tail_last_0p5s.rmse_steer=0.7319`
    - `response_onset_delay_mae_sec=0.6270`
  - built-in diagnostics show bucket-level selectivity is now real:
    - `strong_pos_vs_non_strong_ratio.gate_prob=4.6584`
    - `strong_pos_vs_non_strong_ratio.gate_mean=3.3443`
  - but correlation with actual tail under-amplitude remains weak / slightly negative.
- Impact scope:
  - The live keeper split remains unchanged.
  - Future work on this line, if approved, should stay inside `v58_modular/` and the recalc tool.
  - The current run should be treated as a bracketed failure boundary:
    - `best_by_loss` = fit-preserving but under-repaired
    - `best_by_structured` = strong-pos-repaired but over-regressed
- Evidence:
  - [2026-04-23 daily](daily/2026-04-23.md)
  - [H15_LATE_RESIDUAL_SELECTIVE_v1 summary](../effectiveness_followup_20260423/h15_late_residual_selective_v1_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`
