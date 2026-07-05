请审阅 v222a closeout 结果并给下一轮 bounded 指令。

本地已按你上一轮要求完成 closeout-only candidate gap audit：

- pack: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\v222a_closeout_candidate_gap_audit_pack.zip`
- report: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\reports\v222a_closeout_candidate_gap_audit_cn.md`
- script: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_closeout_candidate_gap_audit_20260622.py`

结论：

- formal headline locked:
  - `loose_main_pool=avg_joint_focus`
  - `strict_main_pool=peak_floor_090`
- `v222a_bounded_residual / v222a_noharm_gate / oracle_safe_gate` 均为 diagnostic-only。
- stop evidence:
  - loose test: validation pass=True, locked test pass=False, RMSE delta `+0.010559`, tail delta `+0.027764`, under reduction `+0.043478`。
  - strict test: validation pass=True, locked test pass=False, RMSE delta `-0.008975`, tail delta `-0.005264`, under reduction `-0.017241`。
- closeout taxonomy:
  - locked test selector_failed_rate: loose `0.407609`, strict `0.413793`, combined `0.410615`。
  - locked test candidate_missing_rate: loose `0.027174`, strict `0.028736`, combined `0.027933`。
  - high-tail candidate_missing_rate: loose `0.119048`, strict `0.135135`, combined `0.126582`。
  - high-tail oracle clear gain rate combined `0.911392`。
- future_route_decision:
  - `v222b_allowed=False`
  - `v223_allowed=False`

验证：

- py_compile pass
- full script run pass
- ZIP `bad_file=None`, 74 files, required files missing `[]`
- leakage guard 6/6 pass
- forbidden scan over closeout tables/reports/logs for `W3_B4_original_soft|oracle_model|true_label|fallback`: no matches
- case figures 61 PNGs, visual spot check pass

请回答：

1. 是否接受 closeout 诊断：当前主要失败是 selector/gate 泛化，而不是 candidate pool 大面积缺曲线？
2. 既然当前 `v222b_allowed=False`、`v223_allowed=False`，下一步唯一值得做的 bounded local step 是什么？
3. 请指定下一步的 exact required files、pass/fail criteria、stop conditions。
4. 明确列出 Codex 下一步必须避免做什么。

限制：不要要求 test retuning、新 tau、`v222a_gate_v2`、multi-router、v222b/v223 训练、或把 oracle/true-label/diagnostic-only row 写入 formal leaderboard/gate/usage/selected configs。
