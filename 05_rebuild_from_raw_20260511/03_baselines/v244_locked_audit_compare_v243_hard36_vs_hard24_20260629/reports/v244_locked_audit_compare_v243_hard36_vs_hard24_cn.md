# v244 locked audit：v243 hard36 vs hard24 对比报告

## 本轮做了什么

- 只读取 v243 已落盘产物，不训练模型，不调权重，不改 validation 规则。
- 对比 validation-selected `v243_metric_hard36_guard08` 和 conservative/test-robust `v243_metric_hard24_guard04`。
- 同时保留 `v243_metric_hard30_guard06_anchor04` 作为参考，因为它在 all/normal 上也很强。
- 本轮是 locked audit/reporting，不是新模型实验。

## 关键限制

- v243 的 npz 只保存了 best guarded 预测，也就是 hard36。
- hard24 没有完整曲线预测、checkpoint 和逐样本 delta，因此 hard24 只能做 aggregate 对比，不能做同级别 per-sample casebook。
- 这个限制不影响 aggregate test 结论，但会阻止把 hard24 直接升级为 formal replacement。

## 候选级结论

- validation-selected：`v243_metric_hard36_guard08`，validation score=0.865386，best_epoch=34。
- hard36 test：all=-0.002128，normal=-0.006139，observe=+0.009219，strong=+0.003896。
- hard24 test：all=-0.003832，normal=-0.003955，observe=-0.006484，strong=-0.003601。
- hard30 test 参考：all=-0.004607，normal=-0.006566，observe=-0.001721，strong=-0.001645。

## Bucket 判断

- all: hard24 mean tail delta=-0.003832，hard36 mean tail delta=-0.002128，preferred=v243_metric_hard24_guard04；hard24 在该 bucket 更稳，尤其要关注 hard36 的迁移风险。
- normal_predictable: hard24 mean tail delta=-0.003955，hard36 mean tail delta=-0.006139，preferred=v243_metric_hard36_guard08；hard36 更适合 normal，但不是 hard bucket 最稳。
- observe_later_like: hard24 mean tail delta=-0.006484，hard36 mean tail delta=+0.009219，preferred=v243_metric_hard24_guard04；hard24 在该 bucket 更稳，尤其要关注 hard36 的迁移风险。
- strong_steer: hard24 mean tail delta=-0.003601，hard36 mean tail delta=+0.003896，preferred=v243_metric_hard24_guard04；hard24 在该 bucket 更稳，尤其要关注 hard36 的迁移风险。

## hard36 逐样本风险

- all: n=1104，tail regression rate=0.451，mean delta=-0.002128，max delta=+0.185908。
- normal_predictable: n=594，tail regression rate=0.429，mean delta=-0.006139，max delta=+0.085701。
- observe_later_like: n=162，tail regression rate=0.475，mean delta=+0.009219，max delta=+0.185908。
- strong_steer: n=480，tail regression rate=0.492，mean delta=+0.003896，max delta=+0.185908。
- zero_cross_or_reverse_or_multi: n=882，tail regression rate=0.435，mean delta=-0.001964，max delta=+0.185908。

## 决策

- `validation_selected_candidate`: `v243_metric_hard36_guard08`。hard36 是 v243 validation 规则下排名第一的 accepted candidate。
- `locked_test_more_stable_candidate`: `v243_metric_hard24_guard04`。hard24 在 observe/strong hard bucket 的变差 delay 数为 2/12，低于 hard36 的 11/12。
- `promote_hard36_as_formal_replacement_now`: `False`。hard36 虽通过 validation，但 locked test 上 observe_later_like 6/6 个 delay tail 变差、strong_steer 5/6 个 delay tail 变差。
- `promote_hard24_as_formal_replacement_now`: `False`。hard24 aggregate test 更稳，但 hard24 没有保存完整预测/checkpoint/逐样本表，且不能用 test 反向改 validation 选择。
- `keep_v241_as_default_until_granular_audit`: `True`。v243 的 hard24/hard36 结论还存在 validation-vs-test 选择冲突；正式替代前必须补齐 hard24 granular audit。
- `recommended_next_step`: `replay_v243_save_all_candidates_then_locked_audit_or_keep_aggregate_v244_as_limit`。若要继续推进 v243，应只重放保存 hard24/hard36 全候选预测和 checkpoint，不改超参，不用 test 调参。
- `hard24_granular_artifact_complete`: `False`。当前 hard24 缺少完整曲线预测、checkpoint 和逐样本 delta。

## Guardrail

- `stage`: `v244_locked_audit_compare_v243_hard36_vs_hard24`
- `source_stage`: `v243_v241_guarded_finetune`
- `new_model_trained`: `False`
- `hyperparameter_changed`: `False`
- `test_used_for_retuning`: `False`
- `test_used_for_locked_audit_reporting`: `True`
- `gate_router_selector_created`: `False`
- `sample_deleted`: `False`
- `formal_headline_changed`: `False`
- `hard24_granular_prediction_available`: `False`
- `source_v243_leakage_pass`: `True`
- `same_event_uid_cross_split_count`: `0`
- `pass`: `True`

## 主要产物

- `tables/v244_validation_vs_test_candidate_compare.csv`
- `tables/v244_per_delay_hard24_hard36_compare.csv`
- `tables/v244_bucket_decision_matrix.csv`
- `tables/v244_hard36_per_sample_risk_summary.csv`
- `tables/v244_hard36_worst_regressions_vs_v241.csv`
- `tables/v244_missing_hard24_granular_audit.csv`
- `tables/v244_next_decision.csv`
- ZIP：`v244_locked_audit_compare_v243_hard36_vs_hard24_pack.zip`
