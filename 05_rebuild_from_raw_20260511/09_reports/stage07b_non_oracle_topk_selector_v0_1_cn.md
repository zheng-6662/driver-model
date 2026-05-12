# Stage 7b：非 oracle top-K selector 轻量实验 v0.1

生成时间：2026-05-13 06:39

本轮不使用生理、EEG、连续风格或被试 ID。selector 输入显式剔除 label-derived 字段。

## Gate

| gate                                 | status      | evidence                                                                                            | decision                |
| ------------------------------------ | ----------- | --------------------------------------------------------------------------------------------------- | ----------------------- |
| stage07b_deployable_selector_upgrade | no_upgrade  | selected test RMSE delta=+0.000000; wrong-side=0.225 vs RBF 0.225; large recall=0.750 vs RBF 0.750. | 当前轻量 selector 不升级主线。    |
| oracle_gap_remaining                 | still_large | oracle RMSE=0.415652; selected RMSE=0.533667.                                                       | 仍需改进非 oracle 选择策略或候选表示。 |
| stage05_physio_eeg_allowed           | blocked     | 车辆-only Stage 7b 轻量 selector 未形成可升级结果。                                                              | 继续阻塞生理/EEG有效性结论。        |

## Selected Test Metrics

| policy_name                                     | rmse_steer | wrong_side_rate | large_response_recall | difficult_top20_rmse | selected_for_report |
| ----------------------------------------------- | ---------- | --------------- | --------------------- | -------------------- | ------------------- |
| oracle_best_of_rbf_topk_not_deployable          | 0.415652   | 0.075000        | 0.875000              | 0.604369             | 0.000000            |
| always_rbf_reference                            | 0.533667   | 0.225000        | 0.750000              | 0.678907             | 0.000000            |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80 | 0.533667   | 0.225000        | 0.750000              | 0.678907             | 1.000000            |
| topk_top1_non_oracle                            | 0.587865   | 0.100000        | 0.750000              | 0.717094             | 0.000000            |

## Diagnostics

| policy_name                                     | oracle_choice_accuracy | mean_confidence | brier_multiclass | ece_5bin |
| ----------------------------------------------- | ---------------------- | --------------- | ---------------- | -------- |
| oracle_best_of_rbf_topk_not_deployable          | 1.000000               |                 |                  | 0.000000 |
| logreg_balanced_c1                              | 0.400000               | 0.645222        | 0.886752         | 0.245222 |
| logreg_balanced_c1__fallback_rbf_conf_lt_0.35   | 0.400000               | 0.645222        | 0.886752         | 0.245222 |
| logreg_balanced_c1__fallback_rbf_conf_lt_0.00   | 0.400000               | 0.645222        | 0.886752         | 0.245222 |
| always_rbf_reference                            | 0.375000               |                 |                  | 0.000000 |
| rf_shallow_balanced__fallback_rbf_conf_lt_0.50  | 0.375000               | 0.324309        | 0.760504         | 0.073424 |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.50 | 0.375000               | 0.520229        | 0.800872         | 0.145229 |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.00 | 0.375000               | 0.520229        | 0.800872         | 0.145229 |
| logreg_balanced_c0_2                            | 0.375000               | 0.520229        | 0.800872         | 0.145229 |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.65 | 0.375000               | 0.520229        | 0.800872         | 0.145229 |
| rf_shallow_balanced__fallback_rbf_conf_lt_0.65  | 0.375000               | 0.324309        | 0.760504         | 0.073424 |
| rf_shallow_balanced__fallback_rbf_conf_lt_0.80  | 0.375000               | 0.324309        | 0.760504         | 0.073424 |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80 | 0.375000               | 0.520229        | 0.800872         | 0.145229 |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.35 | 0.350000               | 0.520229        | 0.800872         | 0.170229 |
| logreg_balanced_c1__fallback_rbf_conf_lt_0.50   | 0.350000               | 0.645222        | 0.886752         | 0.295222 |
| logreg_balanced_c1__fallback_rbf_conf_lt_0.65   | 0.350000               | 0.645222        | 0.886752         | 0.295222 |
| logreg_balanced_c1__fallback_rbf_conf_lt_0.80   | 0.325000               | 0.645222        | 0.886752         | 0.320222 |
| rf_shallow_balanced__fallback_rbf_conf_lt_0.35  | 0.275000               | 0.324309        | 0.760504         | 0.049309 |
| rf_shallow_balanced__fallback_rbf_conf_lt_0.00  | 0.275000               | 0.324309        | 0.760504         | 0.049309 |
| rf_shallow_balanced                             | 0.275000               | 0.324309        | 0.760504         | 0.049309 |

## Coverage Risk

| policy_name                                     | split | coverage | mean_sample_rmse | wrong_side_rate | mean_confidence |
| ----------------------------------------------- | ----- | -------- | ---------------- | --------------- | --------------- |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80 | test  | 0.500000 | 0.499840         | 0.350000        | 0.641921        |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80 | test  | 0.750000 | 0.479778         | 0.300000        | 0.568459        |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80 | test  | 1.000000 | 0.476061         | 0.225000        | 0.520229        |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80 | val   | 0.500000 | 0.447836         | 0.238095        | 0.627866        |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80 | val   | 0.750000 | 0.471354         | 0.156250        | 0.558766        |
| logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80 | val   | 1.000000 | 0.493531         | 0.119048        | 0.505676        |
