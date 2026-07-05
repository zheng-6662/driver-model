# v226 formal robustness / CI audit 报告

- 生成时间：2026-06-23T01:07:37
- 输入：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622`
- 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`
- 范围：audit-only + reporting-only；未训练模型、未调 threshold/tau、未生成 gate/router。
- formal lock：loose_main_pool 使用 avg_joint_focus；strict_main_pool 使用 peak_floor_090。
- 本轮只复用 v225 per-sample formal 表中的 `rmse`、`tail_rmse`、bucket 与 subject/split 元数据；tail 定义继承 v225。

## locked test 指标复现

| pool | formal_model | n | RMSE | tail RMSE | mean sample RMSE | under rate | direction acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| loose_main_pool | avg_joint_focus | 184 | 0.544884 | 0.629752 | 0.468061 | 0.163043 | 0.967391 |
| strict_main_pool | peak_floor_090 | 174 | 0.571770 | 0.658306 | 0.485644 | 0.137931 | 0.948276 |

- 指标复现检查：pass。
- leakage/边界检查：pass。

## sample bootstrap 95% CI（test）

| pool | metric | point | ci_lower | ci_upper |
|---|---:|---:|---:|---:|
| loose_main_pool | rmse | 0.544884 | 0.496066 | 0.593811 |
| loose_main_pool | tail_rmse | 0.629752 | 0.564811 | 0.693788 |
| strict_main_pool | rmse | 0.571770 | 0.511036 | 0.635521 |
| strict_main_pool | tail_rmse | 0.658306 | 0.581652 | 0.736696 |

## subject-block bootstrap 95% CI（test）

| pool | metric | point | ci_lower | ci_upper | n_subjects |
|---|---:|---:|---:|---:|---:|
| loose_main_pool | rmse | 0.544884 | 0.428783 | 0.599684 | 4 |
| loose_main_pool | tail_rmse | 0.629752 | 0.515881 | 0.687686 | 4 |
| strict_main_pool | rmse | 0.571770 | 0.473689 | 0.615000 | 4 |
| strict_main_pool | tail_rmse | 0.658306 | 0.539479 | 0.706505 | 4 |

## tail error 集中度（test）

| pool | top1 share | top5 share | top10 share | top20pct share | gini |
|---|---:|---:|---:|---:|---:|
| loose_main_pool | 0.038498 | 0.179141 | 0.313389 | 0.659320 | 0.612677 |
| strict_main_pool | 0.053691 | 0.205018 | 0.354324 | 0.672493 | 0.630911 |

## readiness 决策

| scope | formal_model | accepted | needs_new_model | needs_gate_or_router | reason |
|---|---|---:|---:|---:|---|
| total | locked_formal_pair | True | False | False | locked formal metrics reproduced and v226 robustness evidence packaged; remaining uncertainty is reporting uncertainty, not a reason to launch a new model. |
| loose_main_pool | avg_joint_focus | True | False | False | locked formal metrics reproduced and v226 robustness evidence packaged; remaining uncertainty is reporting uncertainty, not a reason to launch a new model. |
| strict_main_pool | peak_floor_090 | True | False | False | locked formal metrics reproduced and v226 robustness evidence packaged; remaining uncertainty is reporting uncertainty, not a reason to launch a new model. |

## 输出入口

- `tables/formal_metric_ci_sample_bootstrap.csv`：样本级 bootstrap CI。
- `tables/formal_metric_ci_subject_block_bootstrap.csv`：subject-block bootstrap CI。
- `tables/formal_subject_level_metrics.csv`：subject 级指标分布。
- `tables/formal_tail_error_concentration.csv`：tail error 集中度。
- `figures/`：CI、subject 分布、tail 集中度、低估 profile 和极端峰值概要图。
- `logs/`：复现、边界、禁用名、表对齐、文件清单与 ZIP 校验。