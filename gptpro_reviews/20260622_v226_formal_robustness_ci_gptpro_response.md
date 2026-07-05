## 1. v225 是否接受为 complete

**接受。v225 formal route reconstruction evidence pack 已完成并通过验收。**

理由：

```text
formal model lock 正确
baseline metrics 在 1e-5 内复现
leakage guard 全 pass
formal / diagnostic 隔离 pass
forbidden scan pass
table alignment pass
ZIP 完整
图表数量和抽查 pass
```

正式结论保持不变：

```text
loose_main_pool formal headline = avg_joint_focus
strict_main_pool formal headline = peak_floor_090
```

v222a 相关内容继续保持：

```text
v222a_bounded_residual = diagnostic-only
v222a_noharm_gate = diagnostic-only
oracle_safe_gate = diagnostic-only
ridge_residual_peakfloor = diagnostic reference only
```

---

## 2. 下一步唯一允许 Codex 执行的 bounded local step

下一步做：

```text
v226 formal robustness / confidence interval audit
```

这是 **audit-only + reporting**，不是模型训练，不是 gate，不是 router，不是阈值搜索。

### exact script name

```text
stage03_v226_formal_robustness_ci_audit_20260622.py
```

### exact output directory

```text
F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622
```

### 目的

用 v225 已锁定的 formal model，补齐论文/报告需要的稳健性证据：

```text
1. formal 指标的 bootstrap confidence interval
2. subject-level / route-event-level 稳健性
3. tail error 是否由少数样本主导
4. underestimation 是否集中在特定 bucket
5. formal model 是否足够进入论文主结果阶段
```

这一步不是继续救 v222a。
它是把已锁定 formal result 做成可写进论文的统计证据。

---

## 3. Allowed inputs and forbidden inputs

### Allowed inputs

只允许读取：

```text
v225_formal_route_reconstruction_evidence_pack_20260622/
    tables/formal_model_lock.csv
    tables/formal_reconstruction_metrics_by_pool.csv
    tables/formal_reconstruction_metrics_by_bucket.csv
    tables/formal_reconstruction_metrics_by_route_event.csv
    tables/per_sample_formal_reconstruction_eval.csv
    tables/formal_failure_case_index.csv
    logs/metric_reproduction_check.json
    logs/table_alignment_check.json
```

如 v225 表中没有足够的 point-level SSE / prediction / target 信息，允许只读取 formal cache 中的：

```text
locked formal prediction arrays
ground-truth target arrays
sample metadata
split / pool / subject / route-event labels
```

但只能读取这两个 locked formal models：

```text
loose_main_pool: avg_joint_focus
strict_main_pool: peak_floor_090
```

`y_true` 只能作为 evaluation target 使用，不能作为 model row、candidate row、selected config 或 feature。

### Forbidden inputs

禁止读取或使用：

```text
v222a_noharm_gate outputs
v222a_bounded_residual predictions
oracle_safe_gate outputs
oracle candidate rows
true_label model rows
fallback rows
W3_B4_original_soft
ridge_residual_peakfloor as formal model
any diagnostic-only row as deployable model
test-selected tau
new tau search table
router configs
v222b outputs
v223 outputs
```

禁止使用：

```text
subject_id as personalization feature
per-sample test error as selection feature
oracle best candidate id
future true peak / true under label as model input
```

---

## 4. Required output files

必须生成：

```text
v226_formal_robustness_ci_audit_20260622/
    tables/
        formal_model_lock_recheck.csv
        formal_metric_ci_sample_bootstrap.csv
        formal_metric_ci_subject_block_bootstrap.csv
        formal_subject_level_metrics.csv
        formal_route_event_level_metrics.csv
        formal_bucket_ci_metrics.csv
        formal_tail_error_concentration.csv
        formal_underestimation_profile.csv
        formal_extreme_peak_profile.csv
        formal_sample_influence_audit.csv
        formal_readiness_decision.csv

    figures/
        ci_forest_by_pool/
        subject_level_metric_distribution/
        tail_error_concentration/
        underestimation_profile/
        extreme_peak_cases_summary/

    reports/
        v226_formal_robustness_ci_audit_cn.md

    logs/
        run_manifest.json
        input_file_hashes.json
        bootstrap_config.json
        metric_reproduction_check.json
        leakage_guard_report.json
        forbidden_scan_report.json
        table_alignment_check.json
        file_inventory.json

    v226_formal_robustness_ci_audit_pack.zip
```

---

## 5. Metric and bootstrap rules

### Metric definitions

必须沿用 v225 定义，不允许重新定义 tail mask。

```text
overall RMSE = sqrt(sum squared error over all samples and horizon points / total points)
tail RMSE = sqrt(sum squared error over v225 tail horizon points / total tail points)
```

不要用：

```text
mean(per-sample RMSE)
```

来冒充 aggregate RMSE。

必须同时报告：

```text
rmse
tail_rmse
mean_sample_rmse
median_sample_rmse
p90_sample_rmse
under_rate
direction_acc
strong_steer_rate
extreme_peak_rate
```

### Bootstrap config

固定：

```text
random_seed = 20260622
n_bootstrap = 2000
ci_level = 0.95
```

需要两类 CI：

```text
1. sample bootstrap:
   在 pool/split 内按 sample 重采样

2. subject-block bootstrap:
   在 pool/split 内按 subject 重采样，抽到 subject 后包含该 subject 的全部样本
```

如果某些 bucket 样本太少：

```text
n < 10
```

则写：

```text
ci_status = insufficient_n
```

不要为了补 CI 合并或删 bucket。

---

## 6. Pass / fail criteria

### A. 代码和文件

必须通过：

```text
python -m py_compile pass
full script run pass
ZIP testzip() == None
required files missing == []
```

### B. formal model lock recheck

`formal_model_lock_recheck.csv` 必须只包含：

```text
pool,formal_model
loose_main_pool,avg_joint_focus
strict_main_pool,peak_floor_090
```

任何额外 formal model 出现即 fail。

### C. metric reproduction

`metric_reproduction_check.json` 必须复现：

```text
loose_main_pool / avg_joint_focus / test:
    rmse = 0.544884
    tail_rmse = 0.629752

strict_main_pool / peak_floor_090 / test:
    rmse = 0.571770
    tail_rmse = 0.658306
```

容忍误差：

```text
abs diff <= 1e-5
```

超过即 fail。

### D. leakage guard

`leakage_guard_report.json` 必须全部 pass：

```text
no_training_executed
no_new_tau_created
no_test_retuning
no_router_created
no_gate_created
no_v222b_or_v223
formal_model_lock_exact
no_oracle_in_formal
no_true_label_row_in_formal
no_diagnostic_model_in_formal
sample_id_alignment_pass
pool_filter_pass
split_filter_pass
tail_mask_inherited_from_v225
```

### E. forbidden scan

formal tables、logs、report 主体中不得出现：

```text
W3_B4_original_soft
oracle_model
true_label row
fallback
v222a_noharm_gate as formal
v222a_bounded_residual as formal
oracle_safe_gate as formal
ridge_residual_peakfloor as formal
```

允许在 report 的 appendix 中用一句话说明：

```text
v222a and oracle-related rows remain diagnostic-only and are not used in v226 formal robustness audit.
```

但不得把它们写入任何 formal metric table。

### F. table alignment

必须通过：

```text
no duplicate sample_id within pool/split
no missing formal prediction
prediction shape = N x 21 if arrays are used
horizon length = 21
test n:
    loose_main_pool = 184
    strict_main_pool = 174
```

### G. figure minimums

至少生成：

```text
ci_forest_by_pool >= 2 PNG
subject_level_metric_distribution >= 4 PNG
tail_error_concentration >= 2 PNG
underestimation_profile >= 2 PNG
extreme_peak_cases_summary >= 2 PNG
```

图标题必须包含：

```text
pool
formal_model
split
metric name
```

---

## 7. formal_readiness_decision.csv 规则

必须输出一行总决策和两行 pool 决策。

字段：

```text
scope
formal_model
accepted_for_paper_main_result
needs_new_model
needs_gate_or_router
needs_more_diagnostic_only
reason
```

本轮默认应写：

```text
accepted_for_paper_main_result = True
needs_new_model = False
needs_gate_or_router = False
```

除非 v226 发现严重数据对齐或指标复现失败。

允许写 limitations，例如：

```text
tail error concentrated in a small subset
underestimation remains non-negligible
extreme peak samples remain difficult
```

但这些不解锁 v222b/v223。

---

## 8. Stop conditions

### v226 停止条件

v226 完成 required files 并通过 pass/fail checks 后自动停止。

如果出现任一情况，必须停止并只报告错误，不允许启动修补性模型：

```text
formal metric 无法复现
formal model lock 被破坏
diagnostic-only row 进入 formal table
发现新 tau / gate / router / model training
sample alignment 失败
tail mask 与 v225 不一致
ZIP 不完整
```

### v222b/v223 继续禁止

继续保持：

```text
v222b_allowed = False
v223_allowed = False
```

不解锁原因：

```text
closeout 已显示主要问题是 selector/gate 泛化不稳，
candidate_missing_rate 很低，
因此更复杂的 neural gate / router 不应由当前结果解锁。
```

v226 也不是解锁步骤。
v226 只做 formal robustness statistics。

---

## 9. Codex 必须避免做什么

Codex 下一步必须避免：

```text
不要训练任何模型
不要 v222b
不要 v223
不要 v222a_gate_v2
不要 neural gate
不要 multi-router
不要新 tau
不要 threshold search
不要 test retuning
不要重新选择 formal headline
不要把 diagnostic-only row 写进 formal leaderboard
不要把 oracle / true-label / fallback 当成 candidate 或 model
不要新增 deployable feature 诊断来救 gate
不要删除 test 样本来改善 CI
不要重新定义 tail horizon
不要用 mean sample RMSE 冒充 aggregate RMSE
不要用 test 表现反推下一版配置
```

---

## Codex 下一条可执行指令

```text
Implement stage03_v226_formal_robustness_ci_audit_20260622.py.

This is audit-only and reporting-only.

Use only locked formal models:
- loose_main_pool: avg_joint_focus
- strict_main_pool: peak_floor_090

Read v225 formal outputs and, only if needed, locked formal prediction/target arrays for metric recomputation.

Do not train any model.
Do not tune any threshold.
Do not create any gate or router.
Do not run v222b or v223.
Do not use diagnostic-only rows as formal inputs.

Generate the required v226 tables, confidence intervals, robustness figures, report, logs, and ZIP.
Reproduce v225 locked test metrics within 1e-5.
Run leakage guard, forbidden scan, table alignment, and file inventory.
After v226 pack is complete, stop.
```
