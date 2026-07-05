# GPTPro response - v225 formal route reconstruction evidence pack

Source: ChatGPT Desktop / Pro extension, copied after the v222a closeout result report and compact follow-up prompt.

## 1. 是否接受 closeout 诊断

**接受。**

当前主要失败是：

```text
selector / gate 泛化不稳
```

而不是：

```text
candidate pool 大面积缺曲线
```

依据：

```text
combined selector_failed_rate = 0.410615
combined candidate_missing_rate = 0.027933
high-tail candidate_missing_rate = 0.126582
high-tail oracle clear gain rate = 0.911392
```

这说明现有候选池里经常存在更优曲线，但 deployable gate 学不稳定。
因此：

```text
v222a bounded residual / no-harm gate = diagnostic-only
v222b_allowed = False
v223_allowed = False
```

---

## 2. 下一步唯一允许 Codex 执行的 bounded local step

执行一个 **formal route reconstruction evidence pack**。

建议命名：

```text
stage03_v225_formal_route_reconstruction_evidence_pack_20260622.py
```

输出目录：

```text
v225_formal_route_reconstruction_evidence_pack_20260622/
```

目的：

```text
不再继续优化 v222a。
不再训练新模型。
只用 locked formal baseline 生成正式重建证据、分桶指标、失败案例和论文级图表。
```

正式锁定：

```text
loose_main_pool formal model = avg_joint_focus
strict_main_pool formal model = peak_floor_090
```

v222a 相关内容只能出现在 diagnostic appendix：

```text
v222a_bounded_residual = diagnostic-only
v222a_noharm_gate = diagnostic-only
oracle_safe_gate = upper-bound diagnostic-only
```

---

## 3. 需要产出的 exact files

必须输出：

```text
v225_formal_route_reconstruction_evidence_pack_20260622/
    tables/
        formal_model_lock.csv
        formal_reconstruction_metrics_overall.csv
        formal_reconstruction_metrics_by_pool.csv
        formal_reconstruction_metrics_by_bucket.csv
        formal_reconstruction_metrics_by_route_event.csv
        per_sample_formal_reconstruction_eval.csv
        formal_failure_case_index.csv
        diagnostic_only_v222a_closeout_summary.csv
        excluded_diagnostic_models_audit.csv

    figures/
        formal_examples/
            loose_main_pool/
            strict_main_pool/
        worst_tail_cases/
            loose_main_pool/
            strict_main_pool/
        strong_under_cases/
            loose_main_pool/
            strict_main_pool/
        baseline_sufficient_cases/
            loose_main_pool/
            strict_main_pool/

    reports/
        v225_formal_route_reconstruction_evidence_cn.md

    logs/
        run_manifest.json
        leakage_guard_report.json
        forbidden_scan_report.json
        metric_reproduction_check.json
        file_inventory.json

    v225_formal_route_reconstruction_evidence_pack.zip
```

`formal_model_lock.csv` 必须只包含：

```text
pool,formal_model,source,usage
loose_main_pool,avg_joint_focus,v221_formal_leaderboard,formal_headline
strict_main_pool,peak_floor_090,v221_formal_leaderboard,formal_headline
```

`diagnostic_only_v222a_closeout_summary.csv` 可以引用：

```text
v222a_bounded_residual
v222a_noharm_gate
oracle_safe_gate
ridge_residual_peakfloor
```

但必须标注：

```text
usage = diagnostic_only
allowed_in_formal = False
```

---

## 4. pass/fail criteria

必须全部通过才算完成。

### A. 代码与文件完整性

```text
python -m py_compile pass
script full run pass
ZIP bad_file=None
required files missing=[]
```

### B. formal baseline 复现

`metric_reproduction_check.json` 必须确认 locked test 指标复现：

```text
loose_main_pool avg_joint_focus:
    RMSE = 0.544884
    tail RMSE = 0.629752

strict_main_pool peak_floor_090:
    RMSE = 0.571770
    tail RMSE = 0.658306
```

允许误差：

```text
absolute tolerance <= 1e-5
```

如果超过，说明 sample alignment 或 pool filter 有问题，直接 fail。

### C. formal / diagnostic 隔离

`forbidden_scan_report.json` 必须确认以下名称没有出现在任何 formal usage、formal selected config、formal leaderboard 表中：

```text
W3_B4_original_soft
oracle
oracle_model
true_label
fallback
v222a_noharm_gate
v222a_bounded_residual
oracle_safe_gate
```

这些名称只允许出现在：

```text
diagnostic_only_v222a_closeout_summary.csv
excluded_diagnostic_models_audit.csv
report 的 diagnostic appendix
```

### D. leakage guard

`leakage_guard_report.json` 必须全部 pass：

```text
formal_model_lock_exact
no_training_executed
no_new_tau_created
no_test_retuning
no_router_created
no_v222b_or_v223
no_oracle_in_formal
no_true_label_in_formal
sample_id_alignment_pass
pool_filter_pass
split_filter_pass
```

### E. 指标表一致性

以下表之间必须 row count 和 sample_id 对齐：

```text
per_sample_formal_reconstruction_eval.csv
formal_reconstruction_metrics_by_route_event.csv
formal_failure_case_index.csv
```

要求：

```text
no duplicate sample_id within pool/split
no missing formal prediction
prediction shape = N x 21
horizon length = 21
```

### F. 图表验收

至少生成：

```text
formal_examples >= 12 PNG
worst_tail_cases >= 12 PNG
strong_under_cases >= 8 PNG
baseline_sufficient_cases >= 8 PNG
```

图标题必须显示：

```text
pool
sample_id
formal_model
RMSE
tail RMSE
under flag
```

---

## 5. stop conditions

### v222a 主线停止条件

已经触发，继续保持停止：

```text
validation pass=True
locked test pass=False
且 loose / strict 出现相反失败模式
```

因此：

```text
STOP v222a threshold tuning
STOP v222a no-harm gate optimization
STOP v222a bounded residual formalization
STOP v222a_gate_v2
```

### v225 evidence pack 停止条件

v225 是一次性证据包，完成后自动停止。

如果出现任一情况，必须停止并只报错，不得启动新模型修补：

```text
formal baseline 指标无法复现
sample_id / split / pool 对齐失败
diagnostic-only model 进入 formal 表
出现新 tau / 新 gate / 新 router 配置
出现 v222b 或 v223 训练痕迹
ZIP 或 required files 不完整
```

### v222b/v223 继续禁止条件

本轮继续禁止：

```text
v222b_allowed=False
v223_allowed=False
```

解除禁止的条件不是本轮任务的一部分。
当前 closeout 已经表明：

```text
candidate_missing 不高
oracle clear gain 很高
问题主要是 selector 泛化
```

所以不能用当前结果作为启动 v222b/v223 的理由。

---

## 6. 必须避免做什么

Codex 下一步必须避免：

```text
不要 test retuning
不要新 tau
不要 v222a_gate_v2
不要 multi-router
不要 neural gate
不要 v222b 训练
不要 v223 训练
不要新增大模型
不要重新选择 formal headline
不要把 oracle / true-label / fallback / diagnostic-only row 写入 formal leaderboard
不要把 v222a_noharm_gate 写成 formal model
不要把 oracle_safe_gate 写成 deployable result
不要根据 test failure 改阈值
不要根据 test failure 删除样本
不要把 subject_id 用作 personalization feature
不要新增 deployable feature 诊断来继续救 gate
不要把 case study 图表解释成 formal aggregate improvement
```

Codex 下一条执行指令可以写成：

```text
Implement stage03_v225_formal_route_reconstruction_evidence_pack_20260622.py.

Use only locked formal models:
- loose_main_pool: avg_joint_focus
- strict_main_pool: peak_floor_090

Do not train any model.
Do not tune any threshold.
Do not create any router or gate.
Do not run v222b/v223.
Keep v222a/oracle outputs diagnostic-only.

Generate the exact required files, reproduce locked baseline metrics within 1e-5, run leakage/forbidden scans, create figures and report, zip the pack, then stop.
```
