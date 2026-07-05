我已审阅 `v229_two_month_lessons_failure_taxonomy_pack.zip`。结论是：**v229 接受，模型工作继续停止；下一步只允许做“失败案例人工复核包 / 论文案例证据包”，不是继续训练。**

---

## 1. v229 是否接受

**接受 v229。**

理由：

* v229 是 retrospective-only；
* 没有训练模型；
* 没有生成新预测；
* 没有新 tau / threshold；
* 没有 gate / router / selector；
* 没有解锁 v222b / v223；
* formal lock 保持：

  * `loose_main_pool = avg_joint_focus`
  * `strict_main_pool = peak_floor_090`
* guardrail pass；
* diagnostic-only boundary preserved；
* v229 给出的失败分类与前面 v222a closeout、v225-v228 结论一致。

核心判断接受：

> 当前主要瓶颈不是 candidate pool 大面积缺曲线，而是强反应、极端峰值、尾段延续、反转/多次修正这些局部困难样本，以及 selector/gate 在 current-window deployable feature 下泛化不稳。

因此继续禁止：

```text
v222b
v223
new gate/router
new tau/threshold
test-based retuning
formal headline change
```

---

## 2. 下一步唯一允许的 bounded local step

下一步做：

```text
stage03_v230_failure_case_manual_review_casebook_20260623.py
```

任务类型：

```text
audit-only + paper-case packaging
```

目的：

> 把 v229 已经识别出的失败桶，整理成可人工复核、可放论文、可给导师看的“失败案例证据包”。

这一步**不是新实验**，也不是模型推进。
它只是把最坏尾段、强反应低估、极端峰值、反转/多次修正、普通曲线可控样本整理成 casebook。

---

## 3. 允许读取的输入

只允许读取：

```text
05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/
05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/
05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/
05_rebuild_from_raw_20260511/03_baselines/v229_two_month_lessons_failure_taxonomy_20260623/
```

重点输入文件：

```text
v229/tables/v229_failure_taxonomy_by_pool_event.csv
v229/tables/v229_top_tail_failure_cases.csv
v229/tables/v229_bucket_risk_summary.csv
v229/tables/v229_selector_candidate_diagnosis.csv
v229/tables/v229_next_action_decision_matrix.csv
v228/tables/final_formal_model_lock.csv
v228/tables/final_main_result_table.csv
v228/tables/final_ci_table.csv
v225/tables/formal_failure_case_index.csv
v225/tables/per_sample_formal_reconstruction_eval.csv
```

允许复制已有 figure：

```text
v225/figures/worst_tail_cases/
v225/figures/strong_under_cases/
v225/figures/baseline_sufficient_cases/
v226/figures/tail_error_concentration/
v226/figures/underestimation_profile/
v226/figures/extreme_peak_cases_summary/
v228/figures/selected_main_figures/
v228/figures/selected_appendix_figures/
```

如果某些 v229 case 的 `figure_path` 为空，只允许标记为：

```text
figure_missing
```

不要重新生成预测，也不要为了补图启动模型。

---

## 4. 必须禁止的输入和动作

禁止读取或使用：

```text
v222a_noharm_gate selected configs
v222a bounded residual as formal model
oracle_safe_gate as formal result
oracle candidate rows
true_label model rows
fallback rows
W3_B4_original_soft
v222b outputs
v223 outputs
new prediction arrays
new model cache
new tau / threshold config
new router / gate config
```

禁止动作：

```text
不要训练模型
不要生成新预测
不要重新选择 formal headline
不要调 tau / threshold
不要做 gate/router/selector
不要 test retuning
不要删除困难样本
不要把 oracle 写成可部署结果
不要把 v222a 写成 formal improvement
不要把 case study 结论扩大成 aggregate improvement
```

---

## 5. 必须产出的文件

输出目录：

```text
05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/
```

必须产出：

```text
tables/
    v230_case_selection_index.csv
    v230_manual_review_template.csv
    v230_failure_casebook_table.csv
    v230_bucket_to_claim_mapping.csv
    v230_case_figure_inventory.csv
    v230_formal_boundary_check.csv

reports/
    v230_failure_case_manual_review_casebook_cn.md
    v230_advisor_discussion_notes_cn.md
    v230_paper_failure_case_section_draft_cn.md

figures/
    selected_casebook_figures/
        loose_main_pool/
        strict_main_pool/
        cross_pool_repeated_cases/
        baseline_sufficient_controls/

logs/
    run_manifest.json
    input_file_hashes.json
    guardrail_check.json
    forbidden_scan_report.json
    file_inventory.json
    figure_copy_check.json
    consistency_check.json

v230_failure_case_manual_review_casebook_pack.zip
```

---

## 6. case 选择规则

每个 pool 至少选择：

```text
强反应低估：5 个
极端峰值低估 / 极端峰值尾段难例：4 个
强响应幅值/尾段：5 个
反转或多次修正：3 个
过零/换向边界：3 个
普通曲线可控 baseline sufficient：3 个
```

总数建议控制在：

```text
每个 pool 20-25 个
总计 40-50 个以内
```

优先选择：

```text
tail_rmse 高
under_flag=True
extreme_peak=True
strong_steer=True
reverse=True
figure 存在
cross-pool 重复出现
```

不要为了凑数引入非 formal 模型案例。

---

## 7. `v230_manual_review_template.csv` 必须包含的人工复核字段

```text
pool
sample_id
formal_model
scene_type
route_event
failure_bucket_v229
rmse
tail_rmse
under_flag
strong_steer
extreme_peak
reverse
zero_cross
multi_correction
observed_peak_abs
pred_peak_abs
peak_ratio
figure_path
review_status
human_primary_failure_label
human_secondary_failure_label
is_anchor_suspicious
is_prediction_direction_correct
is_tail_lag_visible
is_peak_flattened
is_reverse_missed_or_delayed
is_vehicle_response_mismatch
paper_figure_candidate
advisor_discussion_candidate
human_notes
```

其中 `human_*` 字段先留空，供人工复核填写。
Codex 不要伪造人工结论。

---

## 8. `v230_bucket_to_claim_mapping.csv` 必须支持的论文 claim

至少包含这些 claim：

```text
C1_direction_is_stable_but_amplitude_hard
C2_tail_error_concentrates_in_difficult_cases
C3_strong_reaction_underestimation_remains
C4_extreme_peak_cases_are_key_limitation
C5_reverse_multi_correction_are_not_solved_by_average_rmse
C6_normal_curve_cases_are_relatively_controlled
C7_v222a_selector_gap_is_diagnostic_not_formal
```

每条 claim 必须标注：

```text
claim_strength = main / limitation / diagnostic_only
allowed_wording
forbidden_wording
supporting_table
supporting_cases
```

---

## 9. pass / fail criteria

必须全部通过：

```text
python -m py_compile stage03_v230_failure_case_manual_review_casebook_20260623.py
full script run pass
ZIP testzip() == None
required files missing == []
guardrail_check.pass == true
consistency_check.pass == true
forbidden_scan_report.hits == []
formal lock exact:
    loose_main_pool = avg_joint_focus
    strict_main_pool = peak_floor_090
no new model outputs created
no new prediction arrays created
no new tau / gate / router config created
no diagnostic-only row appears in formal casebook tables
selected case count >= 40 unless insufficient figures are explicitly logged
each selected case has a valid bucket label
each selected case links to existing evidence table row
figure_copy_check passes or logs figure_missing explicitly
```

---

## 10. stop condition

v230 完成后自动停止。

如果出现任一情况，必须停止并报错，不允许补救性训练：

```text
formal lock 被改变
需要新预测才能完成
需要新 tau / threshold 才能完成
diagnostic-only row 进入 formal casebook
oracle / true_label / fallback 进入 formal 表
v222b / v223 被调用
case 选择规则根据 test 重新优化模型配置
人工复核字段被自动伪造
```

v230 之后的唯一合理下一步是：

```text
人工阅读 casebook
填写 manual_review_template.csv
然后写论文失败案例小节
```

不是启动新模型。

---

## 11. 给 Codex 的下一条指令

可以直接发：

```text
Implement stage03_v230_failure_case_manual_review_casebook_20260623.py.

This is audit-only and paper-case packaging only.

Use existing v225, v226, v228, and v229 outputs.
Do not train models.
Do not generate new predictions.
Do not tune tau or thresholds.
Do not create gates, routers, or selectors.
Do not run v222b or v223.
Do not change formal headline.
Do not use oracle, true-label, fallback, W3_B4_original_soft, or diagnostic-only rows as formal evidence.

Build a compact manual-review casebook from v229 failure taxonomy:
- strong underestimation
- extreme peak failure
- tail amplitude failure
- reverse / multi-correction failure
- zero-cross boundary
- normal-curve control cases

Generate the required tables, reports, copied figure bundle, logs, and ZIP.
Leave human review fields blank.
Run py_compile, full script, ZIP integrity, guardrail, forbidden scan, consistency check, and figure-copy check.
After v230 pack is complete, stop.
```
