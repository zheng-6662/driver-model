# v228 最终论文产物冻结报告

## 结论

v228 按本地 GPTPro 软件端的有效回复执行：接受 v227 作为 reporting-only closeout，并生成最终论文产物冻结包。该版本没有训练模型、没有生成新预测、没有搜索阈值、没有创建新路由/选择器，也没有改变 formal headline。

最终 formal model lock 仍为：

| pool | formal model |
|---|---|
| loose_main_pool | avg_joint_focus |
| strict_main_pool | peak_floor_090 |

## 主结果冻结

| pool | formal model | test n | RMSE | tail RMSE | sample RMSE CI | subject-block RMSE CI |
|---|---|---:|---:|---:|---|---|
| loose_main_pool | avg_joint_focus | 184 | 0.544884 | 0.629752 | 0.496066-0.593811 | 0.428783-0.599684 |
| strict_main_pool | peak_floor_090 | 174 | 0.571770 | 0.658306 | 0.511036-0.635521 | 0.473689-0.615000 |

## claim 与 limitation 边界

- final claim lock 保留 5 条 v227 已有正式 claim，移除了旧的过程性通道阻塞 claim。
- final limitations 保留 6 条正式模型相关 limitation，移除了旧的过程性 pending 项。
- 图文件从 v227 已选图复制而来：主图 6 张，附录图 14 张。

## 输出文件

- `tables/final_formal_model_lock.csv`
- `tables/final_main_result_table.csv`
- `tables/final_ci_table.csv`
- `tables/final_claim_lock_table.csv`
- `tables/final_limitations_table.csv`
- `tables/final_figure_selection_table.csv`
- `tables/final_artifact_manifest.csv`
- `tables/final_guardrail_summary.csv`
- `reports/manuscript_results_section_draft_cn.md`
- `reports/manuscript_claim_boundary_notes_cn.md`
- `logs/consistency_check.json`
- `logs/forbidden_scan_report.json`
- `logs/guardrail_check.json`
- `v228_final_paper_artifact_freeze_pack.zip`
