# 2026-06-23 GPTPro v230 casebook decision

## 结论

接受 GPTPro 对 v229 的审阅结论，并执行唯一允许的本地下一步：

`stage03_v230_failure_case_manual_review_casebook_20260623.py`

## 接受的范围

- 任务类型：audit-only + paper-case packaging。
- 目标：把 v229 已识别的失败桶整理成可人工复核、可给导师讨论、可进入论文失败案例小节的 casebook。
- 允许读取：v225、v226、v228、v229 已完成并验证过的输出。
- 允许复制：v225/v226/v228 既有 figure。
- 若 case 缺少图，只标记 `figure_missing`，不重新生成预测。

## 明确拒绝或继续禁止的范围

- 不训练模型。
- 不生成新预测。
- 不调整 tau / threshold。
- 不创建 gate / router / selector。
- 不运行 v222b / v223。
- 不修改 formal headline。
- 不使用 oracle、true-label、fallback、`W3_B4_original_soft` 或 diagnostic-only 行作为 formal evidence。
- 不把 case study 结论扩大成 aggregate improvement。

## 当前 formal lock

- `loose_main_pool = avg_joint_focus`
- `strict_main_pool = peak_floor_090`

## 本地执行要求

- 生成 v230 tables / reports / figures / logs / ZIP。
- `v230_manual_review_template.csv` 的人工复核字段必须留空，Codex 不伪造人工判断。
- 运行 `py_compile`、完整脚本、ZIP 校验、guardrail、forbidden scan、consistency check 和 figure-copy check。
- v230 完成后停止，不继续启动新模型。
