# v227 写作 / claim readiness 整理报告

## 结论

v227 不是新实验，也不是 GPTPro 新指令解锁的模型路线。由于 v226 结果回报 GPTPro 时桌面端连续空停、Chrome 端需要登录，当前没有新的 GPTPro 正文指令。为了不让本地工作卡死，本轮只做 reporting-only 的写作材料整理：把 v225 锁定 formal 结果和 v226 稳健性 / CI 审计整理成可写入论文或组会材料的 claim/readiness 包。

本轮没有训练模型、没有搜索阈值、没有创建 gate/router、没有运行 v222b/v223、没有改变 formal headline。

## formal 主结果仍然锁定

| pool | formal model | test n | RMSE | tail RMSE | sample RMSE CI | subject-block RMSE CI |
|---|---|---:|---:|---:|---|---|
| loose_main_pool | avg_joint_focus | 184 | 0.544884 | 0.629752 | 0.496066-0.593811 | 0.428783-0.599684 |
| strict_main_pool | peak_floor_090 | 174 | 0.571770 | 0.658306 | 0.511036-0.635521 | 0.473689-0.615000 |

## 可以写入论文的表述边界

- 可以写：v225/v226 共同支持 locked formal result，且指标复现、泄漏检查、forbidden scan、table alignment 和 ZIP 完整性均通过。
- 可以写：v226 给出了 sample bootstrap 与 subject-block bootstrap 的不确定性区间。
- 可以写：tail error 仍集中在少量困难样本上，这是 limitation，不是继续本地模型搜索的解锁条件。
- 不可以写：v227 发现了新模型提升。
- 不可以写：v227 或 GPTPro 解锁了 v222b/v223、new tau、gate/router 或 test retuning。
- 不可以写：诊断-only 行可以进入 formal leaderboard。

## 主要 limitation

- loose test top-20% tail-SSE share = 0.659320，strict = 0.672493，说明尾部误差仍集中。
- loose under_rate = 0.163043，strict under_rate = 0.137931，仍需在论文里解释低估模式。
- 当前 GPTPro 回报通道暂时没有有效回复，因此 v227 只能作为本地写作整理包，后续仍需在 GPTPro 可用时回报。

## 输出文件

- `tables/paper_main_result_table.csv`
- `tables/paper_claim_support_matrix.csv`
- `tables/paper_limitation_table.csv`
- `tables/formal_guardrail_summary.csv`
- `tables/formal_artifact_manifest.csv`
- `tables/figure_selection_index.csv`
- `tables/gptpro_bridge_status.csv`
- `reports/v227_paper_claim_readiness_cn.md`
- `logs/run_manifest.json`
- `logs/input_file_hashes.json`
- `logs/source_artifact_checks.json`
- `logs/no_model_change_guard.json`
- `logs/file_inventory.json`
- `logs/zip_integrity_check.json`
- `v227_paper_claim_readiness_pack.zip`

## 表和图

- claim support rows: 6
- limitation rows: 7
- copied figure rows: 20

## 下一步

当 GPTPro 通道恢复时，应把 v226+v227 的执行结果一起回报 GPTPro，请它只给 bounded writing/claim/reporting 下一步，继续禁止模型训练、new tau、gate/router、v222b/v223 和 test-based retuning。
