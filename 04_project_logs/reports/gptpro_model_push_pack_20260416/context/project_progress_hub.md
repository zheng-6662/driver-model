# 项目进度中枢

这份文件是后续默认入口。

原则很简单：

- 先在这里看当前状态、当前优先级、最近关键判断。
- 想知道某天做了什么，再看 `reports/progress/daily/`。
- 想知道方向为什么变了，再看 `reports/progress/decision_log.md`。
- 想知道某个 run 到底改了什么，再看 `reports/progress/experiment_registry.md`。
- 历史全集继续保留在 `reports/project_progress_master.md`，但后续不再把所有信息都堆回一个文件。

## 当前状态

- 研究主线：极限工况下驾驶员真实反应建模与预测，用于后续辅助驾驶介入/协同控制参考。
- 当前可信判断：`pca_latent` teacher-state 工程链路已经能跑通，但 maintained 主线训练脚本仍未完全落实 protocol 要求的 `subject-level fixed split`，因此现有 smoke / 短跑结果仍应视为原型证据，而不是正式可信结论。
- 当前模型问题：`structured_v2` 相对 `baseline-conditioned` 仍可能有净收益，但 matched-schedule 公平比较下 `boundary_shift_abs_err` 明显恶化，现阶段还不支持直接替代 baseline。
- 当前协作方式：按“用户给目标 + 验收 + 红线，Claude / Codex 持续推进”的模式工作；默认禁删文件、先写短日志再做压缩总结。
- 当前记录方式：进度记录改为“中枢页 + 每日日志 + 决策日志 + 实验登记表”，`project_progress_master.md` 改作历史总档。

## 30 秒白话版

- 现在项目里同时有几条模型线，但真正“能当正式结论”的证据还不多，很多结果目前只能算原型验证。
- `structured_v2` 不是失败了，而是“有亮点，但关键问题还没解决”，所以不能直接顶替 baseline。
- `pca_latent` 这条 teacher-state 新路线已经证明“至少能跑通、能继续投资源”，但还没到可以放心写进正式论文的程度。
- 以后看进度建议先看网页站点或这份中枢页，不要一上来就翻历史总档。

## 当前优先级

1. 让项目状态可快速查阅，不再要求每次都翻历史全集。
2. 让所有新实验都能一眼看出“改了什么、结果怎样、是否继续”。
3. 让真正改变方向的判断单独沉淀，避免和日常排查混在一起。
4. 在模型主线上继续优先守住 protocol-safe、可复现、可解释的证据口径。

## 最近关键判断

| 日期 | 类型 | 一句话结论 | 入口 |
| --- | --- | --- | --- |
| 2026-04-14 | 工作流 | 进度记录正式切换为“中枢 + 分册”，历史总档保留但不再承担全部追加写入 | [决策日志](progress/decision_log.md) |
| 2026-04-14 | 模型 | response-state-aware v1 已形成显式开关 + 最小闭环，后续 run 必须把相关开关与权重当作实验条件的一部分记录 | [决策日志](progress/decision_log.md) |
| 2026-04-13 | 模型 | `pca_latent` teacher-state 路线方向成立，但当前 maintained 主线仍不是 publication-safe 证据链 | [决策日志](progress/decision_log.md) |
| 2026-04-09 | 模型 | `structured_v2` 仍值得继续，但当前证据不支持直接替代 baseline | [决策日志](progress/decision_log.md) |

## 最近实验快照

| 日期 | 实验 / 分析 | 一句话结果 | 入口 |
| --- | --- | --- | --- |
| 2026-04-14 | `pca_latent` 1000 样本短跑 | 主线稳定收敛，但 strong reversal 标签极稀，`rev_head` 退化明显 | [实验登记表](progress/experiment_registry.md) |
| 2026-04-14 | `pca_latent` 256 样本 smoke | 新 teacher-state 主线已能在真实数据上完整跑通 | [实验登记表](progress/experiment_registry.md) |
| 2026-04-09 | `structured_v2_TF0` | teacher forcing 对结果模式有实质影响，但仍缺 baseline TF0 对照 | [实验登记表](progress/experiment_registry.md) |
| 2026-04-09 | `structured_v2_noresid` | 关闭 residual 只小幅缓解 boundary_shift，整体表现反而更差 | [实验登记表](progress/experiment_registry.md) |

## 建议查阅路径

- 2 分钟了解当前状态：看本页。
- 10 分钟了解最近方向变化：看 `progress/decision_log.md` 和 `progress/experiment_registry.md`。
- 追溯完整过程：看 `progress/daily/` 对应日期，再按链接下钻到独立报告或旧总档。

## 快速入口

- [网页看板](project_progress_dashboard.html)
- [决策中心网页](project_progress_decisions.html)
- [实验中心网页](project_progress_experiments.html)
- [日志时间线网页](project_progress_daily.html)
- [术语词典网页](project_progress_glossary.html)
- [每日日志](progress/daily/2026-04-14.md)
- [决策日志](progress/decision_log.md)
- [实验登记表](progress/experiment_registry.md)
- [术语词典与命名指南](progress/glossary.md)
- [记录规则](progress/recording_rules.md)
- [AI 记录协议](progress/ai_recording_protocol.md)
- [模板目录](progress/templates/daily_log_template.md)
- [历史总档](project_progress_master.md)
