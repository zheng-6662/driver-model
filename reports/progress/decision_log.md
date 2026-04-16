# 项目决策日志

这里只记录真正改变方向、标准或边界的判断。

不记录的内容：

- 普通命令执行
- 常规排查过程
- 还没有改变阶段判断的中间观察

## 决策表

| 日期 | 决策 | 白话解释 | 触发原因 | 影响范围 | 证据 / 入口 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-04-14 | 进度记录切换为“中枢页 + 每日日志 + 决策日志 + 实验登记表” | 以后不再把所有内容都塞进一份超长总档里。你先看总览，再看决策和实验，最后才需要翻历史流水。 | `project_progress_master.md` 已变成历史全集，查阅成本高，且状态、实验、基础设施记录彼此混杂 | 后续默认从新结构读写；旧总档改作历史归档 | [项目进度中枢](../project_progress_hub.md), [历史总档](../project_progress_master.md) | active |
| 2026-04-14 | response-state-aware v1 采用“显式开关 + 最小闭环 + 保守权重”的接入方式 | 新模型先小心接进去，不一口气把所有新分支和新损失都开到最猛，避免结果一乱就不知道是谁导致的。 | 需要让新分支进入可控、可对比、可逐步增强的状态，而不是一次性引入过多耦合改动 | 后续 run 比对时必须把相关开关与权重一起记录，不能只看主干脚本名字 | [历史总档](../project_progress_master.md) | active |
| 2026-04-13 | `pca_latent` teacher-state 路线方向成立，但当前 maintained 主线结果仍只算原型证据 | 这条新路线说明“方向大概率没错”，但现在的结果还不能当正式结论，更像是“原型机已经能跑起来”。 | 只读安全审查确认工程链路基本可跑通，但当前脚本仍是 sample-level 随机切分，不满足 `subject-level fixed split` | 对外或论文层面的可信证据仍需 protocol-safe 版本支撑 | [历史总档](../project_progress_master.md) | active |
| 2026-04-09 | `structured_v2` 仍值得继续，但当前证据不支持直接替代 baseline | `structured_v2` 不是彻底失败了，但也远没到“已经赢了 baseline”的程度，所以现在还只能继续观察。 | matched-schedule 公平双跑显示 tail/peak/turning_count 有收益，但 `boundary_shift_abs_err` 恶化明显 | 后续应优先收口 boundary failure mechanism，而不是直接宣称替代 baseline | [历史总档](../project_progress_master.md), [Step 4 决策摘要](../step4_decision_summary_20260408.md) | active |

## 新增格式

后续若有新的关键判断，建议直接按下面模板追加：

### YYYY-MM-DD 决策标题

- 决策：
- 白话解释：
- 触发原因：
- 影响范围：
- 证据链接：
- 后续动作：
- 状态：`active` / `superseded` / `obsolete`
