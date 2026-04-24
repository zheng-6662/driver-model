# 模板 1：Plan 双审模板

## 使用场景
适用于：
- 新功能 / 新实验方案
- 涉及多文件改动
- 风险不确定的任务
- 需要先判断路径是否正确的任务

---

## Claude 初版计划输出模板

```text
[Task Type]
- high-risk / medium-risk / low-risk / interpretation

[Goal]
- 这次要解决什么问题
- 为什么现在做这件事

[Current Understanding]
- 我对仓库现状的理解
- 我认为相关的 active code / 配置 /输出在哪里

[Scope]
- 允许涉及的文件：
- 建议优先读的文件：
- 明确禁止触碰的区域：

[Risk Check]
- 可能的 protocol 风险：
- 可能的 split / leakage 风险：
- 可能的路径误判风险：
- 可能的实验公平性风险：

[Plan v1]
1.
2.
3.

[Validation]
- 最少需要哪些验证证据
- 什么结果算完成

[Need Codex To Review]
- 这个 plan 有没有更简单路径？
- 有没有遗漏风险？
- 范围是否过大？
- 有没有更小切入点？
```

---

## Codex 对 Plan 的 review 模板

```text
[Plan Review Verdict]
- accept / revise / high-risk concern

[What Looks Right]
- 当前计划中合理的部分

[Missing Risks]
- 漏掉了哪些风险

[Boundary Problems]
- 哪些文件范围可能过大/过窄
- 哪些文件可能不该碰

[Simpler Alternative]
- 是否存在更小、更安全、更快验证的路径

[Execution Advice]
- 如果执行，建议从哪一步开始

[If I Challenge This Plan]
- 我最质疑的一个点是什么
```

---

## Claude 收口模板

```text
[Accepted From Codex]

[Rejected From Codex]

[Reason For Rejection]

[Final Execution Scope]
- 最终允许改动文件：
- 最终禁止触碰区域：

[Final Validation Gate]

[Decision]
- proceed / revise again / stop
```
