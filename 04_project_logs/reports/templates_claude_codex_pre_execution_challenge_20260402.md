# 模板 2：Execution 前质疑模板

## 使用场景
适用于 Codex 在真正执行前，先对任务进行一次实现层质疑。

目标不是立刻动手，而是避免：
- 审题不清
- 文件选错
- 范围过大
- 验证不足
- 有更小路径却没走

---

## Codex 执行前必答模板

```text
[Understanding Check]
- 我认为这次任务真正目标是：
- 这次任务不包含的内容是：

[Files To Read First]
1.
2.
3.

[Files That Should Probably Not Be Touched]
1.
2.
3.

[Implementation Risk]
- 我认为最可能出错的点：
- 我认为最容易误改的点：
- 我认为最容易造成副作用的点：

[Validation Sufficiency Check]
- 当前要求的验证是否够？
- 如果不够，还需要什么验证？

[Smaller Entry Point]
- 是否存在更小的实现路径？
- 是否可以先做 smoke change / dry-run / read-only validation？

[Go / No-Go]
- go
- go after clarification
- no-go until plan revised

[If I Were To Challenge The Task]
- 我最想退回重新澄清的一点是：
```

---

## Claude 对质疑结果的处理模板

```text
[Clarification Accepted]

[Clarification Rejected]

[Scope Correction]

[Validation Correction]

[Execution Decision]
- proceed
- revise brief
- stop
```
