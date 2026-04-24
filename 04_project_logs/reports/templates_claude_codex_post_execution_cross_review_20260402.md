# 模板 3：Execution 后交叉 Review 模板

## 使用场景
适用于 Codex 已经完成一轮执行后，由 Claude 主审；必要时再回到 Codex 进行二审或反驳式检查。

---

## Codex 回传模板

```text
[Task Completed]
- 完成了什么

[What I Changed]
- 改了哪些文件
- 每个文件改动的大意是什么

[Why This Implementation]
- 为什么选择这个实现
- 为什么没有选其他方案

[Validation Performed]
- 跑了什么
- 结果是什么

[Known Uncertainty]
- 还有哪些不确定

[Remaining Risk]
- 我仍然担心的风险是什么

[If I Critique My Own Result]
- 如果我要挑自己毛病，最可能的问题是什么
```

---

## Claude 主审模板

```text
[Result Review Verdict]
- accept / revise / reject / escalate

[Correctness Check]
- 是否改到了正确的 maintained code
- 是否碰到了不该碰的目录

[Risk Check]
- split leakage
- time leakage
- label leakage
- protocol / horizon / anchor drift
- hidden fairness issue

[Validation Check]
- 当前验证是否足够支撑结论
- 是否仍需 smoke / read-only / comparison evidence

[What I Accept]

[What I Reject]

[What Still Needs Work]

[Decision]
- close
- send back to Codex
- investigate further with Claude
```

---

## Codex 二审模板（按需触发）

```text
[Why I Agree With Claude]

[Why I Disagree With Claude]

[Over-Conservative Concern]
- Claude 是否过度保守

[Missed Risk By Claude]
- Claude 是否漏看了别的实现问题

[Alternative Patch]
- 如果重做，我会怎么改

[Final Recommendation]
- keep current / revise / rollback / compare alternatives
```
