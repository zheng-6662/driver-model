# Claude + Codex 默认短版工作流

## 适用范围
默认用于：
- 大多数 medium-risk 任务
- 一般方案 review
- 小到中等范围脚本/文档/分析任务
- 需要双向思考，但不值得跑完整重流程的任务

如果任务会影响以下任一项，则自动升级为 high-risk，并额外挂高风险清单：
- protocol / split / label / horizon / anchor
- 数据边界或样本选择逻辑
- 实验公平性 / 可比性
- 结果解释口径
- maintained code 主路径

---

## 一轮默认短版流程

### 1. Claude 先给初版任务卡
```text
Task type:
Goal:
Scope:
Forbidden areas:
Risks to watch:
Validation expectation:
```

### 2. Codex 先 review，不直接执行
```text
What looks right:
Main risk:
Smaller path if any:
What I would change before execution:
Go / revise:
```

### 3. Codex 再执行或给出最小落地建议
```text
What I changed or propose to change:
Validation performed or needed:
Remaining uncertainty:
If I challenge my own result:
```

### 4. Claude 主审收口
```text
Accepted:
Rejected:
Risk check result:
Decision:
Next step:
```

---

## 默认规则
- 不要求每轮都拆成多份长模板
- 不要求 medium-risk 任务默认进入二审
- 只要任务没有升级为 high-risk，就优先使用这张短版任务卡
- 如果试跑发现短版某字段长期没人用，就继续删

---

## 一句话版本
**默认只用一张短版任务卡跑完：Claude 出题，Codex 质疑，Codex 执行/建议，Claude 收口；只有高风险任务才加挂附加清单。**
