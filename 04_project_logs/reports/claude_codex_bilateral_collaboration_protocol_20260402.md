# Claude + Codex 双向思考协作协议（科研仓库版）

## 1. 目标

本协议用于当前科研代码仓库中 Claude 与 Codex 的协作，不采用“Claude 思考、Codex 执行”的单向流水线，而采用：

- 双方都思考
- 双方都查错
- 双方都 review plan 和结果
- 只是主优势不同、主责不同

核心目标：

1. 减少重复思考遗漏
2. 利用双方差异做交叉验证
3. 降低高风险改动误判
4. 提高实验、代码、文献和方案推进质量

---

## 2. 基本原则

### 原则 A：双向思考，不把任何一方降格为纯执行器
- Claude 不是唯一大脑
- Codex 也不是纯手脚
- 两边都必须对任务本身提出判断、质疑和补充

### 原则 B：先审题，再执行
任何非平凡任务都不应直接进入实现，至少先经历一次：
- 任务理解
- 风险识别
- 计划质疑

### 原则 C：高风险任务必须双审
以下任务默认属于高风险：
- protocol_config.json 相关变更
- split 定义变更
- future horizon 变更
- event anchor / label 定义变更
- 训练目标或评估口径变更
- 会影响实验公平性或可比性的改动

### 原则 D：每轮都要收口
任何一轮执行后，都必须回到 Claude 做收口：
- 当前结果是什么
- 哪些已确认
- 哪些未确认
- 是否进入下一轮

### 原则 E：先写详细进度，再输出聊天压缩总结
凡是有实质推进的工作，先写入：
- `reports/project_progress_master.md`

---

## 3. 角色主优势

## Claude 更适合主导
- 仓库理解
- maintained code / historical code 边界识别
- 安全与风险审查
- split / leakage / fairness 复核
- 计划收口
- 结果解释与归因
- 将分散结果整合为下一轮任务

## Codex 更适合主导
- 边界明确的小执行块
- 局部 patch
- 小工具编写
- 批量整理
- 局部验证
- 对具体实现提出替代做法
- 对现有 plan 进行实现层质疑

注意：
这只是主优势，不代表另一方不参与思考。

---

## 4. 任务分级与协作方式

## Level 1：高风险任务（必须双审）
适用：
- 数据协议
- split
- label
- future horizon
- 训练逻辑主路径
- 实验比较口径

流程：
1. Claude 先出 plan v1
2. Codex 专门 review plan v1
3. Claude 收敛为执行版 plan
4. Codex 执行或给出替代实现
5. Claude 主审结果
6. 必要时 Codex 对 Claude 审查结论再做二审

## Level 2：中风险任务（主做 + 强 review）
适用：
- active code 局部改动
- 小分析脚本
- 小范围实验辅助工具
- 结果整理脚本

流程：
1. 一方主做
2. 另一方必须 review
3. 通过后再进入下一步

## Level 3：低风险任务（主做 + 抽检）
适用：
- 文档整理
- 文献结构化归纳
- 报告格式化
- 轻量清单整理

流程：
1. 一方主做
2. 另一方按需抽检

## Level 4：解释型任务（双结论对照）
适用：
- run 为什么变差
- 哪个实验更公平
- 某个研究方向值不值得做
- 某批文献支持什么结论

流程：
1. Claude 独立给判断
2. Codex 独立给判断
3. 比较分歧
4. 再做联合结论

---

## 5. 标准协作流程

## 阶段 A：Plan 双审
### Claude 输出
- Goal
- 当前理解
- 风险点
- 计划步骤
- 禁区
- 验证方法

### Codex review 重点
- 这个 plan 有没有更简单路径？
- 有没有遗漏的风险？
- 文件范围是否过大？
- 验证是否不足？
- 有没有更合理的最小切入点？

### Claude 收口
- 接受哪些意见
- 拒绝哪些意见
- 最终执行边界是什么

---

## 阶段 B：Execution 前质疑
在真正改代码前，Codex 必须先回答：

1. 我是否完全理解目标？
2. 哪些文件必须先读？
3. 哪些文件不该碰？
4. 当前验证方式是否足够？
5. 有没有更小、风险更低的实现路径？

如果这些问题没有回答清楚，不进入执行。

---

## 阶段 C：Execution
Codex 执行时必须受边界约束：
- 不自行扩 scope
- 不擅自改 protocol / split / label
- 不擅自改实验定义
- 不碰 archives / tmp / backup / run outputs
- 必须返回验证证据

---

## 阶段 D：Execution 后交叉 review
### Claude 主审清单
1. 是否改到了正确的 maintained code？
2. 是否误碰历史脚本、输出副本、备份目录？
3. 是否引入 split leakage / time leakage / label leakage？
4. 是否改变了协议定义、horizon、anchor、sampling 假设？
5. 验证证据是否足够？
6. 是否真的解决了原问题，而不是只让代码跑通？

### Codex 二审触发条件
以下情况建议再回到 Codex 做二审：
- Claude 认为实现有问题，但不确定是否过度保守
- 改动涉及多种可行方案，需要替代实现比较
- 验证通过但结果解释不一致
- 需要反驳式检查

---

## 6. 三个核心模板

## 模板 A：Claude → Codex Plan Review Brief
```text
Task type: [high-risk / medium-risk / low-risk / interpretation]
Goal:
Current understanding:
Files in scope:
Files forbidden:
Risks to watch:
Validation expectation:

Your job before execution:
1. Review this plan, not just execute it.
2. Point out missing risks, simpler paths, or boundary mistakes.
3. Only after critique, propose the execution approach.
```

## 模板 B：Codex → Claude Return Brief
```text
Task completed:
What I changed:
Why this implementation:
Validation performed:
What remains uncertain:
Potential risks I still see:
If I had to challenge this result, I would question:
```

## 模板 C：Claude Final Review Brief
```text
Accepted:
Rejected:
Risk check:
- active code path
- protocol/split/label/horizon
- leakage/fairness
- validation sufficiency
Decision:
- accept / revise / escalate
Next step:
```

---

## 7. 适配当前科研仓库的额外约束

### 永远先检查这些风险
- subject leakage
- time leakage
- label leakage
- split contamination
- event anchor 被偷偷改变
- future horizon 改了但没说明
- 改到了实验输出副本而不是源代码

### 默认 maintained 路径
优先从：
- `datasetprocess/final_code`
开始

### 默认禁止区
除非任务明确要求，否则不要改：
- 历史归档
- 程序运行结果
- tmp
- artifacts
- backup 目录
- run folder 内复制脚本

---

## 8. 最推荐的实际工作形态

### 形态 1：方案双审
- Claude 出研究或实现方案
- Codex 先挑错补洞
- Claude 收口

### 形态 2：实现双审
- Claude 划边界和风险
- Codex 先质疑边界
- Codex 执行
- Claude 审结果
- 必要时 Codex 二审

### 形态 3：实验分析双结论
- Claude 先分析
- Codex 独立复核
- 两边对照分歧
- 再形成联合结论

### 形态 4：文献整理双层 synthesis
- Claude 定综述框架
- Codex 批量整理
- Claude 和 Codex 各自归纳
- 再做合并版结论

---

## 9. 最小落地版本

如果不想一开始太复杂，最小版本只执行这四条：

1. Claude 先写 plan
2. Codex 不直接执行，先 review plan
3. Codex 执行后必须提交风险与不确定项
4. Claude 审查后再决定是否结束

只要做到这四条，就已经明显优于单向流水线。

---

## 10. 一句话定义这套协议

**Claude 提案，Codex 质疑；Codex 落地，Claude 审查；必要时 Codex 反审——用差异做交叉验证，而不是把任何一方当作纯执行器。**
