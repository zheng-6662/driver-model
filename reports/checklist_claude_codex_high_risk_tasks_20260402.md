# 模板 4：高风险任务检查清单

## 适用范围
以下任务默认视为高风险：
- protocol_config.json 相关改动
- split 相关改动
- 标签定义改动
- future horizon 改动
- event anchor 改动
- sampling / alignment 改动
- 训练主路径改动
- 实验比较口径改动

---

## A. Active Code 路径检查
- [ ] 是否确认修改的是 `datasetprocess/final_code` 下的 maintained code？
- [ ] 是否确认没有误改历史归档、run 输出副本、tmp、backup？
- [ ] 是否确认没有在实验结果目录里修改复制脚本？

## B. Protocol / Split 检查
- [ ] 是否触及 `protocol_config.json`？
- [ ] 是否改变 subject-level split？
- [ ] 是否改变 train/val/test 边界？
- [ ] 是否改变事件筛选逻辑或样本归属逻辑？

## C. Label / Horizon / Anchor 检查
- [ ] 是否改变未来预测时长？
- [ ] 是否改变目标变量定义？
- [ ] 是否改变 event anchor？
- [ ] 是否改变 primary / response-aligned / full_future_2s_only 等标签语义？

## D. Leakage 检查
- [ ] 是否把未来信息泄漏进输入？
- [ ] 是否把测试主体信息通过路径、缓存、标签泄漏进训练？
- [ ] 是否引入 time leakage？
- [ ] 是否引入 label leakage？

## E. Fairness / Comparability 检查
- [ ] 修改后与历史 run 是否还能公平比较？
- [ ] 如果不能，是否已明确说明“不可直接对比”？
- [ ] 是否把 protocol 变化与 optimization 变化混在一起？

## F. Validation Evidence 检查
- [ ] 是否至少有一轮最小验证？
- [ ] 是否验证了关键路径而不是只验证语法？
- [ ] 是否保留了足够证据让另一方复核？

## G. 双审要求
- [ ] Claude 是否已完成 plan 审查？
- [ ] Codex 是否已完成 pre-execution challenge？
- [ ] Codex 是否已回传执行结果与不确定项？
- [ ] Claude 是否已完成 post-execution review？
- [ ] 如存在争议，是否进行了 Codex 二审？

---

## 高风险任务最终决策模板

```text
Task:
Risk level: high

Active code check:
Protocol/split check:
Label/horizon/anchor check:
Leakage check:
Fairness check:
Validation evidence:

Claude review verdict:
Codex review verdict:

Final decision:
- proceed
- proceed with caveat
- revise
- stop
```
