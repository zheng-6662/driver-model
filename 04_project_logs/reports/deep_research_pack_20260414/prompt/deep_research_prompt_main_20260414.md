# GPT 深度研究提示词（driver response modeling 定向版）

你是一名擅长机器学习建模研究、驾驶行为建模、时序预测、实验协议分析的高级研究助手。请围绕下面这个具体项目问题做深度研究，不要泛泛而谈。

## 项目背景
我在做一个**极限工况下驾驶员反应建模**项目。当前主线任务不是通用自动驾驶轨迹规划，而是：

- 以**事件级样本**为单位；
- 输入为**过去 3 秒 online-visible 的车辆/控制历史**；
- 预测**未来 2 秒驾驶员/车辆响应轨迹**；
- 当前核心输出通道包括：
  - steering
  - yaw rate
  - lateral acceleration
- 当前协议要求：
  - **subject-level fixed split**，不能改成 event-level random split；
  - **online inputs only**，不能引入部署时不可见的未来信息；
  - **full future 2s only**，保证未来窗定义一致；
  - 当前比较非常强调**公平性、可比性、避免数据泄漏**。

## 当前模型设定
当前主模型大致是一个**非自回归 Transformer encoder-decoder**：

- encoder 编码过去 3 秒多变量车辆历史；
- decoder 生成未来 2 秒多步响应轨迹；
- 额外加入了以下机制：
  1. **train-time privileged information / teacher-student distillation**  
     利用生理和 EEG 派生出的 teacher latent state，只在训练时蒸馏给 vehicle-only student，推理时仍只用 vehicle history。
  2. **reversal auxiliary head**  
     专门判断未来 steering 是否出现明显符号翻转/反打。
  3. **trajectory-shape related losses**  
     包括一阶/二阶差分约束、steering amplitude / peak-range 约束等。
  4. **hard sample emphasis**  
     对强反打样本给予更高 sample weight。
  5. **event anchor adaptation**  
     对弯道类事件与直道/紧急变道类事件使用不同 anchor 规则，以减轻多次反打样本的相位不对齐。

## 当前实际困难
目前最关心的不是普通样本的平均 RMSE，而是下面这些**复杂响应事件**：

- 多次反打 / steering sign reversal
- 峰值时序偏移
- 后段结构恢复差
- 预测趋势大体对，但细节不像真实驾驶员响应
- 提升困难样本后，普通样本可能退化

因此我希望你研究的核心问题是：

## 核心研究问题
**在仅使用过去 3 秒 online-visible vehicle/control history、保持 subject-level fixed split 和公平协议不变的前提下，如何提升未来 2 秒驾驶员响应轨迹预测中复杂反打、符号翻转、峰值时序和后段细节结构的建模能力，同时尽量不让普通样本明显退化？**

---

## 你的研究任务
请不要只给“可以试试更大模型/更多数据/更长训练”这类泛建议。请系统研究并输出以下内容：

### 1. 相关方法综述
请重点寻找并总结与以下方向最相关的方法，而不是泛泛的 time-series forecasting：
- event-level driver response modeling
- short-horizon driver control prediction
- maneuver-conditioned forecasting
- sign-aware / reversal-aware trajectory modeling
- peak/phase-aware sequence prediction
- privileged information distillation for forecasting
- rare / tail maneuver robustness

### 2. 方法分类
请把候选改进方案按类别整理，例如：
- 模型结构改进
- 损失函数改进
- teacher-student / privileged information 改进
- 难样本学习与 curriculum
- 分层建模 / mixture / condition routing
- 评估指标改进

### 3. 对我这个项目的适配性判断
对每类方法，请明确判断：
- 为什么它可能适合我的任务；
- 它更可能改善：
  - 反打识别？
  - 符号翻转？
  - 峰值幅值？
  - 峰值时间？
  - 后段结构？
  - rare/tail cases？
- 它是否与“vehicle-only inference, multimodal teacher only at training time”的设定兼容；
- 它是否可能破坏公平比较；
- 它是否可能引入 future leakage 或 hidden protocol drift；
- 它实现成本是低 / 中 / 高；
- 它更适合先做小型 ablation，还是值得直接做主线升级。

### 4. 给出优先级排序
请最后按“**预期收益 × 实现可控性 × 与当前架构兼容性 × 公平比较风险**”综合排序，给出：
- Top 5 最值得优先尝试的方案；
- Top 3 不建议优先投入的方案；
- 每个方案建议的最小可验证 ablation。

### 5. 给出实验设计建议
请针对每个高优先级方案，建议：
- 应保持不变的协议项；
- 需要额外记录的分层指标；
- 应重点观察的 failure cases；
- 如何避免因为 rare-event 提升而 common-case 退化。

### 6. 给出评估指标建议
除了 RMSE/MAE，请研究并建议更适合我的问题的指标，例如：
- sign consistency / zero-crossing accuracy
- peak magnitude error
- peak timing error
- reversal event detection metrics
- event-shape fidelity metrics
- tail-case stratified metrics

并说明这些指标为什么对“驾驶员响应细节恢复”更有意义。

---

## 强约束 / 红线
请你所有建议都必须遵守以下前提：
1. **不能破坏 subject-level fixed split**。
2. **不能使用部署时不可见的 future information**。
3. **不能偷偷改变 target、anchor、future horizon 定义却假装可公平比较**。
4. **不要把问题研究成通用 autonomous driving planning**，重点应是 driver response forecasting。
5. **不要只给调参建议**，要优先给机制性建模建议。
6. **不要默认 inference 时可以使用生理/EEG**；这些模态更适合作为 train-time privileged information。
7. 请优先考虑**短时 2 秒、事件级、复杂反打和相位细节**这个具体问题，而不是泛化到任何时序任务。

---

## 希望的输出格式
请按下面结构输出：

1. **一句话结论**
2. **最相关的研究方向总览表**
3. **候选方案分组表**
4. **每个方案的适配性分析**
5. **Top 5 优先尝试方案**
6. **Top 3 暂不建议优先做的方案**
7. **建议增加的评估指标**
8. **推荐的最小实验矩阵**
9. **如果我是当前代码基础上迭代，最合理的下一步路线图**

请尽量具体、结构化、面向实际建模决策，而不是论文摘要堆砌。

---

## 当前实现摘要补充
当前代码已经包含以下具体设计，请在此基础上判断“哪些改动是增量升级，哪些是范式切换”：

- 输入历史窗：3.0s，200Hz
- 未来窗：2.0s，200Hz
- subject-level fixed split
- future full-2s only
- online-visible inputs only
- Transformer encoder-decoder, non-AR
- train-time teacher-state distillation（physio + EEG derived latent）
- reversal auxiliary head
- derivative-based multi-scale loss
- steering amplitude loss
- reversal sample weighting
- curve vs straight stratified analysis
- weak reversal vs strong reversal stratified analysis

请特别回答：
1. 哪些方案最适合在现有代码上低风险增量实现？
2. 哪些方案虽然理论强，但会明显破坏现有可比性？
3. 哪些方案最可能真正改善“复杂反打细节”，而不是只改善平均 RMSE？
