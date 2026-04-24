# 小论文上下文摘要

## 1. 当前最适合的小论文定位

当前最稳的小论文定位不是“我们已经得到了一个全面最优的新模型”，而是：

- 面向极端工况驾驶员转向响应预测的多模态驾驶员建模框架
- 将驾驶风格作为显式条件先验引入预测模型
- 在训练阶段利用生理和 EEG 信息构造教师状态，对车辆侧学生模型进行指导
- 通过响应结构保真指标而不是 RMSE 单指标来评价模型

一句话版本可以写成：

> 本研究面向极端工况下的驾驶员转向响应预测，提出一种融合驾驶风格先验与训练期生理教师监督的多模态时序建模框架，并建立了面向响应时序、反打结构和尾段行为的评价协议。

## 2. 开题报告与当前实现的对应关系

### 2.1 高度一致的部分

- 题目仍然符合“基于多模态数据驱动的极端工况驾驶员模型研究”这一总方向。
- 开题报告强调的“驾驶风格与驾驶状态联合建模机制”仍然是当前最有价值的创新叙事。
- 开题报告强调“训练阶段引入生理和脑电信息，应用阶段尽量依赖可观测车辆信号”的思路，与当前主线实现高度一致。
- 开题报告强调极端工况下不能只看单一误差指标，这与当前项目里建立的 response-fidelity 指标体系一致。

### 2.2 当前实现相对开题报告的实际收缩

当前代码线最安全的表述是：

- pooled post-trigger steering-response prediction
- 输入主要包括：
  - 触发前车辆动态历史
  - 触发时上下文
  - 驾驶风格先验 `style_id`
  - 已知未来道路几何预览
- 训练阶段再额外引入：
  - physio window means
  - EEG event features
  - teacher-state distillation

当前实现尚不适合直接写成：

- 全场景全任务驾驶员建模
- 在线实时依赖生理和脑电输入
- 单一模型已经全面优于全部对照

### 2.3 当前活跃实现与开题报告在输出上的差异

开题报告中较宏观的设想包含更广的行为和车辆状态输出。
当前 repo 里最稳定、证据最充分的实现切片更聚焦于：

- steer 作为主任务
- yawrate 和 ay 作为辅助任务
- response structure 作为核心评估对象

这意味着小论文应优先围绕“极端工况下转向响应行为预测”来写，而不是把题目扩展到更大的全状态预测叙事。

## 3. 当前证据链

### 3.1 边界与输入证据

- feature-input audit 证明旧 lane 读取存在实质问题，`zx1|lateraldistance` 才是主列。
- speed-unit audit 证明速度单位问题被系统性检查过。
- trigger-to-onset lag analysis 证明任务更适合表述为 post-trigger response prediction，而不是瞬时反应建模。
- INPUT_ROLE_AUDIT 已经明确了：
  - 风格是前向条件
  - physio/EEG 是训练期 teacher-side 信号
  - future road preview 必须写成 conditional preview

### 3.2 keeper 与 no-go 证据

当前形成了比较清晰的 dual-keeper 结构：

- Run A:
  - 响应结构 keeper
  - reversal timing / reversal count 这类指标更强
- `baseline_fixed_input`:
  - fit / tail keeper
  - 是当前 best non-collapse base

后续多条分支提供了“为何不能只看 RMSE”的直接证据：

- `bridge_50_50`
  - fit / tail 数值更好
  - 但 strong-pos tail collapse 阻止晋升
- `H15`
  - overall RMSE 与绝对 tail 明显更好
  - 但 hard-collapse on `strong_pos`
- `H10`
  - ceiling 更高
  - 但只能做 diagnostic evidence，不能直接 promoted
- `OPT_A_20`
  - 说明问题不只是简单优化器没调好
- `OPT_C_BEST` / `CAP_192_BEST` / `WINNER_CONFIRM`
  - 说明轻量 regularization / width / repeat 并没有给出新的统一赢家

## 4. 为什么“不是全局最优”也仍然能写

当前小论文依然成立，原因不是“结果已经足够完美”，而是：

- 模型结构有明确研究价值：
  - 风格条件化
  - 训练期多模态教师监督
  - 事件级极端工况建模
- 评价体系有明确方法价值：
  - 不能只看 RMSE
  - 需要关注 response onset、late peak、first reversal、tail amplitude、tail flatness
- 结果提供了实证发现：
  - raw fit improvement 与 response-structure fidelity 不是同一件事
  - 一些更低误差分支会在 strong-pos tail 上塌缩
- 工程边界是清楚的：
  - 推理时并不要求在线 EEG / physio
  - 这更接近实际部署

换句话说，这篇小论文最强的卖点不是“我们全面赢了”，而是：

> 在极端工况驾驶员响应建模中，如何把个体差异和人因信息正确引入模型，并用更合理的指标判断模型是否真的学到了驾驶员响应结构。

## 5. 当前最推荐的章节结构

### 5.1 Introduction

- 研究场景：
  - 极端工况
  - L2 辅助驾驶仍需考虑驾驶员应激操控
- 文献缺口：
  - 许多驾驶行为预测方法主要依赖车辆和道路信息
  - 缺少对驾驶风格与即时驾驶员状态的联合建模
  - 缺少 response-fidelity 评价而非仅 RMSE 评价
- 核心思想：
  - 风格先验 + 生理教师监督 + 响应结构评价
- 贡献点：
  - 风格条件化极端工况驾驶员响应建模
  - 训练期多模态 teacher-state supervision
  - 面向响应结构的评估协议

### 5.2 Data Collection And Preprocessing

- 驾驶员在环仿真平台
- 极端事件定义与触发锚点
- 多模态同步与预处理
- 车辆、道路、风格、生理、EEG 各自扮演的角色
- 训练期与推理期可用信息边界

### 5.3 Model

- 事件级样本构造
- 输入组成：
  - vehicle history
  - anchor context
  - style prior
  - future road preview
- Transformer 主体
- teacher-state distillation
- auxiliary targets
- checkpoint selection and response-fidelity metrics

### 5.4 Experiments And Ablations

- fixed baseline / Run A / `baseline_fixed_input`
- input ablation
- bridge matrix
- effectiveness follow-up
- 最重要的建议补充：
  - no-style
  - no-state-distill
  - vehicle-plus-road-only

### 5.5 Results And Discussion

- dual-keeper story
- why lower RMSE can still be unsafe
- strong-pos collapse interpretation
- style prior / teacher supervision 的作用与局限
- deployment-safe wording

### 5.6 Conclusion

- 总结框架价值
- 强调当前最稳结论是结构化建模与评价协议
- 诚实指出仍缺单一 unified winner

## 6. 最需要 GPT Pro 帮忙补强的内容

- 把上述技术和证据组织成一篇“像 SCI 小论文”的叙事，而不是项目日志
- 设计最小但足够有说服力的 ablation matrix
- 判断标题、摘要、贡献点和 limitation 的最佳写法
- 给出图表配置方案：
  - 方法图
  - keeper 对比表
  - 关键 case panel
  - ablation table
  - metric-definition table
