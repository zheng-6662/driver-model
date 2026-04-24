# 开题报告与当前小论文方向的对齐摘要

## 1. 开题报告原始主题

开题报告原始题目是：

> 基于多模态数据驱动的极端工况驾驶员模型研究

它的核心关切不是单纯做车辆轨迹预测，而是：

- 在极端工况下理解和建模驾驶员应激操控行为
- 将车辆动态、道路几何、驾驶风格以及生理和脑电信息联系起来
- 为人机协同或辅助驾驶提供更符合真实驾驶员行为的建模基础

## 2. 开题报告中最适合直接继承到小论文里的内容

### 2.1 背景动机

开题报告第一章已经给出很好的大背景：

- 极端工况下驾驶员应激操控是安全关键问题
- L2 及较长时期的人机协同场景下，驾驶员行为不可忽视
- 仅从车辆控制角度难以解释风险链条
- 生理和脑电有助于理解内在状态，但工程上难以长期稳定在线部署

这直接支持当前最稳的论文叙事：

- 训练阶段引入多模态人因信息
- 推理阶段尽量保持车辆侧可部署

### 2.2 原始技术路线

开题报告第三章和第五章里，最重要的技术路线有四步：

1. 多模态数据采集与对齐
2. 驾驶风格与驾驶状态识别
3. 基于多模态信息的驾驶员模型构建
4. 预测验证与模型评估

这四步非常适合直接映射成小论文的方法章节顺序，只需要把当前证据最充分的切片放在中心。

### 2.3 原始创新点

开题报告里最值得保留的创新点是：

- 驾驶风格与驾驶状态的联合建模机制
- 极端工况的多模态驾驶员行为建模

这两个创新点与当前 repo 主线高度一致，比“模型全局最优”更适合写成论文卖点。

## 3. 当前实现与开题报告的最佳对接方式

### 3.1 可以直接继承的部分

- 风格建模
  - 当前代码通过 `style_id` 将驾驶风格先验显式注入上下文
- 驾驶员状态
  - 当前代码通过 physio / EEG 特征构造 teacher-state supervision
- 多模态
  - 当前已经具备 vehicle + road preview + style prior + teacher-side human-state signal
- Transformer
  - 开题报告提到的序列建模主架构与当前主线一致
- 多指标评估
  - 当前项目已经形成了比开题报告更成熟的结构化评价体系

### 3.2 需要收缩后再写的部分

开题报告整体目标更大，但当前最成熟的实现切片更聚焦，因此小论文中建议做以下收缩：

- 不把任务写成“完整极端工况驾驶员全状态生成”
- 不把任务写成“实时生理脑电驱动的在线模型”
- 不把任务写成“全场景统一最优模型”

而应收束为：

- pooled post-trigger steering-response prediction
- style-aware and teacher-guided driver modeling
- response-fidelity-aware evaluation

## 4. 开题报告最有价值的一句思想

开题报告第一章实际上已经给了这篇小论文最重要的立论基础：

> 生理和脑电信号有助于学习驾驶员状态，但由于工程部署限制，更合理的思路是在训练阶段引入多模态信息，在应用阶段仍主要依赖可观测车辆信号。

这句话几乎就是当前小论文方法部分的核心思想，应当作为方法动机保留。

## 5. 开题报告与当前代码的差异提醒

为了避免 GPT Pro 直接照搬开题报告，需要特别注意：

- 开题报告是面向整个课题的总设计，不等于当前主线已经全部完成。
- 当前代码最强的是 steering-response line，不是整个宏观课题都已经闭环。
- 当前代码里的 physio/EEG 最安全表述是 teacher-state / distillation，而不是 deployed multimodal inference.
- 当前最成熟的实验结论强调的是：
  - 结构化建模有价值
  - response-fidelity evaluation 有必要
  - not every lower-RMSE branch is scientifically better

## 6. 从开题报告继承到小论文的推荐结构

### 6.1 Introduction

- 极端工况安全问题
- 人机协同场景下驾驶员建模必要性
- 现有方法多偏车辆和道路，忽视个体差异与状态
- 生理/EEG 的价值与部署矛盾
- 因此提出训练期多模态教师监督 + 推理期车辆侧预测的框架

### 6.2 Data

- 在环平台
- 多模态同步
- 事件级样本
- 风格和状态表征来源

### 6.3 Model

- vehicle history encoder
- style prior
- road preview
- teacher-state supervision
- response-structure-aware objectives

### 6.4 Experiments

- main comparisons
- ablations
- response-fidelity metrics

### 6.5 Discussion

- dual-keeper story
- why RMSE-only is insufficient
- what style prior and teacher-state likely contribute

## 7. 对 GPT Pro 最有用的提醒

如果 GPT Pro 要把这个项目整理成论文，不要直接照着开题报告写成“大而全”的总论文，而应该：

- 以开题报告为研究动机来源
- 以当前 repo 中最成熟的 steering-response 主线为论文主体
- 以风格条件、teacher-state supervision 和 response-fidelity evaluation 为最核心贡献
