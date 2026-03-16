# 极限工况下驾驶员反应研究

本项目旨在研究极限工况下驾驶员的反应特征，重点关注驾驶员在高风险、高负荷驾驶场景中的行为变化以及车辆、生理和脑电等多模态信号之间的关联。

项目的核心目标包括：

- 构建极限工况下的多模态驾驶数据处理流程
- 提取车辆、生理、脑电等多源信号特征
- 构建事件级多模态数据集
- 建立驾驶员反应预测与建模方法
- 为极限驾驶工况下的人机协同与驾驶行为理解提供分析基础

## 数据模态

本项目围绕以下几类数据展开：

- 车辆数据
  - 方向盘、横摆角速度、侧向加速度、车身姿态、LTR 等
- 生理数据
  - ECG、EMG、EDA、RESP 等
- 脑电数据
  - EEG 预处理、坏道处理、ICA 清理、特征提取

## 主要流程

本仓库当前保留的主流程包括：

1. 车辆数据预处理
2. 生理数据预处理
3. 脑电数据预处理
4. 事件级多模态数据集构建
5. 多模态模型训练与诊断评估

## 先看哪里

- 最终整理后的主代码：
  - [datasetprocess/final_code](F:\数据集处理\data_process\datasetprocess\final_code)
- 多模态数据工作区与历史入口：
  - [datasetprocess/多模态数据](F:\数据集处理\data_process\datasetprocess\多模态数据)

如果你只想看当前推荐使用的代码，请直接从 `datasetprocess/final_code/` 开始。

## 推荐阅读顺序

1. `datasetprocess/final_code/README.md`
2. `datasetprocess/final_code/processing/`
3. `datasetprocess/final_code/dataset/`
4. `datasetprocess/final_code/model/training/`
5. `datasetprocess/多模态数据/00_目录说明/README.md`

## 仓库结构

```text
datasetprocess/
├─ final_code/   # 最终整理后的代码
├─ 多模态数据/     # 多模态工作区、历史入口、归档结果
├─ 数据预处理/     # 本地预处理工作区
└─ 补充采集数据/   # 本地补充处理工作区
```

## 使用说明

- 本仓库主要保存源码、说明文档和项目结构。
- 原始数据、模型权重、图片、视频以及自动生成产物没有上传到 GitHub。
- 仓库中的 `final_code/` 是当前最推荐维护和继续使用的代码区域。
