# v230 失败案例人工复核 / 论文案例证据包

- 生成时间：2026-06-23T17:07:50
- 范围：audit-only + paper-case packaging；不训练、不新预测、不调阈值、不建 gate/router/selector。
- 当前 formal lock：`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`。
- 选入 case 数：46；复制图数：85；缺图记录：13。

## 正式主结果边界

| pool | model | test n | RMSE | tail RMSE | direction acc | under rate |
|---|---|---:|---:|---:|---:|---:|
| loose_main_pool | avg_joint_focus | 184 | 0.544884 | 0.629752 | 0.967391 | 0.163043 |
| strict_main_pool | peak_floor_090 | 174 | 0.571770 | 0.658306 | 0.948276 | 0.137931 |

## case 选择分布

| pool | bucket | n |
|---|---|---:|
| loose_main_pool | 反转或多次修正 | 3 |
| loose_main_pool | 强反应低估 | 5 |
| loose_main_pool | 强响应幅值/尾段 | 5 |
| loose_main_pool | 普通曲线可控 | 3 |
| loose_main_pool | 极端峰值失败 | 4 |
| loose_main_pool | 过零/换向边界 | 3 |
| strict_main_pool | 反转或多次修正 | 3 |
| strict_main_pool | 强反应低估 | 5 |
| strict_main_pool | 强响应幅值/尾段 | 5 |
| strict_main_pool | 普通曲线可控 | 3 |
| strict_main_pool | 极端峰值失败 | 4 |
| strict_main_pool | 过零/换向边界 | 3 |

## 人工复核说明

`v230_manual_review_template.csv` 中的人工复核字段已全部留空。后续需要人工逐图填写，Codex 没有自动判断锚点是否可疑、方向是否正确、尾段是否滞后或峰值是否压平。

## 论文使用边界

- 可以写：方向和普通响应相对稳定，但困难样本中的幅值、尾段、极端峰值和反转仍是 limitation。
- 可以写：casebook 用于失败模式展示和人工复核，不是新的模型提升。
- 不可以写：v230 改进了 RMSE、训练了新模型或证明 selector/gate 已可部署。