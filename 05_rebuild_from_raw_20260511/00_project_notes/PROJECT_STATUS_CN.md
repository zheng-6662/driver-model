# R2E-Steering 项目总进度看板

更新时间：2026-05-12 11:29:40

## 当前阶段

阶段 2：事件锚点与样本清单重建已生成 v0.2 候选 manifest；低泄漏道路曲率候选的车辆处理窗口 v0.2 已生成。下一步进入阶段 3 的无学习基线和强车辆基线准备。

## 当前正在做什么

核验阶段 2 manifest 和处理后车辆窗口，准备阶段 3 只基于低泄漏道路曲率候选先做无学习/车辆基线。

## 已完成什么

- 阶段 0 旧流程冻结说明已生成。
- 新流程目录结构已建立。
- 三个原始目录下被试名文件夹内 CSV 清单和哈希已生成。
- 原始车辆/生理/脑电深度审计表已生成。
- 阶段 1 用户查看版中文总结已生成。
- 阶段 0/1 完成审计清单已生成。
- 阶段 2 候选事件清单、样本清单、split 表、道路设计清单和数据版本卡已生成。
- 低泄漏道路曲率候选的处理后车辆窗口 v0.2 已生成，原始 CSV 未修改。

## 正在运行什么任务

当前没有后台审计或训练任务在运行。

## 服务器是否在运行

本阶段 2 和处理后车辆窗口生成均未使用服务器；未读取服务器密码文件。服务器状态未主动检查。

## 最近一次结果

- 阶段 1 纳入审计 CSV：258；车辆/生理/脑电：91/82/85
- 候选事件：11619，其中 old v400 6247、raw road curvature 359、raw vehicle dynamic 5013
- `samples_master.csv/jsonl` 行数：46476
- 道路设计记录：49 个文件，其中多个 CSV 含 curvature/kappa 信息
- 低泄漏道路曲率候选处理后车辆窗口：3 个 NPZ，样本数均为 359，特征数 9
- 处理窗口形状：pre1 输入 `(359,201,9)` 标签 `(359,401)`；pre2 输入 `(359,401,9)` 标签 `(359,401)`；pre3 输入 `(359,601,9)` 标签 `(359,601)`

## 当前最大风险

old v400 仍只能作历史参考；raw vehicle dynamic 锚点来自车辆响应，存在响应结果泄漏风险；raw road curvature 候选较低泄漏但只覆盖道路曲率事件，不能代表全部事件类型。后续阶段 3 只能先从该低泄漏子集做基线。

## 下一步准备做什么

1. 对处理后车辆窗口 v0.2 做无学习基线：零响应、保持当前值、历史趋势外推、同类事件平均轨迹。
2. 建立固定预测图和坏样本图规则。
3. 只在 train split 上拟合任何统计量，验证 split 表是否满足阶段 3 需求。
4. 继续审查道路设计文件能否进一步提供更精确、低泄漏的道路事件锚点。

## 用户可以优先查看哪些文件

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_user_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_v0_2_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/processed_vehicle_windows_v0_2_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/tables/processed_vehicle_window_outputs.csv`
