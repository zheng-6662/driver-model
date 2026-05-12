# R2E-Steering 项目总进度看板

更新时间：2026-05-12 10:55:23

## 当前阶段

阶段 1：原始数据审计与阶段 0/1 完成审计已完成；下一步进入阶段 2 的事件锚点与样本清单重建。

## 当前正在做什么

收尾核验阶段 0/1 产物是否满足长期目标要求，并把完成审计写入项目记录。

## 已完成什么

- 阶段 0 旧流程冻结说明已生成。
- 新流程目录结构已建立。
- 三个原始目录下被试名文件夹内 CSV 清单和哈希已生成。
- 原始车辆/生理/脑电深度审计表已生成。
- 阶段 1 用户查看版中文总结已生成。
- 阶段 0/1 完成审计清单已生成。

## 正在运行什么任务

当前没有后台审计或训练任务在运行。

## 服务器是否在运行

本阶段未使用服务器；未读取服务器密码文件。服务器状态未主动检查。

## 最近一次结果

- 本次纳入审计 CSV：258
- 原始范围 CSV：258
- 原始车辆/生理/脑电：91/82/85
- 三模态齐全组合：76
- overlap>0 组合：91
- zero-dt 文件：151；large gap 文件：26；二者合并后时间复核候选：175

## 当前最大风险

旧事件锚点和旧窗口定义不能直接继承；EMG 可能存在事件后动作结果泄漏；151 个 zero-dt 文件和 26 个 large gap 文件需要在阶段 2/5 逐样本确认。

## 下一步准备做什么

1. 读取原始车辆事件线索和旧事件文件，重建候选事件锚点。
2. 生成 `samples_master.csv/jsonl` 的第一版字段设计。
3. 明确输入窗口、标签窗口和 causal setting。
4. 生成 split_table 和 dataset_version_card。

## 用户可以优先查看哪些文件

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage01_user_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/raw_data_audit_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/stage00_old_flow_freeze_and_rules_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/leakage_risk_report.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/subject_session_modality_matrix.csv`
