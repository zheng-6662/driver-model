# 阶段 0/1 完成审计

审计时间：2026-05-12 10:55:23

## 本轮目标

本轮只完成阶段 0 和阶段 1 的前置工作，不训练模型，不声明连续风格、生理、脑电教师或旧模型上限是否有效。

具体交付标准：

1. 新流程目录和透明化文件存在，并能让用户看到进度、任务、产物、服务器状态和日志。
2. 阶段 0 明确冻结旧流程，把旧流程降级为参考、对照和风险来源。
3. 阶段 1 只审计用户确认的原始数据范围：`原始车辆数据/<被试名>/*.csv`、`原始生理数据/<被试名>/*.csv`、`原始脑电数据/<被试名>/*.csv`。
4. 输出文件清单、哈希、字段、时间戳、采样率、模态完整性、模态重叠、信号质量、EEG 初审和泄漏风险表。
5. 输出中文报告、用户查看版总结、图表、运行日志和产物索引。
6. 服务器密码文件不读取、不记录；本阶段无服务器任务。
7. 关键产物提交到 Git，但不提交服务器密码文件。

## Prompt-to-artifact 核验清单

| 要求 | 核验证据 | 状态 |
|---|---|---|
| 建立新流程目录 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511` | 已完成 |
| 初始化总进度看板 | `00_project_notes/PROJECT_STATUS_CN.md` | 已完成 |
| 初始化每日执行日志 | `00_project_notes/daily_logs/2026-05-12.md` | 已完成 |
| 初始化阶段产物索引 | `00_project_notes/ARTIFACT_INDEX_CN.md` | 已完成 |
| 初始化当前任务队列 | `00_project_notes/TASK_QUEUE_CN.md` | 已完成 |
| 初始化服务器运行记录 | `00_project_notes/SERVER_RUNS_CN.md` | 已完成 |
| 冻结旧流程和重建准则 | `00_project_notes/stage00_old_flow_freeze_and_rules_cn.md` | 已完成 |
| 阶段 0 用户查看版总结 | `09_reports/stage00_user_summary_cn.md` | 已完成 |
| 只扫描三个原始目录 | `01_audit/scripts/raw_csv_audit.py` 中 `RAW_SCOPE_TOP_DIRS` | 已完成 |
| 只扫描被试名文件夹下 CSV | `raw_file_inventory.csv` 中 258 条路径深度均为 `原始目录/被试名/文件.csv` | 已完成 |
| 文件清单和哈希 | `01_audit/tables/raw_file_inventory.csv`，258 行 | 已完成 |
| 字段和行数报告 | `01_audit/tables/raw_schema_report.csv`，258 行 | 已完成 |
| 时间戳连续性报告 | `01_audit/tables/timestamp_continuity_report.csv`，258 行 | 已完成 |
| 采样率报告 | `01_audit/tables/sampling_rate_report.csv`，3 行 | 已完成 |
| 被试/记录/模态矩阵 | `01_audit/tables/subject_session_modality_matrix.csv`，91 行 | 已完成 |
| 模态重叠报告 | `01_audit/tables/modality_overlap_report.csv`，91 行 | 已完成 |
| 生理/车辆信号质量报告 | `01_audit/tables/signal_quality_report.csv`，4031 行 | 已完成 |
| EEG 初审报告 | `01_audit/tables/eeg_artifact_report.csv`，2975 行 | 已完成 |
| 泄漏风险报告 | `01_audit/tables/leakage_risk_report.csv`，7 行 | 已完成 |
| 审计图 | `01_audit/figures/audit` 下 8 张 PNG | 已完成 |
| 阶段 1 中文总结 | `09_reports/raw_data_audit_summary_cn.md` | 已完成 |
| 阶段 1 用户查看版总结 | `09_reports/stage01_user_summary_cn.md` | 已完成 |
| 运行日志无错误输出 | `01_audit/logs/*.stderr.log` 均为空 | 已完成 |
| 不读取/不记录服务器密码 | `SERVER_RUNS_CN.md` 仅写 SSH 命令格式；本阶段未使用服务器 | 已完成 |
| Git 提交 | `e9d302f Add raw rebuild stage 0 and 1 audit` | 已完成 |

## 实际核验结果

- 纳入审计 CSV：258。
- 路径深度：258 条全部为三段结构，符合 `原始目录/被试名/文件.csv`。
- 目录分布：车辆 91、生理 82、脑电 85。
- `raw_scope=True`：258；`derived_file=False`：258。
- 时间解析：车辆、生理、脑电均解析为 datetime。
- 负时间间隔：0。
- 存在零间隔时间戳的文件：151。该现象需要在阶段 2/5 结合原始采样写入策略复核，不能简单等同于坏数据，但必须作为时间解析率和重复点风险记录。
- 存在 large gap 的文件：26。
- 零间隔或 large gap 任一风险命中的文件：175。
- 被试/记录组合：91。
- 三模态齐全组合：76。
- 模态重叠状态：91 条均为 `overlap_ok`。
- 信号有效率低于 95% 的信号/文件组合：3847。
- EEG 近常数通道/文件组合：27。
- 审计图数量：8。
- 所有 `.stderr.log` 文件为空。

## 发现并修正的缺口

1. 产物索引中“重要 Git commit”仍写为“待提交”，已改为 `e9d302f Add raw rebuild stage 0 and 1 audit`。
2. 产物索引缺少 `stage00_user_summary_cn.md`，已补入。
3. 阶段 1 报告原先只强调 26 个 large gap 文件，完成审计发现还应明确 151 个 zero-dt 文件和共 175 个时间复核候选，已补入报告和用户总结。
4. 完成核验时，第一次辅助核验脚本使用了不存在的列名，第二次也使用了旧列名；已改用实际表头重跑并在本文件记录，不影响审计产物本身。

## 结论

阶段 0 和阶段 1 的交付标准已经达到：旧流程已冻结为参考，新流程透明化文件已建立，原始数据审计严格限定到用户确认的三个原始目录和被试名文件夹内 CSV，关键表格、图、中文总结、日志和 Git 提交均已完成。

当前只能进入阶段 2：事件锚点与样本清单重建。仍不能直接进入模型训练，也不能宣称连续风格、生理、EMG、EEG 教师或任何新模型路线有效。
