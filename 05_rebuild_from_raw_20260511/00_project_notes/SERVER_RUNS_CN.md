# 服务器运行记录

## 最新更新：2026-05-12 15:52

- 本次“道路设定引导的车辆失稳事件自动判定 v0.1”完全在本地运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 远程任务状态：无。
- GPU/显存状态：未检查，因为本次只做本地 CSV 表格处理和规则判定。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1`。

## 最新更新：2026-05-12 16:25

- 本次“全部原始车辆 CSV 失稳样本重筛 v0.1”完全在本地运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 远程任务状态：无。
- GPU/显存状态：未检查，因为本次只做本地 CSV 读取、200Hz 插值、规则筛选和表格生成。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1`。

更新时间：2026-05-12 14:03:26

## 连接命令格式

需要连接服务器时，只能记录不含密码的 SSH 命令格式，例如：

`ssh -p <port> <user>@<host>`

禁止在本文件或任何项目文件中写入服务器密码。

## 当前状态

- 当前阶段 2 补充人工事件标注审查包和键盘式人工标注播放器未使用远程服务器。
- 未读取服务器指令与密码文件。
- 当前没有已知后台服务器任务在运行。
- GPU/显存状态：未检查，因为阶段 2 人工标注审查包、键盘播放器和阶段 3 候选诊断均在本地完成。
- 本地标注服务：`http://127.0.0.1:8766/`，PID 34408。该服务只用于本机页面和标签保存，不是远程服务器/GPU 任务。
- Codex 自动事件审阅 v0.1 在本地完成，未使用远程服务器/GPU。

## 运行记录

| 启动时间 | 关闭时间 | 运行任务 | screen/nohup 名称 | 远程项目路径 | 远程日志路径 | 本地拉回路径 | 是否还在跑 | GPU/显存摘要 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | - | 阶段 1 本地原始数据审计 | - | - | - | - | 否 | 未使用服务器 |
| - | - | 阶段 2 本地样本清单与低泄漏车辆窗口处理 | - | - | - | - | 否 | 未使用服务器 |
| - | - | 阶段 3 本地无学习基线与纯车辆 ridge 基线 | - | - | - | - | 否 | 未使用服务器 |
| - | - | 阶段 3 v0.3 本地无被试 ID 车辆基线与诊断 | - | - | - | - | 否 | 未使用服务器 |
| - | - | 阶段 3 v0.4 本地 RBF KRR 候选模型卡 | - | - | - | - | 否 | 未使用服务器 |
| - | - | 阶段 2 补充本地人工事件标注审查包 v0.1 | - | - | - | - | 否 | 未使用服务器 |
| 2026-05-12 13:00:25 | 2026-05-12 13:15:27 | 阶段 2 补充本地键盘式人工事件标注播放器 v0.1 初版整段模式 | 本地 PID 33060 | - | `02_samples/manual_event_keyboard_player_v0_1/logs` | `02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv` | 否，已重启为候选段模式 | 未使用服务器 |
| 2026-05-12 13:17:16 | 2026-05-12 13:20:17 | 阶段 2 补充本地键盘式人工事件标注播放器 v0.2 候选段审查模式初版 | 本地 PID 38204 | - | `02_samples/manual_event_keyboard_player_v0_1/logs` | `02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv` | 否，已重启以加入 N/P 记录切换 | 未使用服务器 |
| 2026-05-12 13:20:17 | 2026-05-12 13:25:51 | 阶段 2 补充本地键盘式人工事件标注播放器 v0.2 候选段审查模式 | 本地 PID 16464 | - | `02_samples/manual_event_keyboard_player_v0_1/logs` | `02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv` | 否，已重启以加入竖线图例 | 未使用服务器 |
| 2026-05-12 13:25:51 | - | 阶段 2 补充本地键盘式人工事件标注播放器 v0.2 候选段审查模式带图例 | 本地 PID 34408 | - | `02_samples/manual_event_keyboard_player_v0_1/logs` | `02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv` | 是，本地服务 | 未使用服务器 |
| 2026-05-12 14:03:26 | 2026-05-12 14:03:26 | 阶段 2 补充 Codex 自动事件审阅 v0.1 | - | - | `02_samples/codex_event_review_v0_1/logs` | `02_samples/codex_event_review_v0_1` | 否 | 未使用服务器 |
| 2026-05-12 16:53:00 | 2026-05-12 17:02:00 | 旧代码测试全原始车辆失稳高置信样本 v0.1：窗口生成、旧车辆基线诊断、旧 vehicle_direct CPU smoke | - | - | `03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/logs`; `03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/logs`; `tmp/event_conditioned_runs/OLD_SMOKE_INSTABILITY_HIGHCONF_V0_1_20260512_165950` | `03_processed_datasets/vehicle_instability_allraw_highconf_v0_1`; `03_baselines/oldcode_vehicle_baselines_on_instability_v0_1`; `tmp/event_conditioned_runs/OLD_SMOKE_INSTABILITY_HIGHCONF_V0_1_20260512_165950` | 否 | 未使用服务器；本地 CPU smoke；未读取服务器指令与密码文件 |
