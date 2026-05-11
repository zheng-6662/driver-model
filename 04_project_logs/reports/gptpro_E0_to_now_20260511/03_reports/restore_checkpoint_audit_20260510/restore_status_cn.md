# 旧实验脚本与检查点恢复记录

更新时间：2026-05-10

## 当前结论

- 本地 `fair_vehicle_event_comparison_20260427` 实验脚本目录曾被清空，已从服务器恢复旧脚本。
- 本地历史运行目录多数还在，指标和预测图多数还在，但 `best_model.pt` 基本缺失。
- 服务器上保留了部分旧检查点，已拉回本地：
  - E10A seed2027/2028；
  - E10B seed2027/2028；
  - E10C seed2027/2028。
- 仍需服务器补跑的关键基础版本：
  - E2 seed2026/2027/2028；
  - E4 seed2026/2027/2028；
  - E7C seed2026/2027/2028；
  - E10C seed2026；
  - E5A seed2026/2027/2028；
  - E6 seed2026/2027/2028；
  - E11A seed2026。

## 服务器信息

- 连接命令：`ssh -p 24309 root@connect.westd.seetacloud.com`
- 注意：密码不写入任何日志、报告或代码。
- 远程项目目录：`/root/autodl-tmp/data_process`
- 远程 Python：`/root/miniconda3/bin/python`
- 远程显卡：`NVIDIA GeForce RTX 4080`，显存约 32GB。
- 数据 manifest 已确认存在：
  `/root/autodl-tmp/data_process/02_code/final_code/model/training/protocol_allphase_control_v2_context_full2s/sample_manifest.csv`

## 已启动的恢复队列

- 队列脚本：
  `/root/autodl-tmp/data_process/04_project_logs/reports/restore_checkpoint_audit_20260510/restore_core_queue_20260510.sh`
- 主日志：
  `/root/autodl-tmp/data_process/04_project_logs/reports/restore_checkpoint_audit_20260510/restore_core_queue_master.log`
- 子任务日志目录：
  `/root/autodl-tmp/data_process/04_project_logs/reports/restore_checkpoint_audit_20260510/remote_restore_logs`
- 运行记录目录：
  `/root/autodl-tmp/data_process/04_project_logs/reports/restore_checkpoint_audit_20260510/remote_restore_records`
- 队列进程号：`2751`
- 并行数：`5`

## 执行顺序

第一阶段补不依赖脑电教师的版本：

- E2 seed2026/2027/2028；
- E4 seed2026/2027/2028；
- E7C seed2026/2027/2028；
- E10C seed2026。

第二阶段等 E4 教师检查点生成后补蒸馏版本：

- E5A seed2026/2027/2028；
- E6 seed2026/2027/2028；
- E11A seed2026。

## 和 G13 的关系

G13 暂停继续执行。先恢复旧脚本和关键旧版本检查点，恢复完成后再继续 G13 的响应类型、物理约束和生理选择性融合实验。

## 恢复完成情况

服务器恢复队列已完成，时间范围约为 2026-05-10 17:27 到 18:38。

已拉回本地的正式恢复版本共 17 个：

| 版本 | seed | test RMSE | tail RMSE | selection |
| --- | ---: | ---: | ---: | ---: |
| E2 | 2026 | 0.4536 | 0.4047 | 0.8707 |
| E2 | 2027 | 0.4614 | 0.3629 | 0.8335 |
| E2 | 2028 | 0.4548 | 0.3768 | 0.8232 |
| E4 | 2026 | 0.4462 | 0.3444 | 0.8028 |
| E4 | 2027 | 0.4630 | 0.3886 | 0.8427 |
| E4 | 2028 | 0.4540 | 0.3804 | 0.8426 |
| E5A | 2026 | 0.4552 | 0.3550 | 0.8227 |
| E5A | 2027 | 0.4510 | 0.3653 | 0.8258 |
| E5A | 2028 | 0.4549 | 0.3828 | 0.8295 |
| E6 | 2026 | 0.4479 | 0.3505 | 0.8280 |
| E6 | 2027 | 0.4493 | 0.3597 | 0.8362 |
| E6 | 2028 | 0.4472 | 0.3840 | 0.8536 |
| E7C | 2026 | 0.4572 | 0.3940 | 0.8519 |
| E7C | 2027 | 0.4656 | 0.4196 | 0.8688 |
| E7C | 2028 | 0.4872 | 0.3775 | 0.8263 |
| E10C | 2026 | 0.4537 | 0.3645 | 0.8298 |
| E11A | 2026 | 0.4648 | 0.3831 | 0.8423 |

本地索引：

- `restored_run_index_20260510.csv`

本地目前 `tmp/event_conditioned_runs` 下已经能找到 23 个 `best_model.pt`，包括服务器原先保留并已拉回的 E10A/E10B/E10C 部分检查点，以及本轮新补的关键恢复版本。

注意：这些恢复检查点是按同一协议和参数重新训练得到的，不是原本丢失的旧 `best_model.pt` 原文件。它们用于恢复可复现训练环境和后续蒸馏/对照，不应和原始结果表混为同一个 run。
