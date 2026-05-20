# 2026-05-20 服务器运行记录：v0.4 样本接入旧流程车辆-only

## 任务

在服务器上运行旧流程 `FAIR09 / E1` 车辆-only粗细双头模型，用 v0.4 新筛选样本作为训练样本。

## 服务器

- 连接格式：`ssh -p 55060 root@connect.westc.seetacloud.com`
- 密码：未写入任何日志、报告、代码或提交。
- 远程项目路径：`/root/autodl-tmp/data_process`
- GPU：NVIDIA GeForce RTX 4080 SUPER
- CUDA 检查：`torch 2.5.1+cu124`，`torch.cuda.is_available() = True`

## 同步内容

只同步了运行必需脚本，没有同步大数据和 checkpoint：

- `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v04_oldflow_fair09_vehicle_only.py`
- `05_rebuild_from_raw_20260511/02_samples/scripts/build_oldcode_deep_clean_vehicle_manifest_v0_1.py`
- `05_rebuild_from_raw_20260511/02_samples/scripts/build_stage2_samples.py`

## 运行步骤

准备阶段：

- 命令入口：`stage03_v04_oldflow_fair09_vehicle_only.py --prepare-only --device cuda`
- 远程日志：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v04_oldflow_fair09_prepare_latest.log`
- 结果：manifest 1422 行，旧流程 loader 保留 1410 行，丢弃 12 行。
- 划分：train=814，val=273，test=323。

训练阶段：

- screen 名称：`oldfair09`
- 命令入口：`stage03_v04_oldflow_fair09_vehicle_only.py --device cuda`
- 远程日志：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v04_oldflow_fair09_train_latest.log`
- 开始时间：2026-05-20 11:18:36
- 结束时间：2026-05-20 11:19:55
- 当前状态：已完成，screen 已退出。
- 结束后 GPU 显存：0 MiB / 32760 MiB。

## 结果

- run root：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848`
- test steer RMSE：0.540754
- primary RMSE：0.348139
- tail RMSE：0.478053
- selection：0.984084
- 最佳验证轮次：epoch 25

## 已拉回本地

- 本地压缩包：`F:/data_set_process/data_process/tmp/stage03_v04_oldflow_fair09_vehicle_only_server_20260520.tar.gz`
- 本地 run 目录：`F:/data_set_process/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848`
- 本地用户报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_v04_oldflow_fair09_vehicle_only_server_result_20260520_cn.md`
