# 服务器运行记录

## 2026-05-19 21:12-22:02 v0.3 样本筛选策略 GPU 快速对比

- 服务器连接命令格式：`ssh -p 55060 root@connect.westc.seetacloud.com`
- 启动时间：2026-05-19 21:12
- 关闭时间：2026-05-19 22:02，任务正常结束，screen 自动退出
- 运行任务：`stage03_v03_screening_sweep_gpu.py`
- screen/nohup 名称：`v03gpu`
- 远程项目路径：`/root/autodl-tmp/data_process`
- 远程日志路径：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_gpu_20260519_211258.log`
- 本地拉回路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_screening_sweep_gpu`
- 是否还有任务在跑：否
- GPU/显存状态摘要：服务器 GPU 为 NVIDIA GeForce RTX 4080 SUPER，PyTorch CUDA 可用；训练阶段确认有 Python 进程占用 GPU 显存。
- 结果摘要：综合排序第一为 `s16_weakpost_lat`；但该版本使用横向偏移特征，需继续复核横向偏移坐标风险。

---

# 服务器运行记录

## 2026-05-19 21:13 v0.3 样本筛选策略 GPU 快速对比

- 服务器连接命令格式：`ssh -p 55060 root@connect.westc.seetacloud.com`
- 启动时间：2026-05-19 21:12
- 关闭时间：运行中
- 运行任务：`stage03_v03_screening_sweep_gpu.py`
- screen/nohup 名称：`v03gpu`
- 远程项目路径：`/root/autodl-tmp/data_process`
- 远程日志路径：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_gpu_20260519_211258.log`
- 本地拉回路径：待任务完成后拉回到 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_screening_sweep_gpu`
- 是否还有任务在跑：是
- GPU/显存状态摘要：服务器 GPU 为 NVIDIA GeForce RTX 4080 SUPER，PyTorch CUDA 可用；该任务用于替代刚才误用 CPU 的 sklearn 筛选循环。
- 备注：旧 CPU screen `v03sweep` 已停止。GPU 快速筛选结果用于样本筛选方向判断，不直接等同于旧 sklearn 核回归模型的公平结果。

---

# 服务器运行记录

## 2026-05-13 03:42-03:46 本地 top-K top1/bestK 差距复盘

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 03:42，本地前台运行。
- 关闭时间：2026-05-13 03:46，本地脚本结束并完成检查。
- 运行任务：`stage03_vehicle_instability_topk_gap_review_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用远程服务器；本轮为本地表格/图表复盘，未训练模型；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-13 03:25-03:34 本地 top-K 车辆-only Transformer 训练与评估

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 03:25，本地前台运行。
- 关闭时间：2026-05-13 03:34，本地脚本结束并完成检查。
- 运行任务：`stage03_vehicle_instability_topk_vehicle_transformer_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用远程服务器；本机 PyTorch 使用 CUDA 训练；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-13 03:10-03:18 本地 RBF/keypoint 多候选车辆-only 复盘

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 03:10，本地前台运行。
- 关闭时间：2026-05-13 03:18，本地脚本结束并完成检查。
- 运行任务：`stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用远程服务器；本轮加载本地 keypoint checkpoint 并重建预测，未训练新模型；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-13 02:48-02:54 本地 RBF/keypoint train/val 选择器训练与评估

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 02:48，本地前台运行。
- 关闭时间：2026-05-13 02:54，本地脚本结束并完成检查。
- 运行任务：`stage03_vehicle_instability_rbf_keypoint_selector_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用远程服务器；本轮为本地轻量 sklearn 选择器；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-13 02:38-02:44 本地 keypoint vs RBF 坏样本差异复盘

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 02:38，本地前台运行。
- 关闭时间：2026-05-13 02:44，本地脚本结束并完成检查。
- 运行任务：`stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用 GPU；未使用远程服务器；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-13 02:20-02:34 本地 keypoint+residual Transformer 训练

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 02:20，本地前台运行。
- 关闭时间：2026-05-13 02:34，本地脚本结束并完成检查。
- 运行任务：`stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用远程服务器；本机 PyTorch 检测到 CUDA 并用于训练；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-13 01:58-02:09 本地 structured Transformer 训练/重跑

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 01:58，本地前台运行并重跑一次报告收口。
- 关闭时间：2026-05-13 02:09，本地脚本结束。
- 运行任务：`stage03_vehicle_instability_structured_vehicle_transformer_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用远程服务器；本机 PyTorch 检测到 CUDA 并用于训练；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-13 01:25-01:43 本地 clean-task Transformer 训练

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 01:25，本地前台运行。
- 关闭时间：2026-05-13 01:43，本地脚本结束。
- 运行任务：`stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用远程服务器；本机 PyTorch 检测到 CUDA 并用于训练；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-13 01:10-01:14 本地响应分解标签生成

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 01:10，本地 CPU 运行。
- 关闭时间：2026-05-13 01:14，本地脚本结束。
- 运行任务：`stage03_vehicle_instability_response_decomposition_labels_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用 GPU；未使用服务器；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-13 00:10-00:18 本地响应任务定义决策

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 00:10，本地 CPU 运行。
- 关闭时间：2026-05-13 00:18，本地脚本结束。
- 运行任务：`build_vehicle_instability_response_task_decision_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用 GPU；未使用服务器；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-12 22:50-22:54 本地窗口覆盖审计

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-12 22:50，本地 CPU 运行。
- 关闭时间：2026-05-12 22:54，本地脚本结束。
- 运行任务：`stage03_vehicle_instability_label_window_coverage_audit_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用 GPU；未使用服务器；未读取服务器指令与密码文件；未记录任何凭据。

## 最新更新：2026-05-12 22:34

- 本次阶段 3 复发坏样本失败来源归因 v0.1 完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 22:30 左右，本地前台运行归因脚本。
- 关闭时间：2026-05-12 22:33 左右，表格、图、报告和验证完成。
- 运行任务：`stage03_vehicle_instability_bad_event_failure_attribution_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因为未使用服务器/GPU。

## 最新更新：2026-05-12 22:18

- 本次阶段 3 复发坏样本详细曲线复盘 v0.1 完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 22:08 左右，本地前台运行曲线复盘脚本。
- 关闭时间：2026-05-12 22:17 左右，表格、图、报告和验证完成。
- 运行任务：`stage03_vehicle_instability_bad_event_curve_review_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因为未使用服务器/GPU。

## 最新更新：2026-05-12 21:44

- 本次阶段 3 稳健性坏样本复盘 v0.1 完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 21:43 左右，本地前台运行坏样本复盘脚本。
- 关闭时间：2026-05-12 21:44 左右，表格、图表和报告生成完成。
- 运行任务：`stage03_vehicle_instability_robustness_bad_sample_review_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因为未使用服务器/GPU。

## 最新更新：2026-05-12 21:37

- 本次阶段 3 强车辆基线稳健性验证 v0.1 完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 21:34 左右，本地前台运行稳健性脚本。
- 关闭时间：2026-05-12 21:36 左右，4 个配置的表格、图表和报告生成完成。
- 运行任务：`stage03_vehicle_instability_strong_vehicle_robustness_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因为未使用服务器/GPU。

## 最新更新：2026-05-12 21:24

- 本次阶段 3 车辆-only 统一对照 v0.1 完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 21:23 左右，本地前台运行汇总脚本。
- 关闭时间：2026-05-12 21:24 左右，表格、图表和报告生成完成。
- 运行任务：`stage03_vehicle_instability_unified_vehicle_comparison_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因为未使用服务器/GPU。

## 最新更新：2026-05-12 21:10

- 本次阶段 3 车辆-only Transformer v0.1 完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 21:00 左右，本地前台运行 Transformer 脚本。
- 关闭时间：2026-05-12 21:05 左右，训练、评估、图表和报告生成完成。
- 运行任务：`stage03_vehicle_instability_vehicle_transformer_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因为未使用服务器/GPU。

## 最新更新：2026-05-12 20:23

- 本次阶段 3 强车辆-only 时序/结构化基线 v0.1 完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 19:50 左右，本地前台运行强车辆-only 基线脚本。
- 关闭时间：2026-05-12 20:22 左右，模型评估、图表和报告生成完成。
- 运行任务：`stage03_vehicle_instability_strong_vehicle_baselines_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因未使用服务器/GPU。

## 最新更新：2026-05-12 19:35

- 本次阶段 3 v0.1 坏样本错误分型完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 19:25 左右，本地前台表格分析和画图。
- 关闭时间：2026-05-12 19:35 左右，错误分型表、图和报告生成完成。
- 运行任务：formal ridge 车辆基线 test 坏样本物理错误分型。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因未使用服务器/GPU。

## 最新更新：2026-05-12 19:20

- 本次阶段 3 v0.1 车辆-only 基线完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 19:10 左右，本地前台基线评估。
- 关闭时间：2026-05-12 19:20 左右，指标和图表生成完成。
- 运行任务：正式车辆失稳样本上的无学习基线和浅层车辆 ridge 基线。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因未使用服务器/GPU。

## 最新更新：2026-05-12 19:05

- 本次 `vehicle_instability_highconf_v0_1` 正式样本清单构建完全在本地运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 启动时间：2026-05-12 19:00 左右，本地前台表格构建。
- 关闭时间：2026-05-12 19:05 左右，样本清单和报告生成完成。
- 运行任务：阶段 2 正式车辆失稳 `samples_master`、split、排除原因、数据版本卡构建。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地输出路径：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1`。
- 是否还有任务在跑：否。
- GPU/显存状态摘要：未检查，因未使用服务器/GPU。

## 最新更新：2026-05-12 18:45

- 本次旧 `vehicle_direct` 全量车辆-only clean 对照完全在本地 CPU 运行。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 远程任务状态：无。
- GPU/显存状态：未检查，因为本次不使用远程 GPU。
- 启动时间：2026-05-12 18:14:13，本地 CPU 训练 run。
- 关闭时间：2026-05-12 18:39:46，本地评估和图表生成完成。
- screen/nohup 名称：无，本地前台命令。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无远程拉回；所有输出直接写入本地项目目录。
- 本地训练 run：`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_FULL_INSTABILITY_HIGHCONF_VEHICLE_DIRECT_CLEAN_V0_1_20260512_181413`。
- 本地训练日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/logs/oldcode_vehicle_direct_full_clean_train_stdout.log`。
- 本地评估输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1`。
- 是否还有任务在跑：否。
- 备注：一次 raw manifest 直读原始 CSV 的本地 run 已判定无效并清理，原因是旧 loader 会把原始缺失点填 0；正式记录只采用 clean manifest 结果。

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
# 最新更新：2026-05-12 17:15

- 本次旧 `vehicle_direct` 全量车辆-only 对照在本地 CPU 启动。
- 未连接远程服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 远程任务状态：无。
- GPU/显存状态：未检查，因为本次不使用远程 GPU。
- 本地输出预计路径：`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_FULL_INSTABILITY_HIGHCONF_VEHICLE_DIRECT_V0_1_*`；补充图表预计写入 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_on_instability_v0_1`。
## 2026-05-13 00:28-00:37 本地 clean-task 车辆-only 基线

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 00:28，本地 CPU 运行。
- 关闭时间：2026-05-13 00:37，本地脚本、报告和图表生成完成。
- 运行任务：`stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用 GPU；未使用服务器；未读取服务器指令与密码文件；未记录任何凭据。
## 2026-05-13 00:48-00:55 本地 B 轨道坏样本复查

- 服务器连接命令格式：未连接服务器；无 SSH 命令。
- 启动时间：2026-05-13 00:48，本地 CPU 运行。
- 关闭时间：2026-05-13 00:55，本地脚本、报告和图表生成完成。
- 运行任务：`stage03_vehicle_instability_clean_task_bad_sample_review_v0_1.py`。
- screen/nohup 名称：无。
- 远程项目路径：无。
- 远程日志路径：无。
- 本地拉回路径：无；所有输出直接生成在本地 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1`。
- 是否还有任务在跑：没有。
- GPU/显存状态摘要：未使用 GPU；未使用服务器；未读取服务器指令与密码文件；未记录任何凭据。

## 2026-05-19 v0.3 excluded 分层加入实验服务器记录

- 服务器连接格式：`ssh -p 55060 root@connect.westc.seetacloud.com`，密码不记录。
- 远程项目路径：`/root/autodl-tmp/data_process`。
- 远程车辆数据映射：`/root/autodl-tmp/data_process/01_datasets/多模态数据/被试数据集合/被试/vehicle/*_vehicle_aligned_cleaned.csv`。
- 本地同步压缩包：`F:/data_set_process/data_process/tmp/v03_excluded_stratified_results_20260519.tar.gz`。
- 远程日志路径：`/root/autodl-tmp/data_process/04_project_logs/reports/server_logs/v03_excluded_stratified_20260519/run.log`。
- 本地日志副本：`F:/data_set_process/data_process/04_project_logs/reports/server_logs/v03_excluded_stratified_20260519/run.log`。
- 运行状态：已完成，结果已拉回本地。
## 2026-05-19 v0.3 样本筛选策略连续对比服务器记录

- 服务器连接格式：`ssh -p 55060 root@connect.westc.seetacloud.com`，密码不记录。
- 启动时间：2026-05-19 20:34:55。
- 运行任务：`stage03_v03_screening_sweep.py`，连续比较 v0.3 多种样本筛选策略。
- screen 名称：`v03sweep`。
- 远程项目路径：`/root/autodl-tmp/data_process`。
- 远程日志路径：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_20260519_203455.log`。
- 本地拉回路径：待任务完成后补充。
- 是否还有任务在跑：是。
- GPU/显存状态摘要：NVIDIA GeForce RTX 4080 SUPER，启动前显存使用 0 MiB；本任务主要是 CPU/表格基线计算。
