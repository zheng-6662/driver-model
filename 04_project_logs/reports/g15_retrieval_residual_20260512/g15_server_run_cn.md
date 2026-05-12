# G15 服务器运行记录

## 运行环境

- 实例：AutoDL `788机`
- 状态：已由用户手动开机，运行中
- SSH 格式：`ssh -p 55060 root@connect.westc.seetacloud.com`
- 远程项目路径：`/root/autodl-tmp/data_process`
- Python：`/root/miniconda3/bin/python`
- GPU：`NVIDIA vGPU-32GB`

说明：本文件不记录服务器密码。

## 同步内容

同步到服务器的主要源码：

- `02_code/final_code/model/training/fair_vehicle_event_comparison_20260427/run_g15_retrieval_residual.py`
- `02_code/final_code/model/training/fair_vehicle_event_comparison_20260427/run_g14_retrieval_reference.py`
- `02_code/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
- `02_code/final_code/model/training/event_conditioned_eval_support.py`
- 相关训练和评价依赖源码

没有同步服务器密码文件、历史大 checkpoint 或无关项目目录。

## 运行记录

第一次运行：

- 远程日志：`/root/autodl-tmp/data_process/04_project_logs/reports/g15_retrieval_residual_20260512/server_logs/g15_route1_20260512_135256.log`
- 结果：完成。
- 备注：发现预测产物中非方向盘通道保留真实未来值。方向盘指标不受影响，但为了产物严谨性进行修复后重跑。

修复后重跑：

- 远程日志：`/root/autodl-tmp/data_process/04_project_logs/reports/g15_retrieval_residual_20260512/server_logs/g15_route1_rerun_20260512_140134.log`
- 结果：完成。
- 本地拉回目录：`F:/data_set_process/data_process/04_project_logs/reports/g15_retrieval_residual_20260512`

## 关键结果

- G15A 相似历史检索：`test_rmse=0.4023`，`primary_rmse=0.3967`，`tail_rmse=0.4414`，`selection=0.9171`，G11 `rmse=0.8300`。
- G15B 检索 + 残差修正：`test_rmse=0.3980`，`primary_rmse=0.3873`，`tail_rmse=0.4356`，`selection=0.8829`，G11 `rmse=0.8087`。

## 当前判断

G15 能降低整体 2 秒平均误差，但主响应、尾段、综合选择指标和 G11 困难样本仍弱于旧强候选。因此它更适合作为“历史原型参考”或后续响应类型候选库，不应直接作为最终主线。
