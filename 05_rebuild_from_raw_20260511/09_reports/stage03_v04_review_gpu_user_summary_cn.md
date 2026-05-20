# v0.4 主训练+次级+待复核样本 GPU 结果

## 这次为什么做

用户希望不要轻易丢掉待复核样本，因此本轮在 v0.4 主训练候选和次级候选基础上，继续加入待复核样本，检查它们是否能扩充样本覆盖并改善车辆-only 预测。

本轮只跑一组，沿用上一轮较稳的设置：去掉横向偏移，不加入连续驾驶风格、生理或脑电。

## 运行设置

- 设备：`cuda`。
- 输入：车辆历史 + 事件/工况上下文字段。
- 标签：锚点后的方向盘相对轨迹。
- 模型：无学习基线 + PyTorch 线性头/小型神经网络；按验证集 RMSE 选模型，再报告测试集。

## 结果

- 可用样本数：1410
- 验证集选择模型：`torch_mlp512_vehicle_context`
- test RMSE：0.8067
- 主阶段 RMSE：0.5786
- 尾段 RMSE：0.9290
- 大响应错侧率：0.1398
- 严重幅值不足率：0.2796
- 大响应召回：0.9032

## 产物位置

- 汇总表：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v04_review_gpu_baseline/tables/v04_review_gpu_summary.csv`
- 输出目录：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v04_review_gpu_baseline`