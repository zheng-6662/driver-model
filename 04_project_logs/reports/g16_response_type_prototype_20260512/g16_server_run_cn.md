# G16 路线2服务器运行记录

## 基本信息

- 日期：2026-05-12
- 服务器：788机
- SSH 命令格式：`ssh -p 55060 root@connect.westc.seetacloud.com`
- 远程项目路径：`/root/autodl-tmp/data_process`
- 本地项目路径：`F:/data_set_process/data_process`
- Python：`/root/miniconda3/bin/python`
- GPU：NVIDIA vGPU-32GB
- 密码记录：未写入任何项目日志、报告或代码。

## 执行内容

本轮执行的是旧流程路线2，也就是：

1. 从当前 FAIR 样本清单读取 train/val/test。
2. 用真实未来方向盘轨迹生成响应类型标签，包括方向、幅值、形态。
3. 在训练集上训练只能使用触发前信息的响应类型判断器。
4. 用预测到的响应类型选择或软组合训练集原型轨迹。
5. 记录整体误差、主响应误差、尾段误差、综合选择指标、G11 困难样本、分被试和分响应类型结果。

## 运行脚本

- 本地脚本：`F:/data_set_process/data_process/02_code/final_code/model/training/fair_vehicle_event_comparison_20260427/run_g16_response_type_prototype.py`
- 远程脚本：`/root/autodl-tmp/data_process/02_code/final_code/model/training/fair_vehicle_event_comparison_20260427/run_g16_response_type_prototype.py`

## 运行日志

第一次运行完成主体计算，但因为服务器 `scikit-learn` 接口不接受旧参数，报告生成阶段发现分类器表缺少字段，已修复脚本后重跑。

- 第一次日志：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/server_logs/g16_route2_20260512_144222.log`
- 兼容修复重跑日志：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/server_logs/g16_route2_rerun_20260512_144646.log`
- 干净版最终日志：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/server_logs/g16_route2_clean_20260512_150423.log`

## 最终输出

- 报告：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/g16_response_type_prototype_report_cn.md`
- 验证集筛选表：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/g16_validation_screening.csv`
- 测试集候选表：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/g16_test_all_candidates.csv`
- 选中结果：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/g16_test_chosen_by_validation.csv`
- 响应类型判断器指标：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/g16_classifier_metrics.csv`
- G11 明细：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/g16_g11_detail.csv`
- 分组结果：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/g16_group_summary.csv`
- 预测数组：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/g16_chosen_predictions_test.npz`
- 固定图：`F:/data_set_process/data_process/04_project_logs/reports/g16_response_type_prototype_20260512/figures/g16_selected_g11_comparison.png`

## 结果判断

G16 不能作为新主线。可部署版本整体 RMSE 虽然低，但主响应、尾段、综合选择指标和 G11 困难样本明显弱于 E5A/E10C。响应类型判断器对细响应形态的识别也不够稳定，说明“先判断类型，再输出类型平均原型”仍然解决不了物理意义问题。
