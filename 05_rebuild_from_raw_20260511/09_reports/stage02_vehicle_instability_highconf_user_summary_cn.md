# 阶段 2 用户查看版：车辆失稳高置信样本清单 v0.1

生成时间：2026-05-12

## 为什么做

之前旧代码已经能在 906 个高置信失稳样本上跑通，但那只是旧代码对照。要进入新流程强车辆基线，必须先有正式、可追溯、无泄漏的 `samples_master`。

## 检查了什么

- 每个样本是否能追溯到原始车辆 CSV、sha256、被试、记录和事件锚点。
- 每个样本是否有明确输入窗口和标签窗口。
- split 是否不依赖未来方向盘标签。
- 生理/脑电是否只是记录可用性，没有提前抽窗口或使用。
- 两个未进入正式样本的事件是否有排除原因。

## 目前发现

908 个高置信车辆失稳事件中，906 个满足完整历史和未来窗口要求；2 个因为窗口覆盖不足被排除。906 个事件各生成 3 个窗口，总样本行数 2718。主窗口 `pre2_label2_old_main` 的 session-level split 为 train 611、val 156、test 139。

## 哪些结果可信

样本锚点来自 `ay/roll_rate` 等非方向盘车辆动态，方向盘没有参与锚点定义。manifest 没有做标准化，后续训练必须只在训练集拟合 scaler。`eval_label_*` 字段只用于评估分层，不能作为模型输入。

## 哪些结果还不能下结论

这还不是强车辆基线结果，也不能证明连续风格、生理或脑电有效。生理和脑电目前只是记录了原始文件是否可用，还没有进入窗口构建和增量验证。

## 下一阶段是否可以继续

可以继续进入新流程车辆基线阶段。下一步应先做无学习基线和强车辆基线，再决定是否进入连续风格和生理验证。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_highconf_v0_1_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/split_feasibility_report.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/sample_exclusion_reasons.csv`
