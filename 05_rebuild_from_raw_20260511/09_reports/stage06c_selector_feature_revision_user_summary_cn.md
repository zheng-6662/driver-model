# Stage 6c 用户查看版：selector feature revision v0.1

## 为什么做

Stage 6b 发现 keypoint selector 主要问题是漏选 keypoint 收益样本，同时也有错选 keypoint 伤害样本。本轮尝试在不使用未来真实标签、不使用生理/脑电/风格的前提下，加入候选模型预测差异特征，看看能不能把 oracle 上限转成可部署选择策略。

## 检查了什么

- 原始 logistic selector。
- 增加候选差异特征后的 logistic selector。
- 一个浅层随机森林 selector。
- 所有候选只在 train 拟合，只用 val 选阈值，test 只最终评估。

## 目前发现

- 当前 val 选择的最佳 selector：`rf_engineered_shallow`。
- RBF test RMSE=0.533667；最佳 selector test RMSE=0.544356，delta=+0.010689。
- 最佳 selector wrong-side=0.175，RBF wrong-side=0.225。
- 最佳 selector large recall=0.875，RBF large recall=0.750。
- 最佳 selector 仍有 FN=6、FP=13 类错误。

## 当前判断

如果 gate 表显示 `no_upgrade_current_revision`，说明这版 feature revision 仍不能升级为主线，只能作为下一版可靠性门控的诊断依据。生理/EEG 仍不能进入有效性结论。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_gate_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_metrics.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/figures/selector_revision_test_rmse.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/figures/selector_revision_physical_metrics.png`
