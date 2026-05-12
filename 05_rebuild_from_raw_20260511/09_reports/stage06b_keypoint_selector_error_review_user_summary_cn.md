# Stage 6b 用户查看版：RBF/keypoint 选择器错误复盘 v0.1

## 为什么做

阶段 6 审计发现，`selector_logreg_rbf_keypoint_no_subject` 没有明显超过 RBF，但它在方向、大幅响应和困难样本上有一些信号。这个阶段要回答：选择器是因为错选 keypoint 变差，还是因为漏掉了 keypoint 本来能改善的样本。

## 检查了什么

- 只看 B 轨道 test 40 个样本。
- 比较每个样本中 RBF 和 keypoint 谁的 RMSE 更低。
- 检查 selector 实际选了谁。
- 统计 TP/FP/FN/TN、oracle regret、相对 RBF 是帮了还是害了。
- 没有使用生理、脑电、连续风格、驾驶员 ID、服务器或服务器密码文件。

## 目前发现

- test 样本数：40
- selector 选择 keypoint 比例：0.275
- oracle 中 keypoint 更优比例：0.425
- TP 选对 keypoint：5；FP 错选 keypoint：6；FN 漏选 keypoint：12；TN 保持 RBF 正确：17
- selector 相对 RBF 帮助样本：5；伤害样本：6
- selector 平均 RMSE delta vs RBF：+0.006945
- selector 平均 oracle regret：0.059122

## 当前判断

选择器不是完全没信号，但当前概率阈值和特征还不能稳定识别“keypoint 真正更好”的样本。下一步应优先复盘 FN 和 FP，而不是直接加入生理/EEG。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_confusion_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_top_regret_samples.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_probability_vs_gain.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_top_regret_samples.png`
