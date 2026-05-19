# v0.3 excluded 分层加入实验（用户查看版）

## 这次为什么做

用户判断 `excluded` 不能直接全丢，因为数据集太小，而且其中可能有可用样本。上一轮直接加入全部 `excluded` 后，结果比“干净集 + 待复核”差，说明 `excluded` 里既有可用样本，也有会拉乱任务的样本。

所以本轮不把 `excluded` 一刀切删除，而是做分层验证：先去掉最容易受坐标跳变影响的 `lateral_distance_selected` 输入，再分别加入低附着、弯道/曲率、横滚/姿态、横向动态来源的 `excluded` 样本。

## 数据路径说明

- 本地新流程目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511`
- 本地原始车辆数据：`F:/data_set_process/data_process/01_datasets/数据预处理/原始车辆数据`
- 本地 v0.3 episode 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/extreme_condition_episodes_all_v0_3.csv`
- 本轮结果目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion`
- 服务器运行路径：`/root/autodl-tmp/data_process`，密码没有写入任何项目文件。

注意：服务器上的车辆数据结构不同，实际映射到 `/root/autodl-tmp/data_process/01_datasets/多模态数据/被试数据集合/被试/vehicle/*_vehicle_aligned_cleaned.csv`。脚本已加入路径映射逻辑，避免误读空目录。

## 汇总结果

| 版本 | 含义 | 样本数 | 最优模型 | test RMSE | 大响应错侧率 | 严重幅值不足率 | 大响应召回 | 原干净子集RMSE |
|---|---|---:|---|---:|---:|---:|---:|---:|
| v03_plus_review_ref | 干净集 + 待复核（原特征参考） | 793 | rbf_kernel_vehicle_context_alpha0.1_g1 | 0.5967 | 0.1923 | 0.6538 | 0.6154 | 0.7945 |
| v03_plus_review_no_lateral | 干净集 + 待复核（去横向偏移） | 793 | rbf_kernel_vehicle_context_alpha0.1_g1 | 0.6043 | 0.2308 | 0.6538 | 0.6346 | 0.7985 |
| v03_plus_review_excluded_all_no_lateral | 干净集 + 待复核 + 全部 excluded（去横向偏移） | 1562 | rbf_kernel_vehicle_context_alpha1_g1 | 0.7321 | 0.1515 | 0.6162 | 0.6364 | 0.7594 |
| v03_plus_review_excluded_low_mu_no_lateral | 干净集 + 待复核 + 低附着 excluded（去横向偏移） | 1204 | rbf_kernel_vehicle_context_alpha1_g1 | 0.6893 | 0.2444 | 0.7333 | 0.6333 | 0.7723 |
| v03_plus_review_excluded_curve_no_lateral | 干净集 + 待复核 + 弯道 excluded（去横向偏移） | 846 | rbf_kernel_vehicle_context_alpha1_g0.5 | 0.6517 | 0.2182 | 0.7455 | 0.6727 | 0.7834 |
| v03_plus_review_excluded_roll_no_lateral | 干净集 + 待复核 + 横滚姿态 excluded（去横向偏移） | 1092 | knn_vehicle_history_context_k9 | 0.6661 | 0.1250 | 0.5000 | 0.6964 | 0.7777 |
| v03_plus_review_excluded_lateral_dyn_no_lateral | 干净集 + 待复核 + 横向动态 excluded（去横向偏移） | 799 | knn_vehicle_history_context_k5 | 0.6536 | 0.2000 | 0.6909 | 0.6727 | 0.8200 |

## 当前结论

1. **目前不能直接全量加入 `excluded`。**
   全量加入并去掉横向偏移后，test RMSE=0.7321，明显差于参考版本 0.5967。这说明 `excluded` 里确实有会拉乱任务的样本。

2. **当前整体最稳的训练范围仍是“干净集 + 待复核”。**
   参考版本样本数 793，test RMSE=0.5967，是本轮最低；去掉横向偏移后反而略差，说明对当前参考集来说，横向偏移不是纯噪声。

3. **横滚/姿态类 `excluded` 有保留价值，但不适合直接混成普通样本。**
   横滚/姿态版本 test RMSE=0.6661，不如参考版本；但大响应错侧率=0.1250，严重幅值不足率=0.5000，比参考版本更好。这说明它可能对“强姿态/强响应”有用，但会改变整体分布，后续更适合单独作为极限工况子集或加权训练，而不是粗暴混入。

4. **低附着和弯道类 `excluded` 暂时不建议直接加入主训练。**
   低附着版本错侧率和严重幅值不足率都变差；弯道版本大响应召回略高，但整体误差和幅值不足偏差仍明显。它们可以继续人工看图复核，但不应自动升级为主训练样本。

## 下一步建议

- 主训练样本先采用“干净集 + 待复核”。
- `excluded` 不丢，但拆成风险池：横滚/姿态优先人工复核，低附着和弯道先看坏样本图再决定是否重标。
- 后续如果要用 `excluded`，更合理的方式不是全量加入，而是：分场景训练、样本加权、或把横滚/姿态作为极限工况子任务。
- 在继续风格/生理之前，建议先检查本轮各版本的坏样本图，确认“横滚/姿态 excluded”到底是在改善强响应，还是只是改变了测试集组成。

## 可查看文件

- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion/tables/v03_excluded_stratified_inclusion_summary.csv`
- 参考版本固定图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion/v03_plus_review_ref/figures/v03_plus_review_ref_fixed_predictions_test.png`
- 参考版本坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion/v03_plus_review_ref/figures/v03_plus_review_ref_bad_samples_test.png`
- 横滚/姿态固定图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion/v03_plus_review_excluded_roll_no_lateral/figures/v03_plus_review_excluded_roll_no_lateral_fixed_predictions_test.png`
- 横滚/姿态坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion/v03_plus_review_excluded_roll_no_lateral/figures/v03_plus_review_excluded_roll_no_lateral_bad_samples_test.png`
- 服务器日志本地副本：`F:/data_set_process/data_process/04_project_logs/reports/server_logs/v03_excluded_stratified_20260519/run.log`
