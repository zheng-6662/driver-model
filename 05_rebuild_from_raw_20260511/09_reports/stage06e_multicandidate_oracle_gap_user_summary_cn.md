# Stage 6e 用户查看版：多候选 oracle gap 复核 v0.1

## 为什么做

Stage 6d 说明当前 RBF/keypoint reliability gate 不能升级。这个阶段不训练新模型，只把已有车辆-only候选放到同一个候选池里，检查“如果有 oracle 按真实标签挑候选，上限有多高”和“实际可部署 selector 离上限差多远”。

## 目前发现

- 当前 RBF/KNN 主参照 test RMSE=0.533667。
- broad oracle pool test RMSE=0.375182，相对 RBF/KNN delta=-0.158484；这个结果不可部署，因为它用真实标签挑选最佳候选。
- 当前最好的可部署 selector test RMSE=0.533912，相对 RBF/KNN delta=+0.000245，没有把 oracle 上限稳定转成实际增益。
- 结论不是“Transformer 更好”，也不是“生理该进来”；结论是车辆-only 多候选路线存在上限，但选择策略还没解决。

## 当前判断

可以进入 Stage 7 的前提不是继续报告 best-of-K，而是建立不用真实标签的候选选择策略、概率校准和坏样本可靠性判断。生理/EEG 仍阻塞。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_gap_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_winner_summary.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/figures/multicandidate_oracle_gap_rmse.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/figures/multicandidate_oracle_winner_counts.png`
