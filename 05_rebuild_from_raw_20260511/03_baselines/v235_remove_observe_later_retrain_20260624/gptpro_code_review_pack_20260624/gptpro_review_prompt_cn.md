请你以严格代码审查和实验审查的角度检查这个 v235 包。

背景：我们正在做行为预测模型，不是写失败机制论文。前面发现一类样本 `observe_later_like=True`，特点是旧锚点前几秒证据弱，但锚点后变化很大。用户提出：先尝试把这类预测效果很差的样本去掉，然后重新训练模型，看结果如何。

本包中的主脚本是：

- `code/stage03_v235_remove_observe_later_retrain_20260624.py`

它复用了：

- `code/stage03_v222a_light_fusion_residual_20260622.py`
- `code/stage03_v222a_candidate_curve_cache_20260622.py` 仅作上游 cache 结构背景

请重点审查以下问题：

1. 删除样本逻辑是否正确
   - 是否确实从 v234 的 `observe_later_like=True` 样本清单取 event_uid/sample_id？
   - 是否从 train/val/test 全部剔除，而不是只剔除 test？
   - strict pool 删除 `117/963` 而 loose pool 删除 `121/1167` 是否能由 pool 交集解释？

2. 数据泄漏与选择纪律
   - feature_matrix 是否没有包含 split、subject、event_uid、true、oracle、RMSE 等目标派生字段？
   - 重训是否只用过滤后的 train split 拟合？
   - 模型选择是否只用过滤后的 validation split？
   - test 是否只在 selected 模型固定后报告？
   - removed holdout 是否只是诊断，不参与训练或选择？

3. 对照是否公平
   - `old full test`、`old filtered test`、`new filtered retrain` 三个数值是否是正确的三种不同口径？
   - 旧模型在同一过滤子集上的指标是否用于区分“删除样本收益”和“重训收益”？
   - formal lock 对照是否只作为参考，没有参与改 headline？

4. 代码实现是否有 bug
   - sample_manifest 与 NPZ event_uid 对齐是否足够严格？
   - `mask_predictions`、`metrics_for_mask`、`removed_holdout` 的 mask 是否可能错位？
   - 导出的 selected prediction NPZ、per-sample metrics、comparison summary 是否可能混入错误 split？
   - 复用 v222a_light 函数时，是否有全局常量或 OUT_DIR 副作用导致写错目录或读错文件？

5. 结论是否过度
   - 当前结果显示：删除样本会显著改善保留 test 指标，但删除后重训相对旧模型同一过滤子集只小幅改善。请判断这个结论是否由表格支持。
   - 是否应该明确说：v235 不能作为正式方法提升 headline，只能作为数据/任务定义诊断？
   - 是否应该继续把 observe_later_like 样本作为短观察层、后移观察点或重锚定对象，而不是永久删除？

关键结果表：

- `tables/v235_comparison_summary.csv`
- `tables/v235_removed_sample_counts.csv`
- `tables/v235_validation_selection_filtered.csv`
- `tables/v235_selected_metrics_filtered.csv`
- `tables/v235_old_selected_metrics_filtered.csv`
- `tables/v235_selected_metrics_removed_holdout.csv`
- `tables/v235_leakage_guard_result.csv`

请输出：

1. 代码/实验中是否存在硬错误。如果有，请指出文件、函数、具体逻辑和会导致什么后果。
2. 是否存在软风险或结论表达风险。
3. 是否认可当前主要结论：收益主要来自过滤后的测试集变容易，重训额外收益很小。
4. 如果要继续推进，下一步应该优先修代码、重跑实验，还是回到短观察层/重锚定路线。
