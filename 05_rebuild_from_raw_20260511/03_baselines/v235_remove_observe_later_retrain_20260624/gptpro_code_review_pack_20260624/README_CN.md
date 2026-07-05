# v235 GPTPro 代码审查包说明

## 审查对象

本包用于审查 `v235_remove_observe_later_retrain_20260624` 的代码与实验口径。用户目标是检查“删除 observe_later_like 差样本后重训模型”这一步是否存在代码问题、数据泄漏、选择口径不公平或结论过度。

## 主脚本

- `code/stage03_v235_remove_observe_later_retrain_20260624.py`
  - 本轮新增脚本。
  - 从 v234 的 `observe_later_like=True` 清单中取删除样本。
  - 从 v222a cache 读取候选曲线、feature_matrix、true_steer 和 split。
  - 从 train/val/test 全部剔除 observe_later_like 样本。
  - 在过滤后的 train 上重训 light residual/convex/absolute Ridge。
  - 在过滤后的 validation 上选择模型。
  - 在过滤后的 test 和 removed holdout 上报告最终 selected 模型。

## 直接依赖代码

- `code/stage03_v222a_light_fusion_residual_20260622.py`
  - v235 复用了其中的 cache 读取、feature schema 审计、指标计算、convex blend、bounded residual 训练和 validation selection 函数。
- `code/stage03_v222a_candidate_curve_cache_20260622.py`
  - 生成 v222a cache 的上游脚本。v235 不直接调用它，但 v235 的输入结构来自该脚本导出的 NPZ/cache。

## 关键结果文件

- `reports/v235_remove_observe_later_retrain_cn.md`
  - 中文报告。
- `tables/v235_comparison_summary.csv`
  - 主对照表。建议优先检查。
- `tables/v235_removed_sample_counts.csv`
  - 各 split 删除规模。
- `tables/v235_validation_selection_filtered.csv`
  - 过滤后 validation selection 排序。
- `tables/v235_selected_metrics_filtered.csv`
  - 删除后重训模型在保留样本上的指标。
- `tables/v235_old_selected_metrics_filtered.csv`
  - 旧 v222a selected 模型在同一保留样本上的指标。
- `tables/v235_selected_metrics_removed_holdout.csv`
  - 删除后重训模型在被删除样本上的诊断指标。
- `tables/v235_leakage_guard_result.csv`
  - 本轮 guard 结果。
- `logs/run_manifest.json`
  - 本轮运行 manifest。

## 输入文件指纹

- `fingerprints/v235_input_file_fingerprints.csv`
  - 记录 v235 关键输入文件的本地路径、大小和 SHA256。
  - 为避免 review 包过大，v222a 原始 cache NPZ 未全部复制进包。v235 自己生成的 selected prediction NPZ 放在 `arrays/`。

## 本轮主要数值

- loose pool 删除 `121/1167`，strict pool 删除 `117/963`。
- 旧 v222a selected full test RMSE：
  - loose `0.555940`
  - strict `0.571966`
- 旧模型在同一过滤 test 子集上：
  - loose `0.482685`
  - strict `0.506547`
- 删除后重训：
  - loose `0.474318`
  - strict `0.504151`
- 被删除 test 样本仍很难：
  - loose removed holdout RMSE `0.868780`
  - strict removed holdout RMSE `0.845273`

## 当前结论边界

本轮不能作为正式方法提升结果直接写入 headline。更合理的解释是：observe_later_like 样本确实显著拉低原始测试指标；删除后保留测试集明显变容易；但删除后重训相对旧模型同一过滤子集只带来小幅额外收益。因此 v235 更像数据/任务定义诊断，不应替代后续短观察层、后移观察点或重锚定路线。
