# v222a 候选曲线缓存导出报告

## 结论

- 已从 v218/v219 历史模块重建每个 pool 的候选曲线，并导出 v222a 可读取的 NPZ 缓存。
- ridge residual 候选全部通过 `selected_by=validation_only` 与 `test_used_for_selection=false` 校验。
- `feature_schema_audit.csv` 未发现 split、subject、true、oracle、RMSE、severe-under 等禁用字段。
- 与 v219 既有指标表的数值交叉检查全部通过。
- 本阶段没有训练新模型，也没有把 `W3_B4_original_soft` 放入 formal 候选或榜单。

## 输出文件

- `candidate_predictions_loose_main_pool.npz`：pool=loose_main_pool，样本=1167，候选=14，predictions_shape=(1167, 14, 21)，feature_shape=(1167, 229)
- `candidate_predictions_strict_main_pool.npz`：pool=strict_main_pool，样本=963，候选=14，predictions_shape=(963, 14, 21)，feature_shape=(963, 229)
- `candidate_manifest.csv`：候选级来源、scope 与 validation-only 元数据。
- `sample_manifest.csv`：样本定位与 split 字段，仅供审计和分组，不作为推理特征。
- `feature_schema_audit.csv`：v219 ridge residual feature schema 泄漏审计。
- `candidate_curve_metrics.csv`：候选曲线按 pool/split 的评估指标。
- `metric_crosscheck_vs_v219.csv`：本次重建与 v219 原表的指标差异。
- `v222a_candidate_curve_cache_pack.zip`：本阶段打包文件。

## 候选范围

- formal 候选行数：16
- diagnostic 候选行数：12
- formal 候选名：avg_joint_focus, global_blend, joint_equal, joint_steer_focus, peak_floor_090, ridge_residual_joint, ridge_residual_peakfloor, steering_only

## Test split formal 候选前三

- loose_main_pool / avg_joint_focus: RMSE=0.544884, tail=0.629752, under=0.163043
- loose_main_pool / global_blend: RMSE=0.549790, tail=0.635805, under=0.173913
- loose_main_pool / joint_equal: RMSE=0.553940, tail=0.629657, under=0.141304
- strict_main_pool / peak_floor_090: RMSE=0.571770, tail=0.658306, under=0.137931
- strict_main_pool / global_blend: RMSE=0.575574, tail=0.672493, under=0.258621
- strict_main_pool / avg_joint_focus: RMSE=0.580618, tail=0.678829, under=0.195402

## 审计摘要

- feature schema 行数：458，fail 行数：0
- v219 交叉检查行数：72，最大差异：0.0019267822736
