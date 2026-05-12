# 阶段 3 技术报告：车辆-only 主参照决策表 v0.2

## 输入产物

本报告只读取既有阶段 3 指标表，不重新训练模型。输入包括 clean-task baselines、direct Transformer、structured Transformer、keypoint+residual、RBF/keypoint selector、top-K reliability selector 和旧 unified comparison。

## 决策结论

- 当前车辆-only 主参照：`rbf_kernel_ridge_context_no_subject`。
- 主参照状态：`not_fully_frozen`。
- 阻塞项：RBF 错侧率仍为 0.225，反向修正完全匹配率为 0.000；top-K fallback 未超过 RBF。
- 风格/生理/EEG 入口：仍阻塞。

## Gate 表

| gate_item | status | evidence | decision_cn |
| --- | --- | --- | --- |
| vehicle_main_reference_available | partial | RBF test RMSE=0.533667, wrong_side=0.225, large_recall=0.750 | 可以作为当前主参照，但不能说物理问题已解决。 |
| strong_vehicle_baseline_frozen | no | RBF 反向修正完全匹配仍为 0，错侧率 0.225；top-K fallback 未超过 RBF。 | 阶段 3 仍未完全冻结，进入风格/生理前需明确接受 RBF 作为保守主参照或继续车辆-only 结构。 |
| topk_reliability_selector_upgrade | no | fallback test RMSE=0.542071 > RBF 0.533667 | 本轮选择器 no-go。 |
| oracle_upper_bound_interpretable | yes_but_not_deployable | best-of-RBF+topK oracle RMSE=0.415652 | 只说明候选池还有上限空间，不能作为模型效果。 |
| style_physio_eeg_allowed_now | no | 强车辆基线/主参照冻结仍未闭环。 | 继续阻塞连续风格、生理和 EEG 增量结论。 |

## 关键模型 test 指标

| 模型 | 角色 | RMSE | 错侧率 | 大幅召回 | 困难 top20 RMSE | 决策 |
|---|---|---:|---:|---:|---:|---|
| RBF KRR | main reference | 0.533667 | 0.225 | 0.750 | 0.678907 | 暂定主参照 |
| keypoint+residual | weak candidate | 0.548994 | 0.125 | 0.875 | 0.728866 | 分支候选 |
| topK fallback | no-go | 0.542071 | 0.225 | 0.750 | 0.678907 | 不升级 |
| RBF+topK oracle | oracle | 0.415652 | 0.075 | 0.875 | 0.604369 | 上限诊断 |

## 解释

本轮决策不是为了宣布阶段 3 完成，而是为了避免后续误用阶段 3 产物。RBF KRR 是当前最稳的车辆-only 主参照，但仍不是“物理响应已经解决”的强结论。keypoint 和 top-K 的价值主要体现为候选池/结构线索，尚未形成可部署增益。
