# v222a Candidate Cache Execution Plan

## Goal

在 v221 统一评估已经完成的基础上，先生成 v222a 需要的固定候选曲线缓存：

- `candidate_predictions_{pool_key}.npz`
- `candidate_manifest.csv`
- `sample_manifest.csv`
- 特征 schema 审计、泄漏守卫结果、候选曲线指标和 ZIP 包

本阶段只做缓存导出与审计，不训练新的神经网络，不改候选池，不把测试集用于选择。

## Guardrails

- `W3_B4_original_soft` 不进入任何 formal 表、gate、oracle 或 usage 表。
- `sample_id/event_uid/split/subject/manifest/true/oracle/RMSE/severe-under` 等字段不得进入推理特征。
- `true_steer` 只能作为训练或评估标签保存在缓存中，不能出现在 `feature_matrix` 或 feature schema 中。
- v219 ridge residual 模型必须声明 `selected_by=validation_only` 且 `test_used_for_selection=false`。
- 所有 shape、样本顺序、候选名和特征名必须通过显式断言。

## Tasks

- [ ] 用 v218/v219 历史模块恢复每个 pool 的基础候选曲线。
- [ ] 从 v219 pickle 恢复 ridge residual 候选，并校验 validation-only 元数据。
- [ ] 导出 NPZ、candidate manifest、sample manifest 和 feature schema audit。
- [ ] 计算候选曲线指标，并与 v219 指标表做数值交叉检查。
- [ ] 运行 `py_compile`、脚本本体、ZIP 校验和禁止字段检查。
- [ ] 若缓存完整，再决定是否进入 v222a 轻量软融合/受限残差训练脚本。
