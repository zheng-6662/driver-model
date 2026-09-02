# 数据和评估脚本清单

| 脚本             | 作用                                                           | 上游只读脚本                                        |
|:-----------------|:---------------------------------------------------------------|:----------------------------------------------------|
| build_dataset.py | 恢复221条来源，20 Hz重采样，构建连续窗口、固定评价映射和人口表 | nan                                                 |
| experiment.py    | 三种子三折驾驶员隔离OOF，训练五模型，汇总指标并生成曲线图      | nan                                                 |
| validate.py      | 独立检查人口、模型、种子、泄漏、表格和Review-light完整性       | nan                                                 |
| nan              | 恢复原始与8月车辆来源及统一通道                                | 02_code/tools/build_multiaction_reframe_audit.py    |
| nan              | 解释Run57 V3 P_full=2323历史对照人口                           | 02_code/scripts/verify_run57_contract_invariants.py |
