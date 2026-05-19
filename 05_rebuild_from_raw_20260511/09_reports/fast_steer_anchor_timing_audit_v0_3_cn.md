# 快速转向候选锚点时序审计 v0.3

用户复核发现部分样本虽然方向盘快速打，但锚点已经落在主要动作之后，锚点后车辆和方向盘都趋于稳定。这类样本会把模型训练成预测稳定尾段，应从极限响应主训练中剔除。

## 数量

- `ANCHOR_USABLE_FAST_RESPONSE`：1 个。
- `EXCLUDE_LATE_ANCHOR_STABILIZED`：5 个。
- `FAST_STEER_WEAK_POST_RESPONSE`：69 个。
- `RISK_LATE_ANCHOR_REVIEW`：3 个。

## 指定样本

- `V03_gzj_2025_09_27_12_28_14_0004`：`EXCLUDE_LATE_ANCHOR_STABILIZED`。主要方向盘动作和车辆动态峰值都在锚点前，锚点后进入恢复/稳定段；不适合作为后续响应预测训练样本。 前/后方向盘角速度峰值比=5.14，前/后车辆动态比=4.65。

## 结论

- 锚点偏晚样本不进入极限主训练。
- 疑似偏晚样本进入人工复核或风险池。
- 后续训练时应增加“锚点时序合格”过滤条件，避免把恢复段当成响应预测任务。

## 文件

- 审计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\fast_steer_anchor_timing_audit_v0_3.csv`
- 图片目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\fast_steer_anchor_timing_split_v0_3`
- 图片说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\fast_steer_anchor_timing_split_v0_3\00_先看这里_锚点时序复核说明.md`