# v0.3 极限工况样本人工复核指南

## 复核目标

这次人工复核不是为了看模型预测好不好，而是确认哪些样本真的属于“极限/近极限工况下的驾驶员响应”。正常弯道和普通驾驶可以保留，但应作为对照集，不应直接混入极限主训练集。

## 推荐查看顺序

- A_先看强响应标尺：12 张图。
- B_重点看横滚姿态候选：20 张图。
- C_待复核高分边界样本：25 张图。
- D_检查正常弯道对照是否混入极限：25 张图。
- E_弱响应保守响应：15 张图。
- F_延迟或无明显转向：15 张图。

## 每张图重点看什么

1. 锚点附近是否真的出现车辆高动态：横滚、横摆、横向加速度、低附着或制动/减速与横向动态同时出现。
2. 方向盘响应是否与这个工况有关：强转向、弱转向、延迟转向、保持通过都可以，但要能解释。
3. 是否只是正常弯道平滑过弯：如果没有明显姿态异常或急剧动态，不要放入极限主样本。
4. 横向偏移是否存在坐标跳变：如果只有横向偏移跳变而其他车辆动态不支持，要标为风险或排除。
5. 输入窗口是否合理：事件前是否已经明显开始转向；如果锚点太晚，只能做恢复阶段样本或排除。

## 建议人工标注结果

- `KEEP_EXTREME_MAIN`：进入极限/近极限主样本。
- `KEEP_WEAK_CONSERVATIVE`：保留为弱响应/保守响应。
- `KEEP_DELAYED`：保留为延迟或无明显转向响应。
- `NORMAL_CONTROL`：普通驾驶或正常弯道对照。
- `RISK_POOL`：可能有用，但存在坐标、窗口、锚点风险，先不进主训练。
- `EXCLUDE`：排除。

## 文件位置

- 优先复核清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\manual_review_priority_list_v0_3.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\review_panels`
- 原始复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`

## 最小复核量建议

先不要全量看。第一轮看优先清单中的前 40 张：A 组 12 张 + B 组 20 张 + C 组前 8 张。看完后如果分类标准稳定，再继续看 D/E/F。

## 第一轮复核后怎么用

如果 B 组横滚/姿态候选里多数能判为 `KEEP_EXTREME_MAIN` 或 `KEEP_WEAK_CONSERVATIVE`，就可以把“横滚/姿态样本池”作为极限姿态扩展集继续训练验证。若 B 组多数是坐标跳变或锚点偏移，则不应把 excluded 直接加入主训练。
