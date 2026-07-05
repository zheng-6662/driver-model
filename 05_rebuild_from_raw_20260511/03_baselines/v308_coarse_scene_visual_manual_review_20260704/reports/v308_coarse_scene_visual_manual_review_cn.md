# v308 coarse scene 视觉人工复核包

## 目的

用户反馈看表不容易区分，因此本版本把 v306 的 high + medium 复核队列改成逐事件曲线图册。

## 如何使用

- 打开 `index.html`：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\index.html`
- 点击任意图可以打开大图。
- 在每张图下方选择“复核结论”和“人工标签”，可写备注。
- 页面会把选择暂存在浏览器 localStorage；完成后点击“导出复核 CSV”。

## 图中信号

- 方向盘角：锚点前历史 steering + 锚点后真实 steering_delta 还原后的方向盘角。
- 方向盘速度：锚点前由方向盘角估计，锚点后使用真实 steering_rate。
- ay / yaw rate / roll：辅助判断车辆是否在猛打方向后开始失稳。
- 曲率/横向距离/车速/制动：辅助判断过弯、横向偏移、制动参与。

## 数量

- 复核图数量：748
- high：529
- medium：219

| coarse_scene_review_priority   | coarse_scene_label                |   n |
|:-------------------------------|:----------------------------------|----:|
| high                           | continuous_lane_change            | 414 |
| high                           | emergency_lane_change_instability | 115 |
| medium                         | other_or_uncertain                | 219 |

## 重要边界

- 图册使用了锚点后真实响应，只用于人工复核标签，不是可部署模型输入。
- 下坡过弯/平路过弯主要由 `scene_type` 给出；当前曲线图不直接显示道路坡度。
- 如果只凭图看不清，应标成 `uncertain` 或 `exclude_or_unclear`，不要为了凑类别强行确认。

## 输出

- HTML 图册：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\index.html`
- 复核队列清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\tables\v308_visual_review_manifest.csv`
- 人工填写模板：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\tables\v308_manual_review_decision_template.csv`
- ZIP 包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\v308_coarse_scene_visual_manual_review_20260704.zip`

生成耗时：880.4s
