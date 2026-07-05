# v309 近期最好模型预测效果图册

## 结论

- 近期最好版本：`v307_coarse_scene_init_aux003_film005_h64`。
- 参照版本：`v300_full_joint_h64_no_subject`。
- test delay0/all：v300 `0.519805` -> v307 `0.496138`。
- test delay0/within_bad_top10：v300 `0.859987` -> v307 `0.777797`。

## 图册

- HTML：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\index.html`
- 代表性样本图数：`54`
- 全 test delay0 指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\tables\v309_test_delay0_prediction_effect_table.csv`
- 图册样本表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\tables\v309_gallery_sample_manifest.csv`

## 如何读图

- 方向盘角第一行：灰色/黑色为真实车辆轨迹，蓝色为 v307，橙色为 v300。
- 第二行是模型真正预测的目标 `steering_delta`，只覆盖 `0-2s`。
- `2s` 后只展示真实后续，帮助判断车辆是否继续失稳、回正或出现二次修正；这不是模型预测范围。

## 边界

- v307 仍使用 v306 粗场景 seed，其中直道连续/紧急子类还需要人工确认。
- 本图册用于观察近期最好模型效果，不代表最终部署模型。

## 验证

- 读取 v307 NPZ 预测包成功。
- test delay0 事件数 `232`。
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\v309_recent_best_prediction_effect_gallery_20260704.zip`。
- 生成耗时：`104.0s`。
