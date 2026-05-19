# 新规则补充：方向盘角速度候选样本 v0.3

用户复核 `06_轻微或坐标风险_优先排除` 后指出，方向盘角速度本身也可能代表紧急性。因此本次从原先优先排除/普通对照倾向样本中，单独提取快速转向候选。

## 筛选条件

- 原自动分组属于 `EXCLUDE_LIGHT_OR_COORD_RISK` 或 `NORMAL_CONTROL_OR_EXCLUDE_LIGHT_STEER`；
- `steer_rate_peak_near >= 4.0`；
- `abs(steer_delta_prepost) >= 0.25`；
- 同时满足工况分数、速度或低附着背景之一：`condition_score_peak >= 3.0`，或车速大于 80 km/h，或最低附着系数小于等于 0.8。

## 数量

- `FAST_STEER_REVIEW_UNCLEAR` / 原分组 `NORMAL_CONTROL_OR_EXCLUDE_LIGHT_STEER`：38 个。方向盘角速度较高，可能代表紧急操作；但原规则因车身动态弱、坐标风险或普通对照倾向未直接纳入。
- `FAST_STEER_TRAIN_LIKE` / 原分组 `NORMAL_CONTROL_OR_EXCLUDE_LIGHT_STEER`：23 个。方向盘角速度高，方向盘变化形成 return/countersteer 结构，建议优先人工复核是否可作为训练样本。
- `FAST_STEER_TRAIN_LIKE` / 原分组 `EXCLUDE_LIGHT_OR_COORD_RISK`：11 个。方向盘角速度高，方向盘变化形成 return/countersteer 结构，建议优先人工复核是否可作为训练样本。
- `FAST_STEER_REVIEW_UNCLEAR` / 原分组 `EXCLUDE_LIGHT_OR_COORD_RISK`：6 个。方向盘角速度较高，可能代表紧急操作；但原规则因车身动态弱、坐标风险或普通对照倾向未直接纳入。

## 结论

这批样本不应直接全部加入训练，但也不应继续放在纯排除类里。建议把 `FAST_STEER_TRAIN_LIKE` 作为第一批人工复核对象；人工确认后可以做“快速转向候选加入训练”的消融实验。

## 文件

- 候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\new_rule_fast_steer_candidates_v0_3.csv`
- 图片目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\fast_steer_review_v0_3`
- 图片说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\fast_steer_review_v0_3\00_先看这里_方向盘角速度候选说明.md`