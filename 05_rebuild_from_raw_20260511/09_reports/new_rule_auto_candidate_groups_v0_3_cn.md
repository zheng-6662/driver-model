# 新人工规则下的自动候选样本分组 v0.3

说明：这是把人工复核新规则转成的自动候选分组，不是最终人工标签。最终是否纳入训练仍以人工看图为准。

## 自动规则核心

- 轻微方向盘摆动、车身姿态/横向动态不明显：优先作为普通对照或排除，不进入极限主训练。
- 车身姿态/横向动态明显：即使方向盘幅度不大、没有明显回正，也保留为极限/近极限候选。
- 窗口不完整或坐标连续性异常但车身姿态很强：不直接丢弃，进入风险候选池，优先人工复核。

## 分组数量

- `RISK_POOL_BODY_STRONG`：614 个。车身姿态/横向动态强，但窗口或坐标连续性有风险，需要人工复核后决定是否抢救
- `NORMAL_CONTROL_OR_EXCLUDE_LIGHT_STEER`：381 个。缺少明显车身姿态/横向动态，更像普通驾驶、正常弯道或直线保持微调
- `KEEP_WEAK_CONSERVATIVE`：165 个。车身姿态/横向动态强，但方向盘幅度不大，符合保守/弱响应
- `MANUAL_REVIEW_BOUNDARY`：109 个。车身动态、方向盘响应或坐标风险边界不清，需要人工复核
- `KEEP_EXTREME_MAIN`：104 个。车身姿态/横向动态强，且方向盘响应明显
- `EXCLUDE_LIGHT_OR_COORD_RISK`：76 个。车身动态不强，且存在窗口或坐标风险，优先排除
- `KEEP_DELAYED`：65 个。车身姿态/横向动态强，但方向盘响应延迟或不明显
- `KEEP_WEAK_CONSERVATIVE`：32 个。中等车身动态，适合保守/弱响应样本
- `KEEP_DELAYED_WEAK`：28 个。中等车身动态，响应延迟或不明显，可保留为延迟/弱极限样本

## 两个截图对应的规则验证

- `V03_gf_2025_09_26_10_52_57_0004`：自动分到 `MANUAL_REVIEW_BOUNDARY`；理由：车身动态、方向盘响应或坐标风险边界不清，需要人工复核；原类别：排除样本；上下文：低附着；横滚/姿态简化分数=2.44。
- `V03_jy_2025_09_26_17_51_46_0002`：自动分到 `RISK_POOL_BODY_STRONG`；理由：车身姿态/横向动态强，但窗口或坐标连续性有风险，需要人工复核后决定是否抢救；原类别：排除样本；上下文：横滚/姿态；横滚/姿态简化分数=6.09。

## 文件

- 全量分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\new_rule_auto_candidate_groups_v0_3.csv`
- 分组数量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\new_rule_auto_candidate_group_summary_v0_3.csv`
- 每组代表样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\new_rule_auto_candidate_representatives_v0_3.csv`