# GPTPro 事件锚点审查回复归档

来源：用户在当前对话中手动粘贴 GPTPro 回复。

归档时间：2026-05-12

## 核心判断

GPTPro 判断：当前暂停盲目改模型、转向重审事件锚点和样本清单是正确且必要的。旧模型中出现的“趋势像但物理意义不对”“真实大幅打方向却预测成轻微变化”“真值和预测在零线两侧”等问题，更像是锚点错位、事件语义混合、左右方向标签混合、响应强弱样本混合造成的，而不只是网络结构问题。

## 对当前方向的评价

1. 事件锚点重建方向正确，应继续推进到 v0.6 后再恢复训练。
2. 当前把事件拆成“场景设计点、显式触发点、车辆响应确认点、方向盘预测标签”四层是合理的。
3. 建议增加一层“被试相关/暴露点”，形成：
   `场景设计点 -> 显式触发点 -> 被试相关/暴露点 -> 车辆响应开始点 -> 响应峰值点 -> 方向盘预测标签`。
4. 方向盘本身不能反推事件开始，只能用于响应确认、响应强弱、峰值时间和预测标签。

## 指出的主要风险

1. v0.5 的高置信复核不能等同最终训练样本。
2. post-window 方向盘指标适合审查，不适合最终纳入规则。
3. 旧锚点接近程度只能作为解释字段，不能作为强评分核心。
4. `middle_section` 是连续负荷段，不应混入单点突发事件池。
5. `longstraight` 显式变道高置信少，不等于触发无效，可能是被试相关性、暴露时刻、窗口长度或响应指标不合适。
6. `differentmu_road` 要区分 raw `mu` 与 cfg 映射，优先使用被试车实际 `mu` 跳变/首次低附着。
7. 弯道入口不应简单等于曲率开始，应细化为曲率开始点、有效曲率阈值点、响应开始点和峰值点。
8. 横向偏移跳变要单独排查，避免把坐标/道路模块切换误判为真实车辆动作。

## v0.6 建议输出

GPTPro 建议 v0.6 不只输出一个训练样本表，而是输出四张表：

1. `primary_training_events_v0_6`：可进入第一版训练的强样本。
2. `manual_review_events_v0_6`：需要人工确认的样本。
3. `response_confirm_only_v0_6`：只能作为响应确认点，不能作为事件锚点。
4. `holdout_or_excluded_v0_6`：暂缓、排除或语义不清样本。

## v0.6 每个样本应包含的关键字段

- `event_type`
- `t_design`
- `t_trigger`
- `t_ego_exposure`
- `t_response_onset`
- `t_response_peak`
- `t_train_anchor`
- `anchor_source`
- `ego_relevance`
- `pre_window_clean`
- `post_response_confirmed`
- `steer_label_quality`
- `coordinate_continuity_ok`
- `confidence_tier`
- `review_status`

## 建议的样本分级

- S 级：第一版训练核心样本。要求外生设计证据明确、被试相关性明确、t0 前没有同一事件响应、t0 后有合理响应、方向盘标签连续、窗口完整、坐标连续、左右/曲率方向可解释。
- A 级：可训练但保留复核标记。允许轻微响应延迟、旧锚点不一致、车辆响应中等或被试相关性需确认。
- B 级：弱响应/负样本/对照样本。可用于是否产生明显转向响应的分类头或 no-response/control。
- C 级：响应确认-only。横摆角速度峰值、横向加速度峰值、方向盘速率峰值等只能确认响应，不能作为因果锚点。
- D 级：排除或暂缓。包括只有旧锚点、只有方向盘峰值、t0 前已响应、窗口不完整、坐标跳变、场景语义不清、`curve3/zd` 覆盖不足、`stop` 缺少前车触发/TTC 证据等。

## 分场景建议

1. `curve1/curve2`：第一批进入，但锚点要从模块入口细化为曲率开始/有效曲率点。
2. `differentmu_road`：第一批进入，但优先 raw `mu` 跳变/首次低附着。
3. `fix_road` 显式变道：第一批或 1.5 批进入，必须人工复核 timing。
4. `middle_section`：暂不进入第一版单点事件模型，应作为连续任务集或 episode 单独处理。
5. `longstraight` 显式变道：暂缓为诊断/小样本复核集，需要先查被试相关性、目标车侵入时刻和响应窗口。
6. `longstraight` 显式停车：单独作为纵向/避让事件，不建议先放入 steering 主训练集。
7. `stop`：暂缓。
8. `curve3/zd`：暂缓。

## 建议第一版可靠样本集

第一版可靠样本集建议采用：

`curve1/curve2 几何锚点 + differentmu raw μ 锚点 + 人工通过的 fix_road 显式变道样本`

不要一开始把 `middle_section`、`longstraight stop`、`stop`、`zd` 混进去。

## 对模型训练顺序的建议

GPTPro 建议：应先训练新锚点下的车辆/道路-only 强基线，暂不加入连续驾驶风格和生理数据。

原因：如果锚点不准，风格/生理数据即使提升指标，也可能只是补偿样本错位、个体差异或噪声，而不是提供真实机制增量。

车辆/道路-only baseline 应覆盖方向符号、峰值幅值、time-to-peak、大幅转向召回、no-response false positive、分事件类型误差等物理指标，而不只看 RMSE。

## 后续路线

1. 冻结 v0.5，不直接训练。
2. 先人工复核 56 张代表图，标注 `pass / early / late / weak_response / continuous / coordinate_issue / unclear / exclude`。
3. 生成 v0.6 样本清单，保留多时间字段：`t_design / t_trigger / t_ego_exposure / t_response_onset / t_response_peak / t_train_anchor`。
4. 第一版训练只用核心场景：`curve1/curve2`、`differentmu_road` raw μ、人工通过的 `fix_road`。
5. 训练车辆/道路-only baseline。
6. 如果 baseline 仍然有零线错侧、幅值不足或时间错位，先继续查左右方向、曲率方向、车道左右关系、方向盘符号、横向偏移坐标跳变和 t0 偏移。
7. 核心事件 baseline 站稳后，再扩展 `middle_section` 连续任务和风格/生理增量验证。

## 最终压缩结论

v0.6 的目标不是“多拿样本”，而是先拿一批因果语义清楚、锚点不混、方向符号可信、响应窗口完整的样本。第一版模型宁可小而干净，也不要大而混杂。
