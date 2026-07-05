# v229 两个月路线经验复盘与失败分类报告

- 生成时间：2026-06-23T14:48:24
- 范围：只读复盘 v220/v225/v228；不训练模型，不生成新预测，不重选 formal headline。
- 当前 formal lock：`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`。

## 一句话结论

这两个月的核心经验不是“模型还没堆够”，而是同一个瓶颈被多条路线反复证明：模型能较稳定抓住方向和普通响应，但强反应幅值、极端峰值、尾段、反转/多次修正仍是主要失败区；候选池经常存在更好上限，真正困难是当前可部署输入下的可靠选择。

## v228 最终正式效果

| pool | formal model | test n | RMSE | tail RMSE | direction acc | under rate | top20 tail share |
|---|---|---:|---:|---:|---:|---:|---:|
| loose_main_pool | avg_joint_focus | 184 | 0.544884 | 0.629752 | 0.967391 | 0.163043 | 0.659320 |
| strict_main_pool | peak_floor_090 | 174 | 0.571770 | 0.658306 | 0.948276 | 0.137931 | 0.672493 |

解读：方向准确率已经很高，但 top20 tail-SSE share 约 0.66-0.67，说明误差集中在少数难例，不能只用平均 RMSE 宣称问题已经解决。

## 两个月路线经验

| 阶段 | 记录数 | 沉淀经验 | v229 对下一步的含义 |
|---|---:|---|---|
| 早期端到端曲线预测与可复现性审计 | 97 | 完整曲线回归太容易被平均化和局部指标冲突拖住，必须把响应机制、强反应、尾段/反转单独拆出来审计。 | 不要再把单一完整曲线回归当作唯一主线；必须单独审计强反应、尾段和反转。 |
| 早期完整曲线/多候选/生理融合探索 | 12 | 候选池中常常有好答案，但当前输入下“选哪个候选”不稳；生理和脑电应先作为机制/状态信号审计，不宜直接塞进完整曲线主模型。 | 候选和外部信号有诊断价值，但下一步应先问选择是否可部署，而不是直接融合更多输入。 |
| 原始数据重筛、旧流程车辆-only与样本规则修正 | 85 | 模型卡住不只是模型问题，样本定义本身会制造假进展和假失败；必须回到事件和锚点定义。 | 模型失败需要先排除锚点、弱响应、道路干扰和样本口径污染。 |
| W2/W3窗口任务、多专家与路由器路线 | 16 | 不能再在同一套 current-window vehicle-only 特征上反复调 router；oracle 好不等于可部署模型好。 | oracle 好不等于 deployable router 好；同空间 current-window router 调参应保持关闭。 |
| Gold-V2锚点重审、人工筛选与低维反应目标 | 43 | 正确方向从“继续调模型”转为“把事件、锚点、反例、弱标签、低维反应拆开”。 | 优先守住事件、锚点、正反例和低维目标定义，避免回到旧数据口径。 |
| 事件门控、趋势标签与低维机制任务 | 39 | 机制标签可以成为辅助或结构约束，但论文目标不能退化成过于简单的二分类；需要继续保留曲线/车辆状态预测目标。 | 保留为追溯材料，不作为新实验解锁依据。 |
| 道路预瞄、高频输入、耦合Transformer与缓存数据集 | 27 | Transformer 作为编码器有用，但直接端到端输出曲线仍偏保守；需要显式处理强反应、峰值、反打和候选选择。 | 高频/道路/风格可作为诊断输入，但不能绕过强幅值低估问题。 |
| 滚动预测、锚点校准、反转/关键点/控制点路线 | 37 | 低维表示成立，但输入预测关键点/模板/机制到曲线的链条不稳定；应采用保守基线 + 候选曲线/小残差/收益门控，而不是硬切换。 | 关键点/控制点有上限价值，但输入到曲线的还原链条仍需按失败桶缩小范围。 |
| 机制优先、联合预测、候选曲线与残差修正 | 19 | 当前主线应回到“根据现有数据同时预测未来车辆状态和驾驶员行为”，模型上采用组合框架，而不是单一模型端到端硬压。 | 组合框架比继续堆大模型更稳；下一步应围绕失败分类决定是否扩展。 |
| 未归类/补充记录 | 10 | 如需正式写论文，可再人工筛掉重复看板类记录。 | 保留为追溯材料，不作为新实验解锁依据。 |

## test split 失败桶

| pool | failure bucket | n | share | avg tail RMSE | max tail RMSE | under rate | top20 tail rate |
|---|---|---:|---:|---:|---:|---:|---:|
| loose_main_pool | 极端峰值低估 | 1 | 0.005 | 1.641 | 1.641 | 1.000 | 1.000 |
| loose_main_pool | 强反应低估 | 15 | 0.082 | 1.090 | 1.650 | 1.000 | 0.867 |
| loose_main_pool | 极端峰值/尾段难例 | 5 | 0.027 | 1.037 | 1.348 | 0.000 | 0.800 |
| loose_main_pool | 强响应幅值/尾段 | 74 | 0.402 | 0.523 | 1.676 | 0.000 | 0.189 |
| loose_main_pool | 反转或多次修正 | 12 | 0.065 | 0.475 | 1.543 | 0.333 | 0.083 |
| loose_main_pool | 过零/换向边界 | 65 | 0.353 | 0.390 | 0.756 | 0.154 | 0.062 |
| loose_main_pool | 普通曲线可控 | 12 | 0.065 | 0.272 | 0.644 | 0.000 | 0.000 |
| strict_main_pool | 极端峰值低估 | 2 | 0.011 | 1.547 | 1.759 | 1.000 | 1.000 |
| strict_main_pool | 强反应低估 | 16 | 0.092 | 1.124 | 1.639 | 1.000 | 0.938 |
| strict_main_pool | 极端峰值/尾段难例 | 4 | 0.023 | 0.973 | 1.580 | 0.000 | 0.750 |
| strict_main_pool | 普通样本高尾误差 | 1 | 0.006 | 0.768 | 0.768 | 0.000 | 1.000 |
| strict_main_pool | 反转或多次修正 | 11 | 0.063 | 0.512 | 1.512 | 0.091 | 0.091 |
| strict_main_pool | 强响应幅值/尾段 | 71 | 0.408 | 0.505 | 2.012 | 0.000 | 0.127 |
| strict_main_pool | 过零/换向边界 | 60 | 0.345 | 0.401 | 0.853 | 0.083 | 0.067 |
| strict_main_pool | 普通曲线可控 | 9 | 0.052 | 0.276 | 0.533 | 0.000 | 0.000 |

## 最严重尾段案例的共同特征

| pool | sample | event | bucket | tail RMSE | under | peak ratio |
|---|---|---|---|---:|---|---:|
| strict_main_pool | rjy_Entity_Recording_2025_09_28_20_02_20_v108_010 | vehicle_strong | 强响应幅值/尾段 | 2.012 | False | 1.987 |
| strict_main_pool | rjy_Entity_Recording_2025_09_28_20_02_20_v108_041 | extreme_peak | 极端峰值低估 | 1.759 | True | 0.456 |
| strict_main_pool | cwh_Entity_Recording_2025_09_26_20_06_19_v108_017 | strong_event | 强响应幅值/尾段 | 1.755 | False | 1.803 |
| loose_main_pool | rjy_Entity_Recording_2025_09_28_19_51_44_v108_023 | strong_event | 强响应幅值/尾段 | 1.676 | False | 0.514 |
| loose_main_pool | tyy_Entity_Recording_2025_09_28_14_23_43_v108_026 | strong_event | 强反应低估 | 1.650 | True | 0.207 |
| loose_main_pool | rjy_Entity_Recording_2025_09_28_20_02_20_v108_031 | extreme_peak | 极端峰值低估 | 1.641 | True | 0.369 |
| strict_main_pool | rjy_Entity_Recording_2025_09_28_19_51_44_v108_023 | strong_event | 强反应低估 | 1.639 | True | 0.474 |
| strict_main_pool | tyy_Entity_Recording_2025_09_28_14_23_43_v108_004 | strong_event | 强反应低估 | 1.597 | True | 0.275 |
| strict_main_pool | rjy_Entity_Recording_2025_09_28_20_02_20_v108_040 | extreme_peak | 极端峰值/尾段难例 | 1.580 | False | 0.525 |
| loose_main_pool | tyy_Entity_Recording_2025_09_28_14_23_43_v108_014 | strong_event | 强反应低估 | 1.571 | True | 0.211 |
| loose_main_pool | rjy_Entity_Recording_2025_09_28_19_51_44_v108_039 | reverse | 反转或多次修正 | 1.543 | False | 1.080 |
| strict_main_pool | tyy_Entity_Recording_2025_09_28_14_23_43_v108_014 | strong_event | 强反应低估 | 1.523 | True | 0.273 |

这些最坏案例主要落在极端峰值、强事件和少量反转/车辆强响应上。它们解释了为什么普通曲线看起来还可以，但一看预测图就会出现“经典幅值压平/尾段跟不上”的感觉。

## selector / candidate 诊断边界

- `combined / future_route_decision`：v222b_allowed=False; v223_allowed=False。当前证据没有解锁 v222b/v223；下一步应先让 GPTPro 审阅路线复盘和失败桶。
- `loose_main_pool / future_route_decision`：v222b_allowed=False; v223_allowed=False。当前证据没有解锁 v222b/v223；下一步应先让 GPTPro 审阅路线复盘和失败桶。
- `loose_main_pool / oracle_safe_gate`：upper-bound diagnostic-only。oracle 只能作为上限诊断，不能写入 formal headline 或可部署结论。
- `loose_main_pool / oracle_safe_gate`：selector_failed_rate=0.4076086956521739; candidate_missing_rate=0.0271739130434782。候选池常有上限，但 learned selector 在 locked test 上不稳；这支持先做失败分类，而不是直接训练更大 gate/router。
- `strict_main_pool / future_route_decision`：v222b_allowed=False; v223_allowed=False。当前证据没有解锁 v222b/v223；下一步应先让 GPTPro 审阅路线复盘和失败桶。
- `strict_main_pool / oracle_safe_gate`：upper-bound diagnostic-only。oracle 只能作为上限诊断，不能写入 formal headline 或可部署结论。
- `strict_main_pool / oracle_safe_gate`：selector_failed_rate=0.4137931034482758; candidate_missing_rate=0.028735632183908。候选池常有上限，但 learned selector 在 locked test 上不稳；这支持先做失败分类，而不是直接训练更大 gate/router。

这部分必须保持 diagnostic-only。oracle 或 selector gap 可以帮助解释路线，但不能写成正式可部署提升，也不能据此直接解锁 v222b/v223。

## v228 limitation 与论文边界

- `loose_main_pool_tail_concentration`：证据 `top20pct tail share=0.659320, tail gini=0.612677`；影响：尾部误差不是均匀分布，论文中需要呈现集中度和代表性失败样本。
- `loose_main_pool_underestimation`：证据 `under_rate=0.163043`；影响：仍存在一定比例低估，尤其需要结合 underestimation profile 表解释。
- `loose_main_pool_sample_size`：证据 `test n=184`；影响：test 样本量有限，subject-block CI 应优先作为跨被试稳健性提示。
- `strict_main_pool_tail_concentration`：证据 `top20pct tail share=0.672493, tail gini=0.630911`；影响：尾部误差不是均匀分布，论文中需要呈现集中度和代表性失败样本。
- `strict_main_pool_underestimation`：证据 `under_rate=0.137931`；影响：仍存在一定比例低估，尤其需要结合 underestimation profile 表解释。
- `strict_main_pool_sample_size`：证据 `test n=174`；影响：test 样本量有限，subject-block CI 应优先作为跨被试稳健性提示。

## 下一步决策矩阵

| 候选动作 | 决策 | 原因 | 重开条件 |
|---|---|---|---|
| 直接训练 v222b/v223 或更大 gate/router | 不建议 | v222a closeout 已显示 learned selector 在 locked test 不稳；继续扩大同类 selector 容易陷入局部过拟合。 | 除非 GPTPro 给出新的 bounded scope，且失败分类证明 candidate_missing 是主因而非 selector_failed。 |
| 新增 tau/threshold 或基于 test 重新调 headline | 禁止 | 违反当前 formal lock 和 test discipline；v228 已冻结最终主结果。 | 不应重开。 |
| 继续写作/论文材料整理 | 可行 | v228 已提供正式主表、CI、claim 和 limitation；结果边界清楚。 | 保留 tail concentration、underestimation、样本量和 subject-block CI 限制说明。 |
| 失败样本 taxonomy + 人工复核少量高尾案例 | 推荐 | 两个月经验表明必须先区分 candidate_missing、selector_failed、强峰值低估、反转/多修正和 input-indeterminate。 | 先产出每类占比和代表图，再决定是否允许一个窄范围机制实验。 |
| 让 GPTPro 审阅 v229 复盘后给一个 bounded 下一步 | 推荐 | 可避免 GPTPro 只基于最新指标给出局部调参建议，并把讨论约束到路线经验和失败分类。 | 发送中文 prompt，要求明确 stop condition、是否只写作、是否允许失败分类审计。 |

## 给 GPTPro 的建议提问方式

不要只问“下一步训练什么模型”。应把 v229 报告发给 GPTPro，请它先确认：当前是否进入写作整理；如果继续实验，是否只允许失败样本 taxonomy/人工复核；是否明确禁止 v222b/v223、大 gate/router、新 tau/threshold 和 test-based retuning；如果允许新实验，必须给出单一窄范围目标和 stop condition。
