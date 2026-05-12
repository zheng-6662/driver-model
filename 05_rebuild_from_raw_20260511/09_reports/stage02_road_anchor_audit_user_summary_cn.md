# 阶段 2 追加：道路事件位置与锚点审计，用户查看版

生成时间：2026-05-12 16:37:03

## 这一步做了什么

这一步没有训练模型，而是专门检查“事件锚点是不是可能有问题”。我把道路设计文件、原始车辆轨迹、旧 v400 事件锚点和当前道路引导失稳候选放到一起对齐。

## 当前发现

1. 道路设计文件可以整理出 16 个道路模块/实例，例如弯道、低附着路面、停车/特殊路段、连接段等。
2. 原始车辆记录可以投影到道路中心线，但可靠性不完全一致。部分记录/片段距离道路中心线较远，所以不能只靠道路模块名称直接定锚点。
3. 旧 v400 锚点不能直接当作最终真值。旧锚点中，只有 736 个在 1 秒内贴近非方向盘车身动态候选，169 个在 1 秒内贴近道路曲率候选，321 个在 1 秒内贴近道路模块边界。
4. 这支持你的怀疑：模型效果卡住，确实可能和样本锚点定义有关。

## 怎么理解

如果旧锚点偏晚，模型训练时看到的“事件后响应”其实已经发生了一部分，模型就容易学成趋势相似但幅值、方向和物理意义不稳定。

如果旧锚点偏早，标签窗口可能还没有覆盖真正响应，模型也会变得很难学。

所以接下来比继续堆模型更重要的是：用道路位置和车身姿态重新定义一批更可信的事件锚点。

## 你可以优先看哪些文件

1. 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\road_event_anchor_audit_v0_1_cn.md`
2. 道路模块位置图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\figures\road_event_position_map_v0_1.png`
3. 锚点审计概览图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\figures\road_anchor_audit_overview_v0_1.png`
4. 旧锚点对齐表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\tables\old_new_anchor_alignment_v0_1.csv`
5. 每条记录道路模块进入/离开时间：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\tables\session_module_entry_exit_v0_1.csv`
6. 代表样本图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\figures\representative_panels`

## 下一步建议

先不要训练。下一步应该用这张旧锚点对齐表，把旧样本分成：

- 锚点可信，可以保留；
- 旧锚点明显偏晚，需要重选；
- 旧锚点明显偏早，需要重选；
- 道路映射不可靠，只能人工复核或暂时不用。

然后再基于新的高可信锚点生成样本清单和强车辆基线。
