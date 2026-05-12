# R2E-Steering 项目总进度看板

更新时间：2026-05-12 12:43:21

## 当前阶段

阶段 2 补充：事件锚点人工确认准备。

说明：阶段 3 已经做过一轮候选车辆基线诊断，但这些结果依赖 `raw_road_curvature_onset` 候选锚点。用户质疑“事件锚点是否已经确定”后，当前正式路线暂停继续训练，把阶段 3 结果降级为候选锚点诊断材料，不作为最终强车辆基线结论。下一步先通过人工标注确认事件从哪里到哪里，再生成 `manual_verified` 样本清单。

## 当前正在做什么

构建人工事件标注审查包：用原始车辆 CSV 重现每个记录的行驶过程参数，叠加三类候选事件位置，让用户在图上判断事件起止、预测锚点、方向和置信度。

## 已完成什么

- 阶段 0 旧流程冻结说明已生成。
- 新流程目录结构已建立。
- 长期目标已写入 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/R2E_STEERING_LONG_GOAL_CN.md`，作为后续新聊天继续执行的正式目标文本。
- 三个原始目录下被试名文件夹内 CSV 清单和哈希已生成。
- 原始车辆/生理/脑电深度审计表已生成。
- 阶段 1 用户查看版中文总结已生成。
- 阶段 0/1 完成审计清单已生成。
- 阶段 2 候选事件清单、样本清单、split 表、道路设计清单和数据版本卡已生成。
- 低泄漏道路曲率候选的处理后车辆窗口 v0.2 已生成，原始 CSV 未修改。
- 阶段 3 候选诊断已完成：无学习基线、v0.2 ridge、v0.3 无被试 ID 车辆基线、v0.4 RBF KRR 候选模型卡。当前这些结果只作为候选锚点诊断，不作为最终可发表结论。
- 人工事件标注审查包 v0.1 已生成：12 个原始车辆记录的多通道行驶过程图、HTML 审查页、人工标注 CSV 模板和中文说明。

## 正在运行什么任务

当前没有后台审计、处理或训练任务在运行。

## 服务器是否在运行

未使用服务器；未读取服务器指令与密码文件；未检查服务器状态；当前没有已知服务器后台任务。

## 最近一次结果

- 阶段 1 纳入审计 CSV：258；车辆/生理/脑电：91/82/85。
- 候选事件：11619，其中 old v400 6247、raw road curvature 359、raw vehicle dynamic 5013。
- 低泄漏道路曲率候选处理后车辆窗口：3 个 NPZ，样本数均为 359，特征数 9。
- 阶段 3 候选车辆模型曾显示：pre2 + session-level test 的 `rbf_krr_vehicle_no_subject` RMSE 0.382337、方向准确率 0.820896、错侧率 0.179104。但该结果依赖候选锚点，当前不能作为最终强车辆结论。
- 人工事件标注审查包 v0.1：生成 12 个记录、12 张整段车辆时间线图、1878 行人工标注模板。
- 人工标注包中每张图重现：道路曲率 `lanecurvatureXY`、方向盘转角 `SteeringWheel`、车速 `v_km/h`、横向位置 `lateraldistance`、横摆角速度 `vyaw`、横向加速度 `ay`、横滚角 `roll`。
- 图中叠加三类候选：蓝色 `raw_road_curvature_onset`、橙色 `old_v400_context_trigger_idx`、红色 `raw_vehicle_dynamic_onset`。

## 当前最大风险

事件锚点仍未人工确认。`raw_road_curvature_onset` 是较低泄漏候选，但不等于真实事件；`old_v400_context_trigger_idx` 是旧流程参考；`raw_vehicle_dynamic_onset` 来自车辆响应，存在把动作结果当事件触发的泄漏风险。必须先完成人工标注或道路设计对齐确认，才能进入正式样本 manifest 和强车辆基线。

## 下一步准备做什么

1. 用户先查看人工标注审查包 v0.1，判断图的参数和格式是否足够支持人工标注。
2. 如果格式可用，把审查包扩展到全部可用原始车辆记录。
3. 用户填写或批注人工事件起止、锚点、方向和置信度。
4. 根据人工标签生成 `manual_verified` 样本清单、版本卡和新的低泄漏处理后窗口。
5. 重新运行阶段 3 车辆基线；此前候选阶段 3 结果只作历史诊断对照。

## 用户可以优先查看哪些文件

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/review_index.html`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/manual_event_labels_template_v0_1.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/session_review_manifest_v0_1.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/figures`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_label_review_pack_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_user_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_anchor_rebuild_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/R2E_STEERING_LONG_GOAL_CN.md`
