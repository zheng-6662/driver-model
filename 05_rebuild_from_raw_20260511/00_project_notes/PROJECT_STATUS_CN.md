# R2E-Steering 项目总进度看板

## 最新更新：2026-05-12 15:52

- 当前阶段：阶段 2 修正，正在从“人工逐条标注失稳候选”转为“道路设定引导的自动综合判定”。
- 当前正在做什么：根据旧项目日志中的 `*_events_v400_context.csv` 事件逻辑、道路中心线模块顺序和原始车辆动态证据，自动判定车辆失稳事件。
- 已完成什么：新增 `road_guided_vehicle_instability_v0_1` 判定版本；全量失稳候选 1227 个，自动/已确认采用 701 个，中间复核 177 个，低证据剔除 349 个。
- 正在运行什么任务：没有远程任务；本地 8766 审查页面仍可作为抽查工具，但当前不要求用户逐条标注。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件。
- 最近一次结果：`road_guided_auto_accepted_events_v0_1.csv` 已生成，可作为下一步车辆失稳样本 manifest 的主输入。
- 当前最大风险：道路中心线最近点映射有 526 个候选为 `very_low` 可靠度，因此道路模块名只能作为弱先验，不能单独证明失稳；最终仍以 `ay/roll_rate/yaw/lateral` 等非方向盘车辆动态证据和旧 v400 近邻上下文为主。
- 下一步准备做什么：基于道路设定引导后的 701 个采用候选生成车辆失稳版 `samples_master`、split 表和处理后车辆窗口，然后重新做无学习/纯车辆基线。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/road_guided_instability_v0_1_cn.md`，以及 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_auto_accepted_events_v0_1.csv`。

更新时间：2026-05-12 14:55:00

## 当前阶段

阶段 2 修正：事件锚点与样本清单重建，当前主线已从“弯道/道路曲率事件”切换为“车辆失稳事件”。

重要修正：

- 之前的 `codex_event_review_v0_1` 产出的 404 个候选，本质是弯道/道路曲率样本，不是用户真正要的车辆失稳样本。
- 这 404 个候选已经降级为道路上下文参考，不再作为主事件样本，也不再用于继续训练正式车辆模型。
- 当前主事件候选改为 `vehicle_instability_onset_codex_v0_1`，锚点来自非方向盘车辆动态异常：`ay` 和 `roll_rate`。
- `steer_rate` 不再作为失稳事件锚点，因为它已经是驾驶员方向盘动作，不能用来定义“失稳开始”；方向盘只作为事件后的响应/标签证据。

## 当前正在做什么

正在整理车辆失稳候选事件版本 v0.1，并把本地浏览器审查工具切换到失稳候选：

- 读取阶段 2 的 `candidate_events_master.csv`。
- 从 `raw_vehicle_dynamic_onset` 中只保留 `ay` 和 `roll_rate` 作为非方向盘动态种子。
- 将相邻动态种子合并为车辆失稳候选片段。
- 计算每段的横向加速度、横滚速率、横摆角速度、横向偏移、车速、事件后方向盘修正幅度等证据。
- 输出全量候选、自动采用候选、需复核候选和概览图。
- 本地页面 `http://127.0.0.1:8766/` 已改为默认读取车辆失稳候选，而不是弯道候选。

## 已完成什么

- 阶段 0：旧流程冻结与重建准则已完成。
- 阶段 1：原始车辆/生理/脑电 CSV 审计已完成；只纳入 `原始车辆数据`、`原始脑电数据`、`原始生理数据` 三个原始目录下被试名文件夹里的 CSV；原始 CSV 未被修改。
- 阶段 2 初版：候选事件总表、样本清单、split 表、道路设计清单和低泄漏道路曲率车辆窗口已生成。
- 阶段 3 诊断版：曾在道路曲率候选上运行过无学习/车辆基线，但现在这些结果只作为诊断材料，不作为正式强车辆基线结论。
- 人工标注播放器已建立，并可在本地用键盘审查候选片段。
- Codex 弯道自动审阅 v0.1 已完成，但已降级为道路上下文参考。
- 新增车辆失稳自动审阅 v0.1：
  - 非方向盘动态种子：1833 个。
  - 合并后车辆失稳候选片段：1227 个。
  - 自动高/中置信采用：358 个。
  - 需要人工复核：462 个。
  - 低失稳证据建议剔除：407 个。
  - 覆盖被试：18 个。

## 正在运行什么任务

当前有一个本地审查服务正在运行：

- URL：`http://127.0.0.1:8766/`
- 本地进程 PID：33512
- 模式：`vehicle_instability_event_reviewer_v0_1`
- 作用：展示车辆失稳候选片段，支持人工复核和键盘标注。
- 标签输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_instability_event_labels_v0_1.csv`

该服务只读原始车辆 CSV 和候选事件表，不训练模型，不读取服务器密码，不修改原始 CSV。

## 服务器是否在运行

未使用远程服务器；未读取服务器指令与密码文件；未检查或启动 AutoDL；当前没有已知远程后台任务。

本地 8766 页面是本机 HTTP 服务，不属于远程服务器任务。

## 最近一次结果是什么

车辆失稳候选 v0.1 已生成：

- 全量表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`
- 自动采用表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_auto_accepted_events_v0_1.csv`
- 需复核表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_needs_human_review_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_event_review_summary_v0_1.csv`
- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/instability_event_review_v0_1_cn.md`
- 概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/instability_event_score_overview_v0_1.png`

按决策统计：

```text
needs_human_review                 462
reject_low_instability_evidence    407
auto_accept_instability_medium     224
auto_accept_instability_high       134
```

按失稳证据类型统计：

```text
instability_ay_only      1150
instability_roll_only      65
instability_ay_roll        12
```

## 当前最大风险

1. 当前失稳锚点来自车辆动态异常，因此任务定义是“检测到车辆失稳动态开始后，预测未来方向盘响应”，不是“失稳发生前预警”。后续样本版本卡必须明确这个因果设定。
2. 高横向加速度可能来自正常过弯，也可能来自车辆失稳；需要结合横滚、横摆、横向偏移、事件后方向盘修正和道路上下文进一步区分。
3. 358 个自动采用样本是 Codex 规则筛选，不是人工真值；只能命名为 `codex_auto_accepted` 或 `vehicle_instability_onset_codex_v0_1`，不能冒充 `manual_verified`。
4. 之前道路曲率上的阶段 3 模型不能继续当正式基线；必须在失稳样本 manifest 确认后重新生成处理窗口和基线。
5. 5000 多个 `raw_vehicle_dynamic_onset` 不能直接当失稳样本数量，因为里面包含大量 `steer_rate`，这属于驾驶员动作结果，不适合定义失稳开始。

## 下一步准备做什么

1. 抽查 `auto_accept_instability_high` 和 `needs_human_review` 的示例图，确认规则是否把正常过弯误判为失稳。
2. 为 `vehicle_instability_onset_codex_v0_1` 写数据版本卡，明确样本定义、排除规则、因果设定和泄漏边界。
3. 生成车辆失稳版本的处理后车辆窗口，窗口必须以失稳锚点为 0 秒，方向盘只作为未来响应标签。
4. 在生成正式 manifest 前，不继续训练新模型。
5. 如果需要人工复核，优先只复核 462 个 `needs_human_review` 和少量高置信抽样，不要求用户逐个看 1227 个。

## 用户可以优先查看哪些文件

1. 车辆失稳说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/instability_event_review_v0_1_cn.md`
2. 车辆失稳数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_v0_1_cn.md`
3. 全量失稳候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`
4. 自动采用候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_auto_accepted_events_v0_1.csv`
5. 需复核候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_needs_human_review_v0_1.csv`
6. 概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/instability_event_score_overview_v0_1.png`
7. 本地审查页面：`http://127.0.0.1:8766/`
8. 当前长期目标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/R2E_STEERING_LONG_GOAL_CN.md`
