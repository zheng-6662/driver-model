# R2E-Steering 项目总进度看板

## 最新更新：2026-05-12 19:05

- 当前阶段：阶段 2 收口，已把全原始重筛的高置信车辆失稳事件整理成新流程正式样本清单 `vehicle_instability_highconf_v0_1`。
- 当前正在做什么：样本清单、split、排除原因、模态可用性和数据版本卡已生成；准备进入新流程阶段 3 的无学习基线和强车辆基线。
- 已完成什么：908 个高置信失稳事件中 906 个满足完整窗口要求，2 个因 3 秒历史窗口不足被排除；906 个事件各生成 3 个窗口，共 2718 行样本。主窗口 `pre2_label2_old_main` 的 session-level split 为 train 611、val 156、test 139。
- 正在运行什么任务：没有训练任务；没有服务器任务；本次只做本地表格/manifest 构建。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件。
- 最近一次结果：`samples_master.csv` 已包含原始车辆文件、sha256、原始时间戳、锚点、输入窗口、标签窗口、split、模态可用性、泄漏标签和 eval-only 响应类型字段。主窗口模态可用性为车辆 906、生理 815、脑电 846、三模态齐全 755。
- 当前最大风险：生理/脑电并非所有样本都可用，后续做生理或 EEG 增量时必须用可比子集和置乱/错位对照；`eval_label_*` 字段来自未来标签，只能用于评估分层，不能作为训练输入、split 或标准化依据。
- 下一步准备做什么：基于正式 `samples_master` 运行新流程无学习基线和强车辆基线；先不进入风格/生理有效性结论。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_vehicle_instability_highconf_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_highconf_v0_1_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`。

## 最新更新：2026-05-12 18:45

- 当前阶段：阶段 3 旧流程历史对照，已完成旧 `vehicle_direct` 全量车辆-only clean run；这仍是旧代码对照，不是新流程最终强车辆基线。
- 当前正在做什么：本地训练和评估已经结束，正在整理产物、日志和 Git 提交。
- 已完成什么：为旧深度入口生成 clean vehicle manifest；84 个原始车辆文件完成 200Hz 插值清洗，906 个高置信失稳样本可用于旧代码；session-level split 为 train/val/test = 611/156/139；旧 `vehicle_direct` 全量 CPU run 已完成；固定预测图和坏样本图已生成。
- 正在运行什么任务：没有本地训练任务；没有远程服务器任务。`http://127.0.0.1:8766/` 仍只是本地审查页面，不是 GPU/服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：clean 版旧 `vehicle_direct` active checkpoint 在 session-level test 上 RMSE=0.637366，主峰方向准确率=0.870504，错侧率=0.129496，严重幅值不足率=0.683453，大幅响应召回=0.142857，反向修正计数完全匹配率=0.086331。structure checkpoint RMSE=0.647720，严重幅值不足率=0.561151。
- 当前最大风险：旧深度入口直接读取原始 CSV 时会把原始交替缺失点填 0，已导致一次 raw direct run 被判定无效并清理；后续旧代码只允许使用 clean manifest。即便 clean run 的 RMSE 比旧 ridge 诊断更低，物理指标仍显示幅值不足、复杂修正识别弱，不能直接升级为新流程强车辆基线。
- 下一步准备做什么：用 906 个高置信失稳事件构建新流程正式 `samples_master`/split/dataset card，并建立不含驾驶员 ID、无泄漏、物理指标齐全的强车辆基线；同时把这次旧 deep 的坏样本作为失败样本库。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_oldcode_vehicle_direct_full_clean_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_direct_full_clean_on_instability_v0_1_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_bad_samples_test.png`。

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

## 最新更新：2026-05-12 16:25

- 当前阶段：阶段 2 修正，全量原始车辆 CSV 失稳样本重筛。
- 当前正在做什么：已按“非方向盘车辆动态 + 旧 v400 事件上下文 + 道路设定先验”的标准，从 `原始车辆数据/<被试名>/*.csv` 直接重筛全部原始车辆文件。
- 已完成什么：91 个原始车辆 CSV 全部可读；检测到 4581 个非方向盘动态种子，合并为 1991 个车辆失稳候选；高置信主清单 908 个，自动/已确认采用扩展清单 1348 个，中间复核 269 个，低证据剔除 374 个。
- 正在运行什么任务：没有远程任务；本地 8766 页面仍可用于抽查，但当前不要求用户逐条标注。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件。
- 最近一次结果：`all_raw_vehicle_instability_primary_high_confidence_v0_1.csv` 和 `all_raw_vehicle_instability_auto_accepted_v0_1.csv` 已生成。
- 当前最大风险：全量重筛使用更直接的 `ay/roll_rate` 种子规则，比旧候选表更宽；正式 manifest 建议先用 908 个高置信主清单，1348 个扩展采用清单作为补充/敏感性分析。
- 下一步准备做什么：用高置信主清单生成车辆失稳版 `samples_master`、split 表和处理后车辆窗口，随后重新做车辆基线。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/all_raw_vehicle_instability_rescreen_v0_1_cn.md` 和 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_primary_high_confidence_v0_1.csv`。

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

## 追加更新：2026-05-12 16:35

- 当前阶段：阶段 2 追加审计，正在确认道路事件位置与旧/新锚点对齐关系。
- 当前正在做什么：不训练模型，先检查道路设计位置、原始车辆轨迹投影、旧 v400 锚点、道路曲率候选、非方向盘车身动态候选和道路引导失稳候选之间的时间关系。
- 已完成什么：已生成 `road_event_anchor_audit_v0_1` 审计包。道路模块/实例 16 个；91 条原始车辆记录全部可投影；道路模块经过片段 890 个；旧 v400 锚点 6247 个完成对齐。
- 最近一次结果：旧 v400 锚点中，1 秒内贴近非方向盘车身动态候选 736 个，贴近道路曲率候选 169 个，贴近道路模块边界 321 个。大量旧锚点被分到“可能早于车身响应”或“可能晚于车身响应”两类，说明旧锚点不能直接作为最终真值。
- 当前最大风险：道路中心线映射质量不均衡，片段级可靠性中 `low/very_low` 占比不低，因此道路模块名称不能单独定义锚点，必须和车身姿态共同使用。
- 下一步准备做什么：先查看审计图和对齐表，再把旧样本分为“可保留、偏早、偏晚、道路映射不可靠”四类；之后再生成高可信新锚点 manifest 和强车辆基线。
- 用户可以优先查看：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_road_anchor_audit_user_summary_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/road_event_anchor_audit_v0_1_cn.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/figures/road_event_position_map_v0_1.png`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/figures/road_anchor_audit_overview_v0_1.png`
  5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/old_new_anchor_alignment_v0_1.csv`

## 追加更新：2026-05-12 17:05

- 当前阶段：阶段 2/3 交界，使用全原始车辆失稳高置信样本测试旧车辆代码。
- 当前正在做什么：已把 `all_raw_vehicle_instability_primary_high_confidence_v0_1.csv` 转成旧阶段 3 诊断 `.npz` 窗口和旧深度模型 `sample_manifest`，并完成旧车辆基线诊断和旧 `vehicle_direct` smoke run。
- 已完成什么：908 个高置信失稳事件中 906 个满足旧代码 3 秒历史 + 2 秒未来要求，2 个因锚点太靠前被排除；生成 2718 行窗口样本；旧深度模型 loader smoke 保留 12/12 样本；旧 `vehicle_direct` smoke 训练闭环成功。
- 正在运行什么任务：没有正在运行的训练任务；本次旧深度模型 smoke 已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件。
- 最近一次结果：pre2 + session-level test 中，旧 `ridge_vehicle_summary` RMSE=0.675055，错侧率=0.273381，严重幅值不足率=0.654676；去掉被试 one-hot 的 `ridge_vehicle_no_subject` RMSE=0.675174，错侧率=0.280576。旧 `vehicle_direct` smoke run 仅用 96/32/32 子集跑通，test steer RMSE=0.400123，但不是正式性能结论。
- 当前最大风险：旧代码诊断仍可能继承旧模型结构和旧评价偏好；smoke 子集太小，不能证明旧深度模型有效，也不能支持风格/生理有效。
- 下一步准备做什么：如果继续使用旧代码，应先跑全量车辆-only 旧模型并生成固定预测图/坏样本图；更稳妥的正式路线仍是把 906 个可用高置信失稳事件整理成新流程 manifest，再建立强车辆基线。
- 用户可以优先查看：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_oldcode_instability_user_summary_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_baseline_on_instability_v0_1_cn.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_direct_smoke_on_instability_v0_1_cn.md`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_baseline_metrics.csv`
  5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_session_level_split.csv`


## 追加更新：2026-05-12 17:18

- 当前阶段：阶段 2 补充审计，已从道路模块位置审计推进到“场景交通对象与触发点审计”。
- 当前完成内容：新增 `scene_trigger_audit_v0_2`，直接解析 SILAB `.aed` 场景文件，提取交通对象、激活点、停车点、换道点，并换算到道路纵向位置和每条被试记录的相对时间轴。
- 最新结果：解析到交通对象 81 行、场景触发点 19 行；场景触发点换算到被试记录时间轴后 1436 行；6247 个旧 v400 锚点已和最近场景触发点完成对齐。
- longstraight 关键发现：25/26 车道附近有交通交互设计。26 车道有车流源；25 车道有 MAN TGL 货车和 Chrysler300 小轿车；Chrysler300 有停车触发；MAN TGL 有向 26 车道换道触发。
- 旧锚点风险：旧锚点 1 秒内接近场景触发点 175 个，2 秒内接近 356 个；大量旧锚点明显早于或晚于最近场景触发点，说明旧锚点不能直接等同于真实场景触发时刻。
- 下一步：做被试车道投影，确认被试通过 longstraight 触发点时到底在 25/26 哪条车道，以及和交通车的相对位置。
- 用户优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_scene_trigger_user_summary_cn.md`；`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/figures/longstraight_scene_trigger_map_v0_2.png`。
# 最新更新：2026-05-12 17:15

- 当前阶段：阶段 2/3 交界，正在用旧流程 `vehicle_direct` 入口对全原始重筛后的高置信失稳样本做全量车辆-only 对照。
- 当前正在做什么：启动旧深度模型全量训练，不使用生理、脑电、连续风格或驾驶员风格向量；输入为 `vehicle_instability_allraw_highconf_v0_1` 的 session-level 旧 manifest。
- 已完成什么：旧代码窗口和 manifest 已生成，906/908 个高置信失稳事件满足旧代码 3 秒历史 + 2 秒未来窗口要求；旧 ridge/无学习诊断和旧 `vehicle_direct` smoke run 已跑通。
- 正在运行什么任务：本地 CPU 全量旧 `vehicle_direct` 训练评估，run prefix 为 `OLD_FULL_INSTABILITY_HIGHCONF_VEHICLE_DIRECT_V0_1`。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件。
- 最近一次结果是什么：smoke run 只证明旧入口可运行；正式对照需要本次全量 run + 固定图/坏样本图后再判断。
- 当前最大风险是什么：旧深度模型结构和旧评价偏好可能不适合作为新流程结论；本次只作为旧代码历史对照，不能替代新流程强车辆基线。
- 下一步准备做什么：训练结束后读取 `run_summary.json`，补逐样本评估表、固定预测图、坏样本图、中文报告，并更新产物索引。
- 用户可以优先查看哪些文件：训练完成后优先看 `09_reports/oldcode_vehicle_direct_full_on_instability_v0_1_cn.md` 和对应固定图/坏样本图。


## 追加更新：2026-05-12 17:25

- 已补充 `longstraight` 被试车道投影 v0.2：在 595 个场景触发点时刻，被试车估计车道为 23 车道 524 行、22 车道 68 行、21 车道 3 行。
- 关键判断：这些估计结果与 `longstraight` 中 25/26 车道的交通触发点不是同车道，也不是同方向侧；因此不能简单说“被试开的车道上有这些车”。更准确的说法是：被试多在 21/22/23 侧行驶，场景触发交通车主要在 25/26 侧，可能构成对向/相邻方向侧交通干扰，需要结合具体场景设计解释。
- 新增图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/figures/longstraight_ego_lane_projection_v0_2.png`。
# 最新更新：2026-05-12 18:40

- 当前阶段：阶段 3 旧代码历史对照，已完成 clean manifest 版旧 `vehicle_direct` 全量车辆-only run。
- 当前正在做什么：整理本次旧代码对照结果并准备提交；没有训练任务正在运行。
- 已完成什么：发现旧深度入口直接读原始 CSV 会把交替缺失点填 0，导致标签高频跳变；已生成 `oldcode_deep_vehicle_csv_v0_1` clean 车辆 CSV 和 `oldcode_manifest_session_level_split_clean_vehicle_v0_1.csv`，并用 clean manifest 重跑全量 `vehicle_direct`。
- 正在运行什么任务：无。clean full run 已结束，评估和固定图/坏样本图已生成。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件。
- 最近一次结果是什么：clean run 的 active legacy checkpoint 为 epoch 5，session-level test RMSE=0.637366，主峰方向准确率=0.870504，错侧率=0.129496，严重幅值不足率=0.683453，大幅响应召回=0.142857；structure checkpoint 为 epoch 9，test RMSE=0.647720。
- 当前最大风险是什么：旧 `vehicle_direct` 在 clean 数据上仍明显幅值不足、反向/多段修正识别差，且只属于旧代码历史对照，不能作为新流程最终强车辆基线，也不能支撑风格/生理有效性结论。
- 下一步准备做什么：把 906 个高置信失稳事件整理成新流程正式 `samples_master` 和强车辆基线；本次旧代码结果只作为历史对照和坏样本来源。
- 用户可以优先查看哪些文件：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_oldcode_vehicle_direct_full_clean_user_summary_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_direct_full_clean_on_instability_v0_1_cn.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_fixed_predictions_test.png`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_bad_samples_test.png`
