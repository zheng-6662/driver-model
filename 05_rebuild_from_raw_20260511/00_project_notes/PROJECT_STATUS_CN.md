# R2E-Steering 项目总进度看板

更新时间：2026-05-12 14:03:26

## 当前阶段

阶段 2 补充：事件锚点 Codex 自动审阅与少量人工复核准备。

说明：阶段 3 已经做过一轮候选车辆基线诊断，但这些结果依赖 `raw_road_curvature_onset` 候选锚点。用户质疑“事件锚点是否已经确定”后，当前正式路线暂停继续训练，把阶段 3 结果降级为候选锚点诊断材料，不作为最终强车辆基线结论。由于逐个播放人工标注成本太高，当前先由 Codex 对低泄漏道路曲率候选做自动审阅，输出高/中置信可采用标签和少量需复核标签。

## 当前正在做什么

整理 Codex 自动事件审阅 v0.1 结果：主线不再要求用户逐个标全部事件，而是用原始车辆、低泄漏道路曲率候选、方向盘响应和附近旧流程/车辆动态候选做第一轮自动审阅。用户后续只需要看低置信或冲突样本。

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
- 键盘式人工事件标注播放器 v0.1 已生成并启动，本地页面为 `http://127.0.0.1:8766/`。
- 键盘播放器已升级为候选段审查模式：长道路曲率段会拆成入弯和出弯/回正短窗口。
- 键盘播放器已补充竖线图例：明确区分粗蓝当前建议段、细蓝道路候选、橙色旧流程参考、红色车辆响应候选、黑色播放时间、紫色手动锚点和绿色已保存人工标签。
- Codex 自动事件审阅 v0.1 已完成：把 359 个道路曲率候选拆成 404 个候选事件段，并基于道路曲率、方向盘响应、横向加速度、车速、旧流程邻近点和车辆动态邻近点打分。

## 正在运行什么任务

当前有一个本地键盘标注服务在运行，作为低置信样本复核工具：

- URL：`http://127.0.0.1:8766/`
- 本地进程 PID：34408
- 作用：只服务本地网页和保存人工标签，不训练模型，不使用远程服务器。

## 服务器是否在运行

未使用远程服务器；未读取服务器指令与密码文件；未检查服务器状态；当前没有已知远程服务器后台任务。本地播放器服务不属于远程服务器任务。

## 最近一次结果

- 阶段 1 纳入审计 CSV：258；车辆/生理/脑电：91/82/85。
- 候选事件：11619，其中 old v400 6247、raw road curvature 359、raw vehicle dynamic 5013。
- 低泄漏道路曲率候选处理后车辆窗口：3 个 NPZ，样本数均为 359，特征数 9。
- 阶段 3 候选车辆模型曾显示：pre2 + session-level test 的 `rbf_krr_vehicle_no_subject` RMSE 0.382337、方向准确率 0.820896、错侧率 0.179104。但该结果依赖候选锚点，当前不能作为最终强车辆结论。
- 人工事件标注审查包 v0.1：生成 12 个记录、12 张整段车辆时间线图、1878 行人工标注模板。
- 键盘播放器接口验证通过：第一条记录 `rjy / 2025_09_28_20_02_20` 返回 7000 个时间点、7 个车辆信号、178 个候选事件。
- 候选段审查模式验证通过：该记录生成 7 个道路候选审查段；第一条长弯道已拆为 `road_001_entry` 入弯窗口 138.565-150.565 秒和 `road_002_exit` 出弯/回正窗口 278.710-290.710 秒。
- 键盘播放器标签输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv`。
- Codex 自动事件审阅：总标签 404；自动高置信采用 224；自动中置信采用 136；需要人工复核 43；证据不足建议剔除 1。
- 自动采用标签合计 360，需人工处理从几百个候选下降到 44 个。
- 自动事件角色：`curve_short` 314、`curve_entry` 45、`curve_exit_or_return` 45。
- 自动审阅输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_reviewed_event_labels_v0_1.csv`。

## 当前最大风险

事件锚点仍未人工最终确认。Codex 自动审阅可以显著减少人工负担，但它不是人工真值。`raw_road_curvature_onset` 是较低泄漏候选；`old_v400_context_trigger_idx` 和 `raw_vehicle_dynamic_onset` 只作为辅助证据，不能当无泄漏锚点。下一步必须把自动采用版本标成 `codex_auto_accepted`，不能冒充 `manual_verified`。

## 下一步准备做什么

1. 用 `codex_auto_accepted_event_labels_v0_1.csv` 生成候选数据版本卡，明确它是 Codex 自动审阅版本，不是人工真值。
2. 只抽查 `codex_needs_human_review_v0_1.csv` 中 44 个低置信/冲突样本。
3. 根据自动采用标签和少量复核标签生成下一版样本清单。
4. 重新生成处理后车辆窗口。
5. 在新样本上重新运行阶段 3 车辆基线；此前候选阶段 3 结果只作历史诊断对照。

## 用户可以优先查看哪些文件

- `http://127.0.0.1:8766/`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_keyboard_player_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/run_manual_event_keyboard_player.py`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/review_index.html`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_label_review_pack_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_user_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_anchor_rebuild_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/R2E_STEERING_LONG_GOAL_CN.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/codex_event_review_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_auto_accepted_event_labels_v0_1.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_needs_human_review_v0_1.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/figures/codex_event_review_score_overview_v0_1.png`
