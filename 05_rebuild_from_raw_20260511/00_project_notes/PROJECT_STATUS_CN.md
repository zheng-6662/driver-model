# 项目状态更新：v0.5 服务器处理后样本重筛 + 被试划分旧流程车辆-only

更新时间：2026-05-20 12:15

当前阶段：旧流程车辆-only 基线复查。按用户要求，本轮使用服务器中已经处理好的车辆 CSV 重新筛选样本，并采用被试划分训练/验证/测试。

当前完成：服务器 4080 SUPER 已完成 v0.5 重筛、manifest 构建、旧流程 loader 检查、FAIR09/E1 车辆-only 粗细双头训练、预测图生成和结果拉回本地。

最近一次结果：旧流程 loader 实际保留 1376 个样本，train/val/test=953/260/163；测试被试为 cwh/gf/tyy，验证被试为 byx/gzj/yyl。车辆-only 测试指标为 test steer RMSE=0.3386，主阶段 RMSE=0.2184，尾段 RMSE=0.3105，selection=0.8206。

当前判断：这次结果看起来明显好于前面一些旧流程车辆-only结果，但不能直接下结论说样本定义已经最终正确，因为本轮测试集样本只有 163 个，并且使用的是固定被试划分。分被试样本级指标显示 tyy 明显更难，尾段误差最高，需要继续看预测图和分样本类型。

当前最大风险：被试划分下 test 样本数量偏少，且 cwh/gf/tyy 的样本类型分布可能和旧测试集不同；因此这个结果更适合作为“服务器处理后样本 + 被试划分”的新检查点，不应和随机划分结果简单横向比较。

下一步准备做什么：先人工查看本轮预测总览图和 12 张固定样本图；如果图像物理意义明显比旧样本更好，再继续在同一 v0.5 样本定义下补车辆+连续风格，之后再考虑生理增量。

用户可以优先查看：

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_server_aligned_subject_oldflow_fair09_user_summary_cn.md`
- `F:\data_set_process\data_process\tmp\event_conditioned_runs\V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144\prediction_figures\test\overview.png`
- `F:\data_set_process\data_process\tmp\event_conditioned_runs\V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144\prediction_figures\test`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_subject_sample_metrics_v0_5.csv`

---

# 项目状态更新：v0.3 样本筛选策略 GPU 快速对比完成

更新时间：2026-05-19 22:05

当前阶段：车辆-only 样本纳入范围审查。GPU 快速筛选已经完成，本轮仍不涉及连续风格、生理或脑电。

当前完成：服务器 4080 SUPER 跑完 19 个样本筛选策略。完整结果已拉回本地。

最近一次结果：综合排序第一为 `s16_weakpost_lat`，即“干净集 + 待复核 + 少量锚点后响应弱样本，并保留横向偏移特征”。它相对基础版本整体 RMSE 略高，但大响应错侧率和严重幅值不足率明显下降：基础版本 test RMSE=0.6376，错侧率=0.2692，严重幅值不足率=0.4038；`s16` test RMSE=0.6446，错侧率=0.2453，严重幅值不足率=0.3019。

当前判断：本轮不支持“全量加入 excluded”。大量加入车身强响应、低附着/横滚/弯道、精选非轻微或全部 excluded 都会让整体任务明显变乱。但少量加入“锚点后响应弱/保守响应”有继续价值，尤其是在物理指标上更好。

当前最大风险：`s16` 使用了横向偏移特征，而横向偏移曾经存在坐标跳变风险；因此它不能直接成为最终样本定义，还需要对这 16 个新增样本和横向偏移质量做图像复核。

下一步准备做什么：围绕 `s16`、`s04` 和基础版本生成对比预测图/新增样本复核图；确认 `s16` 的改善是否来自真正有用的样本，而不是横向偏移或坐标问题。

用户可以优先查看：

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_screening_sweep_gpu_user_summary_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu\tables\v03_screening_sweep_gpu_ranking.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu\tables\v03_screening_sweep_gpu_summary.csv`

---

# 项目状态更新：v0.3 样本筛选策略 GPU 快速对比

更新时间：2026-05-19 21:13

当前阶段：车辆-only 样本纳入范围继续审查。用户指出 CPU 跑筛选太慢是正确的，因此已停止旧的 CPU 版连续筛选任务，改为在服务器 4080 SUPER 上运行 PyTorch GPU 快速筛选。

当前正在做什么：在不加入连续风格、生理、脑电的前提下，对多种 v0.3 样本筛选策略做车辆-only 快速比较，重点看不同样本纳入方式是否改善方向错侧、严重幅值不足、大响应召回和整体误差。

正在运行的任务：服务器 screen `v03gpu` 正在运行 `stage03_v03_screening_sweep_gpu.py`。远程日志路径为 `/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_gpu_20260519_211258.log`。

当前最大风险：GPU 快速筛选模型是为了快速判断样本筛选方向，模型实现从旧的 sklearn CPU 基线改成了 PyTorch 线性/多层感知机，因此结果用于“筛选策略排序和方向判断”，不能直接和之前 sklearn 核回归结果当作同一模型公平对比。

下一步准备做什么：等待 GPU 筛选完成，拉回汇总表和报告；若某些样本策略在物理指标上明显更好，再针对前 2-3 个策略补更完整的车辆-only 基线和预测图。

用户可以优先查看：完成后查看 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_screening_sweep_gpu_user_summary_cn.md` 和 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu\tables\v03_screening_sweep_gpu_summary.csv`。

---

# 项目状态更新：v0.3 全量原始数据极限工况 episode 重筛

更新时间：2026-05-18 19:21

当前阶段：旧流程样本定义重建。当前不再基于旧 v0.2/v0.5/v0.6 候选表继续筛，而是从 `01_datasets/数据预处理/原始车辆数据` 下所有原始车辆 CSV 重新扫描。

当前完成：新增并运行 v0.3 全量原始数据极限/近极限工况 episode 筛选流程。旧候选表只作为上下文贴回，不作为样本入口；`carsim对标.csv` 已作为非被试记录排除，不混入驾驶员 episode。

最近一次结果：扫描 CSV 92 个；其中 89 个被试车辆记录成功读取，2 个记录过短，1 个非被试 CSV 跳过。最终检测 episode 1574 个：强响应型极限工况 49，弱响应/保守响应 208，延迟或无明显转向响应 139，正常驾驶/普通弯道对照 86，待人工复核 311，排除 781。

当前最大风险：v0.3 仍是自动弱标签，不等于最终训练真值。尤其是待人工复核和排除样本占比高，说明全量数据里很多片段存在工况/响应边界不清、窗口或坐标问题。进入模型训练前，应先人工查看代表图，确认强响应、弱响应和延迟/无响应分类是否符合研究目标。

下一步准备做什么：优先复核 v0.3 代表图，确认样本语义后，再构建车辆-only 数据集并跑无学习基线和车辆-only 强基线。只有车辆-only 在新样本上比旧样本更符合物理意义，才继续加入连续驾驶风格和生理数据。

用户可以优先查看：

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_episode_v0_3_user_summary_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\strong_response_episodes_v0_3.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\weak_or_conservative_response_episodes_v0_3.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\delayed_or_no_steer_response_episodes_v0_3.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\review_panels`

---

# 项目状态更新：方向盘动作 episode 样本重建 v0.6

更新时间：2026-05-14

当前阶段：旧流程事件样本重建继续推进。

当前完成：以方向盘角速度启动为主锚点的 episode 自动挖掘，已生成分类样本表、复核图和汇总报告。

最近一次结果：总输出 1574 行；P1=58，P2=13，C=4，N=1415，U=51，X=33。

用户优先查看：

- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episode_summary_v0_6.md`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episodes_all_v0_6.csv`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\primary_positive_episodes_P1_v0_6.csv`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figures`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figure_index_v0_6.csv`

下一步建议：先人工复核 P1、U 和 C 类代表图，再决定是否把 P1/P2 作为“早期方向盘动作预测剩余轨迹”的训练样本。

---

# 项目状态更新：方向盘动作 episode 样本重建 v0.6

更新时间：2026-05-14

当前阶段：旧流程事件样本重建继续推进。

当前完成：以方向盘角速度启动为主锚点的 episode 自动挖掘，已生成分类样本表、复核图和汇总报告。

最近一次结果：总输出 1574 行；P1=58，P2=13，C=4，N=1415，U=51，X=33。

用户优先查看：

- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episode_summary_v0_6.md`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episodes_all_v0_6.csv`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\primary_positive_episodes_P1_v0_6.csv`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figures`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figure_index_v0_6.csv`

下一步建议：先人工复核 P1、U 和 C 类代表图，再决定是否把 P1/P2 作为“早期方向盘动作预测剩余轨迹”的训练样本。

---

# 项目状态更新：方向盘动作 episode 样本重建 v0.6

更新时间：2026-05-14

当前阶段：旧流程事件样本重建继续推进。

当前完成：以方向盘角速度启动为主锚点的 episode 自动挖掘，已生成分类样本表、复核图和汇总报告。

最近一次结果：总输出 1499 行；P1=10，P2=3，C=3，N=1434，U=38，X=11。

用户优先查看：

- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episode_summary_v0_6.md`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episodes_all_v0_6.csv`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\primary_positive_episodes_P1_v0_6.csv`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figures`
- `F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figure_index_v0_6.csv`

下一步建议：先人工复核 P1、U 和 C 类代表图，再决定是否把 P1/P2 作为“早期方向盘动作预测剩余轨迹”的训练样本。

---

# 项目状态更新：方向盘到车辆动态时间差审计

更新时间：2026-05-14

当前阶段：旧流程事件样本与锚点继续审计。

当前刚完成：方向盘动作开始到横向/横摆/侧倾响应开始的时间差审计。

最近一次结果：大多数样本没有稳定的 0.2 秒以上提前量，不能直接假设“方向盘先动很久后车辆才侧倾”。

用户优先查看：

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\steer_to_vehicle_dynamics_latency_v0_1_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`

下一步建议：根据本次比例决定是否把任务拆成“方向盘早期动作预测剩余轨迹”“车辆扰动后纠偏”“几乎同步动作延续”三类，而不是继续混合训练。

---

# 项目状态更新：方向盘到车辆动态时间差审计

更新时间：2026-05-14

当前阶段：旧流程事件样本与锚点继续审计。

当前刚完成：方向盘动作开始到横向/横摆/侧倾响应开始的时间差审计。

最近一次结果：大多数样本没有稳定的 0.2 秒以上提前量，不能直接假设“方向盘先动很久后车辆才侧倾”。

用户优先查看：

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\steer_to_vehicle_dynamics_latency_v0_1_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`

下一步建议：根据本次比例决定是否把任务拆成“方向盘早期动作预测剩余轨迹”“车辆扰动后纠偏”“几乎同步动作延续”三类，而不是继续混合训练。

---

# 项目状态更新：方向盘到车辆动态时间差审计

更新时间：2026-05-14

当前阶段：旧流程事件样本与锚点继续审计。

当前刚完成：方向盘动作开始到横向/横摆/侧倾响应开始的时间差审计。

最近一次结果：大多数样本没有稳定的 0.2 秒以上提前量，不能直接假设“方向盘先动很久后车辆才侧倾”。

用户优先查看：

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\steer_to_vehicle_dynamics_latency_v0_1_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`

下一步建议：根据本次比例决定是否把任务拆成“方向盘早期动作预测剩余轨迹”“车辆扰动后纠偏”“几乎同步动作延续”三类，而不是继续混合训练。

---

# R2E-Steering 项目总进度看板

## 最新更新：2026-05-13 08:09

- 当前阶段：Stage 7j session 多折稳定性验证 v0.1 已完成；gate=`no_upgrade`，Stage 8 生理/EEG 仍阻塞。
- 当前正在做什么：Stage 7j 严格 session-CV 已归档并提交；准备进入 Stage 7k 候选生成/选择规则复核。
- 已完成什么：新增并运行 `stage07j_session_cv_stability_v0_1.py`；每折重训 RBF/KNN 基座；排除固定 split 的 top-K/Transformer/keypoint 预测特征；只使用事件前车辆/道路上下文和该折重训 RBF 形态特征；生成 5 折指标、候选分数、policy 汇总、gate 表、3 张图、用户总结和技术报告；已提交 `11296297 Add stage7j session cv stability audit`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：`stability_penalty_l05` 5 折平均 test delta vs fold RBF=+0.000329，improved fold rate=0.600，difficult improved fold rate=0.800，gate=`no_upgrade`；原始 Stage7g val gate 平均 delta=-0.004170 但也只有 3/5 折改善且 wrong-side 平均略差，不能直接升级。
- 当前最大风险是什么：如果只看单折 Stage7i 或只看原始 val gate 的平均 RMSE，会高估车辆-only 多候选路线；当前还没有稳定、可复验、能同时改善 RMSE 和物理错误的选择规则。
- 下一步准备做什么：Stage 7k 应回到候选生成/选择规则设计，优先处理 wrong-side、幅值不足和困难样本；也可以做完整上游候选重训，但不能进入生理/EEG。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07j_session_cv_stability_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_policy_aggregate.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_policy_fold_deltas.png`。

## 最新更新：2026-05-13 07:55

- 当前阶段：Stage 7i 稳定性校准候选选择 v0.1 已完成；gate=`weak_candidate_continue`，但主线升级仍为 `not_final`，Stage 8 生理/EEG 仍阻塞。
- 当前正在做什么：归档 Stage 7i 结果，确认稳定惩罚规则是否比 Stage 7g 原始 val-best 规则更稳健；本阶段不训练新轨迹模型，只基于已有 Stage 7g 候选做 train/val 选择校准。
- 已完成什么：新增并运行 `stage07i_stability_calibrated_selection_v0_1.py`；修复 Stage7h 稳定表缺少 difficult/wrong-side/large-recall delta 字段的问题；生成候选分数表、policy split 指标、test 汇总、逐样本收益、gate 表、3 张图、用户总结和技术报告；已提交 `d294a520 Add stage7i stability calibrated selection`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：稳定惩罚规则 `stability_penalty_l05` 选中 `segment_resid_rf_blend_25`；test RMSE=0.528046，相对 RBF/KNN delta=-0.005620；困难样本 RMSE delta=-0.029588；wrong-side=0.225，large recall=0.750。
- 当前最大风险是什么：当前收益只来自一个固定 session-level split，且主要改善 RMSE/困难样本 RMSE，错侧率和大幅响应召回没有提升；不能把它直接冻结为最终车辆-only 主线。
- 下一步准备做什么：Stage 7j 应做多折 session validation 或分层 validation 重构，验证 `stability_penalty_l05` 是否能稳定选中有收益候选；生理/EEG/连续风格仍不进入。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07i_stability_calibrated_selection_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_policy_test_summary.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_policy_summary.png`。
## 最新更新：2026-05-13 07:42

- 当前阶段：Stage 7h val/test 选择不稳定诊断 v0.1 已完成；gate=`no_upgrade`，Stage 8 生理/EEG 仍阻塞。
- 当前正在做什么：归档 Stage 7h 诊断结果，明确当前问题是车辆-only 候选选择/校准不稳定，而不是可以进入新模态增量验证。
- 已完成什么：新增并运行 `stage07h_val_test_selection_diagnostics_v0_1.py`；不训练新模型，只读取 Stage 7g 产物，生成候选 split 稳定性、类别/数值分布偏移、逐样本收益、分 bucket 收益、gate 表、4 张诊断图、用户总结和技术报告；已提交 `d990f8e3 Add stage7h selection diagnostics`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：val selected=`segment_abs_rf_blend_25`，test delta=+0.002509；test-best non-oracle=`rbf_resid_keypoint_scaled`，test delta=-0.025129，但它未被 val 选中，只能作为诊断。最大类别偏移是 `response_family`，最大数值偏移是 `prob_entropy`。
- 当前最大风险是什么：如果按 test 表现事后选 `rbf_resid_keypoint_scaled`，会形成选择泄漏；如果忽略 val/test response_family 和置信度分布偏移，会误判多候选路线已经可部署。
- 下一步准备做什么：Stage 7i 应做候选选择校准或验证集重构，例如多折 session validation、按 response bucket/道路模块分层 gate、关键点不确定性评分；仍不进入生理/EEG。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07h_val_test_selection_diagnostics_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_candidate_split_stability.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_candidate_val_test_stability.png`。

## 最新更新：2026-05-13 07:33

- 当前阶段：Stage 7g keypoint/segment 车辆-only 候选 v0.1 已完成；gate=`no_upgrade`。
- 当前正在做什么：收口关键点/分段候选结果，继续把生理/EEG 有效性实验保持阻塞，避免把车辆-only 选择不稳误归因给生理数据。
- 已完成什么：新增并运行 `stage07g_keypoint_segment_candidates_v0_1.py`；只用事件前车辆、道路/事件上下文和候选预测自身形态特征训练关键点回归；生成关键点预测表、target 指标、候选指标、逐样本指标、validation selection、oracle 诊断、gate 表、4 张图、用户总结和技术报告；已提交 `52de7176 Add stage7g keypoint segment candidates`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：val gate 选择 `segment_abs_rf_blend_25`，但 test RMSE=0.536176，比 RBF/KNN 0.533667 差 +0.002509；keypoint/segment oracle RMSE=0.462003。test-only 最好非 oracle 候选 `rbf_resid_keypoint_scaled` RMSE=0.508538，但它在 val 上比 RBF 差，不能事后升级。
- 当前最大风险是什么：关键点/分段候选在 test 上出现了有价值信号，但 validation 不能稳定选中；如果按 test 事后挑 `rbf_resid_keypoint_scaled`，会造成选择泄漏。
- 下一步准备做什么：Stage 7h 应复核 val/test 分布差异和关键点置信度/校准，不进入生理/EEG；重点解释为什么 test 上有好候选但 val gate 选不中。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07g_keypoint_segment_candidates_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_candidate_metrics.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_keypoint_target_scatter.png`。

## 最新更新：2026-05-13 07:19

- 当前阶段：Stage 7f response-factorized vehicle-only candidate v0.1 已完成；gate=`no_upgrade`，当前主线仍是 RBF/KNN 车辆-only 参照。
- 当前正在做什么：收口 Stage 7f 响应分解候选结果，准备把 oracle 空间转化为下一轮可训练的关键点/分段候选设计，而不是进入生理/EEG。
- 已完成什么：新增并运行 `stage07f_response_factorized_candidates_v0_1.py`；基于 Stage 7e 响应类型把候选拆成方向/幅值、峰值时间、尾段、反向修正和多段修正原型；生成 factor 预测、候选逐样本指标、policy 指标、oracle 诊断、gate 表、3 张图、用户查看版总结和技术报告；已提交 `12cef06b Add stage7f response factorized candidates`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：val gate 选择 `rbf_kernel_ridge_context_no_subject`；test RBF/KNN RMSE=0.533667，selected RMSE=0.533667，delta=+0.000000；response-factorized oracle RMSE=0.440217，combo oracle RMSE=0.388119，但两者都是诊断上限，不是可部署性能。
- 当前最大风险是什么：方向、峰值时间等因子有信号，但幅值和尾段等关键因子仍不稳定；如果直接把 response-factorized oracle 当成模型结果，会高估车辆-only 当前能力。
- 下一步准备做什么：Stage 7g 不应继续堆 selector，也不应进入生理/EEG；更合适的是把响应分解原型升级为可训练的 keypoint/segment candidate，并加强幅值、尾段和修正段建模。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07f_response_factorized_candidates_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_factor_prediction_metrics.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_oracle_gain_predictions_test.png`。


## 最新更新：2026-05-13 06:39

- 当前阶段：Stage 7b 非 oracle top-K selector 轻量实验 v0.1 已完成；当前不升级多假设主线。
- 当前正在做什么：验证 Stage 7a 协议下的轻量 selector 是否能不用 test 标签把 top-K/RBF oracle 上限转成可部署收益。
- 已完成什么：新增并运行 `stage07b_non_oracle_topk_selector_v0_1.py`；显式剔除 label-derived 输入字段，使用 37 个允许特征训练 logistic/RF selector 和置信度 fallback；生成 feature audit、allowed features、policy metrics、decision diagnostics、coverage-risk、gate 表、3 张图、用户总结和技术报告；已提交 `d431cd11 Add stage7b non-oracle topk selector`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：val 选中 `logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80`；test RMSE=0.533667，与 RBF/KNN 相同，delta=+0.000000；test 上 RBF 选择比例=1.000，说明实际完全退回主参照；gate=`no_upgrade`，生理/EEG 继续 blocked。
- 当前最大风险是什么：当前 selector 的安全性来自全部退回 RBF，而不是学会了可靠候选选择；如果不导出更完整的候选轨迹差异特征，Stage 7 仍会停在 oracle gap。
- 下一步准备做什么：Stage 7c 或回到 Stage 6：导出完整候选预测轨迹/形态差异，或重新设计候选生成；暂不进入生理/EEG。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07b_non_oracle_topk_selector_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_feature_audit.csv`。

## 最新更新：2026-05-13 06:31

- 当前阶段：Stage 7a 非 oracle 多候选选择协议 v0.1 已完成；Stage 7 还未训练模型。
- 当前正在做什么：把 Stage 7 的候选池、允许特征、禁止信息、评价指标、固定图和升级 gate 固定下来，防止把 oracle/best-of-K 当作可部署模型。
- 已完成什么：新增并运行 `stage07a_non_oracle_selection_protocol_v0_1.py`；候选池 10 个，包含 RBF/KNN 主参照、ridge、KNN/template、keypoint、top-K 分支；生成 candidate pool、feature guard、selection protocol、evaluation plan、fixed plot protocol、gate 表、2 张图、用户总结和技术报告；已提交 `dfacb38d Add stage7a non-oracle selection protocol`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：Stage 7a gate=`ready_for_non_oracle_design`，但 `deployable_selector_status=blocked`；RBF RMSE=0.533667，broad oracle RMSE=0.375182，当前最好可部署 selector RMSE=0.533912，仍未超过 RBF/KNN；生理/EEG 继续 blocked。
- 当前最大风险是什么：如果直接进入多假设训练而不落实非 oracle 选择和校准，仍会得到“oracle 好看、实际不可用”的结果。
- 下一步准备做什么：Stage 7b：导出/整理候选预测轨迹和候选间差异特征，训练只用 train、调参只用 val 的非 oracle selector，并报告 calibration、coverage-risk、固定图和 selector regret。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07a_non_oracle_selection_protocol_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_feature_guard_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_gate_table.csv`。

## 最新更新：2026-05-13 06:24

- 当前阶段：Stage 6e 多候选 oracle gap 复核 v0.1 已完成；Stage 7 多假设路线只允许作为“选择策略问题”继续，不允许只报告 oracle。
- 当前正在做什么：把已有车辆-only 候选合成候选池，量化 oracle 上限和实际可部署 selector 之间的差距。
- 已完成什么：新增并运行 `stage06e_multicandidate_oracle_gap_v0_1.py`；读取 RBF/KNN、ridge、template、direct/structured Transformer、keypoint、top-K 分支和已有 selector 的逐样本表，生成候选池指标、winner 明细、winner 汇总、gap 表、gate 表、2 张图、用户总结和技术报告；已提交 `cb4d8eec Add stage6e multicandidate oracle gap audit`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：RBF/KNN 主参照 test RMSE=0.533667；broad oracle pool test RMSE=0.375182，delta=-0.158484，但这是用真实标签挑候选的不可部署上限；当前最好可部署 selector `selector_logreg_rbf_keypoint_no_subject` test RMSE=0.533912，比 RBF/KNN 差 +0.000245。gate=`oracle_signal_but_deployable_selection_blocked`，生理/EEG 继续 blocked。
- 当前最大风险是什么：如果把 broad oracle 或 best-of-K 当作模型结果，会严重高估当前可部署能力；当前真正瓶颈是候选选择/可靠性校准，不是证明生理或 Transformer 有效。
- 下一步准备做什么：如果进入 Stage 7，必须先设计非 oracle 候选选择策略、概率校准和固定坏样本选择协议；否则继续复查车辆-only 表示和样本规则。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06e_multicandidate_oracle_gap_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_gap_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/figures/multicandidate_oracle_gap_rmse.png`。

## 最新更新：2026-05-13 06:16

- 当前阶段：Stage 6d RBF/KNN reliability gate v0.1 已完成；当前 selector/reliability 路线仍不升级主线。
- 当前正在做什么：把 Stage 6c 的 RBF/keypoint selector 结果做保守门控复核，确认能否在不牺牲 RMSE 的情况下保留错侧率和大幅响应召回收益。
- 已完成什么：新增并运行 `stage06d_reliability_gate_v0_1.py`；扫描 val 选择的 threshold policy，生成 selected policy 表、policy metrics、gate 表、best confusion、2 张图、用户总结和技术报告；已提交 `4264db88 Add stage6d reliability gate`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：RBF/KNN 主参照 test RMSE=0.533667；保守 policy `val_rmse_noninferior_conservative` test RMSE=0.534545，比 RBF/KNN 差 +0.000878，wrong-side=0.225、large recall=0.750，均没有改善；激进 `val_best_rmse` 可把 wrong-side 降到 0.175、large recall 提到 0.875，但 RMSE=0.544356，退化更明显。gate=`no_upgrade`，stage05 physio/eeg=blocked。
- 当前最大风险是什么：oracle/keypoint 确实存在样本级上限，但当前 selector 无法稳定把上限转成可部署增益；如果只看 wrong-side 或 large recall 单项改善，会误判为主线升级。
- 下一步准备做什么：Stage 6 selector/reliability 当前形式降级为诊断候选；下一步更适合转向多假设候选生成/实际选择策略，或复查车辆-only 表示与样本规则，而不是进入生理/EEG。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06d_reliability_gate_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/figures/reliability_gate_physical_metrics.png`。

## 最新更新：2026-05-13 06:06

- 当前阶段：Stage 6c selector feature revision v0.1 已完成；当前修订不升级主线，生理/EEG 仍不进入。
- 当前正在做什么：评估加入候选模型预测差异特征后的 RBF/keypoint selector 是否能把 oracle 上限转成可部署增益。
- 已完成什么：新增并运行 `stage06c_selector_feature_revision_v0_1.py`；比较原始 logistic、工程化 logistic、保守 logistic 和浅层随机森林 selector，生成特征协议、阈值扫描、指标表、gate 表、2 张图、用户总结和技术报告。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：val 选择 `rf_engineered_shallow`；test RMSE=0.544356，比 RBF 差 +0.010689；wrong-side 从 0.225 降到 0.175，large recall 从 0.750 升到 0.875；但 gate=`no_upgrade_current_revision`，stage05 physio/eeg=blocked。
- 当前最大风险是什么：如果只看错侧率和大幅召回改善，可能误把 RMSE 退化和 FP=13 的不稳定选择忽略；当前最多说明 selector/reliability 有物理指标信号，不能升级为主线。
- 下一步准备做什么：围绕 RF selector 的 6 个 FN 和 13 个 FP 做可靠性门控，目标是保留错侧/大幅响应收益，同时控制 RMSE 退化。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06c_selector_feature_revision_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/figures/selector_revision_physical_metrics.png`。

## 最新更新：2026-05-13 05:57

- 当前阶段：Stage 6b RBF/keypoint 选择器错误复盘 v0.1 已完成；生理/EEG 仍不进入。
- 当前正在做什么：复盘 `selector_logreg_rbf_keypoint_no_subject` 在 B 轨道 test 40 个样本上为什么只形成弱候选，准备下一版 selector/reliability 特征修正。
- 已完成什么：新增并运行 `stage06b_keypoint_selector_error_review_v0_1.py`；生成选择器样本明细、混淆表、分组摘要、top regret、漏选 keypoint 样本、错选 keypoint 样本、下一步动作表、3 张图、用户总结和技术报告；已提交 `753525fd Add stage6b keypoint selector error review`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：test 40 个样本中 selector 选 keypoint 比例=0.275，oracle 中 keypoint 更优比例=0.425；TP=5、FP=6、FN=12、TN=17；selector 平均 delta vs RBF=+0.006945，平均 oracle regret=0.059122。
- 当前最大风险是什么：当前 selector 主要问题是漏选 keypoint 潜在收益样本，同时有 6 个错选伤害样本；如果不修选择器，best-of-K 上限无法转化成可部署增益。
- 下一步准备做什么：基于 FN/FP 样本做 selector feature revision 和 reliability gate，优先加入候选差异、响应形态风险、道路模块和历史稳定性特征。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06b_keypoint_selector_error_review_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_confusion_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_probability_vs_gain.png`。

## 最新更新：2026-05-13 05:50（本机实际时间；阶段顺序接在阶段 4 收口之后）

- 当前阶段：阶段 6 车辆-only 结构化路线审计 v0.1 已完成；当前不进入生理/EEG。
- 当前正在做什么：把已有 RBF、direct Transformer、响应分解 Transformer、keypoint、top-K/multihypothesis 和选择器结果统一到车辆-only 结构化 gate 表，准备进入 Stage 6b。
- 已完成什么：新增并运行 `stage06_vehicle_only_structured_route_audit_v0_1.py`；汇总 26 个 B 轨道车辆-only候选，生成 scorecard、delta 表、gate 表、下一步动作表、2 张图、用户查看版总结和技术报告；已提交 `b4d7ac20 Add stage6 vehicle structured route audit`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：RBF 仍为 limited primary reference，test RMSE=0.533667；响应分解 Transformer v0.1 为 no_go_current_form，test RMSE=0.602174；keypoint selector 为 weak_candidate_continue，RMSE delta=+0.000245；最佳 oracle/best-of-K RMSE=0.415652 但不可部署；stage05 physio/eeg=blocked。
- 当前最大风险是什么：如果把 oracle/best-of-K 上限当成实际可用模型，或跳过可部署选择器直接进入生理/EEG，会产生错误增量归因。
- 下一步准备做什么：Stage 6b：复盘 `selector_logreg_rbf_keypoint_no_subject` 的样本级选择错误，把 best-of-K/oracle 上限转化为非 oracle 的可部署选择策略和可靠性门控。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06_vehicle_only_structured_route_audit_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_route_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/figures/vehicle_structured_route_delta_vs_rbf.png`。

## 最新更新：2026-05-13 06:25

- 当前阶段：阶段 4 连续风格路线收口决策 v0.1 已完成；当前连续风格直接残差融合路线不升级为主线。
- 当前正在做什么：准备进入车辆-only 结构化轨迹建模路线，而不是进入生理/EEG。
- 已完成什么：新增并运行 `stage04_style_route_decision_v0_1.py`；汇总风格协议、session-level 探索、subject-level 复核和 gate，形成阶段 4 收口表、下一步动作表、图和中文报告；已提交 `4064bf64 Add style route decision`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：session-level style60 delta RMSE=0.000892；subject-level style60 delta RMSE=-0.001337；物理指标未稳定改善；style route gate=no_go_current_form；physio/eeg=blocked。
- 当前最大风险是什么：如果跳过车辆-only 结构化错误复盘，直接进入生理/EEG，会把车辆主参照未解决的问题错误归因给新模态。
- 下一步准备做什么：阶段 6 车辆-only 结构化轨迹建模，优先响应分解、关键点+残差、多假设/可靠性，继续用固定 RBF 参照和坏样本图。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_route_decision_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_decision_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/figures/style_route_rmse_delta_summary.png`。

## 最新更新：2026-05-13 06:05

- 当前阶段：阶段 4 连续风格跨 split 复核 v0.1 已完成；连续风格有效性结论仍阻塞。
- 当前正在做什么：准备把连续风格路线暂时降级收口，并回到车辆-only 结构化轨迹建模问题。
- 已完成什么：新增并运行 `stage04_style_cross_split_validation_v0_1.py`；在 B 轨道 270 个严格核心样本上完成 session-level 与 subject-level 两类切分的 RBF+风格残差对照，且每个 split 都重新使用 train-only 风格标准化。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：session-level RBF RMSE=0.533667、RBF+last60 风格 RMSE=0.534559；subject-level RBF RMSE=0.484847、RBF+last60 风格 RMSE=0.483510；风格有效性 gate 仍为 blocked。
- 当前最大风险是什么：如果继续强推风格，可能把小样本波动、被试/道路分布或融合方式不足误解释为风格有效或无效；当前证据只支持“当前表示和融合方式下没有形成强证据”。
- 下一步准备做什么：阶段 4 先收口，返回阶段 6/车辆-only 结构化轨迹路线，优先解决错侧、幅值、尾段、反向修正、多段修正和困难样本。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_cross_split_validation_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_metric_summary_test.png`。

## 最新更新：2026-05-13 05:40

- 当前阶段：阶段 4 连续驾驶风格探索性增量对照 v0.1 已完成，仍处于探索阶段。
- 当前正在做什么：已固定 RBF/KNN 类车辆-only 主参照，正在判断事件前连续风格是否值得进入更严格 split 验证。
- 已完成什么：新增并运行 `stage04_style_increment_exploratory_v0_1.py`；在 B 轨道 270 个严格核心失稳响应样本上完成 RBF 残差 Ridge、驾驶员 ID 对照、道路模块对照和多种置乱控制；生成指标表、逐样本表、固定预测图、坏样本图、置乱汇总、gate 表和中文报告。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：RBF test RMSE=0.533667；RBF+last60 风格 test RMSE=0.534559；风格有效性结论仍为 blocked。
- 当前最大风险是什么：session-level 探索性收益可能来自驾驶员身份、道路/场景分布或小样本偶然性；如果物理指标和坏样本图不改善，不能升级为主线。
- 下一步准备做什么：补 subject-level/跨 session 风格验证，继续比较真实风格、驾驶员 ID、同被试置乱、跨被试置乱、道路均衡置乱；生理/EEG 仍阻塞。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_increment_exploratory_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_metric_summary_test.png`。



## 最新更新：2026-05-13 05:08

- 当前阶段：阶段 4 连续驾驶风格协议与候选特征处理 v0.1 已归档并完成 Git 提交。
- 当前正在做什么：阶段 4 协议产物已提交，准备进入固定 RBF 主参照下的连续风格探索性模型对照。
- 已完成什么：新增并运行 `stage04_continuous_style_protocol_v0_1.py`；从 B 轨道 270 个严格核心失稳响应样本中提取事件前 3 秒以前的连续车辆历史风格候选特征，生成 long 表 1080 行、wide 表 270 行、train-only 标准化参数 440 项，其中 436 项可用；已提交 `012f4803 Add continuous style protocol audit`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：风格窗口泄漏检查通过，直接输入窗口重叠 0、标签未来重叠 0；`last30_guard3` 270/270 可用，`last60_guard3` 268/270 可用，`last120_guard3` 267/270 可用，`prefix_until_guard3` 268/270 可用。
- 当前最大风险：本轮只完成候选风格处理和协议设计，还没有模型、置乱、分被试和物理指标对照；不能宣称连续风格有效。
- 下一步准备做什么：在固定 RBF 主参照之上做阶段 4 探索性验证：原始连续风格、驾驶员 ID 对照、被试内/跨被试/跨 session/道路平衡置乱，并只用 train split 拟合标准化和选择规则。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_protocol_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/figures/style_feature_availability_by_window.png`。

## 最新更新：2026-05-13 04:37

- 当前阶段：阶段 3 RBF 主参照冻结审计 v0.1 已归档并完成 Git 提交。
- 当前正在做什么：收口本轮 RBF 冻结审计状态记录，准备进入阶段 4 连续风格协议设计/探索性实验。
- 已完成什么：提交 `112824f7 Add rbf reference freeze audit`；该提交包含冻结审计脚本、RBF 指标/失败/top bad/gate/稳健性表、2 张图、运行摘要、两份中文报告和 04:31 透明化记录。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：RBF 固定为 `limited_reference_freeze` 的保守车辆-only 主参照；`vehicle_only_problem_solved=fail`，`stage04_style_protocol_allowed=conditional_pass`，`stage05_physio_eeg_allowed=blocked`。
- 当前最大风险：后续只能把 RBF 当参照底线，不能把 RBF 的物理缺陷当作已解决；连续风格若要继续，必须证明它在固定 RBF 之外改善物理错误或困难样本。
- 下一步准备做什么：整理阶段 4 连续驾驶风格无泄漏协议、置乱对照和分被试评估计划；生理/EEG 仍保持阻塞。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_reference_freeze_audit_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_freeze_gate_table.csv`。

## 最新更新：2026-05-13 04:31

- 当前阶段：阶段 3 RBF 主参照冻结审计 v0.1 已完成；结论是有限冻结，不是车辆-only 问题已解决。
- 当前正在做什么：归档 RBF 主参照冻结审计产物，并准备把本轮脚本、图表、表格、报告和透明化记录提交 Git。
- 已完成什么：新增并运行 `stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1.py`；生成 RBF 指标画像、失败画像、top bad 样本、冻结 gate 表、稳健性快照、2 张图、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：`reference_identity_fixed=pass_limited`，`vehicle_only_problem_solved=fail`，`stage04_style_protocol_allowed=conditional_pass`，`stage05_physio_eeg_allowed=blocked`。
- 当前最大风险：RBF 虽可作为保守车辆-only 主参照，但错侧、反向修正计数、复杂多段修正和困难样本仍未解决；后续若忽略这些缺陷，会误把风格/生理增量归因为模型已解决的问题。
- 下一步准备做什么：先提交本轮冻结审计；随后可进入阶段 4 连续风格的协议设计/探索性实验，但必须固定 RBF 对照并做置乱、分被试、物理指标和坏样本分析；生理/EEG 仍不能进入有效性验证。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_reference_freeze_audit_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_freeze_gate_table.csv`。

## 最新更新：2026-05-13 04:21

- 当前阶段：阶段 3 车辆-only 主参照决策表 v0.2 已归档并完成 Git 提交；强车辆基线仍未完全冻结。
- 当前正在做什么：收口阶段 3 车辆-only 决策表，准备后续是否冻结 RBF 主参照的判断。
- 已完成什么：提交 `e04bdb2f Add vehicle-only decision table`；该提交包含决策表脚本、候选决策表、gate 状态表、角色汇总、指标库存、3 张图、运行摘要、两份中文报告和 04:18 透明化记录。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：`vehicle_main_reference_available=partial`，`strong_vehicle_baseline_frozen=no`，`style_physio_eeg_allowed_now=no`。
- 当前最大风险：如果不先明确接受 RBF 的物理缺陷或继续车辆-only 结构，就进入新模态验证，会造成增量归因不干净。
- 下一步准备做什么：做 RBF 主参照冻结审查，或者继续阶段 3 的分响应类型/关键点条件多假设；风格、生理、EEG 仍阻塞。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_only_decision_table_user_summary_cn.md`。

## 最新更新：2026-05-13 04:18

- 当前阶段：阶段 3 车辆-only 主参照决策表 v0.2 已完成；强车辆基线仍未完全冻结。
- 当前正在做什么：把阶段 3 的 RBF/KNN、direct Transformer、structured Transformer、keypoint、selector、top-K 和 oracle 上限统一成决策表，防止后续误把弱候选或 oracle 当主线。
- 已完成什么：新增并运行 `stage03_vehicle_instability_vehicle_only_decision_table_v0_2.py`；读取既有阶段 3 指标表，不重新训练；生成车辆-only 候选决策表、gate 状态表、角色汇总、指标库存、3 张图、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：当前主参照仍是 `rbf_kernel_ridge_context_no_subject`，B 轨道 test RMSE=0.533667、错侧率=0.225、大幅响应召回=0.750；但反向修正完全匹配仍为 0，top-K fallback 未超过 RBF，因此 `strong_vehicle_baseline_frozen=no`，`style_physio_eeg_allowed_now=no`。
- 当前最大风险：如果此时进入风格/生理/EEG，会把车辆-only 未解决的错侧、反向修正和多段响应问题误归因给新模态；oracle 上限只能作为潜力，不是可部署性能。
- 下一步准备做什么：提交本轮决策表；之后要么做 RBF 主参照冻结审查并明确接受其缺陷，要么继续更强车辆-only 分响应类型/关键点条件多假设。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_only_decision_table_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_stage3_gate_status_v0_2.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/figures/vehicle_only_decision_key_metrics_test.png`。

## 最新更新：2026-05-13 04:11

- 当前阶段：阶段 3 top-K 可靠性选择/回退 v0.1 已归档并完成 Git 提交；强车辆基线仍未冻结。
- 当前正在做什么：收口可靠性选择 no-go 结果，准备下一步车辆-only 决策。
- 已完成什么：提交 `fbb8d94d Add topk reliability selector`；该提交包含可靠性选择脚本、指标表、逐样本表、决策表、阈值表、分层汇总、6 张图、运行摘要、两份中文报告和 04:07 透明化记录。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：本轮 val 选中的 fallback test RMSE=0.542071，仍差于 RBF 的 0.533667；best-of-RBF+topK oracle RMSE=0.415652 仅作上限。
- 当前最大风险：候选池 oracle 上限容易被误读为可部署收益；当前可部署选择器仍不能超过 RBF/KNN 类主参照。
- 下一步准备做什么：继续阶段 3 的车辆-only 决策，优先考虑 RBF/KNN 类主参照冻结审查或更强分响应类型/关键点条件多假设；继续阻塞风格、生理和 EEG。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_reliability_selector_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_metric_summary_test.png`。

## 最新更新：2026-05-13 04:07

- 当前阶段：阶段 3 top-K 可靠性选择/回退 v0.1 已完成；强车辆基线仍未冻结，RBF/KNN 类车辆基线仍是主参照。
- 当前正在做什么：把 top-K 候选覆盖、可靠性选择和 RBF 回退结果归档，判断是否能升级为车辆-only 主线。
- 已完成什么：新增并运行 `stage03_vehicle_instability_topk_reliability_selector_v0_1.py`；加载已有 top-K checkpoint，重建 RBF 车辆-only 预测，训练 train-only branch/candidate/fallback 选择器，用 val 固定 fallback 阈值，test 只报告；生成指标表、逐样本表、决策表、阈值表、分被试/道路模块汇总、固定图、坏样本图、oracle 增益图、决策计数图、fallback 风险散点图、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；没有服务器任务；本地脚本已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：val 选择 `topk_top1_rbf_fallback_logreg_no_subject`，test RMSE=0.542071，仍差于 RBF 的 0.533667；test 中 39/40 个样本回退到 RBF，1/40 选择 top1；best-of-RBF+topK oracle RMSE=0.415652，说明候选池有潜力但选择机制未解决。
- 当前最大风险：如果只看 oracle 上限，会高估可部署模型；本轮可靠性选择不能升级为强车辆基线，也不能进入风格、生理或 EEG 增量结论。
- 下一步准备做什么：把本轮 no-go 结果提交 Git；之后阶段 3 只能继续做更强的车辆-only 分响应类型/关键点条件多假设，或暂以 RBF/KNN 类车辆基线作为主参照进入更严格基线冻结审查。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_reliability_selector_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_metrics.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_metric_summary_test.png`。

## 最新更新：2026-05-13 03:49

- 当前阶段：阶段 3 top-K top1/bestK 差距复盘 v0.1 已归档并完成 Git 提交；强车辆基线仍未冻结。
- 当前正在做什么：收口 top-K gap review，准备下一步设计车辆-only 可靠性/选择头 v0.2 或关键点条件多假设。
- 已完成什么：提交 `1ace03f2 Add topk gap review`；该提交包含 gap review 脚本、样本详情、摘要、阈值、相关性、分层汇总、top gap 表、top1 worse than RBF 表、4 张诊断图、运行摘要、用户查看版总结、技术报告和 03:46 透明化记录。
- 正在运行什么任务：没有后台任务；没有本地训练任务；没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：低置信规则能捕捉 0.636 的 test 高 gap 样本，简单风险分数捕捉 0.545；说明可靠性特征有信号但不够强。
- 当前最大风险：如果直接把这套简单规则当成部署策略，会把 test 诊断当成训练决策；下一步必须在 train/val 固定规则或模型后再 test。
- 下一步准备做什么：继续阶段 3，做车辆-only 可靠性/选择头 v0.2 或关键点条件多假设；继续阻塞风格、生理和 EEG。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_gap_review_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_overall_summary.csv`。

## 最新更新：2026-05-13 03:46

- 当前阶段：阶段 3 top-K top1/bestK 差距复盘 v0.1 已完成；仍未冻结强车辆基线，仍处于车辆-only 可靠性/选择机制诊断阶段。
- 当前正在做什么：归档 top-K v0.1 的选择失败样本、可靠性信号和下一版模型依据。
- 已完成什么：新增并运行 `stage03_vehicle_instability_topk_gap_review_v0_1.py`；不训练新模型，只合并 top-K 分支诊断、逐样本物理指标、manifest 道路/事件字段和响应分解标签；生成样本详情、总体摘要、阈值、相关性、bucket 汇总、分被试/道路/响应族汇总、top gap 样本、top1 比 RBF 更差样本、4 张诊断图、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；本地复盘脚本已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：test top-1 与 best-of-3 一致率=0.300，平均 top1-bestK gap=0.110531；train 定义的简单高风险分数捕捉 test 高 gap 样本比例=0.545，低置信规则捕捉比例=0.636；最大 gap 样本为 `vehicle_instability_allraw__gf__2025_09_26_10_52_57__000300870__pre3_label3_response_coverage`，gap=0.447251。
- 当前最大风险：当前风险分数/低置信规则只是诊断线索，不是可部署策略；下一步若使用可靠性规则，必须在 train/val 固定后再 test 评价。
- 下一步准备做什么：提交本轮差距复盘；后续优先做可靠性/选择头 v0.2 或关键点条件多假设，继续阻塞连续风格、生理和 EEG 增量结论。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_gap_review_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_top_samples.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_risk_scatter.png`。

## 最新更新：2026-05-13 03:37

- 当前阶段：阶段 3 top-K 车辆-only Transformer v0.1 已归档并完成 Git 提交；强车辆基线仍未冻结。
- 当前正在做什么：收口 top-K v0.1，准备下一步继续车辆-only 选择机制/可靠性头或关键点条件多假设。
- 已完成什么：提交 `03165475 Add topk vehicle transformer`；该提交包含 top-K 脚本、checkpoint、指标表、逐样本表、分支诊断、可靠性分箱、图、运行摘要、用户查看版总结、技术报告和 03:34 透明化记录。
- 正在运行什么任务：没有后台任务；没有本地训练任务；没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：top-K top-1 没有超过 RBF，RBF RMSE=0.533667，top-1 RMSE=0.587883；best-of-3 RMSE=0.477534 说明候选覆盖有潜力，但 top-1 与 best-of-3 一致率只有 0.300，选择机制不足。
- 当前最大风险：若把 best-of-3 当成可部署效果，会严重高估模型；当前证据支持继续研究可靠性/选择头，不支持进入风格、生理或 EEG。
- 下一步准备做什么：分析 top1/bestK 差距样本，决定是改可靠性头，还是构建关键点条件多假设结构。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_vehicle_transformer_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_metrics.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_metric_summary_test.png`。

## 最新更新：2026-05-13 03:34

- 当前阶段：阶段 3 top-K 车辆-only Transformer v0.1 已完成；仍未冻结强车辆基线，仍处于多假设/可靠性车辆-only 探索阶段。
- 当前正在做什么：归档真正 top-K 车辆-only 模型结果，判断它相对 RBF、keypoint、selector 和 oracle 上限的价值。
- 已完成什么：新增并运行 `stage03_vehicle_instability_topk_vehicle_transformer_v0_1.py`；模型只用事件前车辆历史和道路/事件上下文，输出 3 条候选轨迹和分支概率；checkpoint 按 val top-1 RMSE 选择；生成 top-1、best-of-3、各分支、RBF 参照的指标表、逐样本表、分支诊断、可靠性分箱、固定图、坏样本图、top1/bestK 差距图、可靠性散点图、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；本地训练/评估已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。本轮使用本机 CUDA 训练，不是远程服务器。
- 最近一次结果：test RBF RMSE=0.533667、错侧率=0.225、大幅响应召回=0.750；top-1 RMSE=0.587883、错侧率=0.100、大幅响应召回=0.750；best-of-3 RMSE=0.477534、错侧率=0.025、大幅响应召回=0.875；top-1 与 best-of-3 分支一致率=0.300。
- 当前最大风险：best-of-3 上限很好，但 top-1 选择头明显不可靠，不能把 best-of-3 当成可部署效果；本轮 top-K 只能说明“候选覆盖有潜力，选择机制不足”，不能替代 RBF 主参照。
- 下一步准备做什么：提交本轮 top-K 结果；如果继续阶段 3，优先修选择机制/可靠性头或做关键点条件多假设，而不是进入风格、生理或 EEG。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_vehicle_transformer_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_top1_bestk_gap_samples_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_metrics.csv`。

## 最新更新：2026-05-13 03:22

- 当前阶段：阶段 3 RBF/keypoint 多候选车辆-only 复盘 v0.1 已归档并完成 Git 提交；仍未冻结强车辆基线。
- 当前正在做什么：收口多候选复盘，准备下一步转向真正 top-K/可靠性车辆-only 模型设计。
- 已完成什么：提交 `01033e3e Add rbf keypoint multihypothesis review`；提交包含多候选复盘脚本、统一指标、逐样本指标、选择摘要、误选样本表、oracle 增益表、5 张图、运行摘要、用户查看版总结、技术报告和 03:18 透明化记录。
- 正在运行什么任务：没有后台任务；没有本地训练任务；没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：oracle best-of-two 的 RMSE=0.475095 明显低于 RBF 的 0.533667，但 train/val selector 的 RMSE=0.533912 仍只与 RBF 持平；test selector choice accuracy=0.550，说明当前可部署选择器还不能兑现 oracle 上限。
- 当前最大风险：若继续只做离线 oracle 或事后挑选，会误判模型能力；下一步必须把 top-K 候选和可靠性选择做成训练期/验证期可确定的模型或规则。
- 下一步准备做什么：进入车辆-only top-K/可靠性模型设计；继续阻塞连续风格、生理和 EEG 增量结论。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/choice_summary.csv`。

## 最新更新：2026-05-13 03:18

- 当前阶段：阶段 3 RBF/keypoint 多候选车辆-only 复盘 v0.1 已完成；仍处于强车辆基线冻结前的多假设/可靠性评估阶段。
- 当前正在做什么：整理 RBF、keypoint、selector 和 oracle best-of-two 的同图、同表、误选样本和 oracle 增益证据，判断是否值得做真正多假设车辆-only 模型。
- 已完成什么：新增并运行 `stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1.py`；脚本不训练新模型，只重建 RBF 预测、加载 keypoint checkpoint、复用 train/val selector 决策并构造 oracle 上限；生成统一指标表、逐样本表、选择详情表、选择摘要、误选样本表、oracle 增益表、固定图、selector 坏样本图、oracle 增益图、选择混淆图、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；本地脚本已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：test 上 RBF RMSE=0.533667、keypoint RMSE=0.548993、selector RMSE=0.533912、oracle best-of-two RMSE=0.475095；selector 选择准确率=0.550，selector 选择 keypoint 比例=0.275，oracle 需要 keypoint 比例=0.425，平均选择后悔=0.059123。
- 当前最大风险：oracle 上限明显，但 selector 只学到一部分切换规则；如果直接把 oracle 当成模型效果会高估车辆-only 能力。当前只能说明多假设/可靠性路线有潜力，不能说明强车辆基线已冻结，更不能进入风格、生理或 EEG 增量结论。
- 下一步准备做什么：提交本轮多候选复盘；之后若继续阶段 3，应训练或构建真正 top-K/可靠性车辆-only 模型，而不是只在两个已训练模型之间做事后二选一。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_oracle_gap_samples_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/multihypothesis_metrics.csv`。

## 最新更新：2026-05-13 03:06

- 当前阶段：阶段 3 RBF vs keypoint train/val 选择器 v0.1 已归档并完成 Git 提交；仍处于车辆-only 多候选/可靠性策略探索阶段。
- 当前正在做什么：收口本轮 selector 结果，准备把下一步推进到更正式的车辆-only 多假设/可靠性建模，而不是进入风格、生理或 EEG 增量结论。
- 已完成什么：提交 `7e3d53f6 Add rbf keypoint selector`；该提交包含 selector 脚本、结果表、两张图、运行摘要、用户查看版总结、技术报告以及 02:54 透明化记录。
- 正在运行什么任务：没有后台任务；没有本地训练任务；没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：selector 在 test 上与 RBF 整体 RMSE 基本持平但未超过，RBF RMSE=0.533667，selector RMSE=0.533912；selector 将错侧率从 0.225 降到 0.200，将大幅响应召回从 0.750 提到 0.875，将困难 top20 RMSE 从 0.678907 降到 0.648368；oracle best-of-two RMSE=0.475095，说明两候选存在互补但选择器还不够强。
- 当前最大风险：如果只看 oracle 上限会高估实际可部署收益；当前 selector 只能作为多假设/可靠性路线的证据，不能替代强车辆基线，也不能支持风格或生理有效性结论。
- 下一步准备做什么：构建正式多假设车辆-only 评估包，至少同时报告 top-1、best-of-K、train/val selector、oracle 上限、固定预测图和误选样本分析。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_selector_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_metrics.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_test_metrics.png`。

## 最新更新：2026-05-13 02:54

- 当前阶段：阶段 3 RBF vs keypoint train/val 选择器 v0.1 已完成；仍处于车辆-only 多候选/可靠性策略探索阶段。
- 当前正在做什么：归档第一版不看 test 调参的 RBF/keypoint 自动选择器，判断多候选车辆-only 是否有实际空间。
- 已完成什么：新增并运行 `stage03_vehicle_instability_rbf_keypoint_selector_v0_1.py`；selector 只用 train 拟合、只用 val 选择阈值，test 只做最终评价；特征只包含事件/道路上下文和两个候选模型自身预测出的特征，不使用 subject ID、GT peak、sample_rmse、wrong_side、生理、脑电或连续风格；生成选择器训练表、特征表、阈值扫描表、决策表、统一指标表、选择后逐样本表、两张图、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；本轮本地前台脚本已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：val 选出阈值 0.55；test 上 keypoint 选择率 0.275，即 11/40 个样本。test RBF RMSE=0.533667、错侧率=0.225、大幅响应召回=0.750、困难 top20 RMSE=0.678907；selector RMSE=0.533912、错侧率=0.200、大幅响应召回=0.875、困难 top20 RMSE=0.648368；oracle best-of-two RMSE=0.475095。
- 当前最大风险：selector 已经改善方向/大幅响应/困难样本，但整体 RMSE 只是与 RBF 持平，仍不能宣布车辆-only 问题解决；oracle 上限说明 RBF 和 keypoint 互补，但当前可用特征还没完全学会何时切换。
- 下一步准备做什么：若继续阶段 3，应做更正式的多假设/可靠性车辆-only：报告 top-1、best-of-K、可部署选择策略、校准和固定图；在此之前继续阻塞连续风格、生理和 EEG 增量结论。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_selector_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_test_metrics.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_threshold_sweep.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_metrics.csv`。

## 最新更新：2026-05-13 02:44

- 当前阶段：阶段 3 keypoint+residual vs RBF 坏样本差异复盘 v0.1 已完成；仍处于车辆-only 主参照冻结前的错误转移分析阶段。
- 当前正在做什么：归档 keypoint+residual 与 RBF KRR 的逐样本差异，判断 keypoint 的收益是否稳定到可以成为主线，或更适合作为多假设/可靠性分支。
- 已完成什么：新增并运行 `stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1.py`；只读取 B 轨道 test 40 个样本的逐样本指标，不训练模型；生成样本级 RMSE 差异表、错误变化计数、分被试摘要、Top 改善/退化表、两张复盘图、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；本轮本地前台脚本已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：keypoint - RBF 的样本 RMSE 平均差为 +0.025325，说明整体仍略差；RMSE 明显改善 11/40，明显退化 20/40；keypoint 修复错侧 5 个样本、新增错侧 1 个样本；修复大幅响应召回 1 个样本、没有丢失大幅响应召回；新增尾段漂移 1 个样本。
- 当前最大风险：keypoint 的收益主要是方向和大幅响应召回，但启动延迟退化较多、RMSE 退化样本更多；如果把它单独升级为主线，会牺牲稳定整体误差。更合理的方向是多假设候选选择或可靠性/困难样本识别。
- 下一步准备做什么：准备多假设车辆-only 或模型选择/可靠性门控：用 RBF 保住整体 RMSE，用 keypoint 分支覆盖错侧和大幅响应，先做可用选择策略，再考虑是否进入连续风格/生理阶段。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_rmse_delta_top_samples.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_error_change_counts.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_sample_delta.csv`。

## 最新更新：2026-05-13 02:34

- 当前阶段：阶段 3 B 轨道车辆-only 关键点 + 残差 Transformer v0.1 已完成；仍处于强车辆基线冻结前的结构候选筛选阶段。
- 当前正在做什么：归档关键点 + 残差车辆-only 结果，并判断它相对 RBF KRR、direct Transformer、上一版 structured Transformer 的价值。
- 已完成什么：新增并运行 `stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1.py`；模型只在 B 轨道 270 条 3 秒严格核心样本上训练，输入只用事件前车辆历史和道路/事件上下文；关键点标签只作为训练目标，不作为推理输入；生成指标表、逐样本表、关键点误差表、固定图、坏样本图、checkpoint、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；本轮本地前台脚本已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。本轮使用本机可用 CUDA 设备训练，不是远程服务器。
- 最近一次结果：B 轨道 test 上 RBF KRR RMSE=0.533667、错侧率=0.225000、大幅响应召回=0.750000；keypoint+residual RMSE=0.548994、错侧率=0.125000、大幅响应召回=0.875000、尾段漂移风险=0.075000、反向修正完全匹配率=0.025000；direct Transformer RMSE=0.566011；structured Transformer RMSE=0.602174。按 val RMSE 仍选择 RBF KRR，keypoint+residual 是有价值的结构候选，但还不能冻结成主线。
- 当前最大风险：keypoint+residual 改善了错侧率和大幅响应召回，但 RMSE、峰值时间、启动延迟、困难样本和反向修正仍未全面超过 RBF；不能因为它是结构模型就跳到生理/风格有效性结论。
- 下一步准备做什么：对 keypoint+residual 与 RBF KRR 的坏样本差异做复盘；若继续结构路线，优先尝试多假设车辆-only 或可靠性/困难样本识别，而不是继续堆单一轨迹回归头。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/B_response3s_strict_core_keypoint_residual_bad_samples_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_metrics.csv`。

## 最新更新：2026-05-13 02:10

- 当前阶段：阶段 3 B 轨道车辆-only 响应分解/结构化 Transformer v0.1 已完成；仍处于强车辆基线冻结前的结构化车辆模型筛选阶段。
- 当前正在做什么：归档结构化 Transformer 结果，并判断它是否能替代或补强 B 轨道 RBF KRR / direct Transformer 车辆-only 参照。
- 已完成什么：新增并运行 `stage03_vehicle_instability_structured_vehicle_transformer_v0_1.py`；只在 B 轨道 270 条 3 秒严格核心响应样本上训练结构化车辆-only Transformer，生成指标表、逐样本表、辅助标签准确率表、固定预测图、坏样本图、checkpoint、运行摘要、用户查看版总结和技术报告；同表引入上一轮真正 direct Transformer 指标作参照，KNN 只保留为模板参考。
- 正在运行什么任务：没有后台任务；本轮本地前台脚本已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。本轮使用本机可用 CUDA 设备训练，不是远程服务器。
- 最近一次结果：B 轨道 test 上 RBF KRR RMSE=0.533667、错侧率=0.225000、大幅响应召回=0.750000；direct Transformer RMSE=0.566011、错侧率=0.225000、大幅响应召回=0.625000；结构化 Transformer RMSE=0.602174、错侧率=0.225000、大幅响应召回=0.500000、尾段漂移风险=0.250000。结构化版本只把反向修正完全匹配率从 direct 的 0.050000 小幅提高到 0.075000，但整体、幅值、尾段和困难样本明显更差，不能升级为主线。
- 当前最大风险：如果只看到“结构化辅助头”或反向修正指标小幅变化就继续加复杂结构，会掩盖 RMSE、幅值、大幅响应召回、尾段漂移和困难样本同步变差的问题。当前证据更支持把该版本记录为 no-go/弱候选，而不是进入风格、生理或 EEG 结论。
- 下一步准备做什么：把 B 轨道 RBF KRR 作为当前车辆-only 主参照，direct Transformer 作为深度车辆-only参照，结构化 Transformer v0.1 作为失败/弱候选；下一步若继续车辆结构，应优先尝试关键点 + 残差或多假设车辆-only，而不是直接加入生理。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_structured_vehicle_transformer_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_structured_bad_samples_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_metrics.csv`。

## 最新更新：2026-05-13 01:43

- 当前阶段：阶段 3 干净响应任务车辆-only Transformer v0.1 已完成；仍处于强车辆基线冻结前的模型对照阶段。
- 当前正在做什么：在 A/B 两条干净响应任务轨道上补跑真正的车辆-only Transformer，并与 RBF KRR、KNN template、formal ridge 在固定图、坏样本图和物理指标上对照。
- 已完成什么：新增并运行 `stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1.py`；A 轨道 84 条、B 轨道 270 条均完成 Transformer 训练、早停、test 评估、固定图、坏样本图、指标表、逐样本表、模型 checkpoint、用户查看版总结和技术报告。
- 正在运行什么任务：没有后台任务；本轮本地前台脚本已结束。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。本轮使用本机可用 CUDA 设备训练，不是远程服务器。
- 最近一次结果：B 轨道 test 上 RBF KRR RMSE=0.533667、错侧率=0.225000、大幅响应召回=0.750000；直接 Transformer RMSE=0.566011、错侧率=0.225000、大幅响应召回=0.625000。直接 Transformer 已补跑，但当前不能替代 RBF KRR 主车辆参照。
- 当前最大风险：如果只因为模型名称是 Transformer 就升级为主线，会忽略它在 B 轨道 RMSE、大幅响应召回、启动延迟、尾段和坏样本上的不足；下一步应做响应分解/关键点+残差结构，而不是直接进入生理/风格结论。
- 下一步准备做什么：基于已生成的响应分解标签，设计车辆-only 结构化模型：方向、幅值桶、峰值时间/启动延迟、响应形态和尾段状态辅助头，再比较关键点+残差轨迹是否改善 B 轨道坏样本。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_transformer_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_transformer_bad_samples_test.png`。

## 最新更新：2026-05-13 01:14

- 当前阶段：阶段 3 车辆-only 响应分解标签 v0.1 已完成；仍处于强车辆基线/结构化车辆模型准备阶段。
- 当前正在做什么：把 A/B 两条干净响应任务轨道的未来方向盘轨迹拆成方向、幅值、峰值时间、启动时间、尾段状态、零线穿越和反向/多段修正等目标，供下一步车辆-only 响应分解模型使用。
- 已完成什么：新增并运行 `stage03_vehicle_instability_response_decomposition_labels_v0_1.py`；生成 354 条响应分解样本标签、train-only 阈值表、轨道/split/形态/道路/被试汇总、3 张图、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有训练任务、没有评估任务、没有服务器任务；本轮只是本地 CPU 标签/表格/图生成。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：A 轨道 84 个样本，B 轨道 270 个样本；B 轨道 train/val/test=188/42/40，computed multi-correction 比例 0.9296，reverse/multi 合计比例 0.9889，正负方向比例接近均衡。标签只作为训练目标或评价分层，不能作为模型输入。
- 当前最大风险：这些响应分解标签来自事件后方向盘轨迹，若误放进推理输入、split 条件或标准化条件会造成严重泄漏；下一步必须只把它们作为监督目标/辅助任务/评价维度。
- 下一步准备做什么：在 B 轨道优先做车辆-only 响应分解/Transformer 对照，先预测方向、幅值桶、峰值时间桶、启动延迟桶、响应形态和尾段状态，再比较关键点+残差轨迹是否改善坏样本。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_decomposition_labels_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_sample_labels.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/figures/b_track_mean_gt_trajectories_by_morphology.png`。

## 最新更新：2026-05-13 00:18

- 当前阶段：阶段 3 响应任务定义决策 v0.1 已完成；仍不进入连续风格/生理/EEG 增量验证。
- 当前正在做什么：基于标签窗口覆盖审计，把 906 个高置信失稳事件分成 2 秒即时响应、3 秒响应覆盖、手动锚点/尾段复核、长事件/持续控制复核几个任务轨道。
- 已完成什么：新增并运行 `build_vehicle_instability_response_task_decision_v0_1.py`；生成事件级任务决策表、样本级任务 manifest、任务类别/轨道/split/subject 汇总、2 张图、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有训练任务、没有评估任务、没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：906 个事件中，84 个可作为 2 秒即时响应核心候选，294 个可作为 3 秒响应覆盖候选，其中 270 个是 3 秒严格核心候选；588 个需要长事件/持续控制复核；现有 2718 个窗口样本中，下一轮车辆-only 基线可优先使用的候选窗口样本为 462 个。
- 当前最大风险：如果继续用原来的 906 个事件直接训练 2 秒或 3 秒“完整响应”模型，会把大量长事件/持续控制样本混入核心任务，导致模型失败被误归因到结构或生理缺失。
- 下一步准备做什么：基于 `sample_response_task_manifest.csv` 只在两个干净轨道上重跑车辆-only 对照：A 轨道 2 秒即时响应核心候选，B 轨道 3 秒响应覆盖严格核心候选；D 轨道长事件先不进最终训练结论。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_task_decision_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/event_response_task_decision_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`。

## 最新更新：2026-05-12 22:54

- 当前阶段：阶段 3 标签窗口覆盖审计 v0.1 已完成；仍处于冻结车辆-only 主参照前的样本/标签规则复核阶段。
- 当前正在做什么：把 Top 坏样本里暴露出的“标签窗口可能偏短/长事件未拆分”问题扩展到 906 个正式高置信失稳事件全集，判断 2 秒主标签和 3 秒诊断标签是否覆盖完整方向盘响应。
- 已完成什么：新增并运行 `stage03_vehicle_instability_label_window_coverage_audit_v0_1.py`；生成窗口级覆盖统计、事件级推荐窗口策略、split/subject 汇总、Top 12 坏事件叠加表、3 张图、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有训练任务、没有评估任务、没有服务器任务；本轮只是本地 CPU 表格与图表审计。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：906 个事件中，247 个在 3 秒标签中主峰出现在 2 秒之后，635 个在 2 秒之后仍有明显方向盘变化，822 个被标记为“2 秒标签需要复核”；即使用 3 秒标签，仍有 612 个存在峰值靠近末端或尾段未稳定。Top 12 复发坏事件中，12/12 需要复核 2 秒窗口，9/12 在 3 秒标签下仍需复核。
- 当前最大风险：当前 2 秒标签窗口不能直接作为“完整响应”标签；3 秒标签也不一定解决长失稳或持续控制问题。尾段未回零并不等于样本错误，可能是真实保持转向，所以必须先明确任务定义：即时响应、完整响应，还是持续控制。
- 下一步准备做什么：先形成阶段 2/3 的标签窗口决策：保留 2 秒作为即时响应对照、把 3 秒作为完整响应候选，或把长事件拆成启动响应与持续控制两个任务；在决策前不继续训练风格/生理模型。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_label_window_coverage_audit_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_event_policy_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_policy_counts.png`。

## 最新更新：2026-05-12 22:34

- 当前阶段：阶段 3 复发坏样本失败来源归因 v0.1 已完成；仍处于 vehicle-only 主参照冻结前的错误归因阶段。
- 当前正在做什么：把 Top 12 复发坏样本按锚点/窗口/原始信号/车辆-only 结构不足进行自动初筛归因。
- 已完成什么：新增并运行 `stage03_vehicle_instability_bad_event_failure_attribution_v0_1.py`；生成归因明细表、归因旗标统计、主归因统计、归因热图、主归因计数图、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有训练任务、没有评估任务、没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：Top 12 中 10 个事件优先归为 `sample_rule_or_raw_signal_review`，1 个归为 `vehicle_only_model_structure_gap`，1 个归为 `hard_vehicle_only_case`；次级旗标显示 9/12 存在 `vehicle_only_structure_gap`，10/12 标签窗口可能偏短，11/12 反向修正计数基本不匹配。
- 当前最大风险：自动规则只是初筛，尤其“标签窗口可能偏短”和“事件持续超过标签窗口”需要结合单事件曲线复核；不能直接用这些规则否定样本或宣称生理有效。
- 下一步准备做什么：优先复核 10 个 `sample_rule_or_raw_signal_review` 事件，若大部分确认为窗口/锚点问题则回到阶段 2 修 manifest；若样本规则可信，再进入结构化车辆模型。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_failure_attribution_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_failure_attribution_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/figures/bad_event_failure_attribution_flags.png`。

## 最新更新：2026-05-12 22:18

- 当前阶段：阶段 3 复发坏样本详细曲线复盘 v0.1 已完成；仍处于 vehicle-only 主参照冻结前的错误归因阶段。
- 当前正在做什么：把鲁棒性复盘中 Top 12 反复失败事件转成可直接查看的曲线图，用于判断失败来自锚点/窗口/原始车辆信号还是车辆-only 模型结构不足。
- 已完成什么：新增并运行 `stage03_vehicle_instability_bad_event_curve_review_v0_1.py`；重建 4 个鲁棒性配置的 RBF/KNN/template 预测用于绘图；生成 12 张单事件详细曲线、1 张总览拼图、图索引、模型逐事件误差表、运行摘要、用户查看版总结和技术报告。
- 正在运行什么任务：没有训练任务、没有评估任务、没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：Top 12 事件 * 5 个车辆-only 候选模型共 60 条逐事件误差记录；平均样本 RMSE 从低到高为 KNN template 1.249436、RBF KRR 1.271182、peak-scaled template 1.277831、direction-gated KNN 1.283170、formal ridge 1.470337；严重幅值不足率 0.700、错侧率 0.233、反向修正计数完全匹配率 0.033。
- 当前最大风险：这些曲线显示 RMSE 改善后复杂物理结构仍弱，但仍不能直接归因于模型结构；需要先看曲线里是否存在锚点偏早/偏晚、标签窗口覆盖不足或原始车辆局部异常。
- 下一步准备做什么：基于 Top 12 曲线做人工/规则复核，把失败分成“锚点/窗口问题”“原始信号异常”“车辆-only 信息不足”“需要结构化响应模型”几类，再决定是否进入响应分解、关键点+残差或多假设车辆模型。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_curve_review_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/bad_event_curve_contact_sheet.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_model_error_table.csv`。

## 最新更新：2026-05-12 21:44

- 当前阶段：阶段 3 稳健性坏样本复盘 v0.1 已完成；仍处于车辆-only 失败样本归因阶段。
- 当前正在做什么：归档跨配置/跨模型反复失败事件，为下一步画原始波形和预测曲线做候选清单。
- 已完成什么：从强车辆稳健性逐样本指标中读取 3265 行 test 记录，覆盖 389 个事件、20 个 config-model 对照；生成复发坏样本总表、代表坏样本表、物理错误汇总、分被试坏样本汇总、坏样本矩阵和 4 张图。
- 正在运行什么任务：没有训练任务、没有评估任务、没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：复发最高的坏事件是 `vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000337435`，subject=`hzh`，在 15/15 个可见 config-model 对照中进入 top20 高误差；前 4 个复发坏事件都达到 15/15 或 14/15。
- 当前最大风险：这些坏样本还不能直接归因于模型结构，也可能来自锚点偏差、标签窗口覆盖不足或原始车辆局部异常；必须继续画原始车辆波形和预测曲线确认。
- 下一步准备做什么：对代表坏样本前 10-20 个事件画详细曲线，包含事件锚点、车辆姿态、方向盘 GT、RBF/KNN/template 预测；Transformer 只作为已经单独跑过的参照，必要时另行叠加。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_robustness_bad_sample_review_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_representative_bad_events.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_bad_event_matrix.png`。

## 最新更新：2026-05-12 21:37

- 当前阶段：阶段 3 强车辆基线稳健性验证 v0.1 已完成；仍不进入连续风格/生理/EEG 增量结论。
- 当前正在做什么：归档 RBF/KNN/template 在 random-event、subject-level、1 秒窗口、3 秒窗口下的稳健性结果。
- 已完成什么：复用强车辆-only 训练逻辑，在 4 个配置上重新评估 formal ridge、RBF KRR、KNN template、direction-gated KNN、peak-scaled template；生成决策表、完整指标表、逐样本指标表、模型信息表和 RMSE/大幅响应召回/反向修正热图。
- 正在运行什么任务：没有训练任务、没有评估任务、没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：subject-level 主窗口中，val 选择 RBF KRR，test RMSE=0.609792，相比 formal ridge 0.672788 提升约 9.36%；KNN template test RMSE=0.597936 更低，但 train RMSE=0.000001，继续标记为模板记忆风险。窗口敏感性下，1 秒窗口 RBF test RMSE=0.520104，3 秒窗口 KNN test RMSE=0.590207，但 RBF 反向修正匹配率仍很低。
- 当前最大风险：RBF/KNN 的 RMSE 收益有一定稳健性，但反向修正和多段结构仍弱；KNN 的近零训练误差说明它不能作为无条件主线。车辆-only 主参照需要结合结构化响应模型继续推进。
- 下一步准备做什么：复盘 subject-level 和窗口敏感性下的坏样本，决定下一版车辆模型采用响应分解、关键点 + 残差、多假设或可靠性门控。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_robustness_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_decision_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_rmse_heatmap.png`。

## 最新更新：2026-05-12 21:24

- 当前阶段：阶段 3 车辆-only 统一对照 v0.1 已完成；仍处于冻结强车辆基线前的比较阶段。
- 当前正在做什么：把 formal ridge、旧 `vehicle_direct`、RBF/KNN/template 和真正 Transformer 放入同一套指标与图表，准备提交本轮对照汇总。
- 已完成什么：生成 15 个车辆-only test 模型的统一指标表、相对 formal ridge 差异表、候选决策表、坏样本 top28 重合表、关键指标图、物理错误热图、RMSE/错侧权衡图和坏样本重合图。
- 正在运行什么任务：没有训练任务、没有评估任务、没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：KNN template 的 test RMSE 最低，为 0.516941，但被标记为模板记忆风险；RBF KRR test RMSE=0.540287，大幅响应召回=0.600000，但反向修正匹配率=0.043165；真正 vehicle Transformer test RMSE=0.567162，优于 formal ridge 0.649341 和旧 `vehicle_direct active` 0.637366，但多段修正预测仍为明显短板。
- 当前最大风险：如果只按 RMSE 会错误升级 KNN/RBF；如果只看“是否 Transformer”又会忽略 Transformer 仍漏多段修正。因此阶段 3 还需要 subject-level/window 稳健性和坏样本复盘。
- 下一步准备做什么：基于统一对照表做强车辆基线稳健性验证，重点检查 RBF/KNN 是否模板记忆、Transformer 是否需要响应分解/关键点残差结构。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_unified_vehicle_comparison_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_comparison_metrics_test.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/figures/unified_vehicle_physical_failure_heatmap_test.png`。

## 最新更新：2026-05-12 21:10

- 当前阶段：阶段 3 车辆-only Transformer 时序基线 v0.1 已完成；这次是对用户指出“KNN/RBF 不是 Transformer”的纠正与补充。
- 当前正在做什么：归档真正的车辆-only Transformer 结果、固定预测图、坏样本图、指标表和 Git 提交；仍停留在强车辆基线阶段。
- 已完成什么：在正式高置信失稳样本 `vehicle_instability_highconf_v0_1` 的主窗口 `pre2_label2_old_main` + `session_level_split` 上训练并评估 `vehicle_transformer_context_no_subject`。输入只包含事件前 2 秒车辆时序和事件/道路上下文；不使用驾驶员 ID、生理、脑电、连续风格或未来 `eval_label_*` 字段。
- 正在运行什么任务：没有训练任务、没有评估任务、没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：Transformer 在 session-level test 上 RMSE=0.567162、主峰方向准确率=0.820144、错侧率=0.179856、大幅响应召回=0.440000、严重幅值不足率=0.266187、反向修正计数完全匹配率=0.201439、困难 top20 RMSE=1.089107。相比 formal ridge 的 RMSE=0.649341 有提升；但它仍不如上一轮 RBF/KNN 的最低 RMSE，且仍不会预测多段修正，不能直接升级为最终主线。
- 当前最大风险：Transformer 改善了 formal ridge 的整体和若干物理指标，但仍存在大峰值漏预测、多段修正预测为 0 的问题；KNN/RBF 结果虽然 RMSE 更低，但有模板记忆或物理错误风险。因此下一步需要统一比较强车辆候选，而不是只按 RMSE 选主线。
- 下一步准备做什么：把 formal ridge、旧 `vehicle_direct`、RBF/KNN 和 Transformer 放入统一阶段 3 对照表，结合固定图/坏样本图判断强车辆基线主参照；在此之前继续阻塞连续风格、生理和 EEG 有效性结论。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_transformer_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_bad_samples_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_metrics.csv`。

## 最新更新：2026-05-12 20:23

- 当前阶段：阶段 3 更强车辆-only 时序/结构化基线 v0.1 已完成，仍处于“先把纯车辆基线压实”的阶段。
- 当前正在做什么：归档强车辆-only 基线结果、图表、报告和 Git 提交；没有进入连续风格、生理或 EEG 增量验证。
- 已完成什么：在正式高置信失稳样本 `vehicle_instability_highconf_v0_1` 的主窗口 `pre2_label2_old_main` + `session_level_split` 上，评估 formal ridge、rich ridge、RBF kernel ridge、KNN 模板、方向门控模板、峰值缩放模板；全部不使用驾驶员 ID、生理、脑电或连续风格。
- 正在运行什么任务：无训练任务、无服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果：按预设 val RMSE 选择 `rbf_kernel_ridge_context_no_subject`；它在 test 上 RMSE=0.540287、错侧率=0.215827、大幅响应召回=0.600000、严重幅值不足率=0.251799、反向修正精确匹配率=0.043165。test RMSE 最低的是 `knn_template_context_no_subject`，RMSE=0.516941，但不是按 val 选择出的模型，因此只能作为候选观察，不用 test 事后升级。
- 当前最大风险：RBF/KNN 已明显改善 RMSE、幅值不足和大幅响应召回，但反向修正仍很差，RBF 的反向修正不匹配 133/139；KNN 模板 train RMSE 接近 0，存在模板记忆风险，需要跨窗口、subject-level split 或更严格对照后才能升级为主线。
- 下一步准备做什么：先做强车辆基线稳健性验证，包括 subject-level/窗口敏感性、KNN 模板过拟合检查、以及面向反向修正/多段修正的结构化响应模型；继续阻塞风格/生理有效性结论。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_baselines_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_bad_samples_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/strong_vehicle_baseline_metrics.csv`。

## 最新更新：2026-05-12 19:35

- 当前阶段：阶段 3 车辆-only 基线错误分型，已完成 `ridge_vehicle_context_no_subject` 在 test 集上的坏样本物理错误分析。
- 当前正在做什么：归档错误分型表、图和中文报告。
- 已完成什么：对主窗口 `pre2_label2_old_main` + session-level test 的 139 个样本生成逐样本错误标签，并与旧 `vehicle_direct` clean 对照做样本级比较。
- 正在运行什么任务：没有训练任务；没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件。
- 最近一次结果：反向修正计数不匹配 126/139，尾段漂移 87/139，严重幅值不足 81/139，启动延迟大误差 75/139，多段修正结构不匹配 46/139，其中过度预测多段 42/139、漏检多段 4/139，错侧 32/139，大幅响应漏召回 23/139。旧 deep 与 formal ridge 的 top20%坏样本重叠 21/28。
- 当前最大风险：车辆-only 浅层模型的主要问题是复杂响应结构和尾段/幅值，而不是单一平均 RMSE；不能据此跳到“生理会解决”的结论。
- 下一步准备做什么：先建立更强的车辆时序或结构化响应基线，再考虑连续风格和生理增量验证。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_error_analysis_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/formal_error_flag_counts.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/top_bad_sample_error_matrix.png`。

## 最新更新：2026-05-12 19:20

- 当前阶段：阶段 3 初始车辆-only 基线，已在正式 `vehicle_instability_highconf_v0_1` 样本上完成无学习基线和浅层车辆 ridge 基线。
- 当前正在做什么：整理阶段 3 v0.1 结果、图表和提交。
- 已完成什么：在主窗口 `pre2_label2_old_main` + session-level test 上评估零响应、历史趋势、训练均值、按事件均值、`ridge_vehicle_history_no_subject`、`ridge_vehicle_context_no_subject`；全部不使用生理、脑电、连续风格或驾驶员 ID。
- 正在运行什么任务：没有训练任务；没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件。
- 最近一次结果：`ridge_vehicle_context_no_subject` 在主窗口 session-level test 上 RMSE=0.649341、方向准确率=0.769784、错侧率=0.230216、大幅响应召回=0.080000、严重幅值不足率=0.582734、反向修正计数完全匹配率=0.093525。它略差于旧 `vehicle_direct` clean 对照 RMSE=0.637366，但训练边界更清楚。
- 当前最大风险：车辆-only 浅层基线仍然明显无法覆盖大幅响应、多段修正和部分方向；这说明还不能进入“风格/生理有效”的结论，只能继续强化车辆基线和错误分型。
- 下一步准备做什么：检查固定图/坏样本图中的主要失败类型，决定是否先增强车辆模型结构，或建立更强的无泄漏车辆时序模型。
- 用户可以优先查看：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_formal_baselines_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_fixed_predictions_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_bad_samples_test.png`。

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
## 追加更新：2026-05-12 20:30

- 当前阶段：阶段 2 补充审计，正在把“场景交通触发点”进一步拆成“背景交通设计”和“被试方向事件设计”两层。
- 当前正在做什么：根据用户补充，修正 `longstraight` 25/26 车道交通触发的解释，并把其他场景统一纳入被试方向事件来源工作表。
- 已完成什么：新增 `ego_direction_scene_event_source_map_v0_3.csv` 和 `scene_design_working_map_v0_3_cn.md`。当前判断是：显式 `.aed` 交通触发点不能直接作为被试方向主锚点；其他场景有实验设计，但主锚点更可能来自道路几何、路面附着、任务点或车身姿态确认。
- 最近一次结果：全场景 1436 行 `.aed` 交通触发映射中，当前没有一行被判定为被试方向同侧主触发；这不说明场景无事件，只说明交通触发不是主事件锚点。
- 当前最大风险：如果继续把背景车/对侧车触发点当成被试方向锚点，会导致样本语义错位；如果只用车身姿态或方向盘反推锚点，又可能把响应当成触发。
- 下一步准备做什么：继续查旧论文/实验设计日志，结构化解析 `_Area2.cfg` 中道路几何和 `mu` 信息，生成“被试方向设计点 + 车身姿态确认”的候选锚点清单。
- 用户可以优先查看：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_design_working_map_v0_3_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/ego_direction_scene_event_source_map_v0_3.csv`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_scene_trigger_user_summary_cn.md`
## 追加更新：2026-05-12 20:40

- 当前阶段：阶段 2 补充审计，已完成“被试方向设计点与候选锚点重建 v0.4”。
- 当前正在做什么：不训练模型，基于用户提供的小论文、道路配置和已有车辆轨迹投影，整理每类场景更合理的候选锚点来源。
- 已完成什么：提取小论文中的场景设计依据，解析 `_Area2.cfg` 中车道和 `mu` 信息，生成 2401 行候选锚点/上下文点，并输出中文报告和用户查看版总结。
- 最近一次结果：弯道和低附着两类证据最清楚；`curve1/curve2` 可优先比较道路入口、横滚峰值和横向加速度峰值；`differentmu_road` 可优先比较低 `mu` 进入点和 `mu` 跳变点。`longstraight` 25/26 背景车流不作为被试方向主锚点。`fix_road`、`stop`、`zd` 仍需更具体实验设计说明或可视化复核。
- 当前最大风险：候选锚点很多，但不能直接全部进入训练；如果不做可视化复核，仍可能把车身响应峰值或方向盘动作误当成事件触发点。
- 下一步准备做什么：生成候选锚点可视化图，优先检查 `curve1/curve2` 和 `differentmu_road`，并把旧锚点分成“可保留、偏早、偏晚、语义不清”。
- 用户可以优先查看：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_ego_direction_design_anchor_user_summary_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/ego_direction_design_anchor_rebuild_v0_4_cn.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/small_paper_scene_design_extract_v0_4.md`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`
  5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`

## 追加更新：2026-05-12 20:55

- 当前阶段：阶段 2 补充审计，已根据用户说明修正 `middle_section` 的事件语义。
- 当前正在做什么：不训练模型，修正候选锚点规则，把道路连接段从“背景/过渡段”改为“连续超车负荷事件段”。
- 已完成什么：更新 `build_ego_direction_design_anchors_v0_4.py`，重新生成候选锚点表。候选总数从 2401 行更新为 4209 行；其中 `middle_section` 现在有 2260 行连续超车候选，包括入口、中点、横向偏移变化峰值、横向加速度峰值、横摆角速度峰值各 452 行。
- 最近一次结果：`middle_section` 已进入优先可视化复核列表，和弯道、低附着一起作为三类最值得先重建锚点的场景。
- 当前最大风险：连接段是连续负荷，不是单点突发事件；如果把所有连接段入口都当强事件，会引入弱响应/无响应样本。因此必须用横向偏移、横摆和横向加速度筛选。
- 下一步准备做什么：优先生成 `middle_section` 连续超车锚点可视化图，再画 `curve1/curve2` 和 `differentmu_road`。
- 用户可以优先查看：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/middle_section_continuous_overtaking_correction_20260512_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_ego_direction_design_anchor_user_summary_cn.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`

## 追加更新：2026-05-12 21:20

- 当前阶段：阶段 2 补充审计，已根据用户最新说明修正 `longstraight` 和维修路段的变道触发语义。
- 当前正在做什么：不训练模型，继续把道路/场景设计触发点与候选锚点清单对齐。
- 已完成什么：更新 `build_ego_direction_design_anchors_v0_4.py`，将 `longstraight` 的 MAN TGL 25->26 显式变道、Chrysler300 Stop，以及 `fix_road` 的 MAN TGL 25->26、BMW m340 26->25 显式变道纳入候选锚点。
- 最近一次结果：候选锚点总数从 4209 行更新为 4519 行。`longstraight` 现在有场景上下文入口 85 行、显式变道触发点 85 行、显式停车触发点 85 行；`fix_road` 现在有显式变道触发点 140 行。
- 当前最大风险：显式触发点仍不能直接等同最终训练锚点，必须检查触发点附近是否有被试车辆横向/纵向响应；否则仍可能把无响应背景事件混入训练样本。
- 下一步准备做什么：优先生成 `middle_section`、`longstraight`、`fix_road`、`curve1/curve2`、`differentmu_road` 的候选锚点可视化图，比较设计触发点、车身姿态峰值、旧锚点和方向盘响应。
- 用户可以优先查看：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/longstraight_fixroad_lanechange_trigger_correction_20260512_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_ego_direction_design_anchor_user_summary_cn.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`

## 追加更新：2026-05-12 21:45

- 当前阶段：阶段 2 补充审计，已完成事件候选自动筛选 v0.5。
- 当前正在做什么：没有训练模型；当前把 v0.4 的 4519 个候选锚点按“设计证据 + 车身响应 + 可训练窗口 + 旧锚点接近程度”进行自动评分和分层。
- 已完成什么：新增 `filter_event_anchor_candidates_v0_5.py`，输出全部候选评分表、去重后复核清单、高置信复核清单、分场景统计、概览图和 56 张代表性复核图。
- 最近一次结果：4519 个候选中，去重后建议复核 534 个，高置信复核 314 个。高置信复核主要来自 `middle_section` 连续超车段入口 80 个、`curve1/curve2` 道路模块入口 103 个、`fix_road` 显式变道 31 个、`differentmu_road` 低附着/μ 变化候选 80 个、`longstraight` 显式停车 17 个和显式变道 3 个。
- 当前最大风险：自动筛选只是第一关，不能直接当最终训练样本；特别是车身响应峰值只作确认点，不能直接定因果锚点。
- 下一步准备做什么：查看代表性复核图，把候选分成“可进入样本清单、偏早、偏晚、无明显响应、语义不清”，再生成 v0.6 样本清单。
- 用户可以优先查看：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_event_filter_user_summary_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_candidate_filter_v0_5_cn.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidates_high_confidence_v0_5.csv`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/figures/event_candidate_filter_overview_v0_5.png`
  5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/figures/review_panels`

## 追加更新：2026-05-12 21:55

- 当前阶段：阶段 2 补充审计材料整理，已完成给 GPTPro 的事件锚点审查证据包。
- 当前正在做什么：没有训练模型，没有服务器任务。当前只整理可发送材料。
- 已完成什么：把事件筛选报告、核心表格、概览图、按场景精选复核图、`longstraight/fix_road` 修正说明和 v0.4/v0.5 用户总结打包。
- 最近一次结果：压缩包已生成，大小约 3.6 MB，内部 31 个条目，已检查可打开。
- 当前最大风险：GPTPro 回复只能作为外部审查意见，不能替代本地可视化复核和后续样本清单验证。
- 用户可以优先查看或发送：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512.zip`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512/00_README_CN.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512/01_GPTPRO_PROMPT_CN.md`

## 追加更新：2026-05-12 22:35

- 当前阶段：阶段 2 补充审计，已收到并归档 GPTPro 对 v0.4/v0.5 事件锚点分析的回复。
- 当前正在做什么：没有训练模型；正在把 GPTPro 的审查意见转成 v0.6 样本筛选规则。
- 已完成什么：归档 GPTPro 回复摘要、决策记录和行动项；新增 `event_v0_6_screening_rule_from_gptpro_20260512_cn.md`。
- 最近一次结果：GPTPro 支持当前暂停盲目改模型、优先重审事件锚点和样本清单；建议 v0.6 只先拿小而干净的核心样本，优先 `curve1/curve2`、`differentmu_road raw μ`、人工通过的 `fix_road`，暂缓 `middle_section`、`longstraight`、`stop`、`curve3/zd` 进入第一版训练。
- 当前最大风险：不能把 GPTPro 建议直接当结论，仍需要本地复核图和 v0.6 表格验证。
- 下一步准备做什么：基于 56 张代表性复核图生成复核标注表，并按 `pass/early/late/weak_response/continuous/coordinate_issue/unclear/exclude` 标注，随后生成 v0.6 四类事件表。
- 用户可以优先查看：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_response_manualpaste.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_decision_filled.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_action_items_filled.md`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_v0_6_screening_rule_from_gptpro_20260512_cn.md`
## 最新更新：2026-05-13 00:37

- 当前阶段：阶段 3，干净响应任务车辆-only 对照验收完成；仍然没有进入连续风格、生理或 EEG 有效性验证。
- 当前正在做什么：整理 `A_instant2s_core` 和 `B_response3s_strict_core` 两条干净任务轨道的车辆-only 基线结果，并把结果写入项目看板、任务队列、产物索引和每日日志。
- 已完成什么：新增并运行 `stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1.py`；在 84 个 2 秒即时响应核心候选和 270 个 3 秒响应覆盖严格核心候选上评估零响应、趋势外推、训练均值、事件均值、ridge、rich ridge、RBF KRR、KNN template、direction-gated KNN 和 peak-scaled template。
- 正在运行什么任务：无。当前没有训练、没有后台脚本、没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取 `服务器指令与密码.txt`，未记录任何凭据。
- 最近一次结果是什么：A 轨道按 val RMSE 选择 `knn_template_context_no_subject`，test RMSE=0.428130，错侧率=0.333333，大幅响应召回=0.600000；但 A 轨道 test 只有 12 个事件，且 KNN train RMSE 近 0，只能作为诊断候选。B 轨道按 val RMSE 选择 `rbf_kernel_ridge_context_no_subject`，test RMSE=0.533667，错侧率=0.225000，大幅响应召回=0.750000，严重幅值不足率=0.125000；但反向修正计数完全匹配率仍为 0.000000，坏样本图显示多段/反向响应仍明显不足。
- 当前最大风险是什么：如果直接把 KNN 在小样本 A 轨道上的 val/test 表现当作主结论，会把模板记忆风险误判为泛化能力；如果只看 B 轨道 RMSE，又会忽略反向修正、多段修正和长事件仍未解决。
- 下一步准备做什么：优先复查 B 轨道固定图和坏样本图，决定是否把 B 轨道 RBF KRR 作为下一轮结构化车辆-only 参考；A 轨道暂作为即时响应诊断，不升级为主线结论。继续阻止风格/生理/EEG 有效性结论，直到强车辆基线和物理错误闭环更稳定。
- 用户可以优先查看哪些文件：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_baselines_user_summary_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_metrics.csv`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/clean_task_vehicle_metric_summary_test.png`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/B_response3s_strict_core_bad_samples_test.png`
## 最新更新：2026-05-13 00:55

- 当前阶段：阶段 3，B 轨道车辆-only 坏样本物理复查完成；仍未进入连续风格、生理或 EEG 有效性验证。
- 当前正在做什么：把 `B_response3s_strict_core` 上 val 选中的 `rbf_kernel_ridge_context_no_subject` 的 test 坏样本失败类型整理成表、图和中文报告。
- 已完成什么：新增并运行 `stage03_vehicle_instability_clean_task_bad_sample_review_v0_1.py`；只分析 B 轨道 test 40 个样本，不训练新模型，不使用生理、脑电、连续风格、驾驶员 ID 或服务器。
- 正在运行什么任务：无。当前没有训练、没有后台脚本、没有服务器任务。
- 服务器是否在运行：未使用服务器，未读取 `服务器指令与密码.txt`，未记录任何凭据。
- 最近一次结果是什么：B 轨道 RBF KRR 的 high-RMSE top20% 阈值为 sample RMSE>=0.657，共 8 个坏样本；全部 40 个 test 样本中错侧 9/40，严重幅值不足 5/40，大幅响应漏召回 2/40，峰值时间大误差 9/40，启动延迟大误差 7/40，反向修正计数不匹配 40/40。最差 8 个样本里 wrong-side 3/8，严重幅值不足 3/8，大幅响应漏召回 2/8，峰值时间大误差 4/8。
- 当前最大风险是什么：如果只看 B 轨道 RBF KRR 的整体 RMSE，会掩盖反向修正计数全不匹配这一结构性失败；下一步不能直接用生理/风格解释增益，必须先建立结构化车辆-only 响应分解参考。
- 下一步准备做什么：进入车辆-only 响应分解模型设计：优先预测方向、幅值、峰值时间、反向修正/多段修正类型，再预测轨迹；同时保留 B 轨道坏样本表作为固定复查样本。
- 用户可以优先查看哪些文件：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_bad_sample_review_user_summary_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_failure_summary.csv`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_top_bad_samples.csv`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/figures/b_track_rbf_failure_flag_rates.png`
  5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/figures/b_track_rbf_peak_amp_scatter.png`

## 最新更新：2026-05-13 01:05

- 当前阶段：阶段 2 回补，按 GPTPro 最新建议重新生成 `episode-first` 事件样本 v0.6；当前先不继续训练。
- 当前正在做什么：把事件筛选逻辑从“设计触发点是不是事件”改为“是否真实出现车辆动态异常、方向盘响应和回正/纠正 episode”，并把样本分为严格核心、坐标需复核扩展、弱响应/负样本、连续任务复核、场景暂缓复核和不确定复核。
- 已完成什么：更新并运行 `build_episode_first_events_v0_6.py`；脚本优先用横向加速度、横摆角速度和横滚速率构造非横向偏移动态强度，避免横向偏移坐标跳变污染 episode 判定；同时把横向偏移坐标跳变作为复核标记，而不是简单删除所有样本。
- 正在运行什么任务：无。当前没有训练、没有后台脚本、没有服务器任务。
- 服务器是否在运行：本轮未使用服务器，未读取 `服务器指令与密码.txt`，未记录任何凭据。
- 最近一次结果是什么：从 908 个高置信车辆动态 episode 中，得到 19 个第一版最干净核心训练候选、246 个“车辆动态和方向盘响应成立但横向偏移坐标需复核”的扩展候选、298 个车辆动态明显但方向盘响应不足的弱响应/负样本、306 个连续超车任务复核样本、30 个场景语义暂缓复核样本和 9 个因果顺序不清样本。
- 当前最大风险是什么：严格核心样本只有 19 个，直接训练完整轨迹模型样本太少；扩展候选数量足够，但必须人工确认坐标跳变是不是道路坐标重置，并在后续车辆-only 训练中考虑不使用横向偏移特征或单独标记这类样本。
- 下一步准备做什么：先人工查看 36 张分组代表图，确认严格核心、坐标复核扩展和弱响应负样本是否符合直觉；若代表图通过，再用严格核心和“去掉横向偏移特征的扩展候选”分别构建纯车辆/道路预测对照。
- 用户可以优先查看哪些文件：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_episode_first_v0_6_user_summary_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/episode_first_event_v0_6_cn.md`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_candidates_v0_6.csv`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/primary_training_events_v0_6.csv`
  5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/coordinate_flagged_expansion_events_v0_6.csv`
  6. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_decision_summary_v0_6.csv`
  7. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/figures/episode_first_v0_6_summary.png`
  8. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/figures/episode_review_panels`

## 最新更新：2026-05-13 01:29

- 当前阶段：阶段 3，episode-first v0.6 样本的纯车辆/道路预测对照完成；仍未进入连续风格、生理或 EEG 有效性验证。
- 当前正在做什么：用 v0.6 的“严格核心 + 坐标需复核扩展候选”正样本检查纯车辆/道路模型是否比旧混合样本更好，并评估横向偏移坐标是否造成虚高。
- 已完成什么：新增并运行 `stage03_episode_first_vehicle_baselines_v0_1.py`；评估 3 条轨道：2 秒扩展不使用横向偏移、3 秒扩展不使用横向偏移、3 秒扩展保留横向偏移作风险诊断。每条轨道 265 个事件，session-level split 为 train/val/test=183/37/45。
- 正在运行什么任务：无。当前没有训练、没有后台脚本、没有服务器任务。
- 服务器是否在运行：本轮未使用服务器，未读取 `服务器指令与密码.txt`，未记录任何凭据。
- 最近一次结果是什么：旧 B 轨道 RBF KRR 的 test RMSE=0.533667、错侧率=0.225000、大幅响应召回=0.750000；本轮主轨道 `EP3_expanded_no_lateral_3s` 按 val 选择 formal ridge，test RMSE=0.679927、错侧率=0.266667、大幅响应召回=0.250000、严重幅值不足率=0.355556；保留横向偏移的 3 秒轨道 test RMSE=0.680265，未带来改善。
- 当前最大风险是什么：如果只看“新样本没有提升 RMSE”，可能误判为 v0.6 无效；更合理的解释是 episode-first 样本更集中在真实大幅响应、回正、反打和复杂修正上，车辆-only 线性/模板模型更难处理。当前不能说新样本让预测变好，只能说新样本把目标事件筛得更接近研究目标，同时暴露出车辆-only 模型能力不足。
- 下一步准备做什么：不建议马上加生理或连续风格补偿；先基于 v0.6 正样本做车辆-only 响应分解模型，把方向、幅值、峰值时间、回正/反打、多段修正先预测清楚，再进入轨迹回归。
- 用户可以优先查看哪些文件：
  1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_episode_first_vehicle_baselines_user_summary_cn.md`
  2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_metrics.csv`
  3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_val_selected_models.csv`
  4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/figures/episode_first_vehicle_metric_summary_test.png`
  5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/figures/EP3_expanded_no_lateral_3s_bad_samples_test.png`
# R2E-Steering 项目总进度看板
## 最新更新：2026-05-13 06:50

- 当前阶段：Stage 7c 候选轨迹导出与差异审计 v0.1 已完成；当前仍不升级多假设为主线。
- 当前正在做什么：基于已导出的候选轨迹，准备判断下一步应改非 oracle selector，还是重新设计候选生成方式。
- 已完成什么：新增并运行 `stage07c_candidate_trajectory_export_v0_1.py`；加载已有 RBF/KNN、keypoint residual 和 top-K checkpoint，导出 270 个 B 轨道严格核心失稳样本的候选轨迹 npz；生成候选指标表、逐样本指标、候选两两差异、候选特征与标签诊断、oracle 摘要、gate 表、5 张图、用户查看版总结和技术报告；已提交 `48b8c438 Add stage7c candidate trajectory export`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：B 轨道 test 上 RBF/KNN RMSE=0.533667；top-K top1 RMSE=0.587865；RBF+topK oracle RMSE=0.415652，delta=-0.118014；broad oracle RMSE=0.410957，delta=-0.122710。oracle 上限明显，但 Stage 7c 没有训练非 oracle selector，不能作为可部署提升。
- 当前最大风险是什么：候选池存在事后上限，但可部署选择机制仍没有解决；如果直接把 best-of-K/oracle 结果当模型性能，会高估当前车辆-only 能力，也会错误放行生理/EEG。
- 下一步准备做什么：优先做 Stage 7d 非 oracle selector v0.2，使用 Stage 7c 导出的候选差异特征并严格 train/val/test 隔离；若仍退回 RBF/KNN，则转向重新设计候选生成方式，而不是进入生理/EEG。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07c_candidate_trajectory_export_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_export_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_metric_summary_test.png`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/arrays/stage07c_candidate_trajectories.npz`。
# R2E-Steering 项目总进度看板
## 最新更新：2026-05-13 06:58

- 当前阶段：Stage 7d 非 oracle selector v0.2 已完成；gate=`no_upgrade`。
- 当前正在做什么：收口当前多候选选择器路线，准备转向候选生成方式本身，而不是继续堆选择器。
- 已完成什么：新增并运行 `stage07d_non_oracle_selector_v0_2.py`；只用 Stage 7c 的候选预测特征和道路/事件上下文训练 selector；显式排除 test 标签、oracle 特征、subject/session ID、生理、脑电和连续风格；生成 feature audit、allowed features、policy metrics、decision diagnostics、selected decisions、confusion、gate 表、3 张图、用户查看版总结和技术报告；已提交 `eb785f4a Add stage7d non-oracle selector`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：val gate 选择 `always_rbf_reference`；test RMSE=0.533667，与 RBF/KNN 完全相同；test RBF 选择比例=1.000；broad oracle test RMSE=0.410957 仍只是诊断上限。Stage 7d 不能升级为多假设主线。
- 当前最大风险是什么：当前候选池有上限，但非 oracle 特征无法稳定选择；继续堆 selector 很可能只得到“安全退回 RBF/KNN”或“test 退化”的结果。
- 下一步准备做什么：Stage 7e 候选生成重设计协议：把候选显式绑定到方向、幅值、峰值时间、尾段回正/漂移、反向修正和多段修正，而不是先用现有 branch 再补 selector。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07d_non_oracle_selector_v0_2_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_validation_rmse_delta.png`。
# R2E-Steering 项目总进度看板
## 最新更新：2026-05-13 07:05

- 当前阶段：Stage 7e 候选生成重设计审计 v0.1 已完成；当前不再继续 selector-only 路线。
- 当前正在做什么：准备按 response-factorized candidate 思路实现下一版车辆-only 多候选生成，而不是继续在旧 branch 上训练选择器。
- 已完成什么：新增并运行 `stage07e_candidate_generation_redesign_v0_1.py`；从真实方向盘标签提取方向、幅值、峰值时间、尾段、反向修正/多段修正响应类型；用 Stage 7c 候选轨迹审计每类响应的 RBF/KNN 误差、候选 oracle 误差、oracle gain 和 winner 分布；生成候选生成蓝图、下一实验计划、gate 表、4 张图、用户查看版总结和技术报告；已提交 `98552bf3 Add stage7e candidate generation redesign`。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：test RBF/KNN RMSE=0.533667；现有可部署候选池 oracle RMSE=0.410957，delta=-0.122710；test 非 RBF oracle winner 比例=0.700；coverage 中 16 个 test bucket 属于 `selector_gap_candidate_pool_has_signal`，说明候选池有信号但当前 selector-only 路线失败。
- 当前最大风险是什么：如果继续只堆 selector，会反复得到退回 RBF/KNN 的安全策略；如果直接训练复杂多假设模型但不绑定响应物理类型，仍可能生成平滑相似候选。
- 下一步准备做什么：Stage 7f 实现 response-factorized vehicle-only candidate v0.1：保留 RBF/KNN anchor，并增加方向/幅值、峰值时间、尾段模式、反向修正/多段修正和可靠性门控候选。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07e_candidate_generation_redesign_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_candidate_generation_blueprint.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_candidate_generation_blueprint.png`。


## 最新更新：2026-05-18 v0.3 车辆-only 基线

- 当前阶段：v0.3 全量原始数据样本库后的车辆-only 数据集与基线验证。
- 已完成：构建 `v03_vehicle_only_pre2_label5_20hz`，可用样本 482，split={'train': 280, 'test': 114, 'val': 88}。
- 最近结果：当前 test 最好模型 `rbf_kernel_vehicle_context_alpha0.1_g2`，RMSE=0.797252，主响应 RMSE=0.600911，尾段 RMSE=0.905429。
- 下一步：优先查看固定预测图和坏样本图；如果物理意义可接受，再考虑响应类型辅助模型或加入连续风格/生理增量。


## 最新更新：2026-05-18 v0.3 车辆-only 数据集与基线（中文修正版）

- 当前阶段：基于全量原始车辆数据重筛 episode 后，构建车辆-only 数据集并运行无学习基线和车辆-only 强基线。
- 已完成：`v03_vehicle_only_pre2_label5_20hz`，可用样本 482，train/val/test=280/88/114。
- 最近结果：test 最好模型 `rbf_kernel_vehicle_context_alpha0.1_g2`，RMSE=0.797252，主响应 RMSE=0.600911，尾段 RMSE=0.905429。
- 当前判断：车辆-only 比零响应有小幅总体提升，并明显降低大响应错侧率，但大响应召回仍不足；这还不是风格或生理有效性的证据。
- 下一步：优先查看固定预测图、坏样本图、分类型和分被试表，再决定是否调整样本/锚点，或进入响应类型辅助建模。


## 最新更新：2026-05-18 v0.3 车辆-only 数据集与基线（中文修正版）

- 当前阶段：基于全量原始车辆数据重筛 episode 后，构建车辆-only 数据集并运行无学习基线和车辆-only 强基线。
- 已完成：`v03_vehicle_only_pre2_label5_20hz`，可用样本 482，train/val/test=280/88/114。
- 最近结果：test 最好模型 `rbf_kernel_vehicle_context_alpha0.1_g2`，RMSE=0.797252，主响应 RMSE=0.600911，尾段 RMSE=0.905429。
- 当前判断：车辆-only 比零响应有小幅总体提升，并明显降低大响应错侧率，但大响应召回仍不足；这还不是风格或生理有效性的证据。
- 下一步：优先查看固定预测图、坏样本图、分类型和分被试表，再决定是否调整样本/锚点，或进入响应类型辅助建模。

## 最新更新：2026-05-18 v0.3 样本纳入范围消融

- 当前阶段：在不改模型结构的前提下，逐步加入待复核和可成窗排除样本，检查车辆-only 基线是否受益。
- 已完成：3 档纳入范围对照，当前总 test RMSE 最低的是 `v03_plus_review`，RMSE=0.592277。
- 当前判断：该结果只能回答“加样本是否改善车辆-only 基线”，不能证明连续风格或生理数据有效。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_vehicle_only_inclusion_ablation_user_summary_cn.md`。

## 2026-05-19 v0.3 excluded 分层加入实验

- 当前阶段：车辆-only 基线样本纳入范围审查。
- 已完成：服务器跑完 7 个 excluded 分层加入版本，并拉回本地结果。
- 当前结论：`干净集 + 待复核` 仍是整体最稳范围；`excluded` 不能全量加入，但横滚/姿态类有保留价值，适合作为风险池继续复核。
- 本轮结果目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion`
- 用户查看版报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_v03_excluded_stratified_inclusion_user_summary_cn.md`
- 服务器连接格式：`ssh -p 55060 root@connect.westc.seetacloud.com`，密码未写入项目文件。

## 2026-05-19 横滚/姿态 excluded paired 诊断

- 当前阶段：检查横滚/姿态 excluded 版本是否真的改善大响应物理问题。
- 已完成：共同测试样本 paired 对比、新增横滚/姿态 excluded 测试样本统计、改善/恶化样本对比图。
- 用户查看版报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_v03_roll_excluded_pair_diagnosis_user_summary_cn.md`。
- 输出目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis`。

## 2026-05-19 v0.3 临时加入锚点后响应弱样本

- 当前阶段：车辆-only 样本范围继续审查，不涉及连续风格、生理或脑电。
- 本轮动作：在“干净集 + 待复核”基础上，临时加入 `FAST_STEER_WEAK_POST_RESPONSE` 样本，并重跑车辆-only 基线。
- 当前整体 RMSE 最低版本：`v03_weakpost_with_lateral`，test RMSE=0.5889。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_fast_weakpost_temp_train_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train`。
- 注意：这是探索性样本合并实验，不能直接证明最终样本定义正确；需要继续看预测图和大响应物理指标。
## 2026-05-19 v0.3 样本筛选策略连续对比已启动

- 当前阶段：车辆-only 样本筛选策略对比，不涉及连续风格、生理或脑电。
- 当前正在做什么：已在服务器启动 `stage03_v03_screening_sweep.py`，连续比较锚点后响应弱、快速转向、低附着、横滚/姿态、弯道、自动候选标签等多种额外纳入策略。
- 服务器是否在运行：是，screen 名称 `v03sweep`。
- 远程项目路径：`/root/autodl-tmp/data_process`。
- 远程日志路径：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_20260519_203455.log`。
- 最近状态：任务已启动，当前从 `s00_base_nolat` 基准版本开始运行。
- 当前最大风险：这是 CPU/表格基线批量计算，单个版本可能耗时较长；若某一类样本路径或窗口不完整，会在日志中报错并需要修正后续跑。
- 用户可查看：服务器日志路径如上；本地最终结果待任务完成后拉回。

## 2026-05-19 v0.4 极限工况样本重新筛选

- 为什么做：用户指出目标不是继续比较 809 样本版本，而是回到 1574 个初始 episode，按方向盘速度、锚点延时性、锚点后车辆/驾驶员是否仍有变化重新筛选。
- 本轮规则：锚点后车辆有变化即保留，即使驾驶员操作弱；锚点后车和驾驶员都弱则排除；快打方向但车辆变化弱先复核。
- 当前结果：主训练候选 1128，次级训练候选 101，待复核 193，排除 152。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_refilter_v0_4_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4`。

## 2026-05-20 v0.4 重筛样本车辆-only GPU 基线

- 为什么做：用户要求在 v0.4 从 1574 个初始 episode 重筛后的样本上继续跑车辆-only，不使用服务器，直接使用本地 GPU。
- 运行设备：`cuda`。
- 当前综合排序第一：`v04_primary_secondary_nolat`，test RMSE=0.8402，大响应错侧率=0.1707，严重幅值不足率=0.2683，大响应召回=0.9512。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v04_vehicle_only_gpu_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline`。
## 2026-05-20 v0.4 主训练+次级+待复核 GPU 基线

- 为什么做：用户要求在服务器 GPU 上继续跑 v0.4 主训练+次级+待复核样本，检查待复核样本是否能纳入训练。
- 运行位置：服务器，连接格式 `ssh -p 55060 root@connect.westc.seetacloud.com`，密码未写入项目文件。
- 运行设备：NVIDIA GeForce RTX 4080 SUPER。
- 运行版本：`v04_primary_secondary_review_nolat`，即主训练候选 + 次级候选 + 待复核样本，去掉横向偏移。
- 当前结果：原始合并样本 1422，实际可用窗口样本 1410，train/val/test=831/244/335。
- 测试指标：test RMSE=0.8067，主阶段 RMSE=0.5786，尾段 RMSE=0.9290，大响应错侧率=0.1398，严重幅值不足率=0.2796，大响应召回=0.9032。
- 当前判断：加入待复核样本后，整体 RMSE 明显优于上一轮 `v04_primary_secondary_nolat` 的 0.8402，错侧率也下降；但大响应召回下降，说明待复核样本有价值，但可能也稀释了一部分大响应覆盖。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v04_review_gpu_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_review_gpu_baseline`。
- 当前后台任务：服务器 screen 已结束，本地没有继续运行的该任务。


## 2026-05-20 13:48:32 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`


## 2026-05-20 13:58:46 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`


## 2026-05-20 15:10:44 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`


## 2026-05-20 15:14:24 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`

## 2026-05-20 15:16 v0.5 生理机制验证阶段结论

- 当前阶段：在 v0.5 服务器处理后样本集上，验证连续风格、单生理信号、非脑电组合、响应类型辅助和非脑电教师蒸馏。
- 已完成版本：B0、B1、S1、S2、S3、SF1、SF2、SF3、C1、C2、A1、A2、T2，均为 seed2026，40 epochs，batch=64，lr=0.001，cuda。
- 当前最强整体结果：T2（非脑电生理教师 -> 车辆 + 连续风格学生），test RMSE=0.3247，primary RMSE=0.2161，tail RMSE=0.2907，优于 B0 车辆-only 的 0.3386/0.2184/0.3105。
- 当前最强直接输入候选：SF2（车辆 + 连续风格 + 皮电），test RMSE=0.3329，tail RMSE=0.2701，但严重幅值不足率比 B0 高，不能只按 RMSE 升级为主线。
- 方向相关结果：C1（车辆 + 心率 + 皮电 + 肌电）整体 RMSE 不好，但大响应错侧率最低 0.0471，说明非脑电组合可能更偏向方向/状态线索，而不是直接改善整条轨迹。
- 肌电结果：S3 单独肌电整体 RMSE 最差，但严重幅值不足率最低 0.2353、大响应召回最高 0.7647；说明肌电可能对大幅响应幅值有信息，但简单输入会扰乱整体拟合。
- 响应类型辅助：A1/A2 没有形成整体主线，尤其严重幅值不足率偏高；当前不能说辅助任务已经有效。
- 脑电直接输入和含脑电教师路线：暂缓。当前旧脑电事件特征在 v0.5 新锚点下出现全缺失/对齐风险，不能在未重建安全脑电窗口前进入主证据。
- 服务器状态：本轮 screen 已结束，GPU 空闲；没有后台训练继续运行。
- 当前最大风险：目前都是 seed2026 筛选结果，测试样本只有 163 个；T2 和 SF2 需要补 seed2027/2028，并查看预测图和分被试结果后才能形成强结论。
- 用户优先查看：`09_reports/stage03_v05_physio_mechanism_comparison_user_summary_cn.md` 和 `03_baselines/stage03_v05_physio_mechanism_comparison/tables/v05_physio_comparison_table.csv`。

# 项目状态更新：v0.5 脑电数据审计与新锚点特征提取

更新时间：2026-05-20

当前阶段：v0.5 连续风格/生理/脑电机制验证的数据接入修正。

当前完成：已完成原始脑电 CSV、清洗后脑电 FIF、旧脑电特征表和 v0.5 manifest 的对齐审计；已新增脚本按 v0.5 `anchor_s` 重提取锚点前 2 秒脑电特征。

最近一次结果：v0.5 manifest 共 1388 个样本，成功提取严格锚点前脑电特征 1164 个；test 159/165 可用，val 244/263 可用，train 761/960 可用。缺失主要来自部分记录没有清洗后脑电 FIF。

当前判断：脑电不是完全不能用；之前没有跑脑电版本的主要原因是旧脑电特征按横滚峰值对齐，不能直接接入 v0.5 新锚点。现在已经有 v0.5 对齐脑电特征表，可以进入接入检查，但还不能直接宣称脑电有效。

当前最大风险：训练集脑电缺失比例高于测试集，后续必须使用缺失掩码或脑电可用子集公平对照，不能让模型把“脑电是否存在”当成被试或记录线索。

下一步准备：把 `v05_eeg_features_pre_anchor_hist2s.csv` 接入 v0.5 机制实验，先做可用性检查，再决定是否跑车辆+脑电、车辆+连续风格+脑电、脑电教师版本。

用户优先查看：
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_eeg_feature_extraction_user_summary_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_features_pre_anchor_hist2s.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_recording_inventory.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_feature_availability_summary.csv`

---
## 2026-05-20 v0.5 生理/脑电补齐实验完成

- 当前阶段：v0.5 新样本集上的连续风格、生理信号、脑电和多教师机制验证。
- 当前完成：已按 v0.5 新锚点重新提取脑电锚点前 2 秒特征，并接入训练脚本；已在服务器补跑 S4、SF4、C3、C4、A3、T1、T3、T4。
- 服务器状态：本轮 screen 任务已全部结束，GPU 空闲；没有后台训练继续运行。
- 最近结果：当前 seed2026 最强为 T1（脑电教师 -> 车辆 + 连续风格学生），test RMSE=0.3107；其次为 SF4（车辆 + 连续风格 + 脑电），test RMSE=0.3142；B0 车辆-only 为 0.3386。
- 当前判断：脑电路线在 v0.5 样本上重新显示出价值，尤其是“训练时用脑电教师、推理时不用脑电”的 T1；全生理简单融合和全生理教师没有优于脑电教师。
- 当前最大风险：全部仍是 seed2026 筛选结果，测试样本 163 个；必须结合预测图、分被试结果和后续 seed2027/2028 才能形成强结论。
- 用户优先查看：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_eeg_completion_user_summary_cn.md`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_physio_eeg_result_table_white.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_multiversion_overlay_teacher.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_multiversion_overlay_eeg_direct.png`
- 下一步建议：先人工看 T1/SF4/T2 的预测图是否物理上更合理；若可以接受，优先补 T1、SF4、T2 的 seed2027/2028，不优先补 T4/A3。
## 2026-05-20 当前样本集暂存与下一步任务定义修正

- 当前决定：v0.5 服务器对齐样本集先暂存，继续保留其样本表、manifest、训练结果、生理/脑电结果和预测图，作为阶段性对照材料，不再把它直接视为最终样本定义。
- 关键修正：用户指出一次完整实验记录中不应只对应一个 episode。后续应从完整一次实验车辆数据中重建多个驾驶片段，每个片段包含事件开始、驾驶员操作开始、车辆响应开始、峰值、恢复或结束等时间点。
- 任务定义修正：当前研究目标应从“固定锚点后 2 秒方向盘预测”扩展为“极限/近极限工况下驾驶员响应与车辆状态演化建模”。方向盘仍是重要输出，但不再是唯一输出；车速、制动、横向加速度、横摆、横滚、横向偏移等也应纳入后续建模目标。
- 对当前 v0.5 的定位：v0.5 可作为临时实验集和生理/脑电机制筛选证据；但其固定 3 秒输入 + 2 秒预测窗口、单锚点切片方式可能误判锚点偏晚或事件较短的样本。
- 下一步准备：设计完整记录级 episode 重建流程，以一条完整车辆 CSV 为输入，自动识别多个 episode，并为每个 episode 输出变长时间线、驾驶员响应类型、车辆风险等级、锚点质量和后续训练任务建议。
- 补充输入：后续完整记录级重建应参考已有道路/场景信息，包括道路模块、低附着段、弯道/曲率、维修路段、连续超车段、longstraight/fix_road/middle_section 等设计信息。道路信息作为 episode 解释和工况背景，不直接等同于最终事件锚点。

## 2026-05-20 21:53:08 完整记录级 episode 重建 v1.0

- 已运行完整记录级 episode 重建脚本。
- 输入：`F:\data_set_process\data_process\01_datasets\数据预处理\原始车辆数据`
- 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0`
- 处理记录数：3
- 检测 episode 数：62
- 说明：道路/场景信息只作为上下文，不直接作为最终事件锚点；一条完整实验记录允许产生多个 episode。

## 2026-05-20 21:54:17 完整记录级 episode 重建 v1.0

- 已运行完整记录级 episode 重建脚本。
- 输入：`F:\data_set_process\data_process\01_datasets\数据预处理\原始车辆数据`
- 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0`
- 处理记录数：1
- 检测 episode 数：22
- 说明：道路/场景信息只作为上下文，不直接作为最终事件锚点；一条完整实验记录允许产生多个 episode。

## 2026-05-20 22:00:45 完整记录级 episode 重建 v1.0

- 已运行完整记录级 episode 重建脚本。
- 输入：`F:\data_set_process\data_process\01_datasets\数据预处理\原始车辆数据`
- 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0`
- 处理记录数：91
- 检测 episode 数：1766
- 说明：道路/场景信息只作为上下文，不直接作为最终事件锚点；一条完整实验记录允许产生多个 episode。

## 2026-05-20 22:41:04 v1.1 人工复核整理完成

- 用户复核 v1.0 后认为大部分样本可保留，'需要复核/边界复核' 类基本可舍去。
- 已生成 v1.1 复核后样本集：总 episode 1766，主训练候选 1383，对照样本 3，舍弃/暂缓 380。
- 主训练候选来源：核心极限样本 + 保守/弱操作极限样本 + 次级训练样本。
- 输出目录：F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed
- 当前没有训练模型；下一步建议基于 v1.1 构建车辆-only 数据集，并先验证车辆-only 强基线。

