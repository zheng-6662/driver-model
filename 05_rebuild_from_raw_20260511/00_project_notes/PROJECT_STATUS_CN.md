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

# R2E-Steering 项目总进度看板

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
