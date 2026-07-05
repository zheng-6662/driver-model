# 最新状态指针：2026-07-05 已完成 v318-v320 候选门控修正线。v318 保守两段式门控守卫通过但验证失败，核心问题是训练搜索把“全不改”当成安全最优；v319 加入困难代理和双通道配额后仍在训练折外与验证集全不改，原因是失败回退仍偏向零校正，且候选资格条件把可激活样本清空。随后按本地高级模型第320轮建议，v320 改为排序配额修复门控：新增候选正收益概率分类器，强方向盘/困难代理通道按排名强制激活，非强方向盘困难代理通道加更严坏风险上限，普通样本保持不动。v320 验证预算通过并按规则报告测试集：validation 全部样本收益 `0.001961`，强方向盘收益 `0.003729`，困难前20收益 `0.005495`，普通样本校正率和大退化率均为 `0`；test 全部样本收益 `0.000311`，强方向盘收益 `0.000613`，普通样本仍不动，但困难前20 `-0.001521`、困难前10 `-0.005689`。当前结论：第320版已经解决“门控全不改”和普通样本误伤问题，说明候选激活机制打通；但测试困难组仍未稳定改善，不能视为严重样本最终解决。下一步应转向候选排序泛化和困难组候选家族风险预算，而不是继续调全不改门控。

---

# 最新状态指针：2026-07-04 已完成 v316 filtered current-window coarse-scene train。v316 按用户要求“完整跑一次看看”，在 v315 当前窗口保留清单上完整重训 v307 粗场景条件曲线结构：训练/验证/测试保留事件为 `650/211/222`，隔离事件不参与训练、验证选模或测试主统计。选中 `v316_filtered_scene_init_aux003_film005_h64`，validation no-harm 通过，但过滤后 test 没有超过旧 v307：all 为 v300 `0.525580`、旧 v307 `0.496950`、v316 `0.502633`；bad10 为 v300 `0.859987`、旧 v307 `0.777797`、v316 `0.800171`；bad20 为 v300 `0.703038`、旧 v307 `0.651121`、v316 `0.660814`。保留 severe 33 个上，v300 `0.805638`、旧 v307 `0.877334`、v316 `0.886424`。当前结论：v315 过滤清单有助于清理样本边界，但“只过滤再重训”不是突破；后续应针对保留 severe 的幅值、相位、极端峰值跟随不足做模型修正，重锚定候选另开重切窗口线。

---

# 最新状态指针：2026-07-04 已完成 v315 rapid steering filter / reanchor plan。v315 不训练模型，而是把 v314 的方向盘快转来源审计转成训练前数据处理策略：全量 `1167` 个 delay0 事件中，当前窗口训练保留 `1083`，隔离 `84`；其中候选重锚定 `77`，全程快转证据弱候选剔除 `7`。按划分：train 保留 `650/702`，val 保留 `211/233`，test 保留 `222/232`。用户截图 #020 `zx_Entity_Recording_2025_09_27_17_14_07_v108_016` 被隔离为“当前平缓但后续才快转”，候选锚点后移约 `5.02s`，候选 `observation_s=278.52`。当前结论：下一轮当前窗口模型训练应先使用 v315 保留清单；隔离清单不应再作为当前强动作监督。重锚定候选必须重新切车辆窗口和目标曲线，不能只改表后直接训练。

---

# 最新状态指针：2026-07-04 已完成 v314 rapid steering source sample audit。用户明确不想人工逐个复核，改为抽样排查，并强调样本必须由方向盘快速转动引起。v314 读取 v312 的 1167 个 delay0 事件和原始车辆 CSV，用方向盘转动速度审计样本来源：当前 0-2s 有快转来源证据 `1083` 个，当前窗口快转证据不足或来源错位 `84` 个；v309 severe 37 个中只有 `4` 个来源可疑，用户截图 5 个中只有 `1` 个来源可疑，即 #020 `zx_Entity_Recording_2025_09_27_17_14_07_v108_016`。结论：样本总体符合“方向盘快速转动来源”，但应先隔离 `84` 个可疑来源样本；其余严重错误大多不是样本来源错，而是模型对快转后幅值、相位、极端跟随不足。下一步优先做 v315 过滤/重锚定候选训练表，再对来源成立的 severe 样本做幅值与相位修正。

---

# 最新状态指针：2026-07-04 已完成 v312 horizon-aligned label / anchor audit。v312 不训练模型，而是把标签拆成 `local_0_2_motion_label`（模型当前 0-2s 预测窗口内真实局部动作）和 `late_2_6_context_label`（2-6s 后续上下文）。全体 delay0 `1167` 个事件中，`227` 个存在粗标签与 horizon 局部窗口错位嫌疑，`49` 个属于 local flat 但 late 2-6s 大动作，`98` 个属于 local 与 late 方向冲突。v309 severe 37 个中，`11` 个存在错位嫌疑；用户截图 5 个中：`#020` 被标为 `local_0_2_flat_hold + late_2_6_extreme_positive`，建议 `split_local_flat_and_late_context`；`#014` 被标为 `local_0_2_mild_positive + late_2_6_mild_negative`，建议 `split_current_and_late_direction`；`#023` 被标为 `local_0_2_mild_negative + late_2_6_extreme_negative`，建议 `keep_late_context_separate`。当前结论：下一步 v313 不应直接用未来 local 标签当输入，而应先基于 v312 overlay 做人工/规则确认，形成可部署的 horizon-aligned coarse labels 或 validation-only prediction fallback。

---

# 最新状态指针：2026-07-04 已完成 v310/v311 差样本定向修改与锚点窗口审计。v310 从 v307 selected checkpoint 出发，加入 train/val target-shape 权重和方向/幅值/平直三类形状约束；guardrail 通过，选中 `v310_v307init_shape_guard_lo`，常规 test/all 从 v307 `0.496138` 小降到 v310 `0.494998`，test/bad10 从 `0.777797` 小降到 `0.775882`，但 v309 severe 37 个从 v307 `0.888400` 变为 v310 `0.890705`，用户截图 5 个也从 `1.755055` 变为 `1.775683`，说明单纯 loss/权重没有解决“方向/意图级严重错”。随后 v311 审计 37 个 severe 的 raw 0-2s 与 2-6s 后续窗口，发现 `11/37` 存在标签/后续窗口与预测窗口不一致嫌疑，截图 5 个中 `3/5` 命中，其中 `#020` 属于 0-2s 真实近似平直但 2-6s 后续大动作，`#014` 属于 0-2s 与 2-6s 方向相反且模型跟了后续方向。当前结论：下一轮应优先做 horizon-aligned label / anchor 修正，而不是继续加 hard-case loss。

---

# 最新状态指针：2026-07-04 已完成 v309 严重方向/意图错误筛选。基于 v309 近期最好预测图册的 test delay0 `232` 个事件，额外生成严重错误 CSV 和中文复核报告。用户截图命中 `5` 个严重案例：`#014` 平路过弯方向相反、`#017` 紧急变道/失稳极端幅值漏掉、`#019` 下坡过弯大动作曲线/相位误差高、`#020` 连续变道中真实近似无大动作但预测大幅转向、`#023` 平路过弯方向相反且 v307 比 v300 退化。按规则全体筛出 `37` 个严重候选，主要错误型包括 `opposite_peak_direction`、`false_large_maneuver`、`missed_extreme_amplitude`、`large_event_high_rmse`、`regression_vs_v300`。当前结论：v307 虽整体优于 v300，但仍存在方向/意图级严重错误，后续应把这些事件作为人工复核和下一轮训练的 hard-case 集合。

---

# 最新状态指针：2026-07-02 已完成 v300 within-subject full joint-curve retrain。v300 固定使用 v299 的同被试事件级 split，并映射到全部 7002 个 rolling 样本；防泄漏通过：`event_n=1167`，`train/val/test rolling=4212/1398/1392`，`train/val/test event=702/233/232`，`event_in_multiple_splits_n=0`，`duplicate_event_delay_rows_n=0`。本轮从原始 rolling 输入重新 fit scaler 并完整训练 3 个 joint-curve 候选，旧 v249 预测只作诊断参照，不参与训练或 validation 选择。validation 选中 `v300_full_joint_h64_no_subject`；`subject_onehot` 两个候选没有胜出。delay0 test/all RMSE 为 `0.5198`，旧 v249 诊断参照为 `0.3246`，说明完整重训整体能力尚未恢复到旧模型水平；但在 within_bad_top10 上 v300 为 `0.8600`，旧 v249 诊断参照为 `1.0383`，差样本有明显改善。当前结论：同被试完整重训能改善部分极差样本，但 subject one-hot 不是直接解法，下一步应围绕强响应幅值/极端曲线恢复，而不是继续做轻量 residual 或删除样本。

---

# 2026-07-04 v308 coarse scene 视觉人工复核包已完成（最新）

- 当前阶段：承接 v306/v307 的粗场景标签路线，按用户反馈从“看表复核”改成“看图复核”。v308 不是训练模型，而是给用户人工确认 `continuous_lane_change`、`emergency_lane_change_instability`、`other_or_uncertain` seed 的视觉复核入口。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v308_coarse_scene_visual_manual_review_20260704.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704`
- HTML 图册入口：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\index.html`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\reports\v308_coarse_scene_visual_manual_review_cn.md`
- 打包文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\v308_coarse_scene_visual_manual_review_20260704.zip`
- 方法：
  - 读取 v306 `v306_coarse_scene_manual_review_seed_pack.csv`。
  - 只生成 high + medium 复核队列图册，共 `748` 个事件。
  - 每个事件画一张锚点前 `-3s~0s` 与锚点后 `0s~2s` 的曲线图。
  - 曲线包含方向盘角、方向盘速度、`ay`、`yaw rate`、`roll`、曲率/横向距离、车速/制动。
  - 生成静态 `index.html`，支持筛选、逐事件填写复核结论/人工标签/备注，并导出 CSV。
- 核心结果：
  - `review_event_n=748`。
  - high priority `529`，medium priority `219`。
  - `continuous_lane_change=414`。
  - `emergency_lane_change_instability=115`。
  - `other_or_uncertain=219`。
  - `image_n=748`。
  - ZIP 自检 `testzip=None`，压缩包内 `png_n=748`，`has_index=True`。
- 当前判断：
  - v308 解决的是人工复核可读性问题，不是模型性能问题。
  - 图册使用锚点后真实响应，只能作为人工标注确认依据，不能作为预测前可获得输入。
  - 下坡/平路过弯仍主要依赖当前 manifest `scene_type`；当前曲线图不直接显示坡度。
  - 用户完成复核并导出 CSV 后，应先接回生成 confirmed coarse-scene 标签表，再重跑 v307 结构。
- 验证：
  - `python -m py_compile` 通过。
  - 预览 12 张图正常生成。
  - 全量 748 张图正常生成。
  - `logs\guardrail_check.json` 已写入。
  - ZIP 自检通过。

---

# 2026-07-04 v309 近期最好模型预测效果图册已完成（最新）

- 当前阶段：用户希望先看近期最好版本的预测效果。按当前 v306/v307 记录，近期最好版本为 `v307_coarse_scene_init_aux003_film005_h64`，因此 v309 只做预测效果可视化，不继续扩展人工标签复核。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v309_recent_best_prediction_effect_gallery_20260704.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704`
- HTML 图册入口：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\index.html`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\reports\v309_recent_best_prediction_effect_gallery_cn.md`
- 核心结果：
  - test delay0 事件数 `232`。
  - 图册代表性样本数 `54`。
  - test/all：v300 `0.519805` -> v307 `0.496138`。
  - 模型预测范围为 `0-2s`；图中额外显示真实后续到 `+6s`，用于观察事件后续发展，但不是模型预测范围。
  - ZIP 自检 `testzip=None`，压缩包内 `png_n=56`，`has_index=True`。
- 当前判断：
  - v307 是当前近期最好效果版本，适合先作为预测效果观察对象。
  - 图册能直接看出：哪些样本 v307 比 v300 更贴近真实曲线，哪些样本仍存在幅值不足、时序偏差或后续失稳无法覆盖。
  - v307 仍依赖 v306 粗场景 seed，其中直道连续/紧急子类需要人工确认，因此还不是最终可部署结论。
- 验证：
  - `python -m py_compile` 通过。
  - v309 脚本完整运行完成。
  - ZIP 自检通过。

---

# 最新状态指针：2026-07-02 已完成 v299 within-subject split residual calibration。按用户要求，已把每个被试内部样本随机拆成 train/val/test，且 `event_uid` 不重复：`event_n=1167`，`train/val/test=702/233/232`，`duplicate_event_uid_n=0`，`event_in_multiple_splits_n=0`，18 个被试都同时出现在三个 split。固定 v249 预测上做快速 residual 校准，val 选中 `base_curve_meta_subject__extra_trees_d5`，test all delta `-0.0067 RMSE`，test within_bad_top10 delta `-0.0738 RMSE`，达到快速潜力审计的正向标准。重要边界：本轮没有完整重训 v249，且新 test 中 `58.2%` 原本属于旧 v249 train split，因此 v299 只能说明同被试划分/驾驶员校准值得完整重训验证，不能作为 formal final 结果。下一步建议做 v300：按 within-subject split 完整重训当前主模型或轻量等价模型，彻底消除旧 split 暴露。

---

# 最新状态指针：2026-07-02 已完成 v298 event label explanatory audit。当前结论：粗响应标签能识别风险，但不能带来轨迹修正的本质改善；`oracle_strength_label` 识别 test bad_top10 的 AUC 为 `0.7735`，但最佳 future response label-known 残差修正只让 test bad_top10 改善 `-0.0093 RMSE`，远低于 `-0.05 RMSE` 的有效阈值。历史规则标签通过 subject+session+anchor time 只能匹配当前事件 all `22.7%`、test `28.3%`，覆盖不足。当前没有足够覆盖、可部署、锚点前可知的事件标签；下一步如继续标签路线，必须建立当前事件级人工/实验条件标签，而不是直接把 oracle 标签输入模型。生理数据 goal 仍未达成。

---

# 最新状态指针：2026-07-02 已完成 v297 subject style stability audit。当前结论：驾驶风格/被试历史存在弱辅助信号，但不足以作为主线；它更适合做 risk/uncertainty/context，而不是直接决定未来轨迹。下一步优先级转向事件级标签、实验条件标签与响应类型辅助监督。v296 为中断半成品，不作为有效结论；生理数据 goal 仍未达成。

# 历史状态指针：2026-07-02 已完成 v294 post-response candidate wait ranker。v294 把 v293 的 post-response 可见性转成真正 RMSE 任务：等待 1/2/3/5 秒后，用 query/prototype 的 post 生理响应匹配 v292 的 40 个 vehicle-similar train prototype 候选，并由 val 选择是否覆盖 latest。结果：`guardrail.pass=True`，`zip_testzip=True`，`event_n=1167`，`candidate_rows=46680`，`wait_policy_n=4`，`selector_config_n=36`，`uses_post_observation=True`，但 `route_viable_now=false`。候选池 oracle 仍有空间：test bad_top10 oracle delta `-0.0784 RMSE`，candidate_pool_gain_gt_005 oracle delta `-0.1171`；vehicle top1 很差：bad_top10 delta `+0.1453`。v294 的 val no-harm active 策略存在，但 test bad_top10 反而 `+0.0070`；test-best diagnostic 只有 `-0.0112`，且 val bad_top10 `+0.1239`、val all `+0.0606`，不可部署。结论：生理 post-response 能帮助识别风险，但仍不能稳定选择正确未来候选；当前 goal 仍未达成。

---

# 2026-07-02 v294 post-response candidate wait ranker 已完成（最新）

- 当前阶段：在 v293 证明 observation 后 0-3 秒生理响应对 bad_top10 有明显可见性后，进一步检验这种可见性能否转成真实 RMSE 改善。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v294_post_response_candidate_wait_ranker_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v294_post_response_candidate_wait_ranker_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v294_post_response_candidate_wait_ranker_20260702\reports\v294_post_response_candidate_wait_ranker_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v294_post_response_candidate_wait_ranker_20260702_pack.zip`
- 方法：
  - 固定使用 v292 的 40 个 vehicle-similar train prototype 候选。
  - 等待策略包括 `wait1_post0_1`、`wait2_post0_2`、`wait3_post0_3`、`wait5_post0_5`。
  - 对每个 wait 窗口，提取 query 与 prototype 的 v293 post 生理 response 特征。
  - 构造 query/prototype/absdiff/signeddiff pair 特征。
  - 对比 `vehicle_meta_only`、`post_response_pair_top64`、`vehicle_post_response_pair_top96`。
  - 模型包括 `ridge_a10`、`hgb_d3`、`extra_trees_d6`。
  - feature screening 只用 train query；模型只用 train query；threshold 只用 val query；test 只报告。
  - post 特征明确只代表短等待/延迟观测策略，不是原锚点即时输入。
- 核心结果：
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。
  - `route_viable_now=false`。
  - `event_n=1167`，`candidate_rows=46680`。
  - `wait_policy_n=4`，`selector_config_n=36`。
  - `best_val_noharm_active_exists=True`，但泛化失败。
  - val 选择策略：`wait1_post0_1__post_response_pair_top64__extra_trees_d6`，threshold `0.009393`。
  - 该策略 val bad_top10 delta `-0.0057`，val all delta `+0.0011`。
  - 但 test bad_top10 delta `+0.0070`，test all delta `+0.0017`，没有改善。
  - test-best diagnostic：`wait2_post0_2__vehicle_post_response_pair_top96__ridge_a10`，test bad_top10 delta `-0.0112`，bad_top10_vehicle_ambiguous delta `-0.0071`。
  - 但该 test-best 的 val bad_top10 delta `+0.1239`，val all delta `+0.0606`，不可部署。
  - candidate-pool oracle test bad_top10 delta `-0.0784`，bad_top10_vehicle_ambiguous delta `-0.0881`，candidate_pool_gain_gt_005 delta `-0.1171`。
  - vehicle_score_top1 test bad_top10 delta `+0.1453`。
- 当前判断：
  - v293 的 post-response 生理信号确实能识别“哪些样本可能会差”。
  - v294 说明：识别风险不等于能选对未来候选；post 生理 response matching 仍不能稳定从 vehicle top40 里挑出正确 prototype。
  - 目前瓶颈已经从“差样本风险可不可见”转到“看见风险后如何生成/选择更好轨迹”。
  - 当前 goal 仍未完成；继续在候选匹配/reranker 上加模型复杂度，收益依据不足。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成两次；第二次修复了 oracle 分层中的 event-level flag。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`，file_count `13`。

---

# 2026-07-02 v293 physiology response visibility / latency audit 已完成（最新）

- 当前阶段：在 v288-v292 连续证明 observation 前 ECG/RESP/EDA 源生理无法稳定做 selector、reranker、pairwise matching 后，改为审计生理信号的可见时间：锚点前是否真的有信息，还是必须观察锚点后短时间才出现驾驶员响应。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v293_physio_response_visibility_latency_audit_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v293_physio_response_visibility_latency_audit_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v293_physio_response_visibility_latency_audit_20260702\reports\v293_physio_response_visibility_latency_audit_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v293_physio_response_visibility_latency_audit_20260702_pack.zip`
- 方法：
  - 读取 v291 event table 和 v292 pairwise candidate table。
  - 回到 cleaned 200Hz 生理记录，围绕 observation_s 构造 pre 与 post 多窗口特征。
  - baseline 使用 observation 前 `-60s` 到 `-20s`。
  - 窗口包括 `pre10_pre5`、`pre5_pre2`、`pre2_0`、`post0_1`、`post0_2`、`post0_3`、`post0_5`、`post1_3`、`post2_5`、`post5_10`。
  - 信号族包括 ECG、RESP、EDA phasic、EDA tonic、EMG、HR。
  - 每个窗口提取 robust-z 均值、绝对均值、标准差、range、分位数、首尾差、slope、line length 等。
  - 特征筛选只用 train split；分类器只用 train split 训练；val/test 只报告。
  - post-observation 特征明确标记为 diagnostic / waiting-policy evidence，不作为原锚点即时部署输入。
- 核心结果：
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。
  - `event_n=1167`。
  - `feature_n=540`，`screen_feature_n=540`。
  - 生理窗口可用率 `ok_rate=0.91945`。
  - `uses_post_observation=True`，`post_features_are_diagnostic_only=True`。
  - 主差样本 `bad_top10`：
    - pre best test AUC `0.4896`，基本不可见。
    - early-post best test AUC `0.7726`。
    - `window_post0_3` test AUC `0.7254`。
    - `window_post0_2` test AUC `0.7053`。
    - `window_post0_5` test AUC `0.6935`。
  - `bad_top10_vehicle_ambiguous`：
    - pre best test AUC `0.6012`，属于边缘弱信号。
    - early-post best test AUC `0.6627`。
  - `candidate_pool_gain_gt_005`：
    - pre best test AUC `0.5722`。
    - early-post best test AUC `0.5593`。
    - late-post best test AUC `0.5808`。
  - train screen 中最强的窗口/信号主要来自 post EMG，尤其 `post2_5`、`post0_5`、`post1_3` 一带；这更像事件触发后的身体响应，而不是 observation 前稳定驾驶风格。
- 当前判断：
  - v293 解释了 v288-v292 为什么难以成功：原 observation 前生理确实很弱，主差样本 `bad_top10` 上甚至低于随机。
  - 但生理并非完全没用；它在 observation 后 0-3 秒对差样本是否会失败有明显可见性。
  - 这不是原锚点即时预测成功，而是支持下一步测试“短等待/延迟观测/早期响应融合”能否以可接受延迟换来差样本本质改善。
  - 当前 goal 仍未完成；下一步要做的是把 post0-3 秒响应信号转成严格的可部署等待策略，并量化等待代价与 RMSE 收益。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v292 source-physio pairwise candidate ranker 已完成（最新）

- 当前阶段：在 v291 多源生理 event-level selector 失败后，改为 pairwise candidate ranking，直接测试生理能否在车辆相似候选之间做 tie-breaker。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v292_source_physio_pairwise_candidate_ranker_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v292_source_physio_pairwise_candidate_ranker_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v292_source_physio_pairwise_candidate_ranker_20260702\reports\v292_source_physio_pairwise_candidate_ranker_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v292_source_physio_pairwise_candidate_ranker_20260702_pack.zip`
- 方法：
  - 固定使用 v278 `listrank_vehicle` 候选池：每个 query 40 个 train prototype。
  - 明确审计 prototype：val/test query 的 prototype 全部来自 train split，`prototype_train_only=true`。
  - 沿用 v291 train-only source physiology screen，选 `bio_all_top` 和 `bio_lowid_top` 各 45 个 ECG/RESP/EDA 源特征。
  - 对 query/prototype 构造 `absdiff/signeddiff/query/proto` pair 特征。
  - 对照 feature block：vehicle score only、bio pair only、vehicle+bio pair。
  - pairwise 模型只用 train query 训练；覆盖 latest 的阈值只用 val query 选择；test 只报告。
- 核心结果：
  - `route_viable_now=false`。
  - `event_n=1167`，`candidate_rows=46680`。
  - train/val/test event 数 `674/309/184`。
  - `prototype_train_only=true`。
  - `selector_config_n=15`。
  - vehicle top1 在 test bad_top10 上比 latest 差 `+0.1453 RMSE`。
  - candidate-pool oracle 在 test bad_top10 上比 latest 好 `-0.0784 RMSE`。
  - candidate-pool oracle 在 test bad_top10_vehicle_ambiguous 上比 latest 好 `-0.0881 RMSE`。
  - `best_val_noharm_active_exists=false`。
  - val 选择只能 fallback no override，test bad_top10 delta `0.0`。
  - test-best diagnostic：`bio_all_top_pair_only__hgb_d3`，test bad_top10 delta `-0.0248`，test bad_top10_vehicle_ambiguous delta `-0.0314`，但 val bad_top10 delta `+0.1402`、val all delta `+0.0367`，不可部署。
  - 存在一个 no-harm 但非 active 的弱阈值：test bad_top10 delta `-0.0124`，只覆盖约 `1/19` 个 test bad_top10，不能算本质改善。
- 当前判断：
  - v292 证明：差样本不是没有可用候选；车辆 top40 候选池里确实有更好的 prototype。
  - 关键失败点是源生理 pairwise matching 不能稳定从这些候选里选对。
  - 这一步比 v291 更贴近用户提出的“锚点前车辆信息不足、样本相似但后续不同”的本质问题；失败会进一步削弱继续堆 physiology matching/reranker 模型的理由。
  - 当前 goal 仍未达成。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v291 multi-signal physiology supervised probe 已完成

- 当前阶段：在 v288/v289/v290 分别验证 ECG、RESP、EDA 源信号 distance/rerank/gate 不可部署后，进一步测试三路源生理合并后的监督能力。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v291_multisignal_physio_supervised_probe_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v291_multisignal_physio_supervised_probe_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v291_multisignal_physio_supervised_probe_20260702\reports\v291_multisignal_physio_supervised_probe_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v291_multisignal_physio_supervised_probe_20260702_pack.zip`
- 方法：
  - 读取 v288/v289/v290 的 causal source features，不重新抽原始波形。
  - 读取 v278 的 `latest / listrank_vehicle / listrank_vehicle_bio / listrank_vehicle_style_bio` 现成方法池。
  - 用 train split 做源生理特征筛选，构造 `bio_source_all_top`、`bio_source_lowid_top`、vehicle score、vehicle+bio 等 7 个 feature block。
  - 用 train 训练多输出误差回归器，val 选择覆盖 latest 的阈值，test 只报告。
  - 额外训练分类探针，检查源生理是否能识别 `bad_top10`、`vehicle_ambiguous`、`method_oracle_gain_gt_002` 等标签。
- 核心结果：
  - `route_viable_now=false`。
  - `event_n=1167`；train/val/test 为 `674/309/184`。
  - `bio_source_feature_n=1660`，train-only screen 后 `screen_feature_n=1404`。
  - `feature_block_n=7`，`selector_config_n=28`。
  - 现成方法池的事后 oracle 在 test bad_top10 上有 `-0.0402 RMSE` headroom，在 bad_top10_vehicle_ambiguous 上有 `-0.0425 RMSE` headroom。
  - 但 validation 没有任何 no-harm active selector：`best_val_noharm_active_exists=false`。
  - validation fallback 只能选择 no override，test bad_top10 delta `0.0`。
  - test-best diagnostic 非部署选择器仅 `-0.0093 RMSE`，且 val all delta 为 `+0.00477`，不满足 no-harm。
  - 源生理识别 test bad_top10 的最好 AUC 仅 `0.5394`；识别 bad_top10_vehicle_ambiguous 的 test AUC 更低。
- 当前判断：
  - 方法池事后 oracle 有小上限，说明不是完全没有可选空间。
  - 但 ECG/RESP/EDA 源生理合并后，既不能稳定识别差样本，也不能通过 val 选择出可部署覆盖策略。
  - 这比单路源信号 gate 更强地说明：当前生理数据在现有 subject-disjoint / observation-before-anchor 条件下，还不足以本质弥补车辆锚点前信息不足。
  - 下一步若继续生理方向，不应再做 selector/reranker/threshold；只能转成可观测性分层、采集质量/同步复核，或承认生理在当前数据规模下只能作为弱诊断。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v290 EDA/SCR usable-subset source route audit 已完成

- 当前阶段：在 v288 ECG source 和 v289 RESP source 都没有形成可部署 top1 改善后，继续检查 EDA/SCR 源信号是否能补足车辆锚点前信息不足。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v290_eda_scr_usable_subset_route_audit_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v290_eda_scr_usable_subset_route_audit_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v290_eda_scr_usable_subset_route_audit_20260702\reports\v290_eda_scr_usable_subset_route_audit_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v290_eda_scr_usable_subset_route_audit_20260702_pack.zip`
- 方法：
  - 直接读取 cleaned 200Hz `EDA_raw200/EDA_filt200/EDA_Tonic/EDA_Phasic`。
  - 对 `EDA_Phasic` 失效或近常数记录增加 `raw - slow(raw)` 的 fallback phasic。
  - 只使用 observation 前数据，构造 tonic/phasic/SCR-like peak、短窗斜率/波动/积分、相对 baseline delta、质量与可用性标记。
  - 同时评估全体样本 route gate 和 EDA usable / vehicle ambiguous / bad_top10 等可用子集。
- 核心结果：
  - `route_viable_now=false`。
  - `eda_subset_route_viable_now=false`。
  - `eda_source_feature_n=473`。
  - `feature_set_n=29`。
  - EDA 可用事件数 `906/1167`，可用率 `0.77635`。
  - deployable top1 bad_top10 未通过：validation 选择 `eda_duration_dur2_top32`，test delta vs latest `+0.1760`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta vs latest `+0.1601`。
  - test-best top1 diagnostic 未通过：最佳仍比 latest 差 `+0.1409`。
  - test bad_top10 best corr `0.0306`，低于弱相关门槛。
  - EDA usable 子集也未通过：`eda_usable_top1` test delta `+0.0762`，`bad_top10_eda_usable_top1` test delta `+0.1760`。
- 当前判断：
  - EDA 质量覆盖不是主要瓶颈，因为约 77.6% 事件可用，但可用子集没有转成差样本改善。
  - v288/v289/v290 连续说明：ECG、RESP、EDA 源信号重建后都只有弱诊断苗头，不能稳定把 vehicle top40 候选中的更好未来排到 top1。
  - 当前 goal 仍未达成；不应继续在同一类 physiology distance/rerank/gate 框架上堆阈值或模型复杂度。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v287 physiology temporal-window route audit 已完成（最新）

- 当前阶段：在 v285 整体 raw 200Hz shape-state route gate 失败后，检查有效信息是否被窗口/信号族混合稀释。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v287_physio_temporal_window_route_audit_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v287_physio_temporal_window_route_audit_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v287_physio_temporal_window_route_audit_20260702\reports\v287_physio_temporal_window_route_audit_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v287_physio_temporal_window_route_audit_20260702_pack.zip`
- 方法：
  - 复用 v285 causal raw-shape 特征和 train-only screen。
  - 按时间窗口、信号族、特征类型、窗口×信号组合构造 47 个 feature set。
  - route gate 仍复用 v278 vehicle top40 候选池与 v272 差样本标签。
  - test 只报告，不用于选择窗口、信号族、阈值或策略。
- 核心结果：
  - `route_viable_now=false`。
  - feature_set_n `47`。
  - deployable top1 bad_top10 未通过：validation 选择 `signal_eda_top32`，test delta vs latest `+0.2379`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta `+0.2314`。
  - test-best top1 diagnostic 仍未超过 latest：最佳 `combo_pre2_0_ecg_top16`，delta `+0.0941`。
  - test bad_top10 best corr `0.0854`，来自 `combo_pre1_0_ecg_top16`，但这是非部署诊断信号。
  - 单独窗口最好：`win_pre10_pre5_top32`，bad_top10 top1 delta `+0.1144`，top3 oracle delta `-0.0027`。
  - 单独信号族最好：`signal_resp_top32`，bad_top10 top1 delta `+0.1783`，top3 oracle delta `-0.0159`。
- 当前判断：
  - 窗口/信号族拆分没有把生理信号转成可部署 top1 收益。
  - ECG 最近 1-2 秒有弱排序/诊断苗头，但还不能作为轨迹选择策略。
  - 不建议在同一 v285 特征层上继续堆复杂 vehicle+physio 融合模型。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v286 raw-200Hz online subject-aware calibration 已完成

- 当前阶段：在 v285 subject-disjoint route gate 失败后，单独测试 subject-aware / online adaptation 边界。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v286_raw200_online_subject_calibration_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v286_raw200_online_subject_calibration_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v286_raw200_online_subject_calibration_20260702\reports\v286_raw200_online_subject_calibration_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v286_raw200_online_subject_calibration_20260702_pack.zip`
- 边界：
  - 这不是 subject-disjoint 正式结果。
  - global gate 只用 train split 训练。
  - val/test 在线校准只用同 split、同 subject、当前事件之前的历史事件。
  - 生理表示来自 v285 raw 200Hz shape-state train-only feature set。
- 核心结果：
  - test bad_top10 fixed wait-latest：`0.6950`。
  - global vehicle gate：`0.7528`。
  - global vehicle+raw285 gate：`0.8017`。
  - online subject mean vehicle：`0.7112`。
  - online raw285 KNN vehicle：`0.7358`。
  - online subject mean vehicle+raw285：`0.6950`。
  - online raw285 KNN vehicle+raw285：`0.7197`。
  - raw285 KNN online 相对纯 subject mean online 变差 `+0.0246`；vehicle+raw285 后再 KNN 仍相对纯 subject mean 变差 `+0.0085`。
- 当前判断：
  - 即使允许同驾驶员历史反馈，v285 底层生理特征也没有给差样本带来额外校准收益。
  - subject-aware 边界下真正有用的是“等待到 latest”或纯 subject mean 退回 latest，而不是 raw285 生理 KNN。
  - 这进一步削弱了继续加复杂 vehicle+physio 融合模型的理由。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v285 raw-200Hz signal-shape physiology route gate 已完成

- 当前阶段：v284 证明 v260 biomarker 重筛不够后，回到底层 cleaned 200Hz 连续信号重新构造事件前状态。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v285_raw200_shape_state_route_gate_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v285_raw200_shape_state_route_gate_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v285_raw200_shape_state_route_gate_20260702\reports\v285_raw200_shape_state_route_gate_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v285_raw200_shape_state_route_gate_20260702_pack.zip`
- 方法：
  - 从 cleaned 200Hz 层直接提取质量、短窗形态、导数/突变、节律/相位、跨信号耦合、个体内 causal past percentile。
  - 不读取 v260/v284 派生特征表。
  - train-only 选择 6 组特征集：`raw_shape_behavior_top64`、`raw_shape_bad_top64`、`raw_low_identity_top64`、`raw_quality_shape_top64`、`raw_coupling_top48`、`raw_causal_past_top48`。
  - 在 v278 vehicle top40 候选池中复用 v284 route gate 口径。
- 核心结果：
  - `route_viable_now=false`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1958`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta `+0.1826`。
  - test-best top1 diagnostic 未通过：best delta `+0.1578`。
  - test bad_top10 best corr `0.0498`，低于 `0.05` 的弱相关门槛。
- 当前判断：
  - 回到底层 200Hz shape-state 后，没有比 v284 更接近通过 route gate。
  - 当前生理特征层仍不能稳定判断 vehicle top40 中哪个候选更接近真实未来。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v284 dynamic low-identity physiology route gate 已完成

- 当前阶段：按 v283 的硬要求，构造新的低身份、动态生理状态表示，并先过车辆歧义候选 route gate。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v284_dynamic_low_identity_physio_route_gate_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v284_dynamic_low_identity_physio_route_gate_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v284_dynamic_low_identity_physio_route_gate_20260702\reports\v284_dynamic_low_identity_physio_route_gate_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v284_dynamic_low_identity_physio_route_gate_20260702_pack.zip`
- 方法：
  - 使用 v260 的 0ms 事件型 biomarker。
  - 只在 train split 上计算行为 eta、bad_top10 eta、subject/recording identity eta。
  - 构造 `dyn_behavior_identity_top64`、`dyn_bad_identity_top48`、`dyn_noamp_multi_top48`、`low_identity_dyn_top48`、`strict_ratio_noamp_top32` 五组特征。
  - 在 v278 vehicle top40 候选池中重新按生理距离排序，并复用 v272 的 bad_top10 / vehicle_ambiguous 事件口径。
- 核心结果：
  - `route_viable_now=false`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1697`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta `+0.1903`。
  - test-best top1 diagnostic 也未超过 latest：best delta `+0.1525`。
  - 只有弱相关性苗头：test bad_top10 best corr `0.0553`。
- 当前判断：
  - 低身份动态 biomarker 可以略微改善“排序相关性”，但仍不能形成可部署 top1 选择。
  - 不应直接训练更复杂融合模型。
  - 若继续生理目标，下一步只能转向更底层信号重处理或明确 subject-aware 个体校准任务边界。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v283 生理路线 lineage / gap 审计已完成（最新）

- 当前阶段：把 v254b-v282 生理证据链合并，明确哪些路线已经关闭、下一步还剩什么可尝试。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v283_physio_route_lineage_gap_audit_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v283_physio_route_lineage_gap_audit_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v283_physio_route_lineage_gap_audit_20260702\reports\v283_physio_route_lineage_gap_audit_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v283_physio_route_lineage_gap_audit_20260702_pack.zip`
- 核心结论：
  - `current_goal_achieved=false`：没有可部署生理路线稳定改善 test bad_top10。
  - `old_feature_selector_route_closed=true`：v269/v271/v282 已经关闭旧特征筛选、校准和候选消歧微调路线。
  - `physio_source_alignment_ready=true`：200Hz 时间轴与事件窗口覆盖基本可用，失败不能简单归因于对齐。
  - `next_route_requires_feature_redefinition=true`：主要瓶颈是有效信号和身份混淆，不是简单模型容量。
- 下一步硬要求：
  - 不复用旧 bio selector/reranker/reliability filter 微调作为主线。
  - 若继续生理 goal，必须先构造低身份但行为相关的生理状态特征。
  - 新生理特征必须先通过 v282 类车辆歧义样本 route gate，再进轨迹预测模型。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v282 生理歧义消解 route gate 审计已完成（最新）

- 当前阶段：围绕“生理是否能补足车辆锚点前信息不足”做路线门控审计。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v282_physio_ambiguity_route_gate_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v282_physio_ambiguity_route_gate_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v282_physio_ambiguity_route_gate_20260702\reports\v282_physio_ambiguity_route_gate_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v282_physio_ambiguity_route_gate_20260702_pack.zip`
- 核心结果：
  - route gate `route_viable_now=false`。
  - 可部署 `bio_top1`：val 选 raw_set 后，test bad_top10 delta vs latest 为 `+0.1989`；test bad_top10_vehicle_ambiguous delta 为 `+0.2347`。
  - 非部署 `bio_top3` 上限：bad_top10_vehicle_ambiguous 的 val/test 不同向，val `+0.1617`、test `+0.0724`，仍都差于 latest。
  - test bad_top10 任一 raw_set 的生理距离-真实误差排序相关均值最高仅 `0.00985`。
  - v281 可训练 selector 仍没有 deployable 超过 fixed latest。
- 当前判断：
  - 现有生理特征不是稳定的 subject-disjoint 差样本消歧信号。
  - 不建议继续旧特征上的 bio selector / reranker / threshold 微调。
  - 若继续完成生理 goal，下一步应进入 v283：从 200Hz 连续生理层重构事件前状态特征和质量控制特征，再重新验证是否能区分车辆相似但后续分叉样本。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。

---

# 2026-07-02 v279-v281 生理可靠性过滤与 bio-top3 窄化选择已完成（最新）

- 当前阶段：围绕“充分利用生理数据弥补锚点前车辆信息不足”继续做了三轮可部署性实验，但 goal 仍未达成。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v279_physio_reliability_filter_for_listrank_20260702.py`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v280_crossfit_physio_reliability_filter_20260702.py`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v281_bio_top3_constrained_selector_20260702.py`
- 核心结果：
  - v279：用生理/风格判断 v278 vehicle listwise top candidate 是否可信。test-best diagnostic `0.6791`，但 val 不能部署；bio reliability 没有赢过 vehicle reliability。
  - v280：用 recording-group OOF train top 修正 v279 的候选分布偏差。test-best diagnostic 退到 `0.6891`；deployable 仍为 fixed latest `0.6950`；bio 仍不赢 vehicle。
  - v281：只在 vehicle top40 内的 bio top3 候选里训练选择器。bio top3 oracle test bad_top10 `0.6738` 有少量上限，但 val 选择的可部署策略仍为 `0.6950`；test-best diagnostic `0.6842`。
- 当前判断：
  - 生理邻域中偶尔包含更好候选，但可泛化 selector 学不出来；validation 上 bio-top3 oracle 自己也比 latest 差，说明 test 上限更像 split 偶然收益。
  - 不建议继续做同类 bio selector / reranker / reliability filter / threshold 微调。
  - 若继续追求预测效果，主线应回到车辆多未来建模、概率/不确定性输出、或 anchor-aware 任务构造；生理只能作为边界证据或另设 subject-aware 个体校准任务。
- GPTPro：已准备 `gptpro_reviews\20260702_phase280_physio_next_prompt.md`，但 bridge 未能确认 Chrome 当前为 Pro/进阶模式，按规则未发送。
- 验证：
  - v279/v280/v281 均通过 `python -m py_compile`。
  - v279/v280/v281 脚本完整运行完成。
  - 三个 ZIP 均 `testzip=True`。

# 2026-07-02 v278 listwise candidate rank loss（最新追加）

- v278 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v278_listwise_candidate_rank_loss_20260702.py`
- v278 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v278_listwise_candidate_rank_loss_20260702`
- v278 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v278_listwise_candidate_rank_loss_20260702\reports\v278_listwise_candidate_rank_loss_cn.md`
- v278 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v278_listwise_candidate_rank_loss_20260702_pack.zip`
- 方法：
  - 复用 v267 full vehicle top40 候选池。
  - 对每个事件内部的候选按真实 tail RMSE 构造组内相对排序标签 `rank_target_z`。
  - 训练 listwise rank regressor，比较 vehicle-only、vehicle+bio、vehicle+style+bio。
  - 阈值仍只由 val 选择，test 只报告。
- 关键结果：
  - event_n `1167`。
  - candidate_rows `46680`。
  - feature_set_n `3`。
  - search_rows `96`。
  - fixed wait-latest `0.6950`。
  - oracle best `0.6125`。
  - test-best diagnostic：`listrank_vehicle threshold=0.67075`，test bad_top10 `0.6832`，覆盖率 `0.1053`，但 val bad_top10 伤害 `+0.0509`、val all 伤害 `+0.0077`，不可部署。
  - best deployable：`0.6950`，test 覆盖率 `0`。
  - best bio feature diagnostic：`0.6950`，未超过 vehicle-only；`bio_beats_vehicle_diagnostic=false`。
- 判读：
  - 候选选择损失本身是有价值的，能比 v276/v277 诊断出更多 vehicle-only headroom。
  - 但生理/风格没有改善候选排序，反而没有保住 vehicle-only 的 diagnostic headroom。
  - 当前 subject-disjoint 下，生理仍不能作为差样本本质改善的主增量。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检通过。

---

# 2026-07-02 v277 style + calibrated physiology candidate gain model

- v277 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v277_style_bio_candidate_gain_model_20260702.py`
- v277 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v277_style_bio_candidate_gain_model_20260702`
- v277 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v277_style_bio_candidate_gain_model_20260702\reports\v277_style_bio_candidate_gain_model_cn.md`
- v277 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v277_style_bio_candidate_gain_model_20260702_pack.zip`
- 方法：
  - 复用 v276 的 full vehicle top40 candidate gain 框架。
  - 加入 v253a 当前任务口径的 `last60_guard3` 驾驶风格特征。
  - 加入 v271 train-only 筛选后的 calibrated raw physiology summary / PCA 特征。
  - 计算 query-prototype 的 `style_distance_v253_current` 和 `bio271_distance_calibrated`。
  - 比较 6 组 candidate gain 特征集，阈值只由 val 选择，test 只报告。
- 关键结果：
  - event_n `1167`。
  - candidate_rows `46680`。
  - style query feature cap `96`。
  - bio271 query feature cap `96`。
  - feature_set_n `6`。
  - search_rows `195`。
  - fixed wait-latest `0.6950`。
  - oracle best `0.6125`。
  - val-best deployable / active / stable / noharm-all 都是 `candidate_vehicle_style_query threshold=0.0460`，但 test bad_top10 覆盖率 `0`，RMSE `0.6950`。
  - test-best diagnostic 是 `candidate_vehicle_style_bio_dist threshold=0.01509`，test bad_top10 `0.7008`，比 fixed wait-latest 更差。
- 判读：
  - 驾驶风格 + 校准生理没有在 test bad_top10 上提供可部署消歧。
  - 可覆盖 test 的策略反而伤害更大，说明 style/bio distance 没有稳定指向正确候选。
  - 当前 subject-disjoint 生理/风格主线不应继续做同类 feature/ranker 微调。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检通过。

---

# 2026-07-02 v276 bio-assisted candidate gain model

- v276 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v276_bio_assisted_candidate_gain_model_20260702.py`
- v276 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v276_bio_assisted_candidate_gain_model_20260702`
- v276 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v276_bio_assisted_candidate_gain_model_20260702\reports\v276_bio_assisted_candidate_gain_model_cn.md`
- v276 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v276_bio_assisted_candidate_gain_model_20260702_pack.zip`
- 方法：
  - 回到 v267 full vehicle top40 候选池，不再使用 v273 的 bio top5 小候选限制。
  - 对每个 query-candidate pair 训练相对 latest 的候选收益预测器。
  - 比较 `candidate_vehicle`、`candidate_vehicle_bio`、`candidate_bio_only` 三组输入，检查生理是否能辅助候选选择。
  - 每个事件只允许选择预测收益最高的候选；是否覆盖 latest 的阈值只在 val 上选择，test 只报告。
- 关键结果：
  - event_n `1167`。
  - candidate_rows `46680`。
  - feature_set_n `3`。
  - search_rows `96`。
  - fixed wait-latest `0.6950`。
  - oracle best `0.6125`。
  - test-best gain diagnostic `candidate_bio_only threshold=0.04017`：test bad_top10 `0.6858`，覆盖率 `0.0526`。
  - 但该 diagnostic 在 val bad_top10 上伤害 `+0.0277`，val all 伤害 `+0.0079`，stable_pass=false，不可部署。
  - val-best any 为 `threshold=inf`，即不覆盖；test bad_top10 `0.6950`，未超过 fixed wait-latest。
- 判读：
  - 生理确实能在少数 test 差样本上找到一点候选收益信号。
  - 但 val 不支持该规则，说明这仍是事后 diagnostic，不是稳定可部署策略。
  - candidate_vehicle_bio 没有稳定优于 candidate_vehicle，当前生理仍不能成为主增量。
  - 当前 goal 仍未完成；继续沿同类生理 selector/reranker/override 微调收益很低。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检通过。

---

# 2026-07-02 v275 stable bio consensus override

- v275 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v275_stable_bio_consensus_override_20260702.py`
- v275 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v275_stable_bio_consensus_override_20260702`
- v275 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v275_stable_bio_consensus_override_20260702\reports\v275_stable_bio_consensus_override_cn.md`
- v275 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v275_stable_bio_consensus_override_20260702_pack.zip`
- 方法：
  - 承接 v274：默认仍使用 fixed wait-latest。
  - 对同一事件收集所有 calibrated physiology raw_set / pred_col 的候选锚点。
  - 只有多个生理视角支持同一个非 latest 锚点，并且支持票数超过 latest 票数时才允许 override。
  - 阈值只在 val 上选择；test 只报告。
  - 选择时同时约束 val bad_top10、val all、val normal、val strong_steer、val observe_later_like，避免只对少数样本过拟合。
- 关键结果：
  - event_n `1167`。
  - candidate_rows `35010`。
  - grid_rows / search_rows `750`。
  - fixed wait-latest `0.6950`。
  - oracle best `0.6125`。
  - bio-prefilter candidate oracle `0.6466`。
  - test-best consensus diagnostic `0.6881`，覆盖率 `0.1053`，但 val bad_top10 delta `+0.1385`、val all delta `+0.0295`，不可部署。
  - val-best any / active / stable / noharm-all 在 test bad_top10 均为 `0.6950`，覆盖率 `0`。
- 判读：
  - 多生理视角一致投票能在 test diagnostic 中事后找到少量有利样本。
  - 但能在 val 上通过稳定性约束的规则，到 test bad_top10 上没有实际覆盖。
  - 这说明当前生理一致性仍不是稳定可部署的差样本修正信号。
  - 当前 goal 仍未完成；继续做同类 bio override 微调收益很低。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检通过。

---

# 2026-07-02 v274 no-harm bio override（最新追加）

- v274 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v274_noharm_bio_override_20260702.py`
- v274 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v274_noharm_bio_override_20260702`
- v274 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v274_noharm_bio_override_20260702\reports\v274_noharm_bio_override_cn.md`
- v274 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v274_noharm_bio_override_20260702_pack.zip`
- 方法：
  - 承接 v273 的 bio-prefiltered pair 预测结果。
  - 默认策略固定为 `policy_wait_to_latest_anchor`。
  - 只有当 pair model 对某个 bio 候选的预测分数达到阈值，并且相对 latest 的 margin 足够时，才允许覆盖 latest。
  - threshold / margin 只在 val bad_top10 上选择，test 只报告。
- 关键结果：
  - event_n `1167`。
  - candidate_event_model_rows `35010`。
  - threshold_search_rows `3780`。
  - test bad_top10 keep0 `1.1977`。
  - fixed wait-latest `0.6950`。
  - oracle best `0.6125`。
  - bio-prefilter candidate oracle `0.6466`。
  - test-best override diagnostic `override_best_active_subject_seq_pca72_pair_bio_hgb = 0.6902`，覆盖率 `0.0870`，低于 fixed wait-latest `0.0048`，但不是可部署选择。
  - val-best any / active / noharm active 在 test bad_top10 都为 `0.6950`，基本等价于 fixed wait-latest。
- 判读：
  - 稀疏覆盖有很小的 test diagnostic 信号，说明 bio 候选中确实有少数可用点。
  - 但 val 选择无法稳定泛化，不能把这点 headroom 变成正式可部署收益。
  - 当前证据仍不支持继续沿同类 bio selector / bio reranker 小改动投入。
  - 下一步应把主线转向车辆多未来候选、不确定性估计、ranker 与可部署轨迹选择；生理保留为辅助边界证据。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检通过。

---

# 2026-07-02 v273 bio-prefiltered pair reranker（最新追加）

- v273 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v273_bio_prefiltered_pair_reranker_20260702.py`
- v273 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v273_bio_prefiltered_pair_reranker_20260702`
- v273 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v273_bio_prefiltered_pair_reranker_20260702\reports\v273_bio_prefiltered_pair_reranker_cn.md`
- v273 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v273_bio_prefiltered_pair_reranker_20260702_pack.zip`
- 方法：
  - 承接 v272：先用车辆 top40 prototype 建候选池。
  - 再按 v271 calibrated physiology 距离取 bio top5。
  - 只在 bio top5 小候选集合内训练 v267 式监督 pair reranker。
  - raw_set / strategy 由 val bad_top10 选择，test 只报告。
- 关键结果：
  - pair_row_n `35010`。
  - test bad_top10 keep0 `1.1977`。
  - fixed wait-latest `0.6950`。
  - oracle best `0.6125`。
  - bio-prefilter candidate oracle `subject_summary64:pair_candidate_oracle_k5 = 0.6466`。
  - test-best deployable diagnostic `recording_summary64:pair_vehicle_hgb_k10 = 0.7964`，仍高于 fixed wait-latest。
  - val-best vehicle+bio `recording_summary64:pair_vehicle_bio_hgb_k10 = 0.8664`。
- 判读：
  - bio top5 小候选集合确实包含部分好答案，但监督 selector 仍学不会稳定选择。
  - v272 的生理上界无法转化为可部署收益。
  - 继续做同类生理 selector/reranker 已不合理。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=True`。

---

# 2026-07-02 v272 physiology ambiguity disambiguation（最新追加）

- v272 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v272_physio_ambiguity_disambiguation_20260702.py`
- v272 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v272_physio_ambiguity_disambiguation_20260702`
- v272 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v272_physio_ambiguity_disambiguation_20260702\reports\v272_physio_ambiguity_disambiguation_cn.md`
- v272 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v272_physio_ambiguity_disambiguation_20260702_pack.zip`
- 方法：
  - 不训练新模型。
  - 对每个 query，用车辆 topK 找 train prototype 候选。
  - 在车辆候选内部计算 v271 calibrated physiology 距离排序。
  - 检查真正最佳候选在生理排序里的 rank，以及 bio top1/top3/top5 的 oracle 上界。
- 关键结果：
  - diagnostic_row_n `28008`。
  - test bad_top10 vehicle nearest `0.8785`。
  - vehicle candidate oracle k40 `0.6166`。
  - val 选 bio top1 test `0.8940`，比 vehicle nearest 还差。
  - test-best bio top1 diagnostic `0.8744`，仍远高于 fixed wait-latest。
  - test-best bio top3 oracle `0.6738`，低于 fixed wait-latest，但不是可部署策略。
  - bio best candidate rank 均值约 `19-21`，best-in-top3 rate 只有约 `5%-10%`。
- 判读：
  - 生理距离不能稳定把好候选排到前面。
  - 但 bio top3/top5 内有少量上界，因此值得用 v273 验证“bio 预筛 + selector”。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=True`。

---

# 2026-07-02 v271 calibrated raw physiology state（最新追加）

- v271 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v271_calibrated_raw_physio_state_20260702.py`
- v271 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v271_calibrated_raw_physio_state_20260702`
- v271 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v271_calibrated_raw_physio_state_20260702\reports\v271_calibrated_raw_physio_state_cn.md`
- v271 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v271_calibrated_raw_physio_state_20260702_pack.zip`
- 方法：
  - 复用 v256 raw 20s/20Hz 生理序列缓存，只取每个事件 0ms anchor observation 前序列。
  - 先构造 raw summary/FFT，再按 subject 和 recording 做无标签 robust z 校准。
  - 对 subject-centered / recording-centered raw waveform 做 train-only PCA。
  - 只在 train split 做 identity/behavior eta 和特征筛选。
  - 构造 `subject_summary64`、`recording_summary64`、`subject_seq_pca72`、`recording_seq_pca72`、`calibrated_screened64`、`calibrated_low_identity48` 六组特征。
  - 继续复用 wait gate 与 v267 式 query-prototype pair reranker。
- 关键边界：
  - subject/recording baseline 只使用无标签生理输入，不使用未来轨迹标签。
  - 但对 val/test 来说这是 calibrated / transductive setting，不是纯 cold-start subject-disjoint。
- 关键结果：
  - event_n `1167`。
  - raw sequence shape delay0 `[1167, 6, 400]`。
  - raw physio ok rate `0.9195`。
  - raw feature_n `505`，raw_set_n `6`，pair_row_n `280080`。
  - test bad_top10 keep0 `1.1977`。
  - fixed wait-latest `0.6950`。
  - oracle best `0.6125`。
  - pair candidate oracle k40 `0.6166`。
  - wait test-best 仍为 `0.6950`，等价于全 wait-latest。
  - pair test-best deployable 为 `subject_seq_pca72:pair_vehicle_bio_badweighted_hgb_k5 = 0.7853`，仍高于 fixed wait-latest。
  - val-best vehicle+raw 为 `calibrated_low_identity48:pair_vehicle_bio_hgb_k40 = 0.9232`，明显退化。
- 判读：
  - 个体/recording 基线校准确实降低部分身份混淆，但没有形成稳定行为预测增量。
  - v271 比 v270 best diagnostic pair 只改善约 `0.0013`，可视为无实质变化。
  - 当前证据不支持继续在生理表征/融合/reranker 上做小改动。
  - 下一步应回到车辆多未来候选、uncertainty/ranker 和可部署轨迹选择主线。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=True`。

---

# 2026-07-02 v270 raw physiology state latent（最新追加）

- v270 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v270_raw_physio_state_latent_20260702.py`
- v270 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v270_raw_physio_state_latent_20260702`
- v270 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v270_raw_physio_state_latent_20260702\reports\v270_raw_physio_state_latent_cn.md`
- v270 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v270_raw_physio_state_latent_20260702_pack.zip`
- 方法：
  - 复用 v256 raw 20s/20Hz 生理序列缓存，只取每个事件 0ms anchor 对应的 observation 前序列。
  - 信号通道为 `HR_bpm`、`EMG_RMS`、`EDA_Tonic`、`EDA_Phasic`、`RESP_filt200`、`ECG_filt200`。
  - 从 raw waveform 构造 summary/FFT 特征、PCA latent、差分 PCA latent。
  - 所有筛选、PCA scaler、identity/behavior eta 只在 train split 上拟合。
  - 构造 `raw_summary_fft`、`raw_pca96`、`raw_screened64`、`raw_low_identity48` 四组 raw-state latent。
  - 分别验证 wait gate 与 query-prototype pair reranker；val 选择策略，test 只报告。
- 关键结果：
  - event_n `1167`。
  - raw sequence shape delay0 `[1167, 6, 400]`。
  - raw physio ok rate `0.9195`。
  - raw feature_n `277`，raw_set_n `4`。
  - test bad_top10 keep0 `1.1977`。
  - fixed wait-latest `0.6950`。
  - oracle best `0.6125`。
  - pair candidate oracle k40 `0.6166`。
  - wait test-best `wait_raw_raw_summary_fft_gain = 0.6950`，但 latest_rate `1.0`，等价于全 wait-latest。
  - pair test-best deployable `raw_screened64:pair_vehicle_bio_badweighted_hgb_k20 = 0.7866`。
  - val-best vehicle+raw `raw_low_identity48:pair_vehicle_bio_hgb_k5 = 0.8142`。
- 判读：
  - raw waveform latent 没有证明“生理状态能弥补锚点前车辆信息不足”。
  - `raw_summary_fft` 的 identity eta 明显高，说明绝对生理波形仍容易携带 subject/recording 身份信息。
  - candidate oracle 仍接近 full oracle，说明候选空间仍有 headroom；失败点依然是可部署选择信号不足。
  - 后续不应继续做同类 raw 特征筛选或更换 HGB/reranker 小变体；若坚持生理，应验证个体基线、recording 内归一化、subject-aware 校准是否能把身份差异转化为可用状态差异。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=True`。

---

# 2026-07-02 v269 reliable / identity-removed physiology（最新追加）

- v269 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v269_reliable_identity_removed_physio_20260702.py`
- v269 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v269_reliable_identity_removed_physio_20260702`
- v269 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v269_reliable_identity_removed_physio_20260702\reports\v269_reliable_identity_removed_physio_cn.md`
- v269 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v269_reliable_identity_removed_physio_20260702_pack.zip`
- 方法：
  - 复用 v260 的 observation 前事件 biomarker，不使用 test 标签筛选特征。
  - 只在 train split 上计算可靠性、身份 eta、行为 eta，并构造 `reliable_top64`、`dynamic_top48`、`low_identity_top32`、`combo_identity_removed64` 四组特征。
  - 动态特征使用同一生理指标跨窗口差分，例如 pre2_0 与 pre10_pre5 / pre5_pre2 / pre20_pre10 的差。
  - 分别验证 wait gate 与 v267 式 query-prototype pair reranker；threshold / strategy 仍由 val bad_top10 选择，test 只报告。
- 关键结果：
  - test bad_top10 keep0 `1.1977`。
  - fixed wait-latest `0.6950`。
  - oracle best `0.6125`。
  - pair candidate oracle k40 `0.6166`。
  - wait gate best deployable `0.6950`，但 latest_rate `1.0`，实际等价于全 wait-latest。
  - pair test-best deployable `combo_identity_removed64:pair_base_hgb_k40 = 0.7781`，不是生理特征主导策略，且仍高于 fixed wait-latest。
  - test-best vehicle_bio 诊断可到 `low_identity_top32:pair_vehicle_bio_badweighted_hgb_k5 = 0.7981`，仍高于 fixed wait-latest。
  - val-best vehicle+bio 为 `combo_identity_removed64:pair_vehicle_bio_badweighted_hgb_k5`，test bad_top10 `0.8365`。
  - 相比 v267 val-best vehicle+bio `0.8495`，v269 只改善约 `0.0130`。
- 判读：
  - 去掉明显不可用/高身份混淆特征后，生理不是完全无变化，但改善幅度很小。
  - v269 进一步证明问题不是“再筛一次特征”或“再换一个 HGB/MLP/reranker”即可解决。
  - 当前 subject-disjoint 生理路线仍未达到用户 goal 的结束标准。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`，共 `23` 个文件。

---

# 最新状态指针：2026-07-02 已完成 v268 physiology quality / alignment / identifiability audit。v268 不训练新预测模型，而是审计现有生理链路是否能支撑 goal。结论：200Hz 连续生理源层时序稳定，`82` 个 recording / `18` 个 subject，median_hz `200.000`，gap/duplicate 均为 `0`；事件窗口覆盖可用，min split-delay ok_rate `0.889`，post-observation rate `0`。但派生层有结构性问题：`HRV_RMSSD`、`RESP_BPM`、`RESP_Amplitude` usable 均为 `0/82`，EDA 有 `9` 个 recording 近常数/缺失；bio260 的 subject/recording 可分性远高于行为/等待收益可分性，median family identity/behavior eta ratio `68.74`。在 v267 候选重排诊断中，test bad_top10 的 `pred_pair_vehicle_bio_hgb` 仍比 fixed latest 差 `+0.1509`，true best top3 rate 只有 `0.211`。当前 blocker 收敛为：不是源采样断裂，也不是再加深 reranker，而是当前派生生理表征在 subject-disjoint 下可迁移行为信息太弱且身份混淆强。goal 尚未完成；若继续生理路线，下一步必须重建/清洗 HRV、RESP、EDA 等可靠表征，并做去身份化/个体内归一化后再进入候选选择或 wait gate。

---

# 2026-07-02 v268 physiology quality / alignment / identifiability audit（最新追加）

- v268 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v268_physio_quality_identifiability_audit_20260702.py`
- v268 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v268_physio_quality_identifiability_audit_20260702`
- v268 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v268_physio_quality_identifiability_audit_20260702\reports\v268_physio_quality_identifiability_audit_cn.md`
- v268 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v268_physio_quality_identifiability_audit_20260702_pack.zip`
- 方法：
  - 不训练新模型，专门审计生理链路。
  - 检查 source recording 时序质量、核心信号可用性、v260 事件窗口覆盖、bio260 身份/recording 信号与行为信号强弱，以及 v267 候选排序是否真能把最佳候选排到前面。
- 关键结果：
  - source recording：`82` 个 recording，`18` 个 subject，median_hz `200.000`，gap_gt_20ms_total `0`，duplicate_t_total `0`，core columns present rate `1.0`。
  - 信号可用性：ECG/EMG/HR/RESP raw-filt 基础列整体可用；`HRV_RMSSD` usable `0/82`，`RESP_BPM` usable `0/82`，`RESP_Amplitude` usable `0/82`；EDA 相关列 usable `73/82`，存在 `9` 个 near-constant / missing recording。
  - 事件窗口覆盖：train/test ok_rate 约 `0.889/0.897`，val `1.0`；post-observation rate `0`，说明不是大面积窗口缺失或未来泄漏。
  - 事件特征缺失：HRV 事件特征全缺失/零方差；RESP、SCR 缺失率偏高。
  - 身份 vs 行为可识别性：各 family 的 behavior_eta_max_mean 都低于 `0.006`，而 identity_eta_max_mean 约 `0.109-0.211`；median identity/behavior eta ratio 为 `68.74`。
  - 候选排序：test bad_top10 上，candidate oracle 均值 `0.6166`，但 `pred_pair_vehicle_bio_hgb` chosen_rmse `0.8460`，比 fixed latest 差 `+0.1509`，true best top3 rate `0.211`。
- 判读：
  - 原始 200Hz 连续层不是主要失败点。
  - 现有派生生理表征质量不够，且更容易携带 subject/recording 差异，而不是可迁移行为差异。
  - 因此当前生理不能简单拼接到模型里继续堆深度；下一步若继续生理，应先做 reliable physiology rebuild + identity-removed representation，再回到 wait gate / candidate ranker。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`，共 `19` 个文件。

---

# 2026-07-02 v267 supervised bio prototype reranker（最新追加）

- v267 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v267_supervised_bio_prototype_reranker_20260702.py`
- v267 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v267_supervised_bio_prototype_reranker_20260702`
- v267 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v267_supervised_bio_prototype_reranker_20260702\reports\v267_supervised_bio_prototype_reranker_cn.md`
- v267 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v267_supervised_bio_prototype_reranker_20260702_pack.zip`
- 方法：
  - 复用 v266 的 vehicle-matched topK prototype 候选。
  - 构造 `46680` 个 query-prototype pair。
  - 训练 pair_base、pair_vehicle、pair_bio、pair_vehicle_bio、pair_vehicle_bio_badweighted 五个 HGB reranker。
  - 输入只使用 0ms 车辆上下文、observation 前 bio260_sp64、prototype 的 train 已知历史结果与 query/prototype 距离。
  - train-only 拟合；val 选择 K/模型；test 只报告。
- 关键结果：
  - test bad_top10 keep0 `1.1977`。
  - fixed wait-latest `0.6950`。
  - full oracle `0.6125`。
  - candidate oracle k40 `0.6166`。
  - val-best pair vehicle：`pair_vehicle_hgb_k40`，test bad_top10 `0.8746`。
  - val-best pair vehicle+bio：`pair_vehicle_bio_badweighted_hgb_k5`，test bad_top10 `0.8495`。
  - bio 相比 vehicle-only 改善 `0.0251`，但仍高于 fixed wait-latest `0.1545`。
  - test 诊断最好的 bio pair：`pair_vehicle_bio_hgb_k20` 为 `0.8046`，仍未低于 fixed wait-latest，且不是 val 选择结果。
- 判读：
  - v267 比 v266 更强，但仍没有完成 goal。
  - 当前生理在候选内部排序中最多提供弱增量，不能稳定解决“选中正确候选”的核心问题。
  - 继续做同类 pairwise/reranker 生理变体，边际价值较低；若继续 goal，下一步应转向生理数据质量/对齐可识别性审计，或改变任务边界到 subject-aware。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`，共 12 个文件。

---

# 最新状态指针：2026-07-02 已完成 GPTPro phase02 桌面软件复核与 v266 vehicle-matched bio residual prototype。GPTPro 判断：subject-disjoint 下不应再把生理当普通预测模态融合，最多只保留 wait-benefit 决策、vehicle-matched residual prototype reranking、subject-aware online calibration 三条可证伪路线。v265 已覆盖 wait-benefit，未形成 bio 增量；v264 已覆盖 subject-aware online，生理 KNN 无额外收益；v266 覆盖 vehicle-matched prototype reranking。v266 显示：车辆相似 prototype 候选库本身有 headroom，test bad_top10 candidate oracle 可到 `0.6166`，接近完整 oracle `0.6125` 并低于 fixed wait-latest `0.6950`；但 val 选出的可部署 reranker 失败，vehicle-only 为 `0.8890`，vehicle+bio 为 `0.8374`，bio 比 vehicle-only 低 `0.0516`，但仍比 fixed wait-latest 高 `0.1423`。结论：路线2证明“候选存在”，但当前 bio260 仍不能可靠选择正确候选；goal 尚未达成。subject-disjoint 生理主线已完成三条外部建议路线的最小验证，继续堆生理融合/selector 不合理；下一步应转回车辆多未来/不确定性/候选轨迹选择主线，或明确改成 subject-aware 个体校准任务。

---

# 2026-07-02 GPTPro phase02 + v266 vehicle-matched bio residual prototype（最新追加）

- GPTPro 发送方式：ChatGPT 桌面软件，界面显示 Pro / Pro 扩展。
- GPTPro 提问词：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase02_prompt.md`
- GPTPro 回复：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase02_response.md`
- GPTPro 原始可访问性树：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase02_response_raw_accessibility.txt`
- v266 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v266_vehicle_matched_bio_residual_prototype_20260702.py`
- v266 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v266_vehicle_matched_bio_residual_prototype_20260702`
- v266 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v266_vehicle_matched_bio_residual_prototype_20260702\reports\v266_vehicle_matched_bio_residual_prototype_cn.md`
- v266 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v266_vehicle_matched_bio_residual_prototype_20260702_pack.zip`
- 方法：
  - prototype 只来自 train split，不使用 val/test 驾驶员历史。
  - query 只使用 0ms 车辆上下文与 floor 0ms 的 bio260_sp64 特征。
  - 先用车辆上下文找 topK 相似 train 事件。
  - 将 train prototype 的 oracle delay 映射到 query 事件的候选 delay 上，形成少量 vehicle-matched residual/anchor 候选。
  - 生理只在这些候选内部重排序，不直接生成轨迹，不做全局 selector。
  - K/lambda 只根据 val bad_top10 选择，test 不调参。
- 关键结果：
  - test bad_top10 keep0 `1.1977`。
  - fixed wait-latest `0.6950`。
  - full oracle `0.6125`。
  - vehicle-matched candidate oracle k40 `0.6166`，说明相似车辆 prototype 候选库理论上几乎覆盖了 oracle headroom。
  - val 选出的 vehicle-only prototype：`prototype_vehicle_vote_k10`，test bad_top10 `0.8890`。
  - val 选出的 vehicle+bio prototype：`prototype_bio_closest_k3`，test bad_top10 `0.8374`。
  - bio 相比 val-best vehicle-only 改善 `0.0516`，但仍显著弱于 fixed wait-latest `0.6950`。
  - 即使只看 test 上最好的可部署候选，vehicle-only k40 `0.7888`、bio k40 `0.7989`，仍未低于 fixed wait-latest。
- 判读：
  - 路线2不是“候选库没有上限”，而是“可部署选择信号不够强”。
  - 当前 bio260 可以在某些重排比较里带来小幅改善，但没有达到本质改善门槛。
  - 与 v264/v265 合并后，GPTPro 建议的三条 subject-disjoint/边界验证路线均未证明当前生理能完成 goal。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`，共 12 个文件。

---

# 最新状态指针：2026-07-02 追加完成 v265 physiology uncertainty / wait frontier。v265 验证最后一个较合理用途：生理是否能作为不确定性/风险校准信号，在固定等待预算下更准确挑出需要 wait-latest 的样本。结果仍不支持 goal：test bad_top10 上所有风险分数的最佳 RMSE 都退化到全 wait-latest `0.6950`；在等待比例受限时，vehicle+bio_badprob 有局部改善，但不稳定且不支配 vehicle_gain，无法超过 fixed wait-latest，也无法接近 oracle `0.6125`。分数诊断显示 test 上 `score_vehicle_bio_badprob` 的 bad_top10 AUC `0.6175`、`score_bio_only_badprob` AUC `0.6376`，说明生理仍有弱风险信号；但这个信号不能可靠转化为更好的可部署等待策略。至此，生理直接预测、selector、wait gate、subject-invariant selector、online subject-aware 校准、风险前沿均未达到“差样本本质改善”。goal 仍未完成，且当前主要 blocker 是：在正式 subject-disjoint 边界下，现有生理数据缺乏足够强的可部署增量；继续局部加模型已不合理，需要用户确认是否改变任务边界或补充外部复核/数据。

---

# 2026-07-02 v265 physiology uncertainty / wait frontier（最新追加）

- 当前状态：验证“生理作为不确定性/风险校准信号”这一最后一个较合理用途。
- v265 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v265_physio_uncertainty_wait_frontier_20260702.py`
- v265 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v265_physio_uncertainty_wait_frontier_20260702`
- v265 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v265_physio_uncertainty_wait_frontier_20260702\reports\v265_physio_uncertainty_wait_frontier_cn.md`
- v265 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v265_physio_uncertainty_wait_frontier_20260702_pack.zip`
- 方法：
  - 只训练风险/收益分数，不输出新轨迹。
  - policy 只在 keep0 和 wait-latest 间选择。
  - 分数模型只在 train split 拟合。
  - 各等待比例阈值只在 val split 定标。
  - test 只报告等待前沿。
- 风险分数：
  - vehicle gain / vehicle+bio gain / bio-only gain。
  - vehicle keep0 risk / vehicle+bio keep0 risk / bio-only keep0 risk。
  - vehicle badprob / vehicle+bio badprob / bio-only badprob。
  - vehicle oracle gap / vehicle+bio oracle gap。
- 关键结果：
  - test bad_top10 最小 RMSE 全部退化为全 wait-latest `0.6950`。
  - fixed wait-latest `0.6950`，oracle `0.6125`。
  - 在受限等待比例下，vehicle+bio_badprob 有局部改善，例如 target wait `0.4` 时 tail `0.8607`，但实际 selected_latest_rate 已到 `0.6316`，且不稳定。
  - 同等高等待比例下，vehicle_gain 仍更稳：target wait `0.9` 时 vehicle_gain `0.6959`，vehicle+bio_badprob `0.7157`。
  - test 分数诊断中，bio 风险分数有弱 bad_top10 AUC：vehicle+bio_badprob `0.6175`，bio_only_badprob `0.6376`；但这个弱 AUC 没有转化为可部署策略收益。
- 判读：
  - 生理可以弱识别“可能差”的样本，但无法稳定决定哪些样本应该等待或如何减少等待成本。
  - 当前生理信号不能作为 subject-disjoint 正式预测的主增量，也不能作为可靠 uncertainty calibration。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`。

---

# 最新状态指针：2026-07-02 追加完成 v264 online subject-aware physiology calibration，并尝试向 GPTPro 发送 phase02 复核但因无法确认 Pro/进阶模式被桥接脚本拒绝发送。v264 是任务边界实验，不是正式 subject-disjoint 替代结果：它允许同一驾驶员更早事件的已知结果做在线校准。结果显示，online subject 历史反馈能把 test bad_top10 vehicle gate 从 `0.7528` 推近 fixed wait-latest，到 `0.7112`；但 physiology KNN 没有额外收益，`online_physio_knn_vehicle` 仍为 `0.7112`，`online_physio_knn_vehicle_bio` 反而为 `0.7698`。这说明即使放宽到 subject-aware online，真正起作用的是同驾驶员历史反馈和多观察策略，不是当前 bio260 生理特征本身。goal 仍未完成；当前证据更支持“暂停把生理作为 subject-disjoint 主增量”，转向车辆多未来/不确定性主线，或另立 subject-aware 个体校准任务。

---

# 2026-07-02 v264 online subject-aware 生理校准边界实验（最新追加）

- 当前状态：在 v260-v263 后进一步验证“是不是只有 subject-aware / online adaptation 设定下生理才有用”。
- GPTPro phase02 提问词已归档：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase02_prompt.md`
- GPTPro 发送状态：未发送。桥接脚本无法确认 Chrome 当前为 Pro/进阶模式，按规则拒绝发送，避免误发到非 Pro 模式。
- v264 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v264_online_subject_physio_calibration_20260702.py`
- v264 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v264_online_subject_physio_calibration_20260702`
- v264 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v264_online_subject_physio_calibration_20260702\reports\v264_online_subject_physio_calibration_cn.md`
- v264 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v264_online_subject_physio_calibration_20260702_pack.zip`
- 方法：
  - 以 v263 0ms wait gate 为基础，只决策 keep0 或 wait-latest。
  - 全局 gain 模型仍只在 train subjects 上训练。
  - 对 val/test subjects，按 recording + observation_s 顺序，只用同一 subject 更早事件的已知 gain residual 做在线校准。
  - 比较纯 subject residual mean/recent 与 physiology KNN residual，判断生理是否提供额外个体内状态区分能力。
- 关键结果：
  - test bad_top10 keep0 `1.1977`，fixed wait-latest `0.6950`，oracle `0.6125`。
  - global vehicle gate `0.7528`。
  - global vehicle+bio260_sp64 gate `0.8748`，比 vehicle gate 更差。
  - online_subject_mean_vehicle `0.7112`，说明同驾驶员历史反馈能接近 wait-latest。
  - online_physio_knn_vehicle `0.7112`，没有超过纯 subject mean。
  - online_subject_mean_vehicle_bio `0.6950`，实际退化成全 wait-latest。
  - online_physio_knn_vehicle_bio `0.7698`，比 subject mean 和 fixed wait-latest 都差。
- 判读：
  - 放宽到 subject-aware online 后，个体历史反馈有价值。
  - 但 physiology KNN 没有额外收益，甚至在 vehicle+bio 路线上拉低结果。
  - 因此，当前 bio260 生理特征仍未证明能弥补锚点前车辆信息不足；更强证据指向“历史反馈/多观察”而不是“生理状态”。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`。

---

# 最新状态指针：2026-07-02 已完成 v260-v263 事件级生理重构与 wait/anchor 决策复核。结论：v260 从 200Hz 连续波形重算 ECG/EDA/RESP/EMG 事件 biomarker 后，bio260 相比旧 physio200 在 bad_top10 诊断上略有改善，但直接行为预测仍弱；v261 全量 bio260 anchor selector 未超过 vehicle selector；v262 subject-invariant bio260 sp64 在 test bad_top10 上只把 selector tail 从 `0.9419` 小幅降到 `0.9059`，仍远高于固定 wait-latest `0.6950`；v263 0ms wait gate 中 vehicle gate `0.7528` 反而优于 vehicle+bio260 gate `0.8748`，val 调阈值退化为几乎全等 latest。当前证据说明：生理数据不是完全无信号，但尚未形成能“极大弥补锚点前信息不足、让差样本本质改善”的可部署增量。goal 仍未完成；下一步若继续追生理，应转向更明确的 subject-aware 个体校准/额外数据质量修复，或把主线回到车辆多模态不确定性/多未来分布建模。

---

# 2026-07-02 v260-v263 生理重构与决策复核已完成（最新）

- 当前状态：继续执行用户 goal“充分利用生理数据，弥补锚点前车辆信息不足，并让差样本有本质性改善”。本轮在 v257-v259 后继续检查“是不是生理表征不够好、selector 目标太复杂、subject 混淆太强”三个可能原因。
- v260 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v260_event_biomarker_physio_rebuild_20260702.py`
- v260 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v260_event_biomarker_physio_rebuild_20260702`
- v260 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v260_event_biomarker_physio_rebuild_20260702\reports\v260_event_biomarker_physio_rebuild_cn.md`
- v260 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v260_event_biomarker_physio_rebuild_20260702_pack.zip`
- v261 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v261_bio260_anchor_selector_20260702.py`
- v261 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v261_bio260_anchor_selector_20260702`
- v261 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v261_bio260_anchor_selector_20260702\reports\v261_bio260_anchor_selector_cn.md`
- v261 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v261_bio260_anchor_selector_20260702_pack.zip`
- v262 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v262_subject_invariant_bio260_selector_20260702.py`
- v262 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v262_subject_invariant_bio260_selector_20260702`
- v262 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v262_subject_invariant_bio260_selector_20260702\reports\v262_subject_invariant_bio260_selector_cn.md`
- v262 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v262_subject_invariant_bio260_selector_20260702_pack.zip`
- v263 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v263_bio260_wait_gate_20260702.py`
- v263 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v263_bio260_wait_gate_20260702`
- v263 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v263_bio260_wait_gate_20260702\reports\v263_bio260_wait_gate_cn.md`
- v263 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v263_bio260_wait_gate_20260702_pack.zip`
- v260 关键发现：
  - 从 200Hz 连续波形重新派生 ECG peak/IBI/SDNN/RMSSD、EDA/SCR、RESP zero-cross/phase、EMG burst 等事件 biomarker。
  - 对齐守卫通过：`guardrail_check.pass=True`，`bio260_uses_post_observation_max=0`，ok rate 约 `0.919`。
  - subject-disjoint bad_top10 诊断中，旧 physio200 macro-F1 `0.4482`，bio260 `0.4947`，vehicle+bio260 `0.5120`，说明重构后有弱信号。
  - 但 future_cluster4 / high_future_abs_q75 / future summary 回归仍未超过 vehicle-only，不能直接作为行为预测增量。
- v261 关键发现：
  - 使用 v260 bio260 全量特征参与 v247 fine-grid anchor selector。
  - test bad_top10：keep0 `1.1977`，wait-latest `0.6950`，oracle `0.6125`，vehicle selector `0.9425`，bio260-only `1.0180`，vehicle+bio260 `0.9765`，badweighted vehicle+bio260 `0.9837`。
  - 结论：全量 bio260 没有帮助 anchor selector，反而弱于 vehicle selector。
- v262 关键发现：
  - 按 v260 eta2 惩罚 subject/recording 混淆，筛选 subject-invariant bio260 特征。
  - test bad_top10：vehicle selector `0.9419`，vehicle+bio260_sp64 `0.9059`，sp32 `0.9819`，state_change `0.9547`，badweighted sp64 `1.0631`。
  - 结论：去 subject 混淆后出现小幅正增益，但幅度约 `0.036`，仍远离 wait-latest `0.6950` 和 oracle `0.6125`，不能称为本质改善。
- v263 关键发现：
  - 将任务简化为 0ms wait gate：只判断保留 0ms 还是等到 1000ms。
  - test bad_top10：keep0 `1.1977`，wait-latest `0.6950`，oracle `0.6125`，vehicle gate `0.7528`，vehicle+bio260_sp64 gate `0.8748`。
  - val 自动调阈值时，vehicle+bio260 gate 退化为几乎全部 wait-latest，说明不是生理判断起作用，而是“多观察本身”起作用。
- 当前总判断：
  - 生理不是完全没有信息；事件级重构后确实能产生很弱的 bad_top10 风险信号。
  - 但这个信号不足以在 subject-disjoint 正式任务中稳定改善轨迹预测、候选锚点选择或 wait gate。
  - goal 尚未达成。若继续生理路线，建议明确改为 subject-aware 个体校准或先修复/补强生理数据质量；若坚持 subject-disjoint 正式预测，应把主线转回车辆轨迹多未来分布/不确定性建模，而不是继续堆生理融合结构。
- 校验：
  - v260/v261/v262/v263 均通过 `python -m py_compile`。
  - 四个脚本均完整运行完成。
  - 四个 `guardrail_check.pass=True`。
  - 四个 ZIP 均 `testzip=None`。

---

# 最新状态指针：2026-07-02 已完成 v254b-v259 生理深挖阶段。结论：已经从 10Hz/1Hz 统计推进到 200Hz 事件表征、候选轨迹重排序、raw 200Hz CNN、同驾驶员记忆、physio anchor selector、raw 生理 cross-attention 直接预测。当前证据一致显示：现有生理层在正式 subject-disjoint 轨迹预测中没有稳定增量，不能达到用户 goal 中“极大弥补锚点前信息不足、让差样本本质改善”的结束标准。关键 locked 结果：v258 test bad_top10 中固定 wait-latest 可从 `1.1977` 降到 `0.6950`，但 vehicle+physio selector `0.9342` 没有超过 vehicle-only selector `0.9300`；v259 subject-disjoint bad_top10 中 v250 `0.8783`，v259 vehicle-only `0.9267`，vehicle+physio cross-attention `1.0889`，badweighted `1.0351`。下一步若继续生理主线，应优先改变生理数据层或任务边界，例如重算可靠 HRV/RESP/SCR、增加同驾驶员校准，或明确转为 subject-aware 个体化；不建议继续做同类简单融合/浅层 CNN/MLP/attention 盲试。

---

# 2026-07-02 v257-v259 生理深挖后续已完成（最新）

- 当前状态：继续执行用户设置的 goal“充分利用生理数据，弥补锚点前车辆信息不足，并让差样本有本质性改善”。在 v254b-v256 之后，已继续完成同驾驶员记忆、physio-augmented anchor selector、raw 生理 cross-attention 直接预测三条更强路线。
- v257 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v257_subject_personalized_physio_memory_20260702.py`
- v257 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v257_subject_personalized_physio_memory_20260702`
- v257 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v257_subject_personalized_physio_memory_20260702\reports\v257_subject_personalized_physio_memory_cn.md`
- v257 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v257_subject_personalized_physio_memory_20260702_pack.zip`
- v258 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v258_physio_augmented_anchor_selector_20260702.py`
- v258 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v258_physio_augmented_anchor_selector_20260702`
- v258 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v258_physio_augmented_anchor_selector_20260702\reports\v258_physio_augmented_anchor_selector_cn.md`
- v258 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v258_physio_augmented_anchor_selector_20260702_pack.zip`
- v259 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v259_physio_cross_attention_prediction_20260702.py`
- v259 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v259_physio_cross_attention_prediction_20260702`
- v259 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v259_physio_cross_attention_prediction_20260702\reports\v259_physio_cross_attention_prediction_cn.md`
- v259 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v259_physio_cross_attention_prediction_20260702_pack.zip`
- v257 关键发现：
  - 本轮检验 subject-aware 个体化记忆：同一驾驶员历史事件是否能通过车辆/生理状态检索出更接近的未来。
  - validation 选择 `same_subject_vehicle_k3`，不是生理增强策略。
  - test bad_top10：`v250_existing` tail RMSE `0.8383`；`same_subject_vehicle_k3` tail RMSE `1.3054`，delta `+0.4671`。
  - 加入生理/原始序列 PCA 的同驾驶员记忆策略也没有超过 v250。
- v258 关键发现：
  - 本轮检验生理是否能帮助“是否等待/重锚定”的 anchor selector。
  - test bad_top10：`policy_keep_0ms_anchor` tail `1.1977`；`policy_wait_to_latest_anchor` tail `0.6950`；oracle best `0.6125`。
  - `selector_vehicle_hgb` tail `0.9300`；`selector_vehicle_physio_hgb` tail `0.9342`；`selector_vehicle_physio_badweighted_hgb` tail `0.9593`。
  - 结论：等待本身有效，但生理没有帮助 selector 超过 vehicle-only，更没有超过固定 wait-latest。
- v259 关键发现：
  - 本轮不做删样本、不做 gate/router、不做 residual 修正；直接训练车辆时序 + raw 生理时序 cross-attention 单轨迹预测模型。
  - subject-disjoint bad_top10：`v250_existing` tail `0.8783`；`v259_vehicle_attn` tail `0.9267`；`v259_vehicle_physio_crossattn` tail `1.0889`；`v259_vehicle_physio_crossattn_badweighted` tail `1.0351`。
  - subject-aware bad_top10：`v250_existing` tail `0.8383`；`v259_vehicle_attn` tail `1.0038`；`v259_vehicle_physio_crossattn` tail `1.1288`。
  - 结论：更深的 raw 生理 cross-attention 仍没有形成有效增量，反而明显拖累同架构 vehicle-only。
- 当前总判断：
  - v254b-v259 覆盖了手工 200Hz 特征、候选重排序、raw-CNN 融合、同驾驶员记忆、physio anchor selector、raw cross-attention 直接预测。
  - 当前证据不支持“现有生理数据能极大弥补锚点前车辆信息不足、并让差样本本质改善”。
  - goal 尚未完成。若继续强行追生理，需要先改变生理数据层或任务边界，例如更可靠的 HRV/RESP/SCR 重算、更多同驾驶员校准数据，或把正式目标改为 subject-aware 个体化而非 subject-disjoint。
- 验证：
  - v257/v258/v259 均通过 `python -m py_compile`。
  - 三个脚本均完整运行完成。
  - 三个 `guardrail_check.pass=True`。
  - 三个 ZIP 均 `testzip=None`。

---

# 2026-07-02 v254b-v256 生理数据深挖阶段审计完成（最新）

- 当前状态：按用户新 goal“充分利用生理数据，弥补锚点前车辆信息不足，并让预测差样本本质改善”推进。已完成三条不同层级的生理路线：200Hz 事件表征、学习式候选重排序、raw 200Hz CNN 融合预测。
- 本阶段重要前提：
  - 不继续 v222a gate / 删除样本 / 轻量 residual 路线。
  - 不把问题转成失败机制论文，目标仍是行为轨迹预测方法提升。
  - 不使用 observation_s 后生理或 query 未来信息。
  - subject-disjoint 是正式泛化口径；subject-aware 仅作为同一驾驶员个体化潜力诊断。

## v254b：200Hz 连续生理事件相关表征

- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v254b_physio_200hz_event_representation_20260702.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254b_physio_200hz_event_representation_20260702`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254b_physio_200hz_event_representation_20260702\reports\v254b_physio_200hz_event_representation_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254b_physio_200hz_event_representation_20260702_pack.zip`
- 方法：从清洗后 200Hz 连续层直接抽取 observation_s 前事件窗口，使用每个样本自身 `[-60s,-20s]` baseline 做因果归一化；窗口包括 `pre20_pre10 / pre10_pre5 / pre5_pre2 / pre2_0`。
- 覆盖：`ok_rate` test `0.8967`、train `0.8887`、val `1.0`；`uses_post_observation_rate=0`。
- 关键结果：
  - subject-disjoint / high_future_abs_q75：vehicle_only macro-F1 `0.7408`，vehicle+physio200_curated `0.6169`，生理融合变差。
  - subject-disjoint / future_cluster4：vehicle_only `0.7154`，vehicle+physio200_curated `0.6852`，仍弱于车辆。
  - subject-disjoint / bad_top10_v250_diagnostic：vehicle_only `0.4958`，vehicle+physio200_curated `0.5170`，只有很小诊断增量。
  - subject-aware / bad_top10_v250_diagnostic：vehicle_only `0.4578`，vehicle+physio200_norm `0.6095`，说明生理对“同一驾驶员下哪些样本可能差”有一定个体化诊断信号。
- 判断：200Hz 手工事件表征不能带来正式 subject-disjoint 行为预测增量；它更像 subject/recording/个体状态信号。

## v255：生理状态条件化候选轨迹选择

- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v255_physio_conditioned_candidate_ranker_20260702.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v255_physio_conditioned_candidate_ranker_20260702`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v255_physio_conditioned_candidate_ranker_20260702\reports\v255_physio_conditioned_candidate_ranker_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v255_physio_conditioned_candidate_ranker_20260702_pack.zip`
- 方法：车辆输入先取同 delay top60 候选池，再训练 pair ranker 用车辆距离、候选未来原型摘要、query-candidate 生理距离和 recent 生理 index 差异预测候选 future RMSE；val 上用 no-harm 阈值决定是否重排。
- 关键结果：
  - val 上所有 learned ranker 一旦允许重排都会伤害 bad_top10，no-harm 阈值最终全选 `1e9`，即退回 vehicle_rank1。
  - subject-disjoint test bad_top10：vehicle_rank1 `0.9934`，learned_physio_state_guarded `0.9934`，oracle `0.3678`。
  - subject-aware test bad_top10：vehicle_rank1 `0.9838`，learned_physio_state_guarded `0.9838`，oracle `0.4403`。
- 判断：候选池内仍有巨大 oracle 上限，但当前生理状态表示不能可靠选择候选未来；不是“简单最近邻太弱”一个问题，学习式重排序也没选出来。

## v256：raw 200Hz 生理 CNN 融合预测

- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v256_raw_physio_cnn_fusion_20260702.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v256_raw_physio_cnn_fusion_20260702`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v256_raw_physio_cnn_fusion_20260702\reports\v256_raw_physio_cnn_fusion_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v256_raw_physio_cnn_fusion_20260702_pack.zip`
- 方法：每个样本抽取 observation_s 前 20s raw 生理序列，下采样到 20Hz、400 步，通道为 `HR_bpm / EMG_RMS / EDA_Tonic / EDA_Phasic / RESP_filt200 / ECG_filt200`；用 1D CNN 学生理表征，与车辆 MLP 融合预测 21 点未来轨迹。
- 覆盖：`6438/7002` 样本有生理序列，`ok_rate=0.91945`；缺失样本用零序列和 `physio_ok=0` 标记。
- 关键结果：
  - subject-disjoint test bad_top10：`v256_vehicle_only` tail RMSE `0.8411`，`v256_vehicle_physio_cnn` `0.9138`，delta `+0.0727`，融合变差。
  - subject-disjoint test all：vehicle `0.4262`，fusion `0.4671`，delta `+0.0410`。
  - subject-aware test bad_top10：vehicle `0.9272`，fusion `0.9114`，delta `-0.0158`，只有很小改善。
  - 纯生理 CNN 明显弱：subject-disjoint bad_top10 tail RMSE `1.7119`。
- 判断：失败不只是 v254b 手工统计太浅；raw 生理时序 CNN 在正式 subject-disjoint 口径也没有稳定增量。subject-aware 只有极小幅度改善，达不到“本质改善差样本”标准。

## GPTPro 外部复核状态

- 已归档提问词：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase01_prompt.md`
- 发送状态：未发送。桥接脚本无法确认 Chrome 当前为 Pro/进阶模式，按规则拒绝发送，避免误发到非 Pro 模式。

## 当前收口判断

- 到 v256 为止，现有生理数据在正式 subject-disjoint 预测任务中没有证明能“极大弥补锚点前车辆信息不足”。
- 生理信号不是空的，但主要编码 subject/recording/个体状态；它对未来行为的跨驾驶员增量很弱。
- 若论文/模型仍坚持 subject-disjoint 跨驾驶员泛化，不建议继续在当前生理特征上做简单加深、加权、重排序或 CNN/TCN 盲试。
- 更合理的两条后续路线：
  1. 若必须使用生理：转为 subject-aware / 少量个体校准范式，明确“同一驾驶员历史样本可用时，生理作为个体化状态校准信号”。
  2. 若目标仍是正式跨驾驶员预测提升：回到车辆/任务构造主线，例如 anchor-aware 等待决策、显式多模态轨迹分布或更强车辆时序模型；不要把主要希望压在当前生理数据上。
- 当前 goal 的结束标准尚未达成：差样本没有出现本质改善。当前阶段产物的价值是给出强证据边界，避免继续在低收益生理拼接/重排序上消耗实验预算。


# 2026-07-01 v254a physio deep signal audit 已完成（深挖 10Hz 生理后仍未形成行为增量）

- 当前状态：已完成 `v254a_physio_deep_signal_audit_20260701`。本轮回应用户“不是简单拼接，应深层挖掘生理数据”的要求。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v254a_physio_deep_signal_audit_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\reports\v254a_physio_deep_signal_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701_pack.zip`
- 方法：从 `physio_features_10hz.csv` 提取锚点前多窗口特征，不再只用 v253a 的 1Hz 粗窗口；窗口包括 `pre20_pre10/pre10_pre5/pre5_pre2/pre2_0`，特征包括均值、分位数、波动、斜率、首末差和窗口差值。
- 训练边界：只训练轻量 logistic/ridge 诊断头；train 拟合，val/test 只报告；不训练轨迹预测模型，不用 test 调特征。
- 对齐覆盖：10Hz recording 覆盖率全体 `0.919`，test `0.897`，train `0.889`，val `1.000`；所有窗口 `uses_post_observation_rate=0`。
- 特征维度：vehicle_only `268`；physio10hz_deep 原始 `840`、保留 `700`；physio1hz_v253a 原始 `209`、保留 `184`；vehicle+physio10hz 保留 `968`。
- 生理质量：HRV_RMSSD 基本不可用，train finite rate `0.063` 且 near-constant rate `1.0`；RESP_BPM/RESP_Amplitude 仍有较高近常数比例；EDA/HR/EMG/RESP 波形统计覆盖较好。
- 可分性：subject/recording 的 eta² 很高，说明生理含身份/记录结构；但行为标签很弱，future_cluster4 top eta² 约 `0.015`，high_future_abs_q75 top eta² 约 `0.020`，strong_steer_existing top eta² 约 `0.023`。
- test 分类：future_cluster4 macro-F1，vehicle_only `0.7317`，physio10hz_deep `0.2944`，physio1hz_v253a `0.2954`，vehicle+physio10hz `0.5020`；high_future_abs_q75，vehicle_only `0.7112`，physio10hz `0.4897`，vehicle+physio10hz `0.6239`。
- test 回归：future_peak_abs R²，vehicle_only `-0.0640`，physio10hz `-2.6815`，physio1hz `-0.6691`，vehicle+physio10hz `-1.0844`；其他未来摘要同样没有生理增量。
- 判读：v254a 不证明生理没有价值，而是说明“当前 10Hz/1Hz 窗口统计 + subject-disjoint 泛化口径”没有把生理身份/记录结构转化成行为预测增量。
- 下一步：优先重做生理表征，而不是继续简单拼接。建议 `v254b_physio_representation_redesign`：200Hz 连续层事件相关变化、个体内 baseline normalization、EDA/EMG/HR/RESP 专用特征、短时动态编码，并明确 subject-aware vs subject-disjoint 两种评估口径。
- 验证：`python -m py_compile` 通过；完整脚本运行完成；`guardrail_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`。运行中有 sklearn logistic convergence warning，因此诊断头结果只作为方向性审计，不作为最终模型结论。

---

# 2026-07-01 v253b physio/style state tie-break audit 已完成（车辆相似池内用生理/风格重排序未成立）

- 当前状态：已完成 `v253b_physio_state_tiebreak_audit_20260701`。本轮回应用户的新假设：生理状态不应简单全局拼接，而应在“车辆数据看起来很像”的样本之间提供驾驶员状态区别。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v253b_physio_state_tiebreak_audit_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\reports\v253b_physio_state_tiebreak_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701_pack.zip`
- 方法：对每个 test rolling sample，先用 v250/v252 的 vehicle-only 输入在同 delay train 样本里找 top60 候选池；然后分别用 `style_nearest`、`physio_recent_nearest`、`physio_guarded_nearest`、`style_physio_nearest` 在池内选一个候选；用真实未来 RMSE 只做诊断评价，不作为部署输入。
- 关键结果：v250 bad_top10 all-delay，vehicle_rank1 selected future RMSE `0.9934`；style `1.1043`；physio_recent `1.1076`；physio_guarded `1.1930`；style+physio `1.2733`；oracle_best_future_in_vehicle_pool `0.3678`。
- 全体结果：all-delay 全体样本，vehicle_rank1 `0.5814`；style `0.7757`；physio_recent `0.7895`；physio_guarded `0.8034`；style+physio `0.7953`；oracle `0.1846`。
- 相关性：候选池内距离与未来误差的 mean Spearman，bad_top10_v250 中 vehicle `0.152`，style `0.026`，physio_recent `-0.008`，physio_guard `-0.022`，style+physio `-0.002`。当前生理/风格距离没有形成稳定“越近未来越近”的排序信号。
- 重要边界：当前 split 是 subject-disjoint，test subject 为 `cwh/lx/rjy/tyy`，train subject 为 `byx/gf/hzh/jy/xst/yyl/yzy/zt/zx/zxy`；因此本轮不能检验“同一驾驶员个体记忆/个体化模型”，只能检验跨驾驶员状态相似性。
- 决策：v253b 不证明生理没有价值，只证明当前 1Hz 生理统计 + last60_guard3 风格特征在手工距离排序中不可用。下一步应改为可学习的 state-conditioned mixture / uncertainty / mode-prior，或先审计生理特征质量、时间对齐、状态标签可分性。
- 验证：`python -m py_compile` 通过；完整脚本运行完成；`guardrail_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`。

---

# 2026-07-01 v253a state-signal disambiguation audit 已完成（生理/风格可介入，但直接拼接消歧不成立）

- 当前状态：已完成 `v253_state_signal_disambiguation_audit_20260701`。本轮不训练新模型，不改 v250/v251/v252，只检查驾驶风格和生理信号是否能降低 v252 发现的“相似输入后未来分叉”。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v253_state_signal_disambiguation_audit_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\reports\v253_state_signal_disambiguation_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701_pack.zip`
- 旧 style 表匹配审计：`style_feature_candidate_wide.csv` 只有旧样本 270 行，与当前 v252/v250 样本的 `sample_id` 交集 `0`、`event_uid` 交集 `0`、`subject+session+anchor` 匹配行 `0`，因此本轮没有直接复用旧风格表。
- 新驾驶风格特征：从当前 raw vehicle CSV 重新提取 `last60_guard3`，窗口 `[observation_s-63, observation_s-3]`，覆盖 `7002/7002`，`post_observation_any=False`，`overlap_direct_input_any=False`。
- 新生理特征：从 `physio_features_1hz.csv` 提取 `pre5_pre2` 和 `pre2_0`，recording 级覆盖率 `0.919`，`post_observation_any=False`。
- 特征块：vehicle base `268` 维；driving style `127` 维；physio recent/delta `198` 维；physio guarded `170` 维。
- 关键结果：v250 bad_top10 all-delay 的 query-vs-neighbor future RMSE，vehicle-only 为 `1.0627`，style_w0.25 为 `1.0637`，style_w0.50 为 `1.0868`，physio_recent_w0.25 为 `1.0703`，physio_recent_w0.50 为 `1.1311`，physio_guarded_w0.50 为 `1.1323`，style+physio_w0.50 为 `1.1466`。没有任何状态特征组优于 vehicle-only。
- 0ms 局部：style_w0.25 在 bad_top10_v250 0ms 上有极小改善 `-0.0097`，但 all-delay 不改善，且 all/strong/observe_later 多数变差，不能作为路线证据。
- 身份/记录风险：本轮近邻同 subject / 同 recording 比例均为 `0`，说明结果不是被同被试或同记录匹配污染；但也说明当前状态特征没有把 test 样本拉向更可解释的训练近邻。
- 决策：生理和风格可以进入下一阶段，但不建议作为直接拼接输入去训练确定性单曲线模型。更合理用途是作为 `v254` 概率/多模态模型中的条件变量、uncertainty head 输入、mode prior 或校准信号。
- 验证：`python -m py_compile` 通过；完整脚本运行完成；`guardrail_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`。

---

# 2026-07-01 v252 input-similarity future-divergence audit 已完成（相似输入的未来分叉证据成立）

- 当前状态：已完成 `v252_input_similarity_future_divergence_20260701`。本轮是可辨识性审计，不训练新模型，不调整通道，不删除样本，不做 anchor selector / gate / router。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v252_input_similarity_future_divergence_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\reports\v252_input_similarity_future_divergence_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701_pack.zip`
- 方法：固定 `v250_minimal_lateral7` 的 `hist + road + phase` 标准化输入，对每个 test rolling sample 只在同 delay 的 train sample 中找 `K=12` 个近邻，再比较这些近邻的真实未来 steering_delta 是否分叉。
- 总体结果：全 test rolling sample `N=1104`，近邻未来两两 RMSE 均值 `0.707`，query-vs-neighbor 未来 RMSE 均值 `0.686`；当前 v250 bad_top10 样本 `N=111`，近邻未来两两 RMSE 均值升至 `0.837`，query-vs-neighbor 未来 RMSE 均值升至 `1.063`。
- 原始锚点结果：0ms 全体 `N=184`，近邻未来两两 RMSE 均值 `0.836`；0ms v250 bad_top10 `N=30`，近邻未来两两 RMSE 均值 `0.963`，query-vs-neighbor 未来 RMSE 均值 `1.148`。
- 相关分析：`neighbor_future_to_query_mean_rmse` 与 `tail_rmse_v250` 的 all-delay Spearman 为 `0.495`，0ms Spearman 为 `0.491`；而 `neighbor_input_distance_mean` 与 `tail_rmse_v250` 的 all-delay Spearman 仅 `0.047`，0ms 仅 `0.022`。这说明误差主要不是因为找不到相似输入，而是相似输入的后续真实轨迹本身会分叉。
- 重叠分析：all-delay 高未来分叉 q75 覆盖 v250 bad_top10 的 `44.1%`；0ms 高未来分叉 q75 覆盖 v250 bad_top10 的 `56.7%`，覆盖 v241 bad_top10 的 `62.5%`。因此未来分叉是重要原因，但不是唯一原因。
- 典型 casebook：`rjy_Entity_Recording_2025_09_28_19_51_44_v108_039` 0ms 的 input distance mean `0.278`，但 query-vs-neighbor future RMSE `1.658`；`tyy_Entity_Recording_2025_09_28_14_40_01_v108_012` 0ms 的 v250 tail RMSE `1.843`，近邻未来两两 RMSE `1.380`。图中可见锚点前输入近似，但锚点后真实轨迹呈扇形分叉。
- 决策：v252 支持用户判断“锚点前可用有效信息不多，样本之间很像但后续行为差异大”。下一步不建议继续单条确定性轨迹回归上堆复杂度，应进入 `v253 probabilistic / multimodal prediction design`，把输出从一条曲线改成多模态轨迹、概率和不确定性范围。
- 验证：`python -m py_compile` 通过；完整脚本运行完成；`guardrail_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`。

---

# 2026-07-01 v251 locked robustness audit 已完成（v250_minimal_lateral7 稳健性通过）

- 当前状态：已固定 `v250_minimal_lateral7` 完成 `v251_locked_robustness_v250_20260701`。本轮不重新训练、不改通道、不用 test 选择模型，只做 locked test 稳健性审计。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v251_locked_robustness_v250_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\reports\v251_locked_robustness_v250_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701_pack.zip`
- locked test bucket/delay：all、normal_predictable、observe_later_like、strong_steer 的所有 delay mean tail delta 均小于 0，`all_key_bucket_delay_tail_negative=True`。
- event-level bootstrap CI：all-delay tail delta 95% CI 均排除 0 且为负：all `-0.0696 [-0.0926,-0.0467]`，normal `-0.0673 [-0.0989,-0.0361]`，observe_later `-0.0999 [-0.1608,-0.0386]`，strong `-0.0769 [-0.1151,-0.0387]`，bad_top10 `-0.3036 [-0.3818,-0.2268]`。
- subject-level：4 个 test subject 中，多数 subject/bucket 均改善，subject/bucket win rate `0.9375`。唯一主要边界是 `cwh/strong_steer` all-delay mean tail delta `+0.0047`，属于轻微回退，需要后续 case review 保留。
- worst regressions：最大回退集中在 `tyy_Entity_Recording_2025_09_28_14_40_01_v108_012` 和 `tyy_Entity_Recording_2025_09_28_14_57_17_v108_028` 等少数事件，说明 v250 不是逐样本全胜。
- 决策输出：`locked_robustness_pass=True`，`current_status=pass_locked_robustness`，推荐下一步 `v252_mainline_candidate_pack_or_subject_level_retest`。但 `formal_replacement_allowed=False`，正式替代还需要主线打包、最终一致性审计和必要的 subject-level 风险说明。
- 验证：`python -m py_compile` 通过；完整脚本运行完成；`guardrail_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`。

---

# 2026-06-30 v250 history-channel ablation 已完成（精简通道有效，可作为下一候选）

- 当前状态：已按用户判断“历史长度关系不大，先试精简通道”完成 `v250_history_channel_ablation_20260630`。本轮只裁剪 `X_hist` 的历史车辆通道，历史长度仍为 -3.0s 到 0.0s 共 31 点；道路预瞄、phase/current、point query 和 original_remaining target 全部不变。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v250_history_channel_ablation_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\reports\v250_history_channel_ablation_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630_pack.zip`
- 训练候选：`v250_drop_attitude_noise13`、`v250_lateral_core10`、`v250_minimal_lateral7`。三者均从头训练 v241 的 TCN + multi-head query attention 结构，不加载 v241 checkpoint，不做 selector/gate/response-type route，不删除样本。
- best validation model：`v250_minimal_lateral7`，历史通道仅保留 `steering|speed_kmh|ay|yaw_rate|roll|lane_curvature|lateral_distance`；best_epoch=`19`，best_val_loss=`0.487806`，`accepted_as_channel_candidate=True`。
- validation 判断：`v250_minimal_lateral7` 相对 v241 的 normal max tail delta `-0.0813`，all mean tail delta `-0.1380`，observe_later mean tail delta `-0.1368`，strong mean tail delta `-0.1543`；val bad_top10 RMSE delta `-0.3247`，说明精简通道不是只伤害/平滑，而是在 validation hard case 上也有实质收益。
- locked test 对照：`v250_minimal_lateral7` 在 all、normal_predictable、observe_later_like、strong_steer、reverse_or_multi_correction 的所有 delay 上 tail/sample RMSE 均优于 v241。all 平均 tail delta `-0.0696`；normal `-0.0673`；observe_later_like `-0.0999`；strong_steer `-0.0769`。
- shape 结果：test bad_top10_v241 平均 RMSE delta `-0.2433`，strong_steer mean range ratio 从 v241 基础上提高约 `+0.0716`，slope ratio 提高约 `+0.0440`；observe_later_like RMSE 也明显降低。说明 v250 比 v249 更接近用户想要的 hard sample 修正方向。
- 输入邻域审计边界：精简通道没有消除一对多问题。delay=0 v241 bad_top10 的 reduced-channel 邻域仍 `input_ambiguous_rate=1.0`；最低 neighbor future pairwise RMSE 为 `0.891`，来自 `v250_lateral_core10`。因此 v250 证明“原 18 通道有冗余/噪声”，但不能证明 hard case 已完全可判别。
- 决策输出：`accept_reduced_channel_as_next_candidate=True`，`accepted_model_name=v250_minimal_lateral7`，但 `formal_replacement_allowed=False`。下一步推荐 `v250_review_channel_ablation_or_try_multimodal_if_ambiguity_persists`，优先做 locked robustness / subject-level 稳健性审计。
- 验证：`python -m py_compile` 通过；完整训练运行完成；`guardrail_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`。

---

# 2026-06-30 v249 shape-aware curve model 已完成（诊断成立，但不接受为新候选）

- 当前状态：已按 GPTPro 建议和用户确认，在 v241 基础上完成 `v249_shape_aware_curve_model_20260630`。本轮不继续 v222a gate / 删除样本 / 轻量 residual / hard response type，也不做 anchor selector；使用 validation-only 选择候选，test 只作 locked report。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v249_shape_aware_curve_model_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\reports\v249_shape_aware_curve_model_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630_pack.zip`
- 训练候选：`v249a_shape_loss_only`、`v249b_shape_aux_heads`、`v249c_shape_conditioned_residual`。三者均从 v241 checkpoint 初始化；best diagnostic model 为 `v249c_shape_conditioned_residual`，best_epoch=`22`，best_val_loss=`1.155262`。
- validation 判断：v249c `noharm_vs_v236_pass=True`、`upgrade_vs_v241_pass=True`，但 `shape_gain_pass=False`，因此 `accepted_as_shape_candidate=False`。v249a 未通过 v241-upgrade；v249b 也未通过 shape-gain。
- test 对照：v249c 在 all/600ms tail RMSE 相对 v241 小幅改善 `-0.004199`，all/1000ms 改善 `-0.008003`；normal_predictable 全 delay 改善。但 observe_later_like 在 0/200/400/600/800ms tail 均变差，strong_steer 在 0/200/400/600ms 也变差，说明当前 shape-aware loss 主要让普通样本更平滑，并没有修复强变化样本。
- 形状指标：test all 的 range_ratio 从 v241 的偏大变得更保守，bad_top10_v241 的 mean_range_ratio 仍仅 `0.625`、mean_slope_ratio 仅 `0.535`；strong_steer 的 mean_delta_rmse 为 `+0.008156`，mean_delta_range `-0.056487`，mean_delta_slope `-0.080059`。这说明 v249 没有解决幅值/斜率低估。
- 输入邻域审计：test delay=0 的 v241 bad_top10 共 `19` 个，`19/19` 标为 `input_ambiguous`；这些样本在当前可见输入空间中能找到很近的训练邻居，但邻居未来轨迹彼此差异大，neighbor future pairwise RMSE 均值 `0.985`。这提示剩余 hard case 不是单靠同一输入同一确定性曲线就能稳定解决。
- 决策输出：`accept_shape_model_as_next_candidate=False`；当前最强正式候选仍不应被 v249 替换。推荐下一步 `v249_error_review_or_input_ambiguity_followup`，重点审查输入可见信息是否不足、是否需要多解/分布式预测、或是否需要补充更早可见的上下文特征。
- 验证：`python -m py_compile` 通过；完整脚本运行完成；`guardrail_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`。

---

# 2026-06-30 v248 best-anchor 后残余轨迹形状误差审查已完成（锚点不是主要矛盾）

- 当前状态：已按用户观察“锚点原因不大、性能提升不够”完成 `v248_best_anchor_residual_shape_audit_20260630`。本轮只读取 v247 fine-grid + locked v241 预测，不训练新模型、不训练 selector、不调锚点阈值。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v248_best_anchor_residual_shape_audit_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\reports\v248_best_anchor_residual_shape_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630_pack.zip`
- 关键结果：test/all 0ms 平均 RMSE `0.475`，best-anchor 后 `0.253`；test/bad_top10 0ms `1.198`，best-anchor 后 `0.616`；test/very_bad_top5 0ms `1.382`，best-anchor 后 `0.642`。
- 残余错误：test/bad_top10 中 best-anchor 后仍有 `47.4%` 高于 `0.65`；still_bad 组平均 best RMSE `0.828`，range_ratio `0.466`，excursion_ratio `0.438`，slope_ratio `0.405`，说明模型主要把真实的大幅度/快变化轨迹预测成了平滑小幅曲线。
- 判断：v247 的 best anchor label 有上限收益，但 v248 证明剩余主要问题已经不是锚点，而是 trajectory shape modeling。下一步不应优先做 sequential selector，而应围绕完整曲线形状、峰值幅值、斜率/回正速度和反打转折建模。
- 验证：`python -m py_compile` 通过；完整脚本运行完成；`guardrail_check.pass=True`；ZIP `testzip=None`。

---

# 2026-06-26 v242 联合曲线解码模型已训练完成（诊断有效，但不替代 v241）

- 当前状态：按用户要求“继续开始下一步训练模型”，已完成 `v242_joint_curve_decoder_20260626`。本轮把 v241 的逐 future point 查询改成 sample-level 联合曲线解码：一次输出 21 个未来点，并加入轻量曲线差分约束。仍保留 `original_remaining` masked target；不创建 gate/router/selector，不删除样本，不做 response-type hard routing，不用 test 选模型，也不改变 formal headline。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v242_joint_curve_decoder_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\reports\v242_joint_curve_decoder_cn.md`
- 训练结果：使用 `cuda` 训练 `v242_joint_curve_h64_smooth002` 和 `v242_joint_curve_h96_smooth005`。best diagnostic model 为 `v242_joint_curve_h96_smooth005`，best_epoch=`36`，best_val_loss=`0.657444`，validation_selection_score=`1.040291`。
- validation 判断：v242 相对 v236 仍然通过 no-harm，`noharm_vs_v236_pass=True`；但相对 v241 没有通过 upgrade，`upgrade_vs_v241_pass=False`，因此 `accepted_as_next_candidate=False`。关键差异：normal max tail delta vs v241 `+0.039176`，strong 400/1000ms mean tail delta vs v241 `+0.014069`。
- test 对照：v242 相对 v236 仍然全面改善，但相对 v241 多数 delay 变差。strong_steer 相对 v241 的 tail delta 为 `+0.035465/+0.047760/+0.040468/+0.017999/+0.013149/-0.014600`；normal_predictable 相对 v241 全部 delay 都是正 delta。
- 逐样本边界：test all 中 v242 相对 v241 有 `588/1104` 条 tail 回退，mean delta `+0.022558`；strong_400_1000 中有 `80/160` 条 tail 回退，mean delta `+0.012934`。这说明联合曲线解码能学到合理曲线形态，但当前训练方式没有超过 v241。
- 决策输出：`accept_joint_curve_model_as_next_candidate=False`；当前最强候选继续保留 `v241_tcn_mha_h96`；v242 只作为诊断训练结果。推荐下一步 `v243_manual_review_or_loss_redesign_for_sample_regressions`，不要基于 test 继续反调。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；必需文件 `missing=[]`；`guardrail_check.pass=True`；`leakage_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`，共 `24` 个条目；对照图 `3` 张。
---

# 2026-06-26 v241 更强时序模型受控实验已完成（stronger candidate 通过 validation，可进入 locked audit）

- 当前状态：按用户要求“接下来试一下更强的模型”，已完成 `v241_stronger_temporal_model_20260626`。本轮保留 v238/v239 的 `original_remaining` masked point-level target，把 v239 轻量 attention 升级为 temporal convolution + multi-head query attention；不创建 gate/router/selector，不删除样本，不做 response-type hard routing，不用 test 选模型，也不改变 formal headline。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v241_stronger_temporal_model_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\reports\v241_stronger_temporal_model_cn.md`
- 训练结果：使用 `cuda` 训练 `v241_tcn_mha_h64` 和 `v241_tcn_mha_h96`。best diagnostic model 为 `v241_tcn_mha_h96`，best_epoch=`26`，best_val_loss=`0.634865`，validation_selection_score=`0.872780`，`accepted_as_stronger_candidate=True`。
- validation 判断：`v241_tcn_mha_h96` 同时通过 v236 no-harm 和 v239 upgrade 检查。相对 v236，normal max sample delta `-0.143055`，normal max tail delta `-0.164108`，observe_later 0-800ms mean tail delta `-0.263814`，strong 0-600ms mean tail delta `-0.221641`；相对 v239，normal max tail delta `-0.023461`，observe_later 0-800ms mean tail delta `-0.124918`，strong 400/1000ms mean tail delta `-0.187700`。
- test original_remaining 对照：`observe_later_like`、`normal_predictable`、`strong_steer` 在 0/200/400/600/800/1000ms 的 tail RMSE 均优于 v239。strong_steer 相对 v239 的 tail delta 分别为 `-0.228374/-0.257514/-0.231814/-0.129609/-0.098782/-0.088476`。
- 逐样本边界：虽然 bucket 均值全面改善，但不是每个样本都变好。test 全部样本中相对 v239 有 `368/1104` 条 tail 回退，mean delta `-0.128199`，max delta `+1.076289`；strong_400_1000 中有 `47/160` 条 tail 回退，mean delta `-0.160145`，max delta `+0.538267`。因此 v241 可以进入 locked audit，但仍不能直接写成 formal replacement。
- 决策输出：`accept_stronger_model_as_next_candidate=True`，`accepted_model_name=v241_tcn_mha_h96`，`formal_replacement_allowed=False`，推荐下一步 `v242_locked_test_report_for_stronger_temporal_candidate`。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；必需文件 `missing=[]`；`guardrail_check.pass=True`；`leakage_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`，共 `24` 个条目；对照图 `3` 张。
---

# 2026-06-26 v240 locked attention audit 已完成（observe/normal 通过，strong 例外需人工复核）

- 当前状态：按用户要求“开始审查”，已完成 `v240_locked_attention_audit_20260626`。本轮只读取 v239 locked attention 的预测、模型权重和 v238/v236 对照，不训练、不调配置、不用 test 选模型、不创建 gate/router/selector、不做 response-type hard routing，也不改变 formal headline。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v240_locked_attention_audit_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\reports\v240_locked_attention_audit_cn.md`
- 核心判断：`attention_candidate_survives_locked_audit=True`，说明 v239 attention 候选在锁定审查下仍然值得保留；但 `formal_replacement_allowed=False`，因为 strong_steer 的 400ms/1000ms 例外仍需要人工看 casebook。
- locked test 汇总：all mean tail delta `-0.059007`，max tail delta `-0.016376`；`observe_later_like` mean tail delta `-0.142789`，max tail delta `-0.089425`，all-delay tail no-harm=True；`normal_predictable` mean tail delta `-0.069787`，max tail delta `-0.031002`，all-delay tail no-harm=True。
- strong 边界：`strong_steer` mean tail delta `-0.036692`，但 max tail delta `+0.048013`，all-delay tail no-harm=False；`strong_400_1000_positive_regression_cases` 共 `82` 条，mean tail delta `+0.279932`，max tail delta `+1.648318`。
- attention casebook：已生成 `21` 张代表性图，包含真实曲线、v236、v238、v239 attention 预测，以及 history attention / road attention 权重。代表样本平均 history last-1s attention mass `0.544`，road 0-1.2s attention mass `0.607`，说明模型确实主要看近端历史和近端道路预瞄。
- 决策输出：v239 attention 可以作为下一阶段候选，但不能直接写成正式替代；下一步建议 `v241_attention_case_manual_review_and_robustness_ci`，先人工审 strong 例外和 worst regression，再决定是否做稳健性置信区间/重采样。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；必需文件 `missing=[]`；`guardrail_check.pass=True`；`leakage_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`，共 `39` 个条目；attention casebook 图数 `21`。
---

# 2026-06-26 v239 轻量 temporal attention + no-harm 约束实验已完成（attention 可作为下一候选，仍非 formal headline）

- 当前状态：按用户要求“先这样试一下注意力机制效果”，已完成 `v239_light_attention_noharm_20260626`。本轮保留 v238 的 `original_remaining` masked point-level target，在同一个模型内部增加轻量 temporal attention，不创建 gate/router/selector，不做响应类型硬分类，不上完整 Transformer。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v239_light_attention_noharm_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\reports\v239_light_attention_noharm_cn.md`
- 模型结构：历史序列 `31 x 18` 与道路预瞄 `21 x 2` 分别进入轻量 attention；phase + future point 生成 query；attention 输出 history context 和 road context 后接小 MLP head，逐 future point 输出 steering_delta。
- 训练设备与候选：使用 `cuda`；训练 `v239_light_attention_h32` 和 `v239_light_attention_h48` 两个小配置。best diagnostic model 为 `v239_light_attention_h32`，best_epoch=`14`，validation score `1.077325`，validation no-harm pass=True。
- validation no-harm：两个 attention 候选均通过。`h32` 在 validation 上 normal max sample delta `-0.066203`、normal max tail delta `-0.074856`、all max sample delta `-0.029533`、observe_later 0-800ms mean tail delta `-0.138896`、strong 0-600ms mean tail delta `-0.041846`，均满足约束。
- test original_remaining 对照：`observe_later_like` tail 相对 v236 在 0/200/400/600/800/1000ms 分别为 `-0.153984/-0.089425/-0.161281/-0.196791/-0.133886/-0.121369`；`normal_predictable` tail 分别为 `-0.173278/-0.071748/-0.043820/-0.036963/-0.061909/-0.031002`，说明本轮没有复现 v238 MLP 伤 normal 的问题。
- 边界风险：`strong_steer` 在 400ms tail 轻微变差 `+0.009627`，1000ms tail 变差 `+0.048014`；因此 v239 仍是下一候选/诊断原型，不是 formal replacement。formal headline 继续锁定 v225/v226。
- 决策输出：`accept_attention_as_candidate=True`，推荐下一步 `v240_locked_test_report_for_attention_candidate`，即在不扩大到 router/gate/full Transformer 的前提下，对 v239 attention candidate 做锁定报告、可视化与更细分样本审查。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；必需文件 `missing=[]`；`guardrail_check.pass=True`；`leakage_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`，共 `20` 个条目。
---

# 2026-06-26 v238 任务构造与小型 rolling 模型重搭已完成（接受任务构造，不接受当前模型作正式替代）

- 当前状态：按用户要求“仔细审查吸取经验，重新搭建任务构造方式和模型框架”，已完成 `v238_task_model_redesign_20260626`。本轮不继续 v222a gate / 删除样本 / light residual 路线；不重新扫描原始车辆 CSV；复用 v236 rolling 输入，重新定义训练目标并训练受控小模型。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v238_task_model_redesign_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\reports\v238_task_model_redesign_cn.md`
- 核心任务构造：主任务从 v236 的 `receding_2s` 改为 `original_remaining`。每个 delay 只监督 observation time 到 original anchor+2s 的重叠部分；训练形式改成 point-level masked target，让 1000ms 之后落入新行为阶段的点不进入主 loss。
- 样本与窗口：复用 v236 的 `7002` 个 rolling 样本、`1167` 个唯一事件。original_remaining 有效点随 delay 从 `21/19/17/15/13/11` 递减，尾段点每个 delay 均为 `11` 个。
- 模型框架：validation-only 在 point Ridge 与小 MLP 中选择；selected model 为 `v238_point_mlp_96x48_alpha1e-4`，validation score `1.290127`。本轮没有创建 gate/router/selector，没有删除 observe_later_like，没有改变 formal headline，没有用 test 选择模型配置。
- test original_remaining 主要收益：`observe_later_like` tail 相对 v236 在 0/200/400/600/800ms 分别改善 `-0.202554/-0.169130/-0.261414/-0.106553/-0.107993`；`strong_steer` tail 在 0/200/400/600ms 分别改善 `-0.250813/-0.137541/-0.044493/-0.007093`。
- 主要阻塞：`normal_predictable` no-harm 未通过，各 delay sample RMSE 均比 v236 更差；`observe_later_like` 的 1000ms tail 变差 `+0.309477`，`strong_steer` 的 800/1000ms 也变差。因此 v238 只能作为任务构造和小模型原型，不能作为正式替代模型或 formal headline。
- 决策输出：接受 `original_remaining masked point-level target` 作为下一阶段任务构造；不接受当前 selected MLP 作为正式替代；推荐下一步为 `v239_noharm_constrained_original_remaining_model`，重点是 validation no-harm 约束和 late-delay policy，而不是扩大模型。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；必需文件 `missing=[]`；`guardrail_check.pass=True`；`leakage_check.pass=True`；同一 event_uid 跨 split 数 `0`；ZIP `testzip=None`，共 `20` 个条目。
---

# 2026-06-25 v237 rolling target / phase audit 已完成（audit-only，新阶段决策点）

- 当前状态：按 GPTPro 指令完成 `v237_rolling_target_phase_audit_20260624`。本轮只读取 v236 rolling 输出、v225/v226 formal 参考和 v229 taxonomy，不训练模型、不生成新预测、不搜索 alpha/threshold/tau、不创建 gate/router/selector，也不改变 formal headline。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v237_rolling_target_phase_audit_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\reports\v237_rolling_target_phase_audit_cn.md`
- 核心审计结论：target sanity 全部 pass。v236 的 `Y_future` 与 `pred_future` 均在 `steering_delta_from_observation` 空间，`original_remaining` 评估只是裁剪 delay 后与原始 `anchor+2s` 的重叠部分，没有重建新 target。
- receding vs original_remaining：`observe_later_like` 在 receding_2s 下 1000ms 明显变差（test tail RMSE `4.074430`），但在 original_remaining 下 1000ms 只评估原始剩余窗口，tail RMSE 为 `1.199416`；200ms remaining tail RMSE 从 0ms 的 `1.178022` 降到 `1.070851`。这说明 v236 的一部分失败确实来自 target horizon 后移后跨入新行为阶段，而不是“晚观察完全无用”。
- strong_steer 结论保持：test receding tail RMSE 从 0ms `1.018893` 降到 800ms `0.819825`、1000ms `0.814272`；original_remaining 也随 delay 改善到 1000ms `0.529581`。这继续支持 rolling observation 对强转向/强响应样本有帮助。
- 1000ms failure 审计：`observe_later_like` test 共 `27` 个事件，`is_new_phase_after_1000ms` 规则命中率 `0.888889`。反打、zero-cross、多次修正和 late peak 是拉坏 receding horizon 的主要混合因素。
- Ridge underfit 审计：v236 0ms all test RMSE `1.220571`，旧 formal reference RMSE `0.468061`，gap `+0.752510`；`observe_later_like` peak_shrinkage_ratio `0.341520`，`strong_steer` peak_shrinkage_ratio `0.418264`。alpha validation-only 选择 `1000`，且位于最大 alpha 边界，说明 v236 joint Ridge 是可行性小基线而不是足够强的正式模型。
- 决策输出：`v237_next_model_decision.csv` 给出 `v238_allowed=True`，推荐下一步候选为 `v238_small_rolling_model`。但 v237 本身按指令完成后停止，尚未执行 v238；若继续，仍必须禁止 v222a gate/router/selector，按 delay 和 bucket 分开报告，不用 test 选择配置，不改 formal headline。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；必需文件 `missing=[]`；guardrail/leakage/consistency 均 pass；forbidden hits `[]`；同一 event_uid 跨 split 数为 `0`；v236 receding 指标复现最大差异 `2.3841858e-07 < 1e-5`；ZIP `testzip=None`，共 `29` 个条目。
---

# 2026-06-24 连续车辆源数据审计已完成（独立于样本集）

- 当前状态：按用户要求“不看样本集，直接审计那些车辆数据”，已完成 `vehicle_source_audit_20260624`。本轮只读扫描连续车辆 CSV，不使用训练样本、标签或模型预测，不修改原始数据。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\scripts\vehicle_source_audit_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\vehicle_source_audit_20260624_cn.md`
- 审计范围：候选文件 `358` 个，纳入连续车辆审计 `182` 个文件，覆盖 `18` 名被试、`91` 个记录键、约 `25.31` 小时；其中主 `vehicle_aligned_cleaned` 层 `91` 个文件，补充 `(2)_vehicle_fixed_200Hz` 车辆层 `91` 个文件。
- 主要发现：`车辆清理后` 目录存在明显命名混杂，另检出 `82` 个 PhysioLAB 生理文件和 `85` 个 EEG/加速度文件也带 vehicle 文件名/目录；后续不能只按文件名读取车辆源，必须按字段白名单判定。
- 质量判断：主 `vehicle_aligned_cleaned` 层时间轴稳定，median dt 为 `5ms`，关键车辆字段高缺失文件 `0` 个；但主层有 `32/91` 个文件 `ref_nn_ok_rate<95%`，road/curve 分层和横向偏移判断要优先复核这些记录。
- 补充层风险：补充 `(2)_vehicle_fixed_200Hz` 层有 `83/91` 个文件关键字段缺失率超过 `20%`，`45` 个文件 nominal Hz 超出 `150-250`，`42` 个文件 median dt 不接近 `5ms`。它不能和主车辆层直接混用。
- lineage 结论：`91` 个记录簇都能连接主层与补充层，但 `47` 个簇行数不一致、`83` 个簇规范车辆信号抽样哈希不一致；后续样本/模型必须固定唯一源层，并记录每个样本来自哪个源层。
- 被试分布：总时长最高为 `zx=167.2min`、`hzh=122.8min`、`gf=100.6min`；最低为 `zt=20.8min`、`xst=24.8min`、`lx=42.5min`。这支持继续使用 subject/session-level split，不能回到随机样本切分。
- 验证摘要：`py_compile` 通过；脚本完整运行处理 `182/182` 个纳入文件；`run_manifest.json` 中 `errors=[]`；3 张图像均已生成且非空。

---

# 2026-06-24 v236 rolling/reanchor 数据集与小基线已完成（最新）

- 当前状态：按 GPTPro 指令开启新阶段 `v236_rolling_reanchor_dataset_and_baseline_20260624`，停止 v222a gate / 删除样本 / light residual 路线。本轮用 loose 主池 `1167` 个唯一事件生成 rolling observation 数据集，每个事件保留 `0/200/400/600/800/1000ms` 六个观测时刻，共 `7002` 个 rolling 样本。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v236_rolling_reanchor_dataset_and_baseline_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\reports\v236_rolling_reanchor_baseline_cn.md`
- 关键表：`tables/v236_rolling_sample_manifest.csv`、`tables/v236_delay_sample_counts.csv`、`tables/v236_train_val_test_event_split_check.csv`、`tables/v236_baseline_metrics_by_delay.csv`、`tables/v236_baseline_metrics_by_delay_and_bucket.csv`、`tables/v236_observe_later_improvement_curve.csv`、`tables/v236_strong_event_improvement_curve.csv`、`tables/v236_normal_sample_noharm_check.csv`、`tables/v236_metric_vs_old_0ms_formal_reference.csv`。
- 数据构造：输入为 observation time 前 `3s` 历史人车序列、observation time 后 `2s` 道路预瞄和当前可观测 phase features；输出为未来 `2s x 6` joint targets：steering delta、steering rate、roll delta、roll rate、ay、yaw_rate。
- 模型：本轮先用 joint Ridge 小基线验证 rolling 任务是否成立；alpha 只按 validation 选择，selected alpha=`1000`；没有创建 gate/router/selector，没有删除 observe_later_like，没有改 formal headline。
- 关键结果：`strong_steer` test tail mean 从 `0ms=0.961224` 降到 `800ms=0.695632`、`1000ms=0.702697`，strong-under rate 从 `0.7625` 降到 `0.2125` 左右，说明强反应类后续观察明显有帮助。
- 关键风险：`observe_later_like` 在该 joint Ridge 小基线上没有满足“later observation 明显改善”的成功条件；`200ms` 略好于 `0ms`（tail mean `1.100397 -> 1.060875`），但 `400/600/800/1000ms` 没有持续下降，`1000ms` 明显变差。因此不能直接进入更大模型；需要先审查 target 定义、该桶样本组成、线性基线表达能力和极端 outlier。
- 0ms 对旧 formal：v236 0ms baseline 比旧 formal 参考更差，all test sample RMSE `0.641212` vs old `0.468061`；这说明 v236 当前是任务可行性小基线，不是 formal 0ms 替代模型。
- 验证摘要：`py_compile` 通过；完整运行通过；必需文件 `missing=[]`；guardrail `pass`；leakage `pass`；同一 event_uid 跨 split 数 `0`；ZIP `testzip bad=None`、文件数 `22`。

---

# 2026-06-24 v235 删除 observe_later_like 样本后的受控重训实验已完成（最新）

- 当前状态：按用户要求“尝试去掉预测效果很差的样本，然后再训练跑一遍模型”，已完成 `v235_remove_observe_later_retrain_20260624`。本轮把 v234 全量扫描中 `observe_later_like=True` 的事件从 train/val/test 全部剔除，再重训 v222a light residual/融合层，并新增同 feature schema 的 absolute Ridge 对照。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v235_remove_observe_later_retrain_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\reports\v235_remove_observe_later_retrain_cn.md`
- 主对照表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_comparison_summary.csv`
- 删除规模：loose pool 删除 `121/1167` 个样本，其中 train/val/test 为 `58/36/27`；strict pool 按相同 event_uid 交集删除 `117/963` 个样本，其中 train/val/test 为 `57/33/27`。
- 关键结果：旧 v222a selected full test RMSE 为 loose `0.555940`、strict `0.571966`；只删除这类样本后，在同一保留 test 子集上旧模型 RMSE 降到 loose `0.482685`、strict `0.506547`；删除后重训进一步到 loose `0.474318`、strict `0.504151`。
- 解释：收益主要来自“保留测试集变容易”，重训本身相对旧模型同一过滤子集的额外收益较小（loose `-0.00837` RMSE，strict `-0.00240` RMSE）。被删除 test 样本仍很难，重训模型在 removed holdout 上 RMSE 为 loose `0.868780`、strict `0.845273`。
- 方法边界：本轮不是 v216/v218 底座候选网络的端到端重训；底座候选曲线仍来自既有 v222a cache。v235 只能作为“直接删除差样本”的诊断对照，不能直接替代短观察层/重锚定路线，也不应改 formal headline。
- 验证摘要：`py_compile` 通过；脚本完整运行通过；feature schema guard `pass`；selection 使用 filtered validation only；图像非空；ZIP `testzip bad=None`、文件数 `24`。

---

# 2026-06-24 v234 短观察后预测评估层构建包已完成（最新）

- 当前状态：按用户要求继续推进 v233 的 `observe_later_review` 路线，已完成 `v234_short_observation_prediction_layer_20260624`。本轮把旧锚点前证据弱但后续变化大的样本，单独构造成“短观察后预测”评估层，与纯提前预测分开。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v234_short_observation_prediction_layer_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\reports\v234_short_observation_prediction_layer_cn.md`
- 人工审核模板：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_manual_review_template.csv`
- 关键结果：`10` 个 `observe_later_review` 样本，`5` 个观察层（0.0s 纯提前参考 + 0.5/1.0/1.5/2.0s 短观察候选），`50` 条层分配，`1050` 个真实目标曲线点，`10` 张样本图。
- 默认 0.5s 层观察：多数样本 `remaining_peak_fraction` 仍然较高，说明后移 0.5s 后仍有真实未来轨迹要预测；该层不是简单“看见答案后补全”。
- 当前边界：v234 只构建评估层，不训练模型、不修改标签、不改 formal headline；旧 formal prediction 从旧锚点出发，不能硬评到新观察层。
- 下一步：人工审核 `v234_short_observation_manual_review_template.csv`，为每个样本选择 `0.5/1.0/1.5/2.0s` 或拒绝短观察层；确认后才能进入 v235 的短观察层数据清单或模型评估。
- 验证摘要：`py_compile` 通过；脚本完整运行通过；`errors=[]`；图像非空；ZIP `testzip bad=None`。

---

# 2026-06-24 v233 自适应锚点 / 观察时长策略审核包已完成（最新）

- 当前状态：根据用户反馈“有些事件前几秒确实看不出区别，变化很大的样本可否区别对待、后移锚点、多放一点时间观看”，已完成 `v233_adaptive_anchor_observation_policy_20260624`。该包把样本拆成提前重锚定、后移观察点、标准锚点和模糊类，避免把任务可观测性问题错误归为模型失败。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v233_adaptive_anchor_observation_policy_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\reports\v233_adaptive_anchor_observation_policy_cn.md`
- 人工审核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\tables\v233_anchor_observation_policy_review_table.csv`
- 关键结果：在 `29` 个样本中，`10` 个样本属于 `observe_later_review`，即旧锚点前证据弱但后续变化大；`5` 个属于 `reanchor_earlier_review`；`6` 个属于 `reanchor_earlier_or_ambiguous_review`；`1` 个大变化但模糊；`7` 个标准锚点。
- 解释：`observe_later_review` 不是为了刷分，而是单独建立“短观察后预测”层级；它与“提前重锚定”不同，前者是任务可观测性问题，后者是事件起点/锚点定义问题。
- 方法边界：不训练模型、不修改标签、不改 formal headline；不回到硬响应类型分类；不把简单多候选轨迹输出作为主线。
- 验证摘要：`py_compile` 通过；脚本完整运行通过；`errors=[]`；图像非空；ZIP `testzip bad=None`。

---

# 2026-06-24 v232 过晚锚点重锚定候选审核包已完成（最新）

- 当前状态：根据用户反馈，当前主线从“继续改输出结构”前移到“先修过晚锚点和目标窗口”。已完成 `v232_late_anchor_reanchor_candidates_20260624`，对 v230 casebook + v231 六样本的 `29` 个唯一样本读取原始车辆信号，生成重锚定候选和人工审核表。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v232_late_anchor_reanchor_candidates_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\reports\v232_late_anchor_reanchor_candidates_cn.md`
- 人工审核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_reanchor_candidate_review_table.csv`
- 关键结果：`29` 个样本全部打分，输出 `11` 个 P0/P1/P2 重锚定候选。`rjy...010` 为 P0 人工确认晚锚点，算法候选为 `143.100s -> 138.950s`，提前 `4.15s`；P1 另有 `rjy...041`、`rjy...040`、`rjy...032`、`tyy...033`。
- 当前边界：候选新锚点不能自动生效；必须先在人工审核表中填写 `human_decision`、`human_corrected_anchor_s`、`human_use_for_training`。只有人工确认后的样本才允许进入 label window 重建。
- 方法边界：不重启硬响应类型分类；不把简单多候选轨迹输出作为主线；下一步优先是人工确认重锚定候选，再做目标窗口重建。
- 验证摘要：`py_compile` 通过；脚本完整运行通过；`errors=[]`；图像非空检查通过；ZIP `testzip bad=None`。

---

# 2026-06-24 v231 六个最差样本锚点上下文人工审核包已完成（最新）

- 当前状态：按用户要求，从原始车辆 CSV 直接调取 6 个预测效果最差/最有代表性差样本的事件锚点、绝对锚点时间、锚点前后方向盘和车辆状态信号。该包用于判断差样本是否来自锚点/窗口问题，还是模型没有学到行为响应形态；定位目标是方法提升和行为预测，不是写失败机制论文。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\reports\v231_worst_case_anchor_context_cn.md`
- 关键表：`tables\v231_anchor_metadata.csv`、`tables\v231_window_summary.csv`、`tables\v231_anchor_key_points.csv`、`tables\v231_anchor_window_sparse_8s.csv`、`tables\v231_anchor_window_dense_pm3s.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures`
- 数据处理说明：原始车辆 CSV 同一时间戳行上存在不同信号空值，因此关键点表和稀疏窗口按“目标时刻最近非空信号值”输出，并保留 `字段名__time_error_ms`；密集窗口保留原始 200Hz 行，不做填补。
- 初步判断：`rjy...010` 和 `rjy...041` 优先查锚点是否落在事件中段/窗口是否覆盖不足；`rjy...023`、`tyy...026`、`rjy...031` 是反转/多次修正型困难样本；`cwh...017` 未见明显锚点错位，更像幅值/形态预测问题。
- 用户反馈修正：`rjy...010` 已人工确认锚点晚了，后续应进入锚点修正/事件起点定义，不应作为模型形态失败样本；同时撤回“先硬判断响应类型再预测轨迹”作为主线，因为此前已经尝试过，且存在响应类型判断错误导致后续轨迹整体错误的结构性风险。
- 修正说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\reports\v231_user_feedback_method_correction_cn.md`
- 用户第二轮反馈：过晚锚点需要进一步重锚定，不能只标记；一次性输出多个候选轨迹此前也已经尝试过，效果不好，即使 best candidate 仍有偏差，因此简单多候选轨迹输出也不作为下一步主线。
- 下一步：围绕过晚锚点做重锚定准备，生成候选新锚点、移动秒数、证据字段和人工确认字段；只有在锚点确认无误后，才把样本纳入模型方法提升。模型方向优先考虑目标窗口重建、偏差校正、连续相位/延迟建模和对齐鲁棒损失，而不是硬响应类型分类或简单多候选轨迹输出。

---

# 2026-06-23 v230 失败案例人工复核 / 论文案例证据包已完成（最新）

- 当前状态：GPTPro 已审阅并接受 v229，明确要求模型工作继续停止；唯一允许的下一步是 audit-only + paper-case packaging。已完成 `v230_failure_case_manual_review_casebook_20260623`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v230_failure_case_manual_review_casebook_20260623.py`
- 主报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\reports\v230_failure_case_manual_review_casebook_cn.md`
- 导师讨论笔记：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\reports\v230_advisor_discussion_notes_cn.md`
- 论文失败案例小节草稿：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\reports\v230_paper_failure_case_section_draft_cn.md`
- 人工复核模板：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\tables\v230_manual_review_template.csv`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\v230_failure_case_manual_review_casebook_pack.zip`
- case 选择摘要：共 `46` 个 case，每个 pool `23` 个；每池满足 `强反应低估=5`、`极端峰值失败=4`、`强响应幅值/尾段=5`、`反转或多次修正=3`、`过零/换向边界=3`、`普通曲线可控=3`。
- 图复制摘要：复制既有图 `85` 张；case 图缺失 `13` 个已显式记录为 `figure_missing`；没有生成新预测图。
- 验证摘要：`py_compile` 通过；脚本完整运行通过；ZIP `bad_file=None`、文件数 `103`；必需文件缺失 `[]`；guardrail `pass=True`；consistency `pass=True`；forbidden hits `[]`；人工复核字段全部留空。
- 当前边界：v230 完成后自动停止，不启动新模型。下一步只能人工阅读 casebook、填写 `v230_manual_review_template.csv`，再用 `v230_paper_failure_case_section_draft_cn.md` 写/改论文失败案例小节。
---

# 2026-06-23 v229 两个月路线经验复盘与失败分类包已完成（最新）

- 当前状态：根据用户要求，先暂停继续训练/调参，转为分析过去两个月路线经验。已生成 `v229_two_month_lessons_failure_taxonomy_20260623` 复盘包，范围只读 v220/v225/v228 已验证产物，不训练模型、不生成新预测、不重选 formal headline。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v229_two_month_lessons_failure_taxonomy_20260623.py`
- 主报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\reports\v229_two_month_lessons_failure_taxonomy_cn.md`
- GPTPro 中文提问稿：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\reports\v229_gptpro_next_prompt_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\v229_two_month_lessons_failure_taxonomy_pack.zip`
- 核心判断：两个月尝试不是“模型还没堆够”，而是反复证明同一个瓶颈：方向和普通响应较稳，强反应幅值、极端峰值、尾段、反转/多次修正仍是主失败区；候选池经常有更好上限，但 current-window deployable selector 不稳。
- 失败桶摘要：v229 已把 test split 失败样本粗分为 `极端峰值低估`、`强反应低估`、`极端峰值/尾段难例`、`强响应幅值/尾段`、`反转或多次修正`、`过零/换向边界`、`普通曲线可控` 等桶，并导出高尾失败案例表。
- 验证摘要：`py_compile` 通过；脚本完整运行通过；ZIP `bad_file=None`、文件数 `15`；必需文件缺失 `[]`；guardrail `pass=True`，确认未训练、未生成新预测、未新建 gate/router、未新建 tau/threshold、未解锁 v222b/v223、未基于 test 重新选择。
- 当前下一步：如果继续问 GPTPro，应发送 v229 中文复盘 prompt，让 GPTPro 先判断“是否进入写作整理 / 是否只允许失败样本 taxonomy 与人工复核 / 是否继续禁止 v222b/v223、新 gate/router、新 tau/threshold 和 test-based retuning”。在 GPTPro 明确给出 bounded 指令前，不启动新模型或新路由器。
---

# 2026-06-22 v226 formal robustness / CI audit 已完成（最新）

- 当前状态：v226 audit-only robustness / CI pack 已完成并通过独立验收；下一步是把 v226 pack 和验证摘要报告给 GPTPro 获取下一轮 bounded 指令。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\v226_formal_robustness_ci_audit_pack.zip`
- 验证摘要：locked metric reproduction / leakage guard / forbidden scan / table alignment / ZIP / required files / figure count 全部 pass；formal lock 仍为 `loose_main_pool=avg_joint_focus`、`strict_main_pool=peak_floor_090`。
- 边界：仍不允许本地自行进入 v222b/v223、新 tau、新 gate/router 或 test-based retuning。

---

# 2026-06-23 heartbeat：GPTPro 回报通道仍阻塞（最新）

- 当前状态：本轮 heartbeat 继续尝试获取 GPTPro 新指令，但没有拿到有效正文。桌面端 ChatGPT 当前仍显示 v226/v227 handoff prompt 和“已停止思考”，没有六项 bounded 回复；Chrome bridge 再次因无法验证 Pro/进阶模式而拒绝发送。
- 新增阻塞归档：
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v227_heartbeat_gptpro_response_blocked.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v227_heartbeat_gptpro_decision_blocked.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v227_heartbeat_gptpro_action_items_blocked.md`
- v227 ZIP 复核：`zip_exists=True`，`bad_file=None`，`file_count=35`。
- 边界不变：formal headline 仍为 `loose_main_pool=avg_joint_focus`、`strict_main_pool=peak_floor_090`；不进入 v222b/v223、新 tau/threshold、新 gate/router/selector、新模型训练、formal leaderboard/headline 改动或 test-based retuning。
- 下一步：等 GPTPro/ChatGPT Pro 通道恢复后，重新发送 `v227_next_gptpro_prompt_ascii.md`，拿到有效回复后先归档 raw response / decision / action items，再只执行一个通过本地 guardrail 的 bounded 指令。

---

# 2026-06-22 v225 formal route reconstruction evidence pack 已完成

- 状态：按 GPTPro v225 指令完成 `formal route reconstruction evidence pack`。本轮只固化 formal baseline 重建证据，不训练新模型，不调 threshold/tau，不创建 router/gate，不运行 v222b/v223，也不改变 formal headline。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v225_formal_route_reconstruction_evidence_pack_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622`
- GPTPro v225 指令归档：
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v225_evidence_pack_gptpro_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v225_evidence_pack_gptpro_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v225_evidence_pack_gptpro_action_items.md`
- formal model lock：
  - `loose_main_pool = avg_joint_focus`
  - `strict_main_pool = peak_floor_090`
  - `v222a_bounded_residual`、`v222a_noharm_gate`、`oracle_safe_gate`、`ridge_residual_peakfloor` 等只在 diagnostic-only 汇总中出现，不进入 formal lock、formal leaderboard 或 formal usage。
- 核心产物：
  - `tables/formal_model_lock.csv`
  - `tables/formal_reconstruction_metrics_overall.csv`
  - `tables/formal_reconstruction_metrics_by_pool.csv`
  - `tables/formal_reconstruction_metrics_by_bucket.csv`
  - `tables/formal_reconstruction_metrics_by_route_event.csv`
  - `tables/per_sample_formal_reconstruction_eval.csv`
  - `tables/formal_failure_case_index.csv`
  - `tables/diagnostic_only_v222a_closeout_summary.csv`
  - `tables/excluded_diagnostic_models_audit.csv`
  - `reports/v225_formal_route_reconstruction_evidence_cn.md`
  - `logs/run_manifest.json`
  - `logs/leakage_guard_report.json`
  - `logs/forbidden_scan_report.json`
  - `logs/metric_reproduction_check.json`
  - `logs/file_inventory.json`
  - `v225_formal_route_reconstruction_evidence_pack.zip`
- locked test 指标复现：
  - `loose_main_pool / avg_joint_focus`：RMSE `0.544884`，tail RMSE `0.629752`；
  - `strict_main_pool / peak_floor_090`：RMSE `0.571770`，tail RMSE `0.658306`；
  - 四个复现误差均小于 `1e-5`。
- 图表证据：
  - `formal_examples` 12 张；
  - `worst_tail_cases` 12 张；
  - `strong_under_cases` 8 张；
  - `baseline_sufficient_cases` 8 张；
  - 每张图标题含 pool、sample_id、formal_model、RMSE、tail RMSE、under flag，已抽检非空。
- 验证：
  - `python -m py_compile` 通过；
  - v225 脚本完整运行通过；
  - ZIP 校验 `bad_file=None`，必需文件无缺失；
  - `metric_reproduction_check.json` pass；
  - `leakage_guard_report.json` 全 pass；
  - `forbidden_scan_report.json` pass，formal 表未检出 `W3_B4_original_soft/oracle/true_label/fallback/v222a_*` 等禁用名；
  - `table_alignment_check.json` pass，无重复 sample_id、无缺失 formal prediction、prediction shape=`N x 21`、horizon=`21`。
- 当前下一步：把 v225 evidence pack、验证结果和执行边界报告给 GPTPro，获取下一轮 bounded 指令。若 GPTPro 未给出明确 stop condition，不本地开启 v222b/v223、新 gate/router、新 tau 或 test-based retuning。

---

# 2026-06-22 v221 统一评估框架已完成

- 状态：已完成 v221 自包含评估脚本、统一逐样本表、formal/diagnostic 整体表、分组表、失败样本清单、中文报告、HTML 入口和 ZIP 校验。
- 本轮目的：按 GPTPro 最新建议先“收敛评估，不训练新模型”，把 v216/v217/v218/v219 已有候选放到同一套 formal leaderboard 与强事件/普通样本/极强峰值分组指标中。
- 输入说明：当前工作区中 `gptpro_answer/2026.6.22回答.txt` 不存在，本轮依据对话中已提供的 GPTPro 指示和本地 v216-v219 产物执行；未调用缺失的旧源码，不重新训练。
- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v221_formal_model_leaderboard_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622`
- 关键结果：
  - 可用主池整体 test RMSE 最低：`avg_joint_focus = 0.5448793008861739`；
  - 可用主池后 1-2s tail RMSE 最低：`joint_equal = 0.6296311379509313`；
  - 可用主池 formal 低估率最低：`peak_floor_090 = 0.1032608695652173`；
  - 严格主池整体和 tail RMSE 最低：`peak_floor_090 = 0.5717751408320051`，tail `0.6583082313285479`；
  - 严格主池低估率最低：`ridge_residual_peakfloor = 0.0919540229885057`；
  - 强事件均值 RMSE 在两个主池里都由 `ridge_residual_peakfloor` 最低，但可用主池低估率仍是 `peak_floor_090` 更低。
- 当前判断：
  - v221 支持 GPTPro 的判断：不要继续直接重训更大的模型，下一步应先做轻量的候选软融合和受限残差；
  - v218 强峰值训练继续只作为诊断对照，不作为新主线；
  - v222a 若启动，应固定候选、validation-only 选择，并重点平衡 `avg_joint_focus/global_blend` 的普通样本稳定性、`peak_floor_090` 的低估控制、`ridge_residual_peakfloor` 的强事件收益。
- 验证：
  - `python -m py_compile` 通过；
  - v221 脚本完整运行通过；
  - ZIP 校验 `bad_file=None`，包含 `13` 个文件；
  - formal overall 表未检出 `W3_B4_original_soft`、`oracle`、`fallback`、`true_label`。

---

# 2026-06-22 v222a closeout + candidate gap audit 已完成

- 状态：按 GPTPro 最新指令完成 `v222a closeout + candidate gap audit`。正式停止 `v222a bounded residual / no-harm gate` formal 主线；本轮没有训练 v222b/v223，没有新增 router，没有重新选择 tau，也没有根据 locked test 反调配置。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_closeout_candidate_gap_audit_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622`
- GPTPro 回复归档：
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v222a_closeout_gptpro_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v222a_closeout_gptpro_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v222a_closeout_gptpro_action_items.md`
- 核心产物：
  - `tables/formal_headline_decision.csv`
  - `tables/v222a_stop_evidence.csv`
  - `tables/oracle_vs_learned_gap.csv`
  - `tables/candidate_gap_audit.csv`
  - `tables/per_sample_failure_taxonomy.csv`
  - `tables/bucket_failure_summary.csv`
  - `tables/future_route_decision.csv`
  - `reports/v222a_closeout_candidate_gap_audit_cn.md`
  - `logs/closeout_manifest.json`
  - `v222a_closeout_candidate_gap_audit_pack.zip`
- formal headline 已锁定：
  - `loose_main_pool = avg_joint_focus`
  - `strict_main_pool = peak_floor_090`
  - `v222a_bounded_residual / v222a_noharm_gate / oracle_safe_gate` 全部为 diagnostic-only，不进入 formal headline。
- v222a 停止证据：
  - `loose_main_pool`：validation pass=True，locked test pass=False；test RMSE delta `+0.010559`，tail delta `+0.027764`，under reduction `+0.043478`。结论是保住 under 改善但伤 RMSE/tail。
  - `strict_main_pool`：validation pass=True，locked test pass=False；test RMSE delta `-0.008975`，tail delta `-0.005264`，under reduction `-0.017241`。结论是守住 RMSE/tail 但 under 变差。
- closeout 诊断：
  - test `selector_failed_rate`：loose `0.407609`，strict `0.413793`，combined `0.410615`。
  - test `candidate_missing_rate`：loose `0.027174`，strict `0.028736`，combined `0.027933`。
  - high-tail 样本 `candidate_missing_rate`：loose `0.119048`，strict `0.135135`，combined `0.126582`。
  - test oracle clear gain rate 在 high-tail 样本中约 `0.911392` combined，说明现有 formal candidate pool 仍常能提供更好候选，主要问题不是候选池整体缺曲线，而是 learned gate/selector 没稳定抓住。
- future route decision：
  - `v222b_allowed=False`：learned gate validation 过但 locked test 失败，更大 neural gate 风险是继续 overfit selector signal。
  - `v223_allowed=False`：high-tail candidate_missing 未超过 50%，不满足新 candidate generator 的解锁条件。
  - 当前下一步不是继续本地调参，而是把 closeout pack 和结论报告给 GPTPro，获取下一轮 bounded 指令。
- 验证：
  - `python -m py_compile` 通过；
  - closeout 脚本完整运行通过；
  - ZIP `v222a_closeout_candidate_gap_audit_pack.zip` 校验 `bad_file=None`，包内 `74` 个文件，必需文件无缺失；
  - `leakage_guard_result.csv` 六项全 pass；
  - formal headline 禁用名检查未检出 `W3_B4_original_soft/oracle_model/true_label/fallback`；
  - case figure 目视抽检通过，共生成 `61` 张 case 图。

---

# 项目状态更新：v0.5 服务器处理后样本重筛 + 被试划分旧流程车辆-only

---

# 项目状态更新：v2.2 epoch 边界精修审计

更新时间：2026-05-26

当前阶段：旧流程样本锚点与 epoch 起止边界精修。本轮不训练模型，重点解决“完整事件段”和“模型 t0”混在一起导致的锚点偏早、偏晚、结束过早或结束过晚问题。

当前完成：已新增并运行 `build_record_episode_dataset_v2_2_epoch_refined.py`。该脚本基于 v2.1 全量样本和原始车辆 CSV，重新估计每个 episode 的完整活动段、驾驶员动作开始、车辆响应开始、风险峰值、模型锚点、输入窗口和标签窗口，并生成代表性复核图。

最近一次结果：全部 episode 1766 个；边界基本一致 398 个；需要重划边界 1360 个；活动弱或边界不清楚 8 个；v2.2 可进入边界训练池 1721 个；代表性复核图索引 114 张。

主要发现：旧开始偏早 846 个，旧模型锚点偏早 614 个，旧结束偏晚 459 个，旧结束偏早 449 个，旧开始偏晚 154 个，旧模型锚点偏晚 136 个。新开始相比旧开始中位数晚约 0.735 s，新锚点相比旧锚点中位数晚约 0.237 s，说明旧 epoch 里确实混入了大量平稳前奏，也有一部分事件被截断或拖得过长。

当前判断：后续训练不要再直接使用旧 `episode_start_s`、`episode_end_s` 或 `model_anchor_s_v1_8`。完整事件段用于人工复核和事件理解；模型输入输出应优先使用 v2.2 的 `v2_2_model_anchor_s`、`v2_2_obs_start_s`、`v2_2_obs_end_s`、`v2_2_label_start_s`、`v2_2_label_end_s`。

当前最大风险：v2.2 是自动边界精修，不是最终人工真值。需要优先查看“旧结束偏早”“旧锚点偏晚”“活动弱或不清楚”等复核图，确认规则没有把短事件截断，也没有把正常平稳段误纳入标签窗口。

下一步准备做什么：先人工复核 v2.2 代表图；如果边界划定明显更合理，再基于 v2.2 窗口重建 vehicle-only 数据集和共同评价集，之后才训练模型。

用户可以优先查看：

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v2_2_epoch_user_summary_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\record_level_episodes_all_v2_2_epoch_refined.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\training_pool_epoch_refined_v2_2.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\epoch_boundary_rework_needed_v2_2.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\figures\epoch_boundary_review_v2_2`

---

# 项目状态更新：v2.1 横向偏移参考系与道路高程修正后样本表

更新时间：2026-05-26

当前阶段：旧流程样本规则修正。本轮不训练模型，先修正 Goal2 中横向偏移和高度异常过严导致的误排除问题。

当前完成：已新增并运行 `stage03_goal2_v21_reference_height_recovery.py`，生成 v2.1 样本恢复表、分角色统计、分高度规则统计和中文用户报告。

最近一次结果：全部 episode 1766 个；Goal2 严格排除样本 1407 个；v2.1 可进入训练池或复核训练池 1753 个；从 Goal2 严格排除集中恢复 1394 个；v2.1 硬排除 13 个。角色分布为：主训练候选 971，恢复复核候选 463，弱响应/对照候选 319。

当前判断：v2.1 是“恢复候选表”，不是最终干净训练集。它把横向偏移突变降级为 SILAB 道路/车道参考系切换风险提示，把小幅高度变化降级为复核提示，避免把可能可用样本过早删除。

当前最大风险：v2.1 训练池很宽，不能直接整体用于最终训练。下一步需要按主训练、恢复复核、弱响应/对照、高度重点复核和横向偏移参考系风险分层看图，再决定实际训练组合。

下一步准备做什么：先基于 v2.1 表抽样复核；如果恢复样本多数合理，再构建新的 vehicle-only 数据集和共同评价集，之后再训练模型。

用户可以优先查看：

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v2_1_user_summary_cn.md`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_training_pool_v2_1.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\v2_1_role_summary.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_height_review_v2_1.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_lateral_reference_switch_review_v2_1.csv`

---

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

## 2026-05-20 v1.1 完整记录级样本车辆-only GPU 基线

- 为什么做：基于 v1.1 主训练候选样本，先训练车辆-only，检查新 episode 样本定义是否适合建模。
- 运行设备：`cuda`，本地 CUDA。
- 切分：test=cwh/gf/tyy，val=byx/gzj/yyl，其余 train。
- 当前综合排序第一：`v11_vehicle_onset_nolat`，test RMSE=0.3532，大响应错侧率=0.0000，严重幅值不足率=0.5000，大响应召回=0.5000。
- 当前判断：单看 RMSE 最低的是上下文均值模型，不符合极限工况幅值建模目标；车辆响应锚点的方向物理指标更好，但幅值仍明显不足，暂不能升级为最终样本/模型主线。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v11_vehicle_only_gpu_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline`。

## 2026-05-21 完整记录级 episode 样本集 v1.2

- 为什么做：用户指出 v1.1 中 60 秒、80 秒、105 秒 episode 不符合真实单事件逻辑，可能是上下马路/路外恢复或连续过程误合并。
- 本轮动作：加入 `zx|z` 高度、俯仰、横向偏移和时长约束，生成 v1.2 新样本集，不训练模型。
- v1.2 主训练候选：1081；暂缓/复核：302；疑似上下马路/路外恢复：149。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_2_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned`。

## 2026-05-21 完整记录级 episode 样本集 v1.3

- 为什么做：用户指出 v1.2 把一个上下马路/路边恢复样本误判为目标极限事件，又把一个弯道样本误判为上下马路。
- 本轮动作：加入高度去趋势、平滑坡度识别、横向偏移跳变、车速/制动组合风险和用户反例覆盖规则，生成 v1.3 新样本集；本轮不训练模型。
- v1.3 主训练候选：820；暂缓/复核：563；疑似路边恢复或上下马路：393；弯道/坡度复核：128。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_3_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned`。

## 2026-05-21 完整记录级 episode 样本集 v1.4

- 为什么做：用户复核后认为多数高度 z 风险样本确实像上下马路/路边恢复，但高度明显大幅下降的片段应先保留为极限工况样本。
- 本轮动作：在 v1.3 基础上计算 episode 开始后的 z 下坠幅度，保留 `z_drop >= 2.0m` 的高度大幅下降样本，其它上下马路/路边恢复样本先抛弃；本轮不训练模型。
- v1.4 主训练候选：842；高度大幅下降保留样本：22；上下马路但无明显大幅下降抛弃样本：371。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_4_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed`。

## 2026-05-21 完整记录级 episode 样本集 v1.5

- 为什么做：用户复核后指出 v1.4 保留的高度大幅下降样本实际都是弯道路段，应单独判断，不应混入主训练候选。
- 本轮动作：把 v1.4 的 `train_z_drop_extreme_keep` 且属于弯道上下文的样本改为 `review_curve_z_drop_separate`；本轮不训练模型。
- v1.5 主训练候选：820；弯道高度下降单独复核：22。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_5_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated`。

## 2026-05-21 完整记录级 episode 样本集 v1.6

- 为什么做：用户指出弯道不能只看方向盘，需重点看侧倾；开上弯道两侧斜坡造成高度异常的样本也不要作为目标弯道样本。
- 本轮动作：将弯道从主训练候选中完全拆出，按弯道侧倾候选、弯道高度异常排除、弯道普通/弱侧倾复核分层；本轮不训练模型。
- v1.6 非弯道主训练候选：687；弯道侧倾候选：95；弯道高度异常排除：162。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_6_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split`。

## 2026-05-22 完整记录级 episode 样本集 v1.7

- 为什么做：用户指出此前把平滑下坡弯道误判为高度异常；正常弯道应是高度连续下降且允许小波动，异常应看突变、反常波动或不符合正常下坡的高度轨迹。
- 本轮动作：修正弯道高度规则，将平滑下坡弯道从异常类中救回；本轮不训练模型。
- v1.7 非弯道主训练候选：687；平滑下坡弯道侧倾候选：54；弯道高度轨迹异常排除：202。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_7_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised`。

## 2026-05-22 完整记录级 episode 样本集 v1.8

- 为什么做：用户指出部分 episode 起点过早，前面长时间平稳驾驶；弯道小幅高度波动不应排除，大部分弯道待复核样本可以先纳入训练候选。
- 本轮动作：新增模型用锚点，裁掉过早平稳前缀；放宽弯道小波动样本，仅排除高度明显变高或 z 形态异常样本；本轮不训练模型。
- v1.8 全部训练候选：903；弯道训练候选：216；弯道高度异常排除：214；锚点裁掉平稳前缀样本：176。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_8_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised`。

## 2026-05-22 完整记录级 episode 样本集 v1.9 道路坐标判弯道

- 根据用户纠正，本轮废弃“高度下降判弯道”的错误逻辑；弯道改由车辆 `zx|x/zx|y` 匹配道路中心线 `full_centerline_layout.csv` 后的 `curve1/curve2/curve3` 判断。
- 高度 z 只作为异常证据，用于判断疑似上斜坡、下路边或非正常高度跳变；不是弯道定义依据。
- 全部 episode `1766` 个，道路坐标弯道上下文 `506` 个，训练候选 `971` 个，其中弯道候选 `383` 个，非弯道候选 `588` 个。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_9_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised`。

## 2026-05-22 v1.9 非弯道高度微小变化审计

- 为什么做：用户指出直路/非弯道事件中高度 z 也可能有 `0.00x~0.0x m` 的小幅变化，可能来自车身侧倾、姿态、悬架或车辆动态，而不是道路坡度或路外异常。
- 审计结论：判断基本成立。v1.9 非弯道训练候选 `588` 个中，去趋势后 z 波动中位数为 `0.0041 m`，`87.9%` 小于等于 `0.01 m`，`93.2%` 小于等于 `0.02 m`，`95.2%` 小于等于 `0.05 m`。
- 解释边界：`zx|z` 不能直接等同于真实车辆质心高度；目前只能说厘米级 z 微动与横滚姿态变化明显相关，应作为车辆动态辅助信号，不能单独作为路外/异常判据。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_v1_9_noncurve_height_micro_motion_audit_cn.md`。


## 2026-05-22 完整记录级 episode 样本集 v2.0 全量无历史继承重审

- 为什么做：用户指出不能再用“历史上不是候选”作为排除依据；此前未判为候选的 episode 也必须按当前道路坐标和车辆动态重新审查。
- 本轮动作：基于 v1.9 的道路坐标和车辆动态特征，对全部 `1766` 个 episode 重新分类；历史 v1.8/v1.9 标签只作为审计对照，不参与 v2.0 决策。
- v2.0 训练候选：`984`，其中非弯道 `746`，弯道 `238`；从 v1.9 非训练集合中重新纳入训练：`383`；待复核：`463`。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v2_0_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit`。

## 2026-05-22 v2.0 全量无历史继承重审样本车辆-only GPU 基线

- 为什么做：用户指出不能再按历史候选身份筛样本，因此 v2.0 已对 1766 个 episode 全量重审。本轮只训练车辆-only 基线，检查新样本池是否具备建模价值。
- 运行设备：`cuda`，本地 CUDA。
- 模型：无学习基线 + 线性头 + 小型多层感知机；不加入连续风格、生理、脑电或教师蒸馏。
- 划分：test=cwh/gf/tyy，val=byx/gzj/yyl，其余 train。
- 当前综合排序第一：`v20_noncurve_train_anchor_nolat`，test RMSE=0.3906，大响应错侧率=0.0000，严重幅值不足率=0.5000，大响应召回=0.5000。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v20_no_history_vehicle_only_gpu_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline`。

## 2026-05-22 v2.0 待复核样本纳入训练车辆-only GPU 基线

- 为什么做：检查待复核样本是否可以作为训练样本，而不是直接排除。
- 运行设备：`cuda`，本地 CUDA。
- 模型：无学习基线 + 线性头 + 小型多层感知机；不加入连续风格、生理、脑电或教师蒸馏。
- 当前综合排序第一：`v20_train_review_anchor_nolat`，test RMSE=0.3842，大响应错侧率=0.3333，严重幅值不足率=0.5000，大响应召回=1.0000。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v20_review_inclusion_vehicle_only_gpu_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu`。

## 2026-05-25 goal1 v2.0 训练任务重定义执行

- 为什么做：按照 `gptpro_answer/goal1.txt`，把 v2.0 从固定窗口方向盘预测升级为 episode 级车辆-only 联合响应任务。
- 已完成：新版 manifest、可变窗口/掩码数组、E0-E5 车辆-only 实验、预测图和最终报告。
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs\goal1_experiment_summary.csv`。
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_goal1_v2_task_redesign_user_summary_cn.md`。
- 实验输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs`。
- 当前边界：本轮不加入连续驾驶风格、生理数据、脑电或教师蒸馏。

## 2026-05-26 Goal2 clean vehicle-only 任务审计

- 为什么做：修正 goal1 的样本排除、输入泄漏、锚点审计和评价口径问题。
- 已完成：严格 clean manifest、无未来输入特征、anchor audit、common eval、G2_E0-G2_E5 vehicle-only 实验和预测图。
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\goal2_experiment_summary.csv`。
- 用户查看报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_goal2_clean_task_user_summary_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs`。
- 当前边界：仍未加入连续驾驶风格、生理、脑电或教师蒸馏。

## 2026-05-26 Goal2 被排除样本原因拆解

- 为什么做：用户指出 Goal2 的高度异常、旧结论继承和 0.50 高度阈值过严，需要确认被排除样本是否被误伤。
- 已完成：对 1407 个 `excluded_slope_or_offroad` 样本生成逐样本排除原因、恢复优先级和抽查清单。
- 关键发现：`A_优先人工恢复复核` 792 个，`B_较可能可恢复` 265 个，说明 Goal2 严格排除确实大量依赖旧版本文字/标记，不适合作为最终硬规则。
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_goal2_exclusion_recovery_audit_cn.md`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit`。
- 当前结论：旧版本文字和旧标记以后只能作为复核提示，不能继续作为硬排除规则。

## 2026-05-26 Goal2 人工审核图片整理

- 为什么做：方便用户直接查看被 Goal2 排除但可能可恢复的样本图。
- 已完成：按恢复优先级整理图片目录，已复制 487 张现有复核图；920 条样本当前缺少可用图片路径，已单独列出。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\index.html`。
- 优先查看：`00_A_优先看_旧结论可能误伤` 和 `01_B_较可能可恢复_看图确认`。
- 缺图清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\manual_review_images_missing.csv`。

## 2026-05-26 重要规则更新：SILAB 横向偏移不能作为硬排除依据

- 当前确认：SILAB 的横向偏移很可能是相对当前道路/车道参考线计算的量，而不是严格连续的世界坐标横向位移。车辆跨入另一条道路或车道后，当前道路/车道参考系可能切换，横向偏移会出现突兀跳变。
- 后续样本筛选必须记住：
  1. 横向偏移跳变不能单独判定为“坐标异常、下马路、路边恢复、驶出道路”。
  2. 横向偏移跳变只能作为“道路/车道参考系切换风险”的提示字段。
  3. 是否排除样本，需要结合道路坐标、道路设计、车辆高度 z、姿态、速度、制动、横向加速度、横摆、横滚和方向盘变化综合判断。
  4. 旧版本里由横向偏移跳变触发的路边/下马路/高度异常结论，只能作为人工复核提示，不能继续作为硬排除规则。
  5. 后续训练 manifest 重建时，应把相关字段命名为类似“道路/车道参考系切换风险”，而不是直接写成“坐标错误”。
- 对 Goal2 的影响：Goal2 的严格排除很可能误伤了一部分样本。下一步如果继续旧流程，应优先做“道路参考系修正后的样本恢复规则”，再训练 vehicle-only 对照。

## 2026-05-26 重要规则更新：高度异常不能只看十几厘米级变化

- 已检查道路设计文件：`full_centerline_layout.csv`、`curve1_Area2.cfg`、`curve2_Area2.cfg`。
- 当前确认：
  1. `curve1` 设计道路 z 范围约 7 m，`curve2` 设计道路 z 范围约 6 m。
  2. 非弯道模块在中心线总表中的道路 z 基本为 0。
  3. 真实道路下坡/上坡是米级变化，十几厘米到二十厘米的高度变化不能直接判定为下马路、上斜坡或驶出道路。
- 后续样本筛选必须记住：
  1. 小幅高度变化只能作为车辆姿态/悬架/采样/参考系变化提示，不是硬排除依据。
  2. `z_residual_range < 0.20 m` 不应排除。
  3. `0.20 m <= z_residual_range < 0.50 m` 只进入复核。
  4. `z_residual_range >= 0.50 m` 也要结合道路坐标、道路设计、速度、制动、横摆、横滚、方向盘和横向偏移参考系共同判断。
  5. 对 `curve1/curve2`，不能直接用原始 `z_range` 判异常，必须比较车辆 z 与道路设计 z 的残差。
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_goal2_height_rule_design_audit_cn.md`
# 2026-06-22 v222a 候选曲线缓存与轻量受限残差已完成

- 状态：已完成 v222a 的两步本地执行：先导出固定候选曲线缓存，再在固定 formal 候选池上运行轻量软融合/受限残差校准。未训练新的神经网络，未改变候选池，未把 test 用于模型或超参选择。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_candidate_curve_cache_20260622.py`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_light_fusion_residual_20260622.py`
- 新增输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622`
- 候选缓存结果：
  - `loose_main_pool`：`X_hist (1167,31,11)`，`Y_future (1167,21,9)`，`predictions (1167,14,21)`，`feature_matrix (1167,229)`。
  - `strict_main_pool`：`X_hist (963,31,11)`，`Y_future (963,21,9)`，`predictions (963,14,21)`，`feature_matrix (963,229)`。
  - feature schema 审计 458 行，fail 行数 0；formal 候选不含 `W3_B4_original_soft/oracle/fallback/true_label`。
  - 与 v219 指标交叉检查 72 行全部通过，最大差异 `0.0019267822736031`，在显式阈值 `0.002` 内。
- v222a 轻量校准结果：
  - validation selection 表只含 val 排序行，共 108 行；最终每个 pool 选择 1 个输出后才报告 test。
  - `loose_main_pool` 选中 `v222a_bounded_residual_global_blend_a1p0_b0p2`：test RMSE `0.555940`，tail `0.657612`，低估率 `0.108696`，强响应低估率 `0.149254`。对比固定 baseline `avg_joint_focus`：RMSE `0.544884`，tail `0.629752`，低估率 `0.163043`。结论是低估率下降，但整体和尾段 RMSE 变差。
  - `strict_main_pool` 选中 `v222a_bounded_residual_global_blend_a10p0_b0p2`：test RMSE `0.571966`，tail `0.681413`，低估率 `0.137931`，强响应低估率 `0.171875`。对比固定 baseline `peak_floor_090`：RMSE `0.571770`，tail `0.658306`，低估率 `0.137931`。结论是没有形成严格主池的新收益。
- 当前判断：
  - v222a cache 底座是后续 selector/calibration 的可复用产物；
  - 本轮轻量 bounded residual 不能升级为新的 headline 模型，只能作为“降低低估但牺牲 RMSE”的诊断证据；
  - 下一步不应直接进入 v222b/v223 大模型，应先做 no-harm 约束、validation/test mismatch 和逐样本 harm/gain 分解，确认是否能只在低估风险样本上启用校准。
- 验证：
  - 两个新增脚本 `python -m py_compile` 均通过；
  - v222a cache ZIP 校验 `bad_file=None`，包含 11 个文件；
  - v222a light fusion ZIP 校验 `bad_file=None`，包含 15 个文件；
  - `Select-String` 未在 formal/selection/selected 表中检出 `W3_B4_original_soft`、`oracle`、`fallback` 或 `true_label`。

---
# 2026-06-22 v222a no-harm gate 诊断已完成

- 状态：按 GPTPro 新指令完成 `v222a_gain_harm_decomposition -> oracle safe gate upper bound -> binary validation-only no-harm gate`。本轮没有训练 v222b/v223，没有做多候选 router。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_noharm_gate_diagnostic_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622`
- 核心产物：
  - `tables/gain_harm_decomposition.csv`
  - `tables/oracle_safe_gate_report.csv`
  - `tables/val_gate_tradeoff_table.csv`
  - `tables/test_locked_gate_report.csv`
  - `tables/per_sample_gate_decisions.csv`
  - `logs/selected_gate_manifest.json`
- validation 结果：
  - `loose_main_pool`：validation-selected gate 通过 no-harm-first，RMSE delta `-0.018917`，tail delta `-0.013437`，under reduction `0.064725`，strong-under reduction `0.069959`，coverage `0.944984`。
  - `strict_main_pool`：validation-selected gate 通过 no-harm-first，RMSE delta `-0.010182`，tail delta `-0.008429`，under reduction `0.003704`，strong-under reduction `0.004651`，coverage `0.359259`。
- locked test 结果：
  - `loose_main_pool`：formal gate 未通过。under reduction 仍为正 `0.043478`，strong-under reduction `0.037313`，但 RMSE delta `+0.010559`、tail delta `+0.027764`，伤害 RMSE/tail。
  - `strict_main_pool`：formal gate 未通过。RMSE delta `-0.008975`、tail delta `-0.005264`，但 under reduction `-0.017241`、strong-under reduction `-0.023438`，低估反而变差。
- oracle safe gate 上限：
  - `loose_main_pool` test oracle：RMSE `0.520273`，tail `0.597736`，under `0.119565`，strong-under `0.156716`，coverage `0.423913`。
  - `strict_main_pool` test oracle：RMSE `0.538076`，tail `0.618740`，under `0.120690`，strong-under `0.156250`，coverage `0.436782`。
  - 解释：residual 局部有价值，但当前可部署 gate 不能稳定学出“何时启用”。
- 当前判断：
  - v222a bounded residual 继续保持 diagnostic，不升级为 formal headline；
  - learned no-harm gate 在 validation 过关但 locked test 失败，说明继续加复杂 selector 风险很高；
  - 下一步应把本轮结果报告给 GPTPro，重点让它裁决：是否停止 v222a，还是仅做样本级 case study / 更换 gate 特征但不进入 v222b/v223。
- 验证：
  - `python -m py_compile` 通过；
  - ZIP `v222a_noharm_gate_diagnostic_pack.zip` 校验 `bad_file=None`，包含 11 个文件；
  - feature schema 458 行 fail=0；
  - leakage guard 全 pass；
  - 禁用名检查未检出 `W3_B4_original_soft/oracle_model/fallback/true_label`。

---
# 2026-06-22 v226 formal robustness / CI audit 已完成
- 状态：按 GPTPro v226 指令完成 `formal robustness / confidence-interval audit`。本轮只读取 v225 已锁定 formal 输出表，未训练新模型，未调 threshold/tau，未创建 gate/router，未运行 v222b/v223。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v226_formal_robustness_ci_audit_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\v226_formal_robustness_ci_audit_pack.zip`
- formal lock 复核：
  - `loose_main_pool = avg_joint_focus`
  - `strict_main_pool = peak_floor_090`
  - v226 继续保持 v222a/no-harm/oracle/residual 相关行 diagnostic-only，不进入 formal 指标表。
- locked test 指标复现：
  - loose `avg_joint_focus`：RMSE `0.544884`，tail RMSE `0.629752`
  - strict `peak_floor_090`：RMSE `0.571770`，tail RMSE `0.658306`
  - 四项复现误差均小于 `1e-5`。
- 95% sample bootstrap CI（test）：
  - loose RMSE `0.496066-0.593811`，tail `0.564811-0.693788`
  - strict RMSE `0.511036-0.635521`，tail `0.581652-0.736696`
- 95% subject-block bootstrap CI（test，4 个 subject）：
  - loose RMSE `0.428783-0.599684`，tail `0.515881-0.687686`
  - strict RMSE `0.473689-0.615000`，tail `0.539479-0.706505`
- tail error 集中度（test）：
  - loose top-20% tail-SSE share `0.659320`
  - strict top-20% tail-SSE share `0.672493`
- 验证：`py_compile`、完整运行、ZIP `bad_file=None`、required files `[]`、formal lock、metric reproduction、leakage guard、forbidden scan、table alignment、figure count 全部通过。
- 当前下一步：把 v226 pack、CI 结果、readiness 决策和验证摘要报告给 GPTPro，请它给下一轮 bounded 指令。仍然不允许本地自行进入 v222b/v223、新 tau、新 gate/router 或 test-based retuning。
---
# 2026-06-22 v227 paper / claim readiness pack 已完成（最新）

- 当前状态：v226 结果已尝试回报 GPTPro，但桌面端连续出现空的“已停止思考”输出，Chrome 项目桥接被登录页阻塞；已将桥接失败归档。由于没有拿到新的 GPTPro 正文指令，本轮只做 reporting-only fallback：把 v225+v226 已锁定证据整理成写作 / claim readiness 包。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v227_paper_claim_readiness_pack_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\v227_paper_claim_readiness_pack.zip`
- 核心产物：`paper_main_result_table.csv`、`paper_claim_support_matrix.csv`、`paper_limitation_table.csv`、`formal_guardrail_summary.csv`、`figure_selection_index.csv`、`v227_paper_claim_readiness_cn.md`、`v227_next_gptpro_prompt_ascii.md`。
- 验证摘要：`py_compile` 通过；脚本完整运行通过；ZIP `bad_file=None`；required files `[]`；`no_model_change_guard.pass=True`；`source_artifact_checks.pass=True`；formal lock 仍为 `loose_main_pool=avg_joint_focus`、`strict_main_pool=peak_floor_090`。
- v227 prompt 已再次尝试发送给 GPTPro，但 Chrome bridge 仍因无法验证 Pro/进阶模式而拒绝发送；阻塞记录已归档到 `gptpro_reviews\20260622_v227_result_gptpro_*_blocked.md`。
- 边界：v227 不是 GPTPro 新批准的实验方向，不训练模型、不调 tau/threshold、不创建 gate/router、不运行 v222b/v223、不改变 formal headline。下一步仍是等 GPTPro 通道恢复后，把 v226+v227 结果一起回报。

---
# 2026-06-23 goal-level blocked：GPTPro 通道需要用户侧恢复（最新）

- 当前状态：Codex-GPTPro 闭环已连续多轮卡在同一个外部通道问题上。桌面端 ChatGPT 没有给出有效 bounded GPTPro 回复；Chrome bridge 每次都在发送前失败，原因是无法验证 Pro/进阶模式。
- 已新增 goal-level blocked 归档：
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_goal_blocked_gptpro_channel_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_goal_blocked_gptpro_channel_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_goal_blocked_gptpro_channel_action_items.md`
- 当前已经完成的安全本地工作：v226 formal robustness / CI audit、v227 reporting-only paper / claim readiness pack、ZIP 复核、note layer 同步。
- 当前必须停止自动扩展：没有 GPTPro 新指令前，不进入 v222b/v223、新 tau/threshold、新 gate/router/selector、新模型训练、formal leaderboard/headline 改动或 test-based retuning。
- 恢复方式：用户恢复 Chrome ChatGPT Pro/进阶项目通道，或直接把 GPTPro 下一条 bounded 指令粘贴给 Codex；恢复后从 `v227_next_gptpro_prompt_ascii.md` 继续闭环。

---
# 2026-06-23 v228 final paper artifact freeze 已完成（最新）

- 当前状态：用户指出前一轮发给 GPTPro 的问题在本地软件端显示为乱码；已改用本地 ChatGPT Desktop 软件端，并用纯 ASCII handoff/retry 获取到有效 GPTPro 回复。
- GPTPro 决策：接受 v227 作为 reporting-only closeout；下一步只允许 `stage03_v228_final_paper_artifact_freeze_20260623.py`，任务类型为 reporting / packaging / manuscript-readiness，不做模型工作。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v228_final_paper_artifact_freeze_20260623.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v228_final_paper_artifact_freeze_20260623`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v228_final_paper_artifact_freeze_20260623\v228_final_paper_artifact_freeze_pack.zip`
- 关键结果：formal lock 仍为 `loose_main_pool=avg_joint_focus`、`strict_main_pool=peak_floor_090`；主结果 2 行；claim lock 5 条；limitation 6 条；主图 6 张、附录图 14 张。
- 独立验证：`py_compile` 通过；脚本完整运行通过；ZIP `testzip=None`；required files missing `[]`；主指标与锁定 v225/v226 数值差值为 0；CI 行数与 v226 完全一致；forbidden hits `0`；`guardrail_check.pass=true`；`consistency_check.pass=true`。
- 旧结论修正：此前 “GPTPro channel blocked” 是不完整诊断，根因包含 prompt mojibake。当前已经用本地软件端取得有效 GPTPro 回复并完成 v228。
- 当前停止条件：v228 包生成且验证通过后停止；下一轮如继续，应把 v228 执行结果回报 GPTPro，请它只给一个 bounded 下一步。

---
# 2026-06-26 v243 v241 guarded fine-tune 已完成（最新）

- 当前状态：按用户要求回到“方法提升/行为预测”主线，不继续 v222a gate、删除样本或轻量 residual 路线；本轮在 v241 `v241_tcn_mha_h96` backbone 上做 guarded fine-tune。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v243_v241_guarded_finetune_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626\reports\v243_v241_guarded_finetune_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626\v243_v241_guarded_finetune_pack.zip`
- 方法改动：
  - 直接从 v241 checkpoint 初始化，不重建任务，不换成 v242 joint curve。
  - 对 v241 train 残差高、strong/observe/zero-cross/reverse/multi-correction 等困难样本提高 point-level 权重。
  - 加入相对 v241 teacher 的 guard loss：新模型若比 v241 在同一点误差更大，会被惩罚。
  - 对 v241 已经表现好的 normal 样本可启用 teacher anchor，减少正常样本漂移。
  - 第一轮按 point-level loss 早停时三组候选都停在 epoch 0；随后修正为每隔 2 个 epoch 还原曲线并按 validation 分层指标选快照。
- validation 结果：
  - 三个候选都通过 validation no-harm、v241-upgrade、sample-guard 和 meaningful-gain。
  - validation 排名第一：`v243_metric_hard36_guard08`，score `0.865386`，best_epoch `34`，`accepted_as_next_candidate=True`。
  - hard36 vs v241 validation：all 0-800 mean tail delta `-0.007909`，observe 0-800 mean tail delta `-0.004415`，strong 400/1000 mean tail delta `-0.010060`，normal max tail delta `+0.002696`。
- locked test 观察：
  - validation-selected `v243_metric_hard36_guard08` 在 test 的 all/normal 有改善，但 observe_later_like 和 strong_steer 多数 delay 变差：observe mean tail delta `+0.009219`，strong mean tail delta `+0.003896`。
  - test 最均衡候选反而是 conservative `v243_metric_hard24_guard04`：all `-0.003832`，normal `-0.003955`，observe `-0.006484`，strong `-0.003601`；observe/strong 各只有 1 个 delay 的 tail delta 为正。
  - 这不能反向改 validation 选择，也不能把 hard24 直接说成 formal replacement；它说明 v243 需要下一轮 locked audit，把 validation-selected hard36 和 conservative hard24 并列审查。
- 当前判断：
  - v243 证明“在 v241 backbone 上通过 guarded loss + hard weighting 还能挤出增益”，但最佳 validation 候选存在 test bucket 迁移风险。
  - 当前不能把 v243 直接升级为 formal replacement；下一步应做 `v244_locked_audit_compare_v243_hard36_vs_hard24`，重点审查 hard36 的 test 退化与 hard24 的稳健性。
- 验证：
  - `python -m py_compile` 通过。
  - 完整训练运行完成。
  - `guardrail_check.pass=True`，`leakage_check.pass=True`，同一 `event_uid` 跨 split 数为 `0`。
  - ZIP `testzip=None`，条目数 `29`。

---
# 2026-06-29 v244 hard36 vs hard24 locked audit 已完成（最新）

- 当前状态：按用户要求比较 v243 中 validation-selected `hard36` 与 conservative/test-robust `hard24`。本轮只读取 v243 已落盘产物，不训练新模型，不调权重，不改 validation 规则。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v244_locked_audit_compare_v243_hard36_vs_hard24_20260629.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v244_locked_audit_compare_v243_hard36_vs_hard24_20260629`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v244_locked_audit_compare_v243_hard36_vs_hard24_20260629\reports\v244_locked_audit_compare_v243_hard36_vs_hard24_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v244_locked_audit_compare_v243_hard36_vs_hard24_20260629\v244_locked_audit_compare_v243_hard36_vs_hard24_pack.zip`
- 关键限制：
  - v243 的 `.npz` 只保存了 best guarded prediction，即 `v243_metric_hard36_guard08`。
  - `v243_metric_hard24_guard04` 没有完整曲线预测、checkpoint 和逐样本 delta，所以本轮只能对 hard24 做 aggregate locked test 对比，不能做同级别 per-sample casebook。
- 对比结果：
  - validation-selected：`v243_metric_hard36_guard08`，validation score `0.865386`，best_epoch `34`。
  - locked test 更稳：`v243_metric_hard24_guard04`。
  - hard36 test vs v241：all `-0.002128`，normal `-0.006139`，observe `+0.009219`，strong `+0.003896`。
  - hard24 test vs v241：all `-0.003832`，normal `-0.003955`，observe `-0.006484`，strong `-0.003601`。
  - hard30 参考：all `-0.004607`，normal `-0.006566`，observe `-0.001721`，strong `-0.001645`。
  - hard36 在 observe/strong hard bucket 的变差 delay 数为 `11/12`，hard24 为 `2/12`。
- 当前判断：
  - 不应直接把 hard36 升为 formal replacement：它 validation 最优，但 locked test hard bucket 退化明显。
  - 不应直接把 hard24 升为 formal replacement：它 aggregate test 更稳，但缺少 hard24 完整预测/checkpoint/逐样本审计，而且不能用 test 反向改 validation 选择。
  - 在补齐 hard24 granular artifact 前，v241 仍应保持默认候选；若继续推进 v243，应只重放保存 all-candidate predictions/checkpoints，不改超参、不基于 test 调参。
- 验证：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `guardrail_check.pass=True`。
- ZIP `testzip=None`，条目数 `16`。

---
# 2026-06-30 v245 差样本锚点后移效果审查已完成（最新）

- 当前状态：按用户判断“差样本可能需要后移锚点、多观察一点时间”做了定量审查。本轮只读取已有 v241 / v243-hard36 逐样本预测和 v236 rolling 输入，不训练新模型、不调参、不改变 validation 规则。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v245_bad_sample_anchor_shift_effect_audit_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v245_bad_sample_anchor_shift_effect_audit_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v245_bad_sample_anchor_shift_effect_audit_20260630\reports\v245_bad_sample_anchor_shift_effect_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v245_bad_sample_anchor_shift_effect_audit_20260630\v245_bad_sample_anchor_shift_effect_audit_pack.zip`
- 关键发现：
  - test bad_top10 定义：v241 tail RMSE q90=`0.727`，共 `111` 个差样本；其中 base delay<=400ms 的早锚点差样本 `71` 个。
  - 对 v241 bad_top10，固定后移 `+400ms` 平均 tail RMSE delta=`-0.210`，改善率 `83.1%`；固定后移 `+600ms` delta=`-0.288`，改善率 `88.7%`。
  - 对早锚点 bad_top10，`+600ms` 是所有 71 个样本都可比较的最强固定后移：delta=`-0.288`，改善率 `88.7%`。
  - oracle 最佳后移上限：bad_top10 平均 delta=`-0.382`，改善率 `93.1%`；早锚点 bad_top10 平均 delta=`-0.428`，改善率 `95.8%`。
  - 本轮比较的是 original anchor 后 1.0-2.0s tail 段，overlap 点数为 `11`，真实绝对轨迹对齐误差约 `1e-7`，所以改善不是因为少预测了一段，而是后移锚点后同一段后续轨迹确实更容易预测。
- 当前判断：
  - 差样本锚点后移有明确效果，尤其是早锚点差样本。
  - 不建议统一后移全部样本；下一步更合理的是构造“风险样本允许延后观察/重锚定”的 v246 训练任务，并保留普通样本原锚点预测。
- 验证：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`，条目数 `14`。

---

# 2026-06-30 v246 oracle 最佳锚点遍历与 input-only selector 审查已完成（最新）

- 当前状态：按用户建议完成“遍历锚点后移，使每个样本达到最佳锚点”的定量审查。本轮明确区分两件事：`oracle_best_anchor` 是用真实误差挑出来的理论上限，不能部署；`input-only selector / fixed waiting policy` 才是接近部署逻辑的检查对象。
- 新增脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v246_oracle_best_anchor_and_selector_audit_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v246_oracle_best_anchor_and_selector_audit_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v246_oracle_best_anchor_and_selector_audit_20260630\reports\v246_oracle_best_anchor_and_selector_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v246_oracle_best_anchor_and_selector_audit_20260630\v246_oracle_best_anchor_and_selector_audit_pack.zip`
- 关键发现：
  - test bad_top10：当前 v241 tail RMSE `1.008`，oracle 最佳锚点降到 `0.656`，mean delta `-0.352`，改善率 `84.7%`。
  - early bad_top10：当前 `1.021`，oracle 最佳锚点降到 `0.591`，mean delta `-0.431`，改善率 `95.8%`，最常见 oracle shift 为 `+600ms`。
  - RF input-only selector 在 test bad_top10 上只能把 RMSE 降到 `0.908`，mean delta `-0.100`，改善率 `29.7%`，说明仅靠 base 锚点前输入精确判断最佳等待时长仍然困难。
  - Ridge selector 与显式固定策略 `policy_wait_to_latest_anchor` 数值完全一致；在 test bad_top10 上 RMSE `0.685`、mean delta `-0.322`，说明主要收益来自“多看一点/等到最晚锚点”，不是已经学会逐样本精确找最佳锚点。
  - 对 early bad_top10，固定等到最晚锚点 mean delta `-0.391`，接近 oracle `-0.431`，支持“事件前几秒看不出差别的样本可以后移锚点、多观察一点”的判断。
- 当前判断：
  - v246 证明“后移/重锚定”方向有很强上限，且非常贴合差样本问题。
  - 但不能把 oracle 最佳锚点直接写进测试或训练标签选择；下一步应该设计带等待代价或触发条件的重锚定任务，避免模型退化成所有样本一律晚预测。
  - v247 更合理的方向是：普通样本保留当前锚点；风险/早锚点样本允许后移，但必须用 input-only 触发规则或等待成本约束，并分层报告 normal、bad_top10、early_bad_top10、observe_later_like、strong_steer。
- 验证：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`，条目数 `18`。

---
# 2026-06-30 v247 50ms 多分辨率 best anchor discovery 已完成（best label 成立，当前 selector 仍弱于 wait-latest）

- 当前状态：已按用户认可的“不是单纯后移，而是找到最佳锚点”路线完成 `v247_multi_resolution_best_anchor_discovery_20260630`。本轮从原始车辆 CSV 重新构造 `0/50/100/.../1000ms` 共 21 个候选锚点，用锁定 v241 checkpoint 推理每个候选，不训练新的轨迹模型；再用 `score = prediction_error + waiting_cost + instability_penalty` 得到离线 best anchor，并训练 input-only Ridge/RF selector 检查可学习性。
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v247_multi_resolution_best_anchor_discovery_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v247_multi_resolution_best_anchor_discovery_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v247_multi_resolution_best_anchor_discovery_20260630\reports\v247_multi_resolution_best_anchor_discovery_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v247_multi_resolution_best_anchor_discovery_20260630_pack.zip`
- 采样审计：50ms fine grid 可用，`1167` 个事件全部生成 `21` 个候选锚点，共 `24507` 行；`dropped=0`，完整事件比例 `1.000`，最近采样误差最大约 `2ms`。
- v241 replay 审计：fine-grid 里的 coarse delay 行可复现旧 v241 预测，coarse replay mean RMSE `0.000000`，max `0.000001`，说明 fine-grid 推理链路与 v241 checkpoint 对齐。
- primary score：`delay_l05_unstable_m05`。test/all 当前 0ms 平均 tail RMSE `0.475`，oracle best `0.253`，平均 best delay `596.2ms`；test/bad_top10 当前 `1.198`，oracle best `0.616`，平均 best delay `789.5ms`。
- 关键判断：v247 证明“best anchor label”在 50ms 细网格上有明显上限收益，但当前 input-only selector 还没有学到足够强的逐样本选锚规律。RF selector 在 test/bad_top10 上把 RMSE 从 `1.198` 降到 `0.947`，但固定 `policy_wait_to_latest_anchor` 已能降到 `0.695`，明显强于 RF selector。
- 方法含义：这一步支持“部分差样本确实需要更多观察时间”，但不支持立刻把当前 selector 当作可部署策略；下一步若继续，应改进 anchor selector 的输入/结构或把 anchor choice 与轨迹预测做成联合模型，同时必须继续报告 normal 是否受伤、是否只是学会等待到 1000ms。
- 验证：`python -m py_compile` 通过；完整脚本运行完成；`guardrail_check.pass=True`；`ZIP testzip=None`。

---
# 最新状态指针：2026-07-02 已完成 v289 RESP source phase route audit。v289 回到 cleaned 200Hz RESP 源信号，不使用已知弱的 `RESP_BPM/RESP_Amplitude` 派生列，而是重建呼吸周期、相位、幅值、质量和因果同步偏移特征，并复用 v278 vehicle top40 candidate route gate。结果：`guardrail.pass=True`，`zip_testzip=True`，`event_n=1167`，`resp_source_feature_n=575`，`feature_set_n=27`，`uses_post_observation_any=false`，`ok_rate=0.91945`，baseline 有效率中位数 `1.0`，context 呼吸周期中位数 `3.026s`，BPM 中位数 `19.83`。但 `route_viable_now=false`：validation 选出的 deployable top1 在 test bad_top10 上仍比 latest 差 `+0.1553`，在 test bad_top10_vehicle_ambiguous 上差 `+0.1251`；test-best top1 diagnostic 仍差 `+0.0625`，best corr `0.0463`。结论：RESP 源信号重建比 ECG 更接近 latest，但仍没有形成可部署候选选择；当前 goal 仍未达成。
---

# 2026-07-02 v289 RESP source phase route audit 已完成（最新）

- 当前阶段：承接 v288，继续做源信号层面的证据修复；本轮检查 RESP 呼吸相位/周期是否能弥补车辆锚点前信息不足。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v289_resp_source_phase_route_audit_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v289_resp_source_phase_route_audit_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v289_resp_source_phase_route_audit_20260702\reports\v289_resp_source_phase_route_audit_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v289_resp_source_phase_route_audit_20260702_pack.zip`
- 方法：
  - 直接读取 cleaned 200Hz `RESP_filt200/RESP_raw200`。
  - 不使用 `RESP_BPM/RESP_Amplitude` 记录级派生列。
  - 每个事件只使用 observation_s 前数据，基线为 observation 前 `-60s~-20s`。
  - 重建呼吸零交叉、周期、BPM、相位 sin/cos、峰谷幅值、方向斜率、质量和 end0/endm0p5/endm1/endm2 等因果偏移窗口。
  - 只在 train split 上做 feature screen，构造 27 组 RESP feature set，再进入 v284/v285 同口径 route gate。
- 核心结果：
  - `route_viable_now=false`。
  - `resp_source_feature_n=575`，`feature_set_n=27`。
  - `ok_rate=0.91945`。
  - `uses_post_observation_any=false`。
  - `baseline_valid_ratio_median=1.0`。
  - `context_period_s_median=3.0263`，`context_bpm_median=19.8259`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1553`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta vs latest `+0.1251`。
  - test-best top1 diagnostic 仍未超过 latest：最佳 `resp_window_dur3_endm1_top24`，delta `+0.0625`。
  - test bad_top10 best corr `0.0463`，低于 `0.05` 弱相关门槛。
  - test bad_top10_vehicle_ambiguous 中最接近的是 `resp_offset_end0_top32`：top1 delta `+0.0271`，top3 oracle delta `-0.0268`，但仍不是 deployable top1 改善。
- 当前判断：
  - RESP 源信号并非不可用，周期/相位重建有合理生理范围。
  - v289 比 ECG v288 的 test-best top1 更接近 latest，但仍没有过 route gate。
  - 失败不应继续归因于“没有重建 RESP 相位/周期”。
  - 当前生理 goal 仍未完成；不建议直接训练复杂 RESP/vehicle fusion 轨迹模型。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。
---

# 2026-07-02 v288 ECG source-signal route audit 已完成（最新）

- 当前阶段：承接 v287 的 ECG 弱苗头，不再沿 v285/v287 同一特征层继续换融合模型，而是回到 ECG 源信号层做证据修复。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v288_ecg_source_signal_route_audit_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702\reports\v288_ecg_source_signal_route_audit_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702_pack.zip`
- 方法：
  - 直接读取 cleaned 200Hz ECG_filt200 / ECG_raw200。
  - 每个事件只使用 observation_s 前数据，基线为 observation 前 `-60s~-20s`。
  - 提取 ECG 短窗形态、R 峰/RR、噪声/质量、最近窗口相对更早窗口的 delta，以及 end0/endm0p5/endm1/endm2 等因果同步偏移窗口。
  - 只在 train split 上做 feature screen，构造 27 组 ECG feature set，再进入 v284/v285 同口径 route gate。
- 核心结果：
  - `route_viable_now=false`。
  - `ecg_source_feature_n=518`，`feature_set_n=27`。
  - `ok_rate=0.91945`。
  - `uses_post_observation_any=false`。
  - `baseline_valid_ratio_median=1.0`，`dur2_end0_valid_ratio_median=1.0`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1556`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta vs latest `+0.1510`。
  - test-best top1 diagnostic 仍未超过 latest：最佳 `ecg_category_morph_dynamic_top48`，delta `+0.0903`。
  - test bad_top10 best corr `0.0620`，来自 `ecg_duration_dur1_top32`，属于弱排序诊断信号。
  - test bad_top10_vehicle_ambiguous 中 `ecg_duration_dur1_top32` corr `0.1011`，top3 oracle delta `-0.0287`，但 top1 仍差 `+0.0457`，不能作为部署策略。
- 当前判断：
  - ECG 源信号不是整体不可用；有效率和基线覆盖都站得住。
  - 失败不是简单的 ECG 峰检测质量或因果短窗同步偏移问题。
  - ECG 有弱排序信号，但不能稳定把 vehicle top40 中的正确未来候选排到第一。
  - 当前生理 goal 仍未完成；不建议继续做同类 ECG feature/gate 微调。
- 校验：
  - `python -m py_compile` 通过。
  - 脚本完整运行完成。
  - `guardrail_check.pass=True`。
  - ZIP 自检 `testzip=True`。
---
# 最新状态指针：2026-07-02 已完成 v297 subject style stability audit。v297 按用户提出的优先级先审计“同一被试的多次独立 trial 是否存在稳定驾驶风格信号”，而不是直接假设历史事件能预测当前事件。结果：`guardrail.pass=True`、`zip_testzip=True`、`event_n=1167`；train 关键响应描述符的 subject eta mean `0.0598`、median `0.0518`，同被试响应距离 / 异被试响应距离 `0.7103`，说明存在弱到中等的被试层差异；但 rolling history 在 test 上只稳定改善 `v249_rmse` 和 `v249_tail_rmse`，关键目标改善比例仅 `2/7=0.2857`，因此 `style_route_supported_now=false`、`weak_style_signal_exists=true`、`event_label_route_priority=true`。结论：驾驶风格可作为风险/置信度校准辅助，不应单独作为轨迹预测主线；下一步优先转向事件级标签/实验条件标签与响应类型辅助监督。

# 2026-07-02 v297 subject style stability audit 已完成（最新）

- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v297_subject_style_stability_audit_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v297_subject_style_stability_audit_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v297_subject_style_stability_audit_20260702\reports\v297_subject_style_stability_audit_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v297_subject_style_stability_audit_20260702_pack.zip`
- 方法：
  - 从 v249 `delay_ms=0` 的真值/预测提取事件级未来响应描述符：峰值、峰值时间、最终偏移、曲线长度、反向/多段修正、v249 RMSE、tail RMSE、残差均值等。
  - 计算 subject / recording 对这些响应描述符的 eta squared。
  - 计算同被试 vs 异被试的标准化响应距离。
  - 做 rolling history 审计：每个事件只使用同被试、时间上更早的事件均值作为“历史风格”预测当前事件描述符。
  - 生成未来轨迹派生的 oracle 标签候选：强度、时序、形状、方向、误差分层；这些标签只允许用于辅助监督/分层/上限分析，不能作为 test 直接输入。
- 核心结果：
  - `guardrail_check.pass=True`。
  - `zip_testzip=True`。
  - `event_n=1167`。
  - `key_subject_eta_train_mean=0.059843`，`key_subject_eta_train_median=0.051774`。
  - 同被试平均响应距离 / 异被试平均响应距离 `0.710302`，说明同一被试内部的响应描述符整体更接近。
  - 但 rolling history 在 test 上只有 `v249_tail_rmse` 和 `v249_rmse` 明显改善，分别约 `+27.98%` 和 `+27.47%` relative RMSE improvement。
  - 对真实轨迹形状本身，多数目标没有改善甚至变差：`true_peak_abs`、`true_range`、`true_final_delta`、`v249_residual_mean` 等均未被历史风格稳定预测。
  - 二分类 rolling history AUC 较弱：`bad_top10 AUC=0.5630`、`bad_top10_vehicle_ambiguous AUC=0.5999`、`true_reverse_flag AUC=0.5152`。
  - `style_route_supported_now=false`。
  - `weak_style_signal_exists=true`。
  - `event_label_route_priority=true`。
- 当前判断：
  - 用户关于“同一场实验内多次事件是独立 trial，事件之间不应强行建立因果关联”的判断成立。
  - v297 不支持把驾驶风格作为单独主线；它更适合作为“这个被试是否容易被模型预测错”的风险/置信度校准信号。
  - 真正缺的信息更可能是事件级条件或响应类型标签，而不是同被试历史事件序列记忆。
  - 下一步应优先做事件级标签体系：先定义哪些标签在锚点前/实验设计中可知，哪些来自未来轨迹只能作辅助监督或 oracle。
- v296 备注：
  - v296 raw-sequence physiology embedding residual 实验在运行中被用户中断，仅留下部分 raw-sequence 特征文件；没有 guardrail/report/zip，不作为有效结论。

# 2026-07-02 v295 wait1 direct residual + physiology 已完成

- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v295_wait1_direct_residual_physio_20260702.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v295_wait1_direct_residual_physio_20260702`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v295_wait1_direct_residual_physio_20260702\reports\v295_wait1_direct_residual_physio_cn.md`
- ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v295_wait1_direct_residual_physio_20260702_pack.zip`
- 核心结果：
  - `guardrail_check.pass=True`。
  - `route_viable_now=false`，`goal_achieved_now=false`。
  - wait1 baseline 使用 v249 `delay_ms=1000` rolling prediction。
  - validation 选出的 physiology deployable 策略在 test bad_top10 仅 `-0.0011 RMSE`，test all 反而 `+0.0104 RMSE`。
  - 非生理 residual ablation 在 test bad_top10 为 `-0.0107 RMSE`，比生理策略更稳。
  - test-best diagnostic 生理策略可到 `-0.0249 RMSE`，但全样本 `+0.0192 RMSE`，且不是 validation 可部署策略。
- 判断：
  - post0_1 生理和 subject one-hot 没能稳定转成直接轨迹 residual 修正。
  - 生理信号可用于风险识别或边界分析，但当前不适合作为直接轨迹改善主输入。
# 2026-07-03 v302 侧倾诱因输入审计已完成（最新）

- 当前阶段：回应用户判断“车辆一开始侧倾的行为诱因应该作为输入”，完成 v302 因果输入审计。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v302_roll_cause_input_audit_20260703.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v302_roll_cause_input_audit_20260703`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v302_roll_cause_input_audit_20260703\reports\v302_roll_cause_input_audit_cn.md`
- 打包文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v302_roll_cause_input_audit_20260703\v302_roll_cause_input_audit_20260703.zip`
- 方法：
  - 不把 v301 的未来事件标签作为输入。
  - 只使用 v236 现有 anchor 前/anchor 当下/已知道路输入，显式构造 `roll-cause` summary。
  - 比较四类输入：`base_all_v236_preinput`、`raw_roll_cause_subset`、`engineered_roll_cause_summary`、`base_plus_engineered_roll_cause`。
  - 检查三件事：事件类型识别、差样本识别、v300 残差修正。
- 核心输入审计：
  - v236 preinput 总特征数 `609`。
  - 侧倾/横摆/转向/道路相关原始列 `392`。
  - roll 相关列 `95`，ay 相关列 `34`，yaw 相关列 `63`，steer 相关列 `33`。
  - 已包含 `current_roll_abs/current_roll_rate_abs/current_ay_abs/current_yaw_rate_abs`。
  - `steering/speed_kmh/ay/yaw_rate/roll/yaw/roll_rate/roll_acc/brake/lane_curvature/lateral_distance` 均有 `-3.0s~0.0s` 的 31 个历史点；已知道路 `road_curvature/road_lateral_distance` 有 `0.0s~2.0s` 的 21 个点。
- 事件类型识别结果（validation 选模型，再看 test）：
  - `base_all_v236_preinput`：test macro-F1 `0.2284`，balanced accuracy `0.3493`。
  - `raw_roll_cause_subset`：test macro-F1 `0.3037`，balanced accuracy `0.3756`。
  - `engineered_roll_cause_summary`：test macro-F1 `0.3540`，balanced accuracy `0.4261`。
  - `base_plus_engineered_roll_cause`：test macro-F1 `0.3906`，balanced accuracy `0.4400`。
- 差样本识别结果（within_bad_top10_by_v249，validation 选模型）：
  - `base_all_v236_preinput`：test AUC `0.5735`。
  - `raw_roll_cause_subset`：test AUC `0.5815`。
  - `engineered_roll_cause_summary`：test AUC `0.6354`。
  - `base_plus_engineered_roll_cause`：test AUC `0.6228`。
- v300 残差修正：
  - test/all 最好为 `engineered_roll_cause_summary::extra_trees_reg_d6::shrink1.0`：`0.519805 -> 0.510968`，delta `-0.00884`。
  - 但 test/within_bad_top10 没有改善；非零修正基本变差，最好非零结果约 `+0.00234` RMSE。
  - test/within_bad_top20 也没有稳定改善；非零修正多为轻微变差。
- 当前判断：
  - 用户判断成立：侧倾诱因应该作为输入，而且当前 v236/v300 输入里其实已经包含大量侧倾诱因。
  - 显式 roll-cause summary 能明显增强“事件类型识别”，说明这组信号确实有信息。
  - 但它还没有转化为 bad_top10 的轨迹修正收益；全样本小幅改善主要来自普通/较容易样本。
  - 下一步不应再争论“侧倾诱因能不能输入”，而应把它作为专门分支/辅助监督使用，并设置 no-harm 或 bad-focused 约束，防止全样本改善掩盖差样本变差。
- 验证：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `logs\guardrail_check.json` 中 `pass=True`。
  - ZIP 自检 `zip_testzip=True`。

---

# 2026-07-04 v307 coarse scene-label conditioned 曲线模型已完成（最新）

- 当前阶段：按用户确认的粗场景体系完成 v306/v307。v306 先把事件标签收敛为 5 类粗场景：`curve_downhill`（下坡过弯）、`curve_flat`（平路过弯）、`continuous_lane_change`（连续变道/连续左右修正）、`emergency_lane_change_instability`（紧急变道/猛打方向失稳）、`other_or_uncertain`（其他/不确定）；v307 再用这套 `coarse_scene_label` 替换 v304 的细 `event_primary_type` 条件输入重训 fixed-label conditioned 曲线模型。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v306_coarse_predefined_scene_label_table_20260704.py`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v307_coarse_scene_label_conditioned_curve_model_20260704.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v306_coarse_predefined_scene_label_table_20260704`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v307_coarse_scene_label_conditioned_curve_model_20260704`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v306_coarse_predefined_scene_label_table_20260704\reports\v306_coarse_predefined_scene_label_table_cn.md`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v307_coarse_scene_label_conditioned_curve_model_20260704\reports\v307_coarse_scene_label_conditioned_curve_model_cn.md`
- 打包文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v306_coarse_predefined_scene_label_table_20260704\v306_coarse_predefined_scene_label_table_20260704.zip`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v307_coarse_scene_label_conditioned_curve_model_20260704\v307_coarse_scene_label_conditioned_curve_model_20260704.zip`
- v306 标签结果：
  - `event_n=1167`，`coarse_scene_class_n=5`。
  - `curve_downhill`：`277/1167=23.7%`。
  - `curve_flat`：`142/1167=12.2%`。
  - `continuous_lane_change`：`414/1167=35.5%`。
  - `emergency_lane_change_instability`：`115/1167=9.9%`。
  - `other_or_uncertain`：`219/1167=18.8%`。
  - 过弯标签来自当前 rolling manifest 的 `scene_type`，直道内连续/紧急子类仍部分使用 v305/v301 自动 seed。
- v307 训练结果：
  - `guardrail.pass=True`。
  - 使用设备：`cuda`。
  - `event_n=1167`，`rolling_sample_n=7002`，`roll_cause_feature_n=301`。
  - validation-only 选中：`v307_coarse_scene_init_aux003_film005_h64`。
  - validation/all：v300 `0.534133` -> v307 `0.511257`，delta `-0.022876`。
  - validation/within_bad_top10：v300 `1.120367` -> v307 `0.995741`，delta `-0.124626`。
  - validation/within_bad_top20：v300 `0.836189` -> v307 `0.769417`，delta `-0.066772`。
  - test/all：v300 `0.519805` -> v307 `0.496138`。
  - test/within_bad_top10：v300 `0.859987` -> v307 `0.777797`。
  - test/within_bad_top20：v300 `0.690942` -> v307 `0.639121`。
  - test/strong_steer：v300 `0.621347` -> v307 `0.594050`。
  - test/vehicle_ambiguous：v300 `0.525913` -> v307 `0.504829`。
  - test/normal_predictable：v300 `0.391930` -> v307 `0.373519`。
- 对 v304 的关键比较：
  - v304 test/all `0.498102`，v307 `0.496138`，v307 略好。
  - v304 test/within_bad_top10 `0.832204`，v307 `0.777797`，v307 明显更好。
  - v304 test/within_bad_top20 `0.657669`，v307 `0.639121`，v307 更好。
  - 说明粗场景标签没有损失主要条件信息，反而可能比“急左/急右/复合制动”等细 future-derived 标签更贴近当前任务。
- 当前判断：
  - 用户提出的粗场景体系是有效方向：过弯、连续变道、紧急变道失稳这一级标签已经足以给曲线模型提供有用条件信息。
  - 但 v307 仍不能写成最终部署模型，因为 `continuous_lane_change` 与 `emergency_lane_change_instability` 的 seed 仍部分来自 v305/v301 自动标签，需要人工或实验条件确认。
  - 下一步应优先人工复核 v306 high priority 的 `529` 个连续/紧急直道子类，而不是继续扩更多细事件类别。
- 验证：
  - `python -m py_compile` 通过。
  - v306/v307 完整脚本运行完成。
  - v306 `logs\guardrail_check.json` 中 `pass=True`，ZIP 自检 `zip_testzip=True`。
  - v307 `logs\guardrail_check.json` 中 `pass=True`，`candidate_selection_uses_test=false`，`uses_test_error_as_features=false`，ZIP 自检 `zip_testzip=True`。

---

# 2026-07-04 v305 formal predefined event label table 已完成（最新）

- 当前阶段：用户明确“事件可以提前定好，相当于给每个事件打标签”。本轮没有继续盲目训练模型，而是先把这个任务边界落成正式事件标签表，区分“可作为预测前条件输入的主事件类型”和“更像未来过程形状的诊断标签”。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v305_formal_predefined_event_label_table_20260704.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v305_formal_predefined_event_label_table_20260704`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v305_formal_predefined_event_label_table_20260704\reports\v305_formal_predefined_event_label_table_cn.md`
- 打包文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v305_formal_predefined_event_label_table_20260704\v305_formal_predefined_event_label_table_20260704.zip`
- 方法：
  - 从 v301 `v301_event_type_labels.csv` 生成 formal label seed。
  - 主标签收敛为 7 类：`普通/轻微/不确定`、`急停/强减速`、`急左转`、`急右转`、`连续变道/横向避让`、`紧急避让/连续变道`、`复合制动转向`。
  - 将 `晚响应`、`多段修正`、`快速转向`、`高横摆/高横向加速度` 等放入 `formal_secondary_tags`，默认不作为直接输入。
  - 生成 `manual_review_seed_pack.csv`，按高误差、低置信、原 v301 需复核、原标签为未来形状类标签等条件排序。
- 核心结果：
  - `event_n=1167`。
  - `formal_primary_class_n=7`。
  - 主标签分布：
    - `普通/轻微/不确定`：`697/1167=59.7%`。
    - `连续变道/横向避让`：`175/1167=15.0%`。
    - `急停/强减速`：`80/1167=6.9%`。
    - `复合制动转向`：`59/1167=5.1%`。
    - `急左转`：`56/1167=4.8%`。
    - `紧急避让/连续变道`：`54/1167=4.6%`。
    - `急右转`：`46/1167=3.9%`。
  - high priority 人工审核事件数 `869`，medium priority `161`。
- 当前判断：
  - 如果事件主类型确实能在预测前由人工、实验条件、感知/规划模块确定，则 `formal_primary_type` 可以作为合法模型输入。
  - 当前 v305 表仍由 v301 future-derived 自动标签生成，所以它是人工审核 seed，不是最终人工标签。
  - 大量 high priority 不是坏事，而是在提醒：v301 中“多段修正/晚响应”这类未来形状标签占比很高，必须先被人工重新归入可提前定义的主事件类型。
  - 下一步应使用 `v305_manual_review_seed_pack.csv` 做人工复核；人工确认后，用 `v305_formal_event_labels.csv` 替换 v304 的 v301 标签输入，再重跑 fixed event-label conditioned 模型。
- 验证：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `logs\guardrail_check.json` 中 `pass=True`。
  - `task_allows_predefined_event_label_input=true`。
  - `formal_primary_type_can_be_model_input_after_confirmation=true`。
  - `diagnostic_tags_as_direct_input_allowed=false`。
  - ZIP 自检 `zip_testzip=True`。

---

# 2026-07-03 v304 fixed event-label conditioned 曲线模型已完成（最新）

- 当前阶段：按用户提出的“训练前完全确定每个样本是什么事件”完成 v304。该实验把固定事件类型做成 `event embedding`，直接作为曲线预测条件输入，用来检验“事件标签已知”是否能明显弥补锚点前信息不足。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v304_fixed_event_label_conditioned_curve_model_20260703.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v304_fixed_event_label_conditioned_curve_model_20260703`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v304_fixed_event_label_conditioned_curve_model_20260703\reports\v304_fixed_event_label_conditioned_curve_model_cn.md`
- 打包文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v304_fixed_event_label_conditioned_curve_model_20260703\v304_fixed_event_label_conditioned_curve_model_20260703.zip`
- 方法：
  - 输出不变：仍预测 21 点 `steering_delta` 曲线。
  - 主干沿用 v303/v300：`hist / road / phase / point_seq`。
  - 继续使用 v302 的 `roll-cause summary`。
  - 新增固定事件标签输入：`event_primary_type -> event embedding -> token 拼接 + FiLM 调制`。
  - v301 事件标签仍保留辅助头训练。
  - 从 `v300_full_joint_h64_no_subject` 初始化主干权重，新增 roll/event 输入块初始置零。
- 关键边界：
  - 当前固定事件标签来自 v301 `future_behavior_auto_draft`，即由未来真实行为自动派生。
  - 因此 v304 是 known-label / oracle upper-bound 实验，不是无条件可部署模型。
  - 只有当事件标签由人工审核、实验条件或预测前可知的外部系统提供时，才能把该标签当作正式输入。
- validation-only 选模：
  - 选中模型：`v304_fixed_event_init_aux005_film010_h64`。
  - validation/all：v300 `0.534133` -> v304 `0.520151`，delta `-0.013982`。
  - validation/within_bad_top10：v300 `1.120367` -> v304 `1.060815`，delta `-0.059552`。
  - validation/within_bad_top20：v300 `0.836189` -> v304 `0.788535`，delta `-0.047654`。
- test delay0 结果：
  - test/all：v300 `0.519805` -> v304 `0.498102`，delta `-0.021703`。
  - test/within_bad_top10：v300 `0.859987` -> v304 `0.832204`，delta `-0.027783`。
  - test/within_bad_top20：v300 `0.690942` -> v304 `0.657669`，delta `-0.033273`。
  - test/strong_steer：v300 `0.621347` -> v304 `0.595683`，delta `-0.025664`。
  - test/vehicle_ambiguous：v300 `0.525913` -> v304 `0.505271`，delta `-0.020642`。
  - test/normal_predictable：v300 `0.391930` -> v304 `0.376000`，delta `-0.015930`。
- 与 v303 的关系：
  - v303 test/all 为 `0.513617`，v304 降到 `0.498102`。
  - v303 test/within_bad_top10 为 `0.843876`，v304 降到 `0.832204`。
  - v303 test/within_bad_top20 为 `0.669646`，v304 降到 `0.657669`。
  - 说明“事件类型已知”确实有额外上限价值，但幅度仍不是压倒性。
- 当前判断：
  - 用户提出的固定事件标签方向值得保留：它比 v303 更明显改善全样本和差样本。
  - 但当前 v304 不能作为正式部署结果，因为标签源来自未来行为自动草稿。
  - 下一步应优先建立人工/实验条件可知的事件标签体系；若标签能在预测前确定，则 v304 结构可作为正式条件模型。
  - 若标签不能预测前确定，则 v304 只能作为 upper-bound，用来指导 mixture-of-experts 或多模态轨迹模型。
- 验证：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `logs\guardrail_check.json` 中 `pass=True`。
  - `uses_fixed_event_labels_as_features=true`。
  - `fixed_event_label_deployable_without_external_or_manual_label=false`。
  - `candidate_selection_uses_test=false`。
  - ZIP 自检 `zip_testzip=True`。

---

# 2026-07-03 v303 roll-cause 辅助监督多任务曲线模型已完成（最新）

- 当前阶段：在已经确定输出仍为 `21_point_steering_delta_curve` 后，正式开始改模型结构。v303 不再做删除样本、轻量 residual 或旧 gate，而是在 v300 joint-curve backbone 上加入 roll-cause 专门分支与事件类型辅助监督。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v303_roll_aux_multitask_curve_model_20260703.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v303_roll_aux_multitask_curve_model_20260703`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v303_roll_aux_multitask_curve_model_20260703\reports\v303_roll_aux_multitask_curve_model_cn.md`
- 打包文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v303_roll_aux_multitask_curve_model_20260703\v303_roll_aux_multitask_curve_model_20260703.zip`
- 方法：
  - 输出不变：仍预测 21 点 `steering_delta` 曲线。
  - 输入保留 v300 原有 `hist / road / phase / point_seq`，新增 v302 显式 `roll-cause summary`，共 `301` 个 roll-cause 特征。
  - v301 事件类型标签只作为训练期辅助监督，不作为推理输入。
  - 模型结构为 v300/v242 joint curve decoder backbone + roll encoder MLP + event auxiliary head + roll FiLM modulation。
  - 从 `v300_full_joint_h64_no_subject` 初始化主干权重；新增 roll 分支初始影响置小，避免从零训练破坏已有曲线能力。
- validation-only 选模：
  - 选中模型：`v303_roll_init_aux003_film005_h64`。
  - 选择规则：只用 validation 原始剩余样本与 delay0 no-harm gate，不使用 test 误差选模型。
  - validation/all：v300 `0.534133` -> v303 `0.526681`，delta `-0.007452`。
  - validation/within_bad_top10：v300 `1.120367` -> v303 `1.057268`，delta `-0.063099`。
  - validation/within_bad_top20：v300 `0.836189` -> v303 `0.802743`，delta `-0.033447`。
- test delay0 结果：
  - test/all：v300 `0.519805` -> v303 `0.513617`，delta `-0.006188`。
  - test/within_bad_top10：v300 `0.859987` -> v303 `0.843876`，delta `-0.016111`。
  - test/within_bad_top20：v300 `0.690942` -> v303 `0.669646`，delta `-0.021296`。
  - test/strong_steer：v300 `0.621347` -> v303 `0.611574`，delta `-0.009774`。
  - test/vehicle_ambiguous：v300 `0.525913` -> v303 `0.518756`，delta `-0.007158`。
  - test/normal_predictable：v300 `0.391930` -> v303 `0.390394`，delta `-0.001535`。
- 事件辅助头：
  - test/all_rolling：accuracy `0.570402`，balanced accuracy `0.489049`，macro-F1 `0.395503`。
  - test/delay0_only：accuracy `0.538793`，balanced accuracy `0.448250`，macro-F1 `0.416327`。
  - 这说明 roll-cause 分支确实学到了一部分事件/响应类型结构，但它对轨迹误差的转化收益仍偏小。
- 当前判断：
  - v303 是当前阶段第一个“结构确实改变且不伤害 v300”的正向小基线。
  - 结果不是大突破：bad_top10 有改善，但幅度只有 `-0.0161 RMSE`；它证明方向成立，但还没有达到“差样本本质改善”。
  - 从零训练的 v303 候选会破坏全样本和差样本表现；v300 初始化 + 小幅 roll 调制是必要的。
  - 下一步如果继续，应在 v303 上发展 mixture-of-experts / 多模态不确定性输出 / bad-focused 专家，而不是回到删除样本、轻量 residual 或 test 后验 gate。
- 验证：
  - 完整脚本运行完成。
  - `logs\guardrail_check.json` 中 `pass=True`。
  - `model_structure_changed=true`。
  - `uses_future_event_labels_as_features=false`。
  - `uses_event_labels_as_auxiliary_targets=true`。
  - `candidate_selection_uses_test=false`。
  - ZIP 自检 `zip_testzip=True`。

---
# 2026-07-03 v301 事件类型多分类标签草稿与有效性审计已完成（最新）

- 当前阶段：按用户新方向完成“给每个样本标定事件类型，再看多分类是否有用”的第一版系统审计。
- 新增脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v301_event_type_multiclass_label_audit_20260703.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v301_event_type_multiclass_label_audit_20260703`
- 中文报告：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v301_event_type_multiclass_label_audit_20260703\reports\v301_event_type_multiclass_label_audit_cn.md`
- 打包文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v301_event_type_multiclass_label_audit_20260703\v301_event_type_multiclass_label_audit_20260703.zip`
- 方法：
  - 使用 v236/v299/v300 的 delay0 事件，共 `1167` 个事件。
  - 从原始车辆 CSV 和真实未来轨迹提取 anchor 后 `0-2s` 行为特征，生成自动事件类型草稿。
  - 标签包括：`复合急制动转向`、`紧急连续变道/避让`、`强减速/急停`、`连续变道/横向避让`、`急左转`、`急右转`、`多段修正`、`晚响应/长事件`、`普通/轻微`。
  - 明确标记：这些标签来自未来真实行为，当前不能作为部署时直接输入，只能用于人工复核草稿、分层评估、辅助监督或理论上限诊断。
- 核心结果：
  - `guardrail.pass=True`，`zip_testzip=True`。
  - `event_n=1167`，`event_type_n=9`。
  - `label_source_level=future_behavior_auto_draft`。
  - `labels_deployable_as_direct_input_now=false`。
  - `manual_review_required_before_model_input=true`。
  - 自动标签分布中 `多段修正`最多：全量 `660/1167=56.6%`；其次为 `连续变道/横向避让`：`174/1167=14.9%`；`强减速/急停`为 `101/1167=8.7%`。
  - test 上 v300 误差最高的类型：`复合急制动转向` RMSE `1.021`，`急左转` RMSE `0.972`，`急右转` RMSE `0.711`，说明事件标签能解释一部分“哪些样本难”。
  - 但从锚点前输入预测事件类型仍较弱：validation 选出的 `extra_trees_d6` 在 test 上 `accuracy=0.341`，`balanced_accuracy=0.349`，`macro_f1=0.228`。
  - 标签残差修正收益很小：predicted-label residual 在 test/all 从 `0.519805` 到 `0.518730`，只改善约 `0.0011 RMSE`；在 test/within_bad_top10 反而从 `0.859987` 变差到 `0.868272`。
- 当前判断：
  - 事件类型标签有保留价值，主要价值是“分层诊断 + 人工复核 + 辅助监督”，不是直接把未来标签拼进输入。
  - 这一步进一步支持此前判断：锚点前车辆输入对未来强分叉行为的信息仍不足；简单多分类不能单独解决差样本。
  - 下一步应做小规模人工复核：优先复核 v301 manual review pack 中高误差、复合标签、低置信标签样本，确认自动类型定义是否符合驾驶语义。
  - 如果继续进模型，建议把事件类型做成辅助任务或混合专家的训练约束，而不是把未来派生标签作为正式输入。
- 验证：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `logs\guardrail_check.json` 中 `pass=True`。
  - ZIP 自检 `zip_testzip=True`。

---
# 最新状态指针：2026-07-04 已完成第317版二阶段候选门控校正实验。第317版按目标固定第315版保留清单，训练/验证/测试事件为 `650/211/222`，隔离 `84` 个事件不参与训练、验证选模或测试主统计；基础预测使用第316版，候选库共 `20` 条，包含原预测、幅值缩放、时间平移、幅值加时间组合和训练集残差原型。守卫通过：不使用测试集选模，不使用测试误差作特征，不把锚点后真实曲线作为输入，压缩包自检通过。验证结论为失败，因固定方案 `随机森林-候选单选` 未通过验证门槛，故按目标不报告测试集结果。关键验证数值：第316版验证全部样本 `0.531658`，候选最优上限 `0.375611`，固定门控方案 `0.586667`；困难前20为第316版 `0.810938`、候选最优 `0.625778`、固定门控 `0.831892`；困难前10为第316版 `1.002263`、候选最优 `0.766937`、固定门控 `1.033731`。当前判断：候选库覆盖有价值，但门控选择失败且过度修改普通样本；下一步不应扩大候选库优先，而应先做保守门控、原预测优先约束和候选选择损失修正。
---

# 最新状态指针：2026-07-04 已完成本地高级模型第317版修正方案咨询。本轮未训练新模型，而是把第316版后的卡点整理为外部评审问题并在本机问答软件中提交，回答已归档到 `03_baselines/gptpro_reviews/20260704_phase317_response.md`，决策记录在 `20260704_phase317_decision.md`，行动项在 `20260704_phase317_action_items.md`。外部建议与本地证据一致：不要继续重复“过滤后重训”，也不要只加大困难样本权重；下一步应做第316版后的轻量二阶段候选校正器，用锚点前车辆信号和第316版预测摘要构建门控，候选显式覆盖幅值缩放、时间平移和残差原型，同时用验证集整体无伤、普通样本无伤、强方向盘改善、困难样本改善四类门槛约束。该建议是第317版设计输入，不是已完成训练结果。
---
# 最新状态指针：2026-07-05 已完成本地高级模型第318版修正方案咨询。第317版二阶段候选门控验证失败后，已继续向本地高级模型询问下一步修正路线，并将回复清理归档到 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_response.md`。外部建议与本地判断一致：第317版失败点不是候选库没有价值，而是门控把普通样本强制改坏；第318版应固定为保守两段式门控，默认保持第316版原预测，先判断是否值得校正，再选择安全且收益高的候选，最后用小幅残差融合而非全量替换。当前下一步不是报告测试集，也不是扩候选库，而是实现第318版三条递进验证线：可校正门控、可校正门控加候选收益、可校正门控加候选收益加小幅残差融合。
---

# 最新状态指针：2026-07-05 已完成第321版困难样本可视化图册。该轮不训练新模型，不改变第320版结果，只重建第320版测试曲线并把测试困难前20的 `46` 个样本全部画成可浏览图册。结果显示：困难前20中第320版只修正 `6` 个，未修正 `40` 个；实际平均收益为 `-0.001521`，候选最优上限平均收益仍有 `0.223585`。分组上，`34` 个属于“候选有空间但门控未抓住”，`3` 个属于“修正后变坏”，`2` 个属于“修正后变好”。当前判断：第320版方向不能作为最终方案继续硬推；问题不是候选库完全行不通，而是门控选择和风险预算没有稳定学会哪些快速方向盘样本该改、该怎么改。下一步若继续模型线，应针对图册中的门控未抓住和修正变坏样本重做困难样本选择规则，而不是继续微调第320版阈值。
---
