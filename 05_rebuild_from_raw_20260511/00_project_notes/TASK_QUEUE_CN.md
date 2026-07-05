# 最新任务队列指针：2026-07-05 已完成 v318-v320 候选门控修正线。第318版保守两段式门控验证失败，三条方案均全不改；第319版加入困难代理和双通道配额后仍全不改；第320版按本地高级模型建议改成排序配额修复门控，新增候选正收益概率分类器，并将非强方向盘困难代理通道坏风险上限收紧到 `0.62` 及以下。v320 验证预算通过并已报告测试集：validation 全部样本收益 `0.001961`，强方向盘收益 `0.003729`，困难前20收益 `0.005495`，普通样本不动；test 全部样本收益 `0.000311`，强方向盘收益 `0.000613`，普通样本不动，但困难前20 `-0.001521`、困难前10 `-0.005689`。下一步优先：1）读取 v320 测试困难前20/前10被改坏事件，做候选家族和通道贡献诊断；2）把第二段候选选择从点式收益回归进一步改为候选排序/分候选族风险预算；3）继续保持普通样本低覆盖或不改，避免回到第317版普通样本误伤；4）不要把第320版写成最终模型，只能写成“门控激活机制已打通但困难组泛化仍需修正”。

---

# 最新任务队列指针：2026-07-04 已完成 v316 filtered current-window coarse-scene train。v316 按 v315 保留清单完整重训当前窗口模型，结构沿用 v307，完整训练 3 个候选并只用过滤后 validation 选模。选中 `v316_filtered_scene_init_aux003_film005_h64`，validation no-harm 通过；但过滤后 test 上 v316 未超过旧 v307：all 为 v300 `0.525580`、旧 v307 `0.496950`、v316 `0.502633`；bad10 为 v300 `0.859987`、旧 v307 `0.777797`、v316 `0.800171`；bad20 为 v300 `0.703038`、旧 v307 `0.651121`、v316 `0.660814`。保留 severe 33 个上 v316 `0.886424`，旧 v307 `0.877334`，v300 `0.805638`。结论：只过滤来源可疑样本再重训不是新主线；v315 清理边界仍有价值，但后续应转向保留 severe 样本的幅值、相位、极端峰值跟随修正，并另开重锚定候选重切窗口线。

---

# 最新任务队列指针：2026-07-04 已完成 v315 rapid steering filter / reanchor plan。v315 将 v314 的方向盘快转来源审计转成训练前数据处理策略：全量 `1167` 个 delay0 事件中，当前窗口训练保留 `1083`，隔离 `84`；其中重锚定候选 `77`，全程快转证据弱候选剔除 `7`。按划分：train 保留 `650/702`，val 保留 `211/233`，test 保留 `222/232`。用户截图 #020 被隔离为“当前平缓但后续才快转”，候选后移锚点约 `5.02s`，新 observation_s 候选 `278.52`。下一步优先：1）基于 `v315_current_window_keep_manifest.csv` 重跑当前窗口条件模型，检验剔除来源可疑样本后 severe 和整体是否改善；2）对 `v315_reanchor_candidate_manifest.csv` 单独重切窗口和目标曲线；3）保留清单中的 severe 再做幅值、相位、极端动作跟随修正。

---

# 最新任务队列指针：2026-07-04 已完成 v314 rapid steering source sample audit。用户明确改为抽样排查，并强调样本必须由方向盘快速转动引起。v314 用原始方向盘角速度审计 1167 个 delay0 事件，确认 1083 个当前 0-2s 有快转来源证据，84 个当前窗口快转证据不足或来源错位；v309 severe 37 个里只有 4 个来源可疑，用户截图 5 个里只有 #020 来源可疑。下一步优先：1）基于 `suspect_not_current_fast_steer=True` 的 84 个事件生成过滤/重锚定候选训练表；2）把 #020 等“当前平缓、后续才快转”样本从当前窗口强动作训练约束中隔离；3）对来源成立但仍预测差的 severe 样本，转入幅值、相位、极端动作跟随不足修正；4）不继续推进人工逐个复核图册线。

---

# 最新任务队列指针：2026-07-04 已完成 v312 horizon-aligned label / anchor audit。下一步优先顺序：1）让用户优先复核 `v312_v309_severe_horizon_label_overlay.csv` 中 `split_local_flat_and_late_context` 与 `split_current_and_late_direction` 两类，尤其 `#020/#014/#023`；2）生成 confirmed horizon-aligned label 表，把 `local_0_2_motion_label` 与 `late_2_6_context_label` 分开保存；3）若要继续训练，v313 应避免把未来真实 local 标签直接作为部署输入，优先做可部署规则/人工确认标签或 prediction fallback gate 的 validation-only 实验。

---

# 最新任务队列指针：2026-07-04 已完成 v310/v311 差样本定向修改与窗口错位审计。v310 的 hard-case loss/权重只能让常规 test/all 小幅改善，但没有改善 v309 severe 37 个和用户截图 5 个，因此不应作为“严重错已修复”的主线。v311 证明 severe 中有 `11/37` 个存在 0-2s 预测窗口与 2-6s 后续事件不一致的嫌疑。下一步优先任务：做 v312 horizon-aligned coarse label / anchor audit，把标签从“整段后续事件类型”改成“模型预测窗口 0-2s 内实际要预测的局部动作状态”，同时保留 2-6s 后续事件作为单独的 late-event/context 标签，不再让后续动作标签直接驱动 0-2s 预测。

---

# 最新任务队列指针：2026-07-04 已完成 v309 严重方向/意图错误筛选。基于用户截图和 v309 清单，确认截图中的 `5` 个事件为 `#014/#017/#019/#020/#023`；全体 test delay0 `232` 个事件中筛出 `37` 个严重候选。下一步优先人工复核 `v309_severe_direction_or_intent_errors.csv`，把 `opposite_peak_direction`、`false_large_maneuver`、`missed_extreme_amplitude` 分别作为硬错误标签，决定是否回灌到 v306/v307 的 confirmed coarse-scene 标签，或在下一轮训练中加入 hard-case loss/约束。

---

# 最新任务队列指针：2026-07-02 已完成 v300 within-subject full joint-curve retrain。当前结论：v300 已经真正从原始 rolling 输入完整重训，不再固定旧 v249 预测；防泄漏通过，`event_in_multiple_splits_n=0`，`train/val/test event=702/233/232`。validation 选中 `v300_full_joint_h64_no_subject`，两个 `subject_onehot` 候选均未胜出。delay0 test/all RMSE `0.5198`，整体不如旧 v249 诊断参照 `0.3246`；但 within_bad_top10 RMSE `0.8600`，优于旧 v249 诊断参照 `1.0383`。下一步优先分析并修复强响应幅值压缩和极端曲线跟随不足；不建议回到 v222a gate、删除样本、轻量 residual 或单纯 subject id 拼接。

---

# 最新更新：2026-07-04 v308 coarse scene 视觉人工复核包

## 已完成任务

- v308 `coarse_scene_visual_manual_review`：
  - 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v308_coarse_scene_visual_manual_review_20260704.py`
  - 输出：`05_rebuild_from_raw_20260511/03_baselines/v308_coarse_scene_visual_manual_review_20260704`
  - HTML 图册：`05_rebuild_from_raw_20260511/03_baselines/v308_coarse_scene_visual_manual_review_20260704/index.html`
  - 报告：`05_rebuild_from_raw_20260511/03_baselines/v308_coarse_scene_visual_manual_review_20260704/reports/v308_coarse_scene_visual_manual_review_cn.md`
  - ZIP：`05_rebuild_from_raw_20260511/03_baselines/v308_coarse_scene_visual_manual_review_20260704/v308_coarse_scene_visual_manual_review_20260704.zip`
- 结果摘要：
  - 已将 v306 high + medium 复核队列改成图像复核包。
  - 共生成 `748` 张逐事件曲线图，其中 high priority `529`，medium priority `219`。
  - 候选标签分布：`continuous_lane_change=414`，`emergency_lane_change_instability=115`，`other_or_uncertain=219`。
  - 图册支持浏览器筛选、逐事件填写复核结论和人工标签，并导出 CSV。
  - ZIP 自检通过：`testzip=None`，压缩包内 `png_n=748`。

## 当前建议队列

- 首选：用户先打开 v308 `index.html` 看图复核，优先处理 high priority 的连续/紧急直道子类。
- 第二步：将浏览器导出的 `v308_manual_review_decisions.csv` 接回项目，生成 confirmed coarse-scene 标签表。
- 第三步：用 confirmed 标签重跑 v307 结构，形成不依赖自动 future seed 的 confirmed coarse-scene 条件模型。

## 禁止任务

- 不把 v308 图册中的锚点后真实响应当作模型预测前输入。
- 不把用户尚未复核的 v306 直道连续/紧急 seed 写成最终人工标签。
- 不用 `other_or_uncertain` 强行凑成某个明确事件类别。

---

# 最新更新：2026-07-04 v309 近期最好模型预测效果图册

## 已完成任务

- v309 `recent_best_prediction_effect_gallery`：
  - 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v309_recent_best_prediction_effect_gallery_20260704.py`
  - 输出：`05_rebuild_from_raw_20260511/03_baselines/v309_recent_best_prediction_effect_gallery_20260704`
  - HTML 图册：`05_rebuild_from_raw_20260511/03_baselines/v309_recent_best_prediction_effect_gallery_20260704/index.html`
  - 报告：`05_rebuild_from_raw_20260511/03_baselines/v309_recent_best_prediction_effect_gallery_20260704/reports/v309_recent_best_prediction_effect_gallery_cn.md`
  - ZIP：`05_rebuild_from_raw_20260511/03_baselines/v309_recent_best_prediction_effect_gallery_20260704/v309_recent_best_prediction_effect_gallery_20260704.zip`
- 结果摘要：
  - 近期最好版本按当前记录为 `v307_coarse_scene_init_aux003_film005_h64`。
  - test delay0 共 `232` 个事件，图册选取 `54` 个代表性样本。
  - test/all：v300 `0.519805` -> v307 `0.496138`。
  - 图中蓝线为 v307，橙线为 v300 参照，黑线为真实 0-2s。
  - 2s 后灰底区域只显示真实后续到 +6s，不是模型预测范围。

## 当前建议队列

- 首选：先用 v309 `index.html` 看近期最好模型在好样本、中位样本、最差样本、bad_top10、相对 v300 改善/退化样本上的真实表现。
- 第二步：若发现主要失败集中在连续变道或紧急变道失稳，再回到 v308 图册做人工确认标签。
- 第三步：用人工确认后的标签重跑 confirmed coarse-scene 条件模型。

## 禁止任务

- 不把 v309 中 +2s 之后的真实后续误解为模型预测输出。
- 不把 v307 写成最终可部署模型；其直道连续/紧急标签仍有自动 seed 成分。

---

# 最新任务队列指针：2026-07-02 已完成 v299 within-subject split residual calibration。当前结论：同被试内切分后，轻量 subject-aware residual 校准出现明确正向信号；但由于本轮固定旧 v249 预测，正式结论必须完整重训。下一步优先 v300：在同一 within-subject split 上完整重训/复现主模型，确保同一样本不跨 split 且 base model 不接触新 test 样本。

---

# 最新更新：2026-07-02 v299 within-subject split residual calibration

## 正在做的任务

- 生理数据 goal 仍未完成。
- 当前路线出现一个新的可行方向：允许同一被试的不同样本分布在 train/val/test 后，模型可以利用驾驶员历史/个体校准。
- 但 v299 只是固定 v249 预测上的快速 residual 校准，不是完整重训结论。

## 已完成任务

- v299 `within_subject_split_residual_calibration`：
  - 脚本：`03_baselines/scripts/stage03_v299_within_subject_split_residual_calibration_20260702.py`
  - 输出：`03_baselines/v299_within_subject_split_residual_calibration_20260702`
  - 报告：`03_baselines/v299_within_subject_split_residual_calibration_20260702/reports/v299_within_subject_split_residual_calibration_cn.md`
  - ZIP：`03_baselines/v299_within_subject_split_residual_calibration_20260702_pack.zip`
- split guardrail：
  - `event_n=1167`
  - `unique_event_n=1167`
  - `duplicate_event_uid_n=0`
  - `event_in_multiple_splits_n=0`
  - `subject_n=18`
  - `subject_with_all_three_splits_n=18`
  - `train_n=702`
  - `val_n=233`
  - `test_n=232`
- 关键结果：
  - `chosen_method=base_curve_meta_subject__extra_trees_d5`
  - `chosen_test_all_delta=-0.006707`
  - `chosen_test_within_bad_top10_delta=-0.073816`
  - `chosen_test_within_bad_top10_rmse=0.964519`
  - `chosen_test_within_bad_top10_baseline_rmse=1.038335`
  - `within_subject_residual_route_promising=True`
  - `complete_model_retrain_recommended_next=True`
- 关键边界：
  - `full_v249_retrained=False`
  - `fixed_v249_predictions_have_original_split_exposure=True`
  - 新 within-test 中 `within_test_original_v249_train_rate=0.581897`
  - 因此不能把 v299 写成正式模型结果，只能写成完整重训前的潜力审计。

## 下一步候选任务

1. 做 v300 within-subject full retrain：
   - 固定 v299 的 split 表；
   - 重新训练当前主模型或可快速复现的 v249/v241 等价模型；
   - 训练过程只看 within-train，选择只看 within-val，最终只报 within-test；
   - 继续保证同一 `event_uid` 不跨 split。
2. 如果完整重训耗时太长，先做 v300-light：
   - 使用同一 split；
   - 从车辆输入特征直接训练一个轻量 baseline；
   - 和 v299 residual calibration 对齐比较。
3. 若 v300 仍能稳定改善 bad_top10，才把“同驾驶员先验/驾驶习惯学习”作为正式任务边界。

## 禁止任务

- 不把 v299 当 formal final 结果。
- 不忽略旧 v249 训练暴露边界。
- 不让同一 event_uid 跨 train/val/test。
- 不用 test 选择 residual 模型或阈值。
- 不把 recording/session diagnostic 当作可部署主结果。

---

# 最新任务队列指针：2026-07-02 已完成 v298 event label explanatory audit。当前结论：事件/响应标签路线的“粗标签版本”不足以解决差样本轨迹修正；粗响应标签更像风险提示器。下一步不要直接训练 hard response classifier，也不要把 oracle 标签当输入；如继续，应先做当前事件级人工/实验条件标签定义与复核包。

---

# 最新更新：2026-07-02 v298 event label explanatory audit

## 正在做的任务

- 生理数据 goal 仍未完成。
- 当前已经确认：
  - 锚点前车辆信息不足。
  - 生理数据在原锚点前不足以补足分叉信息。
  - 驾驶风格/被试历史只有弱辅助。
  - 粗响应标签可以识别风险，但不能直接修正轨迹。
- 当前最关键的 blocker：没有足够覆盖、可部署、锚点前可知的事件级标签/实验条件标签。

## 已完成任务

- v298 `event_label_explanatory_audit`：
  - 脚本：`03_baselines/scripts/stage03_v298_event_label_explanatory_audit_20260702.py`
  - 输出：`03_baselines/v298_event_label_explanatory_audit_20260702`
  - 报告：`03_baselines/v298_event_label_explanatory_audit_20260702/reports/v298_event_label_explanatory_audit_cn.md`
  - ZIP：`03_baselines/v298_event_label_explanatory_audit_20260702_pack.zip`
- 关键结果：
  - `guardrail_check.pass=True`
  - `event_n=1167`
  - `best_oracle_response_risk_label=oracle_strength_label`
  - `best_oracle_response_risk_test_auc=0.773525`
  - `best_oracle_response_config=oracle_shape`
  - `best_oracle_response_test_badtop10_delta=-0.009262`
  - `best_oracle_response_test_all_delta=-0.001286`
  - `history_match_tol1_rate_all=0.227078`
  - `history_match_tol1_rate_test=0.282609`
  - `deployable_event_label_available_now=False`
  - `history_rule_label_coverage_sufficient_now=False`
  - `coarse_response_labels_are_risk_markers_not_correction_solution=True`
  - `future_derived_labels_used_as_inputs=False`
  - `goal_achieved_now=False`

## 解释

- `oracle_strength_label` 这类未来响应强度标签能较好识别哪些样本容易 bad，但它主要是风险标记。
- 即使直接知道未来粗响应标签，按标签均值修正 v249 残差，对 bad_top10 也只改善约 `0.009 RMSE`，不是本质改善。
- 历史规则标签和当前事件可通过时间近邻匹配一部分，但覆盖率只有约四分之一，不能作为当前全量训练标签。
- 因此下一步不能直接上 hard response classifier；标签必须更贴近事件条件/实验条件/触发机制，而不是只给未来曲线形状粗分类。

## 下一步候选任务

1. 做 v299 当前事件级人工/实验条件标签复核包：
   - 从 v298 casebook 中抽取 test bad_top10、vehicle_ambiguous、强响应但预测正常、预测差但弱响应等典型样本。
   - 给用户一个可审核的标签字典：事件触发类型、道路/场景条件、是否多阶段响应、是否长事件/持续控制、是否需要后续拆段。
2. 如果用户能提供或确认实验条件标签，再做 v300 label-predictability audit：
   - 只用锚点前可知标签/条件；
   - 检查标签对 bad_top10、v249_rmse、未来响应形状的解释力。
3. 只有 v300 通过后，才进入 auxiliary response head 或 soft mixture-of-experts。

## 禁止任务

- 不把 oracle_strength/shape/direction/timing 当作测试输入。
- 不因为 `oracle_strength_label` AUC 高就写成预测效果改善。
- 不做 hard response-type 级联。
- 不继续旧 v222a gate / 删除样本 / 轻量 residual 线。
- 不继续旧 physiology selector / reranker / prototype matching 线。

---

# 最新任务队列指针：2026-07-02 已完成 v297 subject style stability audit。当前结论：驾驶风格/被试历史只支持弱辅助，不支持作为主线；下一步优先做事件级标签、实验条件标签与响应类型辅助监督。v296 是中断半成品，不作为有效结果。

---

# 最新更新：2026-07-02 v297 subject style stability audit

## 正在做的任务

- 生理数据 goal 仍未完成。
- 当前已确认：
  - observation 前生理特征不能稳定弥补锚点前信息不足。
  - wait1 生理 residual 修正不稳定，不能作为当前可部署改进。
  - subject/style 有弱信号，但主要解释“这个被试/这类事件更容易被 v249 预测差”，不稳定解释未来轨迹形状。
  - 同一被试多次事件不能默认存在事件到事件因果连续性，只能作为稳定驾驶倾向/风险先验来审计。
- 下一步应转向事件级标签/实验条件标签，而不是继续堆生理、风格、候选轨迹 selector。

## 已完成任务

- v297 `subject_style_stability_audit`：
  - 脚本：`03_baselines/scripts/stage03_v297_subject_style_stability_audit_20260702.py`
  - 输出：`03_baselines/v297_subject_style_stability_audit_20260702`
  - 报告：`03_baselines/v297_subject_style_stability_audit_20260702/reports/v297_subject_style_stability_audit_cn.md`
  - ZIP：`03_baselines/v297_subject_style_stability_audit_20260702_pack.zip`
- 关键结果：
  - `guardrail_check.pass=True`
  - `event_n=1167`
  - `key_subject_eta_train_mean=0.05984`
  - `same_subject_mean_distance_ratio=0.71030`
  - `rolling_history_test_relative_rmse_improvement_mean_history3=0.06988`
  - `rolling_history_test_positive_target_rate_history3=0.28571`
  - `binary_history_test_auc_mean_history3=0.53109`
  - `style_route_supported_now=False`
  - `weak_style_signal_exists=True`
  - `event_label_route_priority=True`
  - `future_derived_oracle_labels_are_not_deployable_inputs=True`
- 解释：
  - 同被试样本确实比不同被试样本更相似，说明存在弱驾驶风格/响应倾向。
  - rolling history 对 v249 RMSE、tail RMSE 有一定解释力，但对真实轨迹形状、峰值、方向、尾部响应不稳定。
  - 驾驶风格适合作为 risk/uncertainty/context 辅助，不适合作为主预测器。

## 相关补充结果

- v295 `wait1_direct_residual_physio`：
  - 输出：`03_baselines/v295_wait1_direct_residual_physio_20260702`
  - 结论：可部署生理 residual 对 bad_top10 只有极弱改善，且全样本变差。
  - `best_physio_test_badtop10_delta=-0.00111`
  - `best_physio_test_all_delta=+0.01039`
  - `best_nonphysio_test_badtop10_delta=-0.01074`
  - `route_viable_now=False`
  - `goal_achieved_now=False`
- v296 `rawseq_physio_embedding_residual`：
  - 运行被用户中断。
  - 只产生部分中间表，不产生 guardrail/report/zip。
  - 不作为结论引用。

## 下一步候选任务

1. 建立事件级标签字典：严格区分“锚点前可知标签”和“未来派生 oracle/auxiliary 标签”。
2. 从 `v297_event_response_descriptors.csv` 抽取典型样本，让用户人工复核响应类型标签是否符合直觉。
3. 做 v298 label audit：检查标签对 bad_top10、响应形状、v249 误差的解释率。
4. 如果标签确实有用，再做 soft mixture-of-experts / auxiliary response-type head；不要回到硬分类级联。

## 禁止任务

- 不把 future-derived oracle labels 当 test 输入。
- 不把弱 subject/style 信号写成驾驶风格主线成功。
- 不继续旧 v222a gate / 删除样本 / 轻量 residual 线。
- 不继续旧 physiology selector / reranker / prototype matching 线。
- 不用 test 后验选择阈值、等待时间、标签或模型。

---

# 当前任务队列

> 最新指针：2026-07-02 已完成 v294 post-response candidate wait ranker。当前没有正在运行的训练进程。v294 把 v293 的 post-response 可见性转成 RMSE 候选选择任务：等待 1/2/3/5 秒后，用 post 生理响应匹配 v292 的 40 个 train prototype 候选。结果 `route_viable_now=false`：val no-harm active 策略存在，但 test bad_top10 反而 `+0.0070 RMSE`；test-best diagnostic 只有 `-0.0112 RMSE`，且 val bad_top10 `+0.1239`、val all `+0.0606`，不可部署。当前结论是：生理能帮助识别风险，但还不能稳定生成/选择更好轨迹；用户 goal 仍未达成。

---

# 最新更新：2026-07-02 v294 post-response candidate wait ranker

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：post-response 生理信号能识别 bad_top10 风险，但把它用于候选 prototype 选择后，仍没有得到可部署 RMSE 改善。
- 关键转折：问题不再是“差样本是否可见”，而是“看见风险后如何产生更好的轨迹预测”。

## 已完成任务

- v294 `post_response_candidate_wait_ranker`：
  - 脚本：`03_baselines/scripts/stage03_v294_post_response_candidate_wait_ranker_20260702.py`
  - 输出：`03_baselines/v294_post_response_candidate_wait_ranker_20260702`
  - 报告：`03_baselines/v294_post_response_candidate_wait_ranker_20260702/reports/v294_post_response_candidate_wait_ranker_cn.md`
  - ZIP：`03_baselines/v294_post_response_candidate_wait_ranker_20260702_pack.zip`
- 关键结果：
  - `guardrail_check.pass=True`。
  - `zip_testzip=True`，file_count `13`。
  - `route_viable_now=false`。
  - `event_n=1167`，`candidate_rows=46680`。
  - `wait_policy_n=4`，`selector_config_n=36`。
  - `uses_post_observation=True`，`post_features_are_wait_policy_only=True`。
  - `best_val_noharm_active_exists=True`，但 test 不改善。
  - val 选择策略：`wait1_post0_1__post_response_pair_top64__extra_trees_d6`。
  - val bad_top10 delta `-0.0057`，val all delta `+0.0011`。
  - test bad_top10 delta `+0.0070`，test all delta `+0.0017`。
  - test-best diagnostic：`wait2_post0_2__vehicle_post_response_pair_top96__ridge_a10`。
  - test-best bad_top10 delta `-0.0112`，bad_top10_vehicle_ambiguous delta `-0.0071`。
  - 但 test-best 的 val bad_top10 delta `+0.1239`，val all delta `+0.0606`，不可部署。
  - candidate-pool oracle test bad_top10 delta `-0.0784`。
  - vehicle_score_top1 test bad_top10 delta `+0.1453`。

## 下一步候选任务

- 不建议继续在“候选匹配/reranker/selector”上堆模型复杂度。
- 如果继续追求预测改善，下一步应转向“直接建模等待后的轨迹/残差”，而不是从旧 prototype 中选一个：
  - 用 wait1/wait2/wait3 后的早期真实车辆响应 + 生理响应，重新训练后半段轨迹预测器。
  - 或把任务改成 conditional residual / trajectory correction：先用 early response 判断方向，再直接修正曲线。
  - 与 latest、v241/v247/v278 统一比较 RMSE 和差样本 RMSE。
- 生理在下一步里适合作为 risk/state/context 输入，而不是作为 prototype matching 的主排序信号。

## 禁止任务

- 不把 v293 的 post AUC 写成预测改善。
- 不把 v294 的 test-best diagnostic `-0.0112` 写成部署结果。
- 不继续旧 v222a gate / 删除样本 / 轻量 residual 线。
- 不用 test 后验选择等待时间、候选、阈值或模型。

---

# 最新更新：2026-07-02 v293 physiology response visibility / latency audit

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：主差样本 `bad_top10` 在 observation 前生理信息不足，不能支撑原锚点即时预测。
- 重要新结论：observation 后 0-3 秒生理响应对 `bad_top10` 有明显区分度，下一步应转成“短等待/延迟观测/响应可见性”的部署收益验证。

## 已完成任务

- v293 `physio_response_visibility_latency_audit`：
  - 脚本：`03_baselines/scripts/stage03_v293_physio_response_visibility_latency_audit_20260702.py`
  - 输出：`03_baselines/v293_physio_response_visibility_latency_audit_20260702`
  - 报告：`03_baselines/v293_physio_response_visibility_latency_audit_20260702/reports/v293_physio_response_visibility_latency_audit_cn.md`
  - ZIP：`03_baselines/v293_physio_response_visibility_latency_audit_20260702_pack.zip`
- 关键结果：
  - `guardrail_check.pass=True`。
  - `zip_testzip=True`。
  - `event_n=1167`。
  - `feature_n=540`，`screen_feature_n=540`。
  - `feature_set_n=14`。
  - `ok_rate=0.91945`。
  - `uses_post_observation=True`，`post_features_are_diagnostic_only=True`。
  - `bad_top10` pre best test AUC `0.4896`。
  - `bad_top10` early-post best test AUC `0.7726`。
  - `bad_top10` `window_post0_3` test AUC `0.7254`。
  - `bad_top10` `window_post0_2` test AUC `0.7053`。
  - `bad_top10_vehicle_ambiguous` pre best test AUC `0.6012`，属于边缘弱信号。
  - `bad_top10_vehicle_ambiguous` early-post best test AUC `0.6627`。
  - `candidate_pool_gain_gt_005` pre/early-post/late-post best test AUC 分别为 `0.5722 / 0.5593 / 0.5808`。

## 下一步候选任务

- 做 v294：把 post0-3 秒生理响应和同窗口车辆早期响应一起转成“短等待策略”，严格比较：
  - 原锚点即时预测。
  - wait 1s / 2s / 3s 后预测。
  - 只用早期车辆响应。
  - 早期车辆响应 + 生理响应。
  - 只在高不确定/车辆歧义样本上等待。
- v294 必须给出等待代价和预测收益，而不是只给分类 AUC。
- v294 不能把 post 特征包装成原锚点可用信息；必须明确这是延迟观测策略。

## 禁止任务

- 不再继续旧 v222a gate / 删除样本 / 轻量 residual 线。
- 不把 v293 post AUC 写成原锚点即时预测能力。
- 不把 `bad_top10_vehicle_ambiguous` pre AUC `0.6012` 过度解释为主差样本已可预测。
- 不用 test 后验选择等待时间、特征块或部署阈值。

---

# 最新更新：2026-07-02 v292 source-physio pairwise candidate ranker

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：即使把任务改成更贴近核心矛盾的 pairwise candidate ranking，源生理仍不能稳定从车辆相似候选中选对未来。
- 重要新结论：候选池不是没有好候选；好候选存在，但生理无法可靠挑出来。

## 已完成任务

- v292 `source_physio_pairwise_candidate_ranker`：
  - 脚本：`03_baselines/scripts/stage03_v292_source_physio_pairwise_candidate_ranker_20260702.py`
  - 输出：`03_baselines/v292_source_physio_pairwise_candidate_ranker_20260702`
  - 报告：`03_baselines/v292_source_physio_pairwise_candidate_ranker_20260702/reports/v292_source_physio_pairwise_candidate_ranker_cn.md`
  - ZIP：`03_baselines/v292_source_physio_pairwise_candidate_ranker_20260702_pack.zip`
- 关键结果：
  - `guardrail_check.pass=True`。
  - `route_viable_now=false`。
  - `event_n=1167`。
  - `candidate_rows=46680`。
  - train/val/test event 数 `674/309/184`。
  - `prototype_train_only=true`。
  - `selector_config_n=15`。
  - vehicle score top1 在 test bad_top10 上差 `+0.1453 RMSE`。
  - candidate-pool oracle 在 test bad_top10 上改善 `-0.0784 RMSE`。
  - candidate-pool oracle 在 test bad_top10_vehicle_ambiguous 上改善 `-0.0881 RMSE`。
  - `best_val_noharm_active_exists=false`。
  - test-best diagnostic：`bio_all_top_pair_only__hgb_d3`，test bad_top10 delta `-0.0248`，但 val bad_top10 delta `+0.1402`，val all delta `+0.0367`，不可部署。
  - 有一个 no-harm 但非 active 的弱阈值，test bad_top10 delta `-0.0124`，只覆盖约 `1/19` 个 test bad_top10，不能作为路线。

## 下一步候选任务

- 不建议继续堆 physiology matching / pairwise reranker / selector。
- 如果继续围绕生理，需要转成“为什么无效”的证据闭环，而不是再换模型：
  - 生理响应是否发生在 observation 后，导致 observation 前不可见。
  - 生理信号与驾驶行为差异是否主要是 subject 内弱信号，而 subject-disjoint 下不可泛化。
  - 生理采集质量/同步是否使关键时间窗错位。
- 如果继续追求预测改善，应转回车辆侧任务定义：候选池已有 oracle 空间，重点是更好的车辆不确定性/多未来选择，而不是生理 tie-breaker。

## 禁止任务

- 不把 candidate-pool oracle 写成模型改善。
- 不把 test-best diagnostic 写成部署结果。
- 不继续用 test 后验挑 threshold、feature block 或 selector。
- 不再把“候选池里有好轨迹”误解成“生理能选到好轨迹”。

---

# 最新更新：2026-07-02 v291 multi-signal physiology supervised probe

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：即使合并 ECG/RESP/EDA 源信号并做监督 selector，仍无法通过 validation 选择出可部署的差样本改善策略。
- 现成方法池有小的事后上限，但生理无法稳定判断什么时候该覆盖 latest。

## 已完成任务

- v291 `multisignal_physio_supervised_probe`：
  - 脚本：`03_baselines/scripts/stage03_v291_multisignal_physio_supervised_probe_20260702.py`
  - 输出：`03_baselines/v291_multisignal_physio_supervised_probe_20260702`
  - 报告：`03_baselines/v291_multisignal_physio_supervised_probe_20260702/reports/v291_multisignal_physio_supervised_probe_cn.md`
  - ZIP：`03_baselines/v291_multisignal_physio_supervised_probe_20260702_pack.zip`
- 关键结果：
  - `guardrail_check.pass=True`。
  - `route_viable_now=false`。
  - `event_n=1167`，train/val/test 为 `674/309/184`。
  - `bio_source_feature_n=1660`。
  - `screen_feature_n=1404`。
  - `feature_block_n=7`，`selector_config_n=28`。
  - method-pool oracle 在 test bad_top10 上有 `-0.0402 RMSE` 上限，但这是非部署事后上限。
  - `best_val_noharm_active_exists=false`。
  - val 选择只能 fallback no override，test bad_top10 delta `0.0`。
  - test-best diagnostic 非部署选择器只有 `-0.0093 RMSE`，且 val no-harm 不成立。
  - 源生理识别 test bad_top10 的最好 AUC `0.5394`，没有达到可用分类信号。

## 下一步候选任务

- 不建议继续做 physiology selector / reranker / threshold / reliability filter。
- 如果继续生理方向，只建议做两个收口类工作：
  - 生理可观测性分层：说明哪些样本源生理完全不提供信息，哪些只有弱诊断信号。
  - 数据链复核：同步、佩戴质量、subject-disjoint 划分、生理记录是否与事件真实响应阶段错位。
- 如果目标是继续提升预测效果，应把主线转回车辆多未来分布、等待代价、预测不确定性和不可观测样本的任务定义；生理只保留为弱诊断或分层变量。

## 禁止任务

- 不把 v291 的 method-pool oracle 写成部署改善。
- 不把 test-best diagnostic `-0.0093` 写成有效路线。
- 不继续堆更复杂的生理 selector/reranker。
- 不用 test 后验结果选择阈值、特征块或部署策略。

---

# 最新更新：2026-07-02 v290 EDA/SCR usable-subset source route audit

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：EDA/SCR 源信号有较高可用覆盖，但全体样本和 EDA 可用子集都没有形成可部署 top1 改善。
- v288 ECG、v289 RESP、v290 EDA 三条源信号路线都没有把差样本从 latest baseline 之上拉下来。

## 已完成任务

- v290 `eda_scr_usable_subset_route_audit`：
  - 脚本：`03_baselines/scripts/stage03_v290_eda_scr_usable_subset_route_audit_20260702.py`
  - 输出：`03_baselines/v290_eda_scr_usable_subset_route_audit_20260702`
  - 报告：`03_baselines/v290_eda_scr_usable_subset_route_audit_20260702/reports/v290_eda_scr_usable_subset_route_audit_cn.md`
  - ZIP：`03_baselines/v290_eda_scr_usable_subset_route_audit_20260702_pack.zip`
- 关键结果：
  - `guardrail_check.pass=True`。
  - `route_viable_now=false`。
  - `eda_subset_route_viable_now=false`。
  - `eda_source_feature_n=473`。
  - `feature_set_n=29`。
  - EDA 可用事件数 `906/1167`，可用率 `0.77635`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1760`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta `+0.1601`。
  - test-best top1 diagnostic 未通过：最佳仍差 `+0.1409`。
  - test bad_top10 best corr `0.0306`。
  - EDA 可用子集也未通过：`eda_usable_top1` test delta `+0.0762`，`bad_top10_eda_usable_top1` test delta `+0.1760`。

## 下一步候选任务

- 不建议继续在 physiology distance / rerank / gate 这一类框架里做阈值微调或简单加深模型。
- 如果仍坚持生理方向，下一步应先改变任务定义：把生理作为不确定性、等待收益、个体状态分层或可观测性判别，而不是直接用生理距离选择一个未来候选。
- 如果目标是预测效果主线，应转向 vehicle 多未来分布、可等待策略、预测不确定性和差样本可观测性建模；生理只作为辅助解释或分层证据。

## 禁止任务

- 不把 EDA usable 子集写成成功改善。
- 不把 top5 oracle 的小幅改善写成可部署结果。
- 不继续旧 bio selector/reranker/reliability filter 阈值微调。
- 不用 test 后验误差选择 EDA 窗口、特征集、阈值或策略。

---

# 最新更新：2026-07-02 v287 physiology temporal-window route audit

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：把 v285 raw 200Hz 特征按时间窗口/信号族拆开后，也没有得到可部署 top1 改善。
- 新发现：ECG 最近 1-2 秒有弱排序/诊断苗头，但它没有转成 validation 可选择的部署策略。

## 已完成任务

- v287 `physio_temporal_window_route_audit`：
  - 脚本：`03_baselines/scripts/stage03_v287_physio_temporal_window_route_audit_20260702.py`
  - 输出：`03_baselines/v287_physio_temporal_window_route_audit_20260702`
  - 报告：`03_baselines/v287_physio_temporal_window_route_audit_20260702/reports/v287_physio_temporal_window_route_audit_cn.md`
  - ZIP：`03_baselines/v287_physio_temporal_window_route_audit_20260702_pack.zip`
- 关键结果：
  - `route_viable_now=false`。
  - feature_set_n `47`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.2379`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta `+0.2314`。
  - test-best top1 diagnostic 未通过：最佳 `combo_pre2_0_ecg_top16` 仍差 `+0.0941`。
  - test bad_top10 best corr `0.0854`，来自 `combo_pre1_0_ecg_top16`，属于非部署诊断信号。
  - 单独窗口最好：`win_pre10_pre5_top32`，bad_top10 top1 delta `+0.1144`。
  - 单独信号族最好：`signal_resp_top32`，bad_top10 top1 delta `+0.1783`。

## 下一步候选任务

- 不建议在 v285/v287 同一特征层上继续训练复杂 vehicle+physio 融合模型。
- 如果继续生理方向，应只做源信号层面的证据修复：ECG 峰检测质量、同步偏移、原始 1000Hz 到 200Hz 清洗链、EDA 可用记录、RESP 相位重建。
- 如果转回预测效果主线，应基于车辆多未来候选、概率/不确定性输出和等待代价建模；生理仅保留为边界证据或很弱的辅助诊断。

## 禁止任务

- 不把 `combo_pre2_0_ecg_top16` 的 test-best 诊断写成部署结果。
- 不把 corr `0.0854` 写成差样本本质改善。
- 不继续旧 bio selector/reranker/reliability filter 阈值微调。
- 不用 test 后验误差选窗口、信号族、阈值或策略。

---

# 最新更新：2026-07-02 v286 raw-200Hz online subject-aware calibration

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：v285 的 raw 200Hz shape-state 在 subject-disjoint route gate 失败；v286 在 subject-aware online 边界下也没有让 raw285 生理 KNN 形成差样本本质改善。

## 已完成任务

- v286 `raw200_online_subject_calibration`：
  - 脚本：`03_baselines/scripts/stage03_v286_raw200_online_subject_calibration_20260702.py`
  - 输出：`03_baselines/v286_raw200_online_subject_calibration_20260702`
  - 报告：`03_baselines/v286_raw200_online_subject_calibration_20260702/reports/v286_raw200_online_subject_calibration_cn.md`
  - ZIP：`03_baselines/v286_raw200_online_subject_calibration_20260702_pack.zip`
- 关键结果：
  - test bad_top10 fixed wait-latest：`0.6950`。
  - online subject mean vehicle：`0.7112`。
  - online raw285 KNN vehicle：`0.7358`。
  - online subject mean vehicle+raw285：`0.6950`。
  - online raw285 KNN vehicle+raw285：`0.7197`。
  - raw285 KNN 相对纯 subject mean online 变差 `+0.0246`；vehicle+raw285 后再 KNN 仍变差 `+0.0085`。

## 下一步候选任务

- 不建议直接训练更复杂 vehicle+physio 融合模型。
- 继续追求预测效果时，主线应回到车辆多未来候选、概率/不确定性、等待策略代价和任务构造；生理只能作为边界证据。
- 如果仍要继续生理方向，只剩非常底层的数据层问题：重新清洗/重建 EDA、ECG 峰、RESP 相位等源信号，并在进入模型前先过 v285/v286 同级 gate。

## 禁止任务

- 不继续旧 bio selector/reranker/reliability filter 阈值微调。
- 不把 subject-aware 结果写成 subject-disjoint 泛化结论。
- 不把 raw285 KNN 的 online 诊断写成可部署改善。
- 不用 test 后验误差选特征集、阈值或策略。

---

# 最新更新：2026-07-02 v285 raw-200Hz signal-shape physiology route gate

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：回到底层 200Hz shape-state 后，subject-disjoint route gate 仍未通过。

## 已完成任务

- v285 `raw200_shape_state_route_gate`：
  - 脚本：`03_baselines/scripts/stage03_v285_raw200_shape_state_route_gate_20260702.py`
  - 输出：`03_baselines/v285_raw200_shape_state_route_gate_20260702`
  - 报告：`03_baselines/v285_raw200_shape_state_route_gate_20260702/reports/v285_raw200_shape_state_route_gate_cn.md`
  - ZIP：`03_baselines/v285_raw200_shape_state_route_gate_20260702_pack.zip`
- 关键结果：
  - `route_viable_now=false`。
  - raw200 feature 数：`1146`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1958`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta `+0.1826`。
  - test-best top1 diagnostic 未通过：best delta `+0.1578`。
  - test bad_top10 best corr `0.0498`，未超过 `0.05`。

## 下一步候选任务

- v286 已作为后续边界实验完成：subject-aware online 也没有得到 raw285 生理 KNN 额外收益。
- 当前不建议继续在同一生理表征上做更复杂模型。

## 禁止任务

- 不把 raw200 top3/top5 oracle 当成部署结果。
- 不直接把 v285 特征拼进轨迹模型当下一步主线。
- 不用 test 后验误差反选 raw feature set。

---

# 最新更新：2026-07-02 v284 dynamic low-identity physiology route gate

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：旧特征微调关闭之后，新构造的动态低身份 biomarker 也没有通过可部署 route gate。

## 已完成任务

- v284 `dynamic_low_identity_physio_route_gate`：
  - 脚本：`03_baselines/scripts/stage03_v284_dynamic_low_identity_physio_route_gate_20260702.py`
  - 输出：`03_baselines/v284_dynamic_low_identity_physio_route_gate_20260702`
  - 报告：`03_baselines/v284_dynamic_low_identity_physio_route_gate_20260702/reports/v284_dynamic_low_identity_physio_route_gate_cn.md`
  - ZIP：`03_baselines/v284_dynamic_low_identity_physio_route_gate_20260702_pack.zip`
- 关键结果：
  - `route_viable_now=false`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1697`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta `+0.1903`。
  - test-best top1 diagnostic 未通过：best delta `+0.1525`。
  - test bad_top10 best corr `0.0553`，只是弱排序信号。

## 下一步候选任务

- 不建议直接训练更复杂 vehicle+physio 融合模型。
- 若继续生理 goal，只剩两类合理分支：
  - 更底层信号重处理：重新处理 ECG/RESP/EDA 质量、节律、相位和个体内状态，而不是复用现有 v260 biomarker。
  - subject-aware 个体校准：把任务边界改成同驾驶员历史可用的个体化预测，不再宣称纯 subject-disjoint 主增量。

## 禁止任务

- 不继续旧 bio selector/reranker/reliability filter 阈值微调。
- 不把 weak corr 或 bio top3/top5 oracle 写成可部署结果。
- 不用 test 后验误差选特征集、阈值或策略。

---

# 最新更新：2026-07-02 v283 生理路线 lineage / gap 审计

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已完成路线级收口：旧生理特征/旧候选选择器路线不再作为主线继续微调。
- 下一步若继续生理路线，应只做“新生理状态表示 -> route gate -> 预测模型”的顺序。

## 已完成任务

- v283 `physio_route_lineage_gap_audit`：
  - 脚本：`03_baselines/scripts/stage03_v283_physio_route_lineage_gap_audit_20260702.py`
  - 输出：`03_baselines/v283_physio_route_lineage_gap_audit_20260702`
  - 报告：`03_baselines/v283_physio_route_lineage_gap_audit_20260702/reports/v283_physio_route_lineage_gap_audit_cn.md`
  - ZIP：`03_baselines/v283_physio_route_lineage_gap_audit_20260702_pack.zip`
- 关键结果：
  - `current_goal_achieved=false`。
  - `old_feature_selector_route_closed=true`。
  - `physio_source_alignment_ready=true`。
  - `next_route_requires_feature_redefinition=true`。
- 关键解释：
  - 失败不是大面积对齐错误：v254b/v260/v268 都显示 200Hz 时间轴和事件窗口覆盖基本可用。
  - 失败主要来自有效信号弱、派生列不可用/近常数、subject/recording 身份信号强于行为信号。

## 下一步候选任务

- 只保留一个生理主线候选：新建低身份、行为相关的生理状态表示。
  - 先做特征定义和 route gate，不先训练轨迹模型。
  - 通过条件：在 vehicle top40 歧义候选池内，生理距离与候选真实误差排序出现稳定正相关，并且 val/test 同向。
  - 未通过则把生理降级为 subject-aware 个体校准或边界证据。

## 禁止任务

- 不继续旧 bio selector/reranker/reliability filter 阈值微调。
- 不把 bio top3/top5 oracle 写成可部署结果。
- 不用 test 后验误差选 raw_set、阈值或策略。
- 不把 subject-aware / calibrated 结果写成 subject-disjoint 泛化结论。

---

# 最新更新：2026-07-02 v282 生理歧义消解 route gate

## 正在做的任务

- 生理数据 goal 仍未达成。
- 当前已确认：旧的生理特征层无法稳定解决 subject-disjoint 的车辆相似/未来分叉问题。
- 下一步若继续生理路线，应做 v283 级别的特征重构，而不是继续在 v272-v281 的旧候选/旧特征上微调。

## 已完成任务

- v282 `physio_ambiguity_route_gate`：
  - 脚本：`03_baselines/scripts/stage03_v282_physio_ambiguity_route_gate_20260702.py`
  - 输出：`03_baselines/v282_physio_ambiguity_route_gate_20260702`
  - 报告：`03_baselines/v282_physio_ambiguity_route_gate_20260702/reports/v282_physio_ambiguity_route_gate_cn.md`
  - ZIP：`03_baselines/v282_physio_ambiguity_route_gate_20260702_pack.zip`
- 关键结果：
  - route gate `route_viable_now=false`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1989`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta `+0.2347`。
  - oracle top3 bad ambiguous 稳定性未通过：val `+0.1617`、test `+0.0724`。
  - test bad_top10 的最佳生理排序相关均值仅 `0.00985`。

## 下一步候选任务

- 首选 v283：从 200Hz 连续生理层重新构造事件前生理状态特征。
  - 重点不是简单拼接，而是质量控制、个体内 baseline、短窗斜率/能量、响应相位、EDA/RESP/ECG/EMG 的事件前状态变化。
  - 重新验证这些新特征能否区分“车辆输入相似但后续轨迹分叉”的样本。
  - 仍必须用 validation 选策略，test 只报告。
- 若 v283 仍失败，则应把生理主线降级为边界证据或 subject-aware 个体校准分支。

## 禁止任务

- 不继续 v222a gate / 删除样本 / 轻量 residual。
- 不继续旧特征上的 bio selector、bio reranker、bio threshold、bio reliability filter 微调。
- 不把 bio top3/top5 oracle 写成可部署结论。
- 不用 test 后验误差做部署策略。

---

# 最新更新：2026-07-02 v279-v281 生理数据深挖收口

## 正在做的任务

- 生理数据 goal 仍未达成：v279-v281 没有得到“极大弥补锚点前信息不足、让差样本本质改善”的可部署结果。
- 当前不建议继续沿同类 bio selector / reranker / reliability filter 细调。

## 已完成任务

- v279 `physio_reliability_filter_for_listrank`：
  - 目的：生理不直接选轨迹，而是判断 v278 vehicle listwise 第一候选是否可信。
  - 结果：test-best diagnostic `0.6791`，deployable 仍 `0.6950`；生理可靠性没有超过车辆可靠性。
- v280 `crossfit_physio_reliability_filter`：
  - 目的：用 recording-group OOF train top 修正 v279 的训练候选偏乐观问题。
  - 结果：test-best diagnostic `0.6891`，deployable 仍 `0.6950`；生理仍不超过车辆。
- v281 `bio_top3_constrained_selector`：
  - 目的：把 v272 的 bio top3 oracle 上限转成可训练 selector。
  - 结果：bio top3 oracle test bad_top10 `0.6738`，但 val 可部署策略仍 `0.6950`；test-best diagnostic `0.6842`。

## 下一步候选任务

- 首选：停止同类生理候选消歧微调，回到车辆主线。
  - 方向 1：车辆多未来分布/概率预测，而不是单条平均轨迹。
  - 方向 2：anchor-aware 联合任务，显式建模“多看多久”和等待代价。
  - 方向 3：以 v278 的 vehicle listwise headroom 为基础，做更强的车辆生成候选和不确定性估计。
- 如果继续保留生理：
  - 只作为 subject-aware 个体校准或失败边界证据，不再宣称 subject-disjoint 下提供稳定行为预测增量。
  - 需要先由用户确认 ChatGPT Chrome 已处于 Pro/进阶模式，再发送 `gptpro_reviews\20260702_phase280_physio_next_prompt.md` 给 GPTPro 复核。

## 禁止任务

- 不继续 v222a gate / 删除样本 / 轻量 residual。
- 不继续同类 bio selector、bio reranker、bio threshold、bio reliability filter 微调。
- 不把 test-best diagnostic 写成可部署结论。
- 不把 subject-aware 信号写成 subject-disjoint 正式泛化结论。
- 不用 test 后验误差做部署策略。

# 最新更新：2026-07-02 v278 listwise candidate rank loss 完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户 goal 仍未达成。
- v278 已验证“候选选择损失”路线：vehicle-only listwise 有更强 test diagnostic headroom，但生理/风格没有带来增量，val 仍不能部署。

## 已完成任务

- v278：listwise candidate rank loss。
  - 脚本：`03_baselines/scripts/stage03_v278_listwise_candidate_rank_loss_20260702.py`
  - 输出：`03_baselines/v278_listwise_candidate_rank_loss_20260702`
  - 报告：`03_baselines/v278_listwise_candidate_rank_loss_20260702/reports/v278_listwise_candidate_rank_loss_cn.md`
  - ZIP：`03_baselines/v278_listwise_candidate_rank_loss_20260702_pack.zip`
- v277：style + calibrated physiology candidate gain model。
  - 脚本：`03_baselines/scripts/stage03_v277_style_bio_candidate_gain_model_20260702.py`
  - 输出：`03_baselines/v277_style_bio_candidate_gain_model_20260702`
  - 报告：`03_baselines/v277_style_bio_candidate_gain_model_20260702/reports/v277_style_bio_candidate_gain_model_cn.md`
  - ZIP：`03_baselines/v277_style_bio_candidate_gain_model_20260702_pack.zip`
- v276：bio-assisted candidate gain model。
  - 脚本：`03_baselines/scripts/stage03_v276_bio_assisted_candidate_gain_model_20260702.py`
  - 输出：`03_baselines/v276_bio_assisted_candidate_gain_model_20260702`
  - 报告：`03_baselines/v276_bio_assisted_candidate_gain_model_20260702/reports/v276_bio_assisted_candidate_gain_model_cn.md`
  - ZIP：`03_baselines/v276_bio_assisted_candidate_gain_model_20260702_pack.zip`
- v275：stable bio consensus override。
  - 脚本：`03_baselines/scripts/stage03_v275_stable_bio_consensus_override_20260702.py`
  - 输出：`03_baselines/v275_stable_bio_consensus_override_20260702`
  - 报告：`03_baselines/v275_stable_bio_consensus_override_20260702/reports/v275_stable_bio_consensus_override_cn.md`
  - ZIP：`03_baselines/v275_stable_bio_consensus_override_20260702_pack.zip`
- v274：no-harm bio override。
  - 脚本：`03_baselines/scripts/stage03_v274_noharm_bio_override_20260702.py`
  - 输出：`03_baselines/v274_noharm_bio_override_20260702`
  - 报告：`03_baselines/v274_noharm_bio_override_20260702/reports/v274_noharm_bio_override_cn.md`
  - ZIP：`03_baselines/v274_noharm_bio_override_20260702_pack.zip`
- v273：bio-prefiltered pair reranker。
  - 脚本：`03_baselines/scripts/stage03_v273_bio_prefiltered_pair_reranker_20260702.py`
  - 输出：`03_baselines/v273_bio_prefiltered_pair_reranker_20260702`
  - 报告：`03_baselines/v273_bio_prefiltered_pair_reranker_20260702/reports/v273_bio_prefiltered_pair_reranker_cn.md`
  - ZIP：`03_baselines/v273_bio_prefiltered_pair_reranker_20260702_pack.zip`
- v272：physiology ambiguity disambiguation。
  - 脚本：`03_baselines/scripts/stage03_v272_physio_ambiguity_disambiguation_20260702.py`
  - 输出：`03_baselines/v272_physio_ambiguity_disambiguation_20260702`
  - 报告：`03_baselines/v272_physio_ambiguity_disambiguation_20260702/reports/v272_physio_ambiguity_disambiguation_cn.md`
  - ZIP：`03_baselines/v272_physio_ambiguity_disambiguation_20260702_pack.zip`

## v278 关键结果

- 设置：同事件候选组内排序标签，不再回归绝对候选收益。
- 输入比较：
  - `listrank_vehicle`
  - `listrank_vehicle_bio`
  - `listrank_vehicle_style_bio`
- test bad_top10：
  - fixed wait-latest `0.6950`
  - oracle best `0.6125`
  - vehicle-only test-best diagnostic `0.6832`，覆盖率 `0.1053`
  - best deployable `0.6950`，覆盖率 `0`
  - best bio feature diagnostic `0.6950`
- 判断：
  - 候选选择损失是有价值的，说明车辆多未来候选排序仍有可挖空间。
  - 生理/风格没有带来候选排序增量。
  - 当前 goal 仍未完成；若继续提升，应把主线转到车辆多未来排序、选择置信度和不确定性，而不是继续同类生理特征。

## v277 关键结果

- 设置：v276 candidate gain 框架 + 当前任务口径驾驶风格 + v271 校准 raw 生理。
- 输入比较：
  - `candidate_vehicle`
  - `candidate_vehicle_style_dist`
  - `candidate_vehicle_bio271_dist`
  - `candidate_vehicle_style_bio_dist`
  - `candidate_vehicle_style_query`
  - `candidate_vehicle_style_bio_query`
- test bad_top10：
  - fixed wait-latest `0.6950`
  - oracle best `0.6125`
  - val-best deployable `0.6950`，test 覆盖率 `0`
  - test-best diagnostic `0.7008`，比 fixed wait-latest 更差
- 判断：
  - 驾驶风格 + 校准生理没有形成可部署状态消歧。
  - 可覆盖 test 的策略在 val 和 test 都伤害明显。
  - 当前 subject-disjoint 生理/风格路线不应继续做同类微调。

## v276 关键结果

- 设置：回到 v267 full vehicle top40 候选池，对候选轨迹学习相对 latest 的收益。
- 输入比较：
  - `candidate_vehicle`
  - `candidate_vehicle_bio`
  - `candidate_bio_only`
- test bad_top10：
  - fixed wait-latest `0.6950`
  - oracle best `0.6125`
  - test-best gain diagnostic `0.6858`，覆盖率 `0.0526`
  - val-best any `0.6950`，覆盖率 `0`
- 判断：
  - 生理能在 test 的少数差样本上碰到候选收益，但验证集不能稳定选择。
  - candidate_vehicle_bio 没有稳定优于 candidate_vehicle。
  - v276 仍不能算“充分利用生理数据后让差样本本质改善”，只能作为边界诊断。

## v272 关键结果

- test bad_top10：
  - vehicle nearest `0.8785`
  - vehicle candidate oracle k40 `0.6166`
  - val 选 bio top1 `0.8940`
  - test-best bio top1 diagnostic `0.8744`
  - test-best bio top3 oracle `0.6738`
- 判断：
  - 生理距离不能直接把好候选排第一。
  - bio top3/top5 内有少量上界，但这还不是可部署策略。

## v274 关键结果

- 设置：默认 wait-latest，只有当 bio-prefiltered pair model 高置信并有足够 margin 时才覆盖 latest。
- test bad_top10：
  - keep0 `1.1977`
  - fixed wait-latest `0.6950`
  - oracle best `0.6125`
  - bio-prefilter candidate oracle `0.6466`
  - test-best override diagnostic `0.6902`，覆盖率 `0.0870`
  - val-best any / active / noharm active `0.6950`
- 判断：
  - no-harm override 能在 test-best 诊断里挖出极小改善，但验证集不能稳定选中这个规则。
  - 这不是可部署突破，不能算完成“让差样本本质改善”的 goal。
  - 下一步不建议继续围绕同类 bio 阈值、selector、reranker 调参。

## v275 关键结果

- 设置：多生理视角一致投票；只有多个 raw_set / pred_col 支持同一个非 latest 锚点时才允许覆盖 latest。
- 稳定性约束：除 val bad_top10 外，还检查 val all / normal / strong_steer / observe_later_like。
- test bad_top10：
  - fixed wait-latest `0.6950`
  - oracle best `0.6125`
  - bio-prefilter candidate oracle `0.6466`
  - test-best consensus diagnostic `0.6881`，覆盖率 `0.1053`
  - val-best any / active / stable / noharm-all `0.6950`，覆盖率 `0`
- 判断：
  - test-best diagnostic 低于 fixed wait-latest，但 val bad_top10 伤害 `+0.1385`、val all 伤害 `+0.0295`，不可部署。
  - val 稳定规则到 test bad_top10 上不触发覆盖，因此没有收益。
  - 当前生理一致性仍不能作为稳定可部署的差样本修正信号。

## v273 关键结果

- 设置：车辆 top40 -> 生理 top5 预筛 -> 监督式 pair reranker。
- test bad_top10：
  - fixed wait-latest `0.6950`
  - full oracle `0.6125`
  - bio-prefilter candidate oracle `0.6466`
  - test-best deployable diagnostic `0.7964`
  - val-best vehicle+bio `0.8664`
- 判断：
  - 生理 top5 上界没有稳定转成模型收益。
  - 当前生理路线的主要限制不是候选不存在，而是可部署消歧/排序信号不足。

## 当前判断

- 生理路线已覆盖：事件 biomarker、raw waveform、去身份化、个体/recording 校准、wait gate、prototype reranker、在线个体校准、生理消歧诊断、bio 预筛小候选 selector。
- 这些证据整体指向：当前生理数据不足以作为主增量完成差样本本质改善。
- 下一步建议：
  - A：转回车辆多未来候选、uncertainty/ranker 和可部署轨迹选择。
  - B：把生理保留为辅助诊断或边界证据，而不是继续作为主模型输入。
  - C：若仍坚持生理，需要新增外部标签/更长个人基线/新的采集质量，而不是继续用现有表征做小改动。

---

# 最新更新：2026-07-02 v271 calibrated raw physiology state 完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户 goal 仍未达成。
- v271 已完成个体/recording 基线校准 raw 生理状态实验。

## 已完成任务

- v271：calibrated raw physiology state。
  - 脚本：`03_baselines/scripts/stage03_v271_calibrated_raw_physio_state_20260702.py`
  - 输出：`03_baselines/v271_calibrated_raw_physio_state_20260702`
  - 报告：`03_baselines/v271_calibrated_raw_physio_state_20260702/reports/v271_calibrated_raw_physio_state_cn.md`
  - ZIP：`03_baselines/v271_calibrated_raw_physio_state_20260702_pack.zip`
- v270：raw physiology state latent。
  - 脚本：`03_baselines/scripts/stage03_v270_raw_physio_state_latent_20260702.py`
  - 输出：`03_baselines/v270_raw_physio_state_latent_20260702`
  - 报告：`03_baselines/v270_raw_physio_state_latent_20260702/reports/v270_raw_physio_state_latent_cn.md`
  - ZIP：`03_baselines/v270_raw_physio_state_latent_20260702_pack.zip`

## v271 关键结果

- 校准设定：
  - subject / recording robust z summary。
  - subject-centered / recording-centered raw waveform PCA。
  - 对 val/test 属于 calibrated / transductive setting，不是 pure cold-start。
- test bad_top10：
  - keep0 `1.1977`
  - fixed wait-latest `0.6950`
  - oracle best `0.6125`
  - pair candidate oracle k40 `0.6166`
  - wait test-best `0.6950`，等价于全 wait-latest。
  - pair test-best deployable `subject_seq_pca72:pair_vehicle_bio_badweighted_hgb_k5 = 0.7853`
  - val-best vehicle+raw `calibrated_low_identity48:pair_vehicle_bio_hgb_k40 = 0.9232`
- 与 v270 对比：
  - v270 best diagnostic pair `0.7866`。
  - v271 best diagnostic pair `0.7853`。
  - 改善约 `0.0013`，无实质意义。

## 当前判断

- 个体/recording 校准确实降低部分身份混淆，但行为 eta 仍小，无法转化成可部署预测收益。
- 生理路线已完成：事件 biomarker、去身份特征、raw waveform、个体/recording 校准、wait gate、prototype reranker、online subject-aware 等多角度验证。
- 当前证据不支持继续投入同类生理小变体。
- 下一步建议：
  - A：把主线转回车辆多未来候选、uncertainty/ranker 和可部署轨迹选择。
  - B：把 v260-v271 生理实验整理为“边界证据/辅助信息”，不作为主增量路线。
  - C：除非新增更强外部生理标签或更长个人基线，否则不再继续堆生理模型。

---

# 最新更新：2026-07-02 v270 raw physiology state latent 完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户 goal 仍未达成。
- v270 已完成 raw waveform state latent 的可部署验证。

## 已完成任务

- v270：raw physiology state latent。
  - 脚本：`03_baselines/scripts/stage03_v270_raw_physio_state_latent_20260702.py`
  - 输出：`03_baselines/v270_raw_physio_state_latent_20260702`
  - 报告：`03_baselines/v270_raw_physio_state_latent_20260702/reports/v270_raw_physio_state_latent_cn.md`
  - ZIP：`03_baselines/v270_raw_physio_state_latent_20260702_pack.zip`
- v269：reliable / identity-removed physiology。
  - 脚本：`03_baselines/scripts/stage03_v269_reliable_identity_removed_physio_20260702.py`
  - 输出：`03_baselines/v269_reliable_identity_removed_physio_20260702`
  - 报告：`03_baselines/v269_reliable_identity_removed_physio_20260702/reports/v269_reliable_identity_removed_physio_cn.md`
  - ZIP：`03_baselines/v269_reliable_identity_removed_physio_20260702_pack.zip`
- v268：physiology quality / alignment / identifiability audit。
  - 脚本：`03_baselines/scripts/stage03_v268_physio_quality_identifiability_audit_20260702.py`
  - 输出：`03_baselines/v268_physio_quality_identifiability_audit_20260702`
  - 报告：`03_baselines/v268_physio_quality_identifiability_audit_20260702/reports/v268_physio_quality_identifiability_audit_cn.md`
  - ZIP：`03_baselines/v268_physio_quality_identifiability_audit_20260702_pack.zip`
- v267：supervised bio prototype reranker。
  - 脚本：`03_baselines/scripts/stage03_v267_supervised_bio_prototype_reranker_20260702.py`
  - 输出：`03_baselines/v267_supervised_bio_prototype_reranker_20260702`
  - 报告：`03_baselines/v267_supervised_bio_prototype_reranker_20260702/reports/v267_supervised_bio_prototype_reranker_cn.md`
  - ZIP：`03_baselines/v267_supervised_bio_prototype_reranker_20260702_pack.zip`

## v270 关键结果

- raw sequence：
  - event_n `1167`
  - raw sequence shape delay0 `[1167, 6, 400]`
  - raw physio ok rate `0.9195`
- raw 特征集：
  - `raw_summary_fft`
  - `raw_pca96`
  - `raw_screened64`
  - `raw_low_identity48`
- test bad_top10：
  - keep0 `1.1977`
  - fixed wait-latest `0.6950`
  - oracle best `0.6125`
  - pair candidate oracle k40 `0.6166`
  - wait test-best `wait_raw_raw_summary_fft_gain = 0.6950`，但 selected_latest_rate `1.0`，等价于全 wait-latest。
  - pair test-best deployable `raw_screened64:pair_vehicle_bio_badweighted_hgb_k20 = 0.7866`。
  - val-best vehicle+raw `raw_low_identity48:pair_vehicle_bio_hgb_k5 = 0.8142`。

## 当前判断

- v270 说明 raw waveform latent 仍不能完成 goal。
- raw 特征确实能产生一点区分，但没有稳定转化为可部署策略收益。
- wait gate 的最好结果来自“全等 latest”，不是生理判断。
- pair reranker 中候选 oracle 仍接近 oracle，说明候选库仍有 headroom；失败点仍是可部署选择信号不足。
- 同类 raw 特征筛选/融合/reranker 不应继续作为主要方向。
- 下一步候选：
  - A：若坚持生理，验证个体基线、recording 内归一化、subject-aware 校准，明确这是校准任务而不是纯 cold-start subject-disjoint。
  - B：若坚持 subject-disjoint 正式预测，回到车辆多未来候选、不确定性估计和可部署轨迹选择主线。
  - C：整理 v260-v270 生理路线为边界证据，不再消耗同类算力。

---

# 最新更新：2026-07-02 v269 reliable / identity-removed physiology 完成

## v269 关键结果

- 特征集：
  - `reliable_top64`
  - `dynamic_top48`
  - `low_identity_top32`
  - `combo_identity_removed64`
- test bad_top10：
  - keep0 `1.1977`
  - fixed wait-latest `0.6950`
  - oracle best `0.6125`
  - pair candidate oracle k40 `0.6166`
  - wait gate best `0.6950`，但 selected_latest_rate `1.0`，实际退化为全 wait-latest。
  - pair test-best deployable `combo_identity_removed64:pair_base_hgb_k40 = 0.7781`。
  - test-best vehicle_bio 诊断 `low_identity_top32:pair_vehicle_bio_badweighted_hgb_k5 = 0.7981`。
  - val-best vehicle+bio `combo_identity_removed64:pair_vehicle_bio_badweighted_hgb_k5 = 0.8365`。
- 与 v267 对比：
  - v267 val-best vehicle+bio `0.8495`。
  - v269 val-best vehicle+bio `0.8365`。
  - 改善约 `0.0130`，但仍远高于 fixed wait-latest。

## v269 判断

- v269 有微弱正向变化，但不是本质改善。
- wait gate 的最好结果来自“几乎全等 latest”，不是生理判断。
- pair reranker 中候选 oracle 仍接近 oracle，说明候选库仍有 headroom；失败点仍是可部署选择信号不足。
- v269 之后已进一步执行 v270 raw waveform latent，确认底层 raw 表征仍未突破 fixed wait-latest。

---

# 最新更新：2026-07-02 v268 physiology quality / alignment / identifiability audit 完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户 goal 仍未达成。
- v268 已完成现有生理链路的数据质量、对齐覆盖、身份混淆和候选排序可识别性审计。

## 已完成任务

- v268：physiology quality / alignment / identifiability audit。
  - 脚本：`03_baselines/scripts/stage03_v268_physio_quality_identifiability_audit_20260702.py`
  - 输出：`03_baselines/v268_physio_quality_identifiability_audit_20260702`
  - 报告：`03_baselines/v268_physio_quality_identifiability_audit_20260702/reports/v268_physio_quality_identifiability_audit_cn.md`
  - ZIP：`03_baselines/v268_physio_quality_identifiability_audit_20260702_pack.zip`

## v268 关键结果

- source timing：
  - recording_n `82`
  - subject_n `18`
  - median_hz `200.000`
  - gap_gt_20ms_total `0`
  - duplicate_t_total `0`
- signal availability：
  - ECG/EMG/HR/RESP raw-filt 基础列整体可用。
  - `HRV_RMSSD` usable `0/82`。
  - `RESP_BPM` usable `0/82`。
  - `RESP_Amplitude` usable `0/82`。
  - EDA usable `73/82`，有 `9` 个 recording 近常数/缺失。
- event coverage：
  - min split-delay ok_rate `0.889`。
  - post-observation rate `0`。
- identity / behavior：
  - median family identity/behavior eta ratio `68.74`。
  - 各 family 的 behavior_eta_max_mean 均低于 `0.006`。
- candidate rank：
  - test bad_top10 `pred_pair_vehicle_bio_hgb` chosen_rmse `0.8460`。
  - best candidate rmse `0.6166`。
  - chosen_minus_latest `+0.1509`。
  - true best top3 rate `0.211`。

## 当前判断

- 原始 200Hz 连续生理不是主要问题。
- 当前派生生理表征存在两个核心问题：
  - 一部分关键派生列不可用或近常数。
  - 可分性主要来自 subject/recording，而不是可迁移行为状态。
- 继续把当前 bio260 直接拼接到 MLP/attention/reranker，边际价值很低。
- 若继续围绕用户 goal 推进，下一步优先级：
  - A：从 200Hz 连续层重建可靠生理表征，剔除/修复不可用 HRV、RESP、EDA 派生列。
  - B：做个体内归一化、recording residualization、identity-removed 特征筛选。
  - C：只把通过可识别性审计的生理特征接入 wait gate 或 candidate ranker。
  - D：如果上述仍失败，则应承认 subject-disjoint 下当前生理数据无法完成 goal，转回车辆多未来/uncertainty/ranker 主线。

---

# 最新更新：2026-07-02 v267 supervised bio prototype reranker 完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户 goal 仍未达成。
- v267 已完成“更强候选内部监督 reranker”验证。

## 已完成任务

- v267：supervised bio prototype reranker。
  - 脚本：`03_baselines/scripts/stage03_v267_supervised_bio_prototype_reranker_20260702.py`
  - 输出：`03_baselines/v267_supervised_bio_prototype_reranker_20260702`
  - 报告：`03_baselines/v267_supervised_bio_prototype_reranker_20260702/reports/v267_supervised_bio_prototype_reranker_cn.md`
  - ZIP：`03_baselines/v267_supervised_bio_prototype_reranker_20260702_pack.zip`

## v267 关键结果

- test bad_top10：
  - keep0 `1.1977`
  - fixed wait-latest `0.6950`
  - full oracle `0.6125`
  - pair candidate oracle k40 `0.6166`
  - val-best pair vehicle `0.8746`
  - val-best pair vehicle+bio `0.8495`
- 诊断：
  - test 上最好的 bio pair strategy 是 `pair_vehicle_bio_hgb_k20`，bad_top10 `0.8046`，但不是 val 选择结果，且仍未低于 wait-latest。
  - val-best bio 相比 val-best vehicle-only 只改善 `0.0251`。

## 当前判断

- v267 说明即使把候选重排从“简单距离规则”升级为“监督式 query-prototype reranker”，当前 bio260 仍不能把 candidate oracle headroom 转成可部署收益。
- 问题不是缺少一个更深 MLP/attention/reranker，而是 subject-disjoint 下的生理状态信号不足以稳定选择正确候选。
- 下一步优先级：
  - A：做生理数据质量/对齐可识别性审计，检查 bio260 是否被缺失、floor 合并、设备/recording 差异、时间对齐误差稀释。
  - B：改变任务边界为 subject-aware 个体校准。
  - C：如果目标允许回到车辆主线，则做车辆多未来候选 + uncertainty/ranker。

---

> 最新指针：2026-07-02 已完成 GPTPro phase02 桌面软件复核与 v266 vehicle-matched bio residual prototype。当前没有正在运行的训练进程。GPTPro 给出的三条可证伪路线中：wait-benefit 已由 v265 覆盖且失败；subject-aware online 已由 v264 覆盖，生理 KNN 无额外收益；vehicle-matched prototype 已由 v266 覆盖。v266 显示 candidate oracle 有 headroom（test bad_top10 `0.6166`，接近 full oracle `0.6125`，低于 fixed wait-latest `0.6950`），但可部署 reranker 未达标：val-best vehicle-only `0.8890`，val-best vehicle+bio `0.8374`，仍高于 fixed wait-latest。当前结论：生理路线不再缺“又一个更强融合模型”，而是缺能稳定选择正确候选的可部署信号；subject-disjoint 生理主线应暂停，下一步优先回到车辆多未来/不确定性/候选轨迹选择主线，或另立 subject-aware 个体校准任务。

---

# 最新更新：2026-07-02 GPTPro phase02 + v266 完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户 goal 仍未达成。
- GPTPro phase02 已通过 ChatGPT 桌面软件 Pro / Pro 扩展完成，不再是“未发送”状态。
- v266 已完成 GPTPro 路线2的最小可证伪验证。

## 已完成任务

- GPTPro phase02 外部复核：
  - 提问词：`gptpro_reviews/20260702_phase02_prompt.md`
  - 回复：`gptpro_reviews/20260702_phase02_response.md`
  - 原始可访问性树：`gptpro_reviews/20260702_phase02_response_raw_accessibility.txt`
- v266：vehicle-matched bio residual prototype。
  - 脚本：`03_baselines/scripts/stage03_v266_vehicle_matched_bio_residual_prototype_20260702.py`
  - 输出：`03_baselines/v266_vehicle_matched_bio_residual_prototype_20260702`
  - 报告：`03_baselines/v266_vehicle_matched_bio_residual_prototype_20260702/reports/v266_vehicle_matched_bio_residual_prototype_cn.md`
  - ZIP：`03_baselines/v266_vehicle_matched_bio_residual_prototype_20260702_pack.zip`

## v266 关键结果

- test bad_top10：
  - keep0 `1.1977`
  - fixed wait-latest `0.6950`
  - full oracle `0.6125`
  - vehicle-matched candidate oracle k40 `0.6166`
  - val-best vehicle-only prototype `0.8890`
  - val-best vehicle+bio prototype `0.8374`
- v266 的重要含义：
  - 相似车辆 prototype 候选库几乎有足够 headroom，说明“可选候选不存在”不是主因。
  - 但当前 vehicle/bio 重排序选不准，不能稳定把 headroom 变成可部署收益。
  - bio 在 val-best 可部署对照里比 vehicle-only 低 `0.0516`，但仍远高于 fixed wait-latest，不能称为差样本本质改善。

## 当前判断

- GPTPro 建议的三条路线已形成闭环：
  - wait-benefit / CATE-style：v265 已测，bio 无稳定增量。
  - vehicle-matched residual prototype：v266 已测，有 oracle headroom，但 bio reranker 不达标。
  - subject-aware / online calibration：v264 已测，生理 KNN 没有额外收益。
- 因此，subject-disjoint 生理主线应暂停。
- 如果继续追求差样本本质改善，优先级更高的是车辆主线：多未来轨迹候选、候选不确定性、可部署轨迹选择，而不是继续生理拼接/更深融合。

## 下一步候选

- 选择 A：回到车辆主线，基于 v266 暴露的 headroom，做“车辆多未来候选 + 可部署 uncertainty/ranker”，不再依赖生理作为主信号。
- 选择 B：另立 subject-aware 个体校准任务，把同驾驶员历史反馈作为正式部署条件，但不再包装为 subject-disjoint 泛化。
- 选择 C：如果仍坚持生理，必须先做生理数据质量/对齐/任务可识别性审计，而不是继续训练更强融合网络。

---

> 最新指针：2026-07-02 已追加 v265 physiology uncertainty / wait frontier。当前没有正在运行的训练进程。v265 验证“生理作为不确定性/风险校准信号”这一最后一个合理用途：所有风险分数只在 train 拟合，等待比例阈值只在 val 定标。结果：test bad_top10 上所有分数的最佳 RMSE 都退化为全 wait-latest `0.6950`；生理 badprob 有弱 AUC（vehicle+bio `0.6175`，bio-only `0.6376`），但无法稳定转化为可部署等待策略收益。结合 v260-v264，当前生理数据在正式 subject-disjoint 边界下没有形成“差样本本质改善”的建模路线。下一步需要用户确认：改变任务边界为 subject-aware 个体校准，或暂停生理主线转向车辆多未来/不确定性模型，或手动确认 GPTPro Pro/进阶模式后做外部复核。

---

# 最新更新：2026-07-02 v265 physiology uncertainty / wait frontier 完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户 goal 仍未达成。
- v265 已完成最后一类生理用途复核：风险/不确定性校准。

## 已完成任务

- v265：physiology uncertainty / wait frontier。
  - 脚本：`03_baselines/scripts/stage03_v265_physio_uncertainty_wait_frontier_20260702.py`
  - 输出：`03_baselines/v265_physio_uncertainty_wait_frontier_20260702`
  - 报告：`03_baselines/v265_physio_uncertainty_wait_frontier_20260702/reports/v265_physio_uncertainty_wait_frontier_cn.md`
  - ZIP：`03_baselines/v265_physio_uncertainty_wait_frontier_20260702_pack.zip`

## v265 关键结果

- test bad_top10：
  - keep0 `1.1977`
  - fixed wait-latest `0.6950`
  - oracle `0.6125`
  - 所有风险分数最佳 RMSE 都退化为 wait-latest `0.6950`
- 风险诊断：
  - test `score_vehicle_bio_badprob` bad_top10 AUC `0.6175`
  - test `score_bio_only_badprob` bad_top10 AUC `0.6376`
  - 说明生理有弱风险信号。
- 策略前沿：
  - 这个弱风险信号没有稳定转化为同等等待预算下的 RMSE 优势。
  - vehicle+bio 分数不稳定，不支配 vehicle 分数。

## 当前判断

- 生理作为直接预测、selector、wait gate、online KNN、风险校准都未达到 goal。
- 当前 blocker 不是模型层还不够复杂，而是正式 subject-disjoint 边界下的生理信号太弱，无法可靠转化为可部署策略。
- 继续局部加模型不合理；需要用户确认是否改变任务边界或补充外部复核/数据。

## 下一步候选

- 选择 A：改变任务边界，正式做 subject-aware 个体校准任务。
- 选择 B：暂停生理主线，回到车辆多未来分布/不确定性模型。
- 选择 C：用户手动确认 Chrome 为 Pro/进阶模式后，发送 `gptpro_reviews/20260702_phase02_prompt.md` 给 GPTPro 做外部复核。

---

> 最新指针：2026-07-02 已追加 v264 online subject-aware physiology calibration。当前没有正在运行的训练进程。v264 不是正式 subject-disjoint 替代结果，而是边界实验：允许同一驾驶员更早事件结果做在线校准。结果显示 online subject 历史反馈能让 test bad_top10 从 global vehicle gate `0.7528` 接近 fixed wait-latest，到 `0.7112`；但 physiology KNN 没有额外收益，`online_physio_knn_vehicle` 仍为 `0.7112`，`online_physio_knn_vehicle_bio` 为 `0.7698`。GPTPro phase02 提问词已归档，但因无法确认 Chrome 当前为 Pro/进阶模式，桥接脚本拒绝发送。当前 goal 仍未达成；生理主线若继续，应先明确换成 subject-aware 个体校准任务，否则应回到车辆多未来/不确定性建模主线。

---

# 最新更新：2026-07-02 v264 online subject-aware 生理校准边界实验完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户 goal 仍未达成。
- v264 证明：放宽为 online subject-aware 后，同驾驶员历史反馈有价值；但当前 bio260 生理特征没有证明有额外价值。

## 已完成任务

- v264：online subject-aware physiology calibration。
  - 脚本：`03_baselines/scripts/stage03_v264_online_subject_physio_calibration_20260702.py`
  - 输出：`03_baselines/v264_online_subject_physio_calibration_20260702`
  - 报告：`03_baselines/v264_online_subject_physio_calibration_20260702/reports/v264_online_subject_physio_calibration_cn.md`
  - ZIP：`03_baselines/v264_online_subject_physio_calibration_20260702_pack.zip`
- GPTPro phase02 复核：
  - 提问词：`gptpro_reviews/20260702_phase02_prompt.md`
  - 状态：未发送。
  - 原因：桥接脚本无法确认 Chrome 当前为 Pro/进阶模式，按规则拒绝发送。

## v264 关键结果

- test bad_top10：
  - keep0 `1.1977`
  - fixed wait-latest `0.6950`
  - oracle `0.6125`
  - global vehicle gate `0.7528`
  - global vehicle+bio260_sp64 gate `0.8748`
  - online_subject_mean_vehicle `0.7112`
  - online_physio_knn_vehicle `0.7112`
  - online_subject_mean_vehicle_bio `0.6950`
  - online_physio_knn_vehicle_bio `0.7698`

## 当前判断

- online subject 历史反馈可以接近 fixed wait-latest，说明“同驾驶员历史反馈”是有用边界。
- physiology KNN 没有超过纯 subject residual mean，说明当前 bio260 没有提供额外个体内状态区分能力。
- `online_subject_mean_vehicle_bio` 达到 `0.6950` 是因为策略几乎全等 wait-latest，不应解释为生理建模成功。
- 后续若继续生理，必须明确改变任务为 subject-aware 个体校准；如果仍坚持 subject-disjoint，当前生理主线优先级应下降。

## 下一步候选

- 选择 A：正式立一个 subject-aware online adaptation 子任务，目标是评估同驾驶员历史反馈能否作为部署条件，但不要把它写成 subject-disjoint 泛化提升。
- 选择 B：停止把当前生理作为主增量，回到车辆多未来分布/不确定性模型。
- 选择 C：人工确认 Chrome 已在 Pro/进阶模式后，再发送 `gptpro_reviews/20260702_phase02_prompt.md` 做外部复核。

---

> 最新指针：2026-07-02 已完成 v260-v263 生理重构与决策复核。当前没有正在运行的训练进程。用户 goal 仍未达成：v260 事件级 bio260 有弱 bad_top10 风险信号，但 v261 全量 bio260 anchor selector 弱于 vehicle selector；v262 subject-invariant sp64 只把 test bad_top10 selector tail 从 `0.9419` 小幅降到 `0.9059`，远高于 fixed wait-latest `0.6950`；v263 0ms wait gate 中 vehicle gate `0.7528` 优于 vehicle+bio260 gate `0.8748`，val 调阈值退化为几乎全等 latest。结论：当前生理数据尚不能本质弥补锚点前车辆信息不足；后续不建议继续做同类生理拼接/浅层融合，应转向 subject-aware 个体校准，或回到 subject-disjoint 车辆多未来分布/不确定性建模主线。

---

# 最新更新：2026-07-02 v260-v263 生理重构与 wait/anchor 决策复核完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户 goal：充分利用生理数据，弥补锚点前车辆信息不足，并让预测差样本有本质性改善。
- 该 goal 目前尚未达成。v260-v263 的证据显示，生理有弱诊断信号，但没有形成可部署的差样本大幅改善。

## 已完成任务

- v260：事件级 bio260 biomarker 重构。
  - 脚本：`03_baselines/scripts/stage03_v260_event_biomarker_physio_rebuild_20260702.py`
  - 输出：`03_baselines/v260_event_biomarker_physio_rebuild_20260702`
  - 结论：bio260 相比旧 physio200 在 bad_top10 诊断上略好，但 vehicle+bio260 仍不能稳定超过 vehicle-only 的正式未来行为预测。
- v261：bio260 全量 anchor selector。
  - 脚本：`03_baselines/scripts/stage03_v261_bio260_anchor_selector_20260702.py`
  - 输出：`03_baselines/v261_bio260_anchor_selector_20260702`
  - 结论：test bad_top10 中 vehicle selector `0.9425`，vehicle+bio260 `0.9765`，badweighted `0.9837`，生理加入后变差。
- v262：subject-invariant bio260 selector。
  - 脚本：`03_baselines/scripts/stage03_v262_subject_invariant_bio260_selector_20260702.py`
  - 输出：`03_baselines/v262_subject_invariant_bio260_selector_20260702`
  - 结论：sp64 特征使 test bad_top10 从 vehicle selector `0.9419` 小幅降到 `0.9059`，但仍显著弱于 fixed wait-latest `0.6950`。
- v263：0ms bio260 wait gate。
  - 脚本：`03_baselines/scripts/stage03_v263_bio260_wait_gate_20260702.py`
  - 输出：`03_baselines/v263_bio260_wait_gate_20260702`
  - 结论：vehicle gate `0.7528` 优于 vehicle+bio260 gate `0.8748`；val 阈值选择几乎全等 latest，说明主要收益来自多观察而不是生理判断。

## 当前判断

- 生理数据不是完全无信息；事件级 ECG/EDA/RESP/EMG 重构后确实有弱 bad_top10 风险信号。
- 但该信号在 subject-disjoint 正式预测、anchor selector 和 wait gate 中都没有形成稳定可部署增益。
- 若继续生理方向，应优先改任务边界为 subject-aware 个体校准，或先做生理质量/对齐修复；不建议继续盲目加深同类融合结构。
- 若目标仍是 subject-disjoint 正式预测，应把下一步放回车辆主线：显式多未来轨迹分布、不确定性输出、或更强的车辆时序 backbone。

## 下一步候选

- 选择 A：转为 subject-aware 生理个体校准任务，明确要求同一驾驶员历史样本可用。
- 选择 B：暂停生理作为主增量，回到 subject-disjoint 车辆多未来/不确定性建模。
- 选择 C：继续生理前，先单独做生理质量修复审计，例如 HRV/RESP/SCR 可靠重算、缺失记录修复、recording 对齐复核。

## 禁止任务

- 不继续做生理简单拼接、手工权重扫参、浅层 CNN/MLP/attention 盲试。
- 不删除差样本。
- 不把 oracle best anchor / test 后验结果当可部署策略。
- 不使用 observation_s 之后的生理或未来轨迹信息。
- 不把 subject-aware 小幅改善包装成 subject-disjoint 泛化提升。

---

> 最新指针：2026-07-01 已完成 v254a physio deep signal audit。当前没有正在运行的训练任务。v254a 已按“深层挖掘生理数据”路线，从 10Hz 生理表提取锚点前多窗口统计、趋势和窗口差值，并用 train-only 诊断头检查生理是否含未来行为信号。结果：10Hz 生理覆盖率 `0.919`，无 observation 后泄漏；生理确实有 subject/recording 结构，但行为标签可分性很弱，future_cluster4 top eta² 约 `0.015`，high_future_abs_q75 top eta² 约 `0.020`。test 分类中 vehicle_only 仍明显最好：future_cluster4 macro-F1 `0.7317`，physio10hz `0.2944`，vehicle+physio10hz `0.5020`；high_future_abs_q75 vehicle_only `0.7112`，physio10hz `0.4897`，vehicle+physio10hz `0.6239`。下一步不应继续拼接或手工权重，应重做生理表征：200Hz 事件相关变化、个体内归一化、EDA/EMG/HR/RESP 专用特征，并区分 subject-aware 与 subject-disjoint 评估。

---

# 最新更新：2026-07-02 生理数据深挖 v254b-v256 完成，goal 尚未达成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 用户设置的 goal 是：充分利用生理数据，弥补锚点前车辆信息不足，并让预测差样本有本质性改善。
- 该 goal 目前尚未达成：v254b/v255/v256 均未在正式 subject-disjoint 口径下改善 bad_top10 差样本。

## 已完成任务

- v254b：200Hz 连续生理事件相关表征。
  - 脚本：`03_baselines/scripts/stage03_v254b_physio_200hz_event_representation_20260702.py`
  - 输出：`03_baselines/v254b_physio_200hz_event_representation_20260702`
  - 结论：subject-disjoint 下车辆+生理不优于 vehicle_only；subject-aware bad_top10 诊断有一定信号。
- v255：生理状态条件化候选轨迹选择。
  - 脚本：`03_baselines/scripts/stage03_v255_physio_conditioned_candidate_ranker_20260702.py`
  - 输出：`03_baselines/v255_physio_conditioned_candidate_ranker_20260702`
  - 结论：oracle 候选池上限仍强，但 learned physio ranker 在 val 上无法通过 no-harm，test 退回 vehicle_rank1。
- v256：raw 200Hz 生理 CNN 融合轨迹预测。
  - 脚本：`03_baselines/scripts/stage03_v256_raw_physio_cnn_fusion_20260702.py`
  - 输出：`03_baselines/v256_raw_physio_cnn_fusion_20260702`
  - 结论：subject-disjoint bad_top10 上 vehicle+physio CNN 比 vehicle-only 更差；subject-aware bad_top10 只有很小改善。
- GPTPro 复核：
  - 提问词已保存：`gptpro_reviews/20260702_phase01_prompt.md`
  - 未发送原因：Chrome 当前 Pro/进阶模式无法被脚本确认，桥接脚本按规则拒绝发送。

## 当前判断

- 当前生理数据更像 subject/recording/个体状态信号，而不是稳定跨驾驶员行为预测信号。
- 继续做“更复杂的生理拼接、手工权重、候选重排序、浅层 CNN/TCN”优先级很低。
- 如果仍坚持用生理，推荐把问题明确改为 subject-aware 个体化/校准预测，而不是 subject-disjoint 跨驾驶员泛化。
- 如果目标仍是正式 subject-disjoint 差样本大幅改善，下一步应回到车辆与任务构造主线，重点考虑：
  - anchor-aware 等待/重锚定决策；
  - 显式多模态轨迹分布输出；
  - 更强车辆时序 backbone；
  - 但不能使用 test 后验、oracle future、删除样本或 v222a 式 gate/residual。

## 下一步待用户确认

- 选择 A：继续生理路线，但改成 subject-aware 个体化校准任务，明确需要同一驾驶员少量历史样本。
- 选择 B：暂停生理作为主增量，回到 subject-disjoint 正式预测提升，优先做 anchor-aware / 多模态车辆模型。
- 选择 C：用户手动把 Chrome 项目切到 Pro/进阶后，再把 `gptpro_reviews/20260702_phase01_prompt.md` 发给 GPTPro 做外部方法复核。

## 禁止任务

- 不继续盲目做生理简单拼接或权重扫描。
- 不删除差样本。
- 不把 oracle 候选/最佳锚点当可部署策略。
- 不使用 observation_s 后的生理或未来轨迹信息作为输入。
- 不把 subject-aware 小幅改善包装成 subject-disjoint 泛化提升。


## 最新更新：2026-07-01 v254a 生理信号深层挖掘审计已完成

### 正在做任务

- 当前没有正在运行的训练进程。
- v254a 已完成：这是生理表征/可分性/行为增量审计，不是新轨迹预测模型训练。

### 已完成任务

- 已新增并运行 `stage03_v254a_physio_deep_signal_audit_20260701.py`。
- 已生成 v254a 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701_pack.zip`
- 已输出核心表：
  - `tables\v254a_event_physio10hz_deep_features.csv`
  - `tables\v254a_alignment_coverage_summary.csv`
  - `tables\v254a_physio_signal_quality_summary.csv`
  - `tables\v254a_physio_eta2_by_target_feature.csv`
  - `tables\v254a_behavior_classification_diagnostics.csv`
  - `tables\v254a_future_summary_regression_diagnostics.csv`
  - `tables\v254a_feature_block_audit.csv`
- 已输出核心图：
  - `figures\v254a_behavior_classification_macro_f1.png`
  - `figures\v254a_future_summary_regression_r2.png`
  - `figures\v254a_top_physio_eta2.png`
  - `figures\v254a_physio10hz_window_rows.png`

### 结果判断

- v254a 已从 1Hz 粗窗口推进到 10Hz 多窗口深层统计，但 test 上仍未看到跨驾驶员泛化的行为预测增量。
- 生理数据不是空的：subject/recording 可分性很强，说明它含有身份/记录/设备或个体状态信息。
- 但这些信息没有转化成行为标签：future_cluster4、high_future_abs_q75、strong_steer_existing 的 eta² 只有约 `0.015-0.023`。
- 车辆输入仍是行为模式预测主信号；加入 physio10hz 后反而降低 test macro-F1，说明简单高维生理拼接会带来噪声或跨 subject 分布偏移。
- HRV_RMSSD 基本不可用；RESP_BPM/RESP_Amplitude 近常数比例偏高；后续不应把这些列直接当作有效动态特征。

### 下一步候选任务

- 首选：`v254b_physio_representation_redesign`。
  - 从 200Hz 连续层重新构造事件相关生理变化，而不是只用 1Hz/10Hz表格窗口统计。
  - 做个体内 baseline normalization：同一 subject / 同一 recording 内的 z-score、百分位、变化率。
  - 分模态重做专用特征：EDA tonic/phasic response、EMG burst、HR/HRV、RESP phase/amplitude。
  - 区分两套评估：subject-disjoint 泛化与 subject-aware 个体化。
- 可选：先做 `v254b1_physio_representation_quality_only`，只重算特征质量和标签可分性，不训练任何诊断头。

### 禁止任务

- 不把 v254a 解释成“生理数据无效”；只能说“当前 10Hz/1Hz窗口统计没有跨 subject 行为增量”。
- 不继续做简单拼接、手工权重扫描或手工 tie-break。
- 不把 subject/recording 可分性当作行为预测有效性。
- 不用 test 结果选择生理特征。

---

## 最新更新：2026-07-01 v253b 生理/驾驶风格状态 tie-break 审计已完成

### 正在做任务

- 当前没有正在运行的训练进程。
- v253b 已完成：这是状态信号在车辆相似候选池内的排序审计，不是预测模型训练。

### 已完成任务

- 已新增并运行 `stage03_v253b_physio_state_tiebreak_audit_20260701.py`。
- 已生成 v253b 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701_pack.zip`
- 已输出核心表：
  - `tables\v253b_tiebreak_summary.csv`
  - `tables\v253b_tiebreak_per_strategy.csv`
  - `tables\v253b_pool_distance_future_correlation_summary.csv`
  - `tables\v253b_pool_distance_future_correlation_by_sample.csv`
  - `tables\v253b_vehicle_candidate_pool_summary.csv`
  - `tables\v253b_subject_split_table.csv`
- 已输出核心图：
  - `figures\v253b_badtop10_tiebreak_selected_future_rmse.png`
  - `figures\v253b_tiebreak_delta_vs_vehicle_rank1.png`

### 结果判断

- v253b 比 v253a 更贴近用户假设：不是把生理/风格全局拼接进输入，而是在车辆相似候选池内做 tie-break。
- 但当前表示仍不成立：v250 bad_top10 all-delay 上，style / physio_recent / physio_guarded / style+physio 的 selected future RMSE 均高于 vehicle_rank1。
- 候选池 oracle 很强：bad_top10 all-delay oracle 可到 `0.3678`，说明问题不是“池里没有好未来”，而是当前生理/风格距离无法选中它。
- 候选池内距离-未来误差相关几乎为 0：bad_top10 中 style mean Spearman `0.026`，physio_recent `-0.008`，physio_guard `-0.022`，style+physio `-0.002`。
- 当前 split 是 subject-disjoint，不能验证同一驾驶员个体记忆；只能验证跨驾驶员状态相似性。

### 下一步候选任务

- 首选：`v254_state_conditioned_probabilistic_mixture`。
  - 车辆轨迹仍是主输入。
  - 生理/风格不作为手工距离直接决定候选。
  - 用可学习模块预测 mixture weight、mode prior、uncertainty 或 confidence。
  - 对 v252/v253b 中 oracle 上限大的样本输出多模态结果。
- 可选前置审计：`v254a_physio_quality_alignment_audit`。
  - 检查 1Hz 生理特征是否太粗、窗口是否太短、recording 对齐是否足够稳定。
  - 检查生理特征是否能区分 subject / recording / 高低转向强度 / 高低未来分叉。

### 禁止任务

- 不把 v253b 解释为“生理没有价值”；只能说“当前生理/风格表示和手工 tie-break 不可用”。
- 不用 oracle future 做部署选择器。
- 不继续做简单拼接权重扫描。
- 不回到 gate / 删除样本 / response type hard routing。

---

## 最新更新：2026-07-01 v253a 生理/驾驶风格状态信号消歧审计已完成

### 正在做任务

- 当前没有正在运行的训练进程。
- v253a 已完成：这是状态信号可用性和消歧审计，不是预测模型训练。

### 已完成任务

- 已新增并运行 `stage03_v253_state_signal_disambiguation_audit_20260701.py`。
- 已生成 v253a 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701_pack.zip`
- 已输出核心表：
  - `tables\v253a_old_style_match_audit.csv`
  - `tables\v253a_current_style_features_last60_guard3.csv`
  - `tables\v253a_current_physio_features_1hz.csv`
  - `tables\v253a_feature_block_audit.csv`
  - `tables\v253a_neighbor_divergence_by_feature_group.csv`
  - `tables\v253a_summary_by_feature_group_bucket_delay.csv`
  - `tables\v253a_key_comparison_vs_vehicle_only.csv`
- 已输出核心图：
  - `figures\v253a_state_signal_badtop10_disambiguation.png`
  - `figures\v253a_state_signal_delta_vs_vehicle_only.png`

### 结果判断

- 旧 stage04 style 表不能直接复用：与当前样本 `sample_id`、`event_uid`、`subject+session+anchor` 交集均为 `0`。
- 当前样本重新提取的驾驶风格覆盖完整：`7002/7002`，且不使用 observation 后未来、不重叠直接输入最后 3 秒。
- 生理 1Hz 特征 recording 覆盖率为 `0.919`，窗口均不超过 observation_s。
- 但消歧结果不支持“直接拼接状态特征”：
  - bad_top10_v250 all-delay，vehicle-only query-vs-neighbor future RMSE `1.0627`。
  - style_w0.25 `1.0637`，style_w0.50 `1.0868`。
  - physio_recent_w0.25 `1.0703`，physio_recent_w0.50 `1.1311`。
  - style+physio_w0.50 `1.1466`。
- 因此当前状态特征没有让近邻未来更集中，直接作为确定性回归输入不成立。

### 下一步候选任务

- 首选：`v254_probabilistic_multimodal_with_state_conditioning`。
  - 车辆轨迹仍是主输入。
  - 生理/风格不直接拼进确定性曲线回归。
  - 用生理/风格预测 mode prior、uncertainty、confidence 或 mixture weight。
  - 对 v252 高歧义样本输出多条可能未来，而不是强行输出一条平均曲线。
- 可选：先做 v254a 轻量原型，不训练大模型，只训练 mixture weight / uncertainty head，避免一开始复杂化。

### 禁止任务

- 不把旧 stage04 style 表直接回贴到当前样本。
- 不把 v253a 解释为“生理/风格无效”；只能说“当前表示和直接拼接消歧无效”。
- 不用 test 结果选择状态特征权重。
- 不回到 gate / 删除样本 / response type hard routing。

---

## 最新更新：2026-07-01 v252 输入相似样本未来分叉审计已完成

### 正在做任务

- 当前没有正在运行的训练进程。
- v252 已完成：这是可辨识性审计，不是新模型训练。

### 已完成任务

- 已新增并运行 `stage03_v252_input_similarity_future_divergence_20260701.py`。
- 已生成 v252 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701_pack.zip`
- 已输出核心表：
  - `tables\v252_neighbor_divergence_by_sample.csv`
  - `tables\v252_neighbor_detail.csv`
  - `tables\v252_summary_by_delay_bucket.csv`
  - `tables\v252_error_ambiguity_correlation.csv`
  - `tables\v252_high_ambiguity_error_overlap.csv`
  - `tables\v252_casebook_index.csv`
- 已输出核心图：
  - `figures\v252_error_vs_neighbor_future_divergence.png`
  - `figures\v252_neighbor_divergence_by_error_group.png`
  - `figures\v252_delay_future_divergence_summary.png`
  - `figures\v252_casebook_high_error_high_ambiguity.png`
  - `figures\v252_casebook_worst_regression_neighbors.png`

### 结果判断

- 全 test rolling sample：近邻未来两两 RMSE 均值 `0.707`，query-vs-neighbor 未来 RMSE 均值 `0.686`。
- 当前 v250 bad_top10 样本：近邻未来两两 RMSE 均值 `0.837`，query-vs-neighbor 未来 RMSE 均值 `1.063`。
- 0ms 原始锚点更明显：全体近邻未来两两 RMSE `0.836`，v250 bad_top10 0ms 为 `0.963`，query-vs-neighbor 未来 RMSE 为 `1.148`。
- `neighbor_future_to_query_mean_rmse` 与 `tail_rmse_v250` 的 Spearman 为 `0.495`；`neighbor_input_distance_mean` 与 `tail_rmse_v250` 的 Spearman 仅 `0.047`。
- 解释：差样本不是因为简单找不到相似样本，而是即使输入相似，未来真实行为也可能分叉。

### 下一步候选任务

- 首选：`v253_probabilistic_or_multimodal_prediction_design`。
  - 不再强行让模型输出唯一确定曲线。
  - 让模型输出多条候选轨迹、每条轨迹概率、以及不确定性区间。
  - 对可辨识样本保持单峰预测，对高歧义样本显式给出多未来。
- 可选：基于 v252 casebook 人工审查最典型高歧义样本，确认是否存在可补充的上下文变量。

### 禁止任务

- 不回到 v222a gate / 删除样本 / response type hard routing。
- 不把 oracle best anchor 或近邻真实未来当作可部署策略。
- 不基于 test 继续调输入通道。
- 不把 v252 写成模型性能提升；它是任务可辨识性证据。

---

## 最新更新：2026-07-01 v251 locked robustness audit 已完成，v250_minimal_lateral7 稳健性通过

### 正在做任务

- 当前没有正在运行的训练进程。
- v251 已完成：固定 v250 的 `v250_minimal_lateral7`，不重新训练、不改通道、不用 test 做模型选择。

### 已完成任务

- 已新增并运行 `stage03_v251_locked_robustness_v250_20260701.py`。
- 已生成 v251 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701_pack.zip`
- 已输出核心表：
  - `tables\v251_sample_locked_delta.csv`
  - `tables\v251_bucket_delay_locked_summary.csv`
  - `tables\v251_subject_locked_summary.csv`
  - `tables\v251_recording_locked_summary.csv`
  - `tables\v251_event_bootstrap_ci.csv`
  - `tables\v251_worst_regressions.csv`
  - `tables\v251_bad_top10_casebook_index.csv`
  - `tables\v251_next_decision.csv`
- 已输出核心图：
  - `figures\v251_test_bucket_delay_tail_delta.png`
  - `figures\v251_subject_bucket_tail_delta.png`
  - `figures\v251_bootstrap_ci_all_delay.png`
  - `figures\v251_bad_top10_casebook.png`
  - `figures\v251_worst_regression_casebook.png`

### 结果判断

- `locked_robustness_pass=True`。
- all/normal/observe_later_like/strong_steer 的所有关键 test delay tail delta 均小于 0。
- event-level bootstrap all-delay 95% CI 均排除 0 且为负：
  - all：`-0.0696 [-0.0926, -0.0467]`
  - normal_predictable：`-0.0673 [-0.0989, -0.0361]`
  - observe_later_like：`-0.0999 [-0.1608, -0.0386]`
  - strong_steer：`-0.0769 [-0.1151, -0.0387]`
  - bad_top10_v241：`-0.3036 [-0.3818, -0.2268]`
- subject/bucket win rate 为 `0.9375`。
- 主要边界：`cwh/strong_steer` all-delay mean tail delta 为 `+0.0047`，属于轻微回退；worst regressions 主要集中在少数 `tyy` 事件，也有个别 `cwh/rjy` 事件。

### 下一步候选任务

- 首选：`v252_mainline_candidate_pack_or_subject_level_retest`。
  - 固定 7 通道和 v250_minimal_lateral7 方案。
  - 打包为下一主线候选，包括模型定义、输入通道说明、locked robustness 证据、回退样本风险说明。
  - 做最终一致性审计：是否和 v238 original_remaining target、v241 backbone、v250 channel selection 逻辑完全一致。
- 可选：对 `cwh/strong_steer` 和 top worst regression 做人工 case review，确认是否存在特殊事件形态或通道精简导致的过冲。

### 禁止任务

- 不再基于 test 调通道。
- 不把 v251 直接写成 formal replacement；它是 locked robustness 证据。
- 不回到 v222a gate / 删除样本 / response type hard routing。
- 不忽略 worst regression；正式表述必须保留“均值和 bootstrap 稳健，但逐样本不是全胜”的边界。

---

## 最新更新：2026-06-30 v250 history-channel ablation 已完成，精简历史通道可作为下一候选

### 正在做任务

- 当前没有正在运行的训练进程。
- v250 已完成：只精简 `X_hist` 历史车辆通道，不改历史长度、不改道路预瞄、不改 phase/current、不改 point query。

### 已完成任务

- 已新增并运行 `stage03_v250_history_channel_ablation_20260630.py`。
- 已生成 v250 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630_pack.zip`
- 已训练候选：
  - `v250_drop_attitude_noise13`
  - `v250_lateral_core10`
  - `v250_minimal_lateral7`
- 已输出核心表：
  - `tables\v250_model_selection_validation_channel_ablation.csv`
  - `tables\v250_compare_vs_v241_original_remaining.csv`
  - `tables\v250_shape_summary.csv`
  - `tables\v250_input_neighborhood_ambiguity_by_channel.csv`
  - `tables\v250_input_neighborhood_ambiguity_summary.csv`
  - `tables\v250_next_decision.csv`
- 已输出核心图：
  - `figures\v250_tail_delta_by_channel_group.png`
  - `figures\v250_neighbor_ambiguity_by_channel_group.png`

### 结果判断

- best validation model 为 `v250_minimal_lateral7`，保留历史通道：`steering`、`speed_kmh`、`ay`、`yaw_rate`、`roll`、`lane_curvature`、`lateral_distance`。
- `v250_minimal_lateral7` validation：normal max tail delta vs v241 `-0.0813`，all mean tail delta `-0.1380`，observe_later mean tail delta `-0.1368`，strong mean tail delta `-0.1543`，val bad_top10 RMSE delta `-0.3247`。
- locked test：all 平均 tail delta `-0.0696`，normal_predictable `-0.0673`，observe_later_like `-0.0999`，strong_steer `-0.0769`，reverse_or_multi_correction `-0.0704`。
- shape：test bad_top10_v241 mean RMSE delta `-0.2433`；strong_steer range ratio gain `+0.0716`，slope ratio gain `+0.0440`。这说明通道精简不仅降低 RMSE，也部分改善强变化幅值/斜率不足。
- 输入邻域歧义：三组通道下 delay=0 v241 bad_top10 仍 `input_ambiguous_rate=1.0`。最低 neighbor future pairwise RMSE 为 `0.891`，来自 `v250_lateral_core10`；说明通道精简降低了噪声，但没有彻底解决一对多未来问题。

### 下一步候选任务

- 首选：`v251_locked_robustness_for_v250_minimal_lateral7`。
  - 固定 v250_minimal_lateral7，不再用 test 调通道。
  - 做 subject/session-level robustness、bootstrap/CI、逐样本回退审查、bad_top10 casebook。
  - 重点确认 7 通道结果是否稳定，而不是一次随机种子/训练波动。
- 次选：对 `v250_lateral_core10` 做邻域歧义复查，因为它的邻域未来分歧最低，但整体预测不如 minimal7。

### 禁止任务

- 不把 v250 直接写成 formal replacement。
- 不基于 test 继续手动挑通道。
- 不回到 v222a gate / 删除样本 / 轻量 residual / response type hard routing。
- 不把“邻域歧义仍在”误解成 v250 无效；它说明通道精简有效，但一对多问题还需要后续处理。

---

## 最新更新：2026-06-30 v249 shape-aware curve model 已完成，但不接受为新候选

### 正在做任务

- 当前没有正在运行的训练进程。
- v249 已完成：三个候选均从 v241 checkpoint 初始化，使用 validation-only 选择，test 只作 locked report。

### 已完成任务

- 已新增并运行 `stage03_v249_shape_aware_curve_model_20260630.py`。
- 已生成 v249 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630_pack.zip`
- 已训练候选：
  - `v249a_shape_loss_only`
  - `v249b_shape_aux_heads`
  - `v249c_shape_conditioned_residual`
- 已输出核心表：
  - `tables\v249_model_selection_validation_shape.csv`
  - `tables\v249_compare_vs_v241_original_remaining.csv`
  - `tables\v249_shape_summary.csv`
  - `tables\v249_per_sample_shape_delta_vs_v241.csv`
  - `tables\v249_input_neighborhood_ambiguity_audit.csv`
  - `tables\v249_next_decision.csv`
- 已输出核心图：
  - `figures\v249_shape_casebook_test_hard.png`
  - `figures\v249_tail_delta_by_bucket.png`

### 结果判断

- best diagnostic model 为 `v249c_shape_conditioned_residual`，best_epoch=`22`，best_val_loss=`1.155262`。
- v249c 通过 `noharm_vs_v236_pass=True` 和 `upgrade_vs_v241_pass=True`，但 `shape_gain_pass=False`，所以 `accepted_as_shape_candidate=False`。
- test 上，v249c 对 normal_predictable 有稳定改善；但 observe_later_like 在 0-800ms tail 多数变差，strong_steer 在 0-600ms tail 变差。
- hard case 的形状没有被修好：bad_top10_v241 mean_range_ratio 仍为 `0.625`，mean_slope_ratio 仍为 `0.535`；strong_steer 的 range/slope 相对 v241 进一步保守。
- 输入邻域歧义审计显示：test delay=0 的 v241 bad_top10 共 `19` 个样本，`19/19` 标为 `input_ambiguous`，邻居未来轨迹 pairwise RMSE 均值 `0.985`。这说明当前可见输入下存在明显一对多问题。

### 下一步候选任务

- 首选：`v249_error_review_or_input_ambiguity_followup`。
  - 审查 v249 casebook 与邻域审计，确认 hard case 是否确实是输入早期不可判别。
  - 若确认输入歧义存在，应考虑多解/分布式预测或不确定性建模，而不是继续把同一输入压成单条确定性均值曲线。
  - 同步检查是否有可在预测时真实可见的上下文特征可补充，例如更长历史、道路几何变化、速度/横向状态、驾驶员近期操作模式等。
- 次选：在 validation 上做 bounded ambiguity-aware objective，但必须避免硬 response-type route、oracle best-of-K、test 后验选择。

### 禁止任务

- 不把 v249 写成正式替代版本。
- 不继续简单加大 shape loss 权重来追 test bad case。
- 不回到 v222a gate / 删除样本 / 轻量 residual / response type hard routing。
- 不把 oracle best anchor 或 best-of-K 当作可部署结果。
- 不用 test 后验误差训练 selector 或调模型。

---

## 最新更新：2026-06-30 v248 best-anchor 后残余轨迹形状误差审查已完成

### 正在做任务

- 当前没有正在运行的训练进程。
- v248 已完成：读取 v247 fine-grid locked v241 prediction，对 current 0ms 与 best-anchor 后曲线做残余误差分解。

### 已完成任务

- 已新增并运行 `stage03_v248_best_anchor_residual_shape_audit_20260630.py`。
- 已生成 v248 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630_pack.zip`
- 已输出核心表：
  - `tables\v248_best_anchor_residual_decomposition.csv`
  - `tables\v248_peak_underestimation_table.csv`
  - `tables\v248_shape_error_categories.csv`
  - `tables\v248_anchor_vs_shape_summary.csv`
- 已输出核心图：
  - `figures\v248_best_anchor_still_bad_casebook.png`
  - `figures\v248_improved_but_still_wrong_casebook.png`
  - `figures\v248_peak_underestimation_casebook.png`
  - `figures\v248_error_decomposition_scatter.png`
  - `figures\v248_shape_category_summary.png`

### 结果判断

- 锚点有收益，但不是主要矛盾：test/bad_top10 从 `1.198` 降到 `0.616` 后仍然偏高。
- test/bad_top10 中 best-anchor 后仍有 `47.4%` 高于 `0.65`。
- still_bad 组的平均 range_ratio `0.466`、excursion_ratio `0.438`、slope_ratio `0.405`，说明模型主要低估幅值、低估斜率，把强转向/回正轨迹压成平滑小幅曲线。
- 下一步不应优先做 sequential anchor selector；应优先做 trajectory shape modeling。

### 下一步候选任务

- 首选：设计 `v249_shape_aware_curve_model`。
  - 方向一：完整曲线 decoder + peak amplitude loss + slope loss + tail/turning loss。
  - 方向二：基于 v241 的 shape residual corrector，专门修正峰值幅值、斜率、回正速度和反打转折。
  - 方向三：行为形状参数化 decoder，例如预测峰值时间、最大偏转、回正速度、最终偏移，再生成曲线。
- 在做 v249 前，可以先从 v248 casebook 中人工确认 still_bad 样本是否确实是峰值/回正/反打形状错误。

### 禁止任务

- 不把 v247 oracle best anchor 当可部署策略。
- 不继续只优化 anchor selector 来解释当前主要误差。
- 不回到 v222a gate / 删除样本 / 轻量 residual 路线。
- 不用 test 后验误差调模型超参或选择部署规则。
- 不把“等到 1000ms”当作正式预测方法。

## 最新更新：2026-06-26 v242 联合曲线解码模型已训练完成，但不替代 v241

### 正在做任务
- 当前没有正在运行的训练任务。`v242_joint_curve_decoder_20260626` 已完整运行并停止；本轮结论是“v242 相对 v236 有效，但没有通过相对 v241 的 upgrade 检查”。

### 已完成任务
- 已新增并运行：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v242_joint_curve_decoder_20260626.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\reports\v242_joint_curve_decoder_cn.md`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\v242_joint_curve_decoder_pack.zip`
- 已训练候选：`v242_joint_curve_h64_smooth002`、`v242_joint_curve_h96_smooth005`；best diagnostic model 为 `v242_joint_curve_h96_smooth005`。
- 已生成核心表：`v242_model_selection_validation_noharm.csv`、`v242_metrics_by_delay_and_bucket.csv`、`v242_compare_vs_v236_v239_v241_original_remaining.csv`、`v242_per_sample_delta_vs_v241.csv`、`v242_per_sample_delta_summary_vs_v241.csv`、`v242_worst_regressions_vs_v241.csv`、`v242_next_decision.csv`。

### 结果判断
- v242 相对 v236 仍然有用：best model 的 normal max tail delta vs v236 `-0.124933`，observe_later 0-800ms mean tail delta vs v236 `-0.269463`，strong 0-600ms mean tail delta vs v236 `-0.201033`。
- 但 v242 没有超过 v241：normal max tail delta vs v241 `+0.039176`，strong 400/1000ms mean tail delta vs v241 `+0.014069`，因此 `accepted_as_next_candidate=False`。
- test 逐样本层面也支持这个判断：all 中 `588/1104` 条相对 v241 tail 回退，strong_400_1000 中 `80/160` 条回退。
- 当前最强候选仍是 v241 的 `v241_tcn_mha_h96`，v242 只保留为“联合曲线输出不优于 v241”的诊断证据。

### 下一步候选任务
- 推荐下一步：`v243_manual_review_or_loss_redesign_for_sample_regressions`。
- 如果继续训练，应围绕 v241 的逐样本回退做 loss redesign，而不是继续扩大模型或基于 test 反调参数。
- 如果先审查，应固定 v241，进入 locked audit/casebook/robustness CI，避免被 v242 这一轮诊断分散主线。

### 阻塞/禁止任务
- 不要把 v242 写成 formal headline 或替代 v241。
- 不要基于 v242 test 结果继续手动调 smooth_weight、hidden_dim 或阈值。
- 不要回到 v222a gate/router/selector，也不要删除 observe_later_like 样本。
- 不要只因为 v242 相对 v236 好，就忽略它相对 v241 变差的事实。

---

## 最新更新：2026-06-26 v241 更强时序模型受控实验已完成，`v241_tcn_mha_h96` 可进入 locked audit

### 正在做任务
- 当前没有正在运行的训练任务。`v241_stronger_temporal_model_20260626` 已完整运行并停止；本轮结论是“更强时序模型通过 validation，可作为下一阶段候选”，但还不是 formal replacement。

### 已完成任务
- 已新增并运行：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v241_stronger_temporal_model_20260626.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\reports\v241_stronger_temporal_model_cn.md`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\v241_stronger_temporal_model_pack.zip`
- 已训练候选：`v241_tcn_mha_h64`、`v241_tcn_mha_h96`；两者均通过 stronger-candidate validation 检查，best diagnostic model 为 `v241_tcn_mha_h96`。
- 已生成核心表：`v241_model_selection_validation_noharm.csv`、`v241_metrics_by_delay_and_bucket.csv`、`v241_compare_vs_v236_v238_v239_original_remaining.csv`、`v241_per_sample_delta_vs_v239.csv`、`v241_per_sample_delta_summary_vs_v239.csv`、`v241_worst_regressions_vs_v239.csv`、`v241_next_decision.csv`。

### 结果判断
- `v241_tcn_mha_h96` 在 validation 上同时通过 v236 no-harm 和 v239 upgrade 检查：validation score `0.872780`，best_epoch `26`，best_val_loss `0.634865`。
- test 层面三个主 bucket 的全部 delay tail RMSE 都优于 v239，尤其 strong_steer 的 400/1000ms 也从 v240 的例外转为均值改善。
- 但逐样本仍有回退：all test 中 `368/1104` 条 tail 回退；strong_400_1000 中 `47/160` 条 tail 回退。因此本轮不能直接宣布正式替代，只能进入 locked audit。

### 下一步候选任务
- 推荐下一步：`v242_locked_test_report_for_stronger_temporal_candidate`。
- v242 应固定 `v241_tcn_mha_h96`，不重新训练、不调参、不用 test 反选；重点做 locked audit、casebook、逐样本回退解释、strong_400_1000 剩余回退审查和 robustness/CI。

### 阻塞/禁止任务
- 不要把 v241 直接写成 formal headline 或正式替代 v225/v226。
- 不要基于 v241 的 test 对照继续反调 hidden_dim、dropout、delay policy 或阈值。
- 不要回到 v222a gate/router/selector，也不要删除 observe_later_like 样本。
- 在逐样本回退没有审查前，不要只用均值改善来宣布 strong 类完全解决。

---

## 最新更新：2026-06-26 v240 locked attention audit 已完成，attention 候选保留但 strong 例外需人工复核

### 正在做任务
- 当前没有正在运行的训练任务。`v240_locked_attention_audit_20260626` 已完整运行并停止；本轮只做锁定审查，不训练、不调配置、不用 test 反选模型。

### 已完成任务
- 已新增并运行：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v240_locked_attention_audit_20260626.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\reports\v240_locked_attention_audit_cn.md`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\v240_locked_attention_audit_pack.zip`
- 已生成核心表：`v240_locked_overall_summary.csv`、`v240_subbucket_noharm_audit.csv`、`v240_per_sample_locked_metrics.csv`、`v240_worst_regressions.csv`、`v240_strong_400_1000_regressions.csv`、`v240_attention_casebook_index.csv`、`v240_attention_time_focus_summary.csv`、`v240_next_decision.csv`、`v240_split_integrity_check.csv`。
- 已生成 `21` 张 attention casebook 图，覆盖 observe_later 改善例、normal 改善例、strong 400/1000ms 回退例和 worst residual。

### 结果判断
- v239 attention 候选通过 locked audit 的主要条件：all / observe_later_like / normal_predictable 的 tail 对照均优于 v236 original_remaining。
- observe_later_like：mean tail delta `-0.142789`，max tail delta `-0.089425`，说明之前“晚观察/后移观察”这条思路在 attention 下确实有稳定收益。
- normal_predictable：mean tail delta `-0.069787`，max tail delta `-0.031002`，说明 v239 没有复现 v238 MLP 伤普通样本的问题。
- strong_steer：mean tail delta `-0.036692`，但 400ms/1000ms 存在例外；`strong_400_1000_positive_regression_cases` 有 `82` 条，最大变差 `+1.648318`，因此不能直接宣布 strong 已解决。

### 下一步候选任务
- 推荐下一步：`v241_attention_case_manual_review_and_robustness_ci`。
- 先人工审查 `figures\attention_casebook` 和 `tables\v240_strong_400_1000_regressions.csv`，确认 strong 例外到底是锚点/反打/多次修正问题，还是 attention 模型本身在中后段响应幅值上有偏差。
- 若人工审查确认不是明显锚点错误，再做 bootstrap/subject-level robustness CI，判断 v239 attention 是否具备进入正式替代前的统计稳定性。

### 阻塞/禁止任务
- 不要把 v239/v240 直接写成 formal headline 或正式替代 v225/v226。
- 不要基于 v240 的 test casebook 反调 attention 配置、delay policy 或阈值。
- 不要回到 v222a gate/router/selector，也不要删除 observe_later_like 样本。
- 不要在 strong 400/1000ms 例外未解释前扩展到 full Transformer/TCN 大模型。

---

## 最新更新：2026-06-26 v239 轻量 temporal attention + no-harm 约束实验已完成，attention 可作为下一候选
### 正在做任务
- 当前没有正在运行的训练任务。v239 已完整运行并停止；本轮结论是“轻量 attention 通过 validation no-harm，可作为下一候选，但仍不是 formal headline”。
### 已完成任务
- 已新增并运行：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v239_light_attention_noharm_20260626.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\reports\v239_light_attention_noharm_cn.md`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\v239_light_attention_noharm_pack.zip`
- 已训练 attention 候选：`v239_light_attention_h32`、`v239_light_attention_h48`。
- best diagnostic model：`v239_light_attention_h32`，validation score `1.077325`，validation no-harm pass=True。
- 已生成核心表：`v239_model_selection_validation_noharm.csv`、`v239_metrics_by_delay_and_bucket.csv`、`v239_compare_vs_v236_original_remaining.csv`、`v239_attention_training_history.csv`、`v239_next_model_decision.csv`。
### 结果判断
- attention 对 observe_later_like 有稳定收益：test original_remaining tail 在 0-1000ms 全部优于 v236。
- attention 通过 normal no-harm：`normal_predictable` test tail 在 0-1000ms 全部优于 v236，没有复现 v238 MLP 伤 normal 的问题。
- attention 对 strong_steer 整体有收益，但 400ms tail 轻微变差 `+0.009627`，1000ms tail 变差 `+0.048014`，需要在 v240 锁定报告里继续细查。
### 下一步候选任务
- 推荐下一步：`v240_locked_test_report_for_attention_candidate`。
- v240 应只做锁定报告、分桶图、坏例/改善例对照、attention casebook 和 guardrail 审查；不应继续扩大到 full Transformer、gate/router 或 response-type hard routing。
- v240 应重点确认：attention 是否真的改善锚点后移样本；normal no-harm 是否在更多子桶成立；strong_steer 400ms/1000ms 变差来自哪些事件。
### 阻塞/禁止任务
- 不要把 v239 attention 直接写成 formal headline 或正式替代 v225/v226。
- 不要基于 test 反调 attention 配置、delay policy 或阈值。
- 不要回到 v222a gate/router/selector，不要删除 observe_later_like。
- 不要从本轮直接跳到完整 Transformer/TCN 大模型。
---

## 最新更新：2026-06-26 v238 任务构造与小型 rolling 模型重搭已完成，下一步应做 no-harm 约束而不是扩大模型
### 正在做任务
- 当前没有正在运行的训练任务。v238 已完整运行并停止；本轮结论是“接受任务构造，不接受当前 selected MLP 作为正式替代模型”。
### 已完成任务
- 已新增并运行：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v238_task_model_redesign_20260626.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\reports\v238_task_model_redesign_cn.md`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\v238_task_model_redesign_pack.zip`
- 已完成任务重构：从 v236 的 `receding_2s` 改成 `original_remaining` masked point-level target，只训练 original anchor+2s 内的剩余部分。
- 已完成小模型训练：validation-only 在 point Ridge 和小 MLP 中选择，selected model 为 `v238_point_mlp_96x48_alpha1e-4`，validation score `1.290127`。
- 已生成核心表：`v238_task_construction_audit.csv`、`v238_point_training_rows_by_delay.csv`、`v238_model_selection_validation_only.csv`、`v238_metrics_by_delay_and_bucket.csv`、`v238_compare_v236_original_remaining.csv`、`v238_selected_per_sample_metrics.csv`、`v238_next_model_decision.csv`。
### 结果判断
- v238 证明 `original_remaining` 任务构造有价值：它把 delay 后跨入新行为阶段的点从主 loss 中排除，避免把 1000ms receding 失败错误归因给“晚观察无用”。
- 难例有收益：`observe_later_like` 在 0-800ms test tail 相对 v236 全部改善；`strong_steer` 在 0-600ms test tail 改善。
- 但当前模型不能正式使用：`normal_predictable` no-harm 未通过；`observe_later_like` 1000ms 变差；`strong_steer` 800/1000ms 变差。
### 下一步候选任务
- 推荐下一步：`v239_noharm_constrained_original_remaining_model`。
- v239 应继续保留 v238 的 `original_remaining` masked target，但必须把 validation no-harm 写进模型选择条件，尤其保护 `normal_predictable`。
- v239 应明确 late-delay policy：1000ms 不能继续和 0-800ms 用同一个 selected point model 硬混；它应先作为诊断或单独策略，不应直接进入正式主模型。
### 阻塞/禁止任务
- 不要把 v238 selected MLP 写成 formal headline 或正式替代 v225/v226。
- 不要基于 test 反调模型、delay policy 或阈值。
- 不要回到 v222a gate/router/selector，不要删除 observe_later_like。
- 不要因为 v238 有部分难例改善就直接扩大到 Transformer/TCN 大模型。
---

## 最新更新：2026-06-25 v237 rolling target / phase audit 已完成，v238 小 rolling 模型仅作为下一步候选
### 正在做任务
- 当前没有正在运行的训练任务。v237 已按 GPTPro 的 audit-only 指令完成并停止；`v237_next_model_decision.csv` 给出 `v238_allowed=True`，但本地尚未启动 v238。
### 已完成任务
- 已新增并运行：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v237_rolling_target_phase_audit_20260624.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\reports\v237_rolling_target_phase_audit_cn.md`
- 已生成审计 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\v237_rolling_target_phase_audit_pack.zip`
- 已完成两套评估口径：
  - `receding_2s`：从 observation_time 预测到 observation_time+2s，对应 v236 原定义。
  - `original_remaining`：只评估 observation_time 到 original_anchor+2s 的重叠部分，用于判断 delay 后是否变成新任务。
- 已完成 key tables：`v237_target_definition_sanity_check.csv`、`v237_receding_vs_original_remaining_metrics.csv`、`v237_observe_later_subbucket_profile.csv`、`v237_1000ms_failure_audit.csv`、`v237_ridge_underfit_audit.csv`、`v237_alpha_validation_curve_audit.csv`、`v237_next_model_decision.csv`。
### 结果判断
- target definition sanity 全部通过；v236 target/prediction 均在 `steering_delta_from_observation` 空间，未发现 delta/absolute 比较空间不一致。
- `observe_later_like` 的 1000ms receding 变差主要不是单纯“晚观察无用”，而是 delay 后 target horizon 进入了新阶段：1000ms receding tail RMSE `4.074430`，但 original_remaining tail RMSE `1.199416`，且 `1000ms` failure audit 的 new phase 命中率为 `0.888889`。
- `strong_steer` 的 rolling 收益保持：test receding tail RMSE 从 0ms `1.018893` 降到 800ms `0.819825`、1000ms `0.814272`。
- v236 Ridge 有 underfit 证据：旧 formal reference 明显优于 v236 0ms Ridge；`observe_later_like` 与 `strong_steer` 的预测峰值存在明显 shrinkage；alpha validation-only 选择在最大 alpha=`1000` 边界。
### 下一步候选任务
- 若继续执行，下一步可让 GPTPro/用户确认是否启动 `v238_small_rolling_model`：只允许小模型、validation-only 选择、按 delay/bucket 分开报告，且必须保留 v237 guardrails。
- v238 不能基于 test 选择 delay 或模型配置；不能写成 formal headline；不能回到 v222a gate/router/selector；不能删除 `observe_later_like`。
### 阻塞/禁止任务
- 不要在未确认前自动训练 v238。
- 不要进入 Transformer/TCN 大模型。
- 不要做新 gate/router/selector、不要做新 tau/threshold 搜索。
- 不要把 mixed-delay RMSE 或 diagnostic subbucket 结果写成正式模型能力。
---

## 最新更新：2026-06-24 连续车辆源数据审计已完成，下一步不要混用主车辆层和补充车辆层
### 正在做任务
- 解读 `vehicle_source_audit_20260624`：把源级车辆数据质量、目录命名混杂、主/补充层 lineage 差异转成后续样本构建和模型审计约束。
### 已完成任务
- 已新增并运行 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\scripts\vehicle_source_audit_20260624.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\vehicle_source_audit_20260624_cn.md`
- 已完成连续车辆源文件审计：候选文件 `358` 个，纳入车辆审计 `182` 个，覆盖 `18` 名被试、`91` 个记录键、约 `25.31` 小时。
- 已生成关键表：
  - `tables\vehicle_file_inventory.csv`
  - `tables\source_layer_summary.csv`
  - `tables\file_vehicle_quality_summary.csv`
  - `tables\vehicle_numeric_column_summary.csv`
  - `tables\recording_cluster_summary.csv`
  - `tables\subject_vehicle_summary.csv`
  - `tables\road_type_summary.csv`
  - `tables\vehicle_source_audit_findings.csv`
- 已生成图：`vehicle_recording_duration_hist.png`、`vehicle_duration_by_subject.png`、`vehicle_audit_flag_counts.png`。
- 已完成验证：`py_compile` 通过；完整运行 `182/182` 个纳入文件；`errors=[]`；3 张图像均非空。
### 结果判断
- 主 `vehicle_aligned_cleaned` 层是当前更可靠的车辆源：`91` 个文件，median dt=`5ms`，关键车辆字段高缺失文件 `0` 个。
- `车辆清理后` 目录不能按文件名直接当车辆源：另有 `82` 个 PhysioLAB 生理文件和 `85` 个 EEG/加速度文件也在 vehicle 命名/目录下。
- 补充 `(2)_vehicle_fixed_200Hz` 层不能和主车辆层直接混用：关键字段高缺失文件 `83/91`，nominal Hz 异常 `45/91`，median dt 不接近 `5ms` 的文件 `42/91`。
- 主层 road reference 需要复核：`32/91` 个主车辆文件 `ref_nn_ok_rate<95%`，这会影响 road/curve 分层和横向偏移判断。
### 待做任务
- 后续所有样本重建、锚点窗口和模型输入先固定唯一车辆源层，默认优先使用主 `vehicle_aligned_cleaned` 层。
- 把 `recording_cluster_summary.csv` 里的行数/哈希不一致记录映射回既有坏样本，确认失败是否集中来自源层混用或道路参考低覆盖。
- 对 `ref_nn_ok_rate<95%` 的主车辆记录做道路参考复核，再决定是否进入 curve/road 分层训练或人工审核。
### 阻塞/禁止任务
- 不要只按 `*_vehicle_fixed_200Hz.csv` glob 读取车辆数据。
- 不要把主 `vehicle_aligned_cleaned` 层和补充 `(2)_vehicle_fixed_200Hz` 层混在同一训练/审计入口里。
- 不要把补充层字段缺失解释成主车辆层质量差。

---

## 最新更新：2026-06-24 v236 rolling/reanchor 数据集与小基线已完成，下一步先审查 observe_later_like 未改善原因
### 正在做任务
- 解读 `v236_rolling_reanchor_dataset_and_baseline_20260624`：确认 rolling observation 是否真的改善困难样本，并决定是否允许进入更强模型。
### 已完成任务
- 已新增并运行 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v236_rolling_reanchor_dataset_and_baseline_20260624.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624`
- 已生成 `7002` 个 rolling 样本，覆盖 loose 主池 `1167` 个唯一事件，每个事件 `0/200/400/600/800/1000ms` 六个 delay。
- 已完成 joint Ridge 小基线训练，selected alpha=`1000`，只用 validation 选择。
- 已生成必需输出：
  - `tables\v236_rolling_sample_manifest.csv`
  - `tables\v236_delay_sample_counts.csv`
  - `tables\v236_train_val_test_event_split_check.csv`
  - `tables\v236_baseline_metrics_by_delay.csv`
  - `tables\v236_baseline_metrics_by_delay_and_bucket.csv`
  - `tables\v236_observe_later_improvement_curve.csv`
  - `tables\v236_strong_event_improvement_curve.csv`
  - `tables\v236_normal_sample_noharm_check.csv`
  - `tables\v236_metric_vs_old_0ms_formal_reference.csv`
  - `reports\v236_rolling_reanchor_baseline_cn.md`
  - `logs\guardrail_check.json`
  - `logs\leakage_check.json`
  - `logs\file_inventory.json`
  - `v236_rolling_reanchor_dataset_and_baseline_pack.zip`
- 已完成验证：`py_compile` 通过；完整运行通过；必需文件 `missing=[]`；guardrail `pass`；leakage `pass`；同一 event_uid 跨 split 数 `0`；ZIP `bad=None`。
### 结果判断
- `strong_steer` 桶明显受益：test tail mean 从 `0ms=0.961224` 降到 `800ms=0.695632`、`1000ms=0.702697`；strong-under rate 从 `0.7625` 降到约 `0.2125`。
- `observe_later_like` 未满足成功条件：`200ms` 只小幅改善（tail mean `1.100397 -> 1.060875`），但 `400/600/800/1000ms` 没有持续下降，`1000ms` 明显变差。
- v236 0ms baseline 也弱于旧 formal：all test sample RMSE `0.641212` vs old `0.468061`，因此不能把 v236 当成 formal 替代。
### 待做任务
- 先审查 observe_later_like 不改善的原因：target 是否应为 delta-from-observe 还是 absolute future；该桶是否混入反打/多次修正 outlier；Ridge 表达能力是否过弱；1000ms 是否跨入新的行为阶段。
- 在确认 v236 数据/target 口径无误前，不进入更大 Transformer/TCN。
### 阻塞/禁止任务
- 不回到 v222a gate/router/selector。
- 不删除 observe_later_like。
- 不把 mixed-delay RMSE 当 formal headline。
- 不改变 v225/v226 formal headline。

---

## 最新更新：2026-06-24 v235 删除 observe_later_like 样本后的受控重训已完成，下一步不要把“删难样本”当正式方法
### 正在做任务
- 解读 `v235_remove_observe_later_retrain_20260624`：区分“删除样本导致测试集变容易”和“删除后重训模型本身是否变强”。
### 已完成任务
- 已新增并运行 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v235_remove_observe_later_retrain_20260624.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624`
- 已生成关键表：
  - `tables\v235_removed_sample_counts.csv`
  - `tables\v235_validation_selection_filtered.csv`
  - `tables\v235_selected_metrics_filtered.csv`
  - `tables\v235_old_selected_metrics_filtered.csv`
  - `tables\v235_selected_metrics_removed_holdout.csv`
  - `tables\v235_comparison_summary.csv`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\reports\v235_remove_observe_later_retrain_cn.md`
- 已生成对比图：
  - `figures\v235_test_rmse_comparison.png`
  - `figures\v235_test_tail_rmse_comparison.png`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\v235_remove_observe_later_retrain_pack.zip`
- 已完成验证：`py_compile` 通过；完整运行通过；feature schema guard `pass`；selection 使用 filtered validation only；图像非空；ZIP `testzip bad=None`。
### 结果判断
- loose pool 删除 `121/1167` 个样本，strict pool 删除 `117/963` 个样本。
- 旧 v222a selected full test RMSE：loose `0.555940`，strict `0.571966`。
- 旧模型在删除后的同一保留 test 子集上：loose `0.482685`，strict `0.506547`。
- 删除后重训：loose `0.474318`，strict `0.504151`。
- 结论：删除这些样本会明显改善保留测试集指标，但重训相对旧模型同一过滤子集只小幅改善；因此这更像数据分布诊断，不是可直接宣称的方法提升。
### 待做任务
- 不把 observe_later_like 样本简单永久删除；应继续走“人工确认后短观察层/后移观察点”或“重锚定后重建 label window”的路线。
- 若要继续做模型方法，应把 observe_later_like 作为单独任务层或特殊评估桶，而不是混入同一个 pure pre-anchor 指标里。
### 阻塞/禁止任务
- 不把 v235 指标替代 formal headline。
- 不把删难样本后的指标写成模型能力提升。
- 不把 removed 样本丢弃后不解释；被删除 test 样本仍是行为预测中真正困难的可观测性/锚点问题。

---

## 最新更新：2026-06-24 v234 短观察后预测评估层已构建，下一步人工选择观察延迟
### 正在做任务
- 人工审核 `v234_short_observation_manual_review_template.csv`，判断每个 `observe_later_review` 样本是否采用短观察后预测层，以及采用 `0.5s/1.0s/1.5s/2.0s` 哪个观察延迟。
### 已完成任务
- 已新增并运行 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v234_short_observation_prediction_layer_20260624.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624`
- 已生成关键表：
  - `tables\v234_short_observation_layer_definition.csv`
  - `tables\v234_short_observation_layer_assignments.csv`
  - `tables\v234_short_observation_target_curves.csv`
  - `tables\v234_short_observation_context_grid.csv`
  - `tables\v234_short_observation_manual_review_template.csv`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\reports\v234_short_observation_prediction_layer_cn.md`
- 已生成图目录和拼接图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\figures`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\v234_short_observation_prediction_layer_pack.zip`
- 已完成验证：`py_compile` 通过；完整运行通过；`errors=[]`；图像非空；ZIP `testzip bad=None`。
### 待做任务
- 人工逐行填写 `v234_short_observation_manual_review_template.csv`：
  - `human_layer_decision`
  - `human_selected_observe_delay_s`
  - `human_use_for_training`
  - `human_note_cn`
- 对默认 0.5s 层不合理的样本，改选 1.0s、1.5s 或拒绝短观察层。
- 人工确认后，再进入 v235：生成短观察层数据清单或评估对应模型。
### 阻塞/禁止任务
- 不把旧 formal prediction 硬评到短观察层。
- 不自动修改标签。
- 不训练新模型、不改 formal headline。
- 不重启硬响应类型分类路线。
- 不把简单多候选轨迹输出作为下一步主线。

---

## 最新更新：2026-06-24 v233 自适应锚点 / 观察时长策略审核包已完成，下一步人工区分“提前重锚定”和“后移观察点”
### 正在做任务
- 人工审核 `v233_anchor_observation_policy_review_table.csv` 和策略图，确认哪些样本应提前重锚定，哪些样本应进入“短观察后预测 / 后移观察点”评估层。
### 已完成任务
- 已新增并运行 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v233_adaptive_anchor_observation_policy_20260624.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624`
- 已生成关键表：
  - `tables\v233_anchor_observation_policy_table.csv`
  - `tables\v233_anchor_observation_policy_review_table.csv`
  - `tables\v233_observe_delay_grid.csv`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\reports\v233_adaptive_anchor_observation_policy_cn.md`
- 已生成策略图目录和拼接图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\figures`
- 策略分布：`observe_later_review=10`、`reanchor_earlier_review=5`、`reanchor_earlier_or_ambiguous_review=6`、`large_change_standard_or_ambiguous=1`、`standard_anchor_review=7`。
- 已完成验证：`py_compile` 通过；完整运行通过；`errors=[]`；图像非空；ZIP `testzip bad=None`。
### 待做任务
- 优先人工看 `observe_later_review` 的 10 个样本，确认旧锚点前是否确实看不出区别，以及后移 `0.5s/1.0s/1.5s` 后是否出现足够可见证据。
- 对 `observe_later_review` 样本填写 `human_policy_decision`、`human_observe_delay_s`、`human_use_for_training`、`human_note_cn`。
- 若确认后移观察点合理，下一步应建立单独的“短观察后预测”评估层，不与纯提前预测混在同一指标。
- 对 `reanchor_earlier_review` 和 `reanchor_earlier_or_ambiguous_review` 继续沿用 v232 的人工重锚定审核流程。
### 阻塞/禁止任务
- 不把后移观察点当作统一锚点修正；它是任务可观测性层级，不是事件起点重标注。
- 不自动修改训练标签。
- 不训练新模型、不改 formal headline。
- 不重启硬响应类型分类路线。
- 不把简单多候选轨迹输出作为下一步主线。

---

## 最新更新：2026-06-24 v232 过晚锚点重锚定候选审核包已完成，下一步人工确认候选锚点
### 正在做任务
- 人工审核 `v232_late_anchor_reanchor_candidates_20260624` 的 `v232_reanchor_candidate_review_table.csv` 和对应候选图，确认候选新锚点是否确实比旧锚点更接近事件起点。
### 已完成任务
- 已新增并运行 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v232_late_anchor_reanchor_candidates_20260624.py`
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624`
- 已对 v230 casebook + v231 六样本的 `29` 个唯一样本完成原始车辆信号打分。
- 已生成 `11` 个 P0/P1/P2 重锚定人工审核候选：P0=1、P1=4、P2=6。
- 已生成关键表：
  - `tables\v232_target_samples.csv`
  - `tables\v232_reanchor_candidate_all_scored.csv`
  - `tables\v232_reanchor_candidate_review_table.csv`
  - `tables\v232_reanchor_grid_0p05s.csv`
  - `tables\v232_reanchor_key_points.csv`
- 已生成中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\reports\v232_late_anchor_reanchor_candidates_cn.md`
- 已生成候选图目录和拼接图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\figures`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\v232_late_anchor_reanchor_candidates_pack.zip`
- 已完成验证：`py_compile` 通过；脚本完整运行通过；`errors=[]`；图像非空；ZIP `testzip bad=None`。
### 待做任务
- 先人工看 P0/P1 候选：`rjy...010`、`rjy...041`、`rjy...040`、`rjy...032`、`tyy...033`。
- 对每个候选填写 `human_decision`、`human_corrected_anchor_s`、`human_use_for_training`、`human_note_cn`。
- 若算法候选过早或过晚，人工改写 `human_corrected_anchor_s`，不要直接采用算法候选。
- 人工确认后，再启动下一步 label window 重建；未确认前不改训练标签。
### 阻塞/禁止任务
- 不自动采用候选新锚点。
- 不训练新模型、不改 formal headline、不基于 test 重新调阈值。
- 不重启硬响应类型分类路线。
- 不把简单多候选轨迹输出作为下一步主线。

---

## 最新更新：2026-06-24 v231 人工反馈再修正：过晚锚点需要重锚定，多候选轨迹也不作为主线
### 正在做任务
- 从 `v231_worst_case_anchor_context_20260624` 进入过晚锚点重锚定准备：已确认 `rjy...010` 属于锚点晚了，后续不能只标记，需要生成候选新锚点、移动秒数和证据字段供人工确认。
### 已完成任务
- 已从 v225/v230 差样本中固定 6 个代表性样本，并从原始车辆 CSV 调取 `anchor_s ±8s` 信号上下文。
- 已生成信号对齐关键时刻表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_key_points.csv`
- 已生成 0.1 秒稀疏窗口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_window_sparse_8s.csv`
- 已生成原始 200Hz 密集窗口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_window_dense_pm3s.csv`
- 已生成 6 张锚点上下文图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures`
- 已生成中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\reports\v231_worst_case_anchor_context_cn.md`
- 已新增用户反馈修正说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\reports\v231_user_feedback_method_correction_cn.md`
- 已将“一次性输出多个候选轨迹也已尝试且效果不好，即使 best candidate 仍有偏差”写入覆盖表。
### 待做任务
- 对 `rjy...010`：按“人工确认锚点晚”处理，下一步需要重锚定，生成候选新锚点和证据，不再只停留在标记层面。
- 对 `rjy...041`：继续人工确认是否也是锚点落在事件中段。
- 对 `rjy...023`、`tyy...026`、`rjy...031`：不要简单拆成硬响应类型分类，也不要把简单多候选轨迹输出作为主线；先确认锚点和目标窗口无误，再讨论偏差校正、连续相位/延迟或对齐鲁棒损失。
- 若人工确认锚点无误，则下一步回到方法提升：不要把这些样本只写成失败机制，而要作为行为预测模型改进的困难样本集。
### 阻塞/禁止任务
- 本轮不训练新模型、不改 formal headline、不基于 test 重新调阈值。
- 不把 v231 当作失败机制论文产物；它是方法提升前的人工审核证据包。
- 不重启“先硬判断响应类型，再按类型预测轨迹”的路线；该路线此前已尝试过，且存在响应类型判断错误导致后续轨迹整体错误的结构性风险。
- 不把“一次性输出多个候选轨迹”作为下一步主线；该路线此前也已尝试过，且即使 best candidate 仍有偏差。

---

## 最新更新：2026-06-23 v230 失败案例人工复核 / 论文案例证据包已完成，下一步人工复核

### 正在做任务
- v230 已完成并停止。当前不再进入模型工作，下一步是人工阅读 casebook、填写 `v230_manual_review_template.csv`，再整理论文失败案例小节。

### 已完成任务
- 已归档 GPTPro v230 指令：
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v230_casebook_gptpro_prompt.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v230_casebook_gptpro_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v230_casebook_gptpro_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v230_casebook_gptpro_action_items.md`
- 已新增并运行 `stage03_v230_failure_case_manual_review_casebook_20260623.py`。
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623`
- 已生成关键表：
  - `tables/v230_case_selection_index.csv`
  - `tables/v230_manual_review_template.csv`
  - `tables/v230_failure_casebook_table.csv`
  - `tables/v230_bucket_to_claim_mapping.csv`
  - `tables/v230_case_figure_inventory.csv`
  - `tables/v230_formal_boundary_check.csv`
- 已生成关键报告：
  - `reports/v230_failure_case_manual_review_casebook_cn.md`
  - `reports/v230_advisor_discussion_notes_cn.md`
  - `reports/v230_paper_failure_case_section_draft_cn.md`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\v230_failure_case_manual_review_casebook_pack.zip`
- 已完成验证：`py_compile` 通过；完整运行通过；ZIP `bad_file=None`、文件数 `103`；required files `[]`；guardrail `pass=True`；consistency `pass=True`；forbidden hits `[]`；人工复核字段全空。

### 待做任务
- 人工打开 `figures/selected_casebook_figures/` 下的 case 图。
- 人工填写 `tables/v230_manual_review_template.csv` 中的复核字段。
- 根据人工复核结果修改 `reports/v230_paper_failure_case_section_draft_cn.md`。

### 阻塞/禁止任务
- v230 完成后不自动进入任何模型训练。
- 继续禁止 v222b/v223、新 gate/router、新 tau/threshold、新预测、formal headline 改动和 test-based retuning。
---

## 最新更新：2026-06-23 v229 两个月路线经验复盘与失败分类包已完成，下一步先让 GPTPro 审路线边界

### 正在做任务
- 当前转入路线复盘/失败分类阶段：v229 已整理 v220 两个月阶段经验、v225 formal 失败样本与 selector/candidate 诊断、v228 最终锁定指标，形成可直接交给 GPTPro 的中文复盘 prompt。

### 已完成任务
- 已新增并运行 `stage03_v229_two_month_lessons_failure_taxonomy_20260623.py`。
- 已生成输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623`
- 已生成主报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\reports\v229_two_month_lessons_failure_taxonomy_cn.md`
- 已生成 GPTPro 中文提问稿：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\reports\v229_gptpro_next_prompt_cn.md`
- 已生成关键表：
  - `tables/v229_phase_lessons_table.csv`
  - `tables/v229_failure_taxonomy_by_pool_event.csv`
  - `tables/v229_top_tail_failure_cases.csv`
  - `tables/v229_bucket_risk_summary.csv`
  - `tables/v229_selector_candidate_diagnosis.csv`
  - `tables/v229_next_action_decision_matrix.csv`
- 已生成 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\v229_two_month_lessons_failure_taxonomy_pack.zip`
- 已完成验证：`py_compile` 通过；脚本完整运行通过；ZIP `bad_file=None`、文件数 `15`；必需文件缺失 `[]`；guardrail `pass=True`。

### 待做任务
- 如果继续 GPTPro 闭环，优先发送 v229 中文复盘 prompt，而不是直接请求新模型指令。
- 要求 GPTPro 先判断：
  - 是否进入写作/结果整理；
  - 是否只允许失败样本 taxonomy 与人工复核；
  - 是否继续禁止 v222b/v223、新 gate/router、新 tau/threshold、test-based retuning；
  - 若允许新实验，必须只给一个窄范围任务、明确 stop condition 和验收命令。

### 阻塞/禁止任务
- 没有 GPTPro 明确 bounded 指令前，不启动 v222b/v223、大 gate/router、新 tau/threshold、新模型训练或 formal headline 改动。
- 不把 oracle、true label、fallback、diagnostic-only 行写成 formal evidence。
- 不把 `W3_B4_original_soft` 写入 formal leaderboard、formal gate、formal oracle、usage table 或 selected config。
---

## 最新更新：2026-06-22 v226 robustness / CI audit 已完成，下一步报告 GPTPro

### 正在做任务
- Codex-GPTPro 闭环继续运行：v226 formal robustness / confidence-interval audit 已完成，下一步把 v226 pack、CI、readiness、guard/ZIP 验证摘要报告给 GPTPro，等待下一轮 bounded 指令。

### 已完成任务
- 已新增并运行 `stage03_v226_formal_robustness_ci_audit_20260622.py`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\v226_formal_robustness_ci_audit_pack.zip`
- 验证：`py_compile`、完整运行、ZIP `bad_file=None`、required files `[]`、metric reproduction、leakage guard、forbidden scan、table alignment、figure count 全部通过。

### 禁止任务
- 不进入 v222b/v223、新 tau、新 gate/router 或 test-based retuning，除非 GPTPro 给出新的 bounded 指令且满足当前 guardrail。

## 最新更新：2026-06-22 v225 evidence pack 已完成，下一步报告 GPTPro 获取下一轮 bounded 指令

### 正在做任务
- Codex-GPTPro 闭环继续运行：已完成 GPTPro v225 要求的 `formal route reconstruction evidence pack`，下一步把 v225 pack、验证结果、formal lock 和 failure evidence 报告给 GPTPro，请 GPTPro 给下一轮 bounded 指令。

### 已完成任务
- 已归档 GPTPro v225 指令、采纳决策和执行项：
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v225_evidence_pack_gptpro_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v225_evidence_pack_gptpro_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v225_evidence_pack_gptpro_action_items.md`
- 已新增并运行 `stage03_v225_formal_route_reconstruction_evidence_pack_20260622.py`。
- 已生成 v225 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622`。
- 已确认 formal lock：
  - `loose_main_pool=avg_joint_focus`
  - `strict_main_pool=peak_floor_090`
  - v222a residual/no-harm/oracle safe gate/ridge residual 等均为 diagnostic-only，不进入 formal 表。
- 已复现 locked test formal 指标：
  - loose `avg_joint_focus`：RMSE `0.544884`，tail `0.629752`；
  - strict `peak_floor_090`：RMSE `0.571770`，tail `0.658306`；
  - 复现误差均小于 `1e-5`。
- 已完成验证：`py_compile`、完整脚本运行、ZIP `bad_file=None`、必需文件无缺失、metric reproduction pass、leakage guard 全 pass、forbidden scan pass、table alignment pass、figure 抽检正常。

### 待做任务
- 准备并发送下一轮 GPTPro prompt，必须包含：
  - v225 pack 路径；
  - formal model lock；
  - locked test 指标复现；
  - per-sample / bucket / route-event / failure-case 证据入口；
  - diagnostic-only v222a closeout 摘要与 excluded diagnostic audit；
  - ZIP、guard、forbidden scan、table alignment 和 figure count 验证结果；
  - 本轮未训练、未调 tau/threshold、未创建 gate/router、未运行 v222b/v223 的边界说明。
- 等待 GPTPro 给出下一轮 bounded 指令；若新指令涉及 v222b/v223、新 gate/router、新 tau 或 test-based retuning，必须先确认它是否满足当前 guardrail 与 stop condition。

### 阻塞/禁止任务
- 不进入 v222b neural gate / neural soft fusion。
- 不进入 v223 new candidate generator / mechanism Transformer。
- 不继续做 v222a gate_v2、新 tau、新 multi-router 或 test-based config。
- 不把 oracle / true label / fallback / diagnostic-only row 写成 deployable model 或 formal headline。
- 不把 `W3_B4_original_soft` 写入 formal leaderboard、formal oracle、formal gate、usage table 或 selected config。

---

# 最新更新：2026-06-23 heartbeat 仍未取得 GPTPro 新指令

## 正在做任务
- Codex-GPTPro 闭环继续处于外部通道阻塞状态。桌面端只看到 handoff prompt 和“已停止思考”，Chrome bridge 无法验证 Pro/进阶模式并拒绝发送 v227 prompt。

## 已完成任务
- 已新增 2026-06-23 heartbeat 阻塞归档：
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v227_heartbeat_gptpro_response_blocked.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v227_heartbeat_gptpro_decision_blocked.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v227_heartbeat_gptpro_action_items_blocked.md`
- 已复核 v227 ZIP：存在、可读，`bad_file=None`，共 35 个文件。

## 待做任务
- 等 GPTPro 通道恢复后，发送 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\reports\v227_next_gptpro_prompt_ascii.md`。
- 若获得有效 GPTPro 回复，先归档 raw response / decision / action items，再筛掉违反 guardrail 的建议，只执行一个 bounded 安全指令。

## 禁止任务
- 在没有有效 GPTPro 新指令前，不进入 v222b/v223、新 tau/threshold、新 gate/router/selector、新模型训练、formal headline 改动或 test-based retuning。

---

## 最新更新：2026-06-22 v221 已完成，下一步进入 v222a 轻量融合准备

### 正在做任务
- v221 统一评估框架已跑通，当前应基于 `v221_candidate_decision_summary.csv` 固定下一步 v222a 的候选和停止线。

### 已完成任务
- 已新增并运行 `stage03_v221_formal_model_leaderboard_20260622.py`。
- 已统一读取 v216/v217/v218/v219 的逐样本表和整体指标表。
- 已生成 formal/diagnostic 分层结果，正式榜单排除了 oracle、fallback、`W3_B4_original_soft` 等禁用候选。
- 已输出中文报告、HTML、关键 CSV 和 ZIP 包。

### 待做任务
- 做 v222a 之前，先读取：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_candidate_decision_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_model_bucket_metrics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_noharm_vs_reference.csv`
- v222a 第一版只做轻量候选软融合与受限残差，不训练大 Transformer：
  - 可用主池整体基准：`avg_joint_focus = 0.5448793008861739`；
  - 严格主池整体基准：`peak_floor_090 = 0.5717751408320051`；
  - 低估控制候选：`peak_floor_090` 和 `ridge_residual_peakfloor`；
  - 普通样本稳定候选：可用主池 `global_blend`，严格主池 `avg_joint_focus`。

### 阻塞/禁止任务
- 暂不进入 v222b 神经软融合、v223 机制感知联合 Transformer 或 v224 消融。
- 暂不做硬切换 router。
- 暂不把 v218 强峰值训练作为主线。
- v222a 推理特征必须显式排除 `sample_id`、`event_uid`、`split`、`subject`、true/oracle labels、RMSE、低估/错侧等 target-derived 或 candidate true metric 字段。

---

# 当前任务队列

## 最新更新：2026-06-22 v222a closeout 已完成，下一步报告 GPTPro 获取新指令

### 正在做任务
- Codex-GPTPro 闭环继续运行：已完成 GPTPro 要求的 `v222a closeout + candidate gap audit`，下一步把 closeout pack、核心结论和 future route decision 报告给 GPTPro，等待新的 bounded 指令。

### 已完成任务
- 已归档 GPTPro closeout 指令、采纳决策和执行项：
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v222a_closeout_gptpro_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v222a_closeout_gptpro_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v222a_closeout_gptpro_action_items.md`
- 已新增并运行 `stage03_v222a_closeout_candidate_gap_audit_20260622.py`。
- 已生成 closeout 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622`。
- 已确认 formal headline：
  - `loose_main_pool=avg_joint_focus`
  - `strict_main_pool=peak_floor_090`
  - v222a residual / no-harm gate / oracle safe gate 均为 diagnostic-only。
- 已确认 future route：
  - `v222b_allowed=False`
  - `v223_allowed=False`
  - 当前主问题更偏 learned gate/selector 泛化失败，而不是 high-tail 样本中候选池大面积缺曲线。
- 已完成验证：`py_compile`、完整脚本运行、ZIP `bad_file=None`、必需文件无缺失、guard 全 pass、禁用名检查无命中、case 图抽检正常。

### 待做任务
- 准备并发送下一轮 GPTPro prompt，必须包含：
  - closeout pack 路径；
  - formal headline decision；
  - v222a stop evidence；
  - oracle-vs-learned gap；
  - taxonomy 主导类型；
  - `future_route_decision.csv` 的 `v222b_allowed=False`、`v223_allowed=False`；
  - 验证命令与 ZIP 校验结果。
- GPTPro 下一步如果要求继续，必须给出 bounded 指令、明确 stop condition，并且不能要求 test retuning、v222b/v223 训练、multi-router 或 forbidden inference fields。

### 阻塞/禁止任务
- 不进入 v222b neural gate / neural soft fusion。
- 不进入 v223 new candidate generator / mechanism Transformer。
- 不继续做 v222a gate_v2、新 tau、新 multi-router 或 test-based config。
- 不把 oracle / true label / fallback / diagnostic-only row 写成 deployable model 或 formal headline。
- 不把 `W3_B4_original_soft` 写入 formal leaderboard、formal oracle、formal gate、usage table 或 selected config。

---

## 最新更新：2026-05-20 12:15

### 正在做任务
- 当前没有服务器后台训练任务；v0.5 服务器处理后样本重筛与被试划分旧流程车辆-only训练已经完成，进入结果看图和解释阶段。

### 已完成任务
- 已把服务器 `vehicle_aligned_cleaned.csv` 作为数据入口，重新按之前分类要求筛选 1574 个初始 episode。
- 已构建 v0.5 旧流程 manifest，并按被试划分：test=cwh/gf/tyy，val=byx/gzj/yyl，其余为 train。
- 已完成 FAIR09/E1 车辆-only 粗细双头训练，不加连续风格、生理、脑电或教师蒸馏。
- 已生成预测图、逐样本指标、分被试/分道路/分机制标签表，并拉回本地。

### 待做任务
- 人工查看本轮预测总览图和 12 张固定样本预测图，判断物理意义是否比旧样本更合理。
- 重点查看 `tyy` 测试样本，因为它的样本级 RMSE 和尾段误差明显高于 cwh/gf。
- 如果图像质量可接受，再在同一 v0.5 样本定义上补“车辆 + 连续风格”对照。
- 如果图像仍然出现明显错侧、幅值压缩或锚点偏移，先回到 v0.5 样本分层和锚点规则，不急着加生理。

### 阻塞任务
- 暂不声称新样本集最终正确；本轮测试集只有 163 个样本，且是固定被试划分。
- 暂不进入生理/脑电增量验证；必须先确认车辆-only 在新样本定义下的预测图是否站得住。

### 需要服务器的任务
- 当前无正在运行的服务器任务。
- 后续若补车辆+连续风格或多 seed，可继续使用服务器。

### 不需要服务器的任务
- 预测图人工复核、分被试表格解释、中文汇报材料整理。

---

## 最新更新：2026-05-19 22:05

### 正在做任务
- GPU 快速筛选已经完成；当前进入结果解释和候选样本复核准备。

### 已完成任务
- 已停止 CPU 版筛选任务。
- 已使用服务器 4080 SUPER 跑完 19 个 v0.3 样本筛选策略。
- 已拉回 GPU 筛选报告、汇总表、排序表和服务器日志。
- 已确认综合排序第一为 `s16_weakpost_lat`。

### 待做任务
- 重点复核 `s16_weakpost_lat` 新增的 16 个样本，确认是否真实属于可训练的弱/保守响应。
- 对 `s16`、`s04`、`s00` 生成预测图或固定样本对比图，确认不是只在指标上好看。
- 检查横向偏移特征是否存在坐标跳变或道路模块切换影响。
- 决定下一版正式样本集是否采用 `s16`，或采用去横向偏移的 `s04` 作为更稳妥版本。

### 阻塞任务
- 暂不加入连续风格、生理、脑电；先完成 `s16` 新增样本和横向偏移风险复核。

### 需要服务器的任务
- 当前无正在运行的服务器任务。
- 若后续补完整车辆-only 基线或预测图，可继续使用服务器。

### 不需要服务器的任务
- 本轮结果解释、CSV 表格查看、新增样本复核图整理、中文报告更新。

---

# 当前任务队列

## 最新更新：2026-05-19 21:13

### 正在做任务
- 使用服务器 GPU 运行 `stage03_v03_screening_sweep_gpu.py`，对 v0.3 多种样本筛选策略做车辆-only 快速比较。

### 已完成任务
- 已停止旧的 CPU 版 screen `v03sweep`，避免继续用 CPU 跑慢速 sklearn 连续筛选。
- 已确认服务器 GPU 为 NVIDIA GeForce RTX 4080 SUPER，PyTorch CUDA 可用。
- 已同步 GPU 快速筛选脚本到服务器并通过远端语法检查。

### 待做任务
- 等待 GPU 筛选完成后拉回结果表、报告和服务器日志。
- 根据 GPU 快速筛选排序，选择 2-3 个有希望的样本纳入策略，再补完整车辆-only 基线和预测图。
- 将最终筛选建议整理为用户可读中文说明。

### 阻塞任务
- 暂不进入连续风格、生理、脑电模型；必须先确定 v0.3 样本筛选范围是否更合理。

### 需要服务器的任务
- 当前 GPU 快速筛选任务正在服务器 screen `v03gpu` 中运行。

### 不需要服务器的任务
- 结果拉回后的报告整理、表格解释、样本策略命名和人工复核建议。

---

# 当前任务队列

## 最新更新：2026-05-18 19:21

### 正在做任务
- v0.3 全量原始数据极限/近极限工况 episode 样本库已生成，当前需要先看代表复核图，再决定是否进入车辆-only 基线训练。

### 已完成任务
- 已新增并运行 `05_rebuild_from_raw_20260511/02_samples/scripts/build_extreme_condition_episodes_v0_3.py`。
- 已确认主入口是 `01_datasets/数据预处理/原始车辆数据` 下所有原始车辆 CSV，不再从旧候选表筛选。
- 已排除非被试记录 `carsim对标.csv`，避免 subject 出现 `原始车辆数据`。
- 已生成 v0.3 总表、分类表、文件扫描报告、分被试/分上下文统计、161 张代表复核图、用户查看版报告和技术报告。
- 当前最终统计：episode 总数 1574；强响应 49，弱/保守 208，延迟/无明显转向 139，正常对照 86，待复核 311，排除 781。

### 待做任务
- 人工优先复核代表图：强响应、弱/保守、延迟/无明显转向、正常对照各看一批。
- 根据复核结果决定是否调整 v0.3 阈值，尤其是低附着、弯道、横滚/姿态响应和制动主导样本。
- 若复核通过，构建车辆-only 数据集并先跑无学习基线、车辆-only / 道路-only 强基线。
- 车辆基线站住后，再加入连续驾驶风格和生理数据增量验证。

### 阻塞任务
- 暂不直接训练生理/脑电/连续风格模型。
- 暂不把 v0.3 自动弱标签当作最终训练真值。
- 如果代表图显示强响应或弱/保守类别语义不准，需要回到样本筛选规则，而不是继续堆模型。

### 可并行任务
- 复核图人工标注。
- 统计 v0.3 样本按被试、上下文、旧道路模块的分布。
- 准备车辆-only 数据集构建脚本。
- 准备无学习基线和车辆-only 基线评估指标。

### 需要服务器的任务
- 暂无。v0.3 全量扫描已在本机完成。
- 后续若进入车辆-only 神经网络训练，可使用服务器。

### 不需要服务器的任务
- 代表图复核、样本表统计、无学习基线、数据集 manifest 构建可先本地完成。

## 最新更新：2026-05-13 08:09

### 正在做任务
- Stage 7j session 多折稳定性验证已完成并已提交；当前准备进入 Stage 7k 候选生成/选择规则复核。

### 已完成任务
- 已新增并运行 `stage07j_session_cv_stability_v0_1.py`。
- 已完成 5 折 session-CV，每折重训 RBF/KNN 基座，避免固定 split 预测泄漏。
- 已排除固定 split top-K/Transformer/keypoint 预测特征，只使用事件前车辆/道路上下文和 fold-retrained RBF 形态特征。
- 已确认 `stability_penalty_l05` 平均 test delta=+0.000329，gate=`no_upgrade`。
- 已生成 Stage 7j 用户总结、技术报告、5 折指标表、policy 汇总表、feature audit、gate 表和 3 张非空图。
- 已提交 `11296297 Add stage7j session cv stability audit`。

### 待做任务
- Stage 7k：回到候选生成/选择规则设计，重点处理 wrong-side、幅值不足、困难样本和 response_family 分布偏移。
- 复核原始 Stage7g val gate 为什么在严格 CV 中平均 RMSE 略好但 wrong-side 变差，判断是否需要物理指标优先的选择规则。
- 如果要继续多候选路线，需做完整上游候选重训或更保守的候选池，而不是直接用固定 split 的 top-K/Transformer 特征。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- 连续风格强结论继续阻塞。
- Stage7i single-split 弱收益不能升级为主线。

### 可并行任务
- 按 fold/response_family/road module 分析候选退化样本。
- 设计 wrong-side 优先或困难样本优先的 selection gate。
- 生成固定坏样本对照图，解释为什么平均 RMSE 小幅改善不等于物理预测可靠。

### 需要服务器的任务
- 暂无。Stage 7j 是本地 CPU 复核任务。

### 不需要服务器的任务
- Stage 7j 归档、commit、Stage 7k 轻量选择规则诊断均可本地完成。

## 最新更新：2026-05-13 07:55

### 正在做任务
- Stage 7i 稳定性校准候选选择已完成并已提交，当前回写 commit hash 到透明化记录。

### 已完成任务
- 已新增并运行 `stage07i_stability_calibrated_selection_v0_1.py`。
- 已从 Stage 7h 稳定表计算 difficult/wrong-side/large-recall 相对 RBF 的 split delta，修复首跑 `KeyError: 'difficult_rmse_delta_vs_rbf_val'`。
- 已确认稳定惩罚规则 `stability_penalty_l05` 选中 `segment_resid_rf_blend_25`，test RMSE delta=-0.005620，困难样本 RMSE delta=-0.029588。
- 已生成 Stage 7i 用户总结、技术报告、候选分数表、policy 指标表、逐样本收益表、gate 表和 3 张非空图。
- 已提交 `d294a520 Add stage7i stability calibrated selection`。

### 待做任务
- Stage 7j：做多折 session validation 或分层 validation 重构，验证稳定选择规则是否可复现。
- 检查 Stage 7i 选中策略在响应类型、道路模块、置信度分布不同 bucket 下是否稳定。
- 如果多折验证通过，再决定是否把 `segment_resid_rf_blend_25` 或稳定选择规则冻结为车辆-only 弱主候选。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞，直到车辆-only 多候选选择校准更稳。
- 连续风格强结论继续阻塞；当前不允许用风格/生理解释车辆-only 选择不稳定。
- 不能把 single-split 的 `weak_candidate_continue` 直接写成最终主线升级。

### 可并行任务
- Stage 7i 收益样本按 response_family/road_design_module_name/置信度 bucket 复核。
- Stage 7j 多折 split 方案设计。
- 固定坏样本图和增益样本图的人工解释材料整理。

### 需要服务器的任务
- 暂无。Stage 7i 是本地诊断/选择校准任务。

### 不需要服务器的任务
- Stage 7i 归档、commit、Stage 7j split 设计和轻量复核均可本地完成。

## 最新更新：2026-05-13 06:39

### 正在做任务
- Stage 7b 非 oracle top-K selector 轻量实验已完成；当前准备候选轨迹差异复查或候选生成复查。

### 已完成任务
- 已新增并运行 `stage07b_non_oracle_topk_selector_v0_1.py`。
- 已剔除 label-derived 输入字段，保留 37 个允许特征。
- 已确认 val 选中的 learned fallback policy 在 test 上 100% 退回 RBF，RMSE 与 RBF 相同，没有形成新选择能力。
- 已确认 gate 为 `no_upgrade`，生理/EEG 继续阻塞。
- 已提交 `d431cd11 Add stage7b non-oracle topk selector`。

### 待做任务
- 若继续 Stage 7，需要导出完整候选预测轨迹差异特征，而不是只用现有摘要特征。
- 复查是否应回到候选生成本身：让候选分支更可分、更稳定，而不是只改 selector。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- 不能把“全部 fallback 到 RBF”解释为多假设路线有效。

### 可并行任务
- 候选预测轨迹导出。
- oracle gap 样本的候选差异复查。
- 候选生成分支多样性和可靠性复查。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前 Stage 7b 归档、commit 和候选差异复查可本地完成。

## 最新更新：2026-05-13 06:31

### 正在做任务
- Stage 7a 非 oracle 多候选选择协议已完成；当前准备 Stage 7b 非 oracle selector 轻量实验。

### 已完成任务
- 已新增并运行 `stage07a_non_oracle_selection_protocol_v0_1.py`。
- 已固定 10 个车辆-only候选和 RBF/KNN 主参照。
- 已明确允许特征、禁止信息、train/val/test 边界、校准指标、coverage-risk 和固定图协议。
- 已确认 Stage 7 可以进入非 oracle 选择器设计，但不能进入生理/EEG 有效性结论。
- 已提交 `dfacb38d Add stage7a non-oracle selection protocol`。

### 待做任务
- Stage 7b：构建候选预测差异和可靠性特征，训练/验证非 oracle selector。
- 输出 selector top-1、RBF fallback、低置信度 abstain/fallback、oracle upper bound 的公平对照。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- Stage 7 模型升级主线前必须满足 test RMSE 不劣于 RBF，且至少一个物理指标或困难样本改善。

### 可并行任务
- 候选预测轨迹导出/对齐。
- 候选间 disagreement 特征构建。
- 固定坏样本图、oracle gap 样本图和 selector regret 图协议落地。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前 Stage 7a 协议归档、commit 和 Stage 7b 轻量 selector 设计可本地完成。

## 最新更新：2026-05-13 06:24

### 正在做任务
- Stage 6e 多候选 oracle gap 复核已完成；当前准备 Stage 7 非 oracle 选择协议。

### 已完成任务
- 已新增并运行 `stage06e_multicandidate_oracle_gap_v0_1.py`。
- 已确认 broad oracle pool test RMSE=0.375182，相对 RBF/KNN delta=-0.158484，但该结果不可部署。
- 已确认当前最好可部署 selector test RMSE=0.533912，比 RBF/KNN 差 +0.000245，实际选择策略仍未超过主参照。
- 已确认 Stage 7 若继续，必须先解决非 oracle 选择和可靠性校准。
- 已提交 `cb4d8eec Add stage6e multicandidate oracle gap audit`。

### 待做任务
- 设计 Stage 7 非 oracle 多候选选择协议：候选分支、选择特征、校准指标、坏样本固定图和不能使用 test 标签的规则。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- 不能把 broad oracle、best-of-K 或真实标签选择结果当成可部署模型。

### 可并行任务
- 多候选 winner 样本分桶复查。
- 选择策略特征协议设计。
- 固定坏样本图与候选多样性/可靠性图设计。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前 Stage 6e 归档、commit 和 Stage 7 协议设计均可本地完成。

## 最新更新：2026-05-13 06:16

### 正在做任务
- Stage 6d RBF/KNN reliability gate 已完成；当前准备进入下一步路线选择。

### 已完成任务
- 已新增并运行 `stage06d_reliability_gate_v0_1.py`。
- 已确认当前不是把 Transformer 当主线继续训练；RBF/KNN 类车辆-only 强基线仍是主参照。
- 已确认保守 policy `val_rmse_noninferior_conservative` test RMSE=0.534545，比 RBF/KNN 差 +0.000878，wrong-side 和 large recall 没有改善。
- 已确认激进 policy `val_best_rmse` 只改善 wrong-side/large recall，但 RMSE 退化到 0.544356，不能升级。
- 已提交 `4264db88 Add stage6d reliability gate`。

### 待做任务
- Stage 6 selector/reliability 当前形式降级为诊断候选后，准备多假设候选生成/实际选择策略或车辆-only 表示复查。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- 不能根据 oracle/keypoint 上限或物理指标单项改善宣称主线升级。

### 可并行任务
- 多假设候选生成与非 oracle 选择策略设计。
- RBF/KNN 当前坏样本按响应类型、道路模块、尾段/多段修正分桶复查。
- 样本规则和车辆-only 表示的泄漏/信息量复核。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前 Stage 6d 归档、commit、下一步路线决策均可本地完成。

## 最新更新：2026-05-13 06:06

### 正在做任务
- Stage 6c selector feature revision 已完成；下一步准备做可靠性门控。

### 已完成任务
- 已新增并运行 `stage06c_selector_feature_revision_v0_1.py`。
- 已确认 val 选择 `rf_engineered_shallow`，test RMSE=0.544356，比 RBF 差 +0.010689。
- 已确认该 selector 改善错侧率和大幅响应召回，但当前 gate 为 `no_upgrade_current_revision`。

### 待做任务
- 复盘 RF selector 的 6 个 FN 和 13 个 FP。
- 设计 reliability gate，控制 FP，同时尽量保留错侧率和大幅响应召回收益。
- 如果可靠性门控仍不能同时改善 RMSE 和物理指标，则 Stage 6 selector 路线暂时降级为诊断候选。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- 不能用物理指标单项改善直接宣称车辆-only结构化模型已解决。

### 可并行任务
- RF selector 的 threshold/margin 分析。
- FP 高概率样本的道路/响应形态复核。
- FN 低概率样本的候选预测差异复核。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前 reliability gate 设计、表格和报告整理均可本地完成。

## 最新更新：2026-05-13 05:57

### 正在做任务
- Stage 6b RBF/keypoint 选择器错误复盘已完成；下一步准备修 selector/reliability 特征。

### 已完成任务
- 已新增并运行 `stage06b_keypoint_selector_error_review_v0_1.py`。
- 已确认 selector 在 test 上 TP=5、FP=6、FN=12、TN=17，主要问题是漏选 keypoint 可收益样本。
- 已确认 selector 平均 delta vs RBF=+0.006945，没有形成稳定可部署提升。
- 已提交 `753525fd Add stage6b keypoint selector error review`。

### 待做任务
- 复盘 12 个 `FN_missed_keypoint_gain` 样本，找出 keypoint 更好但 selector 概率低的原因。
- 复盘 6 个 `FP_select_keypoint_hurts` 样本，找出错选 keypoint 的风险特征。
- 设计 selector feature revision：候选间差异、响应形态风险、道路模块、历史稳定性、可靠性分数。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- 多假设/关键点路线不能用 oracle 作为结论，必须先形成可部署选择策略。

### 可并行任务
- FN 样本图表复核。
- FP 样本图表复核。
- selector probability 与实际 keypoint gain 的分布复核。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前 selector 错误复盘和下一版特征设计均可本地完成。

## 最新更新：2026-05-13 05:50（本机实际时间；阶段顺序接在阶段 4 收口之后）

### 正在做任务
- 阶段 6 车辆-only 结构化路线审计已完成；下一步准备做 Stage 6b 可部署选择器/可靠性门控。

### 已完成任务
- 已新增并运行 `stage06_vehicle_only_structured_route_audit_v0_1.py`。
- 已形成车辆-only结构化 gate：RBF 保留为 limited primary reference；响应分解 Transformer v0.1 no-go；keypoint selector weak candidate；oracle/best-of-K 只能作为研究上限。
- 已确认生理/EEG 继续阻塞。
- 已提交 `b4d7ac20 Add stage6 vehicle structured route audit`。

### 待做任务
- Stage 6b：复盘 `selector_logreg_rbf_keypoint_no_subject` 的样本级选择错误。
- 将 oracle/best-of-K 上限转化为非 oracle 的可部署选择策略。
- 做可靠性门控和坏样本分桶，重点检查错侧、幅值不足、峰时、尾段、反向修正和多段修正。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- 连续风格有效性强结论继续阻塞；当前风格路线不升级主线。

### 可并行任务
- keypoint selector 错选样本复盘。
- best-of-K oracle 增益样本复盘。
- RBF top bad 与 selector 改善/退化样本交叉表。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前 Stage 6b 选择器复盘、可靠性门控原型和报告整理均可先本地完成。

## 最新更新：2026-05-13 06:25

### 正在做任务
- 阶段 4 连续风格路线已收口；下一步准备进入车辆-only 结构化轨迹建模。

### 已完成任务
- 已新增并运行 `stage04_style_route_decision_v0_1.py`。
- 已形成连续风格 no-go-current-form 决策：当前统计风格 + RBF 残差 Ridge 不升级主线。
- 已确认生理/EEG 继续阻塞。
- 已提交 `4064bf64 Add style route decision`。

### 待做任务
- 阶段 6：车辆-only 响应分解/关键点+残差/多假设路线设计。
- 固定 RBF 坏样本图复核摘要，用来定义结构化模型要解决的物理错误。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- 连续风格有效性强结论阻塞；仅保留为后备表示/融合探索。

### 可并行任务
- 车辆-only 结构化模型方案草稿。
- RBF top bad 样本物理错误摘要。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前收口报告、下一轮结构化车辆-only 方案和轻量原型均可先本地完成。

## 最新更新：2026-05-13 06:05

### 正在做任务
- 阶段 4 连续风格跨 split 复核已完成；当前准备把风格路线暂时降级收口。

### 已完成任务
- 已新增并运行 `stage04_style_cross_split_validation_v0_1.py`。
- 已完成 session-level 与 subject-level 两类切分的 RBF+连续风格残差对照。
- 已确认当前 last60 连续风格在两类切分下都没有稳定超过 RBF：session 0.533667->0.534559，subject 0.484847->0.483510。

### 待做任务
- 写阶段 4 收口说明：当前表示/融合下连续风格没有形成强证据。
- 回到车辆-only 结构化轨迹建模，优先错侧、幅值、尾段、反向修正、多段修正和困难样本。

### 阻塞任务
- 连续风格有效性结论仍阻塞。
- 生理、脑电有效性验证仍阻塞，直到车辆-only 和风格参照更稳。

### 可并行任务
- 固定图/坏样本图人工复核摘要。
- 车辆-only 结构化轨迹模型候选设计。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前阶段 4 收口报告和车辆-only 结构设计可本地完成。

## 最新更新：2026-05-13 05:40

### 正在做任务
- 阶段 4 连续风格探索性增量对照 v0.1 已完成；当前准备进入 subject-level/跨 session 复核。

### 已完成任务
- 已新增并运行 `stage04_style_increment_exploratory_v0_1.py`。
- 已生成 RBF+连续风格、RBF+驾驶员 ID、RBF+道路模块、RBF+风格+ID 和置乱控制的指标、逐样本表、固定预测图、坏样本图和 gate 表。
- 当前 RBF test RMSE=0.533667，RBF+last60 风格 test RMSE=0.534559。

### 待做任务
- 做 subject-level 或留一被试风格验证，检查是否跨被试成立。
- 做跨 session 与道路均衡置乱的更严格复核。
- 逐图复核固定预测图和坏样本图，判断收益是否改善错侧、幅值、尾段、反向修正、多段修正或困难样本。

### 阻塞任务
- 连续风格有效性结论仍阻塞，直到多 split、置乱、驾驶员 ID 对照和物理指标复核完成。
- 生理、脑电有效性验证仍阻塞，直到车辆+连续风格公平参照形成。

### 可并行任务
- 风格 subject-level 输入表和 RBF 参照对齐。
- 风格置乱 seed 扩展。
- 坏样本图人工复核摘要。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前阶段 4 风格轻量对照和报告整理均可本地完成。



## 最新更新：2026-05-13 04:56

### 正在做任务
- 阶段 4 连续风格协议与候选特征 v0.1 已归档并完成 Git 提交，当前准备进入连续风格探索性模型对照。

### 已完成任务
- 已新增并运行 `stage04_continuous_style_protocol_v0_1.py`。
- 已提交 `012f4803 Add continuous style protocol audit`。
- 已生成 B 轨道 270 个严格核心样本的连续风格候选 long/wide 特征表。
- 已生成 session-level train-only 标准化参数、泄漏边界表、置乱对照计划、道路/被试耦合审计、gate 表、3 张图和阶段 4 用户查看版总结。
- 泄漏检查结果：直接输入窗口重叠 0，标签未来重叠 0。

### 待做任务
- 在固定 RBF 主参照上做连续风格探索性模型对照。
- 对照必须包含原始连续风格、驾驶员 ID、被试内置乱、跨被试置乱、跨 session 置乱和道路平衡置乱。
- 评价必须覆盖 RMSE、错侧、幅值、尾段、反向修正、多段修正、困难样本、分被试和分道路结果。

### 阻塞任务
- 连续风格有效性结论仍阻塞，直到模型对照、置乱和分被试验证完成。
- 生理、脑电有效性验证仍阻塞，直到车辆 + 连续风格基线形成公平参照。

### 可并行任务
- 风格模型输入表与 RBF 逐样本指标对齐。
- 风格置乱表生成。
- 道路/被试耦合风险复查。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前阶段 4 协议、轻量特征处理和下一轮小模型对照均可先本地完成。

## 最新更新：2026-05-13 04:37

### 正在做任务
- RBF 主参照冻结审计已提交，当前准备进入阶段 4 连续风格协议设计。

### 已完成任务
- 已提交 `112824f7 Add rbf reference freeze audit`。
- RBF 有限冻结 gate、失败画像、关键指标图和用户查看版总结已归档。

### 待做任务
- 设计阶段 4 连续驾驶风格验证协议：事件前风格来源、train-only 标准化、置乱对照、分被试/分 session 评估和物理指标。
- 固定 RBF 主参照为所有风格候选的底线，不允许用 oracle 上限作为实际性能。

### 阻塞任务
- 连续风格有效性结论仍阻塞，直到置乱和分被试验证完成。
- 生理、脑电有效性验证仍阻塞，直到车辆 + 连续风格基线形成公平参照。

### 可并行任务
- 阶段 4 风格特征候选来源清单。
- RBF top bad 样本逐图复查。
- 风格置乱与分被试 split 方案草案。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 阶段 4 协议设计、表格整理和轻量验证脚本可先本地完成。

## 最新更新：2026-05-13 04:31

### 正在做任务
- RBF 主参照冻结审计 v0.1 已完成，当前准备归档并提交 Git。

### 已完成任务
- 已新增并运行 `stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1.py`。
- 已生成 RBF 指标画像、失败画像、top bad 样本、冻结 gate 表、稳健性快照、2 张图和两份中文报告。
- 已明确 RBF 只能有限冻结为 B 轨道后续增量实验的保守车辆-only 主参照；车辆-only 物理响应问题没有解决。

### 待做任务
- 提交本轮脚本、产物、报告和透明化记录。
- 进入阶段 4 前，整理连续风格协议：固定 RBF 主参照、使用无泄漏事件前风格、设置置乱/分被试/物理指标/坏样本对照。
- 若继续阶段 3，可围绕反向修正、多段修正、错侧和困难样本设计更强车辆-only 结构，但不能用 oracle 当实际性能。

### 阻塞任务
- 生理、脑电有效性验证仍阻塞。
- 连续风格有效性结论仍阻塞；目前只允许做阶段 4 协议设计/探索性实验。

### 可并行任务
- 阶段 4 连续风格无泄漏特征来源清单整理。
- RBF top bad 样本逐图复查。
- oracle 增益样本复查，避免误当可部署性能。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 本轮冻结审计归档和阶段 4 协议设计均可本地完成。

## 最新更新：2026-05-13 04:21

### 正在做任务
- 车辆-only 主参照决策表已提交，当前等待下一步阶段 3 决策。

### 已完成任务
- 已提交 `e04bdb2f Add vehicle-only decision table`。
- 车辆-only 候选角色、gate 状态和主参照状态已归档。

### 待做任务
- 决定是否冻结 RBF KRR 为保守主参照，还是继续做更强车辆-only 结构。
- 若冻结 RBF，需要写清错侧、反向修正、多段修正和困难样本仍未解决。
- 保持连续风格、生理、EEG 增量验证阻塞。

### 阻塞任务
- 连续风格、生理、EEG 增量验证仍阻塞。

### 可并行任务
- RBF 坏样本复查。
- oracle 增益样本复查。
- 阶段 4 前置条件清单整理。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前下一步可本地完成。

## 最新更新：2026-05-13 04:18

### 正在做任务
- 阶段 3 车辆-only 主参照决策表 v0.2 已完成，当前准备归档并提交 Git。

### 已完成任务
- 已新增并运行 `stage03_vehicle_instability_vehicle_only_decision_table_v0_2.py`。
- 已生成候选决策表、gate 状态表、角色汇总、指标库存、3 张图和两份中文报告。
- 已明确 RBF KRR 是当前暂定主参照，但强车辆基线仍未完全冻结。

### 待做任务
- 提交本轮脚本、产物、报告和透明化记录。
- 后续二选一：做 RBF 主参照冻结审查，或继续更强车辆-only 分响应类型/关键点条件多假设。
- 保持连续风格、生理、EEG 增量验证阻塞。

### 阻塞任务
- 连续风格、生理、EEG 增量验证仍阻塞，因为 `strong_vehicle_baseline_frozen=no`。

### 可并行任务
- RBF 反向修正/多段修正失败样本复查。
- keypoint/top-K oracle 增益样本复查。
- 主参照冻结审查材料整理。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 本轮已本地完成；后续冻结审查可本地完成。

## 最新更新：2026-05-13 04:11

### 正在做任务
- top-K 可靠性选择/回退 v0.1 已提交，当前准备下一步车辆-only 决策。

### 已完成任务
- 已提交 `fbb8d94d Add topk reliability selector`。
- 可靠性选择脚本、产物、报告和透明化记录已归档。
- 结论：当前可部署选择策略没有超过 RBF，不能升级强车辆基线。

### 待做任务
- 阶段 3 后续二选一：冻结 RBF/KNN 类主参照并做进入阶段 4 前审查，或继续做更强车辆-only 分响应类型/关键点条件多假设。
- 继续阻塞连续风格、生理、EEG 增量结论。

### 阻塞任务
- 风格、生理、EEG 增量验证仍阻塞。

### 可并行任务
- top-K oracle 增益样本复核。
- RBF/KNN 类主参照冻结审查。
- 更强车辆-only 多假设结构草案。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前下一步可本地完成。

## 最新更新：2026-05-13 04:07

### 正在做任务
- top-K 可靠性选择/回退 v0.1 已完成，当前准备归档并提交 Git。

### 已完成任务
- 已新增并运行 `stage03_vehicle_instability_topk_reliability_selector_v0_1.py`。
- 已生成可靠性选择指标、决策、阈值、分层汇总、固定图、坏样本图、oracle 增益图和两份中文报告。
- 已明确本轮选择策略没有超过 RBF，不能升级为强车辆基线。

### 待做任务
- 提交本轮脚本、产物、报告和透明化记录。
- 继续阶段 3：若继续建模，应做分响应类型/关键点条件多假设或更强 train/val 可靠性头。
- 保持连续风格、生理、EEG 增量验证阻塞。

### 阻塞任务
- 连续风格、生理、EEG 增量验证仍阻塞，因为强车辆基线/可靠性选择还未冻结。

### 可并行任务
- top-K oracle 增益样本逐图复查。
- RBF/KNN 类主参照冻结审查。
- 关键点条件多假设结构草案。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 本轮已本地完成；后续轻量复查仍可本地完成。

## 最新更新：2026-05-13 03:49

### 正在做任务
- top-K gap review 已提交，当前准备车辆-only 可靠性/选择头 v0.2 或关键点条件多假设方案。

### 已完成任务
- 已提交 `1ace03f2 Add topk gap review`。
- gap review 的图、表、报告和透明化记录已归档。

### 待做任务
- 设计 train/val 固定的可靠性/选择头 v0.2。
- 或设计关键点条件多假设结构。
- 继续阻塞风格、生理、EEG 增量结论。

### 阻塞任务
- 连续风格、生理、EEG 增量验证仍阻塞。

### 可并行任务
- 高 gap 样本逐图复查。
- 可靠性标签与损失设计。
- 关键点条件多假设结构草案。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前下一步可本地完成。

## 最新更新：2026-05-13 03:46

### 正在做任务
- 阶段 3 top-K gap review v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_topk_gap_review_v0_1.py`。
- 生成 top1/bestK gap 样本详情、train 阈值、可靠性相关性、分桶汇总、分被试/道路/响应族汇总和 4 张诊断图。
- 明确当前 top-K 瓶颈是选择头/可靠性，而不是完全没有候选覆盖。

### 待做任务
- 提交本轮 gap review。
- 用本轮结果设计可靠性/选择头 v0.2 或关键点条件多假设模型。
- 若继续做可靠性规则，必须 train/val 固定后 test 评价。

### 阻塞任务
- 连续风格、生理、EEG 增量验证继续阻塞。
- 当前可靠性规则不可直接部署，不能冻结强车辆基线。

### 可并行任务
- 高 gap 样本图逐个复查。
- 可靠性标签构造方案。
- 关键点条件多假设结构草案。

### 需要服务器的任务
- 暂无。本轮是本地表格/图表复盘。

### 不需要服务器的任务
- 本轮提交和下一版方案设计。

## 最新更新：2026-05-13 03:37

### 正在做任务
- top-K v0.1 已提交，当前准备进入 top1/bestK 差距复盘与下一版车辆-only 可靠性模型设计。

### 已完成任务
- 已提交 `03165475 Add topk vehicle transformer`。
- top-K v0.1 的脚本、checkpoint、图、表、报告和透明化记录已归档。

### 待做任务
- 分析 top1/bestK 差距样本。
- 设计更强选择机制或关键点条件多假设模型。
- 继续阻塞风格、生理和 EEG 增量结论。

### 阻塞任务
- 连续风格、生理、EEG 增量验证仍阻塞。
- top-K best-of-3 只能作为上限，top-1 不足以冻结车辆基线。

### 可并行任务
- 差距样本复盘。
- 可靠性标签方案。
- 下一版结构草案。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前下一步均可本地完成。

## 最新更新：2026-05-13 03:34

### 正在做任务
- 阶段 3 top-K 车辆-only Transformer v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_topk_vehicle_transformer_v0_1.py`。
- 训练 K=3 车辆-only 多假设 Transformer，checkpoint 只按 val top-1 RMSE 选择。
- 生成 top-1、best-of-3、RBF、各分支指标和分支可靠性诊断。
- 明确 best-of-3 是 oracle 上限，top-1 没有超过 RBF，不能冻结为强车辆主线。

### 待做任务
- 提交 top-K 脚本、结果、图表、报告和透明化记录。
- 复盘 top-1 选择头为什么只与 best-of-3 一致 0.300。
- 设计更强可靠性头或关键点条件多假设结构。

### 阻塞任务
- 连续风格、生理、EEG 增量验证继续阻塞。
- 强车辆基线尚未冻结，top-K v0.1 不能升级为主线。

### 可并行任务
- top-1/best-of-3 差距样本复盘。
- 可靠性标签设计。
- 关键点条件多假设模型方案整理。

### 需要服务器的任务
- 暂无。本轮本地 CUDA 已完成。

### 不需要服务器的任务
- 本轮归档提交、误选样本分析、下一版方案设计。

## 最新更新：2026-05-13 03:22

### 正在做任务
- 阶段 3 多候选车辆-only 复盘已提交，当前准备设计真正 top-K/可靠性车辆-only 模型。

### 已完成任务
- 已提交 `01033e3e Add rbf keypoint multihypothesis review`。
- 多候选复盘的图、表、报告和透明化记录已归档。
- 已确认无服务器任务、无后台任务、未读取密码文件。

### 待做任务
- 设计 top-K 车辆-only 输出或可靠性门控，而不是只复用两个旧候选。
- 把 selector 误选样本转成可靠性/困难样本训练目标。
- 继续维持强车辆基线冻结前的风格/生理/EEG 阻塞规则。

### 阻塞任务
- 连续风格、生理、EEG 增量验证仍阻塞。

### 可并行任务
- top-K 结构方案草拟。
- selector 误选样本归因。
- 可靠性标签与困难样本标签整理。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前下一步方案设计和轻量诊断均可本地完成。

## 最新更新：2026-05-13 03:18

### 正在做任务
- 阶段 3 RBF/keypoint 多候选车辆-only 复盘 v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1.py`。
- 复用 RBF、keypoint checkpoint、train/val selector 和 oracle best-of-two 上限，生成同图同表的多候选诊断。
- 生成固定预测图、selector 坏样本图、oracle 增益样本图、选择混淆图、oracle 增益柱图、误选样本表和用户查看版总结。

### 待做任务
- 把本轮多候选复盘提交到 Git。
- 若继续阶段 3，设计真正 top-K/可靠性车辆-only 模型或候选选择器，不再只做离线复盘。
- 在强车辆基线冻结前继续阻塞连续风格、生理和 EEG 增量结论。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证仍被“强车辆-only 多假设/可靠性基线未冻结”阻塞。
- oracle best-of-two 是事后上限，不能作为可部署性能。

### 可并行任务
- 误选样本原因复盘。
- top-K 车辆-only 结构设计。
- 可靠性/困难样本辅助标签设计。

### 需要服务器的任务
- 暂无。本轮没有训练新模型，当前下一步也可先本地设计。

### 不需要服务器的任务
- 本轮复盘提交、报告检查、下一版 top-K 车辆-only 方案设计。

## 最新更新：2026-05-13 03:06

### 正在做任务
- 阶段 3 RBF/keypoint selector v0.1 已完成提交，当前准备进入正式多假设/可靠性车辆-only 评估包设计。

### 已完成任务
- 已提交 `7e3d53f6 Add rbf keypoint selector`。
- selector 结果、图表、报告和 02:54 透明化记录已经归档。
- 已确认本轮没有服务器任务、没有后台任务、没有读取密码文件。

### 待做任务
- 分析 selector 误选样本和正确切换样本。
- 生成 RBF/keypoint/selector/oracle 同图固定预测图和坏样本图。
- 设计正式 top-1、best-of-K、可部署 selector、oracle 上限和可靠性分层指标。

### 阻塞任务
- 在正式多假设/可靠性车辆-only 结果出来前，连续风格、生理和 EEG 增量验证继续阻塞。

### 可并行任务
- 误选样本表分析。
- 多候选固定图生成。
- 可靠性/困难样本标签方案整理。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前下一步均可本地完成。

## 最新更新：2026-05-13 02:54

### 正在做任务
- 阶段 3 RBF vs keypoint train/val 选择器 v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_rbf_keypoint_selector_v0_1.py`。
- 训练一个不使用 subject ID、生理、脑电、连续风格或 test 标签的 RBF/keypoint 选择器。
- selector 只用 train 拟合，阈值只用 val 选择，test 只做最终评价。
- 生成 selector 训练表、数值/类别特征表、阈值扫描表、test 决策表、统一指标表、选择后逐样本表、test 指标图、阈值扫描图、运行摘要、用户查看版总结和技术报告。

### 待做任务
- 决定是否进入正式多假设车辆-only 版本：报告 top-1、best-of-K、可部署选择策略、固定预测图和校准。
- 分析 selector 在 test 选择的 11 个 keypoint 样本是否对应 keypoint 真正优势样本，以及误选样本的共性。
- 在多候选车辆-only 稳定前，继续阻塞连续风格、生理和 EEG 增量结论。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被“车辆-only 多候选/可靠性策略尚未稳定”阻塞。
- selector 与 RBF RMSE 近似持平但未明显超过 RBF，不能宣称强车辆基线已经最终解决。

### 可并行任务
- 分析 selector 的误选样本。
- 准备多假设车辆-only 预测图，展示 RBF/keypoint/oracle/selector 四种曲线。
- 设计可靠性标签：何时 RBF 更可信、何时 keypoint 更可信、何时两者都不可信。

### 需要服务器的任务
- 暂无。本轮是本地表格/轻量 sklearn 选择器，没有远程服务器任务。

### 不需要服务器的任务
- selector 产物归档、报告更新、Git 提交、多假设/可靠性车辆-only 方案设计。

## 最新更新：2026-05-13 02:44

### 正在做任务
- 阶段 3 keypoint+residual vs RBF 坏样本差异复盘 v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1.py`。
- 对 B 轨道 session-level test 40 个样本逐样本比较 keypoint+residual 与 RBF KRR。
- 生成样本级 RMSE 差异表、错误变化计数表、分被试摘要、Top 改善表、Top 退化表、RMSE 差异图、错误变化计数图、运行摘要、用户查看版总结和技术报告。
- 明确 keypoint+residual 的收益集中在错侧修复和大幅响应召回，不能单独替代 RBF 主参照。

### 待做任务
- 设计多假设车辆-only 或模型选择/可靠性门控，把 RBF 的整体 RMSE 稳定性和 keypoint 的方向/大幅响应优势结合起来。
- 分析 keypoint 退化样本是否集中在某些被试、道路模块、启动延迟大误差或幅值过度预测。
- 在车辆-only 多假设/可靠性策略完成前，继续阻塞连续风格、生理和 EEG 增量结论。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被“车辆-only 多候选/可靠性策略尚未形成”阻塞。
- keypoint+residual 单模型仍未超过 RBF RMSE，不能作为最终强车辆主线。

### 可并行任务
- 统计 keypoint 改善/退化样本的道路模块和响应形态。
- 设计基于 train/val 的模型选择器，避免用 test 事后选择 RBF 或 keypoint。
- 准备多假设车辆-only 评估：top-1、best-of-K、可部署选择策略和不确定性。

### 需要服务器的任务
- 暂无。本轮只是本地表格/图表复盘，没有训练任务。

### 不需要服务器的任务
- 坏样本差异归档、报告更新、Git 提交、下一版多假设/可靠性车辆-only 方案设计。

## 最新更新：2026-05-13 02:34

### 正在做任务
- 阶段 3 B 轨道车辆-only 关键点 + 残差 Transformer v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1.py`。
- 在 B 轨道 `response3s_strict_core_candidate` 上训练关键点 + 残差车辆-only Transformer，输入只使用事件前车辆时序和可因果获得的事件/道路上下文。
- 关键点标签只作为训练目标和评价目标，不作为推理输入；模型不使用生理、脑电、连续风格、驾驶员 ID 或未来 `eval_label_*` 字段。
- 生成 keypoint+residual 指标表、逐样本表、关键点误差表、模型信息表、训练历史、val 选择表、固定预测图、坏样本图、checkpoint、用户查看版总结和技术报告。
- 报告表中已加入 RBF KRR、direct Transformer、上一版 structured Transformer 和 KNN 模板参照。

### 待做任务
- 对 keypoint+residual 与 RBF KRR 的坏样本差异做复盘，明确它改善的是方向/大幅响应，还是只改变了错误分布。
- 如果继续车辆结构路线，优先测试多假设车辆-only 或可靠性/困难样本识别，而不是继续堆单输出回归模型。
- 形成阶段 3 车辆-only 主参照暂定规则：RBF KRR 仍是 val 选择主参照，keypoint+residual 是结构候选。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被“强车辆主参照和困难样本处理策略尚未冻结”阻塞。
- keypoint+residual 还没有在 val 选择、困难样本和反向修正上全面超过 RBF，不能升级为最终车辆主线。

### 可并行任务
- 对 keypoint+residual 的 top 坏样本图做人工摘要。
- 按响应形态、被试和道路模块分析 keypoint+residual 相比 RBF 的收益/退化。
- 准备多假设车辆-only 模型或困难样本可靠性模型脚本骨架。

### 需要服务器的任务
- 暂无。本轮使用本机可用 CUDA 前台完成；没有远程服务器任务。

### 不需要服务器的任务
- keypoint+residual 产物归档、报告更新、Git 提交、下一版车辆-only 结构设计。

## 最新更新：2026-05-13 02:10

### 正在做任务
- 阶段 3 B 轨道车辆-only 响应分解/结构化 Transformer v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_structured_vehicle_transformer_v0_1.py`。
- 在 B 轨道 `response3s_strict_core_candidate` 上训练结构化车辆-only Transformer，输入只使用事件前车辆时序和可因果获得的事件/道路上下文。
- 响应分解标签只作为训练辅助目标和评价目标，不作为推理输入；模型不使用生理、脑电、连续风格、驾驶员 ID 或未来 `eval_label_*` 字段。
- 生成结构化 Transformer 指标表、逐样本表、辅助标签准确率表、模型信息表、训练历史、val 选择表、固定预测图、坏样本图、checkpoint、用户查看版总结和技术报告。
- 报告表中已加入上一轮真正 direct Transformer 指标；KNN 只保留为模板参照。

### 待做任务
- 将结构化 Transformer v0.1 标记为弱候选/no-go，不作为主车辆基线升级。
- 整理 B 轨道 RBF KRR、direct Transformer、structured Transformer 的坏样本差异，决定是否继续关键点 + 残差或多假设车辆-only。
- 若继续结构路线，优先让模型显式预测关键点（主峰、峰值时间、回正点、零线穿越）和残差轨迹，而不是继续堆辅助分类头。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被“强车辆结构化主参照尚未冻结”阻塞。
- 结构化 Transformer v0.1 没有超过 RBF KRR，也没有改善大幅响应/尾段/困难样本，不能据此说明深度结构已经解决车辆-only 问题。

### 可并行任务
- 分析结构化 Transformer 坏样本中辅助标签预测错在哪里。
- 对 B 轨道固定图/坏样本图做人工摘要，列出哪些错误是幅值不足、尾段漂移、峰值时间错位或错侧。
- 准备关键点 + 残差车辆-only 脚本骨架。

### 需要服务器的任务
- 暂无。本轮使用本机可用 CUDA 前台完成；没有远程服务器任务。

### 不需要服务器的任务
- 结构化 Transformer 产物归档、报告更新、Git 提交、下一版车辆-only 结构设计。

## 最新更新：2026-05-13 01:43

### 正在做任务
- 阶段 3 干净响应任务车辆-only Transformer v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1.py`。
- 在 A/B 干净响应轨道上补跑真正的车辆-only Transformer，并与 RBF KRR、KNN template、formal ridge 同表同图对照。
- 生成指标表、逐样本指标表、模型信息表、训练历史、val 选择表、固定预测图、Transformer 坏样本图、checkpoint、运行摘要、用户查看版总结和技术报告。
- Transformer 内部加入 t=0 方向盘增量为 0 的物理约束；不使用生理、脑电、连续风格、驾驶员 ID 或响应分解标签作为输入。

### 待做任务
- 基于响应分解标签做车辆-only 结构化模型，而不是把直接 Transformer 升级为主线。
- 优先验证 B 轨道：方向、幅值桶、峰值时间/启动延迟、响应形态和尾段状态辅助头是否改善坏样本。
- 固定使用 B 轨道 RBF KRR 和本轮 Transformer 作为车辆-only 对照。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被强车辆结构化基线未完成阻塞。
- 直接 Transformer 尚未超过 RBF KRR，不能作为已解决车辆基线的证据。

### 可并行任务
- 整理 B 轨道 Transformer 坏样本与 RBF KRR 坏样本的重合/差异。
- 设计响应分解辅助头的数据读取和损失权重。

### 需要服务器的任务
- 暂无。本轮使用本机可用 CUDA 设备，本地前台完成；没有远程服务器任务。

### 不需要服务器的任务
- Transformer 产物归档、报告更新、Git 提交、下一版结构化车辆-only 模型设计。

## 最新更新：2026-05-13 01:14

### 正在做任务
- 阶段 3 车辆-only 响应分解标签 v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_response_decomposition_labels_v0_1.py`。
- 从 A/B 两条干净响应任务轨道生成方向、幅值、峰值时间、启动时间、尾段状态、零线穿越、反向修正和多段修正等标签。
- 生成 `response_decomposition_sample_labels.csv`、train-only 阈值表、轨道/split/形态/道路/被试汇总、3 张图、运行摘要、用户查看版总结和技术报告。
- 明确这些标签只能作为训练目标、辅助任务目标或评价分层，不能作为模型输入、split 条件、标准化条件或风格/生理特征。

### 待做任务
- 基于 B 轨道 `response3s_strict_core_candidate` 做车辆-only 响应分解/Transformer 对照。
- 对照普通轨迹回归、方向/幅值/峰时/形态辅助头、关键点+残差轨迹是否改善坏样本。
- 继续补固定预测图、坏样本图、分响应形态和分被试结果。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被强车辆结构化基线未完成阻塞。
- 不能把响应分解标签当推理输入，否则会造成事件后标签泄漏。

### 可并行任务
- 整理 B 轨道坏样本的响应形态分组图。
- 准备车辆-only Transformer/响应分解模型的数据读取与评估协议。

### 需要服务器的任务
- 暂无。本轮响应分解标签本地 CPU 完成；下一步小规模车辆-only 结构化对照可先本地验证。

### 不需要服务器的任务
- 响应分解标签归档、报告更新、Git 提交、车辆-only 结构化 baseline 脚本设计。

## 最新更新：2026-05-13 00:18

### 正在做任务
- 阶段 3 响应任务定义决策 v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `build_vehicle_instability_response_task_decision_v0_1.py`。
- 生成事件级任务决策表、样本级任务 manifest、任务类别计数、任务轨道计数、split/subject 汇总、2 张图、运行摘要、用户查看版总结和技术报告。
- 明确下一轮车辆-only 基线不再直接使用全部 906 个事件混合任务，而是优先使用 2 秒即时响应核心候选和 3 秒响应覆盖严格核心候选。

### 待做任务
- 基于 `sample_response_task_manifest.csv` 构建两个干净基线输入：`instant2s_core_candidate` 和 `response3s_strict_core_candidate`。
- 在这两个轨道上重跑无学习/强车辆-only 基线，先看样本规则修正后车辆历史能做到什么程度。
- 长事件/持续控制复核轨道暂不进入最终主线训练，后续单独设计长事件拆分或持续控制任务。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被车辆-only 干净任务基线未完成阻塞。

### 可并行任务
- 抽查 D 轨道 `long_or_unsettled_review` 中的代表事件，判断是持续控制、锚点问题还是需要更长标签。
- 统计 A/B/D 轨道在被试、session、道路上下文和响应形态上的分布。

### 需要服务器的任务
- 暂无。本轮任务定义决策本地 CPU 完成；下一步小规模车辆-only 复跑仍可先本地执行。

### 不需要服务器的任务
- 构建干净轨道基线输入、重跑本地车辆-only 基线、报告更新、Git 提交。

## 最新更新：2026-05-12 22:54

### 正在做任务
- 阶段 3 标签窗口覆盖审计 v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_label_window_coverage_audit_v0_1.py`。
- 对 906 个正式高置信失稳事件的 `pre2_label2_old_main` 与 `pre3_label3_response_coverage` 标签窗口做覆盖审计。
- 生成 `label_window_sample_metrics.csv`、`label_window_event_policy_table.csv`、窗口级 summary、推荐策略计数、split/subject 汇总、Top 12 坏事件 overlay、3 张图、运行摘要、用户查看版总结和技术报告。

### 待做任务
- 形成明确标签窗口决策：2 秒即时响应、3 秒完整响应候选、长事件拆分，或三者并行作为不同任务。
- 若决定修 manifest，回到阶段 2 生成新数据版本卡和样本规则；若决定保留当前标签，必须在阶段 3 报告中明确“2 秒任务只预测即时响应，不代表完整响应”。
- 在标签任务定义冻结前，不继续推进连续风格、生理或 EEG 增量实验。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被标签窗口/强车辆主参照未冻结阻塞。

### 可并行任务
- 抽查 `label_window_bad_event_overlay.csv` 中 Top 12 坏事件，确认 2 秒后变化是否是真实持续控制还是锚点/窗口问题。
- 统计标签窗口策略与被试、session、道路上下文、响应形态之间的关系。

### 需要服务器的任务
- 暂无。本轮窗口覆盖审计本地 CPU 完成。

### 不需要服务器的任务
- 标签窗口决策、manifest v0.2 规则设计、报告更新、Git 提交、后续车辆-only 基线重跑方案设计。

## 最新更新：2026-05-12 22:34

### 正在做任务
- 阶段 3 复发坏样本失败来源归因 v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_bad_event_failure_attribution_v0_1.py`。
- 对 Top 12 复发坏样本生成自动归因：标签窗口可能偏短、峰值接近窗口末端、尾段未回正、锚点前响应、原始信号支持度、共同幅值不足、错侧、反向修正失败和车辆-only 结构不足。
- 生成归因明细表、旗标统计、主归因统计、归因热图、主归因计数图、用户查看版总结和技术报告。

### 待做任务
- 人工/规则复核 `sample_rule_or_raw_signal_review` 的 10 个事件，判断是否需要回到阶段 2 修锚点或标签窗口。
- 对带有 `vehicle_only_structure_gap` 次级旗标的 9 个事件，整理结构化车辆模型需求：方向、幅值、峰值时间、反向修正和多段修正。
- 如果复核后样本规则可信，设计阶段 3 下一版结构化车辆-only 模型；否则先修 manifest。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被车辆-only 主参照未冻结阻塞。

### 可并行任务
- 把归因表与单事件曲线一起复核。
- 统计归因类别和被试/session/道路上下文/响应形态的关系。

### 需要服务器的任务
- 暂无。本轮归因本地 CPU 完成。

### 不需要服务器的任务
- 归因复核、manifest 修正规则设计、结构化车辆模型方案、报告更新、Git 提交。

## 最新更新：2026-05-12 22:18

### 正在做任务
- 阶段 3 复发坏样本详细曲线复盘 v0.1 已完成，当前准备归档并 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_bad_event_curve_review_v0_1.py`。
- 对复发坏样本 Top 12 生成单事件曲线，包含输入窗口、标签窗口、事件锚点、原始车辆波形、GT 方向盘响应和 formal/RBF/KNN/template 预测。
- 生成 `bad_event_curve_contact_sheet.png` 总览拼图、图索引、模型逐事件误差表、运行摘要、用户查看版总结和技术报告。
- 验证输出：12 个事件、60 条模型-事件误差记录；没有使用服务器、凭据、生理、脑电、连续风格或驾驶员 ID 作为模型输入。

### 待做任务
- 对 Top 12 曲线做复核，把失败分成锚点偏差、窗口覆盖不足、原始信号异常、车辆-only 信息不足或模型结构不足。
- 根据复核结果决定下一版车辆-only 模型结构：响应分解、关键点 + 残差、多假设轨迹或可靠性门控。
- 如果曲线显示部分事件锚点/窗口有问题，先回到阶段 2 修样本规则，而不是继续堆模型。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被车辆-only 主参照未冻结阻塞。

### 可并行任务
- 统计 Top 12/Top 20 坏事件与被试、session、道路上下文、响应形态之间的关系。
- 准备结构化响应标签：方向、峰值幅值、峰值时间、启动延迟、反向修正、多段修正、尾段漂移。

### 需要服务器的任务
- 暂无。本轮曲线复盘本地 CPU 完成。

### 不需要服务器的任务
- 曲线复核、表格汇总、报告更新、Git 提交、下一版结构化车辆模型设计。

## 最新更新：2026-05-12 21:44

### 正在做任务
- 阶段 3 稳健性坏样本复盘 v0.1 已完成，当前正在归档并准备 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_robustness_bad_sample_review_v0_1.py`。
- 对 3265 行 test 逐样本指标生成错误标记：高 RMSE、错侧、严重幅值不足、尾段漂移、零线穿越、反向修正、多段修正、大幅响应漏召回、峰值时间和启动延迟大误差。
- 输出复发坏样本总表、代表坏样本表、物理错误汇总、分被试坏样本汇总、坏样本矩阵和 4 张图。

### 待做任务
- 对代表坏样本前 10-20 个事件画详细曲线：原始车辆姿态、锚点、方向盘 GT、RBF/KNN/template 预测；Transformer 只作为已经单独跑过的参照，必要时另行叠加。
- 判断反复失败事件是否来自锚点偏差、窗口覆盖不足、原始数据局部异常或模型结构不足。
- 在确认问题来源后设计结构化车辆响应模型。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证继续被车辆-only 坏样本归因和主参照冻结阻塞。

### 可并行任务
- 复核复发坏样本中的原始车辆波形。
- 统计复发坏样本与被试、session、道路上下文、响应类型的关系。
- 准备结构化标签和坏样本曲线绘图协议。

### 需要服务器的任务
- 暂无。本轮坏样本复盘本地完成。

### 不需要服务器的任务
- 坏样本曲线绘图、原始波形复核、报告更新、Git 提交。

## 最新更新：2026-05-12 21:37

### 正在做任务
- 阶段 3 强车辆基线稳健性验证 v0.1 已完成，当前正在归档并准备 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_strong_vehicle_robustness_v0_1.py`。
- 完成 4 个配置：`random_main`、`subject_main`、`session_pre1`、`session_pre3`。
- 输出 strong vehicle robustness 指标表、逐样本指标表、模型信息表、决策表和三张热图。
- 确认 subject-level 下 RBF 仍优于 formal ridge，但 KNN 仍有模板记忆风险。

### 待做任务
- 对 subject-level 和窗口敏感性下反复失败的样本生成固定坏样本复盘图。
- 设计阶段 3 下一版结构化车辆模型：响应分解、关键点 + 残差、多假设或可靠性门控。
- 在冻结车辆-only 主参照前，继续整理反向修正和多段修正失败类型。

### 阻塞任务
- 连续风格、生理和 EEG 增量验证仍被“车辆-only 主参照尚未结构化冻结”阻塞。

### 可并行任务
- 复核 RBF 在反向修正匹配率很低时为什么仍能降低 RMSE。
- 统计 KNN/RBF/Transformer 在 subject-level top bad 样本中的重合关系。
- 准备结构化响应标签：方向、峰值幅度、峰值时间、启动延迟、反向修正、多段修正。

### 需要服务器的任务
- 暂无。本轮稳健性验证本地 CPU 完成。

### 不需要服务器的任务
- 坏样本复盘、结构化车辆模型设计、报告更新、Git 提交。

## 最新更新：2026-05-12 21:24

### 正在做任务
- 阶段 3 车辆-only 统一对照 v0.1 已完成，当前正在归档并准备 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_unified_vehicle_comparison_v0_1.py`。
- 汇总 formal ridge、旧 `vehicle_direct`、RBF/KNN/template、真正 Transformer 和无学习基线，共 15 个 test 模型。
- 生成统一指标表、相对 formal ridge 差异表、指标排名、候选决策表、top bad 重合表和四张汇总图。
- 输出用户查看版中文总结和技术版中文报告。

### 待做任务
- 对 RBF/KNN/template 做 subject-level split 或窗口敏感性验证，判断收益是否稳定。
- 对 top bad 重合样本画图复盘，确认反复失败是锚点问题、车辆历史不足还是模型结构不足。
- 设计下一版结构化车辆模型：响应分解、关键点 + 残差、多假设或可靠性模型。

### 阻塞任务
- 连续风格、生理、EEG 教师和多模态增量验证仍被“强车辆主参照尚未完成稳健性验证”阻塞。

### 可并行任务
- 复核 RBF/KNN 的模板记忆风险。
- 统计 Transformer 与 RBF/KNN 的坏样本重合模式。
- 准备 subject-level split 的统一评价脚本。

### 需要服务器的任务
- 暂无。本轮统一对照为本地表格和图表汇总。

### 不需要服务器的任务
- 统一对照表复核、坏样本复盘、报告更新、Git 提交、下一版车辆结构设计。

## 最新更新：2026-05-12 21:10

### 正在做任务
- 阶段 3 车辆-only Transformer v0.1 已完成，当前正在归档并准备 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_vehicle_transformer_v0_1.py`。
- 使用 `pre2_label2_old_main` + `session_level_split` 训练真正的 vehicle-only Transformer，而不是 KNN/RBF。
- 输入约束已确认：只用事件前车辆时序和事件/道路上下文；不使用驾驶员 ID、生理、脑电、连续风格或未来标签。
- 生成 Transformer 指标表、逐样本指标、训练历史、模型信息、checkpoint、固定预测图、坏样本图、用户查看版中文总结和技术报告。

### 待做任务
- 生成阶段 3 统一强车辆对照表：formal ridge、旧 `vehicle_direct` clean、RBF/KNN/template 候选和 Transformer 放在同一张表里解释。
- 抽查 Transformer 固定图和坏样本图中的大峰值、错侧、尾段漂移、反向修正、多段修正失败类型。
- 决定阶段 3 后续主车辆基线是 Transformer、RBF/KNN 诊断候选、还是进一步结构化响应模型。

### 阻塞任务
- 连续风格、生理、EEG 教师和多模态增量验证仍被“强车辆基线主参照尚未最终冻结”阻塞。

### 可并行任务
- 整理 RBF/KNN 的模板记忆风险说明。
- 对 Transformer 和 RBF/KNN 的坏样本重合度做交叉表。
- 准备 subject-level split 或其它窗口配置的稳健性检查。

### 需要服务器的任务
- 暂无。本轮 Transformer 在本地 CPU 完成。

### 不需要服务器的任务
- 阶段 3 对照汇总、固定图人工抽查、报告更新、Git 提交、下一版结构化车辆模型设计。

## 最新更新：2026-05-12 20:23

### 正在做任务
- 阶段 3 强车辆-only 基线 v0.1 已完成，当前正在归档产物并准备 Git 提交。

### 已完成任务
- 新增并运行 `stage03_vehicle_instability_strong_vehicle_baselines_v0_1.py`。
- 在 `pre2_label2_old_main` + `session_level_split` 上评估 formal ridge、rich ridge、RBF kernel ridge、KNN 模板、方向门控模板、峰值缩放模板。
- 生成强车辆-only 指标表、逐样本指标、模型信息、固定预测图、坏样本图、指标柱状图、与 formal ridge 的逐样本 RMSE 差异图。
- 生成用户查看版和技术版中文报告。

### 待做任务
- 做强车辆基线稳健性验证：subject-level split、其它窗口、KNN 模板记忆风险、session/subject 分布偏置。
- 设计下一版结构化响应模型，重点处理反向修正计数、多段修正、尾段回正和大幅响应幅值。
- 将强车辆基线与旧 `vehicle_direct`、formal ridge、无学习基线整理成统一阶段 3 对照表。

### 阻塞任务
- 连续风格、生理和 EEG 教师路线仍被“强车辆基线尚未完成稳健性验证”阻塞。

### 可并行任务
- 分析 KNN test RMSE 最低但 train RMSE 近 0 的模板记忆风险。
- 抽查 RBF/KNN 在固定图和坏样本图中的物理错误。
- 准备 subject-level split 的强车辆-only 对照。

### 需要服务器的任务
- 暂无。本轮强车辆-only 基线已在本地 CPU 完成。

### 不需要服务器的任务
- 报告归档、表格核对、图像抽查、Git 提交、下一版本地强车辆基线验证。

## 最新更新：2026-05-12 19:35

### 正在做任务
- 阶段 3 v0.1 坏样本错误分型已完成，当前准备归档和提交。

### 已完成任务
- 生成 `ridge_vehicle_context_no_subject` 的 test 逐样本错误标签表。
- 生成错误标签计数、分被试、分响应类型、分道路/事件等级汇总。
- 生成与旧 `vehicle_direct` clean 对照的样本级 RMSE 对比和 top bad 重叠统计。
- 生成错误标签柱状图、旧 deep 对比散点图、top bad 错误矩阵和分被试错误热图。

### 待做任务
- 建立更强车辆时序/结构化响应基线，重点处理反向修正计数、多段修正结构、尾段漂移和幅值不足。
- 将下一版车辆模型与旧 deep、formal ridge、无学习基线放在统一对照表中。

### 阻塞任务
- 连续风格、生理和 EEG 教师路线仍被强车辆基线不足阻塞。

### 可并行任务
- 从 `per_sample_error_taxonomy.csv` 抽取代表性失败样本做固定图集合。
- 分析错误是否集中在特定被试、道路上下文或响应类型。

### 需要服务器的任务
- 暂无。错误分型本地完成。

### 不需要服务器的任务
- 错误分型报告归档、下一版车辆结构设计、小规模本地验证。

## 最新更新：2026-05-12 19:20

### 正在做任务
- 阶段 3 v0.1 车辆-only 基线已完成，当前正在归档结果并提交。

### 已完成任务
- 在正式 `vehicle_instability_highconf_v0_1` 样本上完成无学习基线：零响应、历史趋势外推、训练集平均、按事件类型平均。
- 完成浅层强车辆基线起点：`ridge_vehicle_history_no_subject` 和 `ridge_vehicle_context_no_subject`，均不使用驾驶员 ID。
- 生成阶段 3 指标表、逐样本指标、模型信息、固定预测图、坏样本图和用户查看版总结。

### 待做任务
- 分析坏样本图，汇总大幅响应、错侧、尾段漂移和多段修正失败类型。
- 决定下一版强车辆基线：更强时序模型、结构化响应分解，或先做固定图协议加严。
- 只有强车辆基线稳定后，再进入连续风格和生理增量验证。

### 阻塞任务
- 连续风格、生理和 EEG 教师路线仍被强车辆基线质量阻塞。

### 可并行任务
- 提取本轮 ridge 的 top bad samples，与旧 `vehicle_direct` bad samples 做交叉表。
- 汇总分被试、分响应类型、分道路上下文的错误分布。

### 需要服务器的任务
- 暂无。阶段 3 v0.1 已本地完成。

### 不需要服务器的任务
- 阶段 3 v0.1 归档、坏样本分析、下一版车辆模型设计。

## 最新更新：2026-05-12 19:05

### 正在做任务
- 阶段 2 正式车辆失稳样本清单 `vehicle_instability_highconf_v0_1` 已完成；当前准备进入阶段 3 新流程车辆基线。

### 已完成任务
- 生成正式 `samples_master.csv/jsonl`：906 个高置信失稳事件、3 个窗口、2718 行样本。
- 生成 `event_anchor_table.csv`：908 个高置信事件全部保留锚点追溯，其中 2 个有明确排除原因。
- 生成 `split_table.csv` 和 `split_feasibility_report.csv`：默认 session-level split 为 train 611、val 156、test 139，同时保留 random-event 和 subject-level split。
- 生成 `sample_exclusion_reasons.csv`：2 个事件因 `history_underflow_for_3s_oldcode` 未进入正式 v0.1 样本。
- 生成数据版本卡和用户查看版总结。

### 待做任务
- 阶段 3：基于 `vehicle_instability_highconf_v0_1` 的主窗口 `pre2_label2_old_main` 建立无学习基线。
- 阶段 3：建立新流程强车辆基线，禁止使用驾驶员 ID，标准化只在 train split 拟合。
- 将旧 `vehicle_direct` 坏样本与新流程车辆基线坏样本做交叉对比。

### 阻塞任务
- 连续风格、生理和 EEG 教师路线仍被新流程强车辆基线阻塞。
- 生理/脑电窗口构建被模态可用性和同步复核阻塞；当前只记录文件可用性，未提取生理/脑电窗口。

### 可并行任务
- 设计新流程强车辆基线特征协议和固定图协议。
- 汇总主窗口响应类型分布和困难样本候选。
- 检查三模态齐全子集是否足够支撑后续生理/EEG 公平对照。

### 需要服务器的任务
- 暂无。阶段 3 初始无学习/车辆基线可以先本地运行。

### 不需要服务器的任务
- 新流程无学习基线、强车辆基线小规模验证、报告和图表生成。

## 最新更新：2026-05-12 18:45

### 正在做任务
- 本轮旧 `vehicle_direct` 全量车辆-only clean 对照已经完成，当前没有训练或评估任务在运行。
- 收尾任务是归档产物、提交 Git，并把本轮结果作为旧流程历史对照和坏样本来源。

### 已完成任务
- 生成旧深度入口专用 clean vehicle manifest：对 84 个原始车辆文件做 200Hz 插值清洗，避免旧 loader 把原始缺失点直接填 0。
- 完成 clean manifest 校验：906 个可用高置信失稳样本，session-level split 为 train/val/test = 611/156/139；clean 轨迹与此前 `.npz` 标签首批 50 个样本最大差异约 2e-6。
- 完成旧 `vehicle_direct` 全量 CPU run：输入只用车辆历史和旧入口上下文，不含生理、脑电、连续风格或教师状态。
- 完成固定预测图、坏样本图、分被试表、分响应类型表和中文报告。
- 已判定并清理一次无效 raw direct run：原因是旧代码直接读原始车辆 CSV 时会把交替缺失点填 0，造成不真实的方向盘标签跳变。

### 待做任务
- 构建新流程正式 `samples_master`、split table 和 dataset version card，主清单暂以 906 个可用高置信失稳事件为起点。
- 建立新流程强车辆基线：不使用驾驶员 ID，不继承旧标准化/旧样本划分假设，覆盖方向、幅值、错侧、尾段、峰值时间、反向修正、多段修正和困难样本指标。
- 将旧 `vehicle_direct` 的坏样本纳入失败样本库，用于后续判断新模型是否真的改善物理错误。

### 阻塞任务
- 连续风格、生理、脑电教师路线仍被新流程正式强车辆基线阻塞；本轮旧代码结果不能作为它们有效的证据。

### 可并行任务
- 汇总旧 deep 坏样本的道路/响应类型分布。
- 为新流程强车辆基线准备固定预测样本协议。
- 检查 clean manifest 与事件锚点、原始时间戳、输入窗口、标签窗口的追溯字段是否足够。

### 需要服务器的任务
- 暂无。本轮旧代码全量对照已在本地 CPU 完成。

### 不需要服务器的任务
- 新流程 manifest 整理、旧 deep 结果分析、图表归档、中文报告和 Git 提交。

## 最新更新：2026-05-12 15:52

### 正在做任务
- 阶段 2 修正：把车辆失稳事件判定从“全人工逐条标注”改为“道路设定 + 旧 v400 事件上下文 + 原始车辆动态证据”的自动综合判定。
- 准备基于 `road_guided_auto_accepted_events_v0_1.csv` 生成车辆失稳版 manifest 和处理后车辆窗口。

### 已完成任务
- 已检索旧项目日志中的道路事件记录：旧流程优先使用 `*_events_v400_context.csv` 提供 `road_type_anchor`、`phase_type`、`event_level`、`trigger_idx`。
- 已读取道路设计模块顺序：`curve1/curve2/curve3` 是弯道上下文，`mu1/differentmu_road` 是高风险路面先验，`fix_road/stop/zd` 是特殊道路段先验。
- 已生成道路设定引导的车辆失稳事件判定 v0.1：全量 1227 个候选，自动/已确认采用 701 个，中间复核 177 个，低证据剔除 349 个。

### 待做任务
- 基于 701 个采用候选生成 `vehicle_instability_road_guided_v0_1` 样本 manifest、split 表和数据版本卡。
- 重新生成车辆失稳版处理后车辆窗口，方向盘只作为未来响应标签，不作为失稳锚点来源。
- 在车辆失稳样本上重新运行无学习基线和强车辆基线；之前 404 个弯道样本上的阶段 3 结果继续降级为诊断材料。

### 阻塞任务
- 风格和生理有效性验证仍被正式车辆失稳 manifest 与强车辆基线阻塞。
- 任何“生理有效”“连续风格有效”的结论仍不能提前宣称。

### 可并行任务
- 抽查 `hybrid_accept_high`、`hybrid_accept_medium`、`hybrid_review_conflict_or_medium` 的代表片段。
- 生成道路模块/旧 v400 上下文/车辆动态证据的可视化汇总。
- 编写车辆失稳版数据版本卡。

### 需要服务器的任务
- 暂无。当前事件判定、manifest 和窗口处理均可本地完成。

### 不需要服务器的任务
- 读取旧日志和道路设计；自动事件判定；本地样本 manifest；本地处理后车辆窗口；中文报告和透明化文件更新。

## 最新更新：2026-05-12 16:25

### 正在做任务
- 阶段 2 修正：基于全部 91 个原始车辆 CSV 的车辆失稳样本重筛已经完成，下一步准备生成正式 `samples_master` 和处理后车辆窗口。

### 已完成任务
- 已直接读取 `原始车辆数据/<被试名>/*.csv` 下全部 91 个原始车辆文件。
- 已按非方向盘动态种子 `ay/roll_rate` 重筛：4581 个动态种子合并为 1991 个候选事件。
- 已叠加旧 v400 事件上下文和道路模块先验：高置信主清单 908 个；自动/已确认采用扩展清单 1348 个；中间复核 269 个；低证据剔除 374 个。
- 已确认旧候选表漏掉的 3 个原始车辆文件本次也被读取，但这 3 个文件没有形成失稳候选。

### 待做任务
- 以 `all_raw_vehicle_instability_primary_high_confidence_v0_1.csv` 为保守主清单生成车辆失稳版样本 manifest。
- 以 `all_raw_vehicle_instability_auto_accepted_v0_1.csv` 作为扩展清单，后续做敏感性对照。
- 为高置信主清单生成处理后车辆输入窗口和方向盘未来响应标签。
- 在新失稳样本上重新运行无学习基线和强车辆基线。

### 阻塞任务
- 正式阶段 3 车辆基线仍被“车辆失稳样本 manifest 和处理后窗口尚未生成”阻塞。
- 连续风格、生理和脑电有效性验证仍然不能开始。

### 可并行任务
- 抽查高置信主清单中的代表片段。
- 统计 908 主清单和 1348 扩展清单在被试、道路模块、旧 v400 上下文中的分布。
- 准备车辆失稳版 `split_table`。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前 manifest、窗口处理、报告和基线小规模验证均可本地执行。

更新时间：2026-05-12 14:55:00

## 正在做任务

- 阶段 2 修正：将主事件定义从“弯道/道路曲率候选”改为“车辆失稳候选”。
- 整理 `vehicle_instability_onset_codex_v0_1` 数据版本，确认它只能作为 Codex 自动筛选版本，不能叫人工真值。
- 使用本地页面 `http://127.0.0.1:8766/` 审查车辆失稳候选片段；当前页面已默认读取失稳候选，不再默认展示 404 个弯道样本。

## 已完成任务

- 初始化新流程目录和透明化文件。
- 写入长期目标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/R2E_STEERING_LONG_GOAL_CN.md`。
- 完成阶段 0：旧流程冻结与重建准则。
- 完成阶段 1：原始车辆/生理/脑电 CSV 审计；只扫描三个原始目录下被试名文件夹内 CSV；原始 CSV 未被修改。
- 完成阶段 2 初版：候选事件总表、样本清单、split 表、道路设计清单和道路曲率车辆窗口。
- 完成阶段 3 诊断版：在道路曲率候选上做过无学习/车辆基线，但已降级为诊断材料。
- 完成人工标注审查包和键盘播放器。
- 完成 Codex 弯道自动审阅 v0.1：404 个弯道候选，现已降级为道路上下文参考。
- 完成车辆失稳自动审阅 v0.1：
  - 非方向盘动态种子 1833 个。
  - 合并后车辆失稳候选 1227 个。
  - 自动高/中置信采用 358 个。
  - 需要人工复核 462 个。
  - 低失稳证据建议剔除 407 个。
- 本地键盘审查服务已重启为 `vehicle_instability_event_reviewer_v0_1`，PID 33512。

## 待做任务

- 抽查失稳候选示例图，尤其是 `auto_accept_instability_high` 和 `needs_human_review`，确认是否存在正常过弯被误判为失稳。
- 写 `vehicle_instability_onset_codex_v0_1` 数据版本卡。
- 生成基于车辆失稳锚点的处理后车辆窗口。
- 生成失稳版本的 `samples_master` / `split_table` 子集或独立 manifest。
- 在失稳 manifest 确认后，重新运行无学习基线和强车辆基线。
- 之后再决定是否进入连续风格和生理增量验证。

## 阻塞任务

- 正式阶段 3 强车辆基线被“失稳样本 manifest 尚未冻结”阻塞。
- 连续驾驶风格有效性验证被“正式强车辆基线尚未完成”阻塞。
- 生理/脑电有效性验证被“正式强车辆基线、无泄漏样本和置乱协议尚未完成”阻塞。
- 旧道路曲率阶段 3 结果不能用于支持最终结论，只能作为历史诊断对照。

## 可并行任务

- 抽查高置信车辆失稳示例图。
- 抽查 `needs_human_review` 候选。
- 整理失稳版本数据版本卡。
- 编写失稳车辆窗口生成脚本。
- 检查旧道路设计记录与失稳候选之间的关系，用于判断“正常过弯”与“失稳”的边界。

## 需要服务器的任务

- 暂无。当前任务均可本地完成。
- 后续如果进行全量多 seed 模型训练，再考虑使用远程服务器。

## 不需要服务器的任务

- 车辆失稳候选审查。
- 本地页面标注。
- 车辆窗口生成。
- 数据版本卡和中文报告整理。
- 失稳 manifest 和 split 表构建。

## 追加更新：2026-05-12 16:35

### 已完成任务

- 完成道路事件位置与锚点重建审计 v0.1。
- 已生成道路模块位置表、每条记录道路映射摘要、每条记录道路模块进入/离开时间表、旧锚点对齐表、道路引导候选对齐表、中文报告、用户查看版总结和概览图。

### 正在做任务

- 当前没有训练任务，也没有服务器任务。
- 当前主线仍是样本与锚点审计，不进入模型训练。

### 待做任务

- 根据 `old_new_anchor_alignment_v0_1.csv` 把旧样本分为：锚点可保留、旧锚点可能偏早、旧锚点可能偏晚、道路映射不可靠。
- 抽查 `old_after_body_onset`、`old_before_body_onset`、`old_unaligned_or_unverified` 的代表图，确认旧流程坏样本是否集中在这些锚点风险组。
- 只用高/中可靠道路映射和强车身姿态证据生成新锚点样本清单。
- 在新锚点样本清单确认后，重新做强车辆基线，再谈连续风格和生理数据。

### 阻塞任务

- 旧流程继续训练被“事件锚点尚未重新冻结”阻塞。
- 风格和生理有效性验证被“新锚点强车辆基线尚未完成”阻塞。

## 追加更新：2026-05-12 17:05

### 已完成任务
- 将 908 个全原始车辆失稳高置信事件转换为旧阶段 3 诊断窗口和旧深度模型 manifest。
- 生成 `vehicle_instability_allraw_highconf_v0_1` 处理后数据包：906 个可用于旧代码的失稳事件，2718 行窗口样本，3 个窗口配置。
- 运行旧车辆基线诊断：无学习基线、旧 `ridge_vehicle_summary`、去掉被试 one-hot 的 `ridge_vehicle_no_subject`。
- 用旧 `run_event_conditioned_trajectory_baseline.py` 的 `vehicle_direct` 入口完成 CPU smoke run，确认旧深度模型训练闭环可读取新 manifest。

### 正在做任务
- 当前没有后台训练任务正在运行。

### 待做任务
- 决定是否对旧 `vehicle_direct` 进行全量车辆-only run。
- 如果进行全量旧模型测试，必须生成固定预测图、坏样本图、分被试结果和分响应类型结果，不能只记录 RMSE。
- 将旧代码测试结果与新流程强车辆基线分开命名，避免把旧代码 smoke 当作正式阶段 3 结论。

### 阻塞任务
- 连续风格、生理和脑电有效性验证仍被正式强车辆基线阻塞。
- 旧深度模型 full run 需要先确认是否继续沿用旧结构；当前 smoke 只证明入口可运行。

### 可并行任务
- 抽查旧代码坏样本图。
- 准备全量旧模型车辆-only run 配置。
- 将 906 个可用高置信失稳事件整理为新流程正式 manifest。

### 需要服务器的任务
- 暂无。全量旧深度模型如果本地 CPU 过慢，再考虑服务器。

### 不需要服务器的任务
- 旧代码窗口处理、旧车辆基线诊断、旧 manifest loader smoke、中文报告整理均已在本地完成。


## 追加任务状态：2026-05-12 17:18

### 已完成
- 场景触发点审计 v0.2：解析 `.aed` 交通对象和触发点，并与旧 v400 锚点对齐。

### 待做
- 被试车道投影：确认每条车辆记录经过 `longstraight` 触发点时的实际车道和相对位置。
- 场景触发点样本候选：把激活、停车、换道触发点分开生成候选锚点，不再混成单一“直道事件”。
- 坏样本交叉检查：检查旧模型坏样本是否集中在旧锚点明显偏早或偏晚的组。

### 不需要服务器
- 当前场景触发点审计、车道投影和报告生成都可以本地完成。
# 最新更新：2026-05-12 17:15

### 正在做任务
- 本地运行旧代码 `vehicle_direct` 全量车辆-only 对照：manifest 为 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_session_level_split.csv`，run prefix 为 `OLD_FULL_INSTABILITY_HIGHCONF_VEHICLE_DIRECT_V0_1`。

### 已完成任务
- 已完成旧代码窗口和 manifest 构建：906 个高置信失稳事件可进入旧深度模型。
- 已完成旧无学习/ridge 车辆诊断和旧 `vehicle_direct` smoke run。

### 待做任务
- 训练结束后生成旧 `vehicle_direct` 全量 run 的逐样本预测表。
- 训练结束后补固定预测图和坏样本图。
- 将本次结果写入中文报告、总看板、每日日志、产物索引和服务器运行记录。

### 阻塞任务
- 连续风格、生理、脑电有效性验证仍然被正式新流程强车辆基线阻塞；本次旧代码全量 run 只做历史对照。

### 可并行任务
- 训练运行期间可整理报告模板和绘图脚本。

### 需要服务器的任务
- 暂无。本次优先本地 CPU 运行；如果明显过慢，再由用户决定是否使用服务器。

### 不需要服务器的任务
- 旧代码全量 `vehicle_direct` 单 seed 车辆-only 对照、逐样本评估、固定图/坏样本图和中文报告。
## 2026-05-12 20:45 阶段 2 补充任务状态

### 已完成

- 提取小论文中的场景设计和锚点依据。
- 解析道路配置中的车道、被试方向和 `mu` 附着系数。
- 生成被试方向候选锚点清单 v0.4。
- 更新用户查看版总结、完整报告、项目状态和产物索引。
- 根据用户补充，将 `middle_section` 从背景/过渡段修正为连续超车负荷事件段，并重新生成候选锚点清单。

### 正在做

- 当前没有训练任务在运行。
- 当前没有服务器任务在运行。

### 待做

- 生成候选锚点可视化图，优先检查 `curve1/curve2` 和 `differentmu_road`。
- 生成 `middle_section` 连续超车可视化图，优先检查连接段入口、横向偏移变化峰值、横向加速度峰值、横摆角速度峰值和旧锚点关系。
- 将旧锚点与候选锚点分为“可保留、偏早、偏晚、语义不清”。
- 继续确认 `fix_road`、`stop`、`zd` 的具体实验设计位置。

### 阻塞或风险

- `curve3` 和 `zd` 当前没有记录级候选，说明现有道路映射或车辆记录覆盖不足，需要继续查原因。
- 4209 行候选锚点不能直接进入训练，必须先做可视化复核。
- `middle_section` 是连续负荷段，不是单点突发事件；需要用车身横向动态筛选明显超车/变道样本。
- 不能把方向盘响应峰值当作事件触发点真值，只能作为旧方案对照或响应标签。

## 2026-05-12 21:20 阶段 2 触发点语义修正

### 已完成

- 根据用户最新说明，修正 `longstraight` 和 `fix_road` 的事件语义。
- `longstraight` 已加入 MAN TGL 25->26 显式变道触发点和 Chrysler300 Stop 触发点。
- `fix_road` 已加入 MAN TGL 25->26 和 BMW m340 26->25 两类显式变道触发点。
- 候选锚点清单已从 4209 行更新为 4519 行。
- 已更新用户总结、完整报告、场景工作表和修正说明。

### 正在做

- 当前没有训练任务在运行。
- 当前没有服务器任务在运行。

### 待做

- 生成 `longstraight` 显式变道/停车触发可视化图，比较触发点、旧锚点、横向加速度、横摆角速度、横向偏移、制动和方向盘响应。
- 生成 `fix_road` 两类显式变道触发可视化图，确认维修/施工变道是否引发被试避让或修正。
- 继续生成 `middle_section` 连续超车、`curve1/curve2` 弯道和 `differentmu_road` 低附着的候选锚点可视化图。
- 将旧锚点与候选锚点分为“可保留、偏早、偏晚、语义不清”四类。

### 阻塞或风险

- 显式触发点只是候选，不是最终真值；必须经过车身姿态和方向盘响应图确认。
- 普通连续车流和显式触发对象要分开处理，不能把整段车流全部当事件，也不能把显式触发全部当背景。

## 2026-05-12 21:45 事件候选筛选 v0.5

### 已完成

- 对 v0.4 的 4519 个候选锚点完成自动评分。
- 生成去重后建议复核清单 534 个。
- 生成高置信复核清单 314 个。
- 生成分场景统计、概览图和 56 张代表性复核图。

### 正在做

- 当前没有训练任务在运行。
- 当前没有服务器任务在运行。

### 待做

- 人工/半自动查看 56 张代表性复核图，先判断锚点是否明显偏早、偏晚或无响应。
- 对高置信复核清单按场景抽样扩大绘图，特别是 `fix_road` 显式变道、`longstraight` 显式变道/停车、`middle_section` 连续超车和 `differentmu_road` 低附着。
- 生成 v0.6 事件样本候选清单，只保留“设计点合理 + 响应合理 + 窗口可训练”的事件。

### 阻塞或风险

- 高置信复核不是最终训练样本。
- 当前高置信中 `longstraight` 显式变道只有 3 个，说明该触发与被试响应关系需要重点看图确认；不能直接说变道无效，也不能直接大量纳入样本。
- `middle_section` 高置信多来自段入口，但连续超车是持续任务，不一定是单点触发；需要看是否发生明显横向动作。

## 2026-05-12 22:35 GPTPro 回复后的任务状态

### 已完成

- 归档 GPTPro 对 v0.4/v0.5 事件锚点分析的回复摘要。
- 生成已填充的决策记录和行动项。
- 生成 v0.6 事件样本筛选规则草案。

### 正在做

- 当前没有训练任务在运行。
- 当前没有服务器任务在运行。

### 待做

- 基于 56 张代表性复核图生成复核标注表。
- 标注字段包括 `pass/early/late/weak_response/continuous/coordinate_issue/unclear/exclude`。
- 生成 v0.6 四类事件表：
  - `primary_training_events_v0_6`
  - `manual_review_events_v0_6`
  - `response_confirm_only_v0_6`
  - `holdout_or_excluded_v0_6`
- 第一版训练候选只考虑 `curve1/curve2`、`differentmu_road raw μ` 和人工通过的 `fix_road`。

### 阻塞或风险

- GPTPro 回复是外部审查意见，不能替代本地复核。
- v0.6 样本表未生成前，不恢复风格/生理训练。
## 2026-05-13 00:37 阶段 3 干净响应任务车辆-only 对照

### 已完成任务
- 完成 `A_instant2s_core` 轨道车辆-only 对照：84 个事件，session-level train/val/test=62/10/12。
- 完成 `B_response3s_strict_core` 轨道车辆-only 对照：270 个事件，session-level train/val/test=188/42/40。
- 输出 clean-task 指标表、逐样本指标表、模型信息表、val 选择表、固定预测图、坏样本图、用户查看版中文总结和技术报告。
- 已确认本轮没有使用生理、脑电、连续风格、驾驶员 ID、服务器或服务器密码文件。

### 正在做任务
- 整理并提交本轮 clean-task 车辆-only 对照产物。

### 待做任务
- 复查 `B_response3s_strict_core_bad_samples_test.png` 和逐样本表，标出 RBF KRR 仍失败的主要物理类型。
- 决定下一轮是否在 B 轨道上推进结构化车辆-only 响应分解模型，而不是继续直接堆 KNN/template。
- 把 A 轨道作为小样本即时响应诊断，不把单次 KNN 排名升级为主线。

### 阻塞任务
- 连续风格、生理和 EEG 有效性验证仍被强车辆基线稳定性和物理错误闭环阻塞。
- 长事件/持续控制轨道仍未进入主线训练，需要单独拆分或复核。

### 可并行任务
- B 轨道坏样本物理归因。
- A 轨道小样本稳定性复核。
- 清理固定图协议和固定坏样本索引。

### 需要服务器的任务
- 暂无。本轮 clean-task 车辆-only 对照已在本地 CPU 完成。

### 不需要服务器的任务
- 指标表复查、坏样本图复盘、报告整理、下一轮结构化车辆-only 方案设计。
## 2026-05-13 00:55 阶段 3 B 轨道车辆-only 坏样本复查

### 已完成任务
- 完成 `B_response3s_strict_core` + `rbf_kernel_ridge_context_no_subject` 的 test 坏样本物理失败类型复查。
- 输出 B 轨道 RBF KRR 的失败标记汇总、top bad 样本表、分响应形态/分被试/分道路模块汇总、3 张图、用户查看版中文总结和技术报告。
- 确认本轮没有训练新模型，没有使用生理、脑电、连续风格、驾驶员 ID、服务器或服务器密码文件。

### 正在做任务
- 整理并提交 B 轨道坏样本复查产物。

### 待做任务
- 基于 B 轨道坏样本表，设计车辆-only 响应分解模型：方向、幅值、峰值时间、反向修正/多段修正类型、轨迹残差。
- 把 B 轨道 top bad 样本固定为下一轮结构化车辆-only 模型的必看图集。

### 阻塞任务
- 连续风格、生理和 EEG 有效性验证继续被强车辆基线结构性失败阻塞。

### 可并行任务
- 响应分解标签表构建。
- B 轨道固定图协议整理。
- A 轨道小样本稳定性复核。

### 需要服务器的任务
- 暂无。本轮坏样本复查已在本地 CPU 完成。

### 不需要服务器的任务
- 失败类型统计、响应分解标签设计、中文报告整理。

## 2026-05-13 01:05 阶段 2 回补：episode-first 事件样本 v0.6

### 已完成任务
- 根据 GPTPro 最新建议，把样本筛选从“触发点是否正确”改成“是否存在车辆动态异常 + 方向盘响应 + 回正/纠正 episode”。
- 完成 `build_episode_first_events_v0_6.py` 更新和运行。
- 输出 v0.6 总表、严格核心表、坐标需复核扩展表、弱响应/负样本表、分桶汇总表、36 张分组代表图、中文技术报告和用户查看版总结。

### 正在做任务
- 人工抽看 v0.6 代表图，判断严格核心和坐标复核扩展是否适合进入下一步纯车辆/道路预测对照。

### 待做任务
- 复核 36 张代表图，重点看严格核心是否真的是完整 episode，坐标复核扩展是否只是道路坐标重置。
- 若通过复核，构建两个纯车辆/道路对照：严格核心小样本对照；扩展候选但不使用横向偏移特征的对照。
- 把旧阶段 3 clean-task 基线结果标记为诊断参考，不直接替代这次 episode-first v0.6。

### 阻塞任务
- 连续风格、生理和 EEG 有效性验证继续阻塞，直到 episode-first 样本和强车辆/道路基线确认。

### 可并行任务
- 代表图人工复核。
- 准备纯车辆/道路 baseline 输入表。
- 整理坐标跳变样本的道路模块和被试分布。

### 需要服务器的任务
- 暂无。本轮 v0.6 样本表和图已在本地完成。

### 不需要服务器的任务
- 事件表复查、代表图复核、报告整理和下一步 baseline 设计。

## 2026-05-13 01:29 阶段 3 episode-first 纯车辆/道路对照 v0.1

### 已完成任务
- 完成 v0.6 episode-first 正样本扩展集的纯车辆/道路预测对照。
- 输出指标表、逐样本指标表、模型信息表、轨道汇总表、val 选择表、固定预测图、坏样本图、用户查看版总结和技术报告。
- 验证“保留横向偏移特征”没有让 3 秒轨道变好，说明坐标特征没有带来可疑虚高。

### 正在做任务
- 整理本轮结论，并决定是否进入车辆-only 响应分解模型。

### 待做任务
- 基于 v0.6 正样本构建响应分解标签：方向、幅值、峰值时间、回正/反打、多段修正、尾段稳定。
- 训练或评估车辆-only 响应分解模型，再决定是否进入连续风格/生理增量验证。

### 阻塞任务
- 生理、脑电和连续风格仍不能进入有效性结论，因为纯车辆/道路对 v0.6 复杂 episode 的方向、幅值和大幅响应召回仍不稳定。

### 可并行任务
- 响应分解标签生成。
- EP3 扩展轨道坏样本归因。
- v0.6 坐标复核扩展候选的人工图审。

### 需要服务器的任务
- 暂无。本轮是 CPU 轻量对照，已本地完成。

### 不需要服务器的任务
- 指标复查、坏样本图复盘、响应分解方案设计。
# R2E-Steering 当前任务队列
## 最新更新：2026-05-13 07:42

## 正在做任务

- Stage 7h val/test 选择不稳定诊断已完成并提交 Git；当前准备 Stage 7i 候选选择校准/验证集重构。

## 已完成任务

- Stage 7h：已生成候选 split 稳定性、类别分布偏移、数值分布偏移、逐样本收益、分 bucket 收益、keypoint target 误差复核、gate 表和 4 张诊断图；已提交 `d990f8e3`。
- 已确认 test-best non-oracle=`rbf_resid_keypoint_scaled` 不能升级，因为它没有被 val gate 选中。
- 已确认当前最大偏移信号：类别为 `response_family`，数值为 `prob_entropy`。

## 待做任务

- Stage 7i：候选选择校准或验证集重构。优先方案包括多折 session validation、response bucket 分层 gate、道路模块分层 gate、关键点不确定性/候选一致性评分。
- 复核 `response_family` 和 `prob_entropy` 偏移是否解释 `rbf_resid_keypoint_scaled` 的 val/test 反转。
- 在没有稳定非 oracle 选择前，不进入生理/EEG。

## 阻塞任务

- 生理/EEG 有效性实验：仍阻塞。车辆-only 候选选择/校准未稳定。
- 多假设主线升级：仍阻塞。Stage 7h 只是诊断，未产生新可部署模型。
- test-only 选择 `rbf_resid_keypoint_scaled`：禁止作为结论，只能作为下一步选择校准的目标。

## 可并行任务

- response_family 分层 gate 设计。
- road module/risk class 分层 gate 设计。
- 关键点不确定性评分和候选一致性特征复核。

## 需要服务器的任务

- 当前无。Stage 7h 未使用服务器，Stage 7i 初版可本地完成。

## 不需要服务器的任务

- Stage 7h 归档、Git 提交、Stage 7i 校准诊断原型。

## 最新更新：2026-05-13 07:33

## 正在做任务

- Stage 7g keypoint/segment 候选已完成并提交 Git；当前准备 Stage 7h val/test 选择不稳定诊断。

## 已完成任务

- Stage 7g：已实现 train-only 关键点回归、分段轨迹候选、RBF 关键点校正候选和 keypoint/segment oracle 诊断；已提交 `52de7176`。
- 已确认 val gate 选择 `segment_abs_rf_blend_25`，但 test RMSE=0.536176，比 RBF/KNN 差 +0.002509，gate=`no_upgrade`。
- 已确认 `rbf_resid_keypoint_scaled` 在 test 上 RMSE=0.508538，但它不是 val 选中的策略，因此只能作为下一步诊断信号，不能升级主线。

## 待做任务

- Stage 7h：复核 val/test 分布差异、关键点回归误差、候选置信度和校准，解释为什么 test 上存在好候选但 val gate 选不中。
- 设计不看 test 标签的候选选择置信度：例如关键点不确定性、RBF/候选一致性、道路模块分层和 response bucket 分层。
- 继续固定 RBF/KNN 为主参照，不进入生理/EEG。

## 阻塞任务

- 生理/EEG 有效性实验：仍阻塞。车辆-only keypoint/segment 候选选择未稳定。
- 多假设主线升级：仍阻塞。Stage 7g 的可部署选中策略未超过 RBF/KNN。
- 按 test 事后选择 `rbf_resid_keypoint_scaled`：禁止。该现象只能作为下一步校准/选择器诊断。

## 可并行任务

- val/test 分布差异表。
- `rbf_resid_keypoint_scaled` 收益样本和 val 失败样本复盘。
- keypoint target scatter 和错误分层复核。

## 需要服务器的任务

- 当前无。Stage 7g 未使用服务器，下一步校准诊断仍可先本地完成。

## 不需要服务器的任务

- Stage 7g 归档、Git 提交、Stage 7h 诊断表和图。

## 最新更新：2026-05-13 07:19

## 正在做任务

- Stage 7f response-factorized vehicle-only candidate v0.1 已完成并提交 Git；当前准备 Stage 7g keypoint/segment candidate 设计。

## 已完成任务

- Stage 7f：已实现响应分解候选原型，输出 factor 预测、候选逐样本指标、policy 指标、oracle 诊断、gate 表、固定图、oracle gain 图、用户总结和技术报告；已提交 `12cef06b`。
- 已确认 validation gate 最终选择 RBF/KNN 主参照，test RMSE=0.533667，delta=+0.000000，gate=`no_upgrade`。
- 已确认 response-factorized oracle RMSE=0.440217、combo oracle RMSE=0.388119 只说明候选空间潜力，不是可部署提升。

## 待做任务

- Stage 7g：把 response-factorized 原型升级为可训练 keypoint/segment candidate，重点改幅值、尾段回正/漂移、反向修正和多段修正。
- 复核 Stage 7f 中幅值因子和尾段因子预测不稳的样本，作为下一版候选生成训练标签和失败样本清单。
- 继续把 RBF/KNN 作为车辆-only 主参照；任何新候选必须用 train 拟合、val 选策略、test 一次报告。

## 阻塞任务

- 生理/EEG 有效性实验：仍阻塞。车辆-only 候选生成和非 oracle 选择仍未形成可部署提升。
- 多假设主线升级：仍阻塞。当前只有 oracle/组合 oracle 上限，没有 validation gate 批准的可部署收益。
- selector-only 继续堆模型：阻塞。Stage 7b、7d、7f 都未通过 gate。

## 可并行任务

- 幅值/尾段/修正段失败样本复盘。
- Stage 7f oracle gain 样本图人工复核。
- 下一版 keypoint/segment candidate 训练标签和固定图清单设计。

## 需要服务器的任务

- 当前无。Stage 7f 未使用服务器，下一步原型仍可先本地完成。

## 不需要服务器的任务

- Stage 7f 归档、Git 提交、Stage 7g 轻量原型设计和图表复核。

## 最新更新：2026-05-13 06:50

## 正在做任务

- Stage 7 后续方向判断：基于 Stage 7c 导出的完整候选轨迹，决定继续做非 oracle selector v0.2，还是重新设计候选生成。

## 已完成任务

- Stage 7c 候选轨迹导出与差异审计 v0.1：已复现 RBF/KNN、keypoint residual、top-K branch/top1 预测，导出 `stage07c_candidate_trajectories.npz`，并生成指标表、差异表、oracle 摘要、gate 表、固定图、高差异图和 oracle gain 图；已提交 `48b8c438`。
- Stage 7b 非 oracle top-K selector v0.1：val 选择的策略在 test 上完全退回 RBF/KNN，未形成可部署增益。
- Stage 7a 非 oracle 选择协议 v0.1：已固定候选池、允许/禁止输入、评估计划和升级 gate。

## 待做任务

- Stage 7d：非 oracle selector v0.2。只允许使用事件前信息和候选预测自身特征，训练只用 train，调参只用 val，test 只报告。
- 如果 Stage 7d 仍不能超过 RBF/KNN：重新设计候选生成，使候选覆盖方向、幅值、峰值时间、尾段回正/漂移和多段修正，而不是继续堆 selector。
- 继续保持 Stage 5 生理/EEG blocked，直到车辆-only 候选选择问题有可复验结果。

## 阻塞任务

- 生理/EEG 有效性实验：当前阻塞。原因是车辆-only 多候选的非 oracle 选择仍未解决，不能把车辆-only 未解决误差归因给生理/EEG。
- 最终多假设主线升级：当前阻塞。原因是 oracle 上限明显，但还没有非 oracle 可部署策略超过 RBF/KNN。

## 可并行任务

- 候选差异特征审查、坏样本图人工快速复核、Stage 7d selector 协议设计、候选生成新路线草案。

## 需要服务器的任务

- 当前无。Stage 7c 未使用服务器；下一步轻量 selector 也优先本地。

## 不需要服务器的任务

- Stage 7d selector v0.2、候选差异图表复核、报告更新、Git 提交。
# R2E-Steering 当前任务队列
## 最新更新：2026-05-13 06:58

## 正在做任务

- Stage 7e 候选生成重设计协议：基于 Stage 7c/7d 的证据，把下一步从“继续堆 selector”改为“重新设计可解释候选生成”。

## 已完成任务

- Stage 7d 非 oracle selector v0.2：已训练 logistic/RF selector 和置信度 fallback；val gate 选择 `always_rbf_reference`；test delta=0，RBF 选择比例=1.0；gate=`no_upgrade`；已提交 `eb785f4a`。
- Stage 7c 候选轨迹导出与差异审计 v0.1：候选池有 oracle 上限，但不是可部署收益。
- Stage 7b 非 oracle top-K selector v0.1：val 选择的策略在 test 上完全退回 RBF/KNN，未形成可部署增益。

## 待做任务

- Stage 7e：定义新的候选生成协议，要求候选覆盖方向、幅值、峰值时间、尾段、反向修正、多段修正和困难样本。
- 决定是否需要先做一个不训练的候选模板/规则上限审计，再进入可训练多假设模型。
- 继续保持 Stage 5 生理/EEG blocked，直到车辆-only 候选生成与选择问题有稳定基线。

## 阻塞任务

- 生理/EEG 有效性实验：仍阻塞。原因是车辆-only 多候选非 oracle 选择未解决。
- 多假设主线升级：仍阻塞。原因是 Stage 7d 选择器无法超过 RBF/KNN。

## 可并行任务

- 复核 Stage 7c oracle gain 样本图；整理候选生成设计维度；准备 Stage 7e 方案表。

## 需要服务器的任务

- 当前无。

## 不需要服务器的任务

- Stage 7e 协议设计、图表复核、报告更新、Git 提交。
# R2E-Steering 当前任务队列
## 最新更新：2026-05-13 07:05

## 正在做任务

- Stage 7f response-factorized vehicle-only candidate v0.1 准备：按 Stage 7e 蓝图实现新的候选生成，而不是继续调旧 selector。

## 已完成任务

- Stage 7e 候选生成重设计审计 v0.1：已生成响应类型表、候选覆盖表、winner 分布、候选生成蓝图、下一实验计划、gate 表和 4 张图；gate 阻塞 selector-only 继续路线；已提交 `98552bf3`。
- Stage 7d 非 oracle selector v0.2：val gate 选择 RBF/KNN，未升级。
- Stage 7c 候选轨迹导出与差异审计 v0.1：候选池有 oracle 上限，但不是可部署收益。

## 待做任务

- Stage 7f：实现 response-factorized candidates。候选至少包括 RBF anchor、方向/幅值 quantile、峰值时间/onset、尾段模式、反向/多段修正、可靠性门控。
- 建立 Stage 7f 固定图和坏样本图：必须覆盖 Stage 7e 的高优先级 response buckets。
- 如果 Stage 7f 仍不能产生非 oracle 增益，再回到样本定义或响应分解标签，而不是进入生理/EEG。

## 阻塞任务

- selector-only 路线：阻塞。Stage 7b 和 Stage 7d 均退回 RBF/KNN。
- 生理/EEG 有效性实验：仍阻塞。车辆-only 候选生成和非 oracle 选择仍未稳定。
- 多假设主线升级：仍阻塞。当前只有 oracle 上限，没有可部署提升。

## 可并行任务

- 复核 Stage 7e 高优先级 bucket；准备 Stage 7f 训练标签；设计固定图样本清单。

## 需要服务器的任务

- 当前无。Stage 7f 初版优先本地小规模实现和验证。

## 不需要服务器的任务

- Stage 7f 脚本实现、协议表、图表和报告。


## 最新更新：2026-05-18 v0.3 车辆-only 基线

### 已完成任务
- 已构建 v0.3 车辆-only 固定窗口数据集 `v03_vehicle_only_pre2_label5_20hz`。
- 已运行无学习基线和车辆-only 强传统基线。
- 已生成指标表、逐样本指标、固定预测图、坏样本图、用户总结和技术报告。

### 待做任务
- 人工查看固定预测图和坏样本图。
- 判断车辆-only 是否已经比旧样本更符合物理意义。
- 决定是否进入响应类型辅助模型，或加入连续风格/生理数据。


## 最新更新：2026-05-18 v0.3 车辆-only 基线（中文修正版）

### 已完成任务
- 构建 v0.3 车辆-only 固定窗口数据集 `v03_vehicle_only_pre2_label5_20hz`。
- 运行无学习基线：零响应、历史趋势外推、训练集均值、类别均值、工况均值。
- 运行车辆-only 强基线：岭回归、近邻模板、核回归。
- 生成总指标、逐样本指标、分类型表、分被试表、分工况上下文表、固定预测图和坏样本图。

### 待做任务
- 人工查看固定预测图和坏样本图。
- 判断车辆-only 预测是否已经比旧样本更符合物理意义。
- 决定下一步是继续修样本/锚点，还是进入响应类型辅助模型。


## 最新更新：2026-05-18 v0.3 车辆-only 基线（中文修正版）

### 已完成任务
- 构建 v0.3 车辆-only 固定窗口数据集 `v03_vehicle_only_pre2_label5_20hz`。
- 运行无学习基线：零响应、历史趋势外推、训练集均值、类别均值、工况均值。
- 运行车辆-only 强基线：岭回归、近邻模板、核回归。
- 生成总指标、逐样本指标、分类型表、分被试表、分工况上下文表、固定预测图和坏样本图。

### 待做任务
- 人工查看固定预测图和坏样本图。
- 判断车辆-only 预测是否已经比旧样本更符合物理意义。
- 决定下一步是继续修样本/锚点，还是进入响应类型辅助模型。

## 最新更新：2026-05-18 v0.3 样本纳入范围消融

### 已完成任务
- 已跑当前干净集、干净集+待复核、干净集+待复核+可成窗排除样本三档车辆-only 对照。
- 已生成汇总表、每档指标表、逐样本指标和预测图。

### 待做任务
- 查看哪一档的坏样本图更符合物理意义。
- 决定后续训练样本是否采用更宽松纳入范围，或只筛选部分待复核/排除样本。

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
## 2026-05-19 v0.3 样本筛选策略连续对比

### 正在做
- 服务器任务 `v03sweep` 正在运行 `stage03_v03_screening_sweep.py`。
- 任务目标：连续比较多种样本纳入策略，找出哪些 excluded/弱响应/快速转向/姿态类样本值得进入训练。

### 待做
- 监控远程日志：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_20260519_203455.log`。
- 任务完成后拉回 `stage03_v03_screening_sweep` 输出目录。
- 生成中文汇总，判断推荐样本集、极限姿态专用样本集和应排除样本集。

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

- 已完成：在服务器 GPU 上运行 `v04_primary_secondary_review_nolat`。
- 样本范围：v0.4 主训练候选 + 次级候选 + 待复核样本，去掉横向偏移。
- 结果摘要：实际可用窗口样本 1410，test RMSE=0.8067，主阶段 RMSE=0.5786，尾段 RMSE=0.9290，大响应错侧率=0.1398，严重幅值不足率=0.2796，大响应召回=0.9032。
- 对比上一轮：相对 `v04_primary_secondary_nolat`，整体 RMSE 从 0.8402 降到 0.8067，错侧率从 0.1707 降到 0.1398；但大响应召回从 0.9512 降到 0.9032。
- 待做：下一步需要看逐样本预测图和大响应样本分布，判断召回下降是否来自待复核样本稀释，还是模型更保守。
- 服务器状态：本轮 screen 已结束；未记录服务器密码。


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

## 2026-05-20 15:16 v0.5 生理机制验证任务队列

### 已完成
- B1、S1、S2、S3、SF1、SF2、SF3、C1、C2、A1、A2、T2 的 seed2026 训练与汇总。
- 生理可用性检查：HR、EDA、EMG、HR+EDA+EMG 可用；EEG 和含 EEG 组合在 v0.5 新锚点下暂不安全。
- 服务器结果拉回本地，表格和中文报告已生成。

### 待做
- 优先补 T2 和 SF2 的 seed2027/2028，验证是否稳定。
- 对 T2、SF2、C1、S3 生成代表预测图对比，重点看方向、幅值、大响应和尾段。
- 重新设计 v0.5 安全脑电窗口后，再讨论 S4、SF4、C3、C4、A3、T1、T3、T4。

### 暂缓
- 所有含脑电直接输入或脑电教师版本，原因是旧脑电事件特征和 v0.5 新锚点没有安全对齐。
- 单纯把 A1/A2 当作主线，原因是辅助任务当前没有改善幅值不足。

### 当前后台任务
- 无。本轮服务器 screen 已结束。

## 最新更新：2026-05-20 v0.5 脑电特征已可按新锚点重提取

### 已完成任务
- 已审计原始脑电 CSV 字段、清洗后脑电 FIF 和旧脑电特征表。
- 已确认旧 `roll_peak_s` 脑电特征不能直接用于 v0.5 新锚点。
- 已按 v0.5 `anchor_s` 提取严格锚点前 2 秒脑电特征：1388 个样本中 1164 个成功。

### 待做任务
- 将 `v05_eeg_features_pre_anchor_hist2s.csv` 接入 v0.5 机制实验的数据可用性检查。
- 设计脑电可用子集公平对照与缺失掩码方案。
- 在确认接入无误后，再跑 `车辆+脑电`、`车辆+连续风格+脑电` 和脑电教师版本。

### 阻塞任务
- 不能继续使用旧横滚峰值脑电特征表作为 v0.5 主证据。
## 2026-05-20 新增任务：完整记录级 episode 重建

### 暂存
- v0.5 服务器对齐样本集：保留当前样本表、manifest、训练结果、生理/脑电结果和预测图，作为阶段性对照材料。

### 待做
- 重新设计 episode 重建规则：一次完整实验记录允许检测出多个 episode，不再默认一条记录只对应一个事件片段。
- 为完整车辆记录建立状态时间线：方向盘角、方向盘角速度、速度、制动、横向加速度、横摆角速度、横滚角、横滚角速度、横向偏移、路面附着系数、道路/曲率信息。
- 接入已有道路/场景信息：参考已整理的道路模块、低附着、弯道、维修路段、连续超车、longstraight/fix_road/middle_section 等信息，作为 episode 上下文和工况解释字段。
- 为每个 episode 标注多个时间点：事件开始、驾驶员操作开始、车辆响应开始、车辆峰值、驾驶员操作峰值、事件结束。
- 区分任务类型：方向盘预测、车辆状态预测、响应类型分类、保守/弱响应识别、正常弯道对照、锚点偏晚样本回溯。
- 生成新的用户复核图：每张图同时标出事件开始、驾驶员操作开始、车辆响应开始、峰值和结束，而不是只画单根锚点线。

### 暂不继续
- 暂不直接在 v0.5 上继续无目标地调筛选阈值。
- 暂不把 v0.5 固定 2 秒方向盘预测结果作为最终样本定义的证明。
## 2026-05-20 完整记录级 episode 重建 v1.0

### 已完成
- 已新增完整记录级 episode 重建脚本，入口为：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scripts\build_record_level_episode_reconstruction_v1_0.py`。
- 已新增配置文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\configs\record_episode_reconstruction_v1_0.json`。
- 已从 `F:\data_set_process\data_process\01_datasets\数据预处理\原始车辆数据` 扫描完整车辆 CSV。
- 已成功处理 91 条车辆记录，自动检测 1766 个 episode。
- 已生成全量 episode 表、分组表、分被试表、上下文统计表、多信号复核图和静态 3D 轨迹图。

### 当前不做
- 不训练模型。
- 不把 v1.0 自动筛选结果直接升级为最终训练集。
- 不把道路设计点、`.aed` 触发点或旧锚点当作最终事件真值。

### 下一步建议
- 优先人工查看 `core_extreme_核心极限样本`、`conservative_extreme_保守/弱操作极限样本` 和 `review_需要复核` 三类复核图。
- 标出明显错误类型：锚点偏晚、正常弯道混入、车辆变化不足、方向盘维持直线微调、坐标跳变、低附着有效事件。
- 根据人工复核结果，把 v1.0 规则细化成 v1.1，再决定是否构建新的车辆-only 数据集。
- 如果 3D 静态图能帮助理解，再进一步生成真正的时间动画；如果帮助不大，先把精力放在多信号复核图和规则修正。

## 2026-05-20 完整记录级 episode 人工复核整理 v1.1

### 已完成
- 已根据用户复核意见生成 v1.1 样本整理结果。
- 保留为主训练候选：核心极限样本、保守/弱操作极限样本、次级训练样本，共 1383 个。
- 暂不进入主训练：需要复核、边界复核样本，共 380 个。
- 正常弯道或普通操控保留为对照样本，共 3 个。
- 已生成全量带复核决策表、主训练候选表、对照表、舍弃/暂缓表和复核图索引。

### 下一步
- 用 v1.1 主训练候选构建新的车辆-only 数据集。
- 训练前先确认任务窗口：是预测后续方向盘，还是同时预测方向盘、车速、制动、横向加速度、横摆、横滚等车辆状态。
- 第一轮建议先跑车辆-only 强基线，暂不加连续风格和生理数据。
- 如果车辆-only 在 v1.1 上仍出现明显错侧、幅值压缩或预测图不合理，再回到样本规则继续改，而不是直接加生理数据。

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

- 已完成：逐样本拆解 1407 个被排除样本的触发原因，并按恢复优先级输出 A/B/C/D/U 五档。
- 待做：用户人工优先查看 `A_优先人工恢复复核` 和 `B_较可能可恢复` 中的样本图；下一版规则应取消旧版本文字硬继承。
- 不做：本步骤不训练模型，不改变现有 Goal2 训练结果。

## 2026-05-26 Goal2 人工审核图片整理

- 已完成：复制并分类 487 张现有复核图，生成 HTML 入口和索引 CSV。
- 待做：用户先看 `00_A_优先看_旧结论可能误伤`、`01_B_较可能可恢复_看图确认` 两个目录；缺图样本如需复核，再单独补画。
# 当前任务队列

## 最新更新：2026-06-22 v222a cache 与轻量受限残差已完成，下一步进入 no-harm/错配诊断

### 正在做任务
- 当前没有正在运行的后台训练任务。
- v222a 的候选曲线缓存和轻量 bounded residual 已完成；当前应转入结果解释和下一步约束策略设计，而不是继续堆大模型。

### 已完成任务
- 已新增并运行 `stage03_v222a_candidate_curve_cache_20260622.py`，生成两个 pool 的 `candidate_predictions_*.npz`、`candidate_manifest.csv`、`sample_manifest.csv`、feature schema audit、候选曲线指标、v219 交叉检查和 ZIP。
- 已新增并运行 `stage03_v222a_light_fusion_residual_20260622.py`，在固定 formal 候选池上完成非负凸融合和 bounded residual 校准。
- 已确认训练纪律：拟合只用 train，选择只用 validation，test 只在最终 validation-selected 输出固定后报告。
- 已确认泄漏守卫：feature schema 458 行 fail=0，输出名不含 `W3_B4_original_soft/oracle/fallback/true_label`。
- 已确认 ZIP：cache 包 `bad_file=None`、11 文件；light fusion 包 `bad_file=None`、15 文件。

### 待做任务
- 先读：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_selected_metrics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_reference_baseline_metrics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_selected_per_sample_metrics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\reports\v222a_light_fusion_residual_report_cn.md`
- 分解 v222a 的 gain/harm：尤其是 loose pool 低估率下降但 RMSE/tail 变差的样本类型，以及 strict pool tail 变差原因。
- 设计下一版只允许“低估风险样本局部启用”的 no-harm gate；gate 仍必须只按 validation 选择。
- 如果 no-harm gate 不能在 validation 和 test 上同时站住，则停止 v222a 校准主线，回到候选曲线解释和样本/事件类型诊断。

### 阻塞/禁止任务
- 暂不进入 v222b 神经软融合、v223 机制感知 Transformer 或 v224 消融。
- 暂不把本轮 v222a bounded residual 写成新的主线突破：它在 test 上没有同时改善 RMSE、tail 和低估率。
- 暂不做硬切换 router；如果继续 selector，必须先有 no-harm 守卫和 validation-only 选择。
- 任何后续推理特征仍必须排除 `sample_id/event_uid/split/subject/true/oracle/rmse/severe_under/wrong_side` 等泄漏字段。

---
# 当前任务队列

## 最新更新：2026-06-22 v222a no-harm gate 已完成，等待 GPTPro 裁决是否停止 v222a

### 正在做任务
- 当前进入 Codex-GPTPro 闭环：本地 no-harm gate 诊断已经完成，下一步把结果报告给 GPTPro 获取下一轮指令。

### 已完成任务
- 已按 GPTPro 指令完成 `v222a_gain_harm_decomposition`。
- 已完成 diagnostic-only `oracle safe gate upper bound`。
- 已完成 binary validation-only no-harm gate。
- 已确认 validation 两个 pool 都能选到通过 no-harm-first 的 gate。
- 已确认 locked test 两个 pool 都未通过完整 formal gate：
  - loose pool 保留 under 改善但伤 RMSE/tail；
  - strict pool 守住 RMSE/tail 但 under 变差。

### 待做任务
- 把 `v222a_noharm_gate_diagnostic_20260622` 的核心结果报告给 GPTPro。
- 让 GPTPro 裁决：
  - 是否停止 v222a，不进入 v222b/v223；
  - 是否只保留 oracle/gain-harm 作为 case study；
  - 是否允许做一轮更窄的 gate feature 诊断；
  - 如果继续，必须给出新的 stop condition。
- 若 GPTPro 要求继续，本地只接受不违反以下纪律的任务：
  - train-only fitting；
  - validation-only selection；
  - test locked report only；
  - no forbidden inference fields；
  - 不做多候选复杂 router。

### 阻塞/禁止任务
- 暂不进入 v222b neural soft fusion。
- 暂不进入 v223 mechanism Transformer。
- 暂不继续扩大 selector 或做 14-candidate multi-router。
- 暂不把 no-harm gate 写成 formal headline，因为 locked test 未通过。

---
# 当前任务队列更新：2026-06-22 v226 robustness / CI audit 已完成，下一步回报 GPTPro

## 正在做任务
- Codex-GPTPro 闭环继续运行：已完成 GPTPro v226 要求的 `formal robustness / confidence-interval audit`，下一步把 v226 pack、CI 结果、readiness 决策、guard/ZIP 验证摘要报告给 GPTPro，请 GPTPro 给下一轮 bounded 指令。

## 已完成任务
- 已归档 GPTPro v226 指令、采纳决策和执行项：
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v226_formal_robustness_ci_gptpro_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v226_formal_robustness_ci_gptpro_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v226_formal_robustness_ci_gptpro_action_items.md`
- 已新增并运行 `stage03_v226_formal_robustness_ci_audit_20260622.py`。
- 已生成 v226 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`。
- 已确认 formal lock：
  - `loose_main_pool=avg_joint_focus`
  - `strict_main_pool=peak_floor_090`
- 已复现 locked test formal 指标：
  - loose `avg_joint_focus`：RMSE `0.544884`，tail `0.629752`
  - strict `peak_floor_090`：RMSE `0.571770`，tail `0.658306`
- 已完成 bootstrap CI：
  - sample CI：loose RMSE `0.496066-0.593811`，loose tail `0.564811-0.693788`；strict RMSE `0.511036-0.635521`，strict tail `0.581652-0.736696`
  - subject-block CI：loose RMSE `0.428783-0.599684`，loose tail `0.515881-0.687686`；strict RMSE `0.473689-0.615000`，strict tail `0.539479-0.706505`
- 已完成验证：`py_compile`、完整脚本运行、ZIP `bad_file=None`、required files `[]`、metric reproduction pass、leakage guard pass、forbidden scan pass、table alignment pass、figure count 满足要求。

## 待做任务
- 准备并发送下一轮 GPTPro prompt，必须包含：
  - v226 pack 路径；
  - formal model lock；
  - locked test 指标复现；
  - sample bootstrap 和 subject-block bootstrap CI；
  - tail error concentration；
  - readiness decision；
  - ZIP、required files、guard、forbidden scan、table alignment 和 figure count 验证结果；
  - 本轮未训练、未调 tau/threshold、未创建 gate/router、未运行 v222b/v223 的边界说明。
- 等待 GPTPro 给出下一轮 bounded 指令；若新指令涉及 v222b/v223、新 gate/router、新 tau 或 test-based retuning，必须先确认是否满足当前 guardrail 和 stop condition。

## 阻塞/禁止任务
- 不进入 v222b neural gate / neural soft fusion。
- 不进入 v223 new candidate generator / mechanism Transformer。
- 不继续做 v222a gate_v2、新 tau、新 multi-router 或 test-based config。
- 不把 oracle / true label / fallback / diagnostic-only row 写成 deployable model 或 formal headline。
- 不把 `W3_B4_original_soft` 写入 formal leaderboard、formal oracle、formal gate、usage table 或 selected config。
---
# 最新更新：2026-06-22 v227 reporting-only 写作/claim readiness 包已完成

## 正在做任务
- Codex-GPTPro 闭环继续运行，但 GPTPro 当前回报通道临时阻塞：桌面端空停，Chrome bridge 无法验证 Pro/进阶模式并拒绝发送 v227 prompt。当前不启动新实验，只保留 v227 写作材料和阻塞归档，等待 GPTPro 恢复后再回报。

## 已完成任务
- 已归档 GPTPro 回报阻塞：
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v226_result_gptpro_response_blocked.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v226_result_gptpro_decision_blocked.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v226_result_gptpro_action_items_blocked.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v227_result_gptpro_response_blocked.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v227_result_gptpro_decision_blocked.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260622_v227_result_gptpro_action_items_blocked.md`
- 已新增并运行 `stage03_v227_paper_claim_readiness_pack_20260622.py`。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\v227_paper_claim_readiness_pack.zip`
- 验证：`py_compile`、完整脚本运行、ZIP `bad_file=None`、required files `[]`、no-model guard、source checks、formal lock 均通过。

## 待做任务
- 等 GPTPro 通道恢复后，发送 `reports/v227_next_gptpro_prompt_ascii.md`，请求下一条 bounded 指令；在恢复前不要继续扩展实验路线。

## 禁止任务
- 仍不进入 v222b/v223、新 tau、新 gate/router、test-based retuning 或任何新模型训练。
- 不把 v227 说成 GPTPro 新批准的实验方向；它只是本地 reporting-only fallback。

---
# 最新更新：2026-06-23 goal-level blocked，等待 GPTPro 通道恢复

## 正在做任务
- 当前不再自动继续本地实验。Codex-GPTPro 闭环已被同一个外部 GPTPro 通道问题连续阻塞：桌面端没有有效 bounded 回复，Chrome bridge 无法验证 Pro/进阶模式。

## 已完成任务
- 已完成并验证 v226 formal robustness / CI audit。
- 已完成并验证 v227 reporting-only paper / claim readiness pack。
- 已新增 goal-level blocked 归档：
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_goal_blocked_gptpro_channel_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_goal_blocked_gptpro_channel_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_goal_blocked_gptpro_channel_action_items.md`

## 待做任务
- 用户恢复 GPTPro / ChatGPT Pro 通道后，重新发送或手动提供 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\reports\v227_next_gptpro_prompt_ascii.md` 的问题。
- 拿到有效 GPTPro 回复后，先归档 raw response / decision / action items，再执行一个符合 guardrail 的 bounded 指令。

## 禁止任务
- 没有有效 GPTPro 新指令前，不进入 v222b/v223、新 tau/threshold、新 gate/router/selector、新模型训练、formal headline 改动或 test-based retuning。

---
# 当前任务队列更新：2026-06-23 v228 final paper artifact freeze 已完成（最新）

## 正在做的任务

- 当前没有正在运行的训练或模型搜索任务。
- Codex 已按本地 ChatGPT Desktop 软件端的有效 GPTPro 回复完成 v228 最终论文产物冻结包。

## 已完成任务

- 已修正 GPTPro handoff 方式：上一轮中文 prompt 在本地软件端显示为乱码，因此改用纯 ASCII handoff/retry。
- 已归档有效 GPTPro 回复：
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v228_local_gptpro_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v228_local_gptpro_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v228_local_gptpro_action_items.md`
- 已新增并运行 `stage03_v228_final_paper_artifact_freeze_20260623.py`。
- 已生成 v228 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v228_final_paper_artifact_freeze_20260623`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v228_final_paper_artifact_freeze_20260623\v228_final_paper_artifact_freeze_pack.zip`
- 已完成验证：`py_compile`、完整脚本运行、ZIP、required files、formal lock、主指标、CI 行数、forbidden scan、guardrail、consistency 全部通过。

## 下一步候选任务

- 如果继续 GPTPro loop：把 v228 结果和验证摘要回报给 GPTPro，只要求一个 bounded 下一步。
- 如果进入论文写作：直接读取 v228 的 `reports/manuscript_results_section_draft_cn.md` 和 `reports/manuscript_claim_boundary_notes_cn.md`。

## 禁止任务

- 不进入 v222b/v223。
- 不做新 tau/threshold 搜索。
- 不做新 gate/router/selector。
- 不训练新模型，不生成新预测。
- 不改变 formal headline，不做 test-based retuning。

---
# 当前任务队列更新：2026-06-26 v243 v241 guarded fine-tune 已完成（最新）

## 正在做的任务

- 当前没有正在运行的训练进程。
- v243 已完成训练、评估、报告、打包和 guardrail/leakage 验证。

## 已完成任务

- 已新增并运行 `stage03_v243_v241_guarded_finetune_20260626.py`。
- 已在 v241 `v241_tcn_mha_h96` 基础上完成 guarded fine-tune：
  - hard sample point weighting；
  - v241 teacher guard loss；
  - optional teacher anchor；
  - 曲线级 validation snapshot selection。
- 已生成 v243 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626\v243_v241_guarded_finetune_pack.zip`
- validation 结论：
  - `v243_metric_hard36_guard08` 排名第一，`accepted_as_next_candidate=True`。
  - `v243_metric_hard24_guard04`、`v243_metric_hard30_guard06_anchor04` 也通过 validation checks。
- test 审查结论：
  - hard36 是 validation-selected，但 test 上 observe_later_like / strong_steer 有 bucket 退化。
  - hard24 是 test 稳定性最均衡的候选，四个核心 bucket 的 test mean tail delta 都优于 v241。
- 已新增候选级稳定性表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626\tables\v243_candidate_test_robustness_summary.csv`

## 下一步候选任务

- 首选：执行 `v244_locked_audit_compare_v243_hard36_vs_hard24`。
  - 固定 v243 已产生的预测，不重新训练。
  - 并列审查 validation-selected `v243_metric_hard36_guard08` 和 conservative/test-robust `v243_metric_hard24_guard04`。
  - 输出 per-delay、per-bucket、per-sample、worst regression casebook，并明确是否继续用 hard36、改用 hard24，或回退 v241。
- 如果继续训练：不要再直接扩大权重；先根据 v244 audit 决定是否需要 revised validation score 或更细的 snapshot selection。

## 禁止任务

- 不做 gate/router/selector。
- 不删除 observe_later_like 或预测差样本。
- 不做 response-type hard routing。
- 不基于 test 反调 v243 权重或选择规则。
- 不改变 formal headline；v243 目前只是进入 audit 的训练候选。

---
# 当前任务队列更新：2026-06-29 v244 hard36 vs hard24 locked audit 已完成（最新）

## 正在做的任务

- 当前没有正在运行的训练进程。
- v244 已完成 hard36 vs hard24 的 locked aggregate audit、报告、图、ZIP 和 guardrail 校验。

## 已完成任务

- 已新增并运行 `stage03_v244_locked_audit_compare_v243_hard36_vs_hard24_20260629.py`。
- 已生成 v244 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v244_locked_audit_compare_v243_hard36_vs_hard24_20260629`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v244_locked_audit_compare_v243_hard36_vs_hard24_20260629\v244_locked_audit_compare_v243_hard36_vs_hard24_pack.zip`
- 已确认 v243 artifact 覆盖：
  - hard36 有完整 best prediction 和逐样本 delta。
  - hard24 没有完整曲线预测、checkpoint、逐样本 delta。
- 已完成候选比较：
  - hard36 是 validation-selected：score `0.865386`，best_epoch `34`。
  - hard24 是 locked test 更稳：all/observe/strong 上优于 hard36。
  - hard36 hard bucket 退化明显：observe/strong 合计 `11/12` 个 delay tail delta 为正。
  - hard24 hard bucket 更稳：observe/strong 合计 `2/12` 个 delay tail delta 为正。

## 下一步候选任务

- 如果继续推进 v243 路线：只做 artifact-replay，不做调参。
  - 目标：重放 v243 并保存 hard24/hard30/hard36 的完整 predictions、checkpoints、per-sample delta。
  - 禁止：改 hard weight、改 guard loss、改 validation rule、根据 test 反调选择。
- 如果不重放：当前应保留 v241 作为默认候选，把 v244 结论作为“v243 尚不能 formal 替代”的审计证据。

## 禁止任务

- 不把 hard36 直接升级为 formal replacement。
- 不把 hard24 因 test aggregate 更好而直接升级为 formal replacement。
- 不做 gate/router/selector。
- 不删除 observe_later_like 或预测差样本。
- 不基于 test 反调 v243 参数或选择规则。

---
# 当前任务队列更新：2026-06-30 v245 差样本锚点后移效果审查已完成（最新）

## 正在做的任务

- 当前没有正在运行的训练进程。
- v245 已完成差样本后移锚点效果审查；本轮只做诊断，不训练、不调参。

## 已完成任务

- 已新增并运行 `stage03_v245_bad_sample_anchor_shift_effect_audit_20260630.py`。
- 已生成 v245 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v245_bad_sample_anchor_shift_effect_audit_20260630`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v245_bad_sample_anchor_shift_effect_audit_20260630\v245_bad_sample_anchor_shift_effect_audit_pack.zip`
- 已完成两个口径的后移比较：
  - remaining-task tail RMSE；
  - 同一原始时间重叠段 absolute steering RMSE。
- 已确认两个口径在本轮 tail 段上数值一致，说明改善不是因为预测段缩短。
- 核心结果：
  - v241 bad_top10 固定 `+400ms`：delta `-0.210`，改善率 `83.1%`。
  - v241 bad_top10 固定 `+600ms`：delta `-0.288`，改善率 `88.7%`。
  - 早锚点 bad_top10 oracle 最佳后移：delta `-0.428`，改善率 `95.8%`。

## 下一步候选任务

- 首选：做 v246 风险样本延后观察 / 重锚定训练任务构造。
  - 普通样本保留当前锚点预测。
  - 高风险/早锚点差样本允许后移观察后再预测。
  - 训练和评估必须分层报告普通样本、风险样本、observe_later_like、strong_steer。
- 在进入训练前，建议先定义 input-only 风险判定规则；不能使用未来真实曲线或 test 后验误差作为部署决策。

## 禁止任务

- 不统一后移全部样本作为 formal 方案。
- 不删除差样本。
- 不基于 test 反调模型参数或选择规则。
- 不把 oracle 最佳后移当成可部署策略。
- hard24 仍缺少完整 prediction/checkpoint/per-sample delta，不能做同级逐样本审查。

---

# 当前任务队列更新：2026-06-30 v246 oracle 最佳锚点遍历与 selector 审查已完成（最新）

## 正在做的任务

- 当前没有正在运行的训练进程。
- v246 已完成 oracle 最佳锚点遍历、input-only selector、固定等待策略对照、报告、图和 ZIP 打包。

## 已完成任务

- 已新增并运行 `stage03_v246_oracle_best_anchor_and_selector_audit_20260630.py`。
- 已生成 v246 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v246_oracle_best_anchor_and_selector_audit_20260630`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v246_oracle_best_anchor_and_selector_audit_20260630\v246_oracle_best_anchor_and_selector_audit_pack.zip`
- 已完成三类对照：
  - oracle 最佳锚点：用真实误差选择每个样本最优后移锚点，只作为理论上限。
  - input-only selector：只看 base 锚点前可见输入和候选等待时长，不看未来真实曲线、人工响应标签、event_uid 或 recording。
  - fixed waiting policy：显式 `policy_wait_to_latest_anchor`，用于判断收益是否只是来自统一多观察。
- 核心结果：
  - test bad_top10 oracle：RMSE `1.008 -> 0.656`，mean delta `-0.352`，改善率 `84.7%`。
  - early bad_top10 oracle：RMSE `1.021 -> 0.591`，mean delta `-0.431`，改善率 `95.8%`。
  - RF selector bad_top10：RMSE `1.008 -> 0.908`，mean delta `-0.100`，改善率 `29.7%`。
  - 固定等到最晚锚点 bad_top10：RMSE `1.008 -> 0.685`，mean delta `-0.322`；early bad_top10 mean delta `-0.391`。

## 下一步候选任务

- 首选：做 v247 “带等待代价/触发条件的重锚定任务构造”。
  - 保留普通样本当前锚点，避免牺牲正常样本。
  - 对风险样本允许延后观察，但要设置等待代价，不能无脑全部等到最晚。
  - 使用 input-only 风险触发规则，不能使用未来真实曲线、测试后验误差或 oracle label 作为部署决策。
  - 分层报告 normal、bad_top10、early_bad_top10、observe_later_like、strong_steer，并明确平均误差、尾部误差、改善率和等待时长分布。
- 如果要训练新模型，训练目标应从“固定原锚点预测”扩展为“可等待决策 + 后移锚点预测”的联合任务；但第一版应先做简单可解释策略，不直接上复杂 router。

## 禁止任务

- 不把 oracle 最佳锚点当成可部署策略。
- 不用 test 后验误差决定是否后移。
- 不统一后移全部样本作为 formal 方案，除非同时报告等待代价和普通样本损伤。
- 不删除差样本。
- 不回到 v222a gate / 删除样本 / 轻量 residual 路线。
- 不把 Ridge selector 的好结果误解成已经学会逐样本最佳锚点；它当前与固定等到最晚锚点策略完全一致。

---
# 最新更新：2026-06-30 v247 50ms 多分辨率 best anchor discovery 已完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- v247 已完成：50ms fine-grid anchor 重采样、锁定 v241 推理、离线 best anchor label、input-only selector、图表、中文报告和 ZIP 打包。

## 已完成任务

- 已新增并运行 `stage03_v247_multi_resolution_best_anchor_discovery_20260630.py`。
- 已生成 v247 输出目录和 ZIP：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v247_multi_resolution_best_anchor_discovery_20260630`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v247_multi_resolution_best_anchor_discovery_20260630_pack.zip`
- 已完成 50ms fine-grid 支持审计：`1167` 个事件、`24507` 个候选锚点、`21` 个 delay、`dropped=0`。
- 已完成 locked v241 inference replay：coarse delay 对齐旧 v241 预测，mean RMSE `0.000000`，max `0.000001`。
- 已输出核心表：
  - `tables\v247_fine_anchor_candidate_table.csv`
  - `tables\v247_best_anchor_by_event.csv`
  - `tables\v247_score_weight_sweep_summary.csv`
  - `tables\v247_selector_training_table.csv`
  - `tables\v247_selector_predictions_by_candidate.csv`
  - `tables\v247_selector_selected_anchor_by_event.csv`
  - `tables\v247_selector_policy_summary.csv`
  - `tables\v247_signal_anchor_diagnostics.csv`
- 已输出核心图：
  - `figures\v247_best_anchor_distribution_by_group.png`
  - `figures\v247_selector_vs_oracle_error.png`
  - `figures\v247_selected_delay_distribution.png`
  - `figures\v247_error_delay_score_curves_examples.png`
  - `figures\v247_signal_anchor_alignment.png`

## 结果判断

- v247 的 50ms best anchor label 有价值：test/all 当前 0ms 平均 RMSE `0.475`，oracle best `0.253`；test/bad_top10 当前 `1.198`，oracle best `0.616`。
- 但当前 input-only selector 仍不够强：RF selector 在 test/bad_top10 上为 `0.947`，虽然优于当前 0ms，但弱于固定 `policy_wait_to_latest_anchor` 的 `0.695`。
- 因此当前结论不是“selector 已可部署”，而是“best anchor 任务定义成立，但 selector 可学习性不足，需要下一轮改进”。

## 下一步候选任务

- 首选：设计 v248 anchor-aware selector / joint anchor-trajectory model。
  - 输入应比当前 RF selector 更适合判断“什么时候信息足够”，例如显式时序编码、候选点间相对变化、候选 anchor 序列级比较。
  - 不能使用未来真实曲线、candidate 真实误差、event_uid、recording 或 test 后验误差作为部署输入。
  - 必须同时对比 `policy_keep_0ms_anchor`、`policy_wait_to_latest_anchor` 和 oracle upper bound。
- 如果先做诊断：优先看 `figures\v247_error_delay_score_curves_examples.png` 和 `tables\v247_signal_anchor_diagnostics.csv`，判断 best delay 与局部信号变化的错位原因。

## 禁止任务

- 不把 oracle best anchor 当成可部署策略。
- 不用 test 后验误差决定锚点。
- 不统一把所有样本后移到 1000ms。
- 不删除差样本。
- 不回到 v222a gate / 删除样本 / 轻量 residual 路线。
- 不把当前 RF/Ridge selector 解释成已经学会逐样本最佳锚点；它目前仍弱于 wait-latest。

---

# 最新更新：2026-07-02 生理深挖 v257-v259 已完成

## 正在做的任务

- 当前没有正在运行的训练进程。
- 生理数据 goal 仍处于未达成状态：v254b-v259 已经覆盖多种合理使用方式，但没有得到“极大弥补锚点前信息不足、差样本本质改善”的结果。

## 已完成任务

- v257：同驾驶员 subject-aware 个体化记忆检索。
  - 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v257_subject_personalized_physio_memory_20260702.py`
  - 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v257_subject_personalized_physio_memory_20260702`
  - 结论：validation 选择 vehicle-only 记忆；test bad_top10 从 v250 `0.8383` 变为 `1.3054`，明显变差。
- v258：生理增强 anchor selector。
  - 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v258_physio_augmented_anchor_selector_20260702.py`
  - 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v258_physio_augmented_anchor_selector_20260702`
  - 结论：fixed wait-latest 对 bad_top10 有大收益 `1.1977 -> 0.6950`，但 vehicle+physio selector `0.9342` 没有超过 vehicle-only selector `0.9300`。
- v259：raw 生理 cross-attention 直接轨迹预测。
  - 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v259_physio_cross_attention_prediction_20260702.py`
  - 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v259_physio_cross_attention_prediction_20260702`
  - 结论：subject-disjoint bad_top10 上 v250 `0.8783`，v259 vehicle-only `0.9267`，vehicle+physio cross-attention `1.0889`，badweighted `1.0351`，生理仍然拖累预测。

## 下一步候选任务

- 若继续坚持“生理数据主线”，优先不要再换一个融合模型盲试，而应先做数据层/定义层决策：
  - 检查能否从原始生理重新计算更可靠的 HRV、RESP phase、EDA/SCR 事件型指标，排除当前 valid_ratio/subject/recording 强混杂。
  - 明确是否接受 subject-aware 个体化校准作为正式任务边界；如果必须 subject-disjoint，当前生理证据不足。
  - 若需要 GPTPro 复核，应先由用户确认 Chrome 当前确实处于 Pro/进阶模式，再发送已准备的 prompt。
- 若目标仍是行为预测效果提升，建议把主线回到车辆/任务构造：
  - 强车辆时序 backbone；
  - 概率/多模态轨迹分布；
  - 等待代价明确的 anchor-aware 预测任务。

## 禁止任务

- 不继续做生理简单拼接、浅层 CNN/MLP、手工权重扫描或同类 cross-attention 盲试。
- 不把 subject-aware 诊断信号写成 subject-disjoint 正式结论。
- 不用 test 后验误差做部署决策。
- 不删除差样本。
- 不回到 v222a gate / 删除样本 / 轻量 residual 路线。

---
# 当前任务队列

> 最新指针：2026-07-02 已完成 v289 RESP source phase route audit。当前没有正在运行的训练进程。v289 从 cleaned 200Hz RESP 源信号重建呼吸周期、相位、幅值、质量和因果同步偏移特征，共 `575` 个 RESP source 特征、`27` 个 feature set，复用 v278 vehicle top40 route gate。结果 `route_viable_now=false`：deployable top1 在 test bad_top10 上仍比 latest 差 `+0.1553`，在 test bad_top10_vehicle_ambiguous 上差 `+0.1251`；test-best top1 也仍差 `+0.0625`，best corr `0.0463`。RESP 源信号比 ECG 更接近 latest，但仍不能转成可部署候选选择。用户 goal 仍未达成。
---

# 最新更新：2026-07-02 v289 RESP source phase route audit

## 正在做的任务

- 生理数据 goal 仍未达成。
- 已确认：从 RESP 源信号重建呼吸周期/相位/幅值后，仍没有得到可部署 top1 改善。
- 当前 blocker 进一步收敛：不是 ECG 源信号没处理，也不是 RESP 相位/周期没重建；现有源信号只能提供弱 headroom，不能稳定完成车辆相似候选消歧。

## 已完成任务

- v289 `resp_source_phase_route_audit`：
  - 脚本：`03_baselines/scripts/stage03_v289_resp_source_phase_route_audit_20260702.py`
  - 输出：`03_baselines/v289_resp_source_phase_route_audit_20260702`
  - 报告：`03_baselines/v289_resp_source_phase_route_audit_20260702/reports/v289_resp_source_phase_route_audit_cn.md`
  - ZIP：`03_baselines/v289_resp_source_phase_route_audit_20260702_pack.zip`
- 关键结果：
  - `guardrail.pass=True`，`zip_testzip=True`。
  - `event_n=1167`，`candidate_rows=46680`。
  - `resp_source_feature_n=575`，`feature_set_n=27`。
  - `uses_post_observation_any=false`。
  - `ok_rate=0.91945`。
  - `baseline_valid_ratio_median=1.0`。
  - `context_period_s_median=3.0263`，`context_bpm_median=19.8259`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1553`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta vs latest `+0.1251`。
  - test-best top1 diagnostic 未通过：最佳 delta `+0.0625`。
  - test bad_top10 best corr `0.0463`。

## 下一步候选任务

- 不建议继续围绕 RESP 同类 phase/cycle feature set 做 gate 或 threshold 微调。
- 如果继续生理方向，最后一个源信号侧候选是 EDA/SCR 可用记录子集和重新清洗，但必须先把 near-constant / missing recording 与 subject-disjoint 泛化问题分开。
- 如果目标优先是预测效果，应转回车辆多未来候选、不确定性、等待代价建模；当前生理证据只能作为弱辅助诊断。

## 禁止任务

- 不把 `resp_window_dur3_endm1_top24` 的 test-best delta `+0.0625` 写成模型改善。
- 不把 bad_top10_vehicle_ambiguous 的 top3 oracle delta `-0.0268` 写成可部署策略。
- 不在 route gate 未通过时直接训练复杂 RESP/vehicle fusion 轨迹模型。
---

# 最新更新：2026-07-02 v288 ECG source-signal route audit

## 正在做的任务

- 生理数据 goal 仍未达成。
- 已确认：从 ECG 源信号层重新做 R 峰/RR、质量和因果同步偏移后，仍没有得到可部署 top1 改善。
- 当前 blocker 更明确：不是 ECG 文件不可用，也不是简单短窗错位；而是 ECG 对“车辆相似但未来分叉”的候选排序信号太弱。

## 已完成任务

- v288 `ecg_source_signal_route_audit`：
  - 脚本：`03_baselines/scripts/stage03_v288_ecg_source_signal_route_audit_20260702.py`
  - 输出：`03_baselines/v288_ecg_source_signal_route_audit_20260702`
  - 报告：`03_baselines/v288_ecg_source_signal_route_audit_20260702/reports/v288_ecg_source_signal_route_audit_cn.md`
  - ZIP：`03_baselines/v288_ecg_source_signal_route_audit_20260702_pack.zip`
- 关键结果：
  - `guardrail.pass=True`，`zip_testzip=True`。
  - `event_n=1167`，`candidate_rows=46680`。
  - `ecg_source_feature_n=518`，`feature_set_n=27`。
  - `uses_post_observation_any=false`。
  - `ok_rate=0.91945`。
  - `baseline_valid_ratio_median=1.0`，`dur2_end0_valid_ratio_median=1.0`。
  - deployable top1 bad_top10 未通过：test delta vs latest `+0.1556`。
  - deployable top1 bad_top10_vehicle_ambiguous 未通过：test delta vs latest `+0.1510`。
  - test-best top1 diagnostic 未通过：最佳 delta `+0.0903`。
  - test bad_top10 best corr `0.0620`。

## 下一步候选任务

- 不建议继续围绕 ECG 同类特征做 feature set / gate / threshold 微调。
- 若仍坚持生理源信号方向，剩余合理检查只应是明显不同的源信号修复：
  - RESP 相位/呼吸周期重建，而不是复用 v285 的粗零交叉特征。
  - EDA/SCR 可用记录子集和重新清洗，先排除 near-constant 记录。
  - 原始 1000Hz 到 cleaned 200Hz 的处理链抽样核查，确认是否存在滤波/同步造成的生理事件被抹平。
- 若目标是尽快改善预测效果，下一步优先回到车辆多未来/不确定性/等待代价主线；生理目前只能保留为边界证据或弱诊断，不应作为主增量直接拼进轨迹模型。

## 禁止任务

- 不把 v288 的 test-best corr `0.0620` 或 bad ambiguous corr `0.1011` 写成可部署改善。
- 不把 top3 oracle 的局部改善写成模型性能。
- 不继续旧 bio selector / reranker / reliability filter 阈值微调。
- 不在 route gate 未通过时直接训练更复杂 vehicle+physio/ECG fusion 轨迹模型。
---
# 最新更新：2026-07-03 v302 侧倾诱因输入审计

## 已完成任务

- v302 `roll_cause_input_audit`：
  - 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v302_roll_cause_input_audit_20260703.py`
  - 输出：`05_rebuild_from_raw_20260511/03_baselines/v302_roll_cause_input_audit_20260703`
  - 报告：`05_rebuild_from_raw_20260511/03_baselines/v302_roll_cause_input_audit_20260703/reports/v302_roll_cause_input_audit_cn.md`
  - ZIP：`05_rebuild_from_raw_20260511/03_baselines/v302_roll_cause_input_audit_20260703/v302_roll_cause_input_audit_20260703.zip`
- 结果摘要：
  - 当前 v236 输入已经包含大量侧倾诱因：`609` 个 preinput 特征中，侧倾/横摆/转向/道路相关列 `392` 个。
  - 显式 roll-cause summary 对事件类型识别有效：`base_all_v236_preinput` test macro-F1 `0.2284`，`base_plus_engineered_roll_cause` test macro-F1 `0.3906`。
  - 对差样本识别有弱改善：within_bad_top10 test AUC 从 base `0.5735` 到 engineered summary `0.6354`。
  - 对轨迹残差修正：test/all 可小幅改善约 `-0.00884 RMSE`，但 test/within_bad_top10 没有改善，非零修正反而轻微变差。

## 当前建议队列

- 首选：把 roll-cause summary 作为事件类型/响应类型辅助监督分支，而不是简单拼接进残差回归。
- 第二步：训练一个 bad-focused / no-harm 的多任务模型，只允许在 validation bad_top10 不变差时采用 roll-cause 分支输出。
- 第三步：人工复核 v301/v302 中高误差复合事件，确认 roll-cause summary 对应的事件类型是否符合驾驶语义。

## 禁止任务

- 不把 v301 未来事件标签直接作为部署输入。
- 不把 test 后验误差作为输入或选择规则。
- 不把 v302 的 test/all 小幅改善写成“差样本本质改善”；bad_top10 尚未改善。
- 不回到 v222a gate / 删除样本 / 轻量 residual 旧线。

---
# 最新更新：2026-07-03 v301 事件类型多分类标签审计

## 已完成任务

- v301 `event_type_multiclass_label_audit`：
  - 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v301_event_type_multiclass_label_audit_20260703.py`
  - 输出：`05_rebuild_from_raw_20260511/03_baselines/v301_event_type_multiclass_label_audit_20260703`
  - 报告：`05_rebuild_from_raw_20260511/03_baselines/v301_event_type_multiclass_label_audit_20260703/reports/v301_event_type_multiclass_label_audit_cn.md`
  - ZIP：`05_rebuild_from_raw_20260511/03_baselines/v301_event_type_multiclass_label_audit_20260703/v301_event_type_multiclass_label_audit_20260703.zip`
- 结果摘要：
  - `guardrail.pass=True`，`event_n=1167`，`event_type_n=9`。
  - 自动标签能把高误差样本分层出来：test 中 `复合急制动转向`、`急左转`、`急右转`误差最高。
  - 但锚点前输入对事件类型的预测能力较弱：选中分类器 `extra_trees_d6`，test `macro_f1=0.228`，`balanced_accuracy=0.349`。
  - 标签残差修正不是有效主线：test/all 仅改善约 `0.0011 RMSE`，test/within_bad_top10 反而变差。

## 当前建议队列

- 首选：人工复核 `tables/v301_manual_review_pack.csv` 中的高优先级样本，先确认“紧急连续变道/急停/急转弯/复合事件”等标签是否符合人工语义。
- 第二步：把人工确认后的事件标签作为辅助监督或样本分层训练信号，做多任务模型或混合专家审计。
- 第三步：如果事件标签仍难从锚点前预测，不要继续堆分类器，应转向不确定性/多未来轨迹建模，并显式承认同一锚点前输入可能对应多种后续行为。

## 禁止任务

- 不把 v301 的未来行为自动标签直接作为部署输入。
- 不把 test 后验误差或真实未来轨迹派生信息用于正式预测输入。
- 不回到 v222a gate / 删除样本 / 轻量 residual 旧线。
- 不把 `macro_f1=0.228` 的事件分类器解释成已经能可靠识别事件类型。

---

# 最新更新：2026-07-04 v306/v307 coarse scene-label conditioned 路线

## 已完成任务

- v306 `coarse_predefined_scene_label_table`：
  - 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v306_coarse_predefined_scene_label_table_20260704.py`
  - 输出：`05_rebuild_from_raw_20260511/03_baselines/v306_coarse_predefined_scene_label_table_20260704`
  - 报告：`05_rebuild_from_raw_20260511/03_baselines/v306_coarse_predefined_scene_label_table_20260704/reports/v306_coarse_predefined_scene_label_table_cn.md`
  - ZIP：`05_rebuild_from_raw_20260511/03_baselines/v306_coarse_predefined_scene_label_table_20260704/v306_coarse_predefined_scene_label_table_20260704.zip`
- v307 `coarse_scene_label_conditioned_curve_model`：
  - 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v307_coarse_scene_label_conditioned_curve_model_20260704.py`
  - 输出：`05_rebuild_from_raw_20260511/03_baselines/v307_coarse_scene_label_conditioned_curve_model_20260704`
  - 报告：`05_rebuild_from_raw_20260511/03_baselines/v307_coarse_scene_label_conditioned_curve_model_20260704/reports/v307_coarse_scene_label_conditioned_curve_model_cn.md`
  - ZIP：`05_rebuild_from_raw_20260511/03_baselines/v307_coarse_scene_label_conditioned_curve_model_20260704/v307_coarse_scene_label_conditioned_curve_model_20260704.zip`
- 结果摘要：
  - v306 已生成 `v306_coarse_scene_event_labels.csv`，共 `1167` 个事件、`5` 类粗场景。
  - 标签分布：下坡过弯 `277`，平路过弯 `142`，连续变道/连续左右修正 `414`，紧急变道/猛打方向失稳 `115`，其他/不确定 `219`。
  - v307 validation-only 选中 `v307_coarse_scene_init_aux003_film005_h64`。
  - test/all：v300 `0.519805` -> v307 `0.496138`。
  - test/within_bad_top10：v300 `0.859987` -> v307 `0.777797`。
  - test/within_bad_top20：v300 `0.690942` -> v307 `0.639121`。
  - 对比 v304：v307 在 test/all、within_bad_top10、within_bad_top20、strong_steer、vehicle_ambiguous、normal_predictable 上均略好或明显更好。
- 关键边界：
  - 过弯两类来自当前 manifest `scene_type`，更接近预测前可知实验场景。
  - 直道内连续/紧急子类仍部分来自 v305/v301 自动 seed，不能直接写成最终人工标签。
  - 当前 v307 是 coarse-scene seed 条件模型，不是最终部署结论。

## 当前建议队列

- 首选：人工复核 v306 high priority 的 `529` 个连续/紧急直道子类，确认 `continuous_lane_change` 与 `emergency_lane_change_instability` 是否符合实验语义。
- 第二步：确认后把 `coarse_scene_manual_review_status` 标为 `confirmed`，重跑 v307 或做 v308 confirmed-coarse-scene 模型。
- 第三步：如果复核发现粗类仍混杂，再只在粗类内部加少量二级标签，不回到 v305 的 7 类细 future-derived 标签。

## 禁止任务

- 不把 v306 直道内连续/紧急 seed 当成最终人工标签。
- 不把 `other_or_uncertain` 强行解释成某个明确事件。
- 不把 v307 写成已经完全可部署；非弯道子类仍需人工或实验条件确认。
- 不回到 v222a gate / 删除样本 / 轻量 residual 旧线。

---

# 最新更新：2026-07-04 v305 formal predefined event label table

## 已完成任务

- v305 `formal_predefined_event_label_table`：
  - 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v305_formal_predefined_event_label_table_20260704.py`
  - 输出：`05_rebuild_from_raw_20260511/03_baselines/v305_formal_predefined_event_label_table_20260704`
  - 报告：`05_rebuild_from_raw_20260511/03_baselines/v305_formal_predefined_event_label_table_20260704/reports/v305_formal_predefined_event_label_table_cn.md`
  - ZIP：`05_rebuild_from_raw_20260511/03_baselines/v305_formal_predefined_event_label_table_20260704/v305_formal_predefined_event_label_table_20260704.zip`
- 结果摘要：
  - 已生成正式事件标签 seed 表 `v305_formal_event_labels.csv`。
  - 已生成可人工复核的 `v305_manual_review_seed_pack.csv`。
  - 主事件类型收敛为 7 类，`event_n=1167`。
  - 主标签分布：普通/轻微/不确定 `697`，连续变道/横向避让 `175`，急停/强减速 `80`，复合制动转向 `59`，急左转 `56`，紧急避让/连续变道 `54`，急右转 `46`。
  - high priority 人工审核 `869`，medium priority `161`。
- 关键边界：
  - `formal_primary_type` 在人工/实验条件确认后可以作为模型输入。
  - `formal_secondary_tags` 默认不作为直接输入，尤其是晚响应、多段修正这类未来过程形状。
  - 当前 seed 仍来自 v301 future behavior auto draft，因此不能直接写成最终人工标签。

## 当前建议队列

- 首选：人工审核 `v305_manual_review_seed_pack.csv`，先处理 high priority 中 v300 RMSE 高、原标签为多段修正/晚响应、自动置信低的样本。
- 第二步：将审核后的 `formal_primary_type` 标记为 `manual_review_status=confirmed`。
- 第三步：用确认后的 v305 formal label 表替换 v304 标签输入，重跑 fixed event-label conditioned 模型。

## 禁止任务

- 不把 `formal_secondary_tags` 直接作为模型输入。
- 不把 v301 future-derived seed 当成最终人工标签。
- 不把当前 v305 写成已经完成人工审核。
- 不回到 v222a gate / 删除样本 / 轻量 residual 旧线。

---

# 最新更新：2026-07-03 v304 fixed event-label conditioned 曲线模型

## 已完成任务

- v304 `fixed_event_label_conditioned_curve_model`：
  - 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v304_fixed_event_label_conditioned_curve_model_20260703.py`
  - 输出：`05_rebuild_from_raw_20260511/03_baselines/v304_fixed_event_label_conditioned_curve_model_20260703`
  - 报告：`05_rebuild_from_raw_20260511/03_baselines/v304_fixed_event_label_conditioned_curve_model_20260703/reports/v304_fixed_event_label_conditioned_curve_model_cn.md`
  - ZIP：`05_rebuild_from_raw_20260511/03_baselines/v304_fixed_event_label_conditioned_curve_model_20260703/v304_fixed_event_label_conditioned_curve_model_20260703.zip`
- 结果摘要：
  - 输出目标不变：21 点 `steering_delta` 曲线。
  - 新结构：v303 roll-cause 分支 + 固定事件标签 `event embedding` 条件输入。
  - validation-only 选中 `v304_fixed_event_init_aux005_film010_h64`。
  - test/all 从 v300 `0.519805` 降到 v304 `0.498102`。
  - test/within_bad_top10 从 v300 `0.859987` 降到 v304 `0.832204`。
  - test/within_bad_top20 从 v300 `0.690942` 降到 v304 `0.657669`。
  - 相比 v303，v304 在 all、bad_top10、bad_top20 也继续改善。
- 重要边界：
  - 当前 event label 来自 v301 `future_behavior_auto_draft`。
  - 因此 v304 当前是 known-label/oracle upper-bound，不是无条件部署模型。
  - 若后续能人工审核或由实验条件在预测前确定事件类型，则 v304 结构可转为正式条件输入模型。

## 当前建议队列

- 首选：整理一套人工/外部可知事件标签体系，区分“预测前可知标签”和“未来轨迹派生标签”。
- 第二步：用人工确认标签替换 v301 自动草稿标签，重跑 v304，检查收益是否保留。
- 第三步：如果人工标签收益稳定，把 v304 扩展为 mixture-of-experts：事件标签负责路由，专家负责不同事件类型的轨迹曲线。

## 禁止任务

- 不把当前 v304 写成无条件可部署结果。
- 不把 v301 future-derived 自动标签混同为预测前真实可知标签。
- 不使用 test 误差选择候选模型。
- 不回到 v222a gate / 删除样本 / 轻量 residual 旧线。

---

# 最新更新：2026-07-03 v303 roll-cause 辅助监督多任务曲线模型

## 已完成任务

- v303 `roll_aux_multitask_curve_model`：
  - 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v303_roll_aux_multitask_curve_model_20260703.py`
  - 输出：`05_rebuild_from_raw_20260511/03_baselines/v303_roll_aux_multitask_curve_model_20260703`
  - 报告：`05_rebuild_from_raw_20260511/03_baselines/v303_roll_aux_multitask_curve_model_20260703/reports/v303_roll_aux_multitask_curve_model_cn.md`
  - ZIP：`05_rebuild_from_raw_20260511/03_baselines/v303_roll_aux_multitask_curve_model_20260703/v303_roll_aux_multitask_curve_model_20260703.zip`
- 结果摘要：
  - 输出目标不变：仍为 21 点 `steering_delta` 曲线。
  - 模型结构已改：v300 joint-curve backbone + roll-cause encoder + event auxiliary head + FiLM 调制。
  - v301 事件类型标签只作为训练辅助监督，不作为推理输入。
  - validation-only 选中 `v303_roll_init_aux003_film005_h64`，通过 v303 no-harm gate。
  - test/all 从 `0.519805` 降到 `0.513617`。
  - test/within_bad_top10 从 `0.859987` 降到 `0.843876`。
  - test/within_bad_top20 从 `0.690942` 降到 `0.669646`。
  - 事件辅助头 test delay0 macro-F1 `0.416327`，说明 roll-cause 分支学到了一部分事件结构。

## 当前建议队列

- 首选：接受 v303 作为“结构改动后的小正向基线”，但不要写成根本突破。
- 第二步：在 v303 初始化策略上继续做 mixture-of-experts 或多模态轨迹分布，让模型承认同一锚点前输入可能对应多种后续行为。
- 第三步：保留 validation no-harm gate，尤其继续分开看 all、within_bad_top10、within_bad_top20、strong_steer、vehicle_ambiguous。

## 禁止任务

- 不把 v301 未来事件标签当成部署输入。
- 不使用 test 误差选择候选模型。
- 不把 v303 的小幅 bad_top10 改善夸大成差样本已经被本质解决。
- 不回到 v222a gate / 删除样本 / 轻量 residual 旧线。

---
# 最新任务队列指针：2026-07-04 第317版二阶段候选门控校正实验已完成但验证失败。下一步优先级改为第318版保守门控修正：1）保留第317版候选库，因为验证集候选最优上限明显优于第316版；2）不要直接扩大候选库或报告测试集；3）先修门控选择机制，要求普通样本默认原预测不改，只有高置信、高风险样本才允许校正；4）增加候选选择的验证阈值或两段式门控，先判定是否需要校正，再选择校正候选；5）继续只用验证集选阈值和模型，测试集仍不得参与。第317版失败类型：候选库有上限，门控选不准，普通样本被过度修改。
---

# 最新任务队列指针：2026-07-04 本地高级模型第317版修正方案咨询已完成。下一步优先级更新为：1）实现第317版轻量二阶段候选校正器；2）使用第315版保留清单和第316版基础预测，构建幅值缩放、时间平移、残差原型候选库；3）训练只用锚点前车辆信号和第316版预测摘要的门控校正器；4）用固定验证门槛判断是否进入测试集。禁止事项：不继续扩大过滤样本作为主线，不用测试集选候选或阈值，不把锚点后真实曲线生成的标签作为部署输入，不把本地高级模型建议直接写成实验结论。
---
# 最新任务队列指针：2026-07-05 下一步执行第318版保守两段式候选门控校正实验。承接第317版验证失败和本地高级模型复询结果，优先新建第318版脚本，复用第316版原预测与第317版候选库，不扩候选、不重训主模型。固定任务顺序：1）训练集内部构造可校正标签和候选收益矩阵；2）训练“是否值得校正”的第一段门控；3）训练候选收益与大退化风险的第二段选择器；4）用训练集内部交叉验证搜索门槛和融合幅度；5）只在验证集上判定三条方案是否通过；6）未通过则继续询问本地高级模型并修正，不报告测试集。硬约束：普通样本必须以保持第316版原预测为默认，整体校正率和普通样本校正率都要有上限。
---

# 最新任务队列指针：2026-07-05 下一步从第321版困难样本图册出发做抽样排查和门控修正。当前不建议继续把第320版排序配额门控当作最终方向硬推，因为测试困难前20实际平均收益为 `-0.001521`，其中 `34/46` 是候选上限有空间但门控未抓住，`3/46` 被第320版改坏，只有 `2/46` 明显改好。优先任务：1）先看图册中“候选有空间但门控未抓住”和“修正后变坏”的代表样本，确认它们是否都符合“方向盘快速转动引起”的问题定义；2）把下一版目标改成困难样本选择和候选家族风险识别，而不是继续放宽第320版阈值；3）如果继续建模，必须先在验证集上证明能减少“该改没改”和“改错候选”两类错误，再允许报告测试集；4）普通样本仍保持不动，避免回到第317版误伤普通样本的问题。
---
