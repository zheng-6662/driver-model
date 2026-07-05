# 项目决策日志

这里只记录真正改变方向、标准或交付边界的判断。

## 决策表

| 日期 | 决策 | 白话解释 | 触发原因 | 影响范围 | 证据 / 入口 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-04-23 | V5.8 active code line switches to modular editing under `02_code/final_code/model/training/v58_modular/`; the old script remains a thin wrapper entrypoint | 以后继续运行老脚本可以，但新增逻辑不再往那个大文件里堆，而是优先写进对应模块。 | active training code had become a multi-thousand-line monolith with duplicated / unreachable sections, and the user explicitly required a literature-style multi-module structure for future work. | all future code edits on the V5.8 training line; wrapper maintenance, training snapshots, and future refactors should all preserve the modular package as the source of truth | [2026-04-23 日志](daily/2026-04-23.md), [current-state](../../references/current-state.md), [v58_modular README](../../../02_code/final_code/model/training/v58_modular/README.md) | active |
| 2026-04-21 | 本轮主线切换到 `D:/下载/codex_next_steps_plan.md`，旧 `current-state` 降为背景锚点 | 本轮不再把“继续追 220918 复现差距”当作唯一下一步，而是先查清输入管线、任务定义、checkpoint 选择和 spike 位置这些更直接的问题。 | 外部计划指出 lane 与 speed 风险直击当前 active script；本地代码与数据抽样已确认这两个风险真实存在。 | 2026-04-21 全部执行顺序、日志落盘规则、第一批分析任务与后续 D/E 路线 | [2026-04-21 日志](daily/2026-04-21.md), [当前状态 handoff](../../references/current-state.md) | active |
| 2026-04-21 | Batch 1 + Batch 2 已完成，下一优先级正式切到 D 输入分组消融，再切到 E bridge 训练 | A/B/F/G 诊断和 active script 输入修复已经闭环，后续不应再停留在“先补工具”阶段，而要进入按计划锁死的 D/E 全矩阵执行。 | 诊断报告与 smoke 验证都已落盘：lane / speed bug 已修，`input_qc` 写出成功，same-tool recalc 兼容新接口，Pareto 与 spike 诊断也已完成。 | 从本条开始，训练侧默认先跑 D，再跑 E；每个 full run 完成后必须 same-tool recalc 并写 daily / registry。 | [2026-04-21 日志](daily/2026-04-21.md), [当前状态 handoff](../../references/current-state.md), [项目中枢](../project_progress_hub.md) | active |
| 2026-04-20 | 稳定 manual-upsample control 继续作为固定基线 | active-script 全跑版本仍没有替换掉这个旧稳定控制线，因此任何新结论都必须先和它比。 | 2026-04-20 的 old-baseline equivalence 检查没有追回旧稳定控制。 | 全部 D/E 训练与后续答辩表述 | [2026-04-20 日志](daily/2026-04-20.md) | active |
| 2026-04-20 | Run A / Run B 保持双 keeper 解读，Run C / Run D 不提升为默认主线 | Run A 更强在 response-structure，Run B 更强在 overall / tail fit；Run C / D 目前只能作为辅助证据。 | 2026-04-20 全量 run + same-tool recalc 的比较结果。 | keeper 选择、答辩叙事、D/E 对照锚点 | [2026-04-20 日志](daily/2026-04-20.md) | active |

## 记录模板

### YYYY-MM-DD 决策标题

- 决策：
- 白话解释：
- 触发原因：
- 影响范围：
- 证据链接：
- 后续动作：
- 状态：`active` / `superseded` / `obsolete`
## 2026-04-21 Final D Closure

- Decision:
  - D input-ablation is closed.
  - `baseline_fixed_input` replaces old Run B as the maintained-line fit / tail keeper.
  - Run A remains the response-structure keeper.
  - `plus_pedals` stays as a late-peak trade-off branch, not a clean replacement.
  - `plus_lat_dyn`, `plus_road_cond`, and `minus_z` are not promoted.
- Plain-language meaning:
  - The biggest D gain came from repairing lane/speed reading, not from adding more optional input families.
  - Turning off `z` is not justified; `USE_Z=1` remains the default.
- Trigger:
  - All five D groups completed with same-tool recalc.
  - `baseline_fixed_input` gives `rmse_steer=0.5559`, `tail_rmse_steer=0.7171`, `late_peak_recall=0.6496`.
  - `minus_z` gives `rmse_steer=0.5760`, `tail_rmse_steer=0.7385`, so it does not beat the repaired baseline on fit/tail.
- Impact scope:
  - The next priority moves to E bridge training.
  - E should bridge Run A response-structure strength against `baseline_fixed_input` fit/tail strength, not the older pre-fix Run B line.
- Evidence:
  - [2026-04-21 daily](daily/2026-04-21.md)
  - [D comparison table](../input_group_ablation_20260421/input_ablation_comparison_table.csv)
  - [D summary](../input_group_ablation_20260421/input_ablation_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-06-23 v228 Final Paper Artifact Freeze

- Decision:
  - The local ChatGPT Desktop GPTPro response is accepted as valid after switching to a clean ASCII handoff/retry.
  - v227 is accepted as a valid reporting-only closeout.
  - v228 is restricted to final paper artifact freeze: reporting, packaging, and manuscript-readiness only.
  - No model training, new prediction generation, threshold search, gate/router/selector work, formal headline change, or test-based retuning is authorized.
- Plain-language meaning:
  - The earlier "GPTPro channel blocked" state was caused in part by the prompt being unreadable in GPTPro. After fixing the encoding path, GPTPro gave a bounded instruction and Codex completed it.
  - v228 turns the already locked v225/v226/v227 evidence into final paper-facing tables, reports, figures, logs, and ZIP.
- Trigger:
  - User observed mojibake in the local GPTPro prompt and instructed Codex to use the local software.
  - GPTPro replied with a six-item v228 instruction.
- Impact scope:
  - Formal headline remains `loose_main_pool=avg_joint_focus` and `strict_main_pool=peak_floor_090`.
  - v228 is the current latest completed step; older heartbeat/blocked records remain process history but are superseded as current-state diagnosis.
  - Next loop step, if requested, is to report v228 results back to GPTPro and ask for exactly one bounded next action.
- Evidence:
  - `gptpro_reviews/20260623_v228_local_gptpro_response.md`
  - `gptpro_reviews/20260623_v228_local_gptpro_decision.md`
  - `gptpro_reviews/20260623_v228_local_gptpro_action_items.md`
  - `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v228_final_paper_artifact_freeze_20260623.py`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/v228_final_paper_artifact_freeze_pack.zip`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/logs/guardrail_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/logs/consistency_check.json`
- Status:
  - `active`

## 2026-06-23 v230 Failure Case Manual Review Casebook

- Decision:
  - GPTPro accepted v229 and kept model work stopped.
  - v230 is restricted to audit-only failure-case manual review / paper-case evidence packaging.
  - No model training, new prediction generation, tau/threshold tuning, gate/router/selector creation, v222b/v223 run, formal headline change, or test-based retuning is authorized.
- Plain-language meaning:
  - The next useful work is no longer model search. The project now needs human review of representative failure cases and paper-ready limitation evidence.
  - Codex may package cases and copy existing figures, but human judgement fields must stay blank until manually reviewed.
- Trigger:
  - GPTPro reviewed `v229_two_month_lessons_failure_taxonomy_pack.zip` and accepted the diagnosis that the main bottleneck is hard local failure cases and unstable current-window deployable selection, not a broad lack of candidate curves.
- Impact scope:
  - Formal headline remains `loose_main_pool=avg_joint_focus` and `strict_main_pool=peak_floor_090`.
  - v230 selected 46 formal cases, 23 per pool, spanning strong underestimation, extreme peak failure, tail-amplitude failure, reverse / multi-correction, zero-cross boundary, and normal-curve controls.
  - The next step after v230 is manual casebook reading and filling `v230_manual_review_template.csv`, then revising the paper failure-case section.
- Evidence:
  - `gptpro_reviews/20260623_v230_casebook_gptpro_response.md`
  - `gptpro_reviews/20260623_v230_casebook_gptpro_decision.md`
  - `gptpro_reviews/20260623_v230_casebook_gptpro_action_items.md`
  - `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v230_failure_case_manual_review_casebook_20260623.py`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/v230_failure_case_manual_review_casebook_pack.zip`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/logs/guardrail_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/logs/consistency_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/logs/figure_copy_check.json`
- Status:
  - `active`

## 2026-06-22 v227 Reporting-Only Paper / Claim Readiness Fallback

- Decision:
  - v227 is accepted only as a reporting / writing-readiness package built from locked v225 and v226 evidence.
  - v227 is not a new model branch, not a new formal leaderboard, and not permission to start v222b/v223, a new tau, a gate/router, selector, or test-based retuning.
  - The next action remains GPTPro handoff once the channel is reachable: report v226+v227 and ask for one bounded writing/claim/reporting instruction.
- Plain-language meaning:
  - GPTPro could not provide a usable next instruction in the current loop because Desktop returned empty stopped-thinking outputs and Chrome required login.
  - To avoid idle looping while preserving guardrails, Codex packaged the already-verified evidence into paper-facing tables, claim support, limitation notes, figure index, and a next GPTPro prompt.
- Trigger:
  - v226 passed robustness/CI audit and was ready for external review.
  - The GPTPro bridge failed to return a valid new response, so local work was restricted to reporting-only synthesis.
- Impact scope:
  - Formal locked models stay `loose_main_pool=avg_joint_focus` and `strict_main_pool=peak_floor_090`.
  - No new training, threshold search, test retuning, or candidate-pool changes are authorized by v227.
  - v227 artifacts may be used for manuscript/result narrative preparation and for the next GPTPro review prompt.
- Evidence:
  - `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v227_paper_claim_readiness_pack_20260622.py`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/reports/v227_paper_claim_readiness_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/reports/v227_next_gptpro_prompt_ascii.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/logs/no_model_change_guard.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/logs/source_artifact_checks.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/v227_paper_claim_readiness_pack.zip`
  - `gptpro_reviews/20260622_v227_result_gptpro_response_blocked.md`
  - `gptpro_reviews/20260622_v227_result_gptpro_decision_blocked.md`
  - `gptpro_reviews/20260622_v227_result_gptpro_action_items_blocked.md`
- Status:
  - `active`

## 2026-06-23 v227 Heartbeat GPTPro Handoff Still Blocked

- Decision:
  - The heartbeat retry did not obtain a valid GPTPro reply.
  - The project remains at the v226/v227 handoff boundary and must not start a new experiment locally.
  - v227 remains reporting-only, not a new model or formal leaderboard direction.
- Plain-language meaning:
  - The Desktop ChatGPT window still shows the handoff prompt and `已停止思考`, without a bounded six-item reply.
  - The Chrome bridge again refused to send because it could not verify Pro/进阶 mode.
- Trigger:
  - The goal-mode heartbeat attempted to continue the Codex-GPTPro loop on 2026-06-23.
- Impact scope:
  - Keep formal models locked as `loose_main_pool=avg_joint_focus` and `strict_main_pool=peak_floor_090`.
  - Continue to forbid v222b/v223, new tau/threshold search, gate/router/selector expansion, new training, formal headline change, and test-based retuning.
  - The only allowed next action is to retry the GPTPro handoff once the Pro/进阶 channel is reachable, then triage a valid GPTPro reply against local guardrails.
- Evidence:
  - `gptpro_reviews/20260623_v227_heartbeat_gptpro_response_blocked.md`
  - `gptpro_reviews/20260623_v227_heartbeat_gptpro_decision_blocked.md`
  - `gptpro_reviews/20260623_v227_heartbeat_gptpro_action_items_blocked.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/v227_paper_claim_readiness_pack.zip`
- Status:
  - `active`

## 2026-06-23 GPTPro Prompt Encoding Correction

- Decision:
  - The earlier Desktop GPTPro handoff must be treated as invalid / superseded because the user observed the prompt rendered as mojibake in GPTPro.
  - The next GPTPro handoff must use the self-contained ASCII-only prompt `gptpro_reviews/20260623_v227_clean_ascii_handoff_prompt.md`.
  - The v226 and v227 local artifacts remain valid; only the external handoff attempt is invalidated.
- Plain-language meaning:
  - GPTPro likely did not answer because the question it received was unreadable.
  - This changes the diagnosis from "GPTPro failed to answer" to "the sent prompt was not validly readable."
- Evidence:
  - User screenshot of garbled GPTPro prompt.
  - `gptpro_reviews/20260623_prompt_encoding_correction.md`
  - `gptpro_reviews/20260623_v227_clean_ascii_handoff_prompt.md`
- Status:
  - `active`

## 2026-06-23 Goal-Level GPTPro Channel Blocked

- Decision:
  - The active Codex-GPTPro execution-loop goal is blocked at the external GPTPro handoff step.
  - The same blocker repeated across consecutive goal turns: Desktop ChatGPT has no valid bounded reply, and Chrome bridge cannot verify Pro/进阶 mode before sending.
  - No further automatic local experiment is allowed from this state.
- Plain-language meaning:
  - Codex has already done the safe local work: v226 audit, v227 reporting package, ZIP checks, and note synchronization.
  - The next required input is external: restore GPTPro / ChatGPT Pro access or manually provide GPTPro's next bounded instruction.
- Trigger:
  - Third consecutive blocked audit of the same GPTPro channel condition.
- Impact scope:
  - Formal headline remains `loose_main_pool=avg_joint_focus` and `strict_main_pool=peak_floor_090`.
  - Continue to forbid v222b/v223, new tau/threshold search, gate/router/selector expansion, new training, formal headline change, and test-based retuning.
- Evidence:
  - `gptpro_reviews/20260623_goal_blocked_gptpro_channel_response.md`
  - `gptpro_reviews/20260623_goal_blocked_gptpro_channel_decision.md`
  - `gptpro_reviews/20260623_goal_blocked_gptpro_channel_action_items.md`
- Status:
  - `blocked`

## 2026-06-22 v222a No-Harm Gate Closure Boundary

- Decision:
  - `v222a_bounded_residual` and its learned no-harm gate are not promoted as a formal headline.
  - The result must be reported back to GPTPro before any next branch is started.
  - Do not enter v222b/v223, a larger selector, or a multi-candidate router from this state without a new bounded GPTPro instruction and explicit stop condition.
- Plain-language meaning:
  - v222a proved residual correction can reduce underestimation locally.
  - It also proved the learned deployable gate is not stable enough yet: validation passes, but locked test does not preserve both RMSE/tail safety and under-reduction.
- Trigger:
  - GPTPro requested `v222a_gain_harm_decomposition -> oracle safe gate upper bound -> binary validation-only no-harm gate`.
  - Locked test `loose_main_pool`: under reduction stays positive (`0.043478`), but RMSE delta is `+0.010559` and tail delta is `+0.027764`.
  - Locked test `strict_main_pool`: RMSE/tail deltas are safe (`-0.008975`, `-0.005264`), but under reduction is negative (`-0.017241`) and strong-under reduction is negative (`-0.023438`).
  - Oracle safe gate remains much stronger than the learned gate, so the residual has local value but the safe enablement rule is not deployable yet.
- Impact scope:
  - Keep v222a as diagnostic / case-study evidence.
  - Preserve validation-only selection, leakage guards, and the fixed formal candidate pool.
  - Next execution step is GPTPro review of this result, not local escalation.
- Evidence:
  - [current-state](../../references/current-state.md)
  - `05_rebuild_from_raw_20260511/03_baselines/v222a_noharm_gate_diagnostic_20260622/reports/v222a_noharm_gate_diagnostic_report_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v222a_noharm_gate_diagnostic_20260622/tables/test_locked_gate_report.csv`
- Status:
  - `active`

## 2026-06-22 v222a Closeout Candidate Gap Audit Boundary

- Decision:
  - `v222a_bounded_residual` and `v222a_noharm_gate` remain diagnostic-only and are formally stopped as headline candidates.
  - Formal headline is locked as `loose_main_pool=avg_joint_focus` and `strict_main_pool=peak_floor_090`.
  - `v222b_allowed=False` and `v223_allowed=False` in the current `future_route_decision.csv`.
  - The next action is GPTPro review of the closeout pack, not local threshold tuning or a larger selector.
- Plain-language meaning:
  - The closeout audit says the current failure is mostly not "we have no candidate curve at all."
  - The fixed candidate pool often has a better oracle option, especially in high-tail cases, but the learned gate does not reliably choose it.
  - That makes v222b risky right now, because a larger learned gate is likely to overfit the same unstable selector signal.
  - v223 is also not unlocked, because high-tail `candidate_missing_rate` is only about `0.126582` combined, far below the >50% unlock condition.
- Trigger:
  - GPTPro explicitly instructed closeout-only candidate gap audit after the no-harm gate failed locked test.
  - Locked test combined `selector_failed_rate=0.410615`.
  - Locked test combined `candidate_missing_rate=0.027933`.
  - High-tail locked test combined `candidate_missing_rate=0.126582`.
  - High-tail locked test oracle clear gain rate is about `0.911392`, so the allowed candidate pool often still contains a better diagnostic option.
- Impact scope:
  - Do not start `v222a_gate_v2`, new tau selection, multi-router, neural gate, v222b, or v223 from this state.
  - Keep `oracle_safe_gate` and best-candidate oracle as diagnostic-only upper bounds.
  - Use the closeout pack as the evidence package for the next GPTPro question.
- Evidence:
  - [current-state](../../references/current-state.md)
  - `05_rebuild_from_raw_20260511/03_baselines/v222a_closeout_candidate_gap_audit_20260622/reports/v222a_closeout_candidate_gap_audit_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v222a_closeout_candidate_gap_audit_20260622/tables/formal_headline_decision.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v222a_closeout_candidate_gap_audit_20260622/tables/oracle_vs_learned_gap.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v222a_closeout_candidate_gap_audit_20260622/tables/future_route_decision.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v222a_closeout_candidate_gap_audit_20260622/v222a_closeout_candidate_gap_audit_pack.zip`
- Status:
  - `active`

## 2026-06-22 v225 Formal Route Reconstruction Evidence Pack Boundary

- Decision:
  - v225 is a formal evidence-pack closure, not a new model branch.
  - Formal headline stays locked as `loose_main_pool=avg_joint_focus` and `strict_main_pool=peak_floor_090`.
  - Diagnostic-only variants, including `v222a_bounded_residual`, `v222a_noharm_gate`, `oracle_safe_gate`, and residual/oracle variants, remain excluded from formal tables and usage.
  - The next action is GPTPro review of the v225 pack, not local v222b/v223 escalation.
- Plain-language meaning:
  - The project now has a packaged, reproducible evidence surface for the locked formal route.
  - v225 does not claim a new performance breakthrough; it makes the current formal baseline, failure cases, bucket/route-event evidence, and exclusion rules auditable.
  - This helps avoid getting stuck in local selector tuning while preserving the evidence needed for the next bounded decision.
- Trigger:
  - GPTPro accepted the v222a closeout diagnosis and instructed a one-shot formal route reconstruction evidence pack.
  - Locked test reproduction passes within `1e-5`: loose `avg_joint_focus` RMSE/tail `0.544884/0.629752`; strict `peak_floor_090` RMSE/tail `0.571770/0.658306`.
  - ZIP `bad_file=None`; required files are present; metric reproduction, leakage guard, forbidden scan, and table alignment all pass.
- Impact scope:
  - Treat v225 as the current handoff pack for GPTPro.
  - Do not start `v222b`, `v223`, a new tau/threshold search, a gate/router, or test-based retuning unless GPTPro gives a new bounded instruction with a valid stop condition.
  - Keep v222a closeout and diagnostic-only rows as appendix evidence, not as formal headline or deployable output.
- Evidence:
  - [current-state](../../references/current-state.md)
  - `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/reports/v225_formal_route_reconstruction_evidence_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/tables/formal_model_lock.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/tables/formal_reconstruction_metrics_by_pool.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/tables/formal_failure_case_index.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/logs/metric_reproduction_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/logs/leakage_guard_report.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/v225_formal_route_reconstruction_evidence_pack.zip`
- Status:
  - `active`

## 2026-04-22 E Interim Update

- Decision:
  - `bridge_50_50` is now the provisional E fit / tail bridge candidate.
  - Run A remains the response-structure keeper.
  - `baseline_fixed_input` stays as the post-D fit/tail anchor and comparator until E closes.
- Plain-language meaning:
  - A stronger static bridge can beat the repaired D baseline on fit/tail, but it still does not recover the Run A structure axis.
- Trigger:
  - `bridge_55_45` completed first and did not change the keeper split.
  - `bridge_50_50` `best_by_structured` gives `rmse_steer=0.5385`, `tail_rmse_steer=0.6846`, `late_peak_recall=0.6197`.
- Impact scope:
  - The last E run, `bridge_schedule_B_to_A`, should now be judged against Run A on structure and `bridge_50_50` on fit/tail.
  - No thesis claim is promoted yet; E is still in progress.
- Evidence:
  - [2026-04-21 daily](daily/2026-04-21.md)
  - [bridge_50_50 structured recalc](../bridge_training_20260421/bridge_50_50/TRAIN_V5_4_STATECOND_REV_20260422_004147/figures/recalc_best_by_structured_summary.json)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-22 E Final Closure

- Decision:
  - E bridge matrix is closed.
  - Run A remains the response-structure keeper.
  - `baseline_fixed_input` remains the fit / tail keeper after the full E guardrail review.
  - `bridge_50_50` is not promoted even though it posts the best fit/tail headline numbers.
  - `bridge_schedule_B_to_A` is mixed and not promoted.
  - `bridge_55_45` is a no-go.
- Plain-language meaning:
  - Weighting / supervision bridge choices do move the trade-off frontier.
  - They still do not collapse fit/tail and response-structure into one promoted checkpoint that survives the full guardrail set.
- Trigger:
  - All three E groups completed with same-tool recalc.
  - `bridge_50_50` `best_by_structured` gives `rmse_steer=0.5385` and `tail_rmse_steer=0.6846`, but `strong_pos.tail_amp_ratio_pred_over_gt=0.4987` and `strong_pos.tail_flatness_rate=0.7368`.
  - `bridge_schedule_B_to_A` `best_by_structured` lifts `late_peak_recall` to `0.6581` and `first_reversal_time_mae_sec` to `0.4923`, but `rmse_steer=0.5819` and `tail_rmse_steer=0.7749` regress too much.
- Impact scope:
  - The live keeper split remains Run A versus `baseline_fixed_input`.
  - `bridge_50_50` becomes fit/tail frontier evidence rather than a keeper; `bridge_schedule_B_to_A` becomes mixed bridge evidence rather than a keeper.
- Evidence:
  - [2026-04-21 daily](daily/2026-04-21.md)
  - [bridge_50_50 structured recalc](../bridge_training_20260421/bridge_50_50/TRAIN_V5_4_STATECOND_REV_20260422_004147/figures/recalc_best_by_structured_summary.json)
  - [bridge_schedule_B_to_A structured recalc](../bridge_training_20260421/bridge_schedule_B_to_A/TRAIN_V5_4_STATECOND_REV_20260422_010228/figures/recalc_best_by_structured_summary.json)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-22 Effectiveness Mainline Switch

- Decision:
  - The active round switches from bridge / gate / loss exploration to the effectiveness plan.
  - The execution order is locked as D0 absolute-window diagnosis, `H15`, `OPT_A_20`, then conditional `H10`, `OPT_C_BEST`, `CAP_192_BEST`, and winner confirmation as budget allows.
  - `H10` is conditional only, not a default mainline run.
  - `OPT_C_BEST` is conditional regularization, not a fixed mandatory `wd=5e-4` slot unless overfit is observed.
- Plain-language meaning:
  - This round asks whether the current ceiling is mainly horizon length, optimization, or mild capacity.
  - It does not reopen bridge / gate / loss as the main frontier.
- Trigger:
  - The external 2026-04-22 revised plan explicitly reprioritized absolute time windows, `1.5 s` viability, and `2.0 s` optimization rescue over more bridge experiments.
- Impact scope:
  - New training must preserve `fixed_v20260421`, the stable GPU path, protocol-safe split, and the existing live keeper split.
- Evidence:
  - [2026-04-22 daily](daily/2026-04-22.md)
  - [effectiveness summary](../effectiveness_followup_20260422/effectiveness_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-22 Effectiveness Round Closure

- Decision:
  - `baseline_fixed_input` remains the current best non-collapse `2.0 s` fit / tail base.
  - Run A remains the response-structure anchor.
  - `H15` is not eligible for `OPT_A_H15 / OPT_B_H15` because it hard-collapses on `strong_pos`.
  - `OPT_A_20` does not rescue the original `2.0 s` task.
  - `H10` remains diagnostic ceiling evidence only.
  - `OPT_C_BEST`, `CAP_192_BEST`, and `WINNER_CONFIRM` do not replace the original baseline anchor.
- Plain-language meaning:
  - Shortening the horizon can produce large overall / tail gains, but the current model loses the strong positive tail behavior.
  - Simple optimizer, regularization, and mild-width changes do not raise the safe `2.0 s` ceiling enough to replace the current baseline.
- Trigger:
  - `H15` `best_by_structured` gives `rmse_steer=0.4930` and `abs_tail_last_0p5s.rmse_steer=0.6022`, but `strong_pos.tail_amp_ratio_pred_over_gt=0.2687` and `strong_pos.tail_flatness_rate=1.0000`.
  - `OPT_A_20` `best_by_structured` gives `rmse_steer=0.5887`, `abs_tail_last_0p5s.rmse_steer=0.7698`, and `late_peak_recall=0.5726`, all worse than the baseline anchor except the first-second prefix.
  - `OPT_C_BEST` and `CAP_192_BEST` are non-collapse but do not beat the baseline anchor on the selection order.
- Impact scope:
  - Future work should keep the live keeper split and only open a new branch if it directly targets `H15` strong-pos collapse or a justified architecture change.
- Evidence:
  - [2026-04-22 daily](daily/2026-04-22.md)
  - [effectiveness comparison table](../effectiveness_followup_20260422/effectiveness_comparison_table.csv)
  - [effectiveness summary](../effectiveness_followup_20260422/effectiveness_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-23 GPT Pro Follow-up Recommendation

- Decision:
  - If a new effectiveness follow-up is approved, prioritize `H15_AC_CF_HLF_v1`.
  - Define it as the `1.5 s` line plus coarse-fine steer decomposition and hard-late fine residual supervision.
  - Align the anti-collapse supervision window to the actual failing slice:
    - `HARD_LATE_START_SEC = 1.00`
    - `HARD_TAIL_START_SEC = 1.00`
  - Do not spend the next budget on plain optimizer or width sweeps.
  - If this anti-collapse attempt fails, escalate directly to `H15_LATE_RESIDUAL_HEAD_v1` instead of continuing loss micro-sweeps.
- Plain-language meaning:
  - `H15` is not bad because it lacks signal.
  - It is bad because it gets better average numbers by flattening the dangerous strong-pos late tail.
  - So the next step should protect late-tail shape directly, not polish optimizer settings.
- Trigger:
  - GPT Pro review judged that `H15` gains are real under the D0 absolute-window metric:
    - `rmse_steer: 0.5559 -> 0.4930`
    - `abs_tail_last_0p5s.rmse_steer: 0.7171 -> 0.6022`
  - The same review also judged that the gains are "bought" by collapse on:
    - `strong_pos.tail_amp_ratio_pred_over_gt = 0.2687`
    - `strong_pos.tail_flatness_rate = 1.0000`
  - `OPT_A_20` and `CAP_192_BEST` are taken as enough evidence that optimizer / width sweeps are no longer the highest-EV next move.
- Impact scope:
  - If approved, the next implementation should land in the modular V5.8 package:
    - `v58_modular/config.py`
    - `v58_modular/modeling.py`
    - `v58_modular/losses.py`
    - `v58_modular/train.py`
    - `v58_modular/evaluation.py`
  - The old monolithic wrapper should remain a launcher only.
- Evidence:
  - [2026-04-23 daily](daily/2026-04-23.md)
  - [GPT Pro review](../gptpro_effectiveness_review_20260423.md)
  - [effectiveness summary](../effectiveness_followup_20260422/effectiveness_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-23 `H15_AC_CF_HLF_v1` Closure

- Decision:
  - `H15_AC_CF_HLF_v1` is closed as a no-go.
  - `H15` is still not promotable.
  - Do not spend more budget on optimizer, width, or generic loss micro-sweeps on this branch.
  - If another run is approved, go directly to `H15_LATE_RESIDUAL_HEAD_v1`.
- Plain-language meaning:
  - The anti-collapse bundle helped, but not enough.
  - It reduced the old `H15` strong-pos flatness failure without restoring enough late-tail amplitude or late-peak recall.
  - The bottleneck now looks architectural in the late slice, not like a tuning-only problem.
- Trigger:
  - `best_by_loss` reaches:
    - `rmse_steer=0.5063`
    - `abs_tail_last_0p5s.rmse_steer=0.6323`
    - `late_peak_recall=0.5786`
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.5141`
    - `strong_pos.tail_flatness_rate=0.3750`
  - `best_by_structured` fails the explicit strong-pos floor:
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.3304 < 0.50`
  - manual review of eight representative `strong_pos` plots still finds `3/8` severe final-tail under-amplitude cases
- Impact scope:
  - Run A remains the response-structure anchor.
  - `baseline_fixed_input` remains the fit / tail keeper.
  - The next approved branch, if any, should be a minimal late residual head for `t >= 1.0 s`.
- Evidence:
  - [2026-04-23 daily](daily/2026-04-23.md)
  - [H15_AC_CF_HLF_v1 summary](../effectiveness_followup_20260423/h15_ac_cf_hlf_v1_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-23 `H15_LATE_RESIDUAL_HEAD_v1` Closure

- Decision:
  - `H15_LATE_RESIDUAL_HEAD_v1` is not promoted as a new keeper.
  - Run A remains the response-structure anchor.
  - `baseline_fixed_input` remains the fit / tail keeper.
  - If the late-residual direction is continued later, only `best_by_structured` from this run should be treated as the control checkpoint.
  - Do not reopen optimizer or width sweeps as the next explanation for this branch.
- Plain-language meaning:
  - Adding extra late-slice capacity helps more than the previous anti-collapse bundle alone, but not enough to make the branch safe to promote.
  - One checkpoint keeps the attractive average metrics and still collapses the dangerous bucket.
  - The other checkpoint repairs the dangerous bucket partially, but gives back too much fit / tail and still misses the amplitude floor.
- Trigger:
  - `best_by_loss` reaches:
    - `rmse_steer=0.4954`
    - `abs_tail_last_0p5s.rmse_steer=0.6284`
    - `late_peak_recall=0.6522`
    - but `strong_pos.tail_amp_ratio_pred_over_gt=0.3163`
    - and `strong_pos.tail_flatness_rate=1.0000`
  - `best_by_structured` improves the old `H15` collapse materially:
    - `strong_pos.tail_amp_ratio_pred_over_gt: 0.2687 -> 0.4904`
    - `strong_pos.tail_flatness_rate: 1.0000 -> 0.5000`
    - `late_peak_recall: 0.6355 -> 0.6656`
  - but `best_by_structured` still misses the amplitude target:
    - `strong_pos.tail_amp_ratio_pred_over_gt = 0.4904 < 0.60`
  - built-in late-residual diagnostics show the new head is active, but only mildly more active on `strong_pos` than on non-strong cases.
- Impact scope:
  - The branch is informative as a mechanism probe, not as a keeper replacement.
  - Future work on this line, if approved, should focus on making the late residual path more selective to the target failure bucket rather than reopening broad hyperparameter searches.
  - Same-tool recalc on the modular path still relies on the temporary shim:
    - `tmp/recalc_v58_metrics_shim_20260423.py`
- Evidence:
  - [2026-04-23 daily](daily/2026-04-23.md)
  - [H15_LATE_RESIDUAL_HEAD_v1 summary](../effectiveness_followup_20260423/h15_late_residual_head_v1_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`

## 2026-04-23 `H15_LATE_RESIDUAL_SELECTIVE_v1` Closure

- Decision:
  - `H15_LATE_RESIDUAL_SELECTIVE_v1` is not promoted as a new keeper.
  - Run A remains the response-structure anchor.
  - `baseline_fixed_input` remains the fit / tail keeper.
  - Do not reopen optimizer, width, bridge, or generic loss sweeps as the next explanation for this branch.
  - If this line is continued later, stay on the selective late-residual path and aim for stronger under-amplitude alignment with fit/head protection.
- Plain-language meaning:
  - The new selective gate is real.
  - It makes the late residual path much more `strong_pos`-aware.
  - But the branch still splits into two bad endpoints:
    - one checkpoint keeps reasonable fit and still under-repairs the dangerous bucket
    - the other checkpoint repairs the dangerous bucket, but damages the main task too much
  - So the next bottleneck is no longer "does late residual work?".
  - The next bottleneck is whether the correction can be aligned tightly enough to the true failure mechanism without hurting fit / tail / prefix / onset.
- Trigger:
  - `best_by_loss` reaches:
    - `rmse_steer=0.5356`
    - `abs_tail_last_0p5s.rmse_steer=0.6745`
    - `late_peak_recall=0.6756`
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.4947`
    - `strong_pos.tail_flatness_rate=0.7500`
  - `best_by_structured` proves the branch can push strong-pos repair much harder:
    - `strong_pos.tail_amp_ratio_pred_over_gt=1.4833`
    - `strong_pos.tail_flatness_rate=0.3750`
  - but `best_by_structured` does so in a globally damaged regime:
    - `rmse_steer=0.6379`
    - `abs_tail_last_0p5s.rmse_steer=0.7319`
    - `response_onset_delay_mae_sec=0.6270`
  - built-in diagnostics show bucket-level selectivity is now real:
    - `strong_pos_vs_non_strong_ratio.gate_prob=4.6584`
    - `strong_pos_vs_non_strong_ratio.gate_mean=3.3443`
  - but correlation with actual tail under-amplitude remains weak / slightly negative.
- Impact scope:
  - The live keeper split remains unchanged.
  - Future work on this line, if approved, should stay inside `v58_modular/` and the recalc tool.
  - The current run should be treated as a bracketed failure boundary:
    - `best_by_loss` = fit-preserving but under-repaired
    - `best_by_structured` = strong-pos-repaired but over-regressed
- Evidence:
  - [2026-04-23 daily](daily/2026-04-23.md)
  - [H15_LATE_RESIDUAL_SELECTIVE_v1 summary](../effectiveness_followup_20260423/h15_late_residual_selective_v1_summary.md)
  - [current-state](../../references/current-state.md)
- Status:
  - `active`
# 2026-06-22 v226 Formal Robustness CI Audit Boundary

- Decision:
  - v226 formal robustness / confidence-interval audit is complete and accepted as an audit/reporting pack, not as a model-search branch.
  - Formal headline remains locked: `loose_main_pool=avg_joint_focus`, `strict_main_pool=peak_floor_090`.
  - No new model, new tau, gate, router, v222b, or v223 is unlocked by v226.
  - Next action is to report v226 results to GPTPro and wait for the next bounded instruction.
- Plain-language meaning:
  - We now have confidence intervals and robustness evidence around the already-locked formal baselines.
  - The result supports packaging the locked models as the current formal main result while clearly reporting uncertainty and tail-error concentration.
  - It does not justify local escalation into another selector/generator search.
- Trigger:
  - GPTPro v226 accepted v225 as complete and requested audit-only robustness/CI evidence from v225 formal outputs.
  - v226 reproduced locked test metrics within `1e-5`: loose RMSE/tail `0.544884/0.629752`; strict RMSE/tail `0.571770/0.658306`.
  - Sample bootstrap test CI: loose RMSE `0.496066-0.593811`, loose tail `0.564811-0.693788`; strict RMSE `0.511036-0.635521`, strict tail `0.581652-0.736696`.
  - Subject-block test CI: loose RMSE `0.428783-0.599684`, loose tail `0.515881-0.687686`; strict RMSE `0.473689-0.615000`, strict tail `0.539479-0.706505`.
  - ZIP, required files, metric reproduction, leakage guard, forbidden scan, table alignment, and figure counts all passed.
- Impact scope:
  - v226 strengthens reporting/readiness evidence for the locked formal pair.
  - It preserves the v222b/v223 stop boundary and the rule that diagnostic rows do not enter formal tables.
  - Future work must come from the GPTPro loop with a bounded instruction and explicit stop condition.
- Evidence:
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/reports/v226_formal_robustness_ci_audit_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/tables/formal_metric_ci_sample_bootstrap.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/tables/formal_metric_ci_subject_block_bootstrap.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/logs/file_inventory.json`
- Status:
  - `active`

---
