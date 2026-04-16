## Supplemental Record: 2026-04-16, Activated official Codex Desktop path while preserving rollback artifacts

- Executor: Codex
- Why this was done:
  - After preserving the old custom bridge/provider workflow in repository snapshots, the user explicitly requested switching the active Codex path back to the official flow.
- What was changed:
  - Backed up the active live Codex files into:
    - `C:\Users\Administrator\.codex\switch_backups\official_switch_20260416_181358`
  - Replaced the active `config.toml` with a minimal config that removes:
    - `model_provider = "zx"`
    - `[model_providers.zx]`
    - `base_url = "http://localhost:8317/v1"`
  - Removed the active API-key auth file from the live position by moving:
    - `auth.json`
    - into the backup directory as `auth.json.backup`
- Important structural finding:
  - `C:\Users\Administrator\.codex` is not a separate directory.
  - It is a junction pointing to:
    - `D:\ClaudeCode\codex-home`
  - Therefore this switch changed the single live Codex home visible through both paths.
- Resulting live state:
  - active `config.toml` now keeps only general settings such as:
    - `model = "gpt-5.4"`
    - `model_reasoning_effort = "xhigh"`
    - `network_access = "enabled"`
    - project trust-level settings
  - active `auth.json` is absent from the live path
- What was intentionally not changed:
  - repository rollback snapshots under `reports/codex_bridge_snapshot_20260416/`
  - the explanatory rollback document
  - project-side `.claude` command files inside the repository
  - bridge launcher snapshots already stored in the repository
- What this means operationally:
  - The active Codex Desktop path is now prepared for official login instead of the custom `zx/8317` provider path.
  - Because the current app session may still hold in-memory state, a full restart of Codex Desktop is still expected before the user signs in again.
- Recommended next step:
  - Restart Codex Desktop completely.
  - On the next launch, sign in with ChatGPT/OpenAI through the official UI.
  - If rollback is ever needed, use:
    - `reports/codex_official_path_rollback_record_20260416.md`
    - plus the repository snapshots
    - plus the local backup directory `switch_backups\official_switch_20260416_181358`

## Supplemental Record: 2026-04-16, Codex official-path rollback record for the old bridge workflow

- Executor: Codex
- Why this was done:
  - The user explicitly asked to preserve the current pre-official-login Codex workflow so it can be restored later without guessing.
  - The machine currently contains both an official Codex desktop path and an older custom bridge/provider path, so a rollback record needed to capture the real files and the exact dependency chain before any future cleanup.
- What was inspected:
  - `C:\Users\Administrator\.codex\config.toml`
  - `C:\Users\Administrator\.codex\auth.json`
  - `D:\ClaudeCode\codex-home\config.toml`
  - `D:\ClaudeCode\codex-home\auth.json`
  - `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
  - `D:\ClaudeCode\codex-bridge\run-codex.cmd`
  - `F:\data_set_process\data_process\.claude\commands\codex-run.md`
  - `F:\data_set_process\data_process\.claude\commands\codex-workflow.md`
  - `F:\data_set_process\data_process\.claude\settings.local.json`
  - existing switch-history documents under `reports/`
- What was found:
  - The old workflow is not just a login mode; it is a chain:
    - project `.claude` command entrypoints
    - `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
    - `D:\ClaudeCode\codex-bridge\codex.exe`
    - custom provider `zx`
    - `http://localhost:8317/v1`
  - Both `C:\Users\Administrator\.codex\` and `D:\ClaudeCode\codex-home\` currently preserve the same provider shape:
    - `model_provider = "zx"`
    - `base_url = "http://localhost:8317/v1"`
    - `requires_openai_auth = true`
    - `auth_mode = "apikey"`
  - The repository already had older API-chain notes, but not a focused Codex rollback packet for the current official-vs-bridge decision.
- What was added:
  - New rollback note:
    - `F:\data_set_process\data_process\reports\codex_official_path_rollback_record_20260416.md`
  - Snapshot files:
    - `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\user.codex.config.toml.snapshot`
    - `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\user.codex.auth.json.snapshot`
    - `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\bridge.codex-home.config.toml.snapshot`
    - `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\bridge.codex-home.auth.json.snapshot`
    - `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\bridge.invoke-codex.ps1.snapshot`
    - `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\bridge.run-codex.cmd.snapshot`
    - `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\project.codex-run.md.snapshot`
    - `F:\data_set_process\data_process\reports\codex_bridge_snapshot_20260416\project.codex-workflow.md.snapshot`
- Why the record is structured this way:
  - A future rollback should not rely on memory of a single config file.
  - The high-risk failure mode is restoring only the login/auth state but forgetting the bridge entrypoint or project-side delegation commands.
- Recommended next step:
  - If the user later decides to switch to the official path, keep this rollback packet unchanged and do not delete the bridge files.
  - When a rollback is needed, restore the snapshot contents first, then validate the bridge path before diagnosing any model/API issue.

## Supplemental Record: 2026-04-16, Maintained Mainline Protocol-Safe Closure and Clean Full Baseline

- Executor: Codex
- Why this was done:
  - After the clean smoke-only checks showed that `W_STEER_REV` and `W_REVSEQ` do not explain the observed “later / weaker / flatter” pattern, the project stopped loss-first probing.
  - The priority shifted to restoring a fair `protocol-safe` training/evaluation loop on the maintained mainline, then running one clean full baseline before judging whether the model body is truly worse than the `2026-04-13` old-good full baseline.
- Minimal active-source changes applied:
  - Only edited:
    - `F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
  - Restored protocol-safe split helpers and outputs:
    - `load_protocol_split(...)`
    - `build_subject_split_indices(...)`
    - `choose_smoke_indices(...)`
    - `export_split_audit(...)`
    - `selected_samples_with_split.csv`
    - `split_audit.json`
    - `split_subject_counts.csv`
    - `split_sample_counts.csv`
  - Restored train-only fitting for feature normalization, curve-threshold fitting, and teacher-state PCA fitting.
  - Fixed training-time validation to iterate over real `val_loader` instead of `test_loader`.
  - Renamed best/final checkpoints to protocol-safe names:
    - `best_model_v5_8_protocol_safe.pth`
    - `model_rollpeak_transformer_v5_8_protocol_safe.pth`
  - Added the missing `load_json(...)` helper so the restored protocol split path can run end-to-end.
- What explicitly did not change:
  - No loss retuning
  - No teacher-state redesign
  - No protocol config edits
  - No edits to script copies inside `tmp` run directories
- Verification before the full run:
  - Syntax check passed:
    - `py -3 -m py_compile F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
  - Read-only split reconstruction under the repaired active source confirmed:
    - total samples: `5229`
    - split counts: `train 4184 / val 517 / test 528`
    - split subject counts: `12 / 3 / 3`
- Clean full baseline run:
  - Run directory:
    - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260416_103752`
  - Environment:
    - `predict_2`
    - `cuda`
    - `seed=2025`
    - `batch_size=64`
    - `epochs=40`
    - `smoke_mode=false`
  - Important note:
    - The run directory prefix still uses the historical `TRAIN_V5_4_STATECOND_REV_*` naming, but the actual run is clean `protocol-safe` full baseline as proven by `run_config.json` and `split_audit.json`.
- Closed-loop evidence from the new run:
  - `split_audit.json` and `selected_samples_with_split.csv` were produced successfully.
  - Training logs show true validation path:
    - `[Epoch xx/40] ... | Val=...`
  - Required evaluation artifacts were produced:
    - `test_metrics.json`
    - `test_metrics_head.json`
    - `test_metrics_tail.json`
    - `test_metrics_peak.json`
    - `test_metrics_reversal_structure.json`
- Same-metric formal comparison targets:
  - old-good full baseline:
    - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260413_174639/figures/baseline_recalc_current_metrics_20260415_summary.json`
  - bad smoke reference:
    - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726/figures/recalc_current_metrics_20260415_summary.json`
  - repaired clean full baseline:
    - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260416_103752/figures/recalc_current_metrics_20260416_full_protocol_safe_summary.json`
- Key numbers:

| Metric | `2026-04-13 old-good full` | `2026-04-15 bad smoke` | `2026-04-16 repaired clean full` |
| --- | ---: | ---: | ---: |
| `rmse_steer` | `0.6557` | `0.8273` | `0.6328` |
| `mae_steer` | `0.4488` | `0.6217` | `0.4330` |
| `head_amp_ratio_pred_over_gt` | `1.6826` | `0.1138` | `1.5856` |
| `head_flatness_rate` | `0.0303` | `0.9417` | `0.0227` |
| `response_onset_delay_sec` | `-0.0467` | `1.6881` | `-0.0990` |
| `tail_amp_ratio_pred_over_gt` | `2.9285` | `0.1560` | `3.3768` |
| `tail_flatness_rate` | `0.4356` | `0.9612` | `0.3883` |
| `late_peak_recall` | `0.4359` | `0.1455` | `0.5855` |
| `peak_time_mae_sec` | `0.4430` | `0.6642` | `0.3850` |
| `strong_pos.tail_amp_ratio_pred_over_gt` | `0.4582` | `0.0652` | `0.6342` |

- What this means:
  - After protocol-safe closure, the maintained mainline no longer supports the earlier “later / weaker / flatter than 4/13” conclusion.
  - The repaired clean full baseline is not obviously worse than `2026-04-13`; on most tracked metrics it is equal or better.
  - The strongest smoke-era failure signatures collapse after regime repair:
    - `head_flatness_rate: 0.9417 -> 0.0227`
    - `response_onset_delay_sec: 1.6881 -> -0.0990`
    - `strong_pos.tail_amp_ratio_pred_over_gt: 0.0652 -> 0.6342`
  - The remaining small gap is narrower and more specific:
    - `head_amp_ratio_pred_over_gt` is slightly lower than `2026-04-13` (`1.6826 -> 1.5856`)
    - this is no longer a broad “whole-trajectory flattening” regression claim
- Important fairness note:
  - `sample_count_diff_vs_protocol` remains non-zero in `split_audit.json`, but this is not a new mismatch between `2026-04-13` and `2026-04-16`.
  - The old-good run and the repaired clean full run have identical `expected_subjects`, `applied_subjects`, `subject_overlap`, and `sample_counts`, so their comparison is fair even though the historical protocol summary CSV still reports smaller counts.
- Updated project judgment:
  - The dominant source of the previously reported regression was regime drift, not teacher-state failure and not first-order responsibility from `W_STEER_REV` / `W_REVSEQ`.
  - The project should stop citing the `2026-04-15` smoke result as direct evidence that the maintained mainline itself is degraded.
  - The new fair baseline for any future single-variable validation should be:
    - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260416_103752`
- Recommended next step:
  - Do not continue blind loss tuning.
  - If another minimal validation is approved, run it under the repaired full protocol-safe regime and target the only remaining small discrepancy first, namely the slightly lower head amplitude ratio, rather than reopening the older broad “later / weaker / flatter” hypothesis.
# 项目进度记录

## 项目主题
极限工况下驾驶员真实反应建模与预测，用于后续辅助驾驶介入/协同控制参考。

## 阅读说明
- 想快速看当前情况时，先看下面的“当前状态速览”，不用从头阅读全部历史。
- 想按“哪天干了什么”来找，先看“日期索引”。
- 想按“做过哪些类型的工作”来找，先看“专题索引”。
- 想看最近一次发生了什么，优先从文末向上看最新的“补充记录”。
- 想追溯某个决策、某次整理、某轮实验为什么发生，再回看对应日期和专题记录。
- 从现在开始，新增记录尽量明确标注执行主体，区分：`Claude`、`Codex`、`用户`、`混合协作`。

## 当前状态速览
- 最近维护日期：`2026-04-14`
- 课题主线：从”固定 2 s 方向盘轨迹预测”转向”极限工况下驾驶员真实反应建模、短时预测、行为识别、共享控制支撑”。
- 最新主结论（2026-04-13）：`pca_latent` teacher-state 工程链路本身基本可跑通，但 maintained 主线训练脚本仍未落实 protocol 要求的 `subject-level fixed split`，现有 smoke/短跑结果因此不能作为正式可信证据。
- 模型主线：当前公平主问题已收口为 `structured_v2` 相对 `baseline-conditioned` 是否仍有净价值，而不是直接和“无 conditioning”比较。
- 当前瓶颈：matched-schedule 公平双跑显示 `structured_v2` 在 tail/peak/turning_count 上仍有收益，但 `boundary_shift_abs_err` 明显恶化，尚不支持直接替代 baseline。
- 文献状态：Zotero 已完成论文主线重整，当前核心写作入口为 `04_论文写作用核心集`。
- 协作状态：Claude / Codex 双向协作协议已收缩为”默认短版工作流 + 高风险附加清单 + 复盘模板”；现已进一步明确可按“用户给目标 + 验收口径，Claude 规划 / Codex 执行 / 强制写日志 / 禁删文件”的目标驱动自治推进模式工作，并补充了三条执行口径：高风险分叉优先指偏离“预测极限工况下驾驶员行为及车辆状态趋势”的主目标、同一小问题默认最多连续尝试 `3-4` 次、训练规模按风险决定而不是机械固定先做极小 smoke；同时已补上一个可复用的 Claude 高权限启动器（`bypassPermissions + add-dir Temp\\claude`）以尽量减少 `tasks/` 临时目录读取弹窗。
- 日志状态：`project_progress_master.md` 已改造成可检索项目日志，顶部提供状态速览、日期索引、专题索引和主体标注约定。
- 当前阶段（2026-04-09）：matched-schedule fairness closure 已完成，结论从“继续调哪一个 knob”推进到“先锁定 boundary failure mechanism，再决定是否做最小 ablation”：
  - 公平结论：`structured_v2` 仍值得继续，但当前证据不支持直接替代 baseline
  - 当前主瓶颈：`boundary_shift_abs_err` 从 `0.535222` 升到 `0.967678`，恶化幅度显著高于其它指标改善幅度
  - 最新 review 结论：更像是 `structured_v2` 的结构轨道注入把 tail 边界附近局部过渡拉宽/拉平/略后移，导致固定 `1.5s` 边界局部导数差计分变差；不更像 selection/matching 假差异
  - 当前动作：先做 boundary failure analysis 收口，再决定是否只做 1–2 个最小 ablation（优先怀疑 `structure_to_steer` 残差支路，其次 `structure_width` / `gate_temperature`）
  - 相关文件：`datasetprocess/final_code/model/training/baseline_eval_primary_aux.py`、`datasetprocess/final_code/model/training/conditioned_trajectory_head.py`、`reports/step4_decision_summary_20260408.md`

## 日期索引
- `2026-04-02`
  - 课题方向从固定 2 秒方向盘轨迹预测，转向极限工况下驾驶员真实反应建模、短时预测、行为识别与共享控制支撑。
  - 完成一轮文献检索、导入、组会方案整理。
  - 建立并强化了 Claude / Codex 的进度记录规则。
  - 完成 Zotero 全量重整、论文写作用核心集构建、旧分类删除、空节点清理。
  - 将 `project_progress_master.md` 顶部重构为更易检索的项目日志入口，便于按当前状态、日期、专题和执行主体快速回看。
- `2026-04-07`
  - 基于指定背景文件收口当前论文模型主线，明确当前真实主推为 allphase + deterministic conditioned v2，而非 multihyp 主线。
  - 确认 conditioned v2 相比 baseline 的已验证收益主要体现在 same-pool 公平比较下的 overall/tail/turning/interation slice 改善。
  - 将下一步优先级收口为：先做快反应样本与事件对齐误差归因，再做 driver/style 切片，最后才考虑最小化短时窗口验证。
- `2026-04-08`
  - Codex 完成 Task 1（context 值域）、Task 2（近似 timestep/boundary）、Task 3（重建训练+导出预测序列），三个任务全部收口。
  - 核心归因结论：Q1_fast tail 退化主因是 tail amplitude/shape 失配（r=0.72），不是 boundary 或 event timing；boundary_shift 恶化是时间位移而非斜率平滑，morphology 主导（single_lobe 最重）。
  - Step 4 tail amplitude penalty 已跑完并判定 No-Go；后续两轮 Codex review 将下一步最小方向收缩为优先审查 `gate_temperature`、其次 `structure_width`，而不是先调 `event_loss_weight`。
  - Claude 收口决策摘要，选择 Step 4 方案 A（tail amplitude penalty）作为唯一修改变量，写入 `reports/step4_decision_summary_20260408.md`。
  - 预测序列已可用：`reports/conditioned_v2_prediction_sequences.npz` 和 `baseline_prediction_sequences.npz`（749×400×2，channels: steer_rel, speed_delta）。
- `2026-04-09`
  - 完成 matched-schedule 公平双跑并确认：两边 config 在排除 `run_prefix` 与 `conditioning_mode` 后 matched，`sample_manifest_used.csv` 一致，`dropped_samples=0`。
  - 公平结论收口为：`structured_v2` 相对 `baseline-conditioned` 仍有信号，但当前不支持直接替代 baseline；关键阻塞项为 `boundary_shift_abs_err` 明显恶化。
  - Claude 发起 Codex 只读 review，进一步把 boundary failure mechanism 收口到“结构轨道注入拉宽/拉平/略后移 tail 边界附近局部过渡”，更像预测波形局部连续性问题，而不是 selection/matching 假差异。
  - 在用户明确允许更主动推进后，Claude 直接执行了 `event_residual_scale=0.0` 的正式单变量实验，验证 `structure_to_steer` 残差是否是当前主嫌疑。
  - 新实验结果表明：关闭 structure steer residual 后，`boundary_shift_abs_err` 仅从 `0.967678` 回落到 `0.900528`，但整体 `steer RMSE`、`tail RMSE`、`selection_score` 都明显变差，因此它不是当前 boundary 退化的唯一主因，也不是可直接默认采用的修复。
  - 下一步不再优先怀疑单独的 `structure_to_steer` 支路，而应把主嫌疑转向更上游的 `structure_to_tgt` / `structure_to_film` 或结构轨道 sharpness（`structure_width` / `gate_temperature`）。
  - 当前动作：teacher-forcing 对称化实验已经显示它是更大的干扰项；下一步优先补 baseline 的 TF0 matched 对照，而不是马上继续结构超参扫描。
- `2026-04-10`
  - Claude 将 maintained 主线重新锚定到真实在用的单文件训练脚本 `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`，而不是继续围绕旧的 `event_conditioned_*` 草稿命名展开下一阶段实现。
- `2026-04-13`
  - Codex 对 maintained teacher-state 扩展做只读安全审查，确认 `pca_latent` 链路方向基本成立，但当前主线脚本仍是 sample-level 随机切分，不符合 protocol 要求的 `subject-level fixed split`，因此现有 smoke/短跑结果不能作为正式可信证据。
  - Codex 将“目标驱动自治推进模式”正式落到仓库文档、模板与 `.claude` 命令入口，明确后续可按“用户给目标 + 验收 + 红线，Claude/Codex 持续推进、禁删文件、先写日志再总结”的方式工作。
  - 用户进一步补充了该模式的执行边界：真正高风险分叉优先指偏离“预测极限工况下驾驶员行为及车辆状态趋势”的主目标；同一小问题默认最多连续尝试 `3-4` 次；在路径与风险已经论证清楚时，可直接上完整训练 / 完整测试，不必机械先做极小规模 smoke。
  - 在用户明确“不着急，可以给 8-9 小时慢慢尝试”后，Codex 将当前阶段量化目标从偏保守版本上调为更强的多指标目标，重点不再只是 RMSE，而是同时纳入趋势相似性、峰值延时和反向预测率。
  - 在用户进一步要求“更狠一点”后，Codex 又将阶段量化目标再上调一档，形成更激进的最终执行版本：更低的 `rmse_steer/rmse_ay` 门槛、更严格的趋势相关性与峰值延时要求、以及更低的反向预测率上限。
  - 先做了只读梳理，确认当前代码里已经具备下一阶段第一版 response-state-aware 所需的四类核心锚点：`A/C` teacher state、`reversal` 标签、`soft peak timing`、以及可复用 trajectory amplitude loss 作为 `peak intensity` 监督代理。
- `2026-04-14`
  - Codex 核对了本机 Claude CLI、项目级 `dontAsk` 配置与 `Temp\\claude\\...\\tasks\\...` 权限弹窗来源，确认该弹窗属于工作区外临时任务目录读取，不是提示词本身失效。
  - Claude 按当前 maintained 主线代码、协议与代表性归因结果，整理了一份可直接发给 GPT 深度研究的单包资料（prompt + code + protocol + evidence），用于围绕 driver response forecasting 的复杂反打/峰值/尾段结构问题做针对性外部研究。
  - 按用户选择的“方案 2”，新增可复用启动器：以项目根目录启动 Claude，并附带 `--permission-mode bypassPermissions` 与 `--add-dir %LOCALAPPDATA%\\Temp\\claude`，用于尽量减少 `tasks/` 临时目录读取弹窗。
  - 启动器已在 `startup/claude_bypass_permissions.ps1` 与 `startup/Claude_Code_BypassPermissions.cmd` 中落地；首次验证发现 `Get-Command claude` 误命中旧版 `D:\\Apps\\nodejs\\claude` (`2.1.91`)，随后修正为优先使用 `D:\\ClaudeCode\\global\\claude` (`2.1.107`)，并再次用 `--help` 验证通过。
  - 在不新增复杂头和不改 split / horizon / anchor 主规则的前提下，先把第一版 response-state-aware 主线收缩为“显式开关 + 最小闭环”：
    - `ENABLE_RESPONSE_STATE_V1`
    - `ENABLE_STATE_DISTILL`
    - `ENABLE_REVERSAL_AUX`
    - `ENABLE_PEAKTIME_AUX`
    - `ENABLE_PEAKINTENSITY_AUX`
  - 对应地将默认训练配置从“代码存在但多数关闭”推进为“第一版正式接入但仍保持保守权重”：
    - `LAMBDA_STATE = 0.08`
    - `LAMBDA_REV = 0.05`
    - `W_PEAKTIME = 0.05`
    - `W_AMP` 继续保留并明确解释为 `peak intensity supervision` 的 trajectory-level 代理
    - `W_REVSEQ`、`W_STEER_REV` 继续保持关闭，避免在第一版就把 correction sequence shaping 也一并推高，导致主干重新失稳
  - 同时把这些 response-state-aware 开关和权重写入 `run_config.json`，并在训练启动日志中显式打印，使后续 run 比较时能清楚知道到底开启了哪些反应状态分支。
  - 训练与验证路径中的 `rev_head` loss 也从先前训练阶段硬关闭，调整为由 `ENABLE_REVERSAL_AUX` 显式控制；这样主线第一次真正形成了“trajectory + state distill + reversal aux + peak timing aux + peak intensity proxy”这一版最小闭环。
  - 随后进一步把原先公式型 `A/C` teacher state 改造成可切换 teacher-state 接口：新增 `TEACHER_STATE_MODE`（当前默认 `pca_latent`）和 `TEACHER_STATE_DIM`，保留 `old_ac` 作为 legacy baseline，同时加入基于 train split SVD 的 PCA latent 构造，使 teacher state 首次可以从现有生理/EEG 基础特征中得到更 data-driven 的连续低维表示。
  - 模型侧同步把 `state_head` 从固定 2 维改为 `state_dim` 可配置，并让 decoder context 维度随 teacher latent 实际维度自动扩展，避免继续把 backbone 写死在 `A/C` 二维语义上。
  - 评估/导出侧也从 `A_veh/C_veh/A_teacher/C_teacher` 固定命名扩展为通用 latent dump：新增 `test_state_meta.json`、按组件名导出 `veh_*` / `teacher_*` 列、样例图标题改成通用 latent summary；同时仍为前两维保留 legacy A/C 列，兼容已有阅读习惯。
  - 本次 PCA latent 仍严格只在脚本当前 train split 上拟合并用于全体样本投影，至少避免了直接在全体样本上拟合 latent 的明显泄漏；但由于这条 maintained 脚本本身仍是 sample-random split，当前实现仍只能视为“主线原型版”，还不能当作最终 publication-safe teacher-state 方案。
  - 已再次用 `py -3 -m py_compile` 对更新后的脚本做语法检查并通过；此前 bash 里的默认 `python` 实际是 3.5.2，导致首次 `py_compile` 对 f-string 报假阳性，已定位为环境版本问题而不是本次改动引入的新语法错误。
  - 当前判断：这一步的真正价值不在于“PCA 一定就是最终最好状态定义”，而在于主线终于摆脱了对手工 A/C 公式的硬绑定，进入“teacher state 可切换、latent 维度可扩展、评估导出可泛化”的状态，为后续继续迭代生理表征、再接在线多模态接口打下了结构基础。
  - 推荐下一步：先做一次极小 smoke run 或短 epoch 预跑，重点检查 `teacher_state_meta.json`、`test_state_meta.json`、latent 数值尺度、`loss_state` 与主任务 loss 的量级关系，以及 PCA latent 下前两维和 peak/reversal 的可解释性；稳定后再决定是否继续做 subject-split-safe 版本或更强的生理 state 表征。 
  - 紧接着 Claude 实际完成了一次基于 `predict_2` CUDA 环境的 smoke run（此前先后修掉了 conda 环境名误用、旧硬编码数据路径、以及 PCA 投影 valid-dim 广播错误三处阻塞），确认这条新主线已经能在真实数据上完整走通。
  - 本次 smoke run 关键信息：
    - 运行目录：`F:\数据集处理\data_process\datasetprocess\多模态数据\程序运行结果\TRAIN_V5_4_STATECOND_REV_20260413_104231`
    - 环境：`predict_2`，设备 `cuda`
    - smoke 配置：`max_samples=256`、`epochs=2`、`batch_size=32`
    - 数据侧：共扫描到 `5229` 个事件样本，smoke 实际取前 `256` 个，其中 train `204` / test `52`
    - teacher state：`mode=pca_latent`、`state_dim=4`、components=`latent_0..latent_3`
    - 训练闭环：2 个 epoch 均正常收敛，`Train 2.768 -> 1.948`，`Test 2.180 -> 1.830`
    - 导出闭环：`teacher_state_meta.json`、`test_state_meta.json`、`test_state_dump.csv`、预测图和 loss 曲线均已成功产出
  - 当前判断：PCA teacher-state 主线已经从“代码设计”进入“可实际跑通的原型”阶段，说明固定 `A/C` 依赖已被结构性打破；下一步不再是修能不能跑，而是判断 teacher latent 本身是否值得继续增强与清洗。
  - 本次 smoke 仍暴露两个值得继续处理的问题：
    - `base_mu/base_sd` 处出现 `Mean of empty slice` / `Degrees of freedom <= 0` 警告，说明 teacher base 12 维里仍有部分列在当前 smoke train 子集上全空或近全空，后续应把这些列的缺失统计显式写入 meta，并在 teacher builder 里更稳健处理
    - 仍有少量事件文件缺失（本次日志里点名了 `lxy/txj/zx/zxy` 的若干文件），虽然不影响主线 smoke 跑通，但正式实验前最好确认这是预期缺口还是数据整理遗漏
  - 推荐下一步：优先补一版 teacher-base 缺失维统计与 valid-mask 日志，然后再做一次稍大一点的短跑（例如 1k~2k 样本），检查 4 维 latent 与 `peak_abs_steer` / `reversal` 的关系是否出现稳定结构；若有，再进入 subject-split-safe 重构。 
  - Claude 随后继续补上了 `teacher_base_missing_stats.json` 与 `teacher_state_meta.json` 中的缺失维摘要，并完成了 `1000` 样本、`2 epoch`、`predict_2` CUDA 环境短跑：`F:\数据集处理\data_process\datasetprocess\多模态数据\程序运行结果\TRAIN_V5_4_STATECOND_REV_20260413_144211`。
  - 这一轮更大样本短跑显示主线稳定性比 256 样本 smoke 更高：
    - train/test=`800/200`
    - `Teacher-base missing dims: 0/12 | all-missing=[]`
    - `teacher_state_mode=pca_latent`、`state_dim=4` 保持稳定
    - `Train 2.126 -> 1.534`，`Test 1.531 -> 1.431`
    - test 指标：`rmse_steer=0.767`、`rmse_yawrate=0.166`、`rmse_ay=2.988`
  - 这说明当前 `PCA teacher latent + trajectory/state` 主线已经不是偶然跑通，而是在更大样本下也保持了正常收敛和完整导出闭环。
  - 当前最突出的新问题转向 `rev_head` 的强反转标签：train 仅 `37/800` 个正样本，`pos_weight=20.622`，测试上 strong reversal 结果退化为 `tp=0, fp=60, fn=9, f1=0.0`；相比之下 weak reversal 已开始出现一定可分性（`f1≈0.374`）。
  - 当前判断：问题更像是“强反转标签过稀 + 当前损失配置对其不稳”，而不是 teacher latent 主线本身失效。因此下一步应优先分析 latent 与反转/峰值关系，或审视 strong/weak reversal 标签与 `LAMBDA_REV` 配置，而不是急着否定 PCA state 路线。
  - 关键文件：`datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
  - 关键锚点：
    - response-state 开关与默认权重：`:218-250`
    - teacher state `A/C` 构造：`:489-526`
    -样本与标签准备（含 reversal label）：`:1320-1448`
    - 训练启动配置打印与 run_config：`:1495-1537`
    - 训练 loss 闭环：`:1552-1609`
    - 验证 loss 闭环：`:1640-1708`
  - 执行主体：Claude
  - Why：用户已明确要继续推进“multimodal-ready baseline-conditioned reaction-state-aware” 新模型，不再停留在结构图和方案层。
  - How to apply：后续所有新 run 如果基于这条脚本，必须在比对时把 response-state 开关和权重一起视为实验条件的一部分，不能只看主干名字。 
  - Codex 复核 `project_progress_master.md`、`README.md`、`CLAUDE.md` 与 `.claude/commands/codex-workflow.md` 后，确认仓库现有规则已经支持“用户只给目标、AI 在禁删文件前提下持续自主推进、每次执行走既有协作工作流并强制写详细日志”的目标驱动自治推进模式；当前真正需要用户额外给定的只有目标、验收标准和硬约束，而不是每一步微指令。

### 补充记录（2026-04-14，GPT 深度研究发送包整理）
- 执行主体：Claude
- Why：用户希望把“面向当前项目问题的深度研究指令”、相关 maintained 主线代码、协议配置和代表性结果一次性整理成单一压缩包，便于直接发送给 GPT 做定向深度研究，而不是在聊天里零散复制。
- What was done：
  1. 读取并收口当前主线研究对象：根 README、`datasetprocess/final_code/README.md`、主协议 `protocol_primary_control_v2_context_full2s/protocol_config.json`、主训练脚本 `future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`。
  2. 提炼适合外部研究模型理解的核心问题：在线可见输入、subject-level fixed split、公平比较、复杂反打/符号翻转、峰值时序、尾段 shape 恢复，而不是泛化成通用自动驾驶规划。
  3. 选取适合随包附带的代表性证据文件：主训练/诊断脚本、protocol split 文件、conditioned v2 快反应/边界归因报告、代表性失败案例摘要、attribution CSV/NPZ 等。
  4. 准备把完整中文版深度研究 prompt 写入单独文件，并按 `prompt / code / protocol / evidence` 目录组织为一个发送包，方便后续直接压缩发送。
- What was found：
  1. 当前最适合 GPT 深度研究的问题，不是“泛化提分”，而是：在 3s online-visible history -> 2s future response 设定下，怎样提升复杂反打、符号翻转、峰值时间和尾段结构恢复，同时不破坏 subject split 与公平协议。
  2. 仓库里已经存在一批高价值证据，尤其是 `reports/conditioned_v2_fast_boundary_attribution_20260408.md`、`reports/conditioned_v2_representative_cases_20260408.md`、`reports/attribution_master_table.csv`、`reports/conditioned_v2_prediction_sequences.npz`，它们能帮助外部研究模型更准确理解“问题到底出在哪”。
  3. 需要避免把临时 run 副本、backup 脚本、`.claude` 配置、无关适配器/启动器材料混入发送包，否则会稀释研究焦点并增加噪声。
- Recommended next step：
  1. 生成发送包后，优先把其中的 prompt 文件与 evidence 一起发给 GPT 深度研究。
  2. 如果第一轮研究输出过泛，可再基于同一包追加“请严格围绕 reversal / peak timing / tail-shape fidelity / privileged-information distillation 回答”的二次提示。

### 补充记录（2026-04-13，道路场景来源与道路列链路安全检查）
- 执行主体：Claude
- Why：用户澄清当前真实实验所对应的道路场景应以新的 SILAB 文件计算结果为准，而此前项目里曾拷贝过旧版道路场景；因此需要核查 maintained code 中道路相关列是否真由当前真实场景派生，还是可能残留旧模板/旧场景结果，从而影响训练公平性、anchor 一致性和论文有效性。
- What was checked：
  1. 读取并核对 maintained 车辆预处理脚本 `datasetprocess/final_code/processing/vehicle/preprocess_vehicle_v14.py`，确认道路相关列是否在 active preprocessing 中生成还是仅被透传/重采样。
  2. 读取 active protocol `datasetprocess/final_code/model/training/protocol_primary_control_v2_context_full2s/protocol_config.json`，确认当前 split policy、future horizon、online-only 假设未被这次道路检查范围直接改写。
  3. 追踪 maintained 训练/诊断脚本对道路列的直接依赖，包括 `future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`、`future_steer_event_rollpeak_transformer_v5_8_diag_eval.py`、`future_steer_speed_subjectsplit_masked.py`、`run_event_conditioned_trajectory_baseline.py`、`event_conditioned_baseline_model.py`、`conditioned_trajectory_head.py`。
  4. 只读回溯历史区 `datasetprocess/多模态数据/数据处理代码/build_road_template.py` 与 `apply_road_template.py`，确认 `road_type_fixed` / `ref_nn_ok` 等道路标签的可疑上游生成逻辑。
  5. 额外请 split-safety-reviewer 做只读复核，专门从 split safety、anchor drift、未来信息泄漏和公平比较角度复审一次。
- What was confirmed：
  1. `preprocess_vehicle_v14.py` 不生成 `zx1|lanecurvatureXY`、`zx1|lateraldistance`、`road_type_fixed`、`ref_nn_ok`；它会读取输入车辆 CSV，统一重采样后连同已有列一起写出到 `*_fixed_200Hz_v14.csv`。
  2. `preprocess_vehicle_v14.py` 的 `resample_to_target()` 会对所有数值列做时间插值，因此若 `road_type_fixed`、`ref_nn_ok` 以数值 0/1 形式存在，会被插值成连续值，语义上不再是严格离散标签。
  3. maintained 主训练脚本 `future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 直接依赖 `zx1|lanecurvatureXY`、`road_type_fixed`、`ref_nn_ok` 来决定弯/直道判定与 anchor 规则，也会消费 `lateraldistance` 家族列生成 `lane_unwrap` 等特征；因此道路列来源若不对，会直接影响训练样本对齐与公平比较。
  4. event-conditioned 主线虽然不直接使用 `road_type_fixed/ref_nn_ok` 作为输入，但通过 `curve_future -> curve_norm` 明确依赖曲率未来序列；若曲率列仍来自旧场景，也会进入新主线。
  5. 当前 protocol config 仍保持 subject-level fixed split、3s history、2s future、online-only vehicle history；本次检查范围内未发现道路列直接破坏主体切分或直接引入未来标签泄漏的证据。
- Risks flagged：
  1. 当前 maintained code 里没有找到这几列的 active 生成器；能明确找到的上游生成逻辑在历史区，而且是“单参考 run 构建模板 + 最近邻投影到所有车辆 CSV”的方案。这说明仅凭 maintained code 不能证明当前道路列一定来自新的真实 SILAB 场景。
  2. 如果项目里实际使用的车辆 CSV 仍承载旧模板/旧场景写入的 `road_type_fixed/ref_nn_ok`，那么 v5.8 主线中的 anchor 规则会发生系统性漂移，旧 run 与新 run 也会失去严格可比性。
  3. `lateraldistance` 在主训练脚本里查找的是无前缀别名（如 `lateraldistance`、`lateraldistance_start`），而旧数据链里常见的是 `zx1|lateraldistance`；这提示存在“你以为用了横向道路偏移，实际上主脚本可能没吃到”的列名不对齐风险。
- Recommended next step：
  1. 直接抽查当前真实训练所用 vehicle CSV 的表头与样本值，确认 `zx1|lanecurvatureXY`、`zx1|lateraldistance`、`road_type_fixed`、`ref_nn_ok` 是否已来自新的真实 SILAB 场景计算结果。
  2. 查明真实使用的 `road_template.npz`（若存在）是由哪条参考 run、哪版 SILAB 场景生成；没有这一步，仍不能把“旧场景残留”从 plausible risk 降到已排除。
  3. 若证实道路标签来自旧场景，最小安全边界应从道路模板/道路列重算开始，至少重跑车辆预处理与所有直接依赖这些列的训练/诊断链路；若 protocol manifest 中的 `road_type_anchor` 也受影响，则还需向前重建相关 manifest/派生表。

### 补充记录（2026-04-13，真实 vehicle CSV 抽查：道路列已在上游写入）
- 执行主体：Claude
- Why：在前一轮只读检查中，已确认 maintained code 会消费道路相关列，但仍缺少“当前真实训练所用 vehicle CSV 里这些列到底是什么状态”的直接证据；因此继续抽查多个被试的原始 `vehicle_aligned_cleaned.csv` 与带 `roadtype_labeled`/`segments` 的配套文件，判断道路列是否早已在更上游写入，以及是否仍呈现模板投影痕迹。
- What was checked：
  1. 抽查 `zx`、`txj`、`byx` 三个被试的原始 `*_vehicle_aligned_cleaned.csv` 首屏数据。
  2. 抽查 `zx` 的 `*_vehicle_aligned_cleaned_roadtype_labeled.csv` 与 `*_roadtype_segments.csv`，确认 labeled 文件和原始文件中的道路列关系。
  3. 搜索原始 `zx` 文件中 `road_type_fixed_str=straight/curve` 的直接行值，确认原始清洗文件内是否已带模板投影后的道路标签。
- What was confirmed：
  1. 原始 `vehicle_aligned_cleaned.csv` 已直接包含 `road_s_ref_m`、`road_type_fixed`、`road_type_fixed_str`、`ref_nn_dist_m`、`ref_nn_ok`，说明这些列在 maintained preprocessing 之前就已经写入真实 vehicle 文件，而不是在 `preprocess_vehicle_v14.py` 或训练脚本里现算。
  2. 抽查到的原始文件同时保留 `zx1|lanecurvatureXY` 与 `zx1|lateraldistance`，因此曲率和横向偏移相关信息也已在更早阶段进入 vehicle CSV。
  3. `*_roadtype_labeled.csv` 里额外出现 `road_s_m`、`kappa_used`、`kappa_smooth`、`road_type`、`road_type_str`，同时仍带 `road_type_fixed/ref_nn_ok`；这与历史脚本“先基于参考 run 建模板，再把 fixed road type 投影回车辆 CSV”的模式一致。
  4. `*_roadtype_segments.csv` 明确给出了长距离 straight/curve 分段，进一步说明 road-type 体系不是训练时临时计算，而是早已固化到上游道路标注产物里。
- Risks flagged：
  1. 现在可以更强地确认：若这些上游道路列对应的仍是旧 SILAB 场景或旧模板，那么后续 maintained 主线将稳定继承旧结果，而不是被 active preprocessing 自动修正。
  2. 抽查样本中 `road_type_fixed=0/ref_nn_ok=1`、`ref_nn_dist_m` 极小、`road_s_ref_m` 连续平滑，整体非常像模板投影后的固定参考标签；这使“旧模板残留”从一般怀疑上升为高可信风险，但仍缺最后一跳证据——当前真实使用的模板是否确实由新版 SILAB 场景生成。
  3. 主训练脚本对 `lateraldistance` 的候选列名仍不包含 `zx1|lateraldistance`；抽查再次证明原始 vehicle 常见的是带前缀版本，所以该列是否被主训练真正消费，仍需单独核对或修正。
- Recommended next step：
  1. 继续追当前实际使用的 `road_template.npz` 或其等价产物来源，确认它到底对应新版 SILAB 还是真有旧场景残留。
  2. 若模板来源坐实不对，最小安全动作应从“重生成道路标签列”开始，而不是只动训练脚本。
  3. 额外抽查一个进入 curve 段的原始 vehicle 文件片段，确认 `road_type_fixed` 与 `road_type_fixed_str` 在原始 CSV 中确实会发生 0→1 切换，并观察与 `zx1|lanecurvatureXY` 的对应关系；这样可以进一步验证当前道路列就是固定模板标签而非其他来源。

### Supplemental Record: 2026-04-09, First Hard-Judgment Experiment — `structured_v2` with Teacher Forcing Disabled
- **Executor**: Claude
- **Why it was done**:
  - After the strict code review, the largest remaining implementation-side threat to mechanism interpretation was the train/infer asymmetry caused by `teacher_forcing_ratio=1.0`.
  - The project therefore switched from small-step probing to a decisive test: keep the current shared runner fixed, but set `teacher_forcing_ratio=0.0` for `structured_v2` and see whether the previously observed failure pattern changes materially.
- **What was done**:
  1. Launched a formal run under the maintained training script with only one intended change relative to the matched structured_v2 reference:
     - run root: `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_20260409_101029`
     - changed `teacher_forcing_ratio: 1.0 -> 0.0`
     - kept `conditioning_mode=structured_v2`, `event_residual_scale=1.0`, manifest, seed, device, epochs, min_epochs, patience, optimizer, model dims, `structure_width`, `gate_temperature`, and `selection_mode=legacy_rmse` unchanged
  2. Read back the resulting `run_summary.json` and `selection_comparison.csv` and compared them against the existing matched baseline and matched structured_v2 references.
- **What was found**:
  1. Disabling teacher forcing changed the result pattern far more than the earlier no-residual test, which strongly suggests that train/infer asymmetry is a major confounder in the current implementation path.
  2. Relative to matched structured_v2 (`teacher_forcing_ratio=1.0`), the TF0 run improved several important test metrics:
     - `steer_rmse`: `0.530886 -> 0.499995`
     - `boundary_shift_abs_err`: `0.967678 -> 0.667954`
     - `peak_time_abs_err_s`: `0.564516 -> 0.528629`
     - `turning_count_abs_err`: `1.391129 -> 1.548387` (worse than structured_v2, but still better than baseline)
  3. The TF0 run did **not** dominate baseline cleanly:
     - baseline still has better `boundary_shift_abs_err`: `0.535222`
     - baseline still has slightly better `tail RMSE`: `0.387507` vs TF0 `0.392474`
     - baseline is slightly better on `selection_score`: `0.879389` vs TF0 `0.877028`
  4. But the key project-level meaning is now much clearer:
     - teacher forcing asymmetry is not a minor detail; it materially changes the observed ranking and failure pattern
     - because only `structured_v2_TF0` has been run so far, the project still lacks a fully matched TF0 baseline control; therefore the current TF0 result is strong evidence that teacher forcing matters, but not yet the final fair answer on whether structured_v2 remains superior under TF-symmetric training
- **Recommended next step**:
  1. Before opening a new structural hyperparameter branch, run `baseline + teacher_forcing_ratio=0.0` as the matched TF0 control.
  2. Use that pair to decide whether the project should keep investing in structured_v2 after removing the biggest known implementation confounder.
  3. Delay the second “hard judgment” shot on `gate_temperature` until after the TF0 pair is complete, because otherwise the next conclusion would still rest on an unresolved teacher-forcing asymmetry.

### Current comparison snapshot after the first TF0 run
| Run | overall steer RMSE | tail RMSE | peak_time_abs_err_s | boundary_shift_abs_err | turning_count_abs_err | selection_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.507124 | 0.387507 | 0.598790 | 0.535222 | 1.951613 | 0.879389 |
| structured_v2 | 0.530886 | 0.379108 | 0.564516 | 0.967678 | 1.391129 | 0.906284 |
| structured_v2_noresid | 0.564264 | 0.412240 | 0.537097 | 0.900528 | 1.346774 | 0.934958 |
| structured_v2_TF0 | 0.499995 | 0.392474 | 0.528629 | 0.667954 | 1.548387 | 0.877028 |

### Key anchors for the TF0 interpretation
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_20260409_101029/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_20260409_101029/selection_comparison.csv`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:448`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:453`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:461`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:717`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py:605`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_head.py:152`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_head.py:157`

- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_20260409_005715/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_20260409_010302/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_NORESID_20260409_094418/run_summary.json`

## 2026-04-14 维护主训练脚本接入固定评估闭环（tail / peak / reversal）

### 补充记录
- 执行主体：Claude
- Why：当前 maintainted 主线 `future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 已能完成 full run，但现有导出主要集中在总体 RMSE/MAE、road-type 和二元 reversal 指标，仍不足以直接判断用户最关心的 tail flattening、峰值时序偏移和反打结构保真是否真正改善。
- What was done：
  1. 在训练脚本评估段新增一组轻量 helper，用于稳定计算 tail、peak、reversal-structure 三类结构指标。
  2. 将 `evaluate_and_plot(...)` 扩展为除原有 `test_metrics.json` / `test_metrics_by_reversal.json` 外，额外导出：
     - `test_metrics_tail.json`
     - `test_metrics_peak.json`
     - `test_metrics_reversal_structure.json`
  3. 同时把 `tail_metrics`、`peak_metrics` 合并写入主 `test_metrics.json`，把 `structured` reversal 指标并入 `test_metrics_by_reversal.json`，避免后续只看主 JSON 时丢失结构诊断信息。
  4. 新增的 tail 指标重点覆盖：末 25% 区间 `tail_rmse_steer`、`tail_mae_steer`、tail std 比、tail amp 比、tail slope 误差、`tail_flatness_rate`，用于直接量化“尾段是否被压平”。
  5. 新增的 peak 指标重点覆盖：峰值时间 MAE/RMSE、峰值幅值 MAE/RMSE、GT 晚峰比例、晚峰 recall，用于区分“是 timing 错还是 amplitude 错”。
  6. 新增的 reversal-structure 指标重点覆盖：首次反打时间误差、反打次数误差、次数完全匹配率，以及按 `straight / weak_pos / strong_pos` 分桶的 steer RMSE、tail RMSE、tail amp ratio、tail flatness rate，用于判断模型到底是“报到了反打”还是“保住了反打结构”。
  7. 用 `py -3 -m py_compile` 对更新后的训练脚本做语法检查，通过；同时再次确认 bash 默认 `conda` 不可用、默认 `python` 仍不适合作为本仓库的权威验证入口，因此本轮继续沿用已验证过的 `py -3` 语法检查路径。
- What was found：
  1. 当前固定评估闭环的最佳落点确实就是训练脚本中的 `evaluate_and_plot(...)`，因为这里已经稳定持有反归一化后的 `pred/true`、reversal label、road-type mask、state dump 和 figure 输出上下文。
  2. 现有脚本在 `evaluate_and_plot(...)` 后半段仍残留一段重复的不可达旧代码（位于函数 `return` 之后），本轮没有顺手清理，避免把评估闭环改动和历史冗余清扫混在一起；后续若继续维护该函数，可考虑单独做一次最小清理。
  3. 这次改动完成后，后续每次 full run 不再只能凭示例图“肉眼判断 tail flattening”，而可以直接从固定 JSON 中读取尾段收缩、晚峰召回和反打结构失真程度。
- Recommended next step：
  1. 用这版脚本跑一次正式评估或下一次 full run，读取新增的 tail/peak/reversal 结构指标，验证它们是否与当前人工看图结论一致。
  2. 若新增指标仍显示 strong/late reversal 样本的 `tail_amp_ratio_pred_over_gt` 持续偏低、`tail_flatness_rate` 偏高，再进入第二阶段最小干预：优先小幅打开 `W_REVSEQ`，必要时再打开 `W_PEAKTIME`。
  3. 后续做 run 对比时，优先对比新增结构指标，而不再只看总体 `rmse_steer`。

## 2026-04-14 Claude-via-Codex adapter background-window fix

### User-visible issue
- The user reported that after closing the visible black window titled `F:\python3.11\python.exe`, the window would immediately appear again.

### Root cause confirmed
- That visible window was not the hidden PowerShell supervisor itself.
- It was the adapter child process launched by `startup/claude_codex_adapter_supervisor.ps1` via `F:\python3.11\python.exe`.
- The supervisor is intentionally a keep-alive process. When the child adapter exits, the supervisor treats that as an unexpected stop and restarts it.
- Therefore, manually closing the child console window caused the supervisor to relaunch the adapter, which looked like "self-starting".

### Change applied
- Updated `F:/data_set_process/data_process/startup/claude_codex_adapter_supervisor.ps1`.
- The supervisor now prefers `F:\python3.11\pythonw.exe` and falls back to `F:\python3.11\python.exe` only if `pythonw.exe` is unavailable.
- Added startup logging of the selected Python runtime so later diagnosis can quickly confirm whether the adapter is running in background-window mode.

### Verification completed
- Restarted the adapter service with `startup/stop_claude_codex_adapter.ps1` and `startup/start_claude_codex_adapter.ps1`.
- Confirmed healthy service state at `http://127.0.0.1:8417`.
- Confirmed adapter child process is now `pythonw.exe` with empty `MainWindowTitle`, meaning no visible console window is attached.
- Re-ran the real launcher:
  - `F:/data_set_process/data_process/startup/claude_via_codex_api.ps1 -ClaudeArgs @('-p','请只输出ok')`
- Verified successful end-to-end response: `ok`

### Operational meaning going forward
- Closing the previously visible adapter console window should no longer be part of the normal stop path because that window should no longer be shown.
- If the service actually needs to be stopped, use:
  - `F:/data_set_process/data_process/startup/stop_claude_codex_adapter.ps1`
- If status needs to be checked, use:
  - `F:/data_set_process/data_process/startup/claude_codex_adapter_status.ps1`

## 2026-04-14 Claude Code CodexAdapter launcher UX + encoding hardening

### User-visible symptoms
- Double-clicking `startup/Claude_Code_CodexAdapter.cmd` opened a `Windows PowerShell` session with the standard startup banner instead of a cleaner Claude-only window.
- The adapter base URL showed port `8418` instead of always `8417`.
- In the Claude TUI, Chinese output could appear as mojibake / mis-decoded text.

### Root-cause reading
- The launcher `.cmd` file was explicitly starting `powershell.exe` without `-NoLogo` or `-NoProfile`, so the normal PowerShell banner and profile-loading noise were expected.
- Port `8418` was not a failure: the supervisor is designed to fall back when preferred port `8417` is already occupied.
- The remaining Chinese mojibake looked consistent with a Windows-side client decoding problem on streamed JSON/SSE payloads carrying raw UTF-8 non-ASCII characters.

### Changes applied
- Updated `F:/data_set_process/data_process/startup/Claude_Code_CodexAdapter.cmd` to use:
  - `powershell.exe -NoLogo -NoProfile -NoExit ...`
- Updated `F:/data_set_process/data_process/startup/Claude_Code_BypassPermissions.cmd` the same way for consistency.
- Updated `F:/data_set_process/data_process/tools/anthropic_codex_adapter.py` so transport JSON written to Claude clients is ASCII-safe:
  - added `transport_json_dumps(...)`
  - switched HTTP JSON responses and SSE `data:` payloads to `ensure_ascii=True`
- This keeps the wire payload ASCII-only while still allowing the client JSON parser to reconstruct Unicode text correctly after parsing.

### Verification completed
- Restarted the adapter service and confirmed healthy state on `http://127.0.0.1:8418`.
- Re-ran:
  - `F:/data_set_process/data_process/startup/claude_via_codex_api.ps1 -ClaudeArgs @('-p','please only output ok')`
- Verified the real launcher still returns `ok`.
- Verified streamed `/v1/messages` responses are now emitted as compact ASCII-safe JSON payloads.

### Practical note for next run
- The already-open old launcher window will still show the previous startup behavior until it is closed and reopened.
- On the next launch from `startup/Claude_Code_CodexAdapter.cmd`, the PowerShell header/profile noise should be reduced.
- Port `8418` should be treated as normal whenever `8417` is busy.

## 2026-04-14 Old desktop launcher permission upgrade simplification

### Clarification
- The user's original desktop shortcut was not pointing to the new Codex-adapter launcher.
- `C:/Users/Administrator/Desktop/Claude Driver Model Project.lnk` points to:
  - `powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File "D:\ClaudeCode\driver-model-project.ps1"`
- That script then calls:
  - `D:/ClaudeCode/claude-router.ps1`
- The old path already used `claude-code-router` (`ccr.cmd code`) rather than a pure direct official Claude path.

### Why the recent work felt overcomplicated
- The recent branch combined two different goals:
  - raise Claude's execution permissions
  - make Claude use the local Codex/OpenAI-backed adapter
- For the first goal alone, the old desktop launcher could have been modified directly with much lower complexity.

### Simplification applied
- Updated `D:/ClaudeCode/claude-router.ps1`.
- The old launcher path now injects:
  - `--permission-mode bypassPermissions`
- This means the original desktop shortcut should now open the familiar project launcher path but with higher execution permissions, without requiring the user to switch to the newer adapter-first launcher for this specific goal.

## 2026-04-14 19:18:24 Stable Claude-via-Codex adapter hardening for auth conflict, auto-restart, and port failover

### Context
- User provided a live Claude screenshot showing two concrete runtime failures on the new local adapter route:
  - auth conflict warning because both `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_API_KEY` were set
  - `Unable to connect to API (ECONNREFUSED)` during an actual Claude request
- User also asked to continue toward a more stable long-running version, preferably with PowerShell-based operational entry points to reduce antivirus friction.

### Why this was done
- The first adapter implementation had already proven the core Anthropic-style translation loop, but the operating path still had two practical weaknesses:
  1. launcher env setup triggered an avoidable Claude auth warning
  2. adapter lifecycle management was still too fragile for long-running use
- The user specifically wanted a "long-term stable" path, not just a minimal proof of concept.

### What was changed
- Expanded the core adapter protocol support in:
  - `F:/data_set_process/data_process/tools/anthropic_codex_adapter.py`
- Added/rewrote PowerShell operational scripts:
  - `F:/data_set_process/data_process/startup/claude_codex_adapter_supervisor.ps1`
  - `F:/data_set_process/data_process/startup/start_claude_codex_adapter.ps1`
  - `F:/data_set_process/data_process/startup/stop_claude_codex_adapter.ps1`
  - `F:/data_set_process/data_process/startup/claude_codex_adapter_status.ps1`
- Updated the user-facing launcher:
  - `F:/data_set_process/data_process/startup/claude_via_codex_api.ps1`

### Operational fixes
- Removed the launcher-side auth conflict source:
  - the launcher now clears `ANTHROPIC_AUTH_TOKEN`
  - and keeps only `ANTHROPIC_API_KEY` for the local Anthropic-style adapter route
- Replaced the old one-shot adapter start with a PowerShell supervisor model:
  - supervisor keeps the adapter alive in the background
  - if the child adapter process exits, supervisor restarts it
  - current state is written to `tmp/claude_codex_adapter/service_state.json`
  - supervisor lifecycle events are written to `tmp/claude_codex_adapter/supervisor.log`
- Added stop/status utilities so the route is inspectable instead of opaque.

### Stability features implemented
- Health-checked startup rather than "fire and hope":
  - `start_claude_codex_adapter.ps1` now waits for a real `/health` success before returning
- Automatic adapter restart:
  - verified by force-killing the child adapter process and observing supervisor restart it with a new PID
- Port conflict handling:
  - if the preferred port `8417` is occupied by another process, supervisor scans up to `8427`
  - the launcher reads the actual chosen port from service state instead of assuming `8417`
- Current launcher now uses the resolved runtime base URL rather than a hard-coded base URL.

### Protocol compatibility improvements
- Added wider Anthropic input compatibility in the adapter:
  - `image` blocks now support `base64`, `url`, and file-backed variants
  - `document` blocks now map into OpenAI-style file input items
  - assistant-side `thinking` blocks are tolerated and converted into reusable text context
  - `redacted_thinking` is safely ignored instead of crashing the adapter
- Existing text/tool-use/tool-result roundtrip support was preserved.

### Validation completed
1. Real launcher path now runs without the previous auth-conflict warning source in its env design.
2. Real launcher script succeeded with:
   - `F:/data_set_process/data_process/startup/claude_via_codex_api.ps1 -ClaudeArgs @('-p','请只输出ok')`
   - observed result: `ok`
3. Supervisor auto-restart was verified:
   - killed active adapter child PID
   - supervisor restarted adapter and health returned to normal
4. Port failover was verified:
   - deliberately occupied `127.0.0.1:8417` with a temporary listener
   - adapter service started on `8418`
   - real launcher inherited `http://127.0.0.1:8418`
   - Claude request still succeeded with result `ok`
5. Status inspection confirmed healthy runtime after failover:
   - `healthy = True`
   - `baseUrl = http://127.0.0.1:8418`

### Current practical conclusion
- The local Claude-via-Codex route is now materially more stable than the first version.
- The exact screenshot failure mode has been addressed at both levels:
  - auth env conflict removed
  - local adapter availability no longer depends on one fragile child process
- The operational entry path is now PowerShell-first even though the protocol translation core still lives in Python.

### Recommended next step
- Use `F:/data_set_process/data_process/startup/Claude_Code_CodexAdapter.cmd` as the normal entry point.
- Use these maintenance commands when needed:
  - start/ensure: `F:/data_set_process/data_process/startup/start_claude_codex_adapter.ps1`
  - status: `F:/data_set_process/data_process/startup/claude_codex_adapter_status.ps1`
  - stop: `F:/data_set_process/data_process/startup/stop_claude_codex_adapter.ps1`
- If the user later wants the protocol core itself also migrated fully into PowerShell, that can be treated as a second hardening phase, but the current PowerShell-led runtime layer is already enough to solve the present reliability problem.

## 2026-04-14 17:42:12 Claude bypassPermissions launcher auth diagnosis

### Context
- User reported that launching Claude from the previously provided bypass script path still produced the same abnormal behavior.
- Initial suspicion was that the launcher path or version selection remained wrong.
- Revalidation showed the launcher now correctly resolves to Claude Code `v2.1.107`, so the failure was no longer a version-selection problem.

### What was confirmed
- The effective Claude config home for the active install is `D:/ClaudeCode/home`, not `C:/Users/Administrator/.claude`.
- This was confirmed by the runtime environment variable `CLAUDE_CONFIG_DIR=D:/ClaudeCode/home` and by Claude debug logs watching `D:/ClaudeCode/home/settings.json`.
- The actual loaded user settings file is `D:/ClaudeCode/home/settings.json`.
- That file currently injects:
  - `ANTHROPIC_AUTH_TOKEN`
  - `ANTHROPIC_BASE_URL=https://aixj.vip`
- Claude debug logs also show `settingsEnv keys: ANTHROPIC_AUTH_TOKEN,ANTHROPIC_BASE_URL` and explicitly mark the host as not first-party.

### Root cause update
- The previously prepared launcher path itself is not the current blocker.
- The current blocker is that the active Claude home config injects third-party API routing, so even when the launcher uses `bypassPermissions`, model requests still go through the proxy/API-key path and fail with:
  - `429 API key 额度已用完`
- This explains why the user felt the supplied path was still "wrong": the binary path was fixed, but the runtime config source was still being loaded from a different home directory than expected.

### Evidence anchors
- `F:/data_set_process/data_process/startup/claude_bypass_permissions.ps1`
- `F:/data_set_process/data_process/startup/Claude_Code_BypassPermissions.cmd`
- `D:/ClaudeCode/home/settings.json`
- `D:/ClaudeCode/home/debug/4408a441-2e9f-4c36-be9a-66ce5527f793.txt`
- `D:/ClaudeCode/home/debug/61548998-93cc-4070-93c6-842442a37311.txt`

### Planned fix
- Back up `D:/ClaudeCode/home/settings.json`.
- Remove the conflicting third-party `env` injection from that file while preserving the non-auth permission convenience setting.
- Re-run a minimal Claude prompt through the launcher to verify:
  - no proxy host injection,
  - no API-key quota error,
  - `bypassPermissions` launcher still points to `D:/ClaudeCode/global/claude.cmd`.

### Implementation result
- Backed up the active Claude user config to:
  - `D:/ClaudeCode/home/backups/settings.json.proxy_env_backup_20260414_174244.json`
- Verified that directly removing the entire `env` block makes Claude lose all login state for this machine.
- Verified that keeping only `ANTHROPIC_AUTH_TOKEN` without `ANTHROPIC_BASE_URL` causes first-party authentication to fail with `401 Invalid bearer token`.
- Therefore the token in the old global home is not a valid official Anthropic bearer token; it is only usable through the third-party route.
- Restored the original `D:/ClaudeCode/home/settings.json` so existing proxy-based behavior is preserved and not silently broken.

### Permanent launcher correction
- Updated the project launcher to stop inheriting the global `CLAUDE_CONFIG_DIR`.
- The launcher now explicitly uses an isolated config home:
  - `D:/ClaudeCode/profiles/driver-model-bypass-official`
- Added a minimal settings file there:
  - `D:/ClaudeCode/profiles/driver-model-bypass-official/settings.json`
- This isolated profile preserves `skipDangerousModePermissionPrompt=true` but does not carry the third-party proxy env injection.
- The launcher also now prints the exact config dir it is using, so future path/debug confusion is easier to diagnose.

### Validation result
- Running the updated launcher now reports:
  - launcher = `D:/ClaudeCode/global/claude.cmd`
  - version = `2.1.107`
  - config dir = `D:/ClaudeCode/profiles/driver-model-bypass-official`
  - profile type = isolated official profile
- Auth status for the isolated profile is now:
  - `loggedIn: false`
  - `authMethod: none`
  - `apiProvider: firstParty`
- So the earlier `429 API key 额度已用完` is confirmed to be unrelated to the launcher path itself and instead comes from the third-party proxy credentials in the old global Claude home.

### Practical next step for the user
- Open the updated launcher path.
- In that Claude window, run `/login` once to complete official first-party authentication for the isolated profile.
- After that, the same launcher path should give:
  - `bypassPermissions`
  - no temp-path permission popups
  - no inherited third-party API quota routing from `D:/ClaudeCode/home`.

## 2026-04-14 17:56:00 Claude-Codex collaboration method recheck from existing project files

### Context
- User clarified that the real intention is not to force Claude onto official Anthropic login, but to let Claude use the existing Codex-side capability/method already prepared in the project.
- User specifically asked to re-check the project's own previously prepared files because a method should already exist there.

### What was inspected
- `F:/data_set_process/data_process/.claude/commands/codex-workflow.md`
- `F:/data_set_process/data_process/.claude/commands/codex-run.md`
- `F:/data_set_process/data_process/.claude/commands/codex-handoff.md`
- `F:/data_set_process/data_process/.claude/agents/codex-coordinator.md`
- `F:/data_set_process/data_process/.claude/settings.json`
- `D:/ClaudeCode/codex-bridge/claude-codex-entry.ps1`
- `D:/ClaudeCode/codex-bridge/run-codex.cmd`
- `D:/ClaudeCode/codex-bridge/invoke-codex.ps1`

### What was found
- The repository already contains a concrete Claude -> Codex bridge workflow.
- The key command path is:
  - `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/ClaudeCode/codex-bridge/claude-codex-entry.ps1" "<PROMPT>"`
- Inside that entry script, Claude hands the task to:
  - `D:/ClaudeCode/codex-bridge/invoke-codex.ps1`
- That bridge script then launches the local executable:
  - `D:/ClaudeCode/codex-bridge/codex.exe`
- The handoff includes:
  - project root binding
  - `--full-auto` execution support
  - automatic extraction of Codex's final message
  - automatic requirement to append substantial project progress into `reports/project_progress_master.md`

### Important interpretation
- The existing project method is **not** "Claude's own backend model is replaced by Codex/OpenAI API".
- The existing method is:

## 2026-04-14 用户提供 GPT 深度研究报告并纳入项目判断

### Context
- 用户提供了外部生成的深度研究文档：`C:/Users/Administrator/Downloads/极限工况驾驶员响应预测研究报告（严格围绕 driver response forecasting 与指定主题）.docx`，要求先阅读并将其要点记录到项目主日志。
- 本轮主线背景是：maintained 训练脚本 `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 已完成一轮最小修复并成功跑通完整训练；当前真正待判断的问题，已经从“代码能否运行”转向“tail flattening / 长值固定不变是否真的缓解”。

### Why this was logged
- 该报告不是随意的泛泛综述，而是围绕本项目当前 driver response forecasting 问题、指定协议、已有 evidence 和 active training code 生成的定向研究材料。
- 它对后续实验主线的真正价值，不在于直接替代仓库内证据，而在于：
  1. 帮助把当前 failure mode 重新整理成更清晰的研究框架；
  2. 提醒哪些方向可以立即吸收进评估与实验设计；
  3. 明确哪些建议因 protocol / evidence / code 口径不一致，只能作为研究参考，不能未经核实直接照搬。

### What was read and extracted
- 通过解包 `.docx` 正文并通读全文，确认该报告显式基于三类材料组织结论：
  1. protocol：`primary_control_v2_context_full2s`，3s history / 2s future、subject-level fixed split、online_inputs_only=true；
  2. evidence：Step4 决策摘要、Q1_fast 归因、boundary 分析、代表性失败案例、attribution 宽表与 prediction sequence artifacts；
  3. code：`future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`、diagnostics 脚本及 event-conditioned 相关评估/绘图脚本。
- 报告还显式标注了几个重要不一致：
  1. protocol 是 primary-only，但 evidence 对应 allphase / mixed phase 口径；
  2. evidence 中预测序列是 2 通道（`steer_rel`, `speed_delta`），而当前 maintained training/diag 脚本仍带 3 通道（`steer/yawrate/ay`）痕迹；
  3. online-visible-only 红线下，future curvature / road preview 若来自真实未来或离线地图，就应被视为 student 不可直接使用的高优先级审计项。

### Main findings worth adopting
1. **报告对当前核心 failure mode 的定性与现有项目判断高度一致**：
   - 当前最主要的问题更像是 `tail amplitude / shape mismatch`，而不是系统性的 boundary / peak timing 全局恶化。
   - 这与当前会话中围绕“后段发平 / 长值固定不变”的诊断方向一致，也支持继续优先从 shape-preserving 的最小修改推进，而不是先做大结构改造。
2. **报告把不同 failure subtype 分开处理的思路是有价值的**：
   - `Q1_fast` 更像 tail shape / amplitude 主导；
   - `single_lobe` 的 boundary shift 更像 time-shift / phase problem；
   - 这提示后续实验和指标不应只依赖单一总损失或总 RMSE，而应区分 shape 问题与 timing 问题。
3. **报告对评估体系的建议可直接吸收**：
   - 后续 run 不能只看 overall RMSE；
   - 应逐步固定为“结构 + 相位 + 形状”三类指标闭环：
     - 结构：reversal exist、zero-cross / turning count、first reversal time；
     - 相位：peak time、boundary shift；
     - 形状：tail MAE / tail RMSE、tail shape corr、tail amplitude。
4. **报告对 protocol 红线的强调是正确且必须保留的**：
   - subject-level fixed split；
   - online-visible inputs only；
   - privileged 信息若使用，应明确限制为 teacher / distillation 路径，而不是 student 推理直接消费。

### Limits / caveats explicitly recorded
- 该报告不能被直接视为“当前仓库事实”的最终替代来源，原因包括：
  1. protocol、evidence、code 口径混杂，尤其 primary-only vs allphase 尚未彻底统一；
  2. 2 通道 evidence 与 3 通道 maintained 脚本之间存在分支差异；
  3. 报告中提到的 future curvature、road preview、80/20 split 痕迹、某些损失权重失效等判断，仍需以当前 active repo 实际代码再次核实，不能只因报告提及就直接当成定论。
- 因此，这份材料适合用于“研究方向校准”和“实验/评估设计参考”，不适合作为未经本地复核就直接执行的大改方案。

### Practical interpretation for current work
- 对当前 maintained 脚本主线，最值得立即采用的不是一口气上 DILATE / Soft-DTW / 多教师蒸馏 / GroupDRO / PCGrad 全套增强，而是先吸收三件事：
  1. 继续坚持 single-change / minimal-change 原则，避免同时叠过多机制后无法归因；
  2. 把 run 结果评估从单纯 loss / RMSE 扩展为 tail / peak / reversal 的固定闭环；
  3. 在后续任何更强 shape-time loss 或蒸馏方案前，先确认当前 successful run 的 `pred_vs_gt_example_*.png` 是否已显示 tail flattening 得到实质缓解。
- 这与本轮已完成的第一阶段最小修复（接通 reversal sample weighting、让 train 侧真正看到 reversal region、并成功完整跑通 40 epoch）是同向的，而不是相互冲突的新路线。

### Recommended next step
1. 先基于当前已成功跑完的 run 产物，做图像级与切片级确认：判断 tail flattening 是否真的减轻，而不是只看总指标。
2. 同步开始补固定评估指标脚本，优先把 tail / peak / reversal 三类指标固化下来。
3. 若图上仍显示尾段偏平，再进入第二轮最小增强：小幅打开 `W_REVSEQ`、必要时再增强 `W_PEAKTIME`，继续保持不改 split / anchor / horizon / 主结构。
4. 在进入更激进的 shape-time loss 或 teacher distillation 增强前，先单独做一次 online-only / future-input 审计，避免形成“离线更好但不合规或不可部署”的假改进。
  - Claude remains the coordinator shell
  - Claude delegates bounded tasks to local Codex through the bridge
  - Codex executes and returns the result
- In other words, this is an orchestration bridge, not a backend protocol replacement.

### Why this matters
- If the user's goal is "let Claude continue the workflow but actually use Codex capability to execute work", then the project already has the right mechanism.
- If the user's goal is "make Claude itself directly consume Codex/OpenAI API as if Codex were Anthropic backend", the currently prepared files do **not** implement that.

### Current practical conclusion
- The already prepared and repo-consistent path is to use the built-in collaboration commands and bridge:
  - Claude plans / hands off
  - Codex executes through `codex.exe`
  - project logging remains enforced
- This is likely the intended method referred to by the user's existing files.

### Recommended next step
- Stop treating the problem as a Claude API-provider replacement problem.
- Re-anchor on the repo's existing bridge workflow and use the prepared Claude commands / bridge scripts as the standard path for Claude-Codex cooperation in this repository.

## 2026-04-14 18:27:00 Anthropic-style local adapter to Codex/OpenAI gateway implemented and validated

### Context
- User explicitly chose the new scheme: do not rely only on the existing Claude -> Codex orchestration bridge.
- The requested target was stronger: add a true local adapter layer so Claude can keep speaking Anthropic-style API while the actual backend consumption goes through the user's Codex/OpenAI side.

### Why this was done
- The existing repository method already supported Claude planning + Codex execution, but that is an orchestration bridge, not a backend protocol replacement.
- The user wanted a more direct path where Claude itself can point to a local Anthropic-compatible endpoint and have that endpoint forward requests into the Codex/OpenAI side.

### What was inspected first
- `D:/ClaudeCode/codex-bridge/invoke-codex.ps1`
- `C:/Users/Administrator/.codex/config.toml`
- `C:/Users/Administrator/.codex/auth.json`
- local gateway behavior at `http://localhost:8317/v1`

### Key findings before implementation
- Codex on this machine is already configured to use a local OpenAI-style gateway:
  - `base_url = http://localhost:8317/v1`
  - `wire_api = responses`
- The local gateway accepts:
  - `GET /v1/models`
  - `POST /v1/responses`
- Streaming responses from that gateway expose usable text and function-call events even though the non-stream `output` field is empty.
- Therefore the adapter should consume the backend through streaming `responses` events and reconstruct Anthropic-style text / tool-use output from those events.

### What was implemented
- Added the core adapter:
  - `F:/data_set_process/data_process/tools/anthropic_codex_adapter.py`
- Added a dedicated local Claude profile for this route:
  - `F:/data_set_process/data_process/startup/claude_codex_profile/settings.json`
- Added an adapter server starter:
  - `F:/data_set_process/data_process/startup/start_claude_codex_adapter.ps1`
- Added a one-click Claude launcher that points Claude to the local adapter and keeps `bypassPermissions`:
  - `F:/data_set_process/data_process/startup/claude_via_codex_api.ps1`
  - `F:/data_set_process/data_process/startup/Claude_Code_CodexAdapter.cmd`

### Adapter behavior
- Exposes Anthropic-style endpoints:
  - `GET /health`
  - `GET /v1/models`
  - `POST /v1/messages`
  - `POST /v1/messages/count_tokens`
- Accepts the query form Claude actually uses:
  - `/v1/messages?beta=true`
- Accepts the preflight/probe method Claude actually uses:
  - `HEAD /`
- Converts Anthropic request structure to OpenAI Responses input:
  - `system` -> `instructions`
  - message text blocks -> `input_text` / `output_text`
  - Anthropic tool definitions -> OpenAI function tools
  - `tool_use` / `tool_result` history -> `function_call` / `function_call_output`
- Converts backend streamed events back into Anthropic-style output:
  - text -> `content_block_delta` with `text_delta`
  - function calls -> `tool_use` blocks with `input_json_delta`
  - final stop -> `message_delta` + `message_stop`

### Validation that was completed
1. Direct adapter health check succeeded:
   - `GET http://127.0.0.1:8417/health`
2. Anthropic-style non-stream text request succeeded:
   - returned text block `ok`
3. Anthropic-style non-stream tool-call request succeeded:
   - returned `tool_use` with `add(a=2,b=3)`
4. Anthropic-style full roundtrip succeeded:
   - user request -> assistant `tool_use` -> user `tool_result` -> final assistant text `5`
5. Anthropic-style stream response succeeded:
   - emitted `message_start -> content_block_start -> content_block_delta -> content_block_stop -> message_delta -> message_stop`
6. Real `claude.cmd` was pointed at the adapter and succeeded in `--print` mode:
   - with `ANTHROPIC_BASE_URL=http://127.0.0.1:8417`
   - with local proxy token
   - result returned `ok`

### Important practical conclusion
- The local Anthropic-style adapter path is now real and working on this machine.
- Claude can now be pointed at the local adapter and indirectly consume the Codex/OpenAI-side local gateway instead of the previous third-party Anthropic proxy route.

### Current limitations / caveats
- This route is validated first in local request tests and Claude `--print` mode.
- The adapter is intentionally text + tool-use focused; no broad multimodal/document coverage was implemented yet beyond basic image-block conversion support in the request mapper.
- The startup route is designed for this machine's current local gateway assumption:
  - backend base URL `http://localhost:8317/v1`
  - backend API key placeholder `sk-dummy`
  - backend model `gpt-5.4`
- If the local Codex/OpenAI gateway configuration changes later, only the launcher/adapter config needs to be updated; Claude-side workflow can stay the same.

### Recommended next step
- Use `F:/data_set_process/data_process/startup/Claude_Code_CodexAdapter.cmd` as the new preferred entry path when the goal is "Claude shell, but backend routed through the local Codex/OpenAI gateway".
- If a later failure appears, inspect:
  - `F:/data_set_process/data_process/tmp/claude_codex_adapter/adapter_stdout.log`
  - `F:/data_set_process/data_process/tmp/claude_codex_adapter/adapter_stderr.log`
- If needed later, expand the adapter with stronger multimodal block coverage or richer model/config selection, but the core Anthropic-to-Codex/OpenAI translation loop is already established.

## 2026-04-14 Maintained v5.8 protocol-safe eval-closure patch

### What was done
- Read the active maintained training script and the existing protocol-safe full-run artifacts for `TRAIN_V5_8_PROTOCOL_SAFE_20260413_174639`, focusing on `evaluate_and_plot`, dataset sample indexing, run-end evaluation, `run_config.json`, `training_summary.json`, `figures/test_metrics.json`, `figures/test_metrics_by_reversal.json`, `figures/test_state_dump.csv`, and `selected_samples_with_split.csv`.
- Patched `F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` so the maintained path can now export event-level response evidence in addition to the existing aggregate metrics.
- Added event-level metric export for three missing acceptance signals: steer trend correlation distribution, primary response peak-time error distribution, and reverse-prediction rate on explicit-direction samples.
- Added a dataset-to-metadata linkage via persistent `sample_index`, allowing exported event tables to trace back to `sample_key`, `subject_id`, `event_idx`, `anchor_idx`, and split.
- Added an eval-only CLI path (`--eval-run-dir`) that rebuilds the protocol-safe dataset with the active source, loads an existing best checkpoint, and writes prefixed derived artifacts into the existing run directory without overwriting historical files.
- Performed syntax validation with `F:/python3.11/python.exe` using in-memory `compile(...)`; this passed. Also confirmed the default shell `python` is 3.5.2, which is too old for this codebase and explains why naive `py_compile` fails before any patch-specific validation.

### Why it was done
- The maintained v5.8 protocol-safe baseline already had split audit, config, basic RMSE/MAE, reversal metrics, and state dump, but it still lacked the three event-level evidence classes required for final acceptance on extreme-condition behavior credibility.
- The immediate slice objective was to close the evaluation loop first, with priority on re-evaluating the existing full run rather than forcing another expensive full retrain.

### What was found
- The existing `evaluate_and_plot` path exported only aggregate `test_metrics.json`, reversal summaries, plots, and a state dump keyed by local dataset `idx`; it did not preserve a stable global sample identifier, so event-level tracing back to protocol metadata was incomplete.
- The missing gap was not in split protocol or checkpoint availability. The main blocker was evaluation plumbing: no event-level response metric computation and no eval-only entry point.
- The existing full run already contains both `best_model_v5_8_protocol_safe.pth` and `model_rollpeak_transformer_v5_8_protocol_safe.pth`, which is enough to support a no-retrain evaluation replay path.
- The final checkpoint already stores feature and normalization statistics (`feat_mean/std`, `y_mean/std`, `curve_mean/std`, `ctx_mean/std`, teacher-state normalization terms), so the eval-only path can reuse them when present.
- Validation detail:
  - `python -m py_compile ...` with the default shell interpreter failed immediately because the machine-default `python` is `Python 3.5.2`.
  - `F:/python3.11/python.exe -c "compile(...)"` succeeded, so the patched file is syntactically valid under a modern interpreter.

### Recommended next step
- In the main Claude session, run the new eval-only command against `TRAIN_V5_8_PROTOCOL_SAFE_20260413_174639` with a unique tag so the existing run receives only new derived artifacts.
- After the replay finishes, inspect the new prefixed files under that run’s `figures/` directory, especially:
  - `<tag>_test_event_response_metrics.json`
  - `<tag>_test_event_response_details.csv`
  - `<tag>_test_metrics_basic.json`
  - `<tag>_test_metrics.json`
  - `<tag>_test_metrics_by_reversal.json`
  - `<tag>_test_state_dump.csv`
- Use the resulting event table to review low-correlation samples, large peak-time-error samples, and explicit-direction reverse-prediction samples before deciding whether the next slice should target evaluation-only acceptance or a new modeling intervention.
- Claude 主会话第一次按建议执行 eval-only replay 时，先用了 `conda run --no-plugins -n predict_2 ...`，但当前 bash shell 中 `conda` 未注入 PATH，导致命令以 `/usr/bin/bash: conda: command not found` 失败；这属于执行入口问题，不是脚本或补丁本身失败。
- 随后改为直接调用已知可用解释器 `D:/ProgramData/anaconda3/envs/predict_2/python.exe` 继续执行同一条 eval-only replay，避免在当前环境里再依赖 shell 级 `conda` 命令可见性。
- 第二次 replay 又暴露了一个环境兼容问题：当前 `predict_2` 环境中的 PyTorch 版本已采用 `torch.load(..., weights_only=True)` 新默认值，而历史 full run checkpoint 里除了 `state_dict` 外还保存了 `numpy` 数组统计量（如 `feat_mean/std`、`y_mean/std`、teacher-state 标准化项等），导致 eval-only 路径在 `_load_torch_payload()` 处因 `UnpicklingError` 失败。这同样不是协议或模型逻辑错误，而是“新 torch 默认 + 旧 checkpoint 结构”之间的加载兼容问题。
- 已在 active source 中把 `_load_torch_payload()` 改为优先显式使用 `torch.load(..., weights_only=False)`，并保留对旧版 torch 不支持该参数时的回退逻辑；由于这里加载的是本项目自己训练生成的本地 checkpoint，属于可控可信来源，因此这一修正满足当前 eval-only replay 的安全边界。
- 第三次 replay 继续前进后，又暴露了另一个 checkpoint 结构细节：`best_model_v5_8_protocol_safe.pth` 实际是纯 `OrderedDict state_dict`，而不是带 `state_dict` 键的包装 dict；eval-only 路径此前按包装结构取 `best_payload["state_dict"]`，因此在样本重建完成后于 `model.load_state_dict(...)` 处触发 `KeyError`。
- 已直接核实该 best checkpoint 的真实结构，并在 active source 中补成兼容写法：若 `best_payload` 自带 `state_dict` 键则取其值，否则直接把载入对象当作 state_dict 本体。这样既兼容历史 best checkpoint，也不影响将来可能保存成包装 dict 的情况。
- 修正后第三次 replay 已成功完成，派生产物已落到：
  - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260413_174639/figures/baseline_recalc_20260414_test_event_response_metrics.json`
  - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260413_174639/figures/baseline_recalc_20260414_test_event_response_details.csv`
  - 以及同前缀的 `test_metrics / test_state_dump / test_metrics_by_reversal / pred_vs_gt_example_*` 等派生文件。
- 本次完整 baseline 补打得到的关键验收面板为：
  - RMSE：`steer=0.6557`、`yawrate=0.1339`、`ay=2.1182`
  - steer 趋势相关性：`median=0.7883`、`p10=0.0564`、`p90=0.9602`
  - 主响应峰值时间误差：`median=0.2575 s`、`p90=1.205 s`
  - 明确方向响应样本反向预测率：`3 / 93 = 3.23%`
- 对照当前阶段验收要求，可直接得出：
  - `explicit_direction_reverse_rate` 已达标（`3.23% <= 5%`）
  - 其余指标仍未达标，其中最强短板是主响应峰值时间误差长尾（`p90=1.205 s`），其次是 `rmse_steer / rmse_yawrate / rmse_ay` 以及趋势相关性中位数略低于门槛。
- 因此下一轮最有价值的切片不应再泛化为“继续整体降 RMSE”，而应优先围绕“降低主响应峰值时序长尾、收敛显著延迟样本，同时不破坏已较低的反向预测率”来做 failure-focused 诊断与改动设计。

### Current practical reading after the full baseline replay
- The protocol-safe maintained baseline is now audit-complete at the event-response level.
- The model is not dominated by widespread sign reversals; explicit-direction reverse predictions are already relatively rare.
- The main remaining credibility problem is timing: many samples still peak too late or at the wrong local response phase, and this long-tail latency is much worse than the median alone suggests.
- This means the next slice should prioritize latency / peak-alignment failure modes over broad untargeted loss tuning.

### Next decision point after the replay
- Start a bounded failure-analysis slice on `baseline_recalc_20260414_test_event_response_details.csv`, especially:
  - worst peak-time-error samples
  - low-correlation samples
  - whether late peaks cluster by road type / subject / event level / anchor policy
- Use that evidence to choose the next modeling move; do not jump straight to a new full training run without first localizing the dominant latency mechanism.

### End state for this replay cycle
- Event-response evaluation closure for the current protocol-safe full baseline is complete.
- The next cycle should be diagnosis-first, not blind retraining.
- Keep the derived baseline artifacts as the fixed comparison point for subsequent interventions.

### 补充记录（2026-04-14，protocol-safe baseline 峰时长尾 failure-focused 诊断收口）
- 执行主体：混合协作（Claude 发起只读诊断，subagent 收口证据，Claude 综合判断）
- Why：baseline 补评估已经确认当前不是“大量反向预测”问题，而是主响应峰值时间误差长尾过大；在进入下一轮建模改动前，需要先判断长尾到底更像系统性早峰/晚峰、少数极端样本、还是局部错峰/相位歧义，否则直接调 loss 权重很容易继续盲试。
- What was checked：
  1. 读取 `baseline_recalc_20260414_test_event_response_metrics.json`、`baseline_recalc_20260414_test_event_response_details.csv`、`baseline_recalc_20260414_test_metrics_by_roadtype.json`、`baseline_recalc_20260414_test_metrics_by_reversal.json`。
  2. 对照 `selected_samples_with_split.csv` 确认 test 被试分布与元信息切片。
  3. 回看 active source 中与 anchor、peak timing aux、主任务损失和评估主峰定义相关的实现位置，重点核对训练侧的 peak-time 目标与评估侧的主峰定义是否一致。
- What was found：
  1. 长尾不像“整体都晚一点”或“整体都早一点”，更像主峰选错了哪个局部峰/哪个相位；详情表里反复出现 GT 峰落在窗口末端 `398/399` 而预测落到 `155/156/131/8`，以及反向模式（GT 较早而预测跳到 `398/399`）。这更像固定错误峰位或相位模板坍缩，而不是单纯连续平移。
  2. 长尾与 low-corr 并不高度重合。多条高误差样本同时仍有较高趋势相关性（例如 `corr≈0.82/0.92/0.95` 仍对应 `>1.2s` 的峰时误差），说明问题不是“趋势没学到”，而是“整体走势能跟住，但主峰挑错了”。
  3. 长尾也不是反向预测主导。当前显式方向样本反向预测率仅 `3/93=3.23%`，且大量长尾样本属于 `is_explicit_direction=0`、`is_reverse_pred=0`。因此当前最强痛点是时序/相位可信度，而不是方向符号错误。
  4. curve 整体 RMSE 确实高于 straight，但峰时长尾并不只是 curve 问题；大量 `>1.2s` 的坏例实际来自 `anchor_source_applied=steer_rate_peak80_first`、`is_curve_applied=0` 的 straight 切片。这意味着主战场更像 straight-anchor 下的多阶段/晚峰响应歧义，而不是“只要优化 curve 就够了”。
  5. 一个非常关键的结构错位被确认：训练里的 `W_PEAKTIME` 当前约束的是 `steer-rate` 的软峰时，而补评估与验收看的是 `abs(steer)` 的主幅值峰时间。对于多阶段响应样本，这两者并不等价；因此下一轮若只是泛调 `W_PEAKTIME`，很可能只是更强地对齐了 steer-rate 峰，而不一定改善最终验收里的主 steer 峰时间。
- Recommended next step：
  1. 下一轮最优先不要泛调一堆 loss 权重，而应先做一个最小、可审计的建模切片：让 timing supervision 更直接面对评估里使用的“主 steer 幅值峰时间”。
  2. 同时补一个轻量分析/导出表，把 `is_curve_applied × is_explicit_direction`、以及 `gt_peak_idx/pred_peak_idx` 分桶后的峰时误差统计直接固化出来，用于验证长尾是否主要来自 straight-anchor 的相位歧义与固定错误峰位。
  3. 在这两步证据补齐前，不建议直接开新的完整训练矩阵；下一轮应是“最小 timing-target 对齐改动 + 有界验证”，而不是继续盲目全局调权。

### 补充记录（2026-04-14，maintained v5.8 timing-target 对齐最小补丁与 GPU smoke 验证）
- 执行主体：Claude
- Why：上一轮 baseline failure-focused 诊断已经收口到一个关键结构错位：训练侧 `W_PEAKTIME` 约束的是 `steer-rate` 软峰时，但补评估与当前验收口径真正看的，是 `abs(steer)` 主幅值峰时间。若继续只泛调 `W_PEAKTIME` 或其它 loss 权重，很可能只是更强地对齐了导数峰，而不是改善最终关注的主响应峰时，因此先做一个最小、可审计、且不改 split / anchor / horizon / online-only 的 timing-target 对齐补丁。
- What was done：
  1. 回读 active maintained 训练脚本 `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 中与 `W_PEAKTIME`、`_soft_peak_time(...)`、训练/验证 loss 闭环，以及评估主峰定义相关代码，确认训练端此前在 `:3485-3495` / `:3541-3550` 一直使用 `abs(diff(steer))` 作为 peak-time aux 的监督对象，而评估端在 `compute_event_response_artifacts(...)` 中使用的是 `argmax(abs(steer))` 的主峰时间。
  2. 在同一 active source 中新增 `_primary_peak_profile(steer_seq)`：先按每条样本真实/预测序列自身的 `abs(steer)` 最大值位置取主峰符号，再把整条 steer 序列按该主符号对齐并裁成非负 profile，作为“主 steer 幅值峰证据”。这样 timing supervision 面对的是和评估定义更接近的目标，而不是导数峰。
  3. 把训练与验证闭环中的 peak-time aux 从：
     - `steer_rate_pred = _diff1(steer_pred).abs()`
     - `steer_rate_true = _diff1(steer_true).abs()`
     - `peak_pred/peak_true = _soft_peak_time(steer_rate_*, temp)`
     改为：
     - `peak_profile_pred = _primary_peak_profile(steer_pred)`
     - `peak_profile_true = _primary_peak_profile(steer_true)`
     - `peak_pred/peak_true = _soft_peak_time(peak_profile_*, temp)`
     其余主任务、reversal、state distill、anchor、dataset build、response horizon、split policy 均保持不变。
  4. 同步把 `W_PEAKTIME` 与 `PEAK_TEMP_FRAC` 注释更新为“primary steer amplitude peak timing alignment”，避免后续 run 比较时误把该辅助项继续理解成 steer-rate timing。
  5. 先做语法验证：默认按用户要求优先尝试 `predict2` 环境 Python，但本机实际可见环境目录名仍是 `predict_2`，`D:/ProgramData/anaconda3/envs/predict2/python.exe` 不存在；随后改用已安装可用解释器 `D:/ProgramData/anaconda3/envs/predict_2/python.exe -m py_compile ...` 完成语法检查并通过。
  6. 继续在 GPU 上做一次有界 smoke 训练验证：
     - 命令环境：`D:/ProgramData/anaconda3/envs/predict_2/python.exe`
     - 设备：`cuda` (`CUDA_VISIBLE_DEVICES=0`)
     - 输出根：`tmp/protocol_safe_runs`
     - smoke 设置：`DRIVER_MODEL_SMOKE=1`，实际取 train/val/test=`238/10/8`，`epochs=2`，`batch_size=32`
     - 运行目录：`F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260414_111310`
- What was found：
  1. 这次最小补丁本身可正常编译，且真实 GPU smoke 训练从数据构建到 best checkpoint 保存、test 指标导出、event-response 指标导出都完整跑通，说明“把 timing supervision 从 steer-rate 峰切到主 steer 幅值峰”没有破坏 maintained 主线的基本训练闭环。
  2. smoke run 在极小样本上给出的 event-response 指标为：
     - `primary_peak_time_error_sec median = 1.08s`
     - `primary_peak_time_error_sec mean = 0.9831s`
     - `steer_trend_corr median = 0.1580`
     - `explicit_direction_reverse_rate = 1/1 = 100%`
     这些数值明显不能当作效果判断依据，但至少说明新的 peak-time aux 路径已真正参与训练与评估，不再停留在静态改码层面。
  3. smoke 的主价值是“验证补丁路径与运行路径是活的”，不是给出性能结论。当前 test 仅 `8` 个样本、显式方向仅 `1` 个样本，因此无论 RMSE 还是 reverse rate 都不具备研究决策意义。
  4. 运行过程中再次确认本机当前实际可执行环境名是 `predict_2` 而不是字面 `predict2`；这与用户口头命名不完全一致，但指向的是同一套已配好的 CUDA 环境。后续若继续自动执行，最稳妥的做法仍是优先使用这一路径型解释器，避免 shell 侧 `conda` 名称/激活差异再次卡住。
- Recommended next step：
  1. 先不要根据这次 smoke 的小样本指标做建模判断；它只说明补丁可运行。
  2. 下一步应在同一补丁上做一次更有判别力的有界验证，优先顺序为：
     - 先复用现有 protocol-safe full baseline 的数据构造方式，做一个中等规模短跑或短 epoch 验证；
     - 同时把 `is_curve_applied × is_explicit_direction` 与 `gt_peak_idx/pred_peak_idx` 分桶统计表补出来，检查 timing-target 对齐后长尾是否真的从 straight-anchor 相位歧义处回落；
     - 若方向对，再决定是否值得上完整训练。
  3. 如果后续要继续严格遵守“运行代码默认走 predict2 + GPU”的用户口径，需要把当前机器上的实际解释器映射记成：用户所说 `predict2` 在本机对应的可用路径是 `D:/ProgramData/anaconda3/envs/predict_2/python.exe`。

### 补充记录（2026-04-14，timing-target 对齐补丁后的中等规模短跑与峰时长尾分桶复核）
- 执行主体：Claude
- Why：最小 timing-target 对齐补丁已经完成并通过小 smoke 路径验证，但要继续朝用户设定目标推进，不能只停在“代码能跑”。需要先用比最小 smoke 更有判别力的短跑看补丁后的方向是否值得继续，再用分桶统计检查长尾是否真的从此前怀疑的 fixed wrong-peak / phase ambiguity 位置回落。
- What was done：
  1. 直接在 active maintained 主线 `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 上，用本机可用 CUDA 解释器 `D:/ProgramData/anaconda3/envs/predict_2/python.exe` 启动一轮中等规模短跑。
  2. 运行参数保持为 protocol-safe 路径下的 smoke-mode 子采样，但把规模提高到：
     - `DRIVER_MODEL_SMOKE=1`
     - `DRIVER_MODEL_SMOKE_MAX_SAMPLES=1024`
     - `DRIVER_MODEL_SMOKE_EPOCHS=4`
     - `DRIVER_MODEL_SMOKE_BATCH_SIZE=32`
     - `CUDA_VISIBLE_DEVICES=0`
     - 输出根：`tmp/protocol_safe_runs`
  3. 本轮实际运行目录为 `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260414_131206`，实际 split 子样本为 train/val/test=`969/29/26`。
  4. 运行完成后，继续读取：
     - 新短跑 `test_event_response_details.csv`
     - 正式 protocol-safe baseline `baseline_recalc_20260414_test_event_response_details.csv`
     并用 `predict_2` Python 生成对照分桶表：
     - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260414_131206/figures/timing_alignment_bucket_summary.csv`
  5. 分桶维度至少包括：
     - `anchor_source_applied`
     - `is_curve_applied`
     - `is_explicit_direction`
     - `anchor_source_applied × is_explicit_direction`
     - `gt_primary_peak_idx × pred_primary_peak_idx` 粗分桶
- What was found：
  1. 这轮短跑本身是稳定完成的，训练/验证/导出链路没有被 timing-target 补丁破坏：
     - `Epoch1 Train=2.2721 Val=1.0899`
     - `Epoch4 Train=1.6346 Val=1.1436`
     - best checkpoint、test metrics、event-response details、plots 全部正常产出。
  2. 但从判别性结果看，这一轮还**不能**支持“timing-target 对齐已经把主问题打下来”：
     - `rmse_steer=0.7546`
     - `trend corr median=0.1191`
     - `primary_peak_time_error median=0.6125s`
     - `primary_peak_time_error p90=1.7675s`
     - `explicit_direction_reverse_rate=0.5 (2/4)`
     这些值距离正式目标非常远，当然也受 test 仅 `26` 条样本、显式方向仅 `4` 条样本影响，不能当正式结论。
  3. 分桶结果说明：这轮短跑没有呈现出“straight-anchor 相位长尾已明显回落”的信号，反而仍保留错峰/错相位特征，只是由于样本很小，当前更像是“方向未证成”，而不是“已经明确失败到可彻底否定”。
     - baseline full：
       - overall peak median `0.2575s`
       - `steer_rate_peak80_first` median `0.2100s`
       - `roll_peak` median `0.3900s`
     - timing patch short：
       - overall peak median `0.6125s`
       - `steer_rate_peak80_first` median `0.4750s`
       - `roll_peak` median `0.6900s`
     - 说明至少在这轮 26 样本短跑里，没有观察到此前怀疑切片上的回落，straight 与 curve 两侧都没出现令人信服的改善信号。
  4. `gt_peak_bin × pred_peak_bin` 粗分桶继续出现了明显的错峰模式，而不是单纯平移：
     - `50-149 -> 0-49`、`150-249 -> 0-49`、`350-399 -> 0-49` 仍反复出现
     - 同时也有 `150-249 -> 250-349`、`250-349 -> 350-399` 这类晚峰跳转
     这与前一轮 baseline diagnosis 的“固定错误峰位 / 相位模板坍缩”判断是一致的，并没有因为这次最小 timing-target 对齐就自动消失。
  5. 一个重要更新是：此前只读诊断把主战场优先押在 straight-anchor 相位歧义上，但这轮小规模实跑分桶显示 `roll_peak` 切片也没有变好，甚至本轮中位峰时误差比 `steer_rate_peak80_first` 更差。这意味着下一轮不能再把问题过窄地理解成“只要修 straight-anchor timing 就够了”；更像是当前补丁还没有真正抓住导致主峰选错的核心监督形态。
- Recommended next step：
  1. 不要把这一版最小 timing-target 对齐直接升级到完整训练；当前证据不支持这么做。
  2. 下一轮应从“只改 peak-time aux 目标”转向“让主峰监督不只约束时间，还更直接约束主峰位置/主峰方向/主峰幅值附近的局部形态”，优先考虑更强但仍有界的主峰对齐监督，而不是继续只在同一 soft peak-time 标量上打转。
  3. 在进入下一轮改动前，保留这次 `timing_alignment_bucket_summary.csv` 作为反证材料：它说明单纯把 timing aux 从 steer-rate 峰切到主 steer 峰，并没有自动消除错峰长尾。

### 补充记录（2026-04-14，第二版更强主峰监督短跑结果）
- 执行主体：Claude
- Why：上一轮最小 timing-target 对齐短跑已经说明，单纯把 peak-time aux 从 steer-rate 峰切到主 steer 峰还不够；因此继续在 maintained 主线上加一层更直接的主峰局部监督，避免模型只学到一个一维 peak-time 标量，却仍在主峰位置和局部形态上选错峰。
- What was done：
  1. 在 `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 中新增：
     - `W_MAINPEAK_LOCAL = 0.10`
     - `_main_peak_local_loss(steer_pred, steer_true)`
  2. 该 loss 的作用方式是：以 GT 主峰位置为中心，按 GT 主峰符号把预测和真值对齐到同一主方向后，对主峰附近局部区域做更强监督，包含：
     - 主峰附近局部形态 L1
     - GT 主峰位置处的峰值幅度对齐
     - 主峰附近错误符号能量惩罚
  3. 将该项同时接入训练与验证 loss 闭环，并完成 `predict_2` 环境下语法检查。
  4. 继续用与上一轮相同的中等规模短跑配置做 GPU 对照验证：
     - 解释器：`D:/ProgramData/anaconda3/envs/predict_2/python.exe`
     - 设备：`cuda`
     - `DRIVER_MODEL_SMOKE_MAX_SAMPLES=1024`
     - `DRIVER_MODEL_SMOKE_EPOCHS=4`
     - `DRIVER_MODEL_SMOKE_BATCH_SIZE=32`
     - 运行目录：`F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260414_144617`
     - 实际 split 子样本：train/val/test=`969/29/26`
- What was found：
  1. 第二版更强主峰监督链路是可运行的：训练、验证、best checkpoint、event-response 指标导出全部正常完成，没有把 maintained 主线跑坏。
  2. 与上一轮 timing-target 对齐短跑（`TRAIN_V5_8_PROTOCOL_SAFE_20260414_131206`）相比，这一版的变化是“有一点点朝对的方向，但幅度很小，而且远远不够支撑升级到正式训练”：
     - `rmse_steer`: `0.7546 -> 0.7627`（略差）
     - `rmse_yawrate`: `0.13433 -> 0.13399`（几乎持平）
     - `rmse_ay`: `2.67686 -> 2.67086`（几乎持平）
     - `trend corr median`: `0.1191 -> 0.1363`（略好）
     - `primary_peak_time_error median`: `0.6125s -> 0.5150s`（有改善）
     - `primary_peak_time_error p90`: `1.7675s -> 1.7675s`（无改善）
     - `explicit_direction reverse_rate`: `0.5 -> 0.5`（无改善）
  3. 因此当前更准确的判断不是“第二版成功了”，而是：
     - 它比上一轮“只改 timing 标量”的方案更接近正确方向；
     - 但改善只停留在小样本短跑上的有限中位数层面；
     - 对我们最在意的长尾（P90）和方向一致性几乎没有触动。

### 补充记录（2026-04-14，显式 main_peak_bin / direction 结构监督接入、报错修复与短跑结论）
- 执行主体：Claude
- Why：前两轮改动都还停留在 trajectory-level 的 timing / local-shape loss 上，虽然第二版对中位峰时有一点改善，但对长尾 P90 和 reverse rate 基本没有撬动；因此下一轮不再继续围绕同一个标量 timing loss 打转，而是直接复用仓库里已有的 `event_targets.py` / `event_head.py` 路径，把更结构化的主峰位置监督（`main_peak_bin` + `main_peak_direction`）接进 maintained v5.8 主线，测试“显式事件目标”是否比隐式轨迹 loss 更能约束错峰问题。
- What was done：
  1. 在 active maintained 脚本 `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 中引入已有事件目标基础设施：
     - `from event_targets import EventTargetConfig, sequence_to_event_targets`
     - `from event_head import EventHead`
  2. 新增结构监督配置与辅助函数：
     - `W_EVENT_PEAKBIN = 0.08`
     - `EVENT_TARGET_CONFIG = EventTargetConfig(future_len=FUTURE_LEN, bin_size=20)`
     - `build_event_peak_targets(...)`
     - `compute_event_peakbin_loss(...)`
     - `compute_event_peakdir_loss(...)`
  3. 扩展 `MultiTaskFutureWithCurveDataset`，让样本除原有 trajectory / state / reversal 标签外，额外携带：
     - `event_peak_bin`
     - `event_peak_dir`
  4. 在模型中新增 `self.event_head = EventHead(...)`，并让 `forward(...)` 在原有 `y_hat, z_veh, rev_logit` 之外额外返回 `event_logits`。
  5. 训练与验证 loss 闭环同时接入：
     - `loss_event_peakbin = CE(main_peak_bin_logits, peak_bin_target)`
     - `loss_event_peakdir = CE(main_peak_direction_logits, peak_dir_target)`
     - 总 loss 新增 `W_EVENT_PEAKBIN * (loss_event_peakbin + 0.5 * loss_event_peakdir)`
  6. 第一次 GPU 短跑时暴露接线错误：训练循环已开始读取 `batch["event_peak_bin"] / batch["event_peak_dir"]`，但主训练 `collate_fn` 尚未把这两个字段真正打进 batch，导致 `KeyError: 'event_peak_bin'`。
  7. 已回读主训练 `collate_fn` 并补齐缺失字段，将：
     - `event_peak_bin_b`
     - `event_peak_dir_b`
     正式加入返回 batch dict；随后用 `D:/ProgramData/anaconda3/envs/predict_2/python.exe -m py_compile ...` 做语法验证并通过。
  8. 重跑同尺度有界 GPU 短训时，又发现一个执行入口问题：该脚本没有 `--smoke` CLI 参数，smoke 实际由环境变量控制。第一次重跑因误传 `--smoke` 直接在 argparse 阶段失败；随即改为正确命令：
     - `DRIVER_MODEL_SMOKE=1`
     - `DRIVER_MODEL_SMOKE_MAX_SAMPLES=1024`
     - `DRIVER_MODEL_SMOKE_EPOCHS=4`
     - `DRIVER_MODEL_SMOKE_BATCH_SIZE=32`
     - 解释器：`D:/ProgramData/anaconda3/envs/predict_2/python.exe`
     - 设备：`cuda`
  9. 正确重跑后的运行目录为：
     - `F:/数据集处理/data_process/datasetprocess/多模态数据/程序运行结果/TRAIN_V5_8_PROTOCOL_SAFE_20260414_161654`
     - 实际 split 子样本：train/val/test=`969/29/26`
- What was found：
  1. 这条“显式事件目标”路线现在已经不是停留在改码层面，而是完成了从 dataset → collate → model → loss → eval 导出的完整闭环，说明结构监督接入本身已经真实跑通。
  2. 但从这轮短跑结果看，它**没有形成值得立刻放大的正向突破**。相对上一轮“第二版更强主峰监督”短跑（`TRAIN_V5_8_PROTOCOL_SAFE_20260414_144617`），当前显式事件目标版表现为：
     - `rmse_steer`: `0.7627 -> 0.8161`（更差）
     - `rmse_yawrate`: `0.1340 -> 0.1184`（更好）
     - `rmse_ay`: `2.6709 -> 2.6452`（略好）
     - `trend corr median`: `0.1363 -> -0.0041`（明显更差）
     - `primary_peak_time_error median`: `0.5150s -> 0.5850s`（变差）
     - `primary_peak_time_error p90`: `1.7675s -> 0.9450s`（明显改善）
     - `explicit_direction reverse_rate`: `0.5 -> 0.5`（无改善）
  3. 这说明当前结构监督路线呈现出一个很关键但也很不稳定的信号：
     - 它可能确实在压缩峰时长尾（P90 从 `1.7675s` 降到 `0.9450s`）；
     - 但同时明显伤害了整体趋势相关性和部分 trajectory 质量，特别是 `rmse_steer` 与 `trend corr median`。
     因此更像是“结构监督把主峰位置拉回来了一些，但方式过硬/过强，导致轨迹整体形态被牺牲”，而不是已经达到可直接升级正式训练的平衡状态。
  4. 与固定 full baseline 相比，这一版仍然远未达到目标：baseline full 的 `trend corr median=0.7883`、`peak median=0.2575s`、`reverse_rate=3.23%`，当前短跑即使在峰时 P90 上看起来比前两版短跑好，也仍不能说明它能在正式尺度上超越 baseline 或满足用户设定的多指标验收口径。
- Recommended next step：
  1. 不要立刻把这版显式 `main_peak_bin` 监督升级到完整训练；当前证据不支持。
  2. 下一轮应优先做“减弱结构监督副作用”的有界切片，而不是回退到完全没有结构监督。最直接的最小动作是：
     - 下调 `W_EVENT_PEAKBIN`
     - 保留长尾压缩信号，同时避免 trajectory 主任务被 event CE loss 压住
  3. 若继续沿这条路线推进，下一轮重点应直接观察：
     - `peak_time_error p90` 是否还能保持低位
     - `trend corr median` 能否从当前塌陷状态回升
     - `rmse_steer` 是否回到至少不劣于上一轮 local-mainpeak 版的区间
  4. 因此当前最合理的判断不是“显式事件目标失败”，而是：它第一次给出了“可能打到长尾”的结构信号，但权重/耦合方式明显还不平衡，下一轮应该做的是**减力校准**，而不是继续加码。
  4. 这说明当前主问题仍不是一个靠局部再加一层轻量 peak loss 就能收口的问题。主峰错选的机制可能更深，仍可能与整体序列生成偏好、晚峰/早峰模板坍缩、或 anchor 后不同阶段响应模式的表征能力不足有关。
- Recommended next step：
  1. 不把第二版直接推到完整训练。当前证据不足。
  2. 下一轮应转向更结构性的有界方向，而不是继续在同一层 loss 上做第三个小修小补：优先考虑把主峰监督从“附加 loss”推进到更显式的事件/主峰目标建模，或回到 teacher/state/response pattern 表征层面重新检查是否缺少区分不同相位模式的判别信号。
  3. 保留第二版结果作为证据：它说明“更强主峰监督”方向比单纯 peak-time 标量稍有希望，但远未达到能替代 baseline 或接近最终目标的程度。

## [2026-04-14 继续推进] 后续记录待续

### 补充记录（2026-04-14，驾驶员反应时延建模文献检索与 Zotero 导入）
- 执行主体：Claude
- Why：用户明确要求围绕当前主线“驾驶员反应时延建模”寻找可用论文，并按仓库既有 ScholarAIO / Zotero 流程执行下载与入库，而不是只给泛泛推荐列表；因此需要在合规边界内做一次面向该方向的定向检索、筛选和实际导入。
- What was done：
  1. 先读取仓库级 `CLAUDE.md` 与 `reports/codex_academic_zotero_workflow.md`，确认本次应优先使用本地 `tools/academic_search_to_zotero.py`，并遵守“不绕过付费墙/登录墙、不能把搜索结果当作已导入结果、只有命令输出确认才算导入成功”的边界。
  2. 使用本地检索命令对多组英文主题词做定向搜索：
     - `py -3.11 ./tools/academic_search_to_zotero.py search "driver reaction time modeling" --limit 8`
     - `py -3.11 ./tools/academic_search_to_zotero.py search "driver response latency prediction" --limit 8`
     - `py -3.11 ./tools/academic_search_to_zotero.py search "driver braking reaction time prediction" --limit 8`
     - `py -3.11 ./tools/academic_search_to_zotero.py search "driver take-over reaction time modeling" --limit 8`
     - `py -3.11 ./tools/academic_search_to_zotero.py search "EEG driver reaction time prediction" --limit 8`
  3. 基于检索结果，优先挑选与当前主线更接近且存在开放获取可能的论文做 DOI 导入到 Zotero 集合 `研究生论文/自动导入`，命令包括：
     - `py -3.11 ./tools/academic_search_to_zotero.py import-doi "10.1177/0361198119842114" --collection "研究生论文/自动导入" --download-pdf`
     - `py -3.11 ./tools/academic_search_to_zotero.py import-doi "10.1109/tr.2017.2778754" --collection "研究生论文/自动导入" --download-pdf`
     - `py -3.11 ./tools/academic_search_to_zotero.py import-doi "10.1109/tetci.2018.2881229" --collection "研究生论文/自动导入" --download-pdf`
     - `py -3.11 ./tools/academic_search_to_zotero.py import-doi "10.1088/2057-1976/adbf25" --collection "研究生论文/自动导入" --download-pdf`
     - `py -3.11 ./tools/academic_search_to_zotero.py import-doi "10.1186/s13638-020-1639-2" --collection "研究生论文/自动导入" --download-pdf`
  4. 随后按用户要求继续排查失败原因，并实际启动 `startup/start_zotero_translation_server.ps1`；脚本通过 Docker 拉起 `zotero/translation-server:latest`，虽然启动日志显示镜像最终下载成功，但本机后续对 `127.0.0.1:1969` 的访问仍被拒绝，因此 translation-server 在当前会话里没有真正进入可用服务状态。
  5. 在该前提下，仍继续重试此前最关键的 3 篇失败论文：
     - `py -3.11 ./tools/academic_search_to_zotero.py import-doi "10.1109/tetci.2018.2881229" --collection "研究生论文/自动导入" --download-pdf`
     - `py -3.11 ./tools/academic_search_to_zotero.py import-doi "10.1088/2057-1976/adbf25" --collection "研究生论文/自动导入" --download-pdf`
     - `py -3.11 ./tools/academic_search_to_zotero.py import-doi "10.1186/s13638-020-1639-2" --collection "研究生论文/自动导入" --download-pdf`
- What was found：
  1. 检索层面，当前与“驾驶员反应时延建模”最贴近、且对现有极限工况/反应建模主线有参考价值的结果，主要落在三类：
     - 行为/认知反应时建模：
       - `10.1016/j.aap.2020.105889` — *Predicting driver reaction time and deceleration: Comparison of perception-reaction thresholds and evidence accumulation framework*
       - `10.1177/0361198119842114` — *Modeling Driver Take-Over Reaction Time and Emergency Response Time using an Integrated Cognitive Architecture*
       - `10.1016/j.aap.2015.02.023` — *What determines the take-over time? An integrated model approach of driver take-over after automated driving*
     - EEG / 脑状态驱动的反应时预测：
       - `10.1109/tetci.2018.2881229` — *Electroencephalogram Based Reaction Time Prediction With Differential Phase Synchrony Representations Using Co-Operative Multi-Task Deep Neural Networks*
       - `10.1088/2057-1976/adbf25` — *An investigation of pre-stimulus eeg for prediction of driver reaction time*
       - `10.1109/tr.2017.2778754` — *Detection of Driver Vigilance Level Using EEG Signals and Driving Contexts*
     - 跟驰 / 多传感器行为预测近邻：
       - `10.1186/s13638-020-1639-2` — *A driver’s car-following behavior prediction model based on multi-sensors data*
  2. 第一轮实际 Zotero 入库成功、且命令已明确返回 `status=imported` 的有两篇：
     - `10.1177/0361198119842114`，Zotero `item_key=NXEMCMEA`
     - `10.1109/tr.2017.2778754`，Zotero `item_key=G7VKGFNP`
     但这两篇在第一轮里都没有拿到 PDF 附件。
  3. 重试后新增成功导入并成功拿到 PDF 的有两篇：
     - `10.1109/tetci.2018.2881229`，Zotero `item_key=F6JSCNF3`，PDF 路径：`E:\文献\paper_download\2019_Electroencephalogram Based Reaction Time Prediction With Differential Phase Synchrony Representations Using Co-Operative Multi-Task Deep Neural Networks.pdf`
     - `10.1088/2057-1976/adbf25`，Zotero `item_key=VRSIGQ86`，PDF 路径：`E:\文献\paper_download\2025_An investigation of pre-stimulus eeg for prediction of driver reaction time.pdf`
  4. `10.1186/s13638-020-1639-2` 在重试时返回 `status=exists`，说明 Zotero 中已存在该条目：
     - Zotero `item_key=L45ESP46`
     - 去重原因：`dedupe_reason=doi`
     - 这意味着它不是“新导入成功”，而是系统检测到已有同 DOI 条目，因此未重复创建；本次输出也没有确认新增 PDF 附件。
  5. 一个重要现象是：虽然 translation-server 在当前会话里仍不可访问，但重试后其中两篇论文依然成功完成了 metadata import + PDF retrieval。这说明本次成功更主要依赖于：
     - Crossref/OpenAlex/Unpaywall 元数据恢复正常
     - 以及候选 OA / repository PDF 链接本轮网络可达
     而不是 translation-server 真正提供了增强能力。
  6. 当前仍未解决的点主要有两个：
     - `127.0.0.1:1969` 仍 refused connection，translation-server 启动状态和端口可用性存在环境问题；
     - 第一轮已导入的 `10.1177/0361198119842114`、`10.1109/tr.2017.2778754` 仍没有 PDF 附件。
  7. 随后又专门针对这两篇已入 Zotero 但缺附件的核心论文重新执行了一轮 `import-doi --download-pdf` 重试：
     - `10.1177/0361198119842114`
     - `10.1109/tr.2017.2778754`
     两条命令都在去重阶段直接返回 `status=exists`，分别命中已存在条目 `NXEMCMEA` 与 `G7VKGFNP`，没有进入新的 PDF 候选下载 / 选择阶段，因此这轮补拉 **没有新增 PDF**。
  8. 这说明当前 CLI 的去重逻辑是：一旦按 DOI 命中已有 Zotero 条目，就直接返回 `exists` 并结束，而不是继续把 `--download-pdf` 当作“为已有条目补附件”的动作来执行。因此用同一条 `import-doi` 命令重跑，并不能自动给旧条目补 PDF。
  9. 随后又直接读取 `tools/academic_search_to_zotero.py`，确认脚本内部已经具备可复用的 PDF 下载与附件上传原语：`build_pdf_candidate_list(...)`、`download_pdf_from_candidates(...)`、以及 `zot.attachment_simple(...)`；缺的只是一个对“已存在条目补附件”的显式命令入口，而不是底层能力本身不存在。
  10. 在不修改脚本源码的前提下，改为用 `py -3.11` 临时加载该模块并直接复用其内部函数，对已有 Zotero 条目执行“候选 PDF 解析 → 下载 → attachment_simple 上传”路径，成功为以下两篇补上 PDF：
      - `10.1177/0361198119842114` / `NXEMCMEA`
        - 下载源：`http://hdl.handle.net/10012/18482`
        - 本地文件：`E:\文献\paper_download\2019_Modeling Driver Take-Over Reaction Time and Emergency Response Time using an Integrated Cognitive Architecture.pdf`
        - Zotero attachment key：`IU3N9HBX`
      - `10.1109/tr.2017.2778754` / `G7VKGFNP`
        - 下载源：`http://xplorestaging.ieee.org/ielx7/24/8305671/08240627.pdf?arnumber=8240627`
        - 本地文件：`E:\文献\paper_download\2018_Detection of Driver Vigilance Level Using EEG Signals and Driving Contexts.pdf`
        - Zotero attachment key：`QBGDF8XJ`
  11. 上传返回里 `attachment_result.unchanged` 而不是 `success`，更像是 Zotero/pyzotero 对该批量接口的返回格式特点，而不是失败；因为结果里已经明确给出了 `attachment` 条目、`parentItem`、`contentType=application/pdf` 和新附件 `key`，可视为附件已挂到对应父条目下。
  12. 随后继续读取这 4 篇核心论文 PDF，并围绕当前 maintaned 主线真正关心的 5 个维度做了文献映射：反应时定义、峰时/延迟监督、EEG 状态表征、个体差异/上下文、以及对当前 extreme-condition driver response 建模的可迁移启发。核心结论如下：
      - `10.1177/0361198119842114`（QN-ACTR）最重要的价值不是神经网络结构，而是把反应时明确拆成“感知-任务切换-决策-动作执行”串行过程，并把 `take-over reaction time / emergency response time` 定义成从提示或事件出现到首次有效转向/制动动作之间的时间；这对当前项目最直接的启发是：不要只把延迟看成一个单点峰时误差，而应考虑把反应过程分段或状态化建模。
      - `10.1109/tr.2017.2778754` 把 vigilance 明确 operationalize 为对 lead-car brake event 的 RT，并证明仅用 EEG 不如“EEG + driving context”稳定；其新增 road curve 上下文后可带来 `2–5%` 精度增益和 `30–80 ms` 更小误差。对当前项目的直接启发是：生理/EEG 状态估计若不和道路/事件上下文联立，很容易把“任务更难导致反应变慢”误判成“驾驶员状态更差”。
      - `10.1109/tetci.2018.2881229` 的核心不是单纯回归 RT，而是把“RT 预测”作为主任务、把“alert vs drowsy”作为辅助任务，做 cooperative multi-task learning；其最佳 MTDNN 相比 SVR 实现 `RMSE -15.49%`、`MAPE -27.15%`、`CC +10.13%`。这对当前项目非常关键：如果目标是更稳地预测 response timing / response shape，单任务纯轨迹回归可能不够，辅助状态任务能帮助 backbone 学到与时延相关的神经状态结构。
      - `10.1088/2057-1976/adbf25` 直接研究 pre-stimulus EEG 对 RT 预测，结论非常贴近当前任务：`2 s` pre-stimulus 窗口最优，1D CNN 比经典回归更强，最佳配置把 MAE 从 `0.51 s` 压到 `0.36 s` 左右，而且作者明确采用 subject-independent 评估。它对当前项目的直接启发是：历史窗长度不能只凭习惯定，短时反应预测里“刺激前 2 秒左右的状态窗”可能比更长窗更稳；同时，应尽量保留 subject-independent / subject-split-safe 的评估口径。
      - 四篇合起来给出的统一方向不是“继续堆更复杂轨迹头”，而是：把当前问题看成“事件发生前的生理/神经准备状态 + 事件/道路上下文 + 反应阶段切换”的联合建模问题，而不是单一序列回归问题。
  13. 基于这 4 篇的综合判断，当前仓库模型主线最可迁移的 4 条文献启发可收口为：
      - **反应时定义层**：把 response latency 明确锚到事件出现/提示出现到首次有效控制动作，而不是只看最终主峰位置；
      - **监督设计层**：除了主轨迹损失，还应显式监督 timing / state / alertness 或至少引入与 response stage 有关的辅助目标；
      - **输入设计层**：EEG/physio 状态必须与 road/event context 联立，避免把任务难度混进个体状态；
      - **评估层**：优先坚持 subject-level split 与 subject-independent 判断，不要让状态表征在 sample-random 条件下看起来过于乐观。
- Recommended next step：
  1. 这轮之后，围绕“驾驶员反应时延建模”的核心 Zotero 入库已经明显前进：至少已有 4 篇明确在库，并且当前核心包里 4 篇都已拿到实际 PDF，可直接进入阅读与整理阶段。
  2. 若后续还要继续给“已存在条目”补附件，最稳妥的长期动作不是再手写临时脚本，而是给 `tools/academic_search_to_zotero.py` 增加一个显式子命令，例如 `attach-doi-pdf` 或 `attach-item-pdf`。
  3. 若要让这轮文献工作直接服务当前建模主线，下一步最值当的不是继续扩论文数量，而是把上述文献启发具体翻译成一版面向 maintained 主线的建模建议表：哪些对应 teacher-state，哪些对应 timing aux，哪些对应 event context，哪些对应 protocol-safe 评估要求。

### 补充记录（2026-04-13，目标驱动自治推进模式确认）
- 执行主体：Codex
- Why：用户明确提出，希望后续不再以“做一点就停下来等下一步指令”的方式推进，而是改成“用户只给必须达到的目标，其间 Claude / Codex 可以自行查资料、读文献、改代码、跑实验、持续推进，直到达成目标才停”；同时用户明确给出关键边界：不能删除其文件，其它权限可开放，并且每次执行都必须经过既有协作工作流并写入项目日志。
- What was checked：
  1. 读取 `reports/project_progress_master.md` 顶部速览与近期记录，确认当前项目主线、阶段瓶颈与既有协作约定。
  2. 读取仓库根 `README.md`，确认项目已经正式规定“凡有实质性进展，先写详细日志，再给压缩总结”，且日志字段至少包括“谁做的 / 做了什么 / 为什么做 / 得到了什么 / 下一步建议”。
  3. 读取 `CLAUDE.md`，确认当前仓库的 maintained code 范围、split/protocol guardrail、高风险失败模式，以及“实质性工作必须先追加到 `reports/project_progress_master.md`”这一硬约束。
  4. 读取 `.claude/commands/codex-workflow.md` 与 `.claude/agents/codex-coordinator.md`，确认仓库现有 Claude -> Codex 协作工作流本身就是“Claude 规划 -> Codex 执行 -> Claude 总结”的结构，并且要求 handoff 中写明范围、约束、验收标准、验证方式与日志义务。
- What was found：
  1. 从仓库现有规则看，用户想要的工作模式是可落地的，而且大部分制度其实已经存在，不需要另起一套全新的流程。
  2. 最接近用户诉求的正式执行形态应是：
     - 用户只给一个明确目标；
     - 同时给出验收口径和不能碰的红线；
     - Claude 负责收口任务、拆成有边界的执行 brief；
     - Codex 负责具体实现、检查、调试、实验、必要时查资料；
     - 每次有实质进展都先补写 `project_progress_master.md`；
     - 未达目标前，不因需要新的“小步指令”而频繁中断。
  3. “不能删文件”可以直接作为默认 guardrail 固化为后续执行边界：后续若无用户单独明确批准，不删除用户现有文件、不做破坏性清理、不用“先删再重建”的方式处理工作区。
  4. 需要对用户保持如实说明的一点是：我们可以承诺“持续推进、不把你当流程调度器、遇阻先自行查证和试错”，但不能诚实地承诺“任何目标都必然一次做到”；当目标暂时无法达到时，应给出基于证据的停点判断、已验证失败路径、以及下一条最优推进路线，而不是假装已经完成。
  5. 对当前项目而言，这种模式尤其适合后续主线推进，例如：
     - 将 maintained v5.8 主线真正改到 protocol-safe 的 `subject-level fixed split`；
     - 在不破坏 horizon / anchor / online-only 前提下验证 response-state-aware 主线；
     - 对 teacher latent、reversal 标签稀疏性、道路列来源等阻塞项做连续闭环处理。
- Recommended next step：
  1. 后续由用户给出单个“必须达到”的目标时，建议同时固定三件事：最终验收标准、不可触碰的红线、可接受的资源边界。
  2. 默认执行口径可以直接采用：
     - 禁删用户文件；
     - 必须遵守既有 Claude / Codex 协作工作流；
     - 每次实质进展先写 `reports/project_progress_master.md`；
     - 优先从 `datasetprocess/final_code` 和 active protocol 出发；
     - 需要资料时允许先查本地文档、再查文献或外部资料；
     - 遇到真正高风险分叉时才停下来请求用户拍板，而不是每一步都问。
  3. 当前最自然的首个目标候选不是继续泛泛“优化模型”，而是把已有近期结论真正落成一个 publication-safe 的下一步，例如：在 maintained 主线中完成 `subject-level fixed split` 落地，并给出 smoke/短跑验证与日志闭环。

### 补充记录（2026-04-13，目标驱动自治推进模式正式落地到仓库规则）
- 执行主体：Codex
- Why：上一条记录已经确认这套模式在原则上可行，但如果只停留在口头说明，后续新会话或新代理仍可能回到“做一步问一步”的旧节奏；因此需要把规则正式写进仓库文档、模板和 `.claude` 命令入口，让它成为可复用、可追溯、可执行的仓库默认能力。
- What was done：
  1. 新增正式工作流文档 `reports/goal_driven_autonomous_workflow.md`，系统定义目标驱动自治推进模式的最小输入、默认边界、标准执行闭环、必须停下来问用户的条件、日志要求以及与现有命令体系的关系。
  2. 新增用户可直接复用的任务模板 `reports/goal_driven_target_template.md`，把“目标 / 验收 / 红线 / 资源边界 / 可接受停点”整理成最小版与推荐版输入格式，并附上适合本项目的 subject-split 示例。
  3. 新增 `.claude/commands/goal-driven-workflow.md`，把“目标驱动自治推进”做成独立命令入口，明确要求先读 `CLAUDE.md` 和新工作流文档，并在需要时循环走 Claude scope -> Codex execute -> Claude review 的有界切片闭环。
  4. 更新 `README.md`，把这套模式暴露给仓库使用者，明确说明现在支持“只给目标，不想每一步都重新下指令”的协作方式，并给出正式说明与模板入口。
  5. 更新 `CLAUDE.md`，新增 `Goal-Driven Autonomous Mode` 小节，把“不删用户文件、持续推进、先写详细日志、只在高风险分叉时停下来问”的口径提升为仓库级约束，而不是临时聊天约定。
  6. 更新 `.claude/commands/codex-workflow.md`、`.claude/commands/codex-run.md` 与 `.claude/agents/codex-coordinator.md`，把目标驱动模式接入现有 Claude / Codex 协作入口，使其在 handoff、direct run、coordinator planning 三个层面都能识别并保留相同 guardrail。
- What was found：
  1. 现有仓库结构本身已经有足够强的协作基础，这次工作的重点不是另造新流程，而是把已有规则“显式化、模板化、命令化”。
  2. 把规则同时落到 `README.md`、`CLAUDE.md`、`reports/` 文档和 `.claude/commands` 之后，后续无论是人读仓库、Claude 读规则还是通过命令入口调用，都会更容易保持同一套执行口径。
  3. 这次落地后，“只给目标、持续推进、禁删文件、先写日志再总结”已经不再只是单次对话承诺，而是仓库里有文档、有模板、有命令入口的正式工作方式。
- Recommended next step：
  1. 后续用户若要启动这套模式，优先按 `reports/goal_driven_target_template.md` 给出目标 brief，可最小化来回确认成本。
  2. 下一次真正执行模型或数据任务时，应直接按这套模式推进一次完整实例，优先候选仍然是 maintained 主线的 `subject-level fixed split` 正式落地与验证。
  3. 若后续发现某类高风险任务仍容易触发反复确认，可再把对应停点规则继续补充回 `reports/goal_driven_autonomous_workflow.md`，让模式迭代得更稳。

### 补充记录（2026-04-13，目标驱动自治推进模式进一步细化）
- 执行主体：Codex
- Why：用户进一步澄清，所谓“真正高风险分叉”不只是泛指代价大或技术复杂，更重要的是不要偏离当前项目的核心研究目标；同时用户明确不希望我们在单个小问题上反复钻牛角尖，并指出在路径已经想清楚时，没有必要机械地总是先做很小规模 smoke 才能进入正式训练。
- What was done：
  1. 更新 `reports/goal_driven_autonomous_workflow.md`，新增“本项目当前目标口径”小节，把当前主目标明确写成“预测极限工况下的驾驶员行为及车辆状态趋势”，并把“趋势相似、延时不能过大、不能系统性大偏差、不能大量反向预测”写成当前阶段更实用的验收口径。
  2. 在同一文档中重写“真正高风险分叉”的定义，明确将“偏离主目标、为局部小问题牺牲趋势相似性、延时和方向一致性、在次要小问题上耗费大量时间”列为优先级更高的高风险情形。
  3. 在同一文档中加入“尝试预算”规则：同一小问题、同一修复假设或同一类小旋钮，连续尝试达到 `3-4` 次后，原则上应停止惯性深挖，除非出现新的强证据或新的机制假设。
  4. 更新训练规模规则：把原先偏保守的“长作业前优先做 smoke / 短跑验证”收敛为“训练规模按风险和把握度决定”；若目标、路径、输出位置和主要风险已经充分论证清楚，可以直接进入完整训练 / 完整测试。
  5. 更新 `reports/goal_driven_target_template.md`，让模板里直接支持填写“趋势相似 / 延时不要太大 / 不要反向预测”和“尝试预算”，并把示例从“1 次 smoke + 1 次短跑”改成更贴近用户当前偏好的“完整训练和完整测试”。
  6. 更新 `CLAUDE.md` 与 `.claude/commands/goal-driven-workflow.md`、`.claude/commands/codex-workflow.md`，把这三条细化规则接到仓库级约束和命令入口层，避免后续执行时口径回退。
  7. 更新 `README.md` 与 `reports/project_progress_master.md` 顶部速览 / 日期索引 / 专题索引，使这次细化不仅存在于细则文件中，也能从项目主日志顶部快速回看。
- What was found：
  1. 用户这次补充后，目标驱动模式从“更少打断用户”进一步升级成了“始终围绕主目标推进”的版本，执行判断标准明显更清楚了。
  2. 这条新口径特别适合当前项目，因为它把“趋势对不对、延时大不大、是否出现反向预测”提升到了比单个局部指标更重要的位置，更符合极限工况驾驶员行为预测当前阶段的真实目标。
  3. 引入 `3-4` 次尝试预算后，可以显著降低在单个小问题上过度消耗时间的风险，也能逼迫我们在局部问题迟迟不收敛时更早切换方向。
  4. 放宽“必须先极小 smoke”的默认习惯后，后续执行会更贴近用户实际诉求：该保守时保守，该直接上完整训练时就直接上，不再让流程本身拖慢项目推进。
- Recommended next step：
  1. 后续所有目标驱动任务，优先按“主目标趋势对齐 -> 关键误差形态 -> 是否值得继续同点深挖”的顺序判断，而不是被单个局部问题牵着走。
  2. 下一次开始正式模型推进时，如果主线路径已经清楚，允许直接把完整训练 / 完整测试作为默认计划，而不是自动回到极小 smoke。
  3. 若后续真的在某个局部问题上达到第 `3-4` 次尝试仍未明显改善，应在日志里显式标记“达到尝试预算，切换方向”，把这一规则真正执行起来，而不只是写在文档里。

### 补充记录（2026-04-13，maintained v5.8 protocol-safe split 改造与正式训练落地）

#### 做了什么
- 读取并核对以下关键文件，确认本轮正式口径必须使用 `protocol_primary_control_v2_context_full2s` 的固定 subject split，而不是 maintained v5.8 里原有的 sample-level shuffle + 80/20：
  - `F:/data_set_process/data_process/CLAUDE.md`
  - `F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
  - `F:/data_set_process/data_process/datasetprocess/final_code/model/training/protocol_primary_control_v2_context_full2s/protocol_config.json`
  - `F:/data_set_process/data_process/datasetprocess/final_code/model/training/protocol_primary_control_v2_context_full2s/frozen_subject_split.json`
  - `F:/data_set_process/data_process/datasetprocess/final_code/model/training/protocol_primary_control_v2_context_full2s/sample_manifest.csv`
  - `F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_speed_subjectsplit_masked.py`
  - `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_primary_v2_context_full2s_baseline.py`
- 在 maintained 主线源码 `F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 中完成主线改造：
  - 为每个样本保留 `subject_id`、`vehicle_file`、`event_idx`、`anchor_idx`、`anchor_source_applied`、`maintained_anchor_policy`、`is_curve_applied` 等元信息。
  - 删除原 `main()` 中的全局 shuffle + 80/20 split，改为从协议固定 split 显式构造 `train_idx / val_idx / test_idx`。
  - 将 `feat_mean/std`、`y_mean/std`、`curve_mean/std / curve_thr`、`ctx_mean/std`、`base_mu/base_sd`、PCA teacher-state 拟合、latent `z_mu/z_sd` 全部改为严格只用 `train_idx` 拟合。
  - 将验证集与测试集正式分开：训练和 early stopping 只看 `val`，训练结束后再对 `test` 做最终评估。
  - 重写 smoke 采样逻辑，改为先按 protocol split，再在各 split 内采样；第二次修正后，进一步保证 smoke 下每个 split 内每个 subject 至少保留 1 个样本，避免某个 val/test subject 被抽空。
  - 为 `run_config.json` 补充协议与审计字段：`protocol_config_path`、`protocol_version`、`split_policy_expected/applied`、`split_source`、各 split subject/sample count、`smoke_mode`、`smoke_sampling_policy`、`teacher_state_fit_split`、`teacher_state_fit_sample_count`、`standardization_fit_split`、`curve_threshold_fit_split`、`anchor_source_expected`、`anchor_source_applied`、`maintained_anchor_policy`。
  - 新增轻量审计产物：`split_audit.json`、`split_subject_counts.csv`、`split_sample_counts.csv`、`selected_samples_with_split.csv`、`val_metrics.json`、`training_summary.json`。
- 运行层面做了两轮执行：
  - 第一次 smoke 因原默认 `RESULT_ROOT` 指向无权限的中文历史结果目录而失败，随后未改源码硬编码，只通过运行时环境变量 `DRIVER_MODEL_RESULT_ROOT=F:/data_set_process/data_process/tmp/protocol_safe_runs` 解决输出权限问题。
  - 第一次 smoke 通过后发现一个细节问题：虽然不存在跨 split 泄漏，但在极小预算下 `val` 的某个 subject 会被抽空；随后修正 smoke 采样器并重新运行 smoke。
  - 之后用实际可用环境 `predict_2`（而不是文档默认的 `predict2`）完成一轮正式训练，保存 `best_model_v5_8_protocol_safe.pth`，并对 test 做最终评估。

#### 为什么这样做
- maintained v5.8 原本的 sample-level shuffle + 80/20 是正式协议违规，并且把 `val=test` 混在一起，会导致：
  - train/val/test 被试泄漏风险；
  - 标准化、teacher-state、latent 拟合等统计量从非 train 数据吸收信息；
  - best model 和最终 test 报告口径不独立；
  - smoke 结果不能用于验证 protocol-safe 主线入口是否真实成立。
- 本轮核心目标不是“结果一定更高”，而是把 maintained 主线第一次变成 protocol-safe 的正式训练入口，并拿到一轮可复现实验结果与 split 审计。

#### 发现了什么
- maintained v5.8 旧逻辑里，随机切分和 `[:n_train]` 前缀切片确实同时污染了：
  - `feat_mean/std`
  - `y_mean/std`
  - `curve_mean/std` 与 `curve_thr`
  - `ctx_mean/std`
  - `base_mu/base_sd`
  - PCA teacher-state 拟合
  - latent `z_mu/z_sd`
- protocol-safe smoke 最终版本运行目录：
  - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260413_175832`
  - smoke split 审计结果：
    - `train_subjects` = 12/12，与协议一致
    - `val_subjects` = 3/3，与协议一致
    - `test_subjects` = 3/3，与协议一致
    - overlap 全空
    - smoke sample counts = train 177 / val 8 / test 7
    - `teacher_state_fit_split=train`, `teacher_state_fit_sample_count=177`
- 正式训练运行目录：
  - `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260413_174639`
  - full split 审计结果：
    - train subject 完全等于协议：`byx, gzj, lx, lxy, rjy, txj, yyl, yzy, zdq, zt, zx, zxy`
    - val subject 完全等于协议：`hzh, jy, xst`
    - test subject 完全等于协议：`cwh, gf, tyy`
    - overlap 全空
    - full sample counts = train 4184 / val 517 / test 528
    - `teacher_state_fit_split=train`, `teacher_state_fit_sample_count=4184`
    - best epoch = 20, early stopping at epoch 28
    - `val_metrics.json`: RMSE all 1.2179, steer 0.6538, yawrate 0.1843, ay 1.9970
    - `test_metrics_basic` / `figures/test_metrics.json`: RMSE all 1.2826, steer 0.6557, yawrate 0.1339, ay 2.1182
- maintained v5.8 与协议 `split_summary.csv` 的样本数差异很大，不是 split 错了，而是样本构建口径本来就不同：
  - 协议 `split_summary.csv`：train 970 / val 241 / test 248，总计 1459
  - maintained v5.8 protocol-safe：train 4184 / val 517 / test 528，总计 5229
  - 差异原因：
    - maintained v5.8 仍使用本脚本自己的样本构建逻辑，基于 `_events_v312.csv`，并筛 `STRONG_LABELS`
    - 协议口径来自 `v400_context` 主事件 manifest，且 `anchor_source=trigger_idx`
    - maintained anchor 仍是 `curve->roll_peak; straight->steer_rate_peak80_first`，与协议 `trigger_idx` 不同
    - 因此本轮修复的是 split plumbing 和 train-only 统计口径，不是让 maintained builder 复制 protocol manifest 的样本定义
- teacher-state 在 protocol-safe split 下仍能正常跑通与收敛：
  - 模式：`pca_latent`
  - 维度：4
  - smoke 和正式训练都成功完成 teacher-state 拟合与下游训练，没有因为 train-only split 收紧而失效
- 运行环境方面，文档默认写的是 `predict2`，但本机实际存在且可用的是 `D:/ProgramData/anaconda3/envs/predict_2/python.exe`；本轮所有 smoke / full run 都实际使用 `predict_2`。

#### 推荐下一步
- 把这次 protocol-safe 版 maintained v5.8 作为之后所有同线训练的正式入口，禁止再回到随机 sample split。
- 若要与 `protocol_primary_control_v2_context_full2s` 的 1459 样本口径做一一公平比较，下一步不要再动 split，而是单独决定是否要把 maintained builder 的事件源 / anchor 定义对齐到 protocol manifest：
  - 事件源从 `_events_v312.csv` 对齐到 protocol 使用的 `v400_context`
  - anchor 从当前 maintained 的 `roll_peak / steer_rate_peak80_first` 对齐到 protocol 的 `trigger_idx`
- 在论文或汇报中如实说明：cross-subject protocol-safe 结果如果比旧随机 split 更难看，这是更真实的泛化难度，不应回退到随机 split 来“修复”成绩。

## [2026-04-13 16:06:39] Codex - Maintained teacher-state extension read-only safety review

### What was done
- Performed a read-only review of the maintained mainline training script `F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`.
- Read the repository guidance files `F:/data_set_process/data_process/README.md` and `F:/data_set_process/data_process/datasetprocess/final_code/README.md` to confirm that `final_code/` is the maintained source of truth and that run-directory script copies are evidence only.
- Read protocol file `F:/data_set_process/data_process/datasetprocess/final_code/model/training/protocol_primary_control_v2_context_full2s/protocol_config.json` to check required split policy, anchor source, horizon, and online-input constraints.
- Reviewed required run artifacts from `TRAIN_V5_4_STATECOND_REV_20260413_104231` and `TRAIN_V5_4_STATECOND_REV_20260413_144211`, including `run_config.json`, `teacher_state_meta.json`, `teacher_base_missing_stats.json`, `feature_names.json`, and `loss_history.csv`.
- Traced the teacher-state build path, PCA valid-mask logic, missing-stat handling, smoke-mode truncation, sample construction, anchor selection, and evaluation exports using exact source line references.

### Why it was done
- Needed to determine whether the recent teacher-state expansion from legacy `old_ac` to switchable `old_ac` / `pca_latent` is safe on the thesis mainline.
- Needed to verify that the change does not introduce regression or data risk in subject-level split semantics, teacher-state statistics, anchor alignment, future-horizon handling, or online-only assumptions.
- Needed to judge whether the two existing validation runs are sufficient evidence for continuing to the next short sprint or formal experiment stage.

### What was found
- The teacher-state mode switch is mostly closed inside training and metadata export: default mode is `pca_latent`, requested dim is 4, PCA is fit on train-only rows, metadata exports component names and PCA parameters, and runtime configs in both runs record `TEACHER_STATE_MODE="pca_latent"` with `ACTUAL_STATE_DIM=4`.
- The PCA valid-mask repair does cover the all-missing feature case at the per-dimension level. Source now computes `all_missing_mask`, z-scores NaNs to train means, then fits PCA only on dimensions where all training rows are finite after imputation. Run `20260413_144211` shows the repaired path working with no all-missing dimensions; run `20260413_104231` shows an earlier weaker state where two dimensions were excluded by `pca_valid_mask` and no separate `teacher_base_missing_stats.json` was exported.
- There is a downstream semantic inconsistency for `pca_latent`: evaluation exports generic latent column names, but still aliases the first two latent dimensions to `A/C` and produces `A_veh/C_veh/A_teacher/C_teacher`-named plots even when `teacher_state_mode != "old_ac"`. This is not a training breakage, but it is not fully consistent for downstream consumers that expect semantic A/C only in legacy mode.
- The mainline script does not enforce the protocol’s subject-level fixed split. It builds samples from all subjects via glob over `ROOT/*/vehicle/*_vehicle_aligned_cleaned.csv`, then shuffles all event samples and performs a random 80/20 split. No code path loads or applies the subject split defined in `protocol_primary_control_v2_context_full2s/protocol_config.json`. This is a material leakage risk and the strongest blocker found in the review.
- Smoke mode changes more than training duration. It truncates the first `SMOKE_MAX_SAMPLES` samples before shuffle/split, so it changes subject/sample composition rather than just shrinking each already-defined split. Because the base split is already sample-level random instead of subject-level, smoke mode also changes split semantics relative to the protocol. Both reviewed runs also used smoke-like training hyperparameters (`EPOCHS=2`, `BATCH_SIZE=32`) rather than source defaults, but the saved `run_config.json` does not explicitly persist a smoke flag.
- Future horizon and anchor definitions remain internally consistent with a 3 s history / 2 s future setup. The script hardcodes `FUTURE_SEC=2.0`, `FUTURE_LEN=400`, uses anchor-relative future targets of length 400, and both runs persist the same values. Anchor selection is event-local and based on event-window roll or steer-rate logic, not on future labels.
- Online-only risk is mixed. Student `src` and `ctx` are built from current/past vehicle features, and the teacher physio window uses only the pre-anchor history window. However, the decoder road-preview input has a fallback path that directly slices future curvature from `X_all[peak_idx+1:peak_idx+1+FUTURE_LEN]` when speed-based projection is unavailable. In the two reviewed runs, `feature_names.json` confirms `zx|vx` and curvature are present, so the leak-prone fallback was likely not used there, but the maintained source still contains that potential protocol violation.

### Recommended next step
- Do not treat the current two runs as sufficient evidence for a formal mainline go decision because the maintained script still violates the protocol split requirement.
- Before any short sprint or formal experiment, require one verification step that proves protocol-aligned subject filtering is actually applied in the maintained source and rerun at least one smoke validation plus one non-smoke run under that split.
- Keep the teacher-state PCA path provisionally acceptable only after that split issue is cleared; the PCA/missing-stat change itself looks directionally safe, but the current evidence is contaminated by the split problem and by incomplete smoke provenance in saved configs.

### 补充记录（2026-04-13，Claude 收口 Codex 审查结论并重排下一步）
- 执行主体：混合协作（Codex 只读审查，Claude 收口）
- Why：在 `pca_latent` teacher-state 主线已经完成 smoke/短跑原型验证后，需要确认当前结论到底能不能进入下一步正式实验；用户也明确要求恢复既有 Claude/Codex 双向协作流程，由 Codex 先做只读 challenge，再由 Claude 收口。
- What was done：
  1. 通过旧的 `codex-bridge` 链路直接调用本地 Codex，只做 maintained 主线只读安全审查，不允许改代码。
  2. Codex 重点复核了 `future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`、`protocol_primary_control_v2_context_full2s/protocol_config.json`，以及两个运行目录 `TRAIN_V5_4_STATECOND_REV_20260413_104231` / `TRAIN_V5_4_STATECOND_REV_20260413_144211` 下的 `run_config.json`、`teacher_state_meta.json`、`teacher_base_missing_stats.json`、`feature_names.json`、`loss_history.csv`。
  3. Claude 进一步将审查结果对照项目协议、当前主线目标和已有实验判断做收口，识别“真正主阻塞”与“次级不一致项”。
- What was found：
  1. `old_ac / pca_latent` 双模式训练链路整体方向基本成立：默认 `pca_latent`、`state_dim=4`、PCA train-only 拟合、metadata 导出与两个 run 中的 `run_config` 基本一致。
  2. PCA `valid_mask` 修复和 teacher-base 缺失统计补充方向是对的；`20260413_144211` 已能证明 `teacher_base_missing_stats.json` 与 `teacher_state_meta.json` 的 repaired path 正常工作。
  3. 评估导出层仍存在一个次级口径问题：即使当前模式是 `pca_latent`，前两维 latent 仍被额外别名成 `A/C` 并沿用 legacy 命名画图；这更像下游解释层不一致，而不是主训练链路失效。
  4. 当前最强阻塞项不是 `pca_latent` 本身，而是 maintained 主线训练脚本并没有落实 protocol 要求的 `subject-level fixed split`。脚本当前是先全量收集样本、随机 shuffle、再做 sample-level 8:2 切分；这与 `protocol_primary_control_v2_context_full2s/protocol_config.json` 明确要求的固定被试级切分直接冲突。
  5. 因此现有两个 smoke/短跑 run 不能再被当作“主线已经安全可继续正式实验”的证据。它们最多只能证明：`pca_latent` teacher-state 工程闭环基本能跑通，但当前证据仍被 split 泄漏风险污染。
  6. smoke mode 也不是单纯缩短训练时长，而是会先截断样本池再做 shuffle/split，因此在当前 sample-level 切分前提下，它还会进一步改变样本组成与 split 语义。
- Recommended next step：
  1. 下一步不要先继续盯 `strong reversal` 或继续放大样本跑，而应优先把 maintained 主线的 split 逻辑对齐到 protocol 的 `subject-level fixed split`。
  2. split 对齐后，先重跑一轮最小 smoke 和一轮非 smoke 短跑，再重新评估 `pca_latent` 是否仍成立、以及 `strong reversal` 的问题到底来自标签稀疏还是 loss 配置。
  3. 在 split 问题修复前，当前更稳的口径应是：“teacher-state PCA 路线方向可继续，但还不能拿现有 run 当正式可信结果。”

### 补充记录（2026-04-13，阶段量化目标从保守版提升到更强版）
- 执行主体：Codex
- Why：用户明确表示当前目标可以更强一些，而且不着急，允许我们用 `8-9` 小时持续尝试；因此阶段目标不应继续停留在偏保守的“先把主线跑顺 + 略微改善 RMSE”层级，而应更贴近最终研究目的，即让模型对极限工况下驾驶员行为与车辆状态的预测在趋势、延时和方向性上都更可信。
- What was checked：
  1. 回看当前正式 protocol-safe full run `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260413_174639/training_summary.json`，提取当前可作为基线的验证 / 测试指标。
  2. 核对当前 full run 的 `val_metrics.json` 与 `figures/test_metrics.json`，确认当前已落地的指标主要是 `rmse_* / mae_*`，其中 test 基线约为：`rmse_steer=0.6557`、`rmse_yawrate=0.1339`、`rmse_ay=2.1182`。
  3. 对照用户最新口径，重新判断当前阶段到底应该把什么定义成“更强但仍合理”的量化目标：不是单纯把一个 RMSE 压得更低，而是要把“趋势相似、延时不要太大、不要反向预测”也一起纳入验收。
- What was found：
  1. 继续把目标只写成某个单一 RMSE 门槛，已经不足以表达这个项目当前真正想要的模型能力，因为那样容易把优化方向重新拉回“数值拟合”而不是“驾驶员真实行为趋势”。
  2. 当前更合理的 stronger target 应是一个多指标组合，而不是单点指标。建议的阶段性量化目标为：
     - `rmse_steer <= 0.58`
     - `rmse_yawrate <= 0.13`
     - `rmse_ay <= 1.95`
     - `steer` 未来序列样本级趋势相关性中位数 `>= 0.75`
     - 主响应峰值时间误差中位数 `<= 0.20 s`
     - 主响应峰值时间误差 `P90 <= 0.35 s`
     - 在“真实存在明确方向响应”的测试样本上，反向预测率 `<= 8%`
  3. 这组目标比此前给出的保守版更强，主要体现在：
     - `rmse_steer` 从保守建议的 `0.60` 再压到 `0.58`
     - `rmse_ay` 从 `2.00` 再压到 `1.95`
     - 趋势相关性从 `0.70` 提高到 `0.75`
     - 峰值延时从中位数 `0.25 s / P90 0.40 s` 收紧到 `0.20 s / 0.35 s`
     - 反向预测率从 `10%` 收紧到 `8%`
  4. 这组 stronger target 之所以仍然合理，是因为它依然围绕主目标，而不是为了“更强”去堆一些与行为真实性关系不大的次级指标。
  5. 当前 still-missing 的关键点是：现有 full run 产物里还没有把“趋势相关性中位数 / 峰值时间误差 / 反向预测率”正式固化为标准评估输出。因此如果要真正按这组 stronger target 执行，下一步第一优先级不是盲目继续调参，而是先把这几项评估指标正式接进 maintained 主线的评估闭环。
- Recommended next step：
  1. 先把 maintained 主线评估从“仅输出 RMSE / MAE”扩展到“RMSE + 趋势相关性 + 峰值延时 + 反向预测率”的正式口径。
  2. 用当前 protocol-safe full run 先产出一版完整基线报告，确认 stronger target 下各项指标距离目标还有多远。
  3. 在后续 `8-9` 小时推进窗口里，允许直接围绕这组 stronger target 做完整训练 / 完整测试，而不是回到偏保守的微小 smoke 口径。

### 补充记录（2026-04-13，更激进的最终执行目标定稿）
- 执行主体：Codex
- Why：用户在确认 stronger target 后，进一步明确希望目标“更狠一点”，但仍然允许较长的连续尝试窗口；因此需要给出一个比前一版更激进、但依然紧扣主目标、并且适合直接交给 Claude 开始执行的最终版本。
- What was checked：
  1. 继续以当前 protocol-safe full run 的 test 基线为出发点：`rmse_steer=0.6557`、`rmse_yawrate=0.1339`、`rmse_ay=2.1182`。
  2. 评估如果再向上加码，哪些指标可以收紧而不至于把目标变成脱离当前主线现实的空目标。
  3. 重新平衡“更强目标”与“8-9 小时内值得冲击”的关系，避免目标设得过于理想化而失去执行价值。
- What was found：
  1. 与前一版 stronger target 相比，更激进但仍可执行的最终版本建议为：
     - `rmse_steer <= 0.55`
     - `rmse_yawrate <= 0.12`
     - `rmse_ay <= 1.85`
     - `steer` 未来序列样本级趋势相关性中位数 `>= 0.80`
     - 主响应峰值时间误差中位数 `<= 0.18 s`
     - 主响应峰值时间误差 `P90 <= 0.30 s`
     - 在“真实存在明确方向响应”的测试样本上，反向预测率 `<= 5%`
  2. 这版目标之所以叫“更激进的最终版本”，是因为它已经不再只是“在当前基线基础上略微提升”，而是明显推动模型向“更像真实驾驶员行为趋势”的方向走：
     - `rmse_steer` 相比当前基线要下降约 `0.10`
     - `rmse_ay` 相比当前基线要下降约 `0.27`
     - 趋势相关性目标提高到 `0.80`
     - 峰值延时与反向预测率都进一步收紧
  3. 同时，这组目标仍保留了“阶段性可执行”的特征：它没有要求不现实的逐点精确复现，也没有把目标写成“必须接近完美预测”，而是继续围绕趋势、延时、方向一致性这些更符合项目主目标的标准。
  4. 如果 Claude 按这版 aggressive target 开始执行，那么第一优先级仍然不应是盲目继续训练，而是：
     - 先把趋势相关性、峰值延时、反向预测率正式接进评估闭环
     - 用当前 protocol-safe full run 产出完整基线
     - 再围绕这组 aggressive target 做完整训练 / 完整测试推进
- Recommended next step：
  1. 将这版 aggressive target 作为后续目标驱动自治推进模式的正式“最终版本”发送给 Claude。
  2. 要求 Claude 先补齐评估口径，再按完整训练 / 完整测试方式围绕这组目标推进，不要被单个局部 loss 牵着走。
  3. 若在推进过程中发现这组 aggressive target 在当前主线和当前时间预算下明显过于激进，也应以证据回报差距，而不是在同一个局部点上无休止反复调小问题。

### 补充记录（2026-04-14，Claude 高权限启动器按方案 2 落地）
- 执行主体：Codex
- Why：用户在实际使用 Claude 时发现，即使项目级 `dontAsk` 已开启，仍会因读取 `C:\Users\Administrator\AppData\Local\Temp\claude\...\tasks\...` 下的后台任务输出而弹出权限确认窗；用户随后明确选择“方案 2”，即采用更高一级的 `bypassPermissions` 会话权限，并把 `Temp\claude` 加入允许访问目录，尽量降低这类弹窗频率。
- What was checked：
  1. 读取项目级权限配置 `F:/data_set_process/data_process/.claude/settings.local.json` 与 `F:/data_set_process/data_process/.claude/settings.json`，确认当前项目确实已配置 `defaultMode = dontAsk`，但只覆盖项目工作区内的常规权限策略。
  2. 读取用户侧 Claude 状态文件 `C:\Users\Administrator\.claude\.claude.json`，确认项目 `F:/data_set_process/data_process` 已被标记为 trusted，但 `C:/Users/Administrator` 项目未完全信任。
  3. 检查 `C:\Users\Administrator\AppData\Local\Temp\claude\...` 目录结构，确认截图中的 `tasks/...output` 路径确实存在，并属于 Claude 的临时任务输出目录，而不是项目仓库内文件。
  4. 直接调用本机 `claude --help`，确认当前安装版本支持：
     - `--permission-mode bypassPermissions`
     - `--dangerously-skip-permissions`
     - `--add-dir`
  5. 查看仓库现有 `startup/` 目录，确认其中已经存在 Claude UTF-8 启动脚本，因此本次更适合沿用相同结构新增一个“高权限启动器”，而不是把命令散落在聊天记录里。
- What was done：
  1. 新增 `F:/data_set_process/data_process/startup/claude_bypass_permissions.ps1`：
     - 自动设置 UTF-8 控制台
     - 自动切到项目根目录
     - 自动确保 `C:\Users\Administrator\AppData\Local\Temp\claude` 存在
     - 自动以 `claude --permission-mode bypassPermissions --add-dir <Temp\\claude>` 启动 Claude
     - 允许继续透传额外 Claude 参数
  2. 新增 `F:/data_set_process/data_process/startup/Claude_Code_BypassPermissions.cmd`，便于用户在 Windows 下直接双击启动，而无需每次手敲完整命令。
  3. 第一次验证时发现脚本通过 `Get-Command claude` 命中了 PATH 中较靠前的旧安装 `D:\Apps\nodejs\claude`，实际版本是 `2.1.91`，与用户截图中的版本一致。
  4. 随后修改启动脚本，改为显式优先选择：
     - `D:\ClaudeCode\global\claude.cmd`
     - `D:\ClaudeCode\global\claude.ps1`
     - 只有不存在时才回退到旧位置或 `Get-Command`
     同时在启动时额外打印实际使用的 Claude 版本，避免后续再靠界面猜测。
  5. 再次使用 `powershell.exe -ExecutionPolicy Bypass -File ... --help` 对修正后的脚本做轻量验证，确认：
     - 脚本能正常切到项目根目录
     - 能正确定位本机 `claude` 启动器
     - 能正确拼接 `bypassPermissions` 与 `add-dir Temp\\claude`
     - 现在实际命中的是 `D:\ClaudeCode\global\claude.cmd`
     - 当前实际版本为 `2.1.107`
     - 不会在验证时误开交互会话
- What was found：
  1. 你此前的提示词和项目级 `dontAsk` 不是完全没效果，而是它们主要作用于项目工作区内；截图里的弹窗之所以仍出现，是因为 `tasks/...output` 属于工作区外的临时目录读取。
  2. 对当前场景而言，单纯继续强化提示词并不能真正解决问题；更有效的是提高会话权限模式，并把 `Temp\\claude` 明确加入允许访问路径。
  3. 用户截图里显示的 `v2.1.91` 不是视觉错觉，而是首次脚本确实拉起了旧安装；问题根因是 PATH 优先级，而不是新脚本参数本身无效。
  4. 修正后，启动器已经稳定改为优先拉起新版 `2.1.107`，这也解释了为什么新旧界面细节和帮助文本会有差异。
  5. “方案 2”相较 `dangerously-skip-permissions` 更平衡：它已经比 `dontAsk` 更强，但仍比完全跳过所有权限检查更保守一些。
  6. 即便如此，也不能保证所有弹窗绝对消失；如果后续仍有极少数非该目录或更高层级的客户端权限弹窗，需要再根据弹窗路径判断是否属于新的工作区外路径。
- Recommended next step：
  1. 后续优先通过 `F:/data_set_process/data_process/startup/Claude_Code_BypassPermissions.cmd` 启动本项目的 Claude 会话，而不是继续用旧入口。
  2. 如果后续仍主要是 `tasks/` 目录相关弹窗，可再考虑叠加环境变量方案（如禁用后台任务系统或改写 `CLAUDE_CODE_TMPDIR`）做第二层缓解。
  3. 若未来需要进一步提高到最强权限，再考虑 `--dangerously-skip-permissions`，但应明确那已经超出当前“方案 2”的风险边界。

## 专题索引
- `课题定义与模型方向`
  - 2026-04-02：重新界定研究问题，聚焦短时预测、反应表征、模式分类与共享控制。
  - 2026-04-13：将当前阶段目标收口为“趋势正确、延时可控、反向预测受限”的 stronger quantitative target，而不是仅用单一 RMSE 代表模型进步。
  - 2026-04-13：进一步上调为更激进的最终执行目标，明确以更低 RMSE、更高趋势相关性、更低延时和更低反向预测率作为阶段冲刺标准。
- `文献检索与导入`
  - 2026-04-02：通过 ScholarAIO 与 Zotero 工作流导入核心综述和相关文献。
- `协作记录规范`
  - 2026-04-02：建立“先写详细进度，再给压缩总结”的规则，并同步到 README / CLAUDE / 命令与桥接层。
  - 2026-04-13：进一步确认可以按“目标驱动自治推进”执行，默认沿用既有 Claude -> Codex 协作工作流、禁删文件约束与强制项目日志记录。
  - 2026-04-13：将目标驱动自治推进模式正式落成仓库级文档、任务模板与 `.claude` 命令入口，避免后续每次重复解释执行方式。
  - 2026-04-13：进一步细化为“目标对齐优先 + 同点尝试预算 `3-4` 次 + 训练规模按风险决定”的版本，明确避免偏离主目标和在局部问题上反复打转。
  - 2026-04-14：为本项目新增 Claude 高权限启动器（`bypassPermissions + add-dir Temp\claude`），用于降低 `tasks/` 临时目录读取弹窗。
- `项目日志结构优化`
  - 2026-04-02：将总进度日志改造成更易检索的项目日志，新增阅读说明、当前状态速览、日期索引、专题索引与执行主体标注建议。
- `Zotero 结构整理`
  - 2026-04-02：重建论文导向分类树、建立论文写作用核心集、删除旧分类外壳、清理空节点。
- `conditioned v2 归因与诊断`
  - 2026-04-08：Codex 完成 Task 1/2/3；Claude 收口机制判断（amplitude 失配 + boundary 时移）和 Step 4 决策（方案 A：tail amplitude penalty）。
  - 2026-04-09：matched-schedule 公平双跑确认 structured_v2 仍有净信号，但 boundary_shift_abs_err 是当前主阻塞；Codex 只读 review 进一步收口为结构轨道注入导致的 boundary-local continuity 恶化。
- `teacher-state 主线与 split 安全`
  - 2026-04-10：Claude 将 maintained 单文件训练脚本扩展为可切换 `old_ac / pca_latent` 的 teacher-state 主线，并完成 smoke/短跑原型验证。
  - 2026-04-13：Codex 只读审查确认 `pca_latent` 工程链路方向基本成立，但指出 maintained 主线仍未应用 protocol 要求的 `subject-level fixed split`，这是当前最强阻塞项。

## 记录格式建议
- 日期层：继续使用 `## YYYY-MM-DD` 作为当天主标题。
- 条目层：补充记录继续使用 `### 补充记录（YYYY-MM-DD 时间，主题）`。
- 主体层：每条新增记录开头尽量单独写明：
  - `执行主体：Claude`
  - `执行主体：Codex`
  - `执行主体：用户`
  - `执行主体：混合协作`
- 索引维护：
  - 当某一天出现新的重要工作时，同时更新“日期索引”中的一句话摘要。
  - 当出现新的长期主题时，同时更新“专题索引”。
- 如果未来日志继续变长：
  - 可继续保留本文件作为总索引与总日志；
  - 再按月或按阶段拆分归档文件，但当前暂时还不需要。

## 2026-04-02

### 今天做了什么
- 明确当前核心问题：现有模型预测未来 2 s 驾驶员方向盘轨迹时，前约 1 s 趋势还能对上，后 1 s 逐渐失真，和真实驾驶员反应不一致。
- 明确研究目标不应被“固定 2 s 方向盘角轨迹预测”绑死，而应转向“未来一段时间驾驶员反应映射到动作表征”的建模。
- 形成了几个候选研究方向：
  1. 缩短预测时域，重点研究 0.5–1.0 s 的短时反应预测。
  2. 将预测目标从单一方向盘角扩展为多表征动作变量，如方向盘角、方向盘角速度、横向速度、纵向速度等。
  3. 先做驾驶员操作模式分类/聚类，再做分模式建模或条件预测。
  4. 面向极限工况与共享控制/辅助驾驶协同，强调“真实反应”而不是“几何轨迹拟合”。
- 按现有 ScholarAIO 工作流检查了文献检索路径：
  - 关键词检索使用 OpenAlex。
  - DOI / URL 可进一步导入 Zotero。
  - 当前脚本：`tools/academic_search_to_zotero.py`
  - 工作流说明：`reports/codex_academic_zotero_workflow.md`
- 用 ScholarAIO 检索了 4 类相关方向文献：
  1. driver behavior / motion prediction
  2. steering / vehicle dynamics / loss of control
  3. maneuver classification / driving style clustering
  4. shared control / driver model

### 当前判断
- 现有问题很可能不是单纯“模型没调好”，而是任务定义本身有偏差：
  - 极限工况下驾驶员未来 2 s 操作具有更强的不确定性。
  - 后段轨迹可能受车辆状态演化、驾驶员纠正动作、闭环反馈影响更大，因此固定长时域回归容易均值化、失真。
- 方向盘角速度可能比方向盘角本身更直接表征“反应强度/反应启动”，值得作为目标之一做对比。
- 老师提出的“先聚类/分类，再预测”路线是合理的，因为极限工况下不同反应模式（制动、转向修正、联合操作）混在一起，会削弱单一回归模型的可学习性。

### 初步文献线索
1. `A survey on motion prediction and risk assessment for intelligent vehicles`  
   DOI: `10.1186/s40648-014-0001-z`
2. `Understanding and Modeling the Human Driver`  
   DOI: `10.1076/vesd.40.1.101.15875`
3. `A Review of Shared Control for Automated Vehicles: Theory and Applications`  
   DOI: `10.1109/THMS.2020.3017748`
4. `An Overview on Study of Identification of Driver Behavior Characteristics for Automotive Control`  
   DOI: `10.1155/2014/569109`
5. `Modeling and Recognizing Driver Behavior Based on Driving Data: A Survey`  
   DOI: `10.1155/2014/245641`
6. `A Review of Intelligent Driving Style Analysis Systems and Related Artificial Intelligence Algorithms`  
   DOI: `10.3390/s151229822`
7. `Real-time estimation and prediction of tire forces using digital map for driving risk assessment`  
   DOI: `10.1016/j.trc.2019.08.016`
8. `Review of Integrated Chassis Control Techniques for Automated Ground Vehicles`  
   DOI: `10.3390/s24020600`

### 还存在的问题
- 还没有把“你这个具体课题”压缩成最适合投稿/汇报的一句研究问题。
- 还没有明确最优预测目标是：
  - 方向盘角
  - 方向盘角速度
  - 横纵向速度
  - 还是这些变量的联合表征。
- 还没有明确当前数据里哪些标签最能定义“操作模式/反应类型”。
- 还没有验证：如果把预测时域从 2 s 缩到 1 s 或分段预测，现有模型结果会不会明显改善。

### 下一步建议
1. 做一个小型任务重定义：
   - 比较未来 0.5 s / 1.0 s / 1.5 s / 2.0 s 的可预测性差异。
2. 做目标变量对比实验：
   - 方向盘角 vs 方向盘角速度 vs 联合动作变量。
3. 做反应模式划分：
   - 先用规则或聚类把样本分成制动主导、转向主导、联合修正等类型。
4. 做条件建模：
   - 先分类后回归，或把模式标签作为条件输入。
5. 文献侧继续补齐：
   - 更聚焦“driver intent / steering reversal / corrective steering / shared control / loss-of-control recovery”关键词。

### 补充记录（2026-04-02，文献导入与方案整理）
- 已按 ScholarAIO 工作流将多篇核心文献导入 Zotero 分类：`研究生论文/极限工况驾驶员反应建模/短时预测_行为识别_共享控制`。
- 已成功导入的核心条目包括：
  - `10.1186/s40648-014-0001-z`
  - `10.1076/vesd.40.1.101.15875`
  - `10.1109/THMS.2020.3017748`
  - `10.1155/2014/569109`
  - `10.1155/2014/245641`
  - `10.3390/s151229822`
  - `10.1016/j.trc.2019.08.016`
- 其中部分条目同时成功下载并上传了开放 PDF，部分条目仅完成元数据入库。
- `10.3390/s24020600` 在导入过程中遇到外部 SSL / 握手超时，当前未确认是否成功入库，需要后续单独复查。
- 已形成一版组会方案草案：`reports/driver_reaction_modeling_group_meeting_plan_20260402.md`

### 备注
- 本文件用于持续累计项目进度、问题、结论与后续计划。
- 后续每次重要讨论、实验结论、失败原因、老师反馈，都继续追加到这里。

### 补充约定：协作过程先落进度再压缩总结
- 从这次开始，这个文件作为项目协作的主进度日志之一。
- 后续无论是我还是 Codex 在本项目里执行命令、检索资料、修改脚本、整理方案、分析结果，只要产生了实质性进展，都优先把进度详细追加到这里，再做对话里的压缩总结。
- 进度记录尽量写清楚以下内容：
  - 谁做的：本条记录尽量标注执行主体，例如 `Claude`、`Codex`、`用户`、`混合协作`。
  - 做了什么：执行了哪些命令、看了哪些文件、改了哪些脚本、跑了哪些实验、整理了哪些文献。
  - 为什么做：当前动作对应的问题背景、判断依据、任务目标。
  - 得到了什么：关键发现、结论、异常、失败原因、未解决点。
  - 下一步是什么：建议继续做的检查、实验、修改或决策点。
- 如果某次工作量较大，优先写“详细过程版”到本文件，再在聊天中输出“压缩总结版”。
- 如果只是很小的确认性动作，可以简写，但不能省掉对当前项目状态的更新。
- 当日志累计变长后，不要求每次从头回看；应优先维护“当前状态速览”，必要时再按专题或日期追溯历史。
- 为了便于后续索引，新增记录时尽量同步维护文件顶部的“日期索引”和“专题索引”。

### 补充记录（2026-04-02 晚，日志结构优化与记录规则固化）
- 执行主体：混合协作（用户 + Codex + Claude）
- 用户明确提出新的协作要求：以后在本项目中进行命令执行、分析、修改、整理等工作时，不要只在聊天里做压缩式汇报，而要先把“目前已经干了什么、做到哪一步、得到什么判断、接下来做什么”详细写进 `reports/project_progress_master.md`。
- Codex 已先把该日志顶部改造成更适合回看的项目日志入口，核心变化包括：
  - 新增“阅读说明”；
  - 新增“当前状态速览”；
  - 新增“日期索引”；
  - 新增“专题索引”；
  - 新增“记录格式建议”，明确后续记录尽量标注执行主体。
- Claude 已补充将这次顶部结构优化正式纳入当前状态、日期索引与专题索引，避免它只停留在文件结构层面而没有进入可追溯的项目进展描述。
- 当前判断：这一步解决的不是“又多写了一份文档”，而是把原本越来越长、越来越难回看的累计日志，调整成“先看总览、再按日期/专题回溯”的可检索项目日志，后续再开新会话时也更容易快速恢复上下文。
- 已确认当前这份进度文档内容完整、编码正常（UTF-8），可继续作为后续迭代的累计进度日志使用。
- 已将“先写详细进度、再做压缩总结”的记录约定正式补充进本文档，作为后续协作默认执行规则。
- 下一步建议：
  - 后续每次新增重要记录时，同步维护顶部“日期索引”和“专题索引”；
  - 若日志继续明显变长，再考虑拆分 `project_progress_master.md` 总览 + `reports/progress_daily/` 明细，但当前阶段暂不必提前拆分。

### 补充记录（2026-04-02 晚，规则同步到仓库说明）
- 执行主体：Claude
- 用户同意将上述协作约定进一步同步到项目公共说明文件，避免规则只存在于单一进度日志中，后续其他代理或后续会话不容易看到。
- 已检查仓库根目录说明文件，确认存在以下两个适合落规则的位置：
  - `README.md`：面向项目整体使用和协作入口。
  - `CLAUDE.md`：面向代理在本仓库中的执行约束与工作方式。
- 已在 `README.md` 中新增“协作记录约定”小节，明确：
  - 当前主进度日志文件位置；
  - 先写详细进度、再给压缩总结；
  - 进度记录至少应包含“做了什么、为什么做、得到了什么、下一步建议”。
- 已在 `CLAUDE.md` 中补充代理执行要求，明确对于实质性工作：
  - 在输出压缩聊天总结前，先将详细进展写入 `reports/project_progress_master.md`；
  - 命令执行、文件检查、脚本修改、实验分析、文献/方案整理等，只要对项目有实质推进，都应计入进度。
- 这样处理后，规则现在同时存在于：
  - 项目进度日志；
  - 仓库 README；
  - 仓库代理工作说明。
- 后续如果还需要，我可以继续把这条规则同步到 `.claude/commands/` 或专门的项目协作规范文档中，使自动化工作流也默认遵守。

### 补充记录（2026-04-02 晚，规则写入 Claude / Codex 命令体系）
- 执行主体：Claude
- 用户进一步明确希望不只是普通说明文件知道这条规则，而是 Claude 和 Codex 的命令模板本身也要知道：只要是在执行与论文模型、实验分析、文献整理、方案推进有关的工作，就应把“记忆/进展”写进项目进度日志，而不是只在聊天里简单带过。
- 为落实这一点，本次检查并修改了仓库中的命令与代理相关文件，重点查看了：
  - `.claude/commands/`
  - `.claude/agents/`
  - `.claude/skills/`
- 已补充规则的命令文件包括：
  - `codex-handoff.md`
  - `codex-review.md`
  - `codex-run.md`
  - `codex-workflow.md`
  - `compare-runs.md`
  - `data-check.md`
  - `failure-analysis.md`
  - `session-sync.md`
  - `summarize-run.md`
- 已补充规则的代理与 skill 文件包括：
  - `.claude/agents/codex-coordinator.md`
  - `.claude/skills/codex-handoff/SKILL.md`
  - `.claude/skills/codex-review/SKILL.md`
  - `.claude/skills/session-sync/SKILL.md`
- 本次写入的核心约束是：
  - 只要任务对论文/模型进展有实质推进，就应先把详细进度写入 `reports/project_progress_master.md`，再返回压缩总结。
  - 详细进度至少包含：
    - 做了什么
    - 为什么做
    - 得到了什么
    - 下一步建议
  - 对 Codex 的 handoff / workflow / run 类命令，要显式把这条要求写进交接提示词或执行简报。
  - 对 compare-runs / data-check / failure-analysis / summarize-run 这类分析命令，也要求先写进度日志再输出压缩结论。
  - 对 session-sync 这类会话压缩命令，要求在生成紧凑 checkpoint 之前先确认详细进度已经写入项目日志。
- 当前状态下，这条规则已经不只存在于说明文档，而是进入了：
  - 项目 README；
  - 根目录 `CLAUDE.md`；
  - Claude 命令模板；
  - Codex 交接相关命令模板；
  - 代理协作说明；
  - 相关 skills。
- 这样做的意义是，后续无论是直接分析、交接给 Codex、复盘实验、生成 session checkpoint，还是做 run summary，只要属于论文模型推进工作，系统级默认要求都会倾向于先保留详细进度。

### 补充记录（2026-04-02 晚，规则下沉到 Codex 桥接脚本）
- 执行主体：Claude
- 在将规则写入命令模板后，又进一步检查了 Claude 调用 Codex 的本地桥接层，避免出现“命令模板里写了要求，但真正传给 Codex 的 prompt 没有稳定带上”的情况。
- 本次查看的桥接文件包括：
  - `D:\ClaudeCode\codex-bridge\claude-codex-entry.ps1`
  - `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
  - `D:\ClaudeCode\codex-bridge\run-codex.cmd`
- 检查结果表明：
  - `claude-codex-entry.ps1` 只是一个很薄的入口，主要负责把 prompt 转交给 `invoke-codex.ps1`。
  - 真正适合统一补项目级前置约束的位置，是 `invoke-codex.ps1` 中把 prompt 组装后再发送给 Codex 的环节。
- 已在 `D:\ClaudeCode\codex-bridge\invoke-codex.ps1` 中新增项目级 prompt 前缀注入逻辑：
  - 当 `ProjectRoot` 为 `F:\data_set_process\data_process` 时，自动在用户 prompt 前补一段仓库专属说明。
  - 说明内容明确要求：如果任务对论文/模型进展有实质推进，应先把详细进度写入 `reports/project_progress_master.md`，再返回压缩总结。
  - 说明里还列明了哪些工作属于应记录的进展，例如：
    - 命令执行
    - 文件检查
    - 脚本或文档修改
    - 实验分析
    - run comparison
    - failure analysis
    - 文献整理
    - 方案细化
  - 同时明确进度日志至少需要包括：
    - 做了什么
    - 为什么做
    - 得到了什么
    - 下一步建议
- 这样处理后，即使后续某个 Claude/Codex 命令模板忘了重复强调，只要还是通过这套 bridge 在当前项目根目录下调用 Codex，桥接层也会自动把这条规则带入 prompt。
- 当前对该桥接改动做了静态核对，已确认新增的 prompt 前缀逻辑和目标日志路径写入成功；本次未额外做一次真实 Codex 往返调用测试，以避免为单纯规则注入再产生无意义的代理运行成本。

### 补充记录（2026-04-02 晚，进度日志固定命名）
- 执行主体：混合协作（用户 + Claude）
- 用户提出将进度日志从按日期命名改为长期固定名，避免后续命令、桥接脚本、说明文档持续绑定到某一天的文件名，随着日期变化产生维护成本。
- 已将原文件 `reports/project_progress_20260402.md` 重命名为 `reports/project_progress_master.md`，作为后续长期使用的主进度日志。
- 本次同步更新了所有已知关键引用位置，包括：
  - 项目说明文件：`README.md`
  - 仓库代理约束：`CLAUDE.md`
  - Claude / Codex 命令模板：`.claude/commands/`
  - 协作代理说明：`.claude/agents/codex-coordinator.md`
  - 相关 skills：`.claude/skills/`
  - Codex 本地桥接脚本：`D:\ClaudeCode\codex-bridge\invoke-codex.ps1`
- 这样调整后，后续不需要因为日期变化继续迁移日志文件名；默认都应写入 `reports/project_progress_master.md`。

### 补充记录（2026-04-02 晚，Zotero 按论文主线重整）
- 执行主体：Codex
- 用户提出新的文献管理任务：当前 Zotero 文献管理较乱，希望按文献内容重新整理全部文献；主基调是服务当前驾驶员反应建模/论文主线，与论文关系弱的内容单独归档；允许重新设计 collection 布局和命名。
- 本次首先检查了当前 Zotero 相关基础设施，确认：
  - 仓库中已有 Zotero 修复与规范化痕迹，包括：
    - `zotero_repair_summary.txt`
    - `zotero_normalization_summary.txt`
    - `zotero_linked_attachment_repair_report.csv`
    - `tools/academic_search_to_zotero.py`
  - `startup/academic_zotero_config.json` 中存在可用的 Zotero Web API 配置（user library `13300964`），因此本次优先采用 Web API，而不是直接操作本地 SQLite。
- 读取 Zotero 库现状后，得到一版整体画像：
  - 顶层文献条目约 `209` 条；
  - collections 约 `27` 个；
  - 现有结构呈现明显的“历史任务叠加型混乱”：
    - 当前论文主线相关：极限驾驶工况、驾驶行为建模、车辆判稳、共享控制、人因、多模态信号、脑电、生理等；
    - 方法借鉴类：驾驶员分心状态检测、机器视觉、域泛化、元学习、数据增强等；
    - 明显旁支：机器人课程、智能制造、电动飞行器/电推进等；
    - 还有少量未归类顶层条目。
- 判断后采用的整理策略不是“删除旧 collections 重来”，而是更稳妥的：
  - 新建一套论文导向 collection 树；
  - 将所有顶层文献重新加入新树中的目标位置；
  - 保留旧 collections 作为安全网和历史结构，不做破坏性清理。
- 为保证后续可复用，本次没有手工逐条拖拽，而是新增脚本：
  - `tools/organize_zotero_for_thesis.py`
- 该脚本的核心功能包括：
  - 读取 Zotero API 配置；
  - 拉取全部 top-level items 与 collections；
  - 依据标题、摘要、期刊、tag、现有 collection 路径进行内容优先分类；
  - 自动创建新的论文导向 collection 层级；
  - 将条目加入新 collection；
  - 输出 JSON + Markdown 报告，便于复查。
- 本次设计的新 Zotero 主结构为：
  - `论文整理_驾驶员反应建模`
    - `01_核心主线`
      - `综述与研究定位`
      - `驾驶员行为建模与短时预测`
      - `极限工况_车辆稳定性_风险评估`
      - `共享控制_接管_人机协同`
      - `多模态_生理_脑电`
    - `02_方法借鉴`
      - `驾驶状态_分心_情绪识别`
      - `机器视觉_域泛化_迁移学习`
      - `多模态学习_信号处理方法`
    - `03_待处理`
      - `待人工复核`
    - `90_旁支归档`
      - `机器人与课程作业`
      - `智能制造`
      - `电动飞行器与电子电气`
      - `其他非当前论文`
- 在正式写回前，先连续跑了多版 `dry-run`，并针对错误分配做了规则修正，主要修正点包括：
  - 降低旧 collection 路径对分类的支配权，避免被历史错误目录拖偏；
  - 强化“内容相关性”判断，让真正与驾驶员建模/共享控制/极限工况/多模态信号相关的文献优先回到论文主线；
  - 将明显 generic 或明显无关条目移出主线；
  - 为 `lane change intention`、`anticipatory driving`、`conditional automation`、`takeover` 等与当前论文更相关的主题增加识别规则。
- 最终 `apply` 写回结果：
  - 脚本实际更新了 `209` 条顶层文献的 collection 归属；
  - 生成最终应用报告：
    - `reports/zotero_thesis_organize_report_20260402_194719.json`
    - `reports/zotero_thesis_organize_report_20260402_194719.md`
  - 按最终报告统计，新结构中的分类数量为：
    - `驾驶状态_分心_情绪识别`：50
    - `机器视觉_域泛化_迁移学习`：46
    - `多模态_生理_脑电`：26
    - `极限工况_车辆稳定性_风险评估`：20
    - `驾驶员行为建模与短时预测`：15
    - `共享控制_接管_人机协同`：13
    - `综述与研究定位`：9
    - `电动飞行器与电子电气`：11
    - `智能制造`：6
    - `机器人与课程作业`：5
    - `其他非当前论文`：4
    - `待人工复核`：4
- 当前剩余待人工复核项只有 4 条，分别是：
  - `Addon Item`
  - `Deviation Sequence Neural Network Control for Path Tracking of Autonomous Vehicles`
  - 2 条无标题异常记录
- 这说明整理体系已经可用，但仍建议后续手工做最后一轮轻量清理：
  - 删除或修复无标题异常条目；
  - 判断 `Path Tracking of Autonomous Vehicles` 更适合放在主线控制方法还是旁支；
  - 视需要将旧 collections 逐步弱化，或只保留作历史参考。
- 额外说明：
  - 在 `apply` 结束后尝试再次通过 Web API 回读新树做二次验证时，遇到一次远端 SSL / connection reset 中断；
  - 但 `apply` 脚本本身已经返回成功，且最终应用报告已正常生成，因此当前认为主要写回动作已经完成；
  - 后续如有必要，可再单独做一轮只读核对，确认 Zotero 在线端显示与报告一致。

### 补充记录（2026-04-02 晚，Zotero 第二轮精修与论文写作用核心集）
- 执行主体：Codex
- 在完成第一轮全量整理后，用户同意继续执行下一步，因此又对新结构做了第二轮精修，目标有两个：
  - 清理残留的边界/异常条目；
  - 从全量整理结构中再抽出一套更适合论文写作与文献综述的核心精选集。
- 为此继续增强了脚本：
  - `tools/organize_zotero_for_thesis.py`
- 第二轮脚本增强内容包括：
  - 新增 `03_待处理 / 异常条目_待清理` 分类，用于承接明显无效或元数据损坏的条目；
  - 对少数边界条目增加手工 override 规则；
  - 新增 `04_论文写作用核心集` collection 体系；
  - 增加“managed collections”同步逻辑，确保新整理树内部不会残留旧的错误归属。
- 本次新增的论文写作用核心集结构为：
  - `论文整理_驾驶员反应建模 / 04_论文写作用核心集`
    - `A_研究定位与综述`
    - `B_驾驶员行为建模与短时预测`
    - `C_极限工况_风险_稳定性`
    - `D_共享控制_接管_人机协同`
    - `E_多模态_生理_脑电`
    - `F_方法借鉴_状态识别`
    - `G_方法借鉴_域泛化`
- 第二轮 `apply` 写回结果：
  - 第一次精修写回时：
    - 更新主分类条目 4 条；
    - 加入核心集归属 56 条。
  - 随后又执行了一次 managed collections 同步，以清除新整理树内部的旧残留归属：
    - 更新条目 168 条；
    - 核心集条目总量仍为 56 条。
- 第二轮最终应用报告：
  - `reports/zotero_thesis_organize_report_20260402_200230.json`
  - `reports/zotero_thesis_organize_report_20260402_200230.md`
- 第二轮最终状态下：
  - `待人工复核` 已清空；
  - 仅剩 `3` 条进入 `异常条目_待清理`：
    - `Addon Item`
    - 2 条无标题异常记录
  - `Deviation Sequence Neural Network Control for Path Tracking of Autonomous Vehicles` 已被移入：
    - `90_旁支归档 / 其他非当前论文`
- 针对这些边界条目又额外做了只读核对，确认当前实际 Zotero 归属为：
  - `Addon Item` -> `异常条目_待清理`
  - 两条无标题异常记录 -> `异常条目_待清理`
  - `Path Tracking of Autonomous Vehicles` -> `其他非当前论文`
- 论文写作用核心集当前已自动筛出 56 条，覆盖：
  - 研究定位与综述
  - 驾驶员行为建模与短时预测
  - 极限工况/风险/稳定性
  - 共享控制/接管/人机协同
  - 多模态/生理/脑电
  - 状态识别方法借鉴
  - 域泛化方法借鉴
- 这意味着当前 Zotero 现在已经形成两层可直接使用的结构：
  - 全量论文导向整理库：便于长期累积；
  - 论文写作用核心集：便于直接阅读、综述和写作。
- 额外说明：
  - 报告中的 `current_paths` 字段是生成报告时的分配快照，个别异常项在第二次 managed sync 前后可能显示旧路径组合；
  - 本次又专门对 4 条边界条目做了单独回读，已确认 Zotero 中的实际最终归属是正确的。

### 补充记录（2026-04-02 晚，删除旧 Zotero 分类外壳）
- 执行主体：Codex
- 用户查看 Zotero 左侧目录后，觉得仍然显得杂乱，希望把之前的旧分类删掉；前提是文献已经被整理进新的分类体系中。
- 为避免误删，本次没有直接执行删除，而是先做了覆盖验证：
  - 检查旧 collections 中的文献，是否都已经至少挂到新的 `论文整理_驾驶员反应建模` 树下。
  - 验证结果：
    - 旧 collections 数量：`27`
    - 新整理树 collections 数量：`26`
    - 仅存在于旧 collections、但未进入新树的条目数：`0`
- 因为覆盖验证通过，说明旧分类已不再承担唯一归档作用，所以可以安全删除旧 collections 外壳，而不会把文献条目本身从 Zotero 库中删掉。
- 为保证这个动作以后可复用，又新增了一个清理脚本：
  - `tools/prune_zotero_old_collections.py`
- 该脚本的逻辑是：
  - 读取全部 Zotero collections；
  - 将 `论文整理_驾驶员反应建模` 及其全部子 collection 视为“保留树”；
  - 将其他 collections 视为“旧树候选”；
  - 先检查是否有条目仍只存在于旧树中；
  - 只有在没有漏项时，才执行删除。
- 本次先跑了 dry-run，生成报告：
  - `reports/zotero_old_collection_prune_20260402_201528.json`
  - `reports/zotero_old_collection_prune_20260402_201528.md`
- dry-run 中列出的待删除旧 collections 共 `27` 个，主要包括：
  - `研究生论文` 及其历史子树；
  - `驾驶行为建模`；
  - `ScholarAIO`；
  - `贺老师发表论文 / savedrecs`；
  - `智能制造系统`；
  - `机器人技术结课论文参考文献`；
  - 以及其他旧的历史 collection 入口。
- 随后执行了正式删除，生成最终 apply 报告：
  - `reports/zotero_old_collection_prune_20260402_201605.json`
  - `reports/zotero_old_collection_prune_20260402_201605.md`
- 正式删除结果：
  - 旧 collections 共删除 `27` 个；
  - 删除后再次回读 Zotero collections，当前左侧只剩新的 `论文整理_驾驶员反应建模` 树及其子结构。
- 当前 Zotero 目录主结构为：
  - `01_核心主线`
  - `02_方法借鉴`
  - `03_待处理`
  - `04_论文写作用核心集`
  - `90_旁支归档`
- 额外说明：
  - 当前 `03_待处理 / 待人工复核` 仍是一个空 collection，它不是旧分类残留，而是新结构里预留的后续缓冲位；
  - 如果后续也想把这个空节点去掉，可以再做一次轻量清理。

### 补充记录（2026-04-02 晚，清理空节点并解释父节点计数为 0）
- 执行主体：Codex
- 用户继续指出 Zotero 左侧目录里仍有一个视觉上不舒服的点：
  - `03_待处理 / 待人工复核` 已经为空；
  - `02_方法借鉴`、`90_旁支归档` 这些父节点旁边显示为 `0`，容易让人误以为整理失败。
- 本次已执行的实际清理：
  - 删除空 collection：`论文整理_驾驶员反应建模 / 03_待处理 / 待人工复核`
  - 删除后回读确认：
    - `03_待处理` 还保留，但其子 collection 数量从 2 变为 1；
    - 当前只剩 `异常条目_待清理` 作为该节点下的有效子文件夹。
- 关于父节点旁边显示 `0` 的原因，本次也做了 API 层核对：
  - `02_方法借鉴`：
    - `numItems = 0`
    - `numCollections = 2`
  - `90_旁支归档`：
    - `numItems = 0`
    - `numCollections = 4`
  - `03_待处理`：
    - `numItems = 0`
    - `numCollections = 1`
- 这说明 Zotero 左侧列表中父 collection 右侧的数字，显示的是：
  - 当前这个 collection 自己“直接挂了多少条目”
  - 而不是“把所有子 collection 里的条目一起加总后的总数”
- 因此：
  - `02_方法借鉴` 显示 `0`，并不表示里面没有文献；
  - 它只是表示“没有文献直接放在 `02_方法借鉴` 这一层”，文献都放在它下面的两个子分类里。
- 当前这属于 Zotero 的显示逻辑，不是本次整理出错。
- 如果以后想让父节点右侧也不是 `0`，理论上只有两种办法：
  - 把部分文献直接也挂到父节点；
  - 或接受 Zotero 这个“只显示当前层直接条目数”的默认行为。
- 现阶段为了保持结构干净，没有把文献重复挂到父节点，因此保留 `0` 是更合理的做法。

### 补充记录（2026-04-02 晚，清理中间态自动报告文件）
- 执行主体：Codex
- 用户随后注意到 `reports/` 目录里出现了较多带时间戳的 Zotero 报告文件，担心目录显得过于杂乱。
- 说明原因：
  - 这些文件并不是额外的业务文档，而是本次整理过程中为了稳妥起见多次执行 `dry-run`、规则修正、`apply`、旧分类清理时自动生成的阶段性报告；
  - 每一轮都会生成一对文件：
    - `.md`：便于人工查看；
    - `.json`：便于脚本复查与程序化对比。
- 为减少干扰，本次按“保留最终结果，删除中间迭代”的原则进行了清理。
- 删除的中间态文件包括：
  - 较早几轮 `zotero_thesis_organize_report_*.md/.json`
  - 较早一轮 `zotero_old_collection_prune_*.md/.json`
- 当前保留的 Zotero 相关最终结果文件为：
  - `reports/zotero_thesis_organize_report_20260402_200230.json`
  - `reports/zotero_thesis_organize_report_20260402_200230.md`
  - `reports/zotero_old_collection_prune_20260402_201605.json`
  - `reports/zotero_old_collection_prune_20260402_201605.md`
- 清理后，`reports/` 目录中的 Zotero 整理结果只保留最终版本，避免中间态报告继续堆积。

### 补充记录（2026-04-02 晚，Claude Code + Codex 协同流程调研）
- 执行主体：Claude
- 用户当前希望先暂停代码实验推进，转而调研 Claude Code 和 Codex 协同合作的成熟方式，重点关注：
  - Claude 做 plan、repo understanding、safety check；
  - Codex 做 execution、patching、validation；
  - 双方互相 review plan、handoff 和结果。
- 本次首先尝试通过 GitHub CLI 检索公开仓库与代码案例，但当前环境不存在 `gh` 命令，因此无法直接走 GitHub CLI。
- 为弥补这一点，本次改用两条路径：
  1. 网页搜索 GitHub 公开结果；
  2. 调用独立研究代理梳理 Claude Code 与 Codex 官方文档中最接近该协作模式的能力证据。
- 本次调研得到的关键结论是：
  - 没有检到一个成熟、公开、直接把“Claude Code + Codex”同时写成固定工作流模板的代表性 GitHub 项目；
  - 但两边官方能力已经足够支撑这种协作方式自然拼接落地。
- Claude Code 侧可直接支持的协作能力包括：
  - common workflows
  - subagents
  - agent teams
  - hooks
  - code review
- Codex 侧可直接支持的协作能力包括：
  - CLI 本地执行
  - approval modes
  - local code review
  - automation
  - subagents
- 因此，本次形成的推荐协作原则不是“去照搬某个现成双工具项目”，而是：
  - Claude 作为总控、计划器和安全门；
  - Codex 作为受边界约束的执行器；
  - 每轮执行后回到 Claude 做验收、风险复核和下一轮 handoff。
- 针对当前科研仓库，本次整理出的推荐 SOP 为：
  1. Claude 先做 repo understanding、风险识别、计划拆解；
  2. Claude 输出边界清晰的 Codex handoff brief；
  3. Codex 只处理一个 bounded task，并返回 patch、validation evidence、blockers；
  4. Claude 对结果做 review，重点检查：
     - 是否改到 maintained code 而不是归档/输出副本；
     - 是否触及 split、protocol、event anchor、future horizon 等高风险定义；
     - 是否存在 look-ahead、time leakage、label leakage；
     - 验证证据是否足够支持进入下一步；
  5. 若通过，由 Claude 生成下一轮 handoff 或收口总结；
  6. 若不通过，由 Claude 重新收口并改写 brief 后再交给 Codex。
- 结合当前仓库特点，额外明确了分工建议：
  - Claude 更适合：计划、仓库理解、安全审查、实验公平性判断、结果归因、结构化交接；
  - Codex 更适合：有清晰边界的小修补、小工具、批量整理、局部验证、按模板回传结果；
  - 双方都不应自行无限扩 scope，必须通过 Claude 做每轮任务封装与验收。
- 当前判断：
  - 这个协作模式完全可落地；
  - 真正关键的不是继续搜索“有没有现成名字很响的项目”，而是尽快把 handoff 模板、回传模板、review checklist 固定下来。
- 推荐下一步：
  1. 写本仓库专用的 Claude→Codex handoff 模板；
  2. 写 Codex→Claude 结果回传模板；
  3. 写 review checklist，覆盖 active code、split、protocol、leakage、validation evidence。

### 补充记录（2026-04-02 晚，协作模式进一步澄清）
- 执行主体：混合协作（用户 + Claude）
- 用户进一步澄清：理想的 Claude + Codex 协作不应只是“Claude 思考、Codex 执行”。
- 用户真正希望的是：双方 agent 都参与思考、查错、review 计划与结果，形成双向审查和交叉验证，而不是把 Codex 降格成单纯执行器。
- 这一澄清会直接改变后续协作模板设计方向：
  - 不再采用单向流水线；
  - 改为“双向评审 + 定向执行”的结构；
  - 典型形态应是：Claude 先提出方案，Codex 先 review / 挑错 / 补充，再执行；执行后 Claude 做主审，必要时再回到 Codex 做二审或反驳检查。
- 当前新的设计目标应是：
  - 让两边都承担思考责任；
  - 让两边都能发现计划漏洞与实现漏洞；
  - 把双方差异当成交叉验证机制，而不是简单分工。
- 推荐下一步因此调整为：
  1. 设计“双向思考型”协作协议，而不是普通 handoff 模板；
  2. 明确哪些阶段必须双审（如 plan、high-risk code change、experiment interpretation）；
  3. 明确哪些阶段可由一方主做、另一方抽检（如小工具、批量整理、低风险文档工作）。
- 用户随后要求：在正式写这套协作协议之前，再做一轮更聚焦的 GitHub 复核，确认是否真的没有接近“Claude Code + Codex 双向思考、互相 review、互相挑错”的公开项目、模板或操作范式；如果有，应优先借鉴而不是重复造轮子。
- 因此，当前下一步不是立刻写协议，而是继续做一轮更有针对性的公开案例复核。

### 补充记录（2026-04-02 晚，GitHub 协同案例二次复核启动）
- 执行主体：Claude
- 用户担心过早下结论“没有现成案例”会导致重复造轮子，因此要求在写协作协议前，再做一次更聚焦的 GitHub 复核。
- 本轮复核目标不再是泛搜“Claude + Codex”，而是重点验证以下几类公开证据：
  1. GitHub 仓库中是否存在 Claude/Codex 双工具并用的 README、脚本、命令模板或工作流说明；
  2. 是否存在把 planner/reviewer 与 executor/reviewer 组合起来的多 agent coding workflow；
  3. 是否存在虽然不直接点名 Claude+Codex，但其结构可直接映射到当前需求的开源模板。
- 本轮复核完成前，暂不把“没有现成项目可借鉴”视为最终结论。
- 复核完成后，需要把结果分成三档给用户：
  - 可直接借鉴；
  - 可部分借鉴并改造；
  - 没有现成模板，只能基于能力拼接自建。
- 二次复核完成后，结论进一步明确：
  - 仍未找到一个可直接照搬的、公开成熟的“Claude Code + Codex 双向思考、互相 review、互相挑错”的 GitHub 成品模板；
  - 但找到了一批可以借结构、不能整套照抄的参考骨架，包括：
    - `MetaGPT`
    - `ChatDev`
    - `crewAI`
    - `LangGraph`
    - `AutoGen`
    - `SWE-agent`
    - `OpenHands`
    - `Aider`
    - `anthropics/claude-code`
    - `openai/codex`
- 这些候选分别可借：
  - 角色分工思路；
  - manager/worker/validator 编排；
  - issue -> patch -> test 的执行闭环；
  - agent 命令边界与本地执行接口；
  - 但没有任何一个项目已经把当前需要的“双向互审协议层”直接做成成品。
- 因此，当前判断从“可能没有现成模板”升级为更稳的结论：
  - 有很多可拼接的零件；
  - 没有可直接复用的整机；
  - 后续应采用“借骨架、自定义最小协议层”的策略，而不是继续泛搜现成成品。
- 在这个基础上，已开始产出本仓库专用的双向协作协议文档，目标不是脱离现有生态重新发明，而是把：
  - MetaGPT / ChatDev 的角色分工思想；
  - crewAI / LangGraph / AutoGen 的流程结构；
  - SWE-agent / OpenHands / Aider 的执行验证闭环；
  - Claude Code / Codex 的实际 agent 能力边界；
  汇总成一套适合当前科研仓库的最小可执行协议。

### 补充记录（2026-04-02 晚，双向思考协作协议成稿）
- 执行主体：Claude
- 在完成二次 GitHub 复核后，正式开始编写本仓库专用的 Claude + Codex 双向思考协作协议，而不再延迟等待一个并不存在的现成开源成品。
- 已新增协议文档：
  - `reports/claude_codex_bilateral_collaboration_protocol_20260402.md`
- 本次协议文档的核心改动方向是：
  - 明确不采用“Claude 想、Codex 做”的单向流水线；
  - 明确采用“Claude 提案，Codex 质疑；Codex 落地，Claude 审查；必要时 Codex 反审”的交叉验证结构。
- 协议中已包含的关键内容包括：
  - 基本原则
  - 角色主优势
  - 任务分级
  - 标准协作流程
  - 三个核心模板
  - 适配当前科研仓库的额外约束
  - 最小落地版本
- 当前这份协议已经足够作为后续实际协作的 v1 版本使用；如果用户认可，后续可继续把它拆成：
  - plan 双审模板
  - execution 前质疑模板
  - execution 后交叉 review 模板
  - 高风险任务双审规则清单
  并继续嵌入到 Claude / Codex 命令体系中。

### 补充记录（2026-04-02 晚，后续待做）
- 执行主体：Claude
- 当前最值得继续推进的，不再是泛化调研，而是把刚写好的协议继续工程化：
  1. 抽成可复用模板；
  2. 嵌入现有 Claude / Codex 协作命令；
  3. 结合当前科研仓库高风险点，做具体检查清单。
- 这样下一次真实任务到来时，不需要再口头解释协作方式，而能直接按协议执行。

### 补充记录（2026-04-02 晚，双向协作协议拆分为可直接复用模板）
- 执行主体：Claude
- 用户认可继续推进，因此本次没有停留在“大协议文档”层，而是继续把双向协作协议拆成可直接拿来用的模板与检查清单。
- 为避免每次真实任务再重新口头解释，本次新增了 4 份面向实际协作的落地文件：
  1. `reports/templates_claude_codex_plan_dual_review_20260402.md`
  2. `reports/templates_claude_codex_pre_execution_challenge_20260402.md`
  3. `reports/templates_claude_codex_post_execution_cross_review_20260402.md`
  4. `reports/checklist_claude_codex_high_risk_tasks_20260402.md`
- 这 4 份文件分别承担的作用是：
  - Plan 双审模板：用于 Claude 给出初版计划后，让 Codex 先做质疑与补洞，再由 Claude 收口；
  - Execution 前质疑模板：用于 Codex 在真正执行前，先检查理解、文件范围、风险点和验证是否充分；
  - Execution 后交叉 review 模板：用于 Codex 回传执行结果后，由 Claude 主审，必要时再触发 Codex 二审；
  - 高风险任务检查清单：用于 protocol / split / label / horizon / anchor / training path 等高风险任务的系统性复核。
- 这一步的意义在于：
  - 把“协作理念”变成“可复制执行动作”；
  - 让后续真实任务可直接按模板运行，而不是重新发明协作方式；
  - 把当前科研仓库最关键的高风险点（active code 路径、protocol/split、leakage、公平性、验证证据）固定成标准检查项。
- 当前状态：
  - 协议 v1 已有；
  - 4 个落地模板已就绪；
  - 下一步如果继续推进，最自然的方向是把这些模板嵌入现有 Claude / Codex 命令体系或直接用于第一轮真实任务试跑。
- 推荐下一步：
  1. 把这 4 个模板再压成“最常用短版”；
  2. 选一个中等风险真实任务做第一次协议试跑；
  3. 根据试跑结果，再反向修订协议和模板。

### 补充记录（2026-04-02 晚，启动第一次真实协议试跑）
- 执行主体：混合协作（用户 + Claude）
- 用户确认不再停留在文档层，决定直接用“当前 Claude/Codex 协作协议任务本身”作为第一次真实试跑案例。
- 这意味着当前任务不再只是我单独继续写模板，而是要真正把一个与当前主题高度相关的任务正式交给 Codex，按“双向思考 + 互相 review”的方式跑一次完整闭环。
- 本次试跑的目的不是为了让 Codex 单纯代劳，而是验证以下几点是否真的可执行：
  1. Claude 能否先给出明确计划与边界；
  2. Codex 能否先 review 计划，而不是直接执行；
  3. Codex 的输出能否包含风险、质疑和替代建议，而不是只有结果；
  4. Claude 能否对返回结果做主审和收口。
- 当前选定的试跑任务是：
  - 围绕“Claude + Codex 双向思考协作协议”本身，要求 Codex 按我们刚定义的想法先做 plan review，再提出如何进一步工程化落地这套协议。
- 这个任务被视为中等风险：
  - 不涉及 protocol / split / 训练主路径；
  - 但会影响后续整个协作体系设计，因此很适合做第一次流程验证。
- 当前下一步动作：
  - 由 Claude 按 Plan 双审模板生成一份面向 Codex 的真实 handoff brief；
  - 再让 Codex 先 review，再决定是否继续执行下一层落地建议。
- 这样做的价值是：
  - 协议第一次不是纸上推演，而是立即作用于自己；
  - 能最快发现模板是否太空、太长、太难执行，或者边界不够清晰。
- 试跑完成后，需要重点复盘：
  - Codex 是否真的先审计划；
  - 它提出的质疑是否有价值；
  - 这套协议是否还需要压缩、改写或补充触发条件。

### 补充记录（2026-04-02 晚，第一次真实试跑的 Claude 初版计划）
- 执行主体：Claude
- 为了让第一次试跑有明确边界，本次 Claude 已先形成执行前的任务定义：
  - 任务类型：medium-risk
  - 目标：让 Codex 先 review 当前双向协作协议与模板体系，指出是否仍有结构漏洞、执行摩擦点、冗余模板、缺失检查项，并给出更适合直接落地到真实任务中的精简建议。
  - 范围：只读为主，重点检查 `reports/claude_codex_bilateral_collaboration_protocol_20260402.md` 及 4 个模板文件；如有必要，可建议如何重组，但当前不要求它直接大改文件。
  - 禁区：不涉及 `datasetprocess/final_code`、protocol_config、split、训练脚本、run outputs、tmp、backup 等。
  - 验证期望：Codex 返回内容中必须包含：
    - 它认为当前方案合理的部分；
    - 它认为最需要修改的部分；
    - 它建议优先落地的最小实战版本；
    - 如果要质疑这套协议，它最质疑的一点是什么。
- 这次 handoff 的重点不是“让 Codex 写更多文档”，而是故意把它放在 reviewer / critic 位置，验证它是否真的能承担双向思考角色。
- 试跑完成后，Claude 需要基于其 review 再做一次主审收口，决定：
  - 是否接受其建议；
  - 是否改协议文档；
  - 是否进入下一轮工程化嵌入。

### 补充记录（2026-04-02 晚，第一次真实协议试跑交给 Codex 执行）
- 执行主体：混合协作（Claude -> Codex）
- 在完成上述边界定义后，已正式把这次真实试跑任务交给 Codex。
- 交付方式不是普通“帮我做完”，而是明确要求它：
  - 先 review 当前协议和模板；
  - 先指出结构问题和执行摩擦；
  - 再给出最小落地建议；
  - 并明确说明如果它要反对当前设计，最反对哪一点。
- 当前处于等待 Codex 返回结果阶段。
- 本轮返回后，Claude 将按 post-execution cross review 模板对其结果做主审，不会把它的结论直接当最终答案。
- 这一步完成后，才能真正判断这套双向协议是否已经从“概念成立”进入“流程成立”。

### 补充记录（2026-04-02 晚，双向协作协议与模板体系的批判性 review）
- 执行主体：混合协作（Codex critique，Claude 主审收口）
- 本轮没有继续扩写协议文档，而是按用户要求先对当前已经形成的协作体系做了一次批判性审阅，重点阅读了以下 5 个文件：
  - `reports/claude_codex_bilateral_collaboration_protocol_20260402.md`
  - `reports/templates_claude_codex_plan_dual_review_20260402.md`
  - `reports/templates_claude_codex_pre_execution_challenge_20260402.md`
  - `reports/templates_claude_codex_post_execution_cross_review_20260402.md`
  - `reports/checklist_claude_codex_high_risk_tasks_20260402.md`
- 同时补读了 `reports/project_progress_master.md` 中与协作协议形成过程直接相关的最近记录，重点回看了：
  - 从“先调研是否存在现成成熟范式”到“确认只能借骨架、自定义最小协议层”的判断链；
  - 从“大协议文档成稿”到“拆成 4 份模板与 1 份高风险清单”的演化过程；
  - 第一次真实试跑的设计目的，即检验这套体系是否真的能从概念走向流程。
- 这样做的原因是：当前风险不在于“文档写得不够多”，而在于协议体系可能已经开始进入“治理结构先于真实使用”的状态；如果不先做 critique，就容易继续叠模板、叠流程、叠检查项，最后造成落地摩擦。
- 本轮审阅后形成的核心判断包括：
  - 当前协议最合理的部分，是已经抓住了真正有价值的骨架：`先 plan review，再执行，再回传不确定项，最后主审收口`；这一点是可落地的，也是相比单向 handoff 真正有增益的部分。
  - 当前体系最明显的结构问题，不是原则错，而是“协议层”和“模板层”已经开始重复表达同一件事：计划审查、边界确认、验证要求、风险说明、主审收口，在协议正文、plan 模板、pre-execution 模板、post-review 模板、高风险 checklist 中都出现了不同版本。
  - 目前最容易产生执行摩擦的点，是一次中等风险任务若严格完整走全流程，需要经历：协议说明、plan 双审、pre-execution challenge、execution 回传、Claude 主审、必要时 Codex 二审；这对高风险任务合理，但对真实高频的中风险任务偏重，容易让使用者为了“合规走模板”而不是为了“更快暴露问题”去填表。
  - 当前“最小落地版本”写在协议里，但没有真正成为整个体系的默认入口；现实上更像是先有完整协议和 4 份模板，再由人记住“其实也可以只用短版”。这会导致系统默认姿态偏重。
  - 高风险清单本身是有价值的，但它仍然偏“检查项集合”，缺少一个更关键的判断门：什么情况下必须升级为高风险，什么情况下即使名义上没碰 `protocol_config` 也应视为高风险，比如会改变实验可比性、会改变数据选择边界、会改变结果解释口径但表面上没改核心配置。
  - 当前体系还缺少一次“试跑后反向裁剪”的明确闭环，也就是：真实任务跑完以后，不只是审任务结果，还要审协议本身，决定删掉哪些模板字段、哪些步骤可以合并、哪些检查只保留在高风险任务里。
- 基于以上发现，本轮更推荐的方向不是继续加文件，而是做最小化收缩：
  - 把真实默认工作流收敛为 1 个短版主模板 + 1 个高风险附加清单；
  - 将 `pre-execution challenge` 合并进 `plan review` 或作为其中一个必答段，而不是继续保留为独立文件；
  - 将 `post-execution cross review` 保留，但压缩为“结果 / 验证 / 不确定项 / 是否接受”四块，不再让中风险任务默认进入过长的来回审查。
- 下一步建议：
  1. 不要继续新增模板文件，先把现有 4 份模板压缩成真正的默认短版；
  2. 选一个真实的 medium-risk 任务，用“短版主模板 + 如有必要再挂高风险清单”的方式试跑一次；
  3. 试跑后专门输出一份“协议复盘”，只回答三个问题：哪些字段从头到尾没人真正用到、哪些步骤阻塞了推进、哪些检查项真的抓到了风险；
  4. 只有在试跑证明某一步确实持续产生价值后，才保留为标准动作，否则就应删掉。

### 补充记录（2026-04-02 晚，按 Codex 反馈收缩为默认短版工作流）
- 执行主体：Claude
- 在第一次真实试跑中，Codex 的批判性 review 给出的最重要反馈不是“方向错”，而是“默认形态开始变重”，尤其是：
  - 协议层和模板层重复；
  - `pre-execution challenge` 与 `plan review` 重叠太高；
  - medium-risk 任务如果默认跑完整套模板，会有明显流程摩擦。
- 基于这一反馈，本次没有继续新增更多长模板，而是直接做了结构收缩，目标是让“最小可行版本”真正变成默认入口，而不是藏在正文里的说明。
- 本次新增了 2 份新的收缩版文件：
  1. `reports/claude_codex_minimal_workflow_shortform_20260402.md`
  2. `reports/claude_codex_protocol_retrospective_template_20260402.md`
- 这两份文件的作用分别是：
  - `minimal_workflow_shortform`：把默认工作流收敛为一张短版任务卡，只保留：Claude 初版任务卡 -> Codex review -> Codex 执行/建议 -> Claude 收口；只有触发高风险条件时才外挂高风险清单。
  - `protocol_retrospective_template`：在真实任务跑完后，不是只复盘任务结果，而是专门复盘协议本身，判断哪些字段没人用、哪些步骤阻塞推进、哪些检查项真的有价值。
- 这一步的核心变化是：
  - 让“短版”从补充方案变成默认方案；
  - 让“高风险清单”从通用伴随文件变成按条件触发的附加层；
  - 让协议拥有“试跑后反向裁剪”的闭环，而不是只会越写越厚。
- 当前推荐的实际使用方式已经更新为：
  - 默认先用 `claude_codex_minimal_workflow_shortform_20260402.md`；
  - 只在高风险任务上附加 `checklist_claude_codex_high_risk_tasks_20260402.md`；
  - 每次真实试跑后，再用 `claude_codex_protocol_retrospective_template_20260402.md` 做协议复盘。
- 目前状态：
  - 大协议文档仍保留，作为背景和原则；
  - 旧的 4 份长模板暂时保留，作为完整版本参考；
  - 但默认入口已收缩为“短版主模板 + 高风险附加清单 + 复盘模板”。
- 推荐下一步：
  1. 选一个新的 medium-risk 真实任务，直接只按短版工作流试跑一次；
  2. 跑完后立即做协议复盘；
  3. 如果短版足够好用，再考虑把旧的长模板降级为附录或历史版。
### 补充记录：2026-04-02 晚，Claude API 链路切换留档与直连配置切换
- 执行主体：Codex
- 本次任务不是继续改动建模代码，而是处理当前科研仓库配套的 Claude/Codex 工具链路：用户要求先把“旧的特殊渠道 Codex API 工作流”完整记清楚，以便未来可随时切回，然后再把 Claude Code 改成直接走新购买的 Claude API。
- 先确认并固化了旧链路的真实结构，而不是只凭印象描述：
  - 项目内的 Codex 委派入口仍在 `F:\data_set_process\data_process\.claude\commands\codex-run.md` 与 `F:\data_set_process\data_process\.claude\commands\codex-workflow.md`；
  - 项目本地权限仍显式允许 `D:\ClaudeCode\codex-bridge\claude-codex-entry.ps1`；
  - 真正执行落在 `D:\ClaudeCode\codex-bridge\invoke-codex.ps1` -> `D:\ClaudeCode\codex-bridge\codex.exe`；
  - Codex 用户级 provider 仍来自 `C:\Users\Administrator\.codex\config.toml`，其 `base_url` 指向 `http://localhost:8317/v1`，`wire_api` 为 `responses`。
- 为了让未来回滚时不需要重新排查，本次新增了独立留档文件：
  - `F:\data_set_process\data_process\reports\claude_api_chain_switch_20260402.md`
- 该留档文件明确区分了两套链路：
  - 旧链路：`Claude 项目命令 -> codex-bridge -> codex.exe -> localhost:8317/v1 -> 特殊渠道 provider`
  - 新链路：`Claude Code -> C:\Users\Administrator\.claude\settings.json -> ANTHROPIC_* -> 新购买的 Claude API`
- 在不删除旧链路的前提下，新增了用户级 Claude Code 配置文件：
  - `C:\Users\Administrator\.claude\settings.json`
- 新配置采用：
  - `ANTHROPIC_AUTH_TOKEN`
  - `ANTHROPIC_BASE_URL=https://code.pchat.vip`
  - `ANTHROPIC_SMALL_FAST_MODEL=claude-3-5-haiku-20241022`
  - 默认 `model=claude-3-5-haiku-20241022`
- 这样做的原因是：
  - 用户提供的购买文档明确建议使用该渠道和该小模型；
  - 先用兼容性更高的固定模型把直连链路跑通，后续若该供应商支持更高模型，再单独放宽。
- 本次切换刻意保留了旧桥接链路，不做删除和覆盖：
  - 不删除 `codex-bridge`
  - 不删除项目内 `/codex-run`、`/codex-workflow`
  - 不改写 `C:\Users\Administrator\.codex\config.toml`
- 推荐下一步：
  1. 用最小非侵入命令验证 Claude Code 是否已能通过新 API 正常返回；
  2. 如果验证通过，再根据实际体验决定是否保留 Haiku 作为默认模型，或继续尝试该渠道是否支持更高模型；
  3. 如果未来要切回旧方案，优先查看 `reports/claude_api_chain_switch_20260402.md`，再决定是否撤销 `C:\Users\Administrator\.claude\settings.json`。
### 补充记录：2026-04-02 晚，Claude 新链路最小连通性验证通过
- 执行主体：Codex
- 在完成用户级 `C:\Users\Administrator\.claude\settings.json` 配置后，继续做了一次不改仓库代码的最小连通性验证，目的是确认这次切换不是“纸面切换”，而是实际可用。
- 使用的验证命令是：
  - `claude -p --setting-sources user --model claude-3-5-haiku-20241022 "请只回复：OK"`
- 验证结果：
  - Claude Code 已成功通过新 API 返回 `OK`；
  - 说明用户级直连链路已经生效。
- 同时做了一个对根域名的极简 HTTP 访问测试，但该测试并不能代表 Claude API 能力本身，因此不作为失败依据；真正有意义的判断以 `claude -p` 的实际返回为准。
- 推荐下一步：
  1. 后续如果继续使用该渠道，先保持 `claude-3-5-haiku-20241022` 为默认模型，确保稳定；
  2. 若需要更强模型，再单独验证该供应商是否支持更高档位模型名；
  3. 若未来要回滚，优先查看 `reports/claude_api_chain_switch_20260402.md` 中的旧链路结构和恢复入口。
### 补充记录：2026-04-02 晚，新 Claude 渠道更高模型兼容性实测
- 执行主体：Codex
- 用户在确认新链路已经打通后，进一步要求验证这套新购买的 Claude 渠道是否也支持比 Haiku 更高档的模型；本次不再靠文档猜测，而是直接做实际模型调用测试。
- 采用的测试方式是：
  - 使用已经生效的用户级 Claude Code 配置；
  - 分别显式指定模型名，执行最小命令 `claude -p --setting-sources user --model <MODEL> "请只回复：OK"`；
  - 观察是否能成功返回，并补充用调试日志确认请求中确实带上了对应模型名，而不是本地默认回退。
- 本次已实测成功返回 `OK` 的模型包括：
  - `claude-sonnet-4-6`
  - `claude-opus-4-6`
  - `sonnet`
  - `opus`
  - 以及此前已验证通过的 `claude-3-5-haiku-20241022`
- 为避免“CLI 别名成功但实际没切模型”的误判，又额外生成了调试日志并确认：
  - `tmp/claude_sonnet_debug.log` 中出现 `model=claude-sonnet-4-6`
  - `tmp/claude_opus_debug.log` 中出现 `model=claude-opus-4-6`
- 由此可以确认：
  - 该渠道不只支持截图文档里推荐的 Haiku 小模型；
  - 至少 Sonnet 和 Opus 档位也可以通过当前 Claude Code 新链路正常工作。
- 当前建议：
  1. 默认模型暂时仍保留 `claude-3-5-haiku-20241022`，因为它已明确写在供应商文档里，稳定性和成本更可控；
  2. 后续遇到更复杂的代码分析、计划、长上下文任务时，可以显式切到 `claude-sonnet-4-6`；
  3. `claude-opus-4-6` 已能使用，但更适合少量高复杂度任务按需启用，不建议在尚未观察该渠道稳定性前直接改成默认。
### 补充记录：2026-04-02 晚，Claude 默认模型切到 Sonnet
- 执行主体：Codex
- 在确认新渠道支持更高模型后，用户明确同意采用此前推荐方案，将 Claude Code 的默认模型改为 Sonnet。
- 实际修改时发现：
  - `C:\Users\Administrator\.claude\settings.json` 中的 `ANTHROPIC_BASE_URL` 已不再是之前写入的 `https://code.pchat.vip`，而是 `https://aixj.vip`；
  - 由于该变动很可能来自用户后续手动调整或其他有效配置，本次没有覆盖该地址，只在现有配置基础上新增默认模型字段。
- 本次实际修改为：
  - `C:\Users\Administrator\.claude\settings.json`
  - 新增 `model = "claude-sonnet-4-6"`
- 修改后做了不带 `--model` 参数的最小验证：
  - 命令成功返回 `OK`
  - 调试日志 `tmp/claude_default_model_debug.log` 中确认出现 `model=claude-sonnet-4-6`
- 由此可以确认：
  - 当前用户级默认模型已经不是 Haiku，而是 Sonnet；
  - 之后直接运行 `claude` 或 `claude -p --setting-sources user ...`，若不额外指定模型，将默认走 `claude-sonnet-4-6`。
- 推荐下一步：
  1. 先按当前 Sonnet 默认模型使用一段时间，观察该渠道在真实任务下的稳定性和速度；
  2. 如果需要更省钱或更快，可以临时显式切回 Haiku；
  3. 如果需要更强推理，再临时显式切到 `claude-opus-4-6`。
### 补充记录：2026-04-02 夜，恢复 Claude 侧的文献检索固定入口（方案 B）
- 执行主体：Codex
- 用户在切换到 Claude 直连 API 后，明显感觉之前“和 Claude 联合工作、搜索文献、接 ScholarAIO/Zotero”的流程像是丢了；本次没有把原因简单归结为模型变化，而是重新拆分了工作流层次，确认问题主要出在“API 已换，但原来的本地工具链入口没有作为固定默认入口被重新挂回 Claude”。
- 实际核对发现：
  - 项目级规则 `F:\data_set_process\data_process\CLAUDE.md` 仍在；
  - 项目级命令 `F:\data_set_process\data_process\.claude\commands\codex-run.md`、`codex-workflow.md` 等仍在；
  - Claude 项目记忆 `C:\Users\Administrator\.claude\projects\F--data-set-process-data-process\memory\MEMORY.md` 仍保存着“论文模型诊断要结合文献”等偏好；
  - 但文献检索和 Zotero 导入的老链路主要依赖 `ScholarAIO`、`tools/academic_search_to_zotero.py`、以及必要时的 `codex-bridge`，这些不是单靠 API 或默认模型就会自动恢复的。
- 基于用户选择的方案 B，本次没有再去改 API，而是把“Claude 对话 + 旧 ScholarAIO/Zotero/Codex 工具链执行”重新做成项目固定入口：
  - 新增项目命令：`F:\data_set_process\data_process\.claude\commands\literature-workflow.md`
  - 该命令明确规定：
    - 搜索优先走 `tools/academic_search_to_zotero.py` 或 ScholarAIO CLI；
    - 导入 DOI / URL / query 优先走本地 Zotero 工作流；
    - 多步骤、重执行、或明确要求协作时，再通过 `D:\ClaudeCode\codex-bridge\claude-codex-entry.ps1` 走 Codex bridge；
    - 文献类实质推进仍要先写 `reports/project_progress_master.md` 再给压缩总结。
- 同时为避免后续又退化成“普通网页搜索优先”，还做了两层补强：
  - 在 `F:\data_set_process\data_process\CLAUDE.md` 的 `Common Tasks Claude Should Help With` 中新增明确规则：本项目的文献检索、筛选、Zotero 导入应优先走恢复后的 ScholarAIO/Zotero 工作流，而不是先退回 generic web search；
  - 在项目级 Claude memory `C:\Users\Administrator\.claude\projects\F--data-set-process-data-process\memory\MEMORY.md` 中新增稳定偏好：本项目里找文献、筛文献、导入 Zotero 时优先用恢复后的 ScholarAIO/Zotero 老链路，较重联合任务再挂 Codex bridge。
- 这样做的结果是：
  - 保留了当前可用的 Claude 直连 API；
  - 同时把你习惯的文献工作流重新固定到了 Claude 侧；
  - 后续你不需要重新解释“别只给我普通网页搜索”，项目本身已经把这条路写清楚了。
- 推荐下一步：
  1. 之后实际要找文献时，优先直接在项目里使用新的 `/literature-workflow` 入口；
  2. 真实跑一轮后，再观察是否还存在“像忘了流程”的感觉；
  3. 若仍有遗漏，再继续把最常用的文献子流程拆成更细的固定命令，例如 search-only、import-only、synthesis-only。
### 补充记录：2026-04-02 夜，提高当前项目的 Claude 权限到 dontAsk
- 执行主体：Codex
- 用户明确要求提高当前项目内 Claude 的权限，并在三个方案中选择了“只改当前项目，把权限提高到几乎不再询问”的第二档，而不是修改全局配置。
- 在核对项目配置后确认：
  - 当前项目的本地高优先级设置文件是 `F:\data_set_process\data_process\.claude\settings.local.json`
  - 其中原本的 `defaultMode` 为 `acceptEdits`
  - 项目本身已经存在较多显式放行规则，包括 `codex-bridge`、`scholaraio`、`py`、`gh` 和部分 `WebFetch` 域名，因此把默认模式继续上调，主要影响的是“交互时是否频繁询问”，而不是重新定义整套 allow/deny 清单。
- 本次实际修改为：
  - 将 `F:\data_set_process\data_process\.claude\settings.local.json` 中的 `defaultMode` 从 `acceptEdits` 改为 `dontAsk`
- 这意味着：
  - 之后在这个项目里启动 Claude，会更倾向于不再对常见执行步骤反复发起询问；
  - 但项目里已有的 deny 区域、hook、以及本地规则文件仍然存在，并没有被删除。
- 本次没有做的事：
  - 没有修改全局 `C:\Users\Administrator\.claude\settings.json`
  - 没有删除 `.claude\settings.json` 里的 deny 规则
  - 没有移除编辑前保护 hook `protect-edit-targets.ps1`
- 推荐下一步：
  1. 后续直接在项目根目录开启新的 Claude 会话，让新的 `dontAsk` 模式自然生效；
  2. 如果仍觉得提示过多，再看是否要进一步改成更激进的项目级配置；
  3. 若发现权限过高导致不安心，可以随时把 `defaultMode` 改回 `acceptEdits`。

### 补充记录（2026-04-02 晚，重新识别当前论文模型的真实最新主线）
- 执行主体：Claude
- 用户指出：此前按 `primary_control_v2 / d2 / d3` 协议名去理解当前模型进展，可能不是项目里真正最新在推进的主线；希望重新按“最新真实进展”而不是旧文件名判断。
- 本次因此没有继续沿用旧协议线做推断，而是重新检查了以下证据：
  - `reports/project_progress_master.md` 当前顶部状态与 2026-04-02 记录；
  - `reports/group_meeting_model_progress_ppt_20260327/build_group_meeting_model_progress_ppt.py`；
  - `datasetprocess/final_code/model/training/protocol_allphase_control_v2_context_full2s/protocol_config.json`；
  - `reports/fair_baseline_same_pool_check_20260328/fair_baseline_same_pool_summary.md`；
  - `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260327_000432/run_summary.json`；
  - `tmp/single_output_d3_runs/EXP2_ALLPHASE_V2_CONTEXT_FULL2S_TRUE2S_SUP_20260324_224343/run_summary.json`。
- 重新核对后的关键判断是：
  - 当前最像“真实最新主线”的，不是早先讨论到的 `primary_control_v2 / d2 / d3` 协议命名线；
  - 当前更接近主线的是：基于 `allphase_control_v2_context_full2s` 样本池的 `event-conditioned trajectory` 路线；
  - 其中当前主推版本是 `deterministic conditioned v2`，而不是单纯 baseline，也不是已经切到 multi-hypothesis 主线。
- 支撑这一判断的直接证据包括：
  - `protocol_allphase_control_v2_context_full2s/protocol_config.json` 明确给出当前样本池为 `core_full2s_allphase`，`sample_count = 6238`，说明主训练池已经不是 earlier primary-only 小池；
  - `fair_baseline_same_pool_summary.md` 明确说明 baseline 与 conditioned v2 已在相同 sample pool、公平 split 下比较，并给出 conditioned v2 在 overall 2s RMSE、tail RMSE、interaction-slice tail RMSE 上优于 baseline；
  - 组会 PPT 构建脚本明确把研究主线写成：围绕“2 s 方向盘转角预测中的关键事件对齐”持续收敛，并把 `deterministic conditioned v2` 写成当前主推版本。
- 同时也澄清了一个容易误判的点：
  - `tmp/single_output_d3_runs/` 目录名里虽然带 `d3_runs`，但其中 formal baseline run 实际 run prefix 是 `ALLPHASE_V2_CONTEXT_FULL2S_TRUE2S_SUP`；
  - 因此不能仅凭目录名把当前主线误判为 `d3 protocol`。
- 关于样本量，本次也形成了一个更细的判断：
  - 总样本量层面，当前 allphase 主池 `6238` 已明显高于 earlier primary-only 方案，因此当前很多问题不再像“主因只是总样本太少”；
  - 但困难子集样本量仍可能偏少，例如 conditioned v2 formal run 中 interaction sample count 在 val/test 仍只有 `51 / 27`，因此后续要区分“总样本量够不够”和“关键困难切片样本量够不够”。
- 基于这些证据，后续诊断方向已经从“继续围绕旧协议名做三选一”调整为：
  1. 先围绕当前真正主线（allphase + conditioned v2）检查 protocol、formal run、fair comparison 与组会叙事是否一致；
  2. 再判断当前瓶颈更像：事件/标签定义问题、关键事件对齐问题、时序建模问题、训练/选择规则问题，还是评估口径限制；
  3. 诊断时把样本量拆成两层：总样本量 vs interaction / 难例子集样本量。
- 用户随后明确同意：后续分析按这条“真正最新主线”继续，不再被旧协议名字带偏；并再次强调，对于论文模型相关工作，每次有实质进展都应先记录到 `reports/project_progress_master.md`。
- 下一步建议：
  1. 直接围绕 conditioned v2 主线做一轮定向诊断，而不是再对旧协议名做抽象比较；
  2. 先梳理当前主线的训练脚本、protocol、run summary、fair comparison 和关键报告之间的一致性；
  3. 然后再判断最值得优先改的是关键事件对齐、标签定义、horizon 切分、还是更复杂的 multi-hypothesis / 交互建模。

### 补充记录（2026-04-02 夜，围绕 conditioned v2 主线开始做“代码+run+文献”联合诊断）
- 执行主体：Claude
- 用户进一步强调：后续不要只在仓库内部“闭门造车”，需要把文献视角也并入模型诊断，尤其关注：
  - 驾驶员反应时延；
  - 驾驶习惯 / 驾驶风格差异；
  - 个体差异；
  - 这些因素如何影响当前论文模型的下一步改进。
- 本次因此把诊断范围从“当前主线是什么”继续扩展为“当前主线为什么会卡住，以及文献是否支持这种卡点解释”。
- 本轮读取与核对的关键文件包括：
  - `reports/driver_reaction_modeling_group_meeting_plan_20260402.md`
  - `reports/v3_selection_conditioned_interaction_pilot_20260327/task_2_conditioned_v2/formal_eval/eval_summary.json`
  - `reports/event_plus_conditioned_trajectory_baseline_20260326/task_D_formal_run/eval_summary.json`
  - `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260327_000432/run_summary.json`
  - `tmp/interaction_multihyp_runs/EXP_INTERACTION_MULTIHYP_PILOT_FORMAL_20260327_010459/run_summary.json`
  - 同时额外调用了只读研究代理，补充公开文献中关于 reaction latency、driver heterogeneity 与多模态未来的解释框架。
- 当前形成的阶段性判断如下：
  1. **项目内的高层研究问题，已经明显开始从“固定 2 s 轨迹回归”转向“短时真实反应建模”**。
     - `driver_reaction_modeling_group_meeting_plan_20260402.md` 已经把现象总结得很清楚：前约 1 s 还有趋势一致性，后约 1 s 明显失真，且结果容易变得“平滑但不像真实人”。
     - 这份方案也已经明确提出：长时域固定 2 s 可能不是最优任务定义，应该考虑短时预测、动作表征重构、操作模式划分与协同控制落点。
  2. **但当前仓库里真正已经落地并跑到 formal comparison 的主线，仍然是 allphase + conditioned trajectory，而不是新的短时/分模式主线。**
     - 也就是说，项目在“研究叙事层”已经往真实反应建模转向，但“工程实现层”目前仍以 conditioned v2 为中心。
  3. **conditioned v2 相比 baseline 的主要收益，更像来自“关键结构/事件条件化”而不是纯回归误差的大幅下降。**
     - same-pool 公平比较里 conditioned v2 在 overall 2s RMSE、tail RMSE、turning count error、interaction-slice tail RMSE 上均优于 baseline；
     - 同时 `run_summary.json` 里仍保留明显的 event-related 指标，例如 `event_loss`、`main_peak_idx_mae`、`boundary_shift_abs_err`，说明当前主线其实已经不再是单纯“只看 trajectory MSE”。
  4. **当前最突出的结构性瓶颈，仍然是关键事件和边界对齐，而不是‘模型完全不会预测’。**
     - 组会 PPT 构建脚本和 conditioned v2 的 selection summary 都在强调 tail、turning、boundary、peak-time 这些结构指标；
     - 这与用户此前口头现象“前段还行，后段失真”是相互吻合的。
  5. **interaction-only multi-hypothesis pilot 暂时还不适合直接接管主线。**
     - 该 pilot 的 train/val/test count 仅为 `205 / 51 / 27`，明显只是 interaction 困难子集；
     - formal pilot 中 top1 test `overall_primary_steer_rmse = 0.5320`，明显劣于 conditioned v2；
     - 但 oracle test `overall_primary_steer_rmse = 0.4370`，明显优于 top1，且 `rmse_tail_abs_steer` 也从 `0.3875` 降到 `0.2562`；
     - 这说明“多解空间本身是有价值的”，但“当前 hypothesis selection / ranking 还不稳定”，所以它更像下一阶段可验证方向，而不是当前主推 formal 主线。
- 文献侧，本轮补充得到的直接启发包括：
  1. **驾驶员 reaction latency 不是常数，而是受预期性、复杂度和选择负荷影响的分布。** 这意味着当前固定 anchor 后直接要求所有样本同步进入可预测 future，可能把一部分潜伏期差异硬压成标签噪声。
  2. **极限工况下，驾驶员更像是在做短时局部纠偏决策，而不是输出一条平滑、单解、长时连续轨迹。** 这支持为什么当前 2 s full future 容易出现后段均值化。
  3. **驾驶风格和个体差异更像结构信号，而不是纯噪声。** 若快反应激进纠偏与慢反应温和纠偏被放在同一统一回归目标里，模型很容易学成平均行为。
  4. **长时域单一轨迹回归天然倾向均值化；多假设预测只有在 hypothesis ranking 足够稳定时才会真正转化成 top1 收益。** 当前 multihyp pilot 恰好印证了这一点：oracle 有空间，但 top1 还不稳。
- 基于当前证据，下一步的判断门槛已经更清楚：
  - 不是简单问“要不要直接换成更复杂 multihyp”；
  - 而是先判断 conditioned v2 当前的限制，究竟主要来自：
    1. event/anchor/边界定义仍不够贴近真实反应时延；
    2. interaction 困难子集样本偏少且异质性过强；
    3. 没有显式建模驾驶风格/个体差异；
    4. 还是 selection rule / top1 hypothesis ranking 不够好。
- 当前更偏向的中间结论是：
  - **总样本池已经不小，但困难 interaction 子集仍小；**
  - **conditioned v2 已经证明“事件条件化”是有效方向；**
  - **真正值得优先推进的，不一定是立刻全面切多假设，而更可能是：先把 reaction latency / style heterogeneity 明确纳入当前主线的标签解释、切片分析或条件输入中。**
- 下一步建议：
  1. 继续直接围绕 conditioned v2 主线，排查 event/anchor 与 peak/boundary 指标之间的误差来源；
  2. 结合文献，把“反应时延分布”和“驾驶风格差异”映射成可落地的实验：例如 latency-aware 切片、driver-style 条件输入、分驾驶员误差分析；
  3. 把 multihyp 保持为并行支线，优先研究 hypothesis selection 为什么没有把 oracle 优势转成 top1 优势；
  4. 再决定下一轮是先改 protocol/标签，还是先改模型结构。

### 补充记录（2026-04-02 夜，按既定协议启动 Claude/Codex 双向思考协作）
- 执行主体：Claude
- 用户确认：本轮后续分析可以正式按此前已经约定的 Claude/Codex 协作方式执行，不采用“Claude 只想、Codex 只做”的单向流水线，而是采用双向思考、互相 review、互相挑错的结构。
- 因此本轮任务的执行方式被进一步明确为：
  1. 先由 Claude 基于当前已核对的主线证据、run 结果和文献视角，收口出一个更清晰的诊断问题边界；
  2. 再将其中一部分交给 Codex 做交叉 review / 定向分析，重点不是单纯复述，而是补漏洞、挑问题、检验当前判断是否站得住；
  3. 最后再由 Claude 汇总 Codex 的补充意见，给出更稳的阶段性结论与下一步实验建议。
- 当前这一步的意义，不是增加流程感，而是刻意利用 Claude 与 Codex 在视角和偏好上的差异，减少闭门推断和单代理误判。
- 结合当前项目状态，本轮特别适合用双向协作的原因是：
  - 现在的问题已经不只是读一个脚本或看一个指标，而是在判断 conditioned v2 的真正瓶颈究竟更像 event/anchor 定义、latency 错位、驾驶风格异质性、还是 multihyp selection；
  - 这类问题很容易因为单一代理过早收敛到某一种解释，因此更适合做交叉 review。
- 当前下一步动作已经明确：
  - 继续沿当前真实主线（allphase + conditioned v2）推进；
  - 由 Claude 先收口出“下一轮最值得先做的 3 组实验/分析”；
  - 再让 Codex 从另一个角度 review 这些建议是否遗漏了关键风险、可比性问题或更省样本的替代方案。
- 推荐下一步：
  1. 先形成一版 Claude 侧的最小实验矩阵草案；
  2. 再交给 Codex 做 challenge/review；
  3. 最后再确定真正开跑的实验优先级。

### 补充记录（2026-04-02 夜，Codex 风格交叉审查当前最小实验矩阵）
- 执行主体：混合协作（Claude 提案，Codex 风格 reviewer challenge，Claude 收口）
- 在完成当前主线和文献视角的第一轮联合诊断后，本轮继续按既定协议推进：由 Claude 先提出一个“当前最小实验矩阵”方向，再刻意从 Codex 视角做批判性 challenge，而不是顺着原方案点头。
- 被审查的原始 3 组优先实验是：
  1. latency-aware slicing 与误差切片；
  2. 分驾驶员/风格误差分析，或 style-conditioned 输入；
  3. 保持 multihyp 为支线，重点分析为什么 oracle 优势没转成 top1 优势。
- Codex 风格 challenge 给出的最核心批评不是“方向完全错”，而是：
  - 这 3 组实验碰到了最可能的问题，但仍不够“干净”；
  - 主要缺口是：把“分析问题来源”和“改模型/改任务定义”混在同一张清单里；
  - 在真正改输入或改结构之前，还缺一个更公平、更省样本、也更贴当前主线的中间层实验。
- 这次交叉审查追加强调的“中间层实验”是：
  - **固定当前主线 protocol 和模型不变，只做事件对齐误差归因与重打分**；
  - 具体应围绕当前已有结构指标展开，而不是立刻发明新输入：
    - `main_peak_idx_mae`
    - `boundary_shift_abs_err`
    - `rmse_tail_abs_steer`
    - turning count / turning abs err
    - interaction vs non-interaction 切片
    - 可能的 latency proxy
    - driver / session 维度切片
- Codex 风格审查给出的更稳优先级排序是：
  1. **第 1 位：固定当前主线 protocol，不改训练，只做事件对齐误差归因 + latency-aware 误差切片。**
     - 这是最公平、最省样本、也最能直接回答“当前 conditioned v2 为什么仍卡住”的分析。
  2. **第 2 位：分驾驶员 / 分 session 误差分析，但先不做 style-conditioned 输入。**
     - 先确认 heterogeneity 是主因还是次因，再决定是否值得把 style 当作条件输入。
  3. **第 3 位：multihyp 支线只做 oracle-top1 gap 归因，不做主线升级。**
     - 当前它的首要价值是揭示 hypothesis selection / ranking 问题，而不是证明它已经适合接管主线。
  4. **第 4 位：只有在前 1–3 项都支持时，才考虑最小化的 style-conditioned 或 latency-aware task variant。**
     - 这一步已经会开始改输入或改任务定义，必须单列为新 protocol 变体，而不能与当前 formal fairness comparison 混在一起。
- 这轮交叉审查特别强调的高风险点包括：
  1. **latency-aware slicing 可作为分析切片，但如果直接拿 future 派生 latency 反向改训练输入或筛样本，就会把诊断偷偷变成任务重定义。**
  2. **driver/style 分析本身安全，但若把 driver ID / style embedding 直接喂给模型，容易变成身份记忆器，并与当前输入集合不再可比。**
  3. **oracle 不是 deployable metric，multihyp 若只拿 oracle 证明“多解空间存在”是可以的，但不能直接用来对打当前 deterministic 主线。**
  4. **任何把 anchor、future start、valid boundary 直接改掉的实验，本质上都已经在改任务定义，而不是单纯提升当前模型结构。**
- Claude 收口后的当前中间结论是：
  - 原来的 3 组实验方向仍然成立，但应先插入一个“事件对齐误差归因表”作为真正的第一步；
  - 也就是说，现在最该优先做的不是马上上新结构，而是先把当前 conditioned v2 已经产出的结构指标用足；
  - 先证明：当前尾段失真到底主要由 peak/boundary 错位驱动、由 latency 分布驱动，还是由 driver heterogeneity 驱动。
- 这一步带来的实际方法论变化是：
  - 从“先想一个可能有效的新模型改法”
  - 收缩为“先把当前主线里已经可见的错误结构做归因，再决定该改标签、改分析、还是改结构”。
- 推荐下一步：
  1. 优先形成一版基于当前主线输出的 `事件对齐误差归因表` 设计；
  2. 再设计最小化的 latency-aware / driver-aware 分析切片；
  3. multihyp 只保留为 oracle-top1 ranking failure analysis 支线；
  4. 在这些证据出现之前，先不要把 style-conditioned 输入或新的任务变体当成正式主实验。

### 补充记录（2026-04-02 夜，开始收口“事件对齐误差归因表”的可执行分析设计）
- 执行主体：Claude
- 在完成主线识别、文献并入和 Codex 风格交叉审查后，用户同意继续往下走，因此本轮开始把“该先做什么”从抽象优先级，收口成可直接执行的分析设计。
- 当前决定优先推进的不是新模型结构，也不是新的 protocol 变体，而是：
  - **固定当前主线 protocol 与模型不变**；
  - 基于现有 conditioned v2 输出，先做一版 `事件对齐误差归因表`；
  - 目标是把 tail 失真、peak 错位、boundary 错位、latency 错位、driver heterogeneity 之间的关系拆清楚。
- 这样做的原因是：
  1. 当前 same-pool fairness 已经成立，继续直接换模型容易把问题来源和结构增益混在一起；
  2. 当前主线已有足够多的结构指标（peak、boundary、tail、turning、interaction slice），还没有被系统地交叉利用；
  3. 若不先做这一步，后续无论是 style-conditioned 输入、latency-aware 任务变体，还是 multihyp 升级，都容易缺乏干净依据。
- 本轮接下来将明确：
  - 误差归因表按哪些维度切片；
  - 每个维度必须看哪些指标；
  - 哪些变量只能作为诊断切片、不能直接进训练；
  - 最终如何根据结果判断下一步更适合改标签、改分析口径、还是改模型结构。
- 推荐下一步：
  1. 先形成 `事件对齐误差归因表` 的字段与切片设计；
  2. 再决定是否需要写一个小分析 helper 放到 `tools/`；
  3. 只有当归因结果明确支持时，才进入新的 style / latency / multihyp 实验。

### 补充记录（2026-04-03，先把当前分析结果整理成更直观的可读文件）
- 执行主体：Claude
- 在继续执行“方案 A / 方案 B”之前，用户提出一个很实际的反馈：当前分析虽然已经逐步收敛，但直接在对话里看仍然比较繁琐，希望先把本轮判断用更形象、更易浏览的方式整理出来，最好生成一个单独文件。
- 这个反馈意味着：当前阶段不应立刻继续往下展开更多分析细节，而应先做一次中间层收口，把已经得到的关键判断、优先级和下一步顺序，以更直观的方式输出，降低后续沟通与决策成本。
- 因此本轮新增动作调整为：
  1. 先把当前主线识别结果、核心问题、风险点、优先实验顺序、以及“为什么暂时不要急着上新结构”整理成一份更易读的摘要文件；
  2. 文件形式优先采用人可直接阅读的报告/路线图，而不是继续只在聊天中解释；
  3. 待用户先通过这个中间收口版本形成统一理解后，再继续推进后续的字段设计或数据来源路径核对。
- 推荐下一步：
  1. 在 `reports/` 下生成一份面向当前模型问题的可视化/结构化摘要文件；
  2. 文件中优先突出：当前真正主线、主要瓶颈、不要急做的事、最推荐先做的分析顺序；
  3. 用户确认这份中间总结后，再继续进入具体执行方案。

### 补充记录（2026-04-03，用户选择先执行方案 B：对接现有结果文件的数据来源）
- 执行主体：Claude
- 在生成 `reports/current_model_diagnosis_visual_summary_20260403.md` 之后，用户没有要求继续抽象化成新的流程图，而是明确选择继续执行“方案 B”。
- 这意味着当前优先任务从“继续解释概念”切换为“回到仓库实际文件，确认事件对齐误差归因表能从哪里拿数据”。
- 本轮目标被收口为：
  1. 不急着写新分析脚本；
  2. 先盘点当前仓库里已有的 run summary、eval summary、诊断输出、CSV 或 artifacts，确认哪些字段已经存在；
  3. 再把这些字段按‘可直接复用 / 需二次加工 / 当前缺失’三类整理出来，作为后续是否写 helper 的依据。
- 这样做的原因是：
  - 当前项目里已经存在较多 run outputs、formal eval 和 artifacts；
  - 若不先盘点现有字段来源，就直接设计新表或写新脚本，容易重复造轮子；
  - 方案 B 更贴近当前仓库实际，也更有利于判断后续工作量和最小实现路径。
- 推荐下一步：
  1. 优先核对 conditioned v2 formal run、baseline formal run、multihyp pilot formal run 及其附近 formal_eval/artifacts；
  2. 明确哪些字段已经有样本级来源，哪些目前只有汇总级来源；
  3. 在此基础上再决定是否需要新增一个小分析 helper 放到 `tools/`。

### 补充记录（2026-04-03，方案 B 第一轮盘点：事件对齐误差归因表的现有字段来源）
- 执行主体：Claude
- 按用户已选定的方案 B，本轮没有直接新写分析脚本，而是先围绕当前真实主线附近的现有 artifacts 做字段来源盘点，目标是回答：后续想做的“事件对齐误差归因表”，哪些字段已经现成可用，哪些只能通过 join/加工得到，哪些当前仍缺。
- 本轮重点核对的文件/结果包括：
  - `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260327_000432/run_summary.json`
  - `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260327_000432/metrics.json`
  - `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260327_000432/sample_manifest_used.csv`
  - `tmp/single_output_d3_runs/EXP2_ALLPHASE_V2_CONTEXT_FULL2S_TRUE2S_SUP_20260324_224343/run_summary.json`
  - `tmp/interaction_multihyp_runs/EXP_INTERACTION_MULTIHYP_PILOT_FORMAL_20260327_010459/test_top1_sample_metrics.csv`
  - `tmp/interaction_multihyp_runs/EXP_INTERACTION_MULTIHYP_PILOT_FORMAL_20260327_010459/test_oracle_sample_metrics.csv`
  - `reports/v3_selection_conditioned_interaction_pilot_20260327/task_3_interaction_multihyp/formal_eval/interaction_sample_level_comparison.csv`
  - `reports/style_probe_artifacts/` 附近的 style 相关 CSV / JSON。
- 当前盘点后的核心结论是：
  1. **multihyp 支线的样本级结构误差字段最完整。**
     - 已直接发现 top1/oracle 的 sample-level CSV；
     - 其中已经包含 `rmse_2s_abs_steer`、`rmse_tail_abs_steer`、`tail_trend_corr`、`tail_direction_match`、`boundary_shift_abs_err`、`turning_count_abs_err`、`first_reversal_time_abs_err_s`、`peak_time_abs_err_s`、`extrema_count_abs_err`；
     - 同时还自带 `sample_key`、`subj`、`eval_morphology_label`、`interaction_slice`、`reversal_slice`、`effective_mechanism_tag` 等切片字段。
  2. **conditioned v2 主线已有较完整的样本元数据，但当前直接看到的更多是 summary/aggregate，而不是同等丰富的 sample-level error CSV。**
     - `sample_manifest_used.csv` 已提供后续 join 所需的大量主键和样本属性：`sample_key`、`subj`、`split`、`phase_type`、`recording_id`、`anchor_s`、`mechanism_tag`、`eval_morphology_label`、`structure_slice`、`reversal_slice`、`interaction_slice` 等；
     - `run_summary.json` / `metrics.json` 已证明主线确实有 `main_peak_idx_mae`、`boundary_shift_abs_err` 等结构指标；
     - 但本轮附近盘点中，尚未直接找到与 multihyp 那样现成展开到每个样本的结构误差 CSV，因此 conditioned v2 若要做完整归因表，大概率需要再定位隐藏输出、或从现有 eval 产物二次导出。
  3. **baseline 线当前更像“汇总指标充分、样本级误差表不明显”。**
     - allphase baseline formal run 一侧已确认有 manifest、metrics_long、若干机制/形态切片汇总；
     - 但本轮第一眼能直接用来做样本级误差归因的 CSV 不如 multihyp 明确。
  4. **style / heterogeneity 相关 artifacts 已经存在，不是从零开始。**
     - 已确认 `driver_style_vectors.csv`、`session_style_features.csv`、`prior_session_style_vectors.csv`、`probe_manifest_with_style.csv`、`style_probe_metrics.json` 等文件存在；
     - 这说明后续若要把 driver/session/style 作为归因切片，不一定要先新建特征，可先尝试和主线 manifest / sample_key 体系对齐。
- 因此，本轮对“事件对齐误差归因表”的字段来源可先分成三类：
  - **可直接复用：** multihyp sample-level 结构误差字段；conditioned v2 manifest 的样本元数据字段；已有 interaction/morphology/reversal 切片字段；style artifacts 中的 driver/session/style 侧信息。
  - **需二次加工：** conditioned v2 主线的样本级结构误差表（目前更像藏在 eval 过程里，需要导出或重拼）；baseline 与 conditioned v2 的逐样本可比表；latency proxy 相关字段。
  - **当前仍不明确/可能缺失：** 一个直接可读、统一 schema 的“conditioned v2 deterministic sample-level attribution table”成品文件。
- 这一步带来的实际推进是：
  - 已经可以较有把握地说，后续若要先做归因而不改训练，并不是完全没数据可用；
  - 真正的卡点主要不是字段概念设计，而是：**如何把 conditioned v2 主线的样本级结构误差抽出来，并与 manifest/style/multihyp 对齐成统一表。**
- 推荐下一步：
  1. 继续定点查找 conditioned v2 formal eval 附近是否已有未注意到的 sample-level metrics CSV；
  2. 若没有，就把“从现有 eval 输出导出 conditioned v2 sample-level 结构误差”列为最小 helper 目标；
  3. helper 写法应优先保持只读/重组，不改训练协议和主模型逻辑。

### 补充记录（2026-04-03，方案 B 第二轮追查：conditioned v2 样本级误差文件已定位）
- 执行主体：Claude
- 在用户同意继续追查 conditioned v2 样本级指标文件后，本轮继续针对 `event_conditioned_runs` 与 `reports/v3_selection_conditioned_interaction_pilot_20260327/task_2_conditioned_v2/formal_eval/` 做定点搜索，目标是确认：此前未直接看到的 conditioned v2 sample-level error files 是否其实已经存在于 formal_eval 报告目录。
- 本轮直接定位到的关键文件包括：
  - `reports/v3_selection_conditioned_interaction_pilot_20260327/task_2_conditioned_v2/formal_eval/conditioned_trajectory_sample_metrics.csv`
  - `reports/v3_selection_conditioned_interaction_pilot_20260327/task_2_conditioned_v2/formal_eval/conditioned_event_sample_metrics.csv`
  - `reports/v3_selection_conditioned_interaction_pilot_20260327/task_2_conditioned_v2/formal_eval/baseline_trajectory_sample_metrics.csv`
  - `reports/v3_selection_conditioned_interaction_pilot_20260327/task_2_conditioned_v2/formal_eval/baseline_event_sample_metrics.csv`
  - `reports/v3_selection_conditioned_interaction_pilot_20260327/task_2_conditioned_v2/formal_eval/sample_level_comparison.csv`
  - `reports/v3_selection_conditioned_interaction_pilot_20260327/task_2_conditioned_v2/formal_eval/eval_summary.json`
- 这一步非常关键，因为它直接修正了上一轮的暂时判断：
  - **conditioned v2 的样本级结构误差文件并不是缺失，而是已经存在于 formal_eval 报告目录，只是上一轮主要在 run 目录附近盘点时没有直接命中。**
- 当前已确认 `conditioned_trajectory_sample_metrics.csv` 中，逐样本字段已经非常接近“事件对齐误差归因表”的核心需求，包含：
  - 主键/切片类：`split`、`seed`、`local_idx`、`subj`、`sample_key`、`phase_type`、`road_type_anchor`、`mechanism_tag`、`is_curve`、`structure_slice`、`structure_heavy`、`valid_future_len`、`eval_morphology_label`、`interaction_slice`、`reversal_slice`、`effective_mechanism_tag`
  - 轨迹/尾段误差类：`rmse_2s_abs_steer`、`rmse_pre_tail_abs_steer`、`rmse_tail_abs_steer`、`tail_pre_gap_abs_steer`、`tail_pre_ratio_abs_steer`、`late_mean_abs_err_steer`
  - 趋势/形状类：`trend_corr`、`tail_trend_corr`、`shape_corr`、`tail_shape_corr`
  - 结构/事件对齐类：`direction_match`、`tail_direction_match`、`tail_slope_abs_err`、`boundary_slope_abs_err`、`boundary_shift_abs_err`、`turning_count_abs_err`、`turning_has_reversal_match`、`first_reversal_time_abs_err_s`、`peak_time_abs_err_s`、`peak_abs_amp_err`、`range_abs_err`、`extrema_count_abs_err`
- 同时，`conditioned_event_sample_metrics.csv` 还补充了更细的事件级字段，例如：
  - `event_name`
  - `presence_acc`
  - `time_abs_err_s`
  - `direction_acc`
  - `support_true`
  - `support_matched`
  - 以及 `interaction_slice`、`structure_slice`
- 更重要的是，`sample_level_comparison.csv` 已经不是单模型单表，而是：
  - 按 `sample_key` 对齐 baseline 与 conditioned 的逐样本结果；
  - 同时直接给出 `delta_rmse_2s_abs_steer`、`delta_rmse_tail_abs_steer`、`delta_tail_trend_corr`、`delta_turning_count_abs_err`、`delta_peak_time_abs_err_s`、`delta_boundary_shift_abs_err` 等差值字段；
  - 因此它已经可以直接支持“哪类样本上 conditioned v2 变好/变坏、到底主要改善了什么结构指标”的归因分析。
- 基于这轮结果，当前关于字段来源的判断需要更新为：
  1. **conditioned v2 主线并不缺样本级结构误差字段，且 baseline 与 conditioned 的逐样本比较表也已存在。**
  2. **事件对齐误差归因表的最小版本，已经基本不需要从训练代码里重新导出，可直接从 formal_eval CSV 重组。**
  3. **后续真正可能仍需补的，不是主干误差字段，而更像是额外分析维度：driver/session/style join、latency proxy、以及与 multihyp top1/oracle 的统一对接。**
- 这意味着当前最小实现路径又进一步收缩：
  - 不是先去改训练脚本；
  - 也不一定需要先写“导出 conditioned v2 指标”的 helper；
  - 更合理的是优先写一个**只读聚合/重组 helper**，把：
    - conditioned trajectory sample metrics
    - conditioned event sample metrics
    - baseline-conditioned sample-level comparison
    - manifest/style artifacts
    - （后续可选）multihyp sample metrics
    统一整理成一张更适合做切片归因的分析表。
- 推荐下一步：
  1. 优先基于 `task_2_conditioned_v2/formal_eval/` 这些现成 CSV，定义一版归因总表的最小 schema；
  2. 再决定是否写一个轻量 helper 放到 `tools/` 仅做 join/重命名/字段筛选；
  3. 暂时不要动训练主线，因为当前缺口已经不在训练输出，而在分析表重组层。

### 补充记录（2026-04-03，收口事件对齐误差归因总表的最小 schema）
- 执行主体：Claude
- 在确认 conditioned v2 / baseline 的 sample-level CSV 已存在后，本轮继续把“下一步做什么”从“继续找文件”收口成“应该怎样组织一张真正可分析的归因总表”。
- 当前判断是：后续如果直接拿已有多个 CSV 分别做切片，仍会比较散；更稳的方式是先定义一张统一 schema，再决定是否用一个小 helper 只做重组。
- 这张归因总表当前建议采用 **单样本一行（sample-level wide table）** 的最小结构，而不是事件级长表作为主表。原因是：
  1. 当前主要问题是解释为什么某类样本在 tail / boundary / peak 上改善或恶化；
  2. baseline-conditioned comparison 已天然是 sample-level；
  3. driver/session/style/latency proxy 等后续扩展也更容易挂在 sample_key 上；
  4. event-level metrics 更适合作为附表或展开表 join，而不是主表骨架。
- 因此，最小 schema 建议分成 6 个字段层：
  1. **主键层（identity）**
     - `sample_key`
     - `split`
     - `subj`
     - `local_idx_baseline` / `local_idx_conditioned`（如需要回溯原表）
  2. **样本属性层（sample attributes）**
     - `phase_type`
     - `road_type_anchor`
     - `mechanism_tag`
     - `effective_mechanism_tag`
     - `eval_morphology_label`
     - `is_curve`
     - `valid_future_len`
     - `structure_slice`
     - `structure_heavy`
     - `interaction_slice`
     - `reversal_slice`
  3. **baseline 指标层（baseline metrics）**
     - 以 `_baseline` 结尾保留原字段命名，例如：
       - `rmse_2s_abs_steer_baseline`
       - `rmse_tail_abs_steer_baseline`
       - `tail_trend_corr_baseline`
       - `boundary_shift_abs_err_baseline`
       - `turning_count_abs_err_baseline`
       - `peak_time_abs_err_s_baseline`
       - `first_reversal_time_abs_err_s_baseline`
       - `extrema_count_abs_err_baseline`
  4. **conditioned 指标层（conditioned metrics）**
     - 以 `_conditioned` 结尾保留原字段命名，例如：
       - `rmse_2s_abs_steer_conditioned`
       - `rmse_tail_abs_steer_conditioned`
       - `tail_trend_corr_conditioned`
       - `boundary_shift_abs_err_conditioned`
       - `turning_count_abs_err_conditioned`
       - `peak_time_abs_err_s_conditioned`
       - `first_reversal_time_abs_err_s_conditioned`
       - `extrema_count_abs_err_conditioned`
  5. **差值归因层（delta attribution）**
     - 直接保留 comparison 表中已有 delta 字段：
       - `delta_rmse_2s_abs_steer`
       - `delta_rmse_tail_abs_steer`
       - `delta_tail_trend_corr`
       - `delta_turning_count_abs_err`
       - `delta_peak_time_abs_err_s`
       - `delta_boundary_shift_abs_err`
     - 同时建议补两个派生标签字段：
       - `improved_overall_flag`（如 `delta_rmse_2s_abs_steer < 0`）
       - `improved_tail_flag`（如 `delta_rmse_tail_abs_steer < 0`）
     - 这两个 flag 只作为分析辅助列，不改变原指标。
  6. **可选扩展层（optional joins, nullable）**
     - 当前先预留但允许为空：
       - `driver_style_cluster`
       - `driver_style_vector_id`
       - `session_style_cluster`
       - `latency_proxy_bucket`
       - `multihyp_top1_rmse_tail_abs_steer`
       - `multihyp_oracle_rmse_tail_abs_steer`
       - `multihyp_oracle_gap_tail`
- 关于事件级字段，本轮建议采用“附表”而不是塞进主表全部展开：
  - 推荐保留一个单独的 `event-level table`，主键至少包括：
    - `sample_key`
    - `model_name`
    - `event_name`
  - 核心字段包括：
    - `presence_acc`
    - `time_abs_err_s`
    - `direction_acc`
    - `support_true`
    - `support_matched`
  - 后续只有在需要回答“是哪一类事件先错、哪些事件时间对齐最差”时，再和主表按 `sample_key` 聚合或展开。
- 当前最推荐的最小实现顺序也因此更清晰：
  1. 先把 `sample_level_comparison.csv` 作为主骨架；
  2. 再按 `sample_key` 补来自 manifest / style 的附加字段；
  3. event-level CSV 暂不完全扁平化进主表，只在需要时做二级 join；
  4. multihyp top1/oracle 字段放在下一阶段再并入，避免第一版 schema 过重。
- 这样做的好处是：
  - 第一版归因总表已经足够回答当前最关键的问题：conditioned v2 相对 baseline 在哪些样本、哪些结构切片上变好或变坏；
  - 同时又不会把 driver/style/latency/multihyp 全部一次性混进来，保持分析过程干净。
- 推荐下一步：
  1. 若用户同意，就按这版最小 schema 写一个只读 helper 到 `tools/`；
  2. helper 第一版只输出 sample-level attribution master table，不做训练、不改协议；
  3. 后续再视需要增量接入 event-level 附表与 multihyp gap 字段。

---

### 2026-04-03 Attribution Master Table 脚本完成与首次运行

- **执行人**: Claude (延续 04-02 schema 设计)
- **所做工作**:
  1. 按照 04-02 设计的 6 层 schema，编写了只读聚合脚本 `tools/build_attribution_master_table.py`。
  2. 脚本读取 5 个源 CSV（comparison、manifest、event_conditioned、baseline_event、driver_style），join 成两张输出表：
     - `reports/attribution_master_table.csv` — sample-level 宽表（749 行 × 100 列）
     - `reports/attribution_event_table.csv` — event-level 附表（4494 行 × 10 列）
  3. 使用 `F:\python3.11\python.exe` 成功执行，输出验证通过：行数与 comparison 一致（749），join 无膨胀，核心字段零 NaN。
  4. 脚本为纯只读，不修改任何训练代码、协议配置或源数据。

- **关键产出与发现**:
  - **整体改善**: conditioned v2 在 tail RMSE 上有 56.1% 样本优于 baseline（420/749），overall RMSE 约 50.2% 改善。
  - **被试差异显著**: cwh 改善率最高（64.3%），gf 最弱（50.2%），tyy 居中（56.6%）。提示：个体差异是 tail 退化的主要混淆因子之一。
  - **interaction 切片效果**: interaction 样本改善率 74.1%（27 样本），远高于 unknown 54.2%（655 样本）。非交互也有 67.2%，说明 conditioned 信号不只在交互场景有效。
  - **latency proxy 发现**: Q1（快反应）样本 delta_rmse_tail 均值为 +0.015（恶化），Q3/Q4（慢反应）改善显著（-0.136 / -0.044）。提示：对于反应快的事件，conditioned 信号可能引入干扰。
  - **boundary_shift 恶化**: 在 reverse_correction（+0.107）和 single_lobe（+0.182）形态下，boundary_shift 明显变大。说明 conditioned 虽然改善了 tail 整体 RMSE，但在转向边界判定上可能引入偏移。
  - **Event table**: 包含 conditioned（event_conditioned_baseline）和 baseline（unconditional_baseline）两个模型的 3 种事件（first_major_turn_onset, first_reversal, main_peak），每种 749 × 2 = 1498 条记录，总计 4494 行。

- **产出文件**:
  - `tools/build_attribution_master_table.py` — 可复现的只读聚合脚本
  - `reports/attribution_master_table.csv` — 749 行 × 100 列 sample-level 归因宽表
  - `reports/attribution_event_table.csv` — 4494 行 × 10 列 event-level 附表

- **推荐下一步**:
  1. **快反应样本恶化归因**：进一步分析 Q1_fast 桶中 conditioned 为什么恶化——是事件信号噪声、anchor 时间太近导致预测窗口不够、还是 conditioned 在短 latency 场景过拟合？
  2. **boundary_shift 专题分析**：检查 reverse_correction / single_lobe 形态下 boundary_shift 恶化是否集中在特定被试或特定触发类型。
  3. **event-level 交叉分析**：用 event table 回答"哪种事件类型的时间对齐最差"、"presence_acc 是否因被试不同而有差异"等问题。
  4. **multihyp gap 字段接入**：在归因表基础上接入 multi-hypothesis top1/oracle 指标，量化多假设是否能弥补 boundary_shift 恶化。

### 补充记录：2026-04-03，Claude 直连渠道切换到 xuedingtoken

- **执行人**: Codex
- **所做工作**:
  1. 根据用户提供的新渠道截图，调整了用户级 Claude Code 配置文件 `C:\Users\Administrator\.claude\settings.json`。
  2. 本次只切换 Claude 直连接口，不修改 Codex 老链路，也不改默认模型。
  3. 同时补入该渠道文档建议的两个环境变量：
     - `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`
     - `CLAUDE_CODE_ATTRIBUTION_HEADER=0`

- **变更范围**:
  - 已切换：
    - `ANTHROPIC_BASE_URL` 指向新的 `xuedingtoken` 渠道
    - `ANTHROPIC_AUTH_TOKEN` 更新为新渠道密钥
  - 保持不变：
    - 默认模型仍为 `claude-opus-4-6`
    - `C:\Users\Administrator\.codex\config.toml` 仍保留旧的 Codex 特殊链路
    - 项目内 `codex-bridge` 相关命令与 ScholarAIO / Zotero 工作流不受影响

- **原因**:
  - 用户希望试用另一家本质相似的 Claude 代理渠道；
  - 当前阶段优先完成切换本身，不额外做探针请求，避免立即产生新的计费记录。

- **推荐下一步**:
  1. 在项目根目录重新启动一次 `claude` 会话，让新的用户级配置自然生效；
  2. 若要验证是否成功切换，再做一次最小探针请求并对照新渠道后台；
  3. 如果后续要回切，优先查看 `reports/claude_api_chain_switch_20260402.md` 中的渠道切换记录。

### 补充记录：2026-04-05，Claude 直连渠道回切到 aixj

- **执行人**: Codex
- **所做工作**:
  1. 根据用户要求，将 Claude 直连渠道从 `xuedingtoken` 回切到上一个已验证可计费、可正常返回的 `aixj` 渠道。
  2. 只修改了用户级 Claude 配置文件 `C:\Users\Administrator\.claude\settings.json`。
  3. 本次没有再做最小探针测试，目的是优先恢复稳定工作状态，避免继续产生测试计费。

- **变更范围**:
  - 已恢复：
    - `ANTHROPIC_BASE_URL` 指回 `https://aixj.vip`
    - 对应恢复为此前可用的 Claude 直连密钥配置
  - 已移除：
    - `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`
    - `CLAUDE_CODE_ATTRIBUTION_HEADER`
  - 保持不变：
    - 默认模型仍为 `claude-opus-4-6`
    - `C:\Users\Administrator\.codex\config.toml` 仍保留旧的 Codex 特殊链路
    - 项目内 ScholarAIO / Zotero / Codex 协作入口不受影响

- **原因**:
  - 用户反馈 `xuedingtoken` 渠道在实际对话时明显卡住，虽然网络层可达且请求可发出，但没有稳定流式返回，因此不适合作为当前默认工作渠道。

- **推荐下一步**:
  1. 在项目根目录重新开启一个新的 `claude` 会话，让回切后的用户级配置生效；
  2. 后续如需再次验证，可用一次极小探针请求确认 `aixj` 已恢复；
  3. 暂时不要把 `xuedingtoken` 当作默认工作渠道，除非后续单独排查出其兼容性问题。

### 补充记录：2026-04-07，Claude 直连渠道再次切回 xuedingtoken

- **执行人**: Codex
- **所做工作**:
  1. 按用户要求，将 Claude 直连渠道从 `aixj` 再次切回 `xuedingtoken`。
  2. 只修改用户级 Claude 配置文件 `C:\Users\Administrator\.claude\settings.json`。
  3. 没有执行新的探针请求，避免切换瞬间继续产生额外测试计费。

- **变更范围**:
  - 已切换：
    - `ANTHROPIC_BASE_URL` 指向 `https://xuedingtoken.com`
    - `ANTHROPIC_AUTH_TOKEN` 换回对应的 `xuedingtoken` 渠道密钥
    - 重新加入：
      - `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`
      - `CLAUDE_CODE_ATTRIBUTION_HEADER=0`
  - 保持不变：
    - 默认模型仍为 `claude-opus-4-6`
    - `C:\Users\Administrator\.codex\config.toml` 仍保留旧的 Codex 特殊链路
    - 项目内 ScholarAIO / Zotero / Codex 协作入口不受影响

- **原因**:
  - 用户当前优先目标是重新试用 `xuedingtoken` 渠道，因此先完成配置切换，不在本次动作中继续争论稳定性问题。

- **推荐下一步**:
  1. 在项目根目录重新开启一个新的 `claude` 会话，让新的用户级配置真正生效；
  2. 如需确认是否切换成功，再用一次最小探针请求单独验证；
  3. 若再次出现明显卡顿，再决定是否回切到 `aixj` 或继续排查 `xuedingtoken` 的兼容性。

### 补充记录（2026-04-07，基于指定背景文件收口 conditioned v2 主线判断与下一步优先级）

- **执行主体**：Claude
- **为什么做**：
  - 用户明确要求本轮先不扩展到整个仓库，也不先做代码实现，而是只基于 4 份指定背景文件收口当前研究主线、已验证进展、关键瓶颈和下一步最值得优先推进的问题。
  - 本轮目标不是再发散方案，而是判断：在尽量少改当前主线的前提下，下一轮最值得先做什么分析或最小实验。

- **本轮读取与参考范围**：
  1. `reports/project_progress_master.md`
  2. `reports/current_model_diagnosis_visual_summary_20260403.md`
  3. `reports/driver_reaction_modeling_group_meeting_plan_20260402.md`
  4. `reports/fair_baseline_same_pool_check_20260328/fair_baseline_same_pool_summary.md`

- **当前收口判断**：
  1. **当前真正主线**不是泛泛的“驾驶员轨迹预测”，而是基于 `allphase_control_v2_context_full2s` 样本池的 `event-conditioned trajectory / deterministic conditioned v2`。
  2. **conditioned v2 是当前主推版本**，不是 baseline，也不是 multi-hypothesis 主线；multi-hyp 目前更适合作为旁路诊断线，而不是直接接管主线。
  3. **conditioned v2 相比 baseline 的收益已经被公平验证**：在 same-pool / same-split 前提下，overall 2s RMSE、tail RMSE、turning count abs err、interaction-slice tail RMSE 均优于 baseline，因此“事件条件化”本身是有效方向。
  4. **当前最核心瓶颈**不是“模型完全不会预测”，而是关键结构仍不稳，主要体现在：
     - peak timing 错位；
     - boundary 对齐偏移；
     - tail 修正段失真；
     - 快反应样本上 conditioned 信号可能引入干扰；
     - driver/style 异质性可能在混淆主结果。
  5. **multi-hyp 的当前证据**更支持“存在多解空间，但 top1 ranking 没接住 oracle 优势”，因此暂时不适合直接扶正为主线。

- **对下一步优先级的收口**：
  1. **第一优先**：先做 `conditioned v2` 的快反应样本恶化归因与事件对齐误差专题分析。
     - 核心想回答：Q1_fast 或相近 latency proxy 样本中，退化主要由 `boundary_shift`、`peak_time` 还是 `tail` 形状误差驱动。
     - 这一步优先级最高，因为它最贴近当前主线瓶颈，且不需要先改训练。
  2. **第二优先**：做 driver / session / style 维度误差切片，但先停留在分析层。
     - 核心想回答：主线收益不稳定到底是不是少数 driver/style 主导，还是普遍现象。
     - 当前不建议直接把 driver ID 或 style embedding 喂回模型，否则会把“分析判断”与“身份记忆效应”混在一起。
  3. **第三优先**：只在不改 protocol、不改 split、不改 anchor 的前提下，做现有结果的短时窗口重算或最小化验证。
     - 目的不是马上改任务，而是先确认：当前 conditioned v2 的有效信息是否主要集中在前半段，还是问题更像 timing / boundary 错位。

- **当前明确不建议立刻推进的事项**：
  1. 不建议马上让 multi-hypothesis 接管主线；
  2. 不建议现在就上 style-conditioned / driver embedding；
  3. 不建议现在直接改 protocol / split / label / horizon / anchor；
  4. 不建议一边改任务定义一边改模型结构再一起比较。

- **原因**：
  - 上述动作都可能显著影响实验可比性、训练主路径或论文叙事干净程度；
  - 当前更需要的是把主线里已经存在的结构误差与切片信息用足，再决定下一轮究竟该改分析、改任务还是改模型。

- **如后续交给 Codex，建议的最小可执行任务**：
  1. 只读分析 `attribution_master_table.csv` 与相关 formal_eval CSV，专做“快反应样本退化归因”；
  2. 只读分析 driver/style 切片，判断个体异质性是否为主因；
  3. 只对现有预测结果做短时切窗重算，不改训练、不改 protocol。
  - 以上 3 类都属于**低风险或低到中风险**任务；
  - 但一旦触及 protocol、split、label、horizon、anchor、训练主路径或实验可比性，就应视为**高风险任务**，不能在没有额外确认的情况下直接推进。

- **推荐下一步**：
  1. 若要最快获得下一轮修改方向，优先把“快反应样本退化归因”交给 Codex 做成一个最小只读分析任务；
  2. 该任务返回前，应先把详细进度写入 `reports/project_progress_master.md`；
  3. 等快反应 / boundary / peak 错位关系更清楚后，再决定是继续深挖分析，还是进入最小化任务改动实验。

### 补充记录：2026-04-08，conditioned v2 快反应退化与 boundary 恶化专题归因
- **执行主体**：Codex
- **做了什么**：
  1. 读取并核对了 `reports/attribution_master_table.csv`（749 × 100）与 `reports/attribution_event_table.csv`（4494 × 10）的字段名、样例行和关键分类字段，确认 `latency_proxy_bucket` 可直接用于识别 `Q1_fast`。
  2. 在只读前提下新增分析脚本 `tools/conditioned_v2_fast_boundary_attribution_20260408.py`，集中完成以下统计：
     - `Q1_fast` vs 非 `Q1_fast` 的关键指标对比；
     - `Q1_fast` 中 `delta_rmse_tail_abs_steer > 0` 样本的结构指标相关性与条件均值差排序；
     - `eval_morphology_label × subj` 的 `delta_boundary_shift_abs_err` 交叉统计；
     - `gf / cwh / tyy` 三位被试的 baseline / conditioned `boundary_shift` 分布对比；
     - event-level 中 `first_major_turn_onset` / `main_peak` 的 `time_abs_err_s` 对比，以及同一样本的 `conditioned - baseline` 差值汇总。
  3. 运行脚本并生成以下产物：
     - `reports/conditioned_v2_fast_boundary_attribution_20260408.md`
     - `reports/conditioned_v2_q1fast_summary_20260408.csv`
     - `reports/conditioned_v2_boundary_event_summary_20260408.csv`
  4. 对生成结果做了二次复核，特别核查了：
     - `Q1_fast` 上 tail 指标是否确实退化；
     - `boundary_shift` 恶化是否主要集中在特定 `subj × eval_morphology_label`；
     - event-level 时间对齐是否支持 “Q1_fast 因对齐更差而退化” 这一解释。

- **为什么做**：
  - 当前 deterministic conditioned v2 主线已经有 attribution master table 的初步发现，但还缺少对 `Q1_fast` 退化、`boundary_shift` 恶化切片和被试差异的定向归因。
  - 本轮任务限定为只读分析，目标不是修改训练或任务定义，而是把“问题究竟集中在 boundary、timing，还是 tail shape / amplitude”这个判断收紧，为后续最小化验证提供依据。

- **发现了什么**：
  1. `Q1_fast` 的 `delta_rmse_tail_abs_steer` 均值为 `+0.0155`，而非 `Q1_fast` 为 `-0.0345`，说明快反应桶的 tail 退化确实存在。
  2. 但 `Q1_fast` 的 `delta_boundary_shift_abs_err` 均值仅 `+0.0316`，低于非 `Q1_fast` 的 `+0.0842`；`delta_peak_time_abs_err_s` 也没有转成明显更差，因此 `Q1_fast` 退化并不是由更重的 boundary / peak timing 恶化主导。
  3. 在 `Q1_fast` 且 `delta_rmse_tail_abs_steer > 0` 的样本内，如果只看任务要求的结构性 delta 指标，相关性最高的是 `delta_peak_time_abs_err_s`（Pearson `r=0.267`），其次是 `delta_turning_count_abs_err`（`r=0.240`），而 `delta_boundary_shift_abs_err` 几乎不相关（`r=0.027`）。
  4. 若扩展到 conditioned 结构指标，最强信号来自：
     - `shape_corr_conditioned` 下降（`|r|=0.621`）
     - `peak_abs_amp_err_conditioned` 升高（`|r|=0.599`）
     这说明 `Q1_fast` tail 退化更像 shape / amplitude 失配，而不是 boundary 漂移。
  5. `boundary_shift` 恶化并非锁定在单一被试，而是更强地集中在 morphology：
     - `single_lobe × cwh` 的 `delta_boundary_shift_abs_err` 均值最高，为 `+0.2202`；
     - `single_lobe × gf` 次之，为 `+0.2034`；
     - 三位被试在 `single_lobe` 与 `reverse_correction` 上都呈正向恶化；
     - `multi_correction` 明显更轻，其中 `cwh` 甚至接近不恶化。
  6. 被试层面上，三位被试的 `boundary_shift_abs_err_conditioned` 分布都整体右移；按 `delta_boundary_shift_abs_err` 均值排序为：
     - `cwh`: `+0.1249`
     - `gf`: `+0.0655`
     - `tyy`: `+0.0460`
     这与已有 subject heterogeneity 观察一致，但仍不是单一被试独占现象。
  7. event-level 时间对齐不支持 “Q1_fast 因 conditioned 带来的时间对齐更差而退化”：
     - `first_major_turn_onset` 上，`conditioned - baseline` 在 `Q1_fast` 为 `-0.0078`，仍是改善；
     - `main_peak` 上，`conditioned - baseline` 在 `Q1_fast` 为 `-0.0137`，同样是轻微改善而非恶化。

- **推荐下一步**：
  1. 继续保持只读分析，优先抽取 `Q1_fast` 中 `peak_abs_amp_err_conditioned` 高、`shape_corr_conditioned` 低的代表样本，做 trajectory 级可视化核查，确认是否存在尾段幅值不足、过冲或回摆失真。
  2. 针对 `single_lobe` 与 `reverse_correction` 分别追加小样本可视化，判断 `boundary_shift` 恶化更像边界提前/滞后，还是边界附近局部幅值与斜率失真。
  3. 若后续需要进入最小化验证，建议先围绕 “tail shape / amplitude 失配” 这一方向组织假设，而不是先把问题归结为 boundary 或 event timing。

### 补充记录：2026-04-08，代表样本可视化总览面板
- **执行主体**：Codex
- **做了什么**：
  1. 回看 04-08 的归因记录与 `reports/conditioned_v2_fast_boundary_attribution_20260408.md`，确认当前最值得优先让用户“直接看图”的不是再做一轮更抽象的统计，而是把已经存在的 formal-eval 单样本曲线图，重新排成一版更贴当前结论的代表样本面板。
  2. 核对了 `reports/v3_selection_conditioned_interaction_pilot_20260327/task_2_conditioned_v2/formal_eval/figures/` 中现有的样本图，确认当前 formal run 没有像 multihyp pilot 那样直接保存整套 `baseline / conditioned / GT` 预测序列 `npz`，因此本轮优先复用已经生成好的单样本 PNG，而不是冒然重跑主线或猜测缺失序列来源。
  3. 新增只读脚本 `tools/build_conditioned_v2_representative_panel_20260408.py`，从 `reports/attribution_master_table.csv` 读取关键指标，并把 5 个代表样本重新组合为一张总览图，同时为每个样本补上：
     - `latency_proxy_bucket`
     - `eval_morphology_label`
     - `delta_rmse_tail_abs_steer`
     - `delta_boundary_shift_abs_err`
     - `shape_corr_conditioned`
     - `peak_abs_amp_err_conditioned`
     - 一句面向当前归因结论的解释性 caption
  4. 运行脚本并生成：
     - `reports/conditioned_v2_representative_cases_20260408.png`
     - `reports/conditioned_v2_representative_cases_20260408.md`

- **为什么做**：
  - 用户已经能够理解统计名词后，下一步最需要的是“把统计结论落到肉眼可见的具体样本上”，尤其是区分：
    - `Q1_fast` 中的 tail shape / amplitude 失配；
    - `single_lobe` / `reverse_correction` 中的 morphology-driven boundary 问题；
    - 以及作为对照的 conditioned 改善样本。
  - 在 formal run 未直接保留完整逐样本预测序列文件的情况下，先复用现有 formal-eval 单样本图，是风险最低且最快能让用户看见问题结构的做法。

- **发现了什么**：
  1. 当前可直接复用的代表样本里，`tyy ... #65` 是最贴 `Q1_fast` 退化结论的现成坏例：
     - `latency_proxy_bucket = Q1_fast`
     - `eval_morphology_label = single_lobe`
     - `delta_rmse_tail_abs_steer = +1.525`
     - `shape_corr_conditioned = -0.747`
     - `peak_abs_amp_err_conditioned = 1.947`
     它非常适合作为“快反应尾段 shape / amplitude 失真”的可视化主例。
  2. `tyy ... #54` 与 `cwh ... #6` 分别提供了：
     - `single_lobe` 下 boundary 偏移更明显的例子；
     - `reverse_correction` 下相对温和但仍恶化的对照例子。
  3. `tyy ... #52` 作为明显改善样本被保留下来，作用不是证明“所有样本都好”，而是提醒后续判断应聚焦“为什么有些形态/时延会坏”，而不是误判为 conditioned 主线整体无效。

- **推荐下一步**：
  1. 先让用户直接查看 `reports/conditioned_v2_representative_cases_20260408.png`，用肉眼确认这几类失真模式是否与统计结论一致。
  2. 若用户希望继续深挖，再进入下一轮更细的 trajectory 级可视化，但那一步需要继续确认原始预测序列文件是否能从现有 artifact 重建，或是否只对少量样本做最小化补绘。

### 补充记录：2026-04-08，基于归因结果的机制判断与下一轮实施方案

- **执行主体**：Claude
- **为什么做**：
  - 04-08 的 Codex 归因分析已经产出了 Q1_fast 退化归因、boundary_shift morphology 交叉表、event-level 时间对齐核查等关键结果。
  - 用户要求 Claude 基于这些结果给出研究判断和可落地的实施方案，而不是继续发散或重复背景。

- **做了什么**：
  1. 通读了归因报告、代表样本报告、两份 CSV 汇总表和代表样本面板图。
  2. 深入阅读了训练脚本的 event conditioning 实现（context 5 维向量 + uniform broadcast 到全部 T_out 时间步）和 loss 函数结构（MSE + amplitude L1 + derivative losses，其中 W_PEAKTIME=0.0 和 W_REVSEQ=0.0 被禁用）。
  3. 基于数据证据和代码结构，形成了对 Q1_fast 退化和 boundary_shift 恶化的机制判断。
  4. 设计了 5 步实施方案，明确了每步的验证目标、停止条件和风险等级。

- **核心机制判断**：
  1. **Q1_fast tail 退化主因**：context embedding 被 uniform broadcast 到全部 400 个未来时间步。Q1_fast 样本的 anchor 处 steer_rate/amplitude 偏高，前段动态集中但 tail 段已收敛，uniform broadcast 导致 tail 段幅值失配。次因是 time-step weighting 基于 steer_rate，对 tail 段权重不足。
  2. **boundary_shift 恶化主因**：MSE loss 对 single_lobe 的尖锐边界产生 hedging 平滑效应，conditioned context 加剧了边界附近的不确定性。次因是 peak timing loss 和 reversal probability loss 被禁用，缺少对边界结构的显式约束。

- **实施方案摘要**（详见本次对话输出）：
  - Step 1（低风险）：逐时间步误差曲线，验证退化是否集中在 tail 段
  - Step 2（低风险）：context 向量值域分析，验证 Q1_fast context 是否系统性偏高
  - Step 3（低风险）：边界斜率对比，验证 boundary 恶化是平滑还是平移
  - Step 4（中风险）：基于 Step 1-3 结论选择单一 loss 修改（tail amplitude penalty / 启用 W_PEAKTIME / context time-decay）
  - Step 5：决策节点，根据 Step 4 结果决定继续优化还是进入任务重定义讨论

- **明确不做的事项**：不启用 multi-hyp 接管主线、不改 anchor/horizon/split/protocol、不同时改多个 loss 权重、不在诊断完成前跳到修改实验。

- **推荐下一步**：
  - 优先将 Step 1 和 Step 3 交给 Codex 并行执行（两者都是只读分析，但都依赖逐样本预测序列文件的可用性）。
  - 如果预测序列文件缺失，需要先用现有 checkpoint 做一次只读推理保存序列，再执行分析。
  - Step 2 可以独立于 Step 1/3 执行，因为它只需要 dataset 中的 context 向量，不需要预测序列。

### 补充记录：2026-04-08，预测序列缺失确认与 3 个 Codex 任��拆分

- **执行主体**：Claude
- **做了什么**：
  1. 系统排查了 conditioned v2 formal run 附近的所有 artifact：
     - `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260327_000432/` 只有 5 个文件：`loss_history.csv`、`metrics.json`、`run_summary.json`、`sample_manifest_used.csv`、`selection_comparison.csv`
     - 没有 `.pt`、`.pth`、`.npz`、`.npy` 等 checkpoint 或预测序列文件
     - baseline formal run 同样没有 checkpoint
     - `run_summary.json` 引用的 `init_checkpoint` 路径文件已不存在
  2. 排查 conditioned v2 训练脚本来源：
     - 维护代码区 `final_code/model/training/` 中仅存 `future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`，该脚本是 base 训练脚本，不含 `structured_v2` conditioning
     - grep 整个仓库未找到包含 `conditioning_mode`、`event_embed_dim=96`、`structured_v2` 的 `.py` 文件
     - 但在 `__pycache__/` 中发现了对应 `.pyc` 文件
  3. 最终在 git 历史 commit `418f869`（"Add v3 conditioned trajectory and interaction pilot"）中找到完整源码：
     - `conditioned_trajectory_head.py`（162行）— `ConditionedTrajectoryHead` 类，含 `structured_v2` mode
     - `run_event_conditioned_trajectory_baseline.py`（748行）— 训练脚本，支持 `--conditioning-mode baseline` 和 `structured_v2`
     - `eval_event_conditioned_trajectory.py`（815行）— 评估脚本，计算 sample-level metrics 并生成图表
     - 以及 `event_conditioned_baseline_model.py`、`event_head.py`、`event_targets.py` 等支撑模块
  4. 确认 eval 脚本在推理时收集了 `preds`/`trues` 数组，但没有保存为文件，只用来计算 metrics
  5. 基于以上发现，拆分出 3 个 Codex 任务并编写了结构化 handoff brief

- **为什么做**：
  - 5 步实施方案的 Step 1（逐时间步误差曲线）和 Step 3（边界斜率对比）都依赖逐样本预测序列
  - 必须先确认这些序列是否可用，否则后续分析无从下手
  - 同时需要给出绕过方案（用现有 CSV 近似）和彻底方案（重训+保存序列）

- **发现了什么**：
  1. **Checkpoint 和预测序列完全缺失**：conditioned v2 和 baseline formal run 的所有 `.pt` 文件已被删除
  2. **训练脚本源码已从工作树删除**：但在 git commit `418f869` 中完整保留，可通过 `git show` 恢复
  3. **模型架构关键差异**：
     - 维护代码中的 `Past2FutureMultiTaskRoadPreview` 是 base 模型，context = [steer, steer_rate, ay, yawrate, style_id] + z_veh（7 dim）
     - conditioned v2 额外有 `ConditionedTrajectoryHead` 层，实��� event-level structured conditioning（gaussian/sigmoid tracks, gate, residual scale）
  4. **context 值域分析可绕过缺失**：不需要预测序列，可直接从 vehicle 原始数据 + sample_manifest 提取 anchor 处信号值
  5. **近似 timestep/boundary 分析可绕过缺失**：用已有 `rmse_pre_tail` vs `rmse_tail`、`boundary_slope_abs_err` 等 CSV 字段近似

- **产出文件**：
  - `tools/codex_handoff_task1_context_analysis.md` — 任务 1 handoff brief
  - `tools/codex_handoff_task2_approx_analysis.md` — 任务 2 handoff brief
  - `tools/codex_handoff_task3_reconstruct_conditioned_v2.md` — 任务 3 handoff brief

- **推荐下一步**：
  1. 任务 1 和 2 可并行交给 Codex 立即执行（低风险，只读）
  2. 任务 3 单独交给 Codex（中风险），需先恢复源码再训练
  3. 任务 1+2 返回后由 Claude 做主审，判断近似分析是否足以回答当前问题
  4. 任务 3 返回后，用预测序列执行 Step 1 和 Step 3 的精确版本
  5. 三个任务全部完成后，综合结果决定进入哪种最小 loss 修改实验


### Supplemental Record: 2026-04-08, Task 1 Context / Anchor Signal Value Range Analysis
- **Executor**: Codex
- **What was done**:
  1. Read `tools/codex_handoff_task1_context_analysis.md` and inspected the concrete inputs: `sample_manifest.csv`, `reports/attribution_master_table.csv`, the maintained training script context build at lines 722-729, and representative raw vehicle CSV schemas.
  2. Implemented the read-only analysis script `tools/attribution_context_value_range_analysis.py`.
  3. In the script, loaded test-split samples from the manifest, grouped samples by vehicle CSV, computed `steer_rate` with `np.gradient(steer, 1/200)`, extracted anchor-time `steer`, `steer_rate`, `ay`, and `yawrate`, and joined `latency_proxy_bucket`, `eval_morphology_label`, and `delta_rmse_tail_abs_steer` from the attribution master table.
  4. Generated grouped statistics for `latency_proxy_bucket`, direct `Q1_fast` vs `non_Q1_fast` comparison, second-level `latency_proxy_bucket x eval_morphology_label` breakdown, and Pearson correlations against `delta_rmse_tail_abs_steer`.
  5. Ran the script with `F:\python3.11\python.exe` and produced:
     - `reports/context_value_range_by_latency_bucket_20260408.csv`
     - `reports/context_value_range_by_latency_bucket_20260408.md`
- **Why it was done**:
  - The immediate thesis/model question was whether `Q1_fast` tail degradation could be explained by systematically larger anchor-point context signals, which would support the hypothesis that uniform context broadcast amplifies tail mismatch.
- **What was found**:
  1. All 749 test samples were extracted successfully, and every sample matched exactly by manifest `anchor_idx` with zero anchor-time error.
  2. `Q1_fast` did not show a uniform four-signal elevation. Compared with `non_Q1_fast`, it had only slightly higher `abs(steer_rate)` (`+0.1322` mean), while `abs(steer)` (`-0.0672`), `abs(ay)` (`-0.7351`), and `abs(yawrate)` (`-0.0277`) were lower.
  3. Across all 749 test samples, Pearson correlations between anchor-signal magnitudes and `delta_rmse_tail_abs_steer` were weak overall, with max `|r| = 0.1304`.
  4. Within the 188 `Q1_fast` samples, the strongest single relationship was `abs(steer)` with `|r| = 0.2759`, which is still only moderate.
  5. The result does not support a broad “Q1_fast has uniformly stronger raw anchor context” explanation. At most, the data supports a narrower `steer_rate`-intensity difference, not a consistent multi-signal anchor-value gap.
- **Recommended next step**:
  1. Shift the next read-only diagnosis toward mechanisms beyond raw anchor magnitudes, especially context broadcast behavior over time or structured conditioning effects.
  2. Prioritize the next analysis on time-localized trajectory error or boundary/tail shape behavior for high-tail-error `Q1_fast` samples, because anchor signals alone explain little of the observed variance.

### Supplemental Record: 2026-04-08, Task 2 Approximate Timestep / Boundary Analysis From Existing CSVs
- **Executor**: Codex
- **What was done**:
  1. Read `tools/codex_handoff_task2_approx_analysis.md` and verified the available inputs: `reports/attribution_master_table.csv`, the baseline / conditioned formal-eval sample metric CSVs, and `sample_level_comparison.csv`.
  2. Implemented the read-only analysis script `tools/attribution_approx_timestep_boundary_analysis.py`.
  3. In the script, aligned the baseline and conditioned sample-metric CSVs against `sample_level_comparison.csv` on `sample_key` and confirmed exact equality for the reused metrics (`rmse_pre_tail_abs_steer`, `rmse_tail_abs_steer`, `tail_slope_abs_err`, `boundary_slope_abs_err`, `boundary_shift_abs_err`, `peak_abs_amp_err`, `shape_corr`).
  4. Computed approximate front-vs-tail concentration metrics:
     - `tail_to_front_ratio = rmse_tail_abs_steer / rmse_pre_tail_abs_steer`
     - `delta_tail_to_front_ratio = conditioned_ratio - baseline_ratio`
     - `delta_front_rmse` and `delta_tail_rmse`
     - grouped summaries by `latency_proxy_bucket` and by `eval_morphology_label x latency_proxy_bucket`
  5. Computed approximate boundary-mode diagnostics:
     - `delta_boundary_slope_abs_err`
     - `delta_boundary_shift_abs_err`
     - `delta_tail_slope_abs_err`
     - `delta_peak_abs_amp_err`
     - morphology-level summaries and boundary-mode counts for `single_lobe` and `reverse_correction`
  6. Built the required `Q1_fast x single_lobe` view, including a case table plus a scatter of `peak_abs_amp_err_conditioned` vs `boundary_shift_abs_err_conditioned`, with point color by `subj` and point size driven by `shape_corr_conditioned`.
  7. Ran the script with `F:\python3.11\python.exe` and produced:
     - `reports/approx_timestep_boundary_analysis_20260408.csv`
     - `reports/approx_timestep_boundary_analysis_20260408.md`
     - `reports/approx_boundary_slope_shift_scatter_20260408.png`
     - `reports/approx_q1fast_single_lobe_amp_boundary_scatter_20260408.png`
- **Why it was done**:
  - Raw prediction sequences are currently missing, so the original per-timestep and boundary-slope plan could not be executed directly.
  - This task was the low-risk workaround: reuse the existing formal-eval sample metrics to answer the same causal questions approximately, without touching training code or rerunning experiments.
- **What was found**:
  1. `Q1_fast` degradation is **not tail-only overall**. Its mean front RMSE delta is `+0.0299`, while its mean tail RMSE delta is `+0.0155`, and the mean tail/front ratio drops from `1.2623` to `1.2207` (`delta = -0.0417`).
  2. The clearest tail-focused worsening sits in the `Q1_fast x single_lobe` intersection:
     - mean front delta `+0.0263`
     - mean tail delta `+0.0562`
     - mean ratio delta `+0.0461`
     - tail-driven share `65.4%`
  3. `Q1_fast x reverse_correction` does **not** support a worsening narrative on mean RMSE; both front and tail are still slightly improved (`-0.0157` front, `-0.0110` tail), even though the tail remains relatively heavier than the front.
  4. `single_lobe` boundary worsening is better explained by **time-shift dominance** than by slope flattening:
     - mean `delta_boundary_shift_abs_err = +0.1821`
     - mean `delta_boundary_slope_abs_err = +0.0244`
  5. `reverse_correction` shows the same direction, only weaker:
     - mean `delta_boundary_shift_abs_err = +0.1065`
     - mean `delta_boundary_slope_abs_err = +0.0333`
  6. At the stricter `Q1_fast x single_lobe` intersection, the slope term is not worsened on average:
     - mean `delta_boundary_shift_abs_err = +0.0584`
     - mean `delta_boundary_slope_abs_err = -0.0232`
     This argues against “slope flattening” as the main explanation for that slice.
  7. The worst `Q1_fast x single_lobe` cases are more **amplitude-driven** than boundary-driven:
     - `corr(delta_rmse_tail_abs_steer, peak_abs_amp_err_conditioned) = 0.7195`
     - `corr(delta_rmse_tail_abs_steer, boundary_shift_abs_err_conditioned) = -0.1544`
     - the largest failure case (`tyy#65`) combines `delta_rmse_tail_abs_steer = +1.5254` with `peak_abs_amp_err_conditioned = 1.9472`, while its boundary shift error is only `0.2607`
- **Recommended next step**:
  1. Treat the immediate mechanism as two-layered:
     - global `Q1_fast` degradation is mixed and not tail-only
     - the most actionable bad slice is `Q1_fast x single_lobe`, where tail amplitude mismatch is much more convincing than boundary-only drift
  2. If a minimal intervention experiment is started next, prioritize a fix aimed at tail amplitude / tail-shape behavior before a boundary-only loss tweak.
 3. If Task 3 reconstructs raw prediction sequences successfully, use those sequences to rerun the precise timestep-curve and boundary-shape checks, mainly to confirm the approximate conclusions already obtained here.

### Supplemental Record: 2026-04-08, Task 3 Reconstruct Conditioned V2 Training + Prediction Sequence Export
- **Executor**: Codex
- **What was done**:
  1. Read `tools/codex_handoff_task3_reconstruct_conditioned_v2.md` and restored the deleted conditioned-trajectory source files from git commit `418f869` back into `datasetprocess/final_code/model/training/`.
  2. During import validation, restored additional missing dependencies required by the recovered pipeline:
     - `baseline_eval_primary_aux.py`
     - `future_steer_speed_subjectsplit_masked.py`
     - `run_primary_v2_context_full2s_baseline.py`
     - `protocol_d3_response_aligned_extended_v1/__init__.py`
     - `protocol_d3_response_aligned_extended_v1/dataset_builder.py`
  3. Updated `datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py` so it now:
     - loads both the reconstructed baseline-mode run and the reconstructed `structured_v2` run from the same event-conditioned pipeline
     - saves per-sample prediction sequences as compressed `.npz`
     - stores `pred`, `true`, `sample_keys`, `mask`, `channel_names`, `channel_note`, `run_root`, and `split`
  4. Confirmed the usable GPU environment is `D:\ProgramData\anaconda3\envs\predict_2\python.exe` with `torch 2.7.1+cu118`, `cuda_available=True`, and device `NVIDIA GeForce RTX 2060`.
  5. Trained a reconstructed baseline formal run on GPU:
     - run root: `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_BASELINE_FORMAL_20260408_182418`
     - config used: manifest/split unchanged, seed `2026`, device `cuda`, epochs `12`, batch size `64`, lr `0.001`, conditioning mode `baseline`
  6. Trained a reconstructed conditioned v2 formal run on GPU using the new baseline checkpoint as initialization:
     - run root: `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260408_183236`
     - config used: manifest/split unchanged, seed `2026`, device `cuda`, init checkpoint from the reconstructed baseline run, epochs `3`, min epochs `3`, patience `2`, batch size `64`, lr `0.0003`, selection mode `structure_aware_primary`, conditioning mode `structured_v2`, `structure_width=0.065`, `gate_temperature=0.04`, `event_residual_scale=1.2`
  7. Ran the reconstructed evaluation/export step on the test split and produced:
     - `reports/baseline_prediction_sequences.npz`
     - `reports/conditioned_v2_prediction_sequences.npz`
     - `reports/event_conditioned_reconstruct_eval_20260408/` with comparison CSVs, summary JSON, and figures
- **Why it was done**:
  - The original formal conditioned v2 run had summary metrics only, with no checkpoint and no per-sample prediction sequences, which blocked direct sequence-level diagnosis.
  - Reconstructing the deleted source and rerunning the formal pipeline was the only reliable way to recover model checkpoints and export aligned prediction trajectories.
- **What was found**:
  1. The reconstructed source pipeline runs successfully after restoring the additional dependency files listed above.
  2. The new baseline formal reconstruction achieved:
     - best val steer RMSE: `0.5972`
     - final test steer RMSE: `0.5191`
  3. The new conditioned v2 formal reconstruction achieved:
     - best val steer RMSE: `0.5751`
     - final test steer RMSE: `0.4973`
  4. On this reconstruction, conditioned v2 remains better than the reconstructed baseline on test steer RMSE (`0.4973` vs `0.5191`) and test speed RMSE (`1.1397` vs `1.2989`).
  5. The exported sequence files contain `749` test samples with shapes:
     - `pred`: `(749, 400, 2)`
     - `true`: `(749, 400, 2)`
     - `sample_keys`: `(749,)`
     - `mask`: `(749, 400)`
  6. The handoff note requested `(N, 400, 3)` with channels `(steer, yawrate, ay)`, but the restored model family actually predicts only 2 channels from `y_seq`. To avoid silently fabricating channels, the `.npz` files store the real outputs and include:
     - `channel_names = ['steer_rel', 'speed_delta']`
     - `channel_note` explaining the 2-channel reality of the restored pipeline
- **Recommended next step**:
 1. Use `reports/conditioned_v2_prediction_sequences.npz` and `reports/baseline_prediction_sequences.npz` as the sequence source for the precise per-timestep / boundary-shape diagnosis that was only approximated in Task 2.
 2. When consuming these files downstream, treat them as 2-channel trajectory outputs (`steer_rel`, `speed_delta`), not as 3-channel `(steer, yawrate, ay)` tensors.

### Supplemental Record: 2026-04-08, Step 4 Tail Amplitude Penalty Experiment
- **Executor**: Codex
- **What was done**:
  1. Read `tools/codex_handoff_step4_tail_amp_penalty.md` and executed the requested single-variable loss intervention without changing the architecture, protocol, split, anchor, horizon, or data pipeline.
  2. Modified `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py` in exactly two places:
     - added tail-penalty constants at lines `49-51`
     - replaced the training-loop loss block at lines `468-480`
  3. The exact training-script change was:
     - `TAIL_START = 200`
     - `W_TAIL_AMP = 0.3`
     - training loop only: computed `tail_mask`, `pred_amp`, `true_amp`, `tail_amp_loss`, and added `W_TAIL_AMP * tail_amp_loss` to the existing `traj_loss + event_loss_weight * event_breakdown.total`
     - validation loss path was left unchanged
  4. Launched the Step 4 conditioned v2 retraining run from the reconstructed Task 3 baseline checkpoint using the handoff config:
     - `run_prefix=EXP_EVENT_CONDITIONED_TRAJECTORY_V2_TAILAMP_STEP4`
     - `seed=2026`
     - `device=cuda`
     - `epochs=3`, `min_epochs=3`, `patience=2`
     - `batch_size=64`, `lr=0.0003`
     - `event_loss_weight=0.5`
     - `conditioning_mode=structured_v2`
     - `structure_width=0.065`, `gate_temperature=0.04`, `event_residual_scale=1.2`
     - `selection_mode=structure_aware_primary`
     - `init_checkpoint=tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_BASELINE_FORMAL_20260408_182418/best_model.pt`
  5. Training run path:
     - `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_TAILAMP_STEP4_20260408_193916`
  6. Ran the existing eval/export pipeline on the test split and produced:
     - `reports/step4_tailamp_eval_20260408/`
     - `reports/step4_tailamp_prediction_sequences.npz`
  7. Verified exported sequence contents:
     - `pred`: `(749, 400, 2)`
     - `true`: `(749, 400, 2)`
     - `sample_keys`: `(749,)`
     - `mask`: `(749, 400)`
     - `channel_names = ['steer_rel', 'speed_delta']`
  8. Computed the Step 4 slice metrics by joining `reports/step4_tailamp_eval_20260408/conditioned_trajectory_sample_metrics.csv` with `reports/attribution_master_table.csv` on `sample_key`, using the attribution table's baseline-reference columns so the Task 3 reference values exactly matched the handoff anchors (`+0.0155`, `-0.0345`, `+0.1821`).
- **Why it was done**:
  - Prior attribution and failure-analysis tasks pointed to tail amplitude mismatch, especially in the `Q1_fast` degradation pattern, as the strongest evidence-backed mechanism.
  - This experiment was the minimal intervention to test whether explicitly penalizing steer-tail amplitude in training could reduce the fast-slice tail regression without materially degrading overall test RMSE.
- **What was found**:
  1. Step 4 training completed successfully on GPU and produced a new checkpointed run with no manifest or split changes.
  2. Overall test steer RMSE remained within the allowed tolerance relative to Task 3:
     - Task 3 conditioned v2: `0.4973`
     - Step 4: `0.5013`
     - delta: `+0.0040`
  3. Overall test tail RMSE was essentially unchanged:
     - Task 3 conditioned v2: `0.3619`
     - Step 4: `0.3623`
     - delta: `+0.0004`
  4. The key `Q1_fast` tail metric improved versus Task 3, but did not cross the Go threshold:
     - Task 3 conditioned v2 mean `delta_rmse_tail_abs_steer`: `+0.0155`
     - Step 4 mean `delta_rmse_tail_abs_steer`: `+0.0057`
     - delta vs Task 3: `-0.0098`
  5. Outside `Q1_fast`, tail behavior weakened relative to Task 3:
     - Task 3 non-`Q1_fast` mean `delta_rmse_tail_abs_steer`: `-0.0345`
     - Step 4 non-`Q1_fast` mean `delta_rmse_tail_abs_steer`: `-0.0184`
     - delta vs Task 3: `+0.0162`
  6. `single_lobe` boundary-shift worsening increased substantially:
     - Task 3 mean `delta_boundary_shift_abs_err`: `+0.1821`
     - Step 4 mean `delta_boundary_shift_abs_err`: `+0.3855`
     - delta vs Task 3: `+0.2034`
  7. Comparison table:

| Metric | Task 3 conditioned v2 | Step 4 tail amp penalty | Delta |
|---|---:|---:|---:|
| test rmse_2s_abs_steer (overall) | 0.4973 | 0.5013 | +0.0040 |
| test rmse_tail_abs_steer (overall) | 0.3619 | 0.3623 | +0.0004 |
| Q1_fast: mean delta_rmse_tail_abs_steer | +0.0155 | +0.0057 | -0.0098 |
| non-Q1_fast: mean delta_rmse_tail_abs_steer | -0.0345 | -0.0184 | +0.0162 |
| single_lobe: mean delta_boundary_shift | +0.1821 | +0.3855 | +0.2034 |

  8. Go / No-Go verdict: **No-Go**
     - overall RMSE passed the tolerance check (`0.5013 <= 0.508`)
     - but `Q1_fast` mean `delta_rmse_tail_abs_steer` stayed positive (`+0.0057 > 0.0000`)
     - the intervention therefore improved the targeted fast-slice tail regression only partially and also worsened the `single_lobe` boundary-shift behavior
- **Recommended next step**:
  1. Do not promote the Step 4 tail-amplitude penalty as the next default method; it is insufficient under the handoff decision rule.
  2. Keep the Step 4 artifacts for reference, but move to the next minimal intervention that explicitly addresses temporal/boundary alignment in `single_lobe` cases, since boundary-shift damage increased even while `Q1_fast` tail RMSE improved modestly.
  3. If another ablation is run, compare against the same attribution-table baseline columns again so the slice metrics remain anchored to the established Task 3 reference values.

### Supplemental Record: 2026-04-08, Review of Whether the Next Minimal Ablation Should Be Event-Loss-Weight Only
- **Executor**: Codex
- **What was done**:
  1. Read `CLAUDE.md` first and followed the repository review order for active modeling work.
  2. Performed a read-only inspection of the current active training path:
     - `datasetprocess/final_code/model/training/event_conditioned_baseline_model.py`
     - `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
     - `datasetprocess/final_code/model/training/event_conditioned_eval_support.py`
  3. Verified the exact loss structure in code:
     - `compute_event_loss(...)` already aggregates turn / reversal / peak sub-losses into `event_breakdown.total`
     - training uses `traj_loss + args.event_loss_weight * event_breakdown.total + W_TAIL_AMP * tail_amp_loss`
     - `--event-loss-weight` is exposed as a standalone CLI argument with default `0.50`
     - the structure-aware validation summary still tracks `tail_score`, `turning_score`, `continuity_score`, and `boundary_shift_abs_err`
  4. Read the existing Step 4 records in this same progress log to align the review against the established project decision state instead of reinterpreting the experiment from memory.
- **Why it was done**:
  - The immediate thesis/model decision was whether the next smallest ablation after the Step 4 tail-amplitude No-Go should prioritize changing only `event_loss_weight`, while keeping the current deterministic conditioned v2 mainline and retaining the tail penalty.
  - The user explicitly requested review only: no retraining, no code edits, and no long-running commands.
- **What was found**:
  1. `event_loss_weight` is the cleanest currently exposed single scalar for changing how strongly the optimizer trades trajectory fit against the already-defined event objective, because it is a dedicated CLI knob and does not alter architecture, protocol, labels, split, anchor, or event-target definitions.
  2. However, it is not a perfectly isolated mechanism-level ablation:
     - it rescales the entire bundled event objective at once
     - that bundled objective mixes turn onset, reversal, and peak tasks with fixed internal weights
     - therefore any observed change would still be interpretable only as "more or less total event supervision" rather than as a boundary-specific or peak-specific causal test
  3. Keeping the current tail penalty while tuning only `event_loss_weight` is still explainable at review level, but the interpretation must be precise:
     - the ablation would test whether rebalancing `event` supervision against the fixed `traj + tail-amplitude` objective can recover some structure without reopening the Step 4 tail-penalty design itself
     - it would not answer whether tail penalty is good in isolation, because the Step 4 result is already a No-Go and the retained tail term remains a known confounder for boundary behavior
  4. Given the recorded Step 4 outcome, the main limitation is directional:
     - Step 4 only partially improved `Q1_fast` tail RMSE
     - Step 4 substantially worsened `single_lobe` boundary shift
     - this makes an "event-loss-weight only" follow-up relatively weak as the next priority if the real unresolved bottleneck is temporal / boundary alignment rather than insufficient total event pressure
  5. Review verdict:
     - `event_loss_weight` is a clean optimizer-level single-variable entry
     - but it is not the best next ablation priority after Step 4, because the dominant failure that remains on record is boundary-shift damage in `single_lobe`, and `event_loss_weight` moves all event heads together rather than targeting that failure mode
- **Recommended next step**:
  1. Do not prioritize `event_loss_weight`-only as the very next ablation.
  2. Prefer the next review-level direction to be a boundary / timing-focused minimal intervention concept, for example: add or re-enable a narrowly scoped temporal-alignment term that targets boundary / peak timing behavior while leaving architecture, protocol, and dataset unchanged.
  3. If `event_loss_weight` is still tested later, frame it explicitly as an optimization-balance ablation under a fixed tail-penalty regime, and judge it against `Q1_fast` tail metrics plus `single_lobe` `boundary_shift_abs_err` together rather than tail metrics alone.

### Supplemental Record: 2026-04-08, Focused Review of Whether Existing Structured-Time Hyperparameters Should Be Prioritized Before `event_loss_weight`
- **Executor**: Codex
- **What was done**:
  1. Read `CLAUDE.md` first, then performed a read-only inspection of the active files that directly determine the current decision:
     - `datasetprocess/final_code/model/training/conditioned_trajectory_head.py`
     - `datasetprocess/final_code/model/training/event_head.py`
     - `datasetprocess/final_code/model/training/event_conditioned_baseline_model.py`
     - `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
  2. Verified that `conditioned_trajectory_head.py` already contains explicit structured-time controls for the conditioned decoder:
     - `structure_width`
     - `gate_temperature`
     - `_gaussian_track(...)`
     - `_sigmoid_track(...)`
     - `peak_pulse`
     - `reversal_pulse`
     - `post_peak`
     - `post_reversal`
     - `tail_gate`
  3. Verified the exact way those controls are used:
     - `structure_width` directly sets the Gaussian spread of turn / peak / reversal pulses
     - `gate_temperature` directly sets the sigmoid sharpness of post-turn / post-peak / post-reversal gates
     - the resulting structure tracks feed `structure_to_tgt`, `structure_to_film`, and `structure_to_steer`, so they directly shape the decoder target tokens, FiLM modulation, and steer residual
  4. Verified the training objective remains:
     - `traj_loss + event_loss_weight * event_breakdown.total + W_TAIL_AMP * tail_amp_loss`
  5. Verified that `event_breakdown.total` supervises turn / reversal / peak tasks, but `event_loss_weight` is only an outer scalar on that bundled event objective.
  6. Verified that training still uses `teacher_forcing_ratio=1.0` by default, so when teacher events are provided, the structured summary sent into the trajectory head is teacher-based during those steps rather than purely predicted.
- **Why it was done**:
  - The user requested a narrowly scoped review-only decision on the next smallest direction after the Step 4 tail-amplitude No-Go, with special attention to whether the existing structured-time hyperparameters are closer to the current boundary / timing failure than changing `event_loss_weight`.
- **What was found**:
  1. The current bad point on record is `single_lobe` boundary-shift worsening after Step 4, which is a boundary / timing-alignment failure, not primarily a missing-loss-weight failure.
  2. `structure_width` and `gate_temperature` are mechanically closer to that failure mode than `event_loss_weight`:
     - `structure_width` controls how narrow or diffuse the pulse support is around peak / reversal centers
     - `gate_temperature` controls how abruptly the post-event state turns on
     - both therefore act directly on the temporal edge sharpness and transition placement of the structured decoder tracks
  3. `event_loss_weight` is one level farther away from the observed problem:
     - it changes the optimizer pressure on the whole bundled event objective
     - it does not directly change the shape, width, or transition steepness of the structured tracks already injected into the decoder
     - any benefit would be indirect, through better event prediction quality, and that indirect path is weakened during teacher-forced training steps
  4. For a review-level minimal direction, prioritizing the existing structured-time hyperparameters is therefore more targeted than prioritizing `event_loss_weight`.
  5. Among the two structured-time knobs, `gate_temperature` is the most boundary-adjacent conceptually because the failure is about boundary shift / transition placement, while `structure_width` is the next most relevant because it governs pulse spread around peak / reversal timing.
- **Recommended next step**:
  1. If only one review-level direction is chosen now, prefer reviewing and then testing the existing structured-time hyperparameters first, especially `gate_temperature`, then `structure_width`.
  2. Defer `event_loss_weight` until after the structured-time review, or treat it only as a secondary optimizer-balance ablation rather than the first response to the current boundary / timing issue.

### Supplemental Record: 2026-04-08, Third Short Review on Whether `gate_temperature` Is the Smallest Next Direction
- **Executor**: Codex
- **What was done**:
  1. Read `CLAUDE.md` first and kept the review read-only.
  2. Re-checked the active implementation path only:
     - `datasetprocess/final_code/model/training/conditioned_trajectory_head.py`
     - `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
  3. Verified the exact mechanism under discussion:
     - default `gate_temperature=0.040`
     - `_sigmoid_track(...)` uses `gate_temperature` directly as the sigmoid temperature
     - the resulting `post_turn`, `post_peak`, `post_reversal`, and `tail_gate` are concatenated into `structure_tracks`
     - `structure_tracks` then drive `structure_to_tgt`, `structure_to_film`, and `structure_to_steer`
     - the training script passes `args.gate_temperature` straight into the model
  4. Kept the review anchored to the already confirmed decision state:
     - Step 4 tail penalty stays
     - the current primary failure is `single_lobe` `boundary_shift` worsening
     - no training, no code edits, and no long-running commands were executed
- **Why it was done**:
  - The user asked for a third, shorter review that answers only whether `gate_temperature` is the right single next entry point for the current boundary / timing failure, and what the main risks and directionality are if only that one hyperparameter is changed.
- **What was found**:
  1. `gate_temperature` is the most boundary-adjacent single existing knob in the active head because it directly changes the steepness and onset softness of the post-event sigmoid tracks that define post-turn / post-peak / post-reversal transitions and the derived `tail_gate`.
  2. Relative to other already-exposed scalars, it is the closest single entry point to the recorded failure mode:
     - closer than `event_loss_weight`, which only rescales bundled event supervision indirectly
     - slightly more timing-edge-specific than `structure_width`, which mainly changes pulse spread rather than post-event turn-on sharpness
  3. But it is still not a perfectly isolated boundary-only knob:
     - one scalar moves four structured channels at once
     - the same change affects both early transition sharpness and later tail gating
     - because these tracks feed token bias, FiLM modulation, and steer residual together, the effect can propagate globally through the decoder rather than staying local to the boundary
  4. Review-level risk assessment if only `gate_temperature` is changed:
     - risk 1: improving boundary timing while degrading tail amplitude or tail persistence, because `tail_gate` is built from the same softened/sharpened sigmoids
     - risk 2: making transitions too sharp and causing overshoot / brittle timing sensitivity, or too soft and worsening the existing boundary lag, because the parameter directly sets sigmoid slope
     - risk 3: creating ambiguous attribution, because any gain/loss will reflect a joint change in `post_turn`, `post_peak`, `post_reversal`, and `tail_gate`, not a single isolated structure component
  5. Directional judgment at review level:
     - if forced to choose a sign, the safer first move is to increase `gate_temperature` modestly rather than decrease it
     - reasoning: the current issue is boundary-shift worsening under a regime that already keeps the tail penalty, so making the gates even sharper is more likely to amplify abrupt boundary commitment and timing brittleness; a modest increase is the more conservative way to reduce over-hard switching while preserving the same mechanism
     - confidence on sign is moderate rather than high, because without a direct before/after decomposition of boundary lag versus tail under this exact knob, direction remains an inference from mechanism, not a verified run result
- **Recommended next step**:
  1. Treat `gate_temperature` as the best current single-parameter entry for the next minimal boundary/timing-focused check.
  2. If only one sign is chosen first at review level, start with a modest increase, not a decrease.
  3. Judge it jointly on `single_lobe` boundary timing and tail behavior, because the same knob controls both transition sharpness and tail gating.

### Supplemental Record: 2026-04-08, Smoke-Test Entry Review for `gate_temperature`
- **Executor**: Codex
- **What was done**:
  1. Read `CLAUDE.md` first, then performed a strictly read-only review of the active path and the latest same-day decision records:
     - `datasetprocess/final_code/model/training/conditioned_trajectory_head.py`
     - `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
     - `datasetprocess/final_code/model/training/event_conditioned_baseline_model.py`
     - `reports/project_progress_master.md`
     - `reports/step4_decision_summary_20260408.md`
  2. Re-verified the exact live mechanism instead of relying on prior chat memory:
     - default `gate_temperature=0.040`
     - default `structure_width=0.065`
     - default `event_loss_weight=0.50`
     - `gate_temperature` goes directly into `_sigmoid_track(...)`
     - the resulting `post_turn`, `post_peak`, `post_reversal`, and `tail_gate` enter `structure_tracks`
     - those tracks feed `structure_to_tgt`, `structure_to_film`, and `structure_to_steer`
     - training remains `traj_loss + event_loss_weight * event_breakdown.total + W_TAIL_AMP * tail_amp_loss`
  3. Re-checked that `event_loss_weight` acts as an outer loss scalar while training still defaults to `teacher_forcing_ratio=1.0`, which weakens its directness as a response to the current boundary/timing failure.
  4. Kept the review under the user constraints:
     - no training
     - no code edits
     - no long-running commands
- **Why it was done**:
  - The immediate thesis/model decision was whether, with Step 4 tail penalty retained and the active failure still centered on `single_lobe` `boundary_shift`, `gate_temperature` is still the most justified minimal single-variable direction and whether that conclusion is strong enough to justify an execution-before-full-run smoke test.
- **What was found**:
  1. Yes: under the current constraints, `gate_temperature` remains the strongest minimal single-variable entry point.
     - it is more directly coupled to boundary/transition placement than `event_loss_weight`
     - it is slightly more boundary-edge-specific than `structure_width`
  2. The reason it stays first is mechanistic closeness:
     - `gate_temperature` directly changes sigmoid onset softness/steepness for post-event tracks
     - the same tracks define both transition behavior and derived `tail_gate`
     - this matches the current recorded joint problem: boundary-shift damage with tail sensitivity still in play
  3. `structure_width` is still the next-best backup knob, but it is broader:
     - it changes Gaussian support around centers
     - that is relevant to temporal spread, but less directly targeted at post-event boundary turn-on than `gate_temperature`
  4. `event_loss_weight` remains less suitable as the next smallest direction:
     - it rescales bundled event supervision indirectly
     - it does not directly reshape the structured tracks already entering the decoder
     - with teacher forcing on by default, some of the hoped-for benefit path is one step removed from the current structural failure
  5. Review-level directional confidence is not absolute:
     - the safer first sign is still a modest increase in `gate_temperature`
     - but confidence on sign is only moderate, not high, because the same scalar also moves `tail_gate` and could trade boundary timing against tail persistence
  6. Smoke-test worthiness:
     - yes, it is worth a smoke-test-level experiment before any broader run
     - the reason is not high confidence in effect sign, but high confidence that this is the most interpretable next single-knob check under the current no-protocol/no-architecture-change regime
- **Recommended next step**:
  1. If only one execution-level probe is allowed next, use a `gate_temperature`-only smoke test first.
  2. Treat the sign as "prefer modest increase first" rather than "known-correct increase."
  3. Gate the smoke-test decision on joint readout of `single_lobe` boundary timing and tail behavior, not on tail metrics alone.
### Supplemental Record: 2026-04-08, Gate-Temperature-Only Smoke Test (`0.040 -> 0.050`) After Pre-Execution Audit
- **Executor**: Codex
- **What was done**:
  1. Read `CLAUDE.md` first, then re-checked only the active path and same-day decision files requested by the user:
     - `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
     - `datasetprocess/final_code/model/training/conditioned_trajectory_head.py`
     - `reports/project_progress_master.md`
     - `reports/step4_decision_summary_20260408.md`
  2. Verified the smoke-test brief stayed clean before execution:
     - Step 4 tail penalty is still active in training as `traj_loss + event_loss_weight * event_breakdown.total + W_TAIL_AMP * tail_amp_loss`
     - default live values remained `gate_temperature=0.040`, `structure_width=0.065`, `event_loss_weight=0.50`
     - the target knob still enters `_sigmoid_track(...)` directly and therefore is the intended single-variable structured-time control
     - no protocol, split, anchor, horizon, dataset, model family, or eval-script changes were introduced
  3. Confirmed environment and execution scope before launch:
     - `conda run -n predict2 ...` failed because of a local conda wrapper issue (`NoWritableEnvsDirError`), so execution switched to the installed environment interpreter `D:\\ProgramData\\anaconda3\\envs\\predict_2\\python.exe`
     - confirmed `torch 2.7.1+cu118` sees `cuda_available=True`
     - confirmed GPU device: `NVIDIA GeForce RTX 2060`
     - confirmed output roots:
       - training run: `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_SMOKE_GATETEMP005_20260408_223608`
       - eval report: `reports/gate_temperature_smoke_eval_20260408_223608`
  4. Executed exactly one smoke-test-level training run with no persistent source edit:
     - runtime override only: `--gate-temperature 0.050`
     - all other Step 4 control settings kept fixed at runtime, including:
       - `--structure-width 0.065`
       - `--event-loss-weight 0.5`
       - `--conditioning-mode structured_v2`
       - `--selection-mode structure_aware_primary`
       - `--teacher-forcing-ratio 0.75`
       - `--event-residual-scale 1.2`
       - same manifest / protocol path
       - same baseline init checkpoint
     - smoke limits only:
       - `--smoke-train-samples 96`
       - `--smoke-val-samples 32`
       - `--smoke-test-samples 32`
       - `--smoke-epochs 2`
  5. Ran the existing evaluation script without changing eval logic:
     - `datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py`
     - compared the new smoke conditioned run against `EXP_EVENT_CONDITIONED_TRAJECTORY_BASELINE_FORMAL_20260408_182418`
     - used the generated `sample_level_comparison.csv` and `trajectory_subset_comparison.csv` to aggregate smoke-scope metrics on the overlapping sample pool
- **Why it was done**:
  - The user requested a strict "review first, execute second" workflow for the next minimal thesis/model probe.
  - After Step 4 tail-amplitude penalty was marked No-Go, same-day review records had already narrowed the next single-variable entry from `event_loss_weight` to `gate_temperature`, with the preferred first sign being a modest increase.
  - The purpose of this run was not to promote a new mainline, but to check whether `gate_temperature` can survive a minimal smoke test while keeping the Step 4 tail-penalty regime and avoiding any high-risk scope drift.
- **What was found**:
  1. The pre-execution audit passed:
     - this remained a single-variable probe
     - Step 4 tail penalty stayed on
     - no high-risk protocol / split / anchor / horizon / dataset / eval changes were made
     - no persistent training-source edit was required; the change was injected only through the CLI override
  2. The smoke result itself was clearly negative on the smoke sample pool (`n=32` test samples):
     - overall `rmse_2s_abs_steer_conditioned = 0.5664` vs baseline `0.4125` (`delta = +0.1539`)
     - overall `rmse_tail_abs_steer_conditioned = 0.5365` vs baseline `0.4085` (`delta = +0.1280`)
  3. The requested fast-response readout also worsened on the smoke pool:
     - `Q1_fast mean delta_rmse_tail_abs_steer = +0.0695` on `n=8`
     - this is directionally worse than the 2026-04-08 Step 4 formal context, where the same metric was `-0.0188` on the full test pool
  4. `single_lobe` boundary behavior on the smoke pool did not justify continuation:
     - `single_lobe mean boundary_shift_abs_err_conditioned = 0.6393`
     - same-pool baseline was `0.5509`, so the smoke delta was still positive (`+0.0885`)
     - it is numerically lower than the Step 4 formal full-pool conditioned value (`0.8561`), but that is not a fair head-to-head improvement claim because the smoke slice only has `n=6` `single_lobe` samples
  5. Coarse training/eval stability was also weak:
     - smoke run `best_val_steer_rmse = 0.4689`
     - smoke run printed test steer RMSE `0.7286`
     - together with the subset eval results, this is not a "soft maybe"; it is a fail-fast signal for this exact `0.050` first move
- **Recommended next step**:
  1. Do not escalate `gate_temperature=0.050` to a full run.
  2. Keep the Step 4 tail penalty in place; this smoke result does not justify removing it.
  3. Treat the current `0.050` first-sign hypothesis as a smoke-test No-Go under the present setup, and keep mainline defaults unchanged.
  4. If structured-time tuning is still pursued, require a new explicit review before any further run, with the next candidate framed as a fresh minimal probe rather than a continuation of this `0.050` branch.

### Supplemental Record: 2026-04-08, Updated Priority Reset After Incorporating External GPT Analysis
- **Executor**: Claude
- **What was done**:
  1. Read the user-provided Word document containing an earlier GPT analysis summary and extracted its text for direct comparison against the repository-grounded findings accumulated later in this session.
  2. Cross-checked the GPT document's major claims against the now-established evidence chain in this repo:
     - attribution closure on `Q1_fast` tail amplitude/shape mismatch
     - morphology-driven `boundary_shift` time-shift worsening
     - Step 4 tail-amplitude penalty No-Go
     - review verdict that `event_loss_weight` is not the best next knob
     - `gate_temperature=0.050` smoke-test No-Go
  3. Reconciled what still holds from the GPT document versus what must now be updated, and used that comparison to reset the next-step priority order.
- **Why it was done**:
  - The earlier GPT analysis was directionally useful but written before the latest review, probe, and smoke-test evidence existed.
  - A fresh priority reset was needed so the project does not keep acting as if the failure modes are still ambiguous or as if the latest No-Go probes never happened.
- **What was found**:
  1. The GPT document remains strong on methodology: it was right to emphasize fairness, same-schedule comparability, and analysis-before-structure-escalation.
  2. However, its stage framing is now outdated:
     - the project is no longer at "we still haven't split the failure cleanly"
     - it is now at "the main failure families are largely identified, but two minimal repair knobs have already failed"
  3. Because both minimal probes are now negative (`tail amplitude penalty` and `gate_temperature=0.050`), the next highest-value step is no longer another local knob tweak.
  4. The priority should reset to:
     - **P0**: matched-schedule `conditioning_mode` single-variable comparison
     - **P1**: formal closure of fair-comparison conclusions
     - **P2**: rewrite thesis narrative so mainline value and local failure slices are separated
     - only after that, if conditioned still shows net value under matched schedule, consider a more conservative next probe
  5. Under this reset, continuing the current failed local branches (`tail amplitude penalty`, `gate_temperature=0.050`) is not recommended.
- **Recommended next step**:
  1. Move the next execution priority to a matched-schedule `conditioning_mode` single-variable control experiment.
  2. In parallel or just before it, finalize a formal fairness/comparison summary so future discussion stops relying on mixed active-vs-active wording.
  3. Treat any future `gate_temperature` or `structure_width` attempt as conditional on matched-schedule evidence that the conditioned mainline still has net value worth saving.

### Supplemental Record: 2026-04-09, Read-Only Audit of Whether the Active Training Path Can Support a Fair Matched-Schedule `conditioning_mode` Control Matrix
- **Executor**: Codex
- **What was done**:
  1. Read `CLAUDE.md` first, then performed a strictly read-only audit of the active path and the latest directly relevant records:
     - `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
     - `datasetprocess/final_code/model/training/conditioned_trajectory_head.py`
     - `datasetprocess/final_code/model/training/event_conditioned_baseline_model.py`
     - `datasetprocess/final_code/model/training/event_head.py`
     - `datasetprocess/final_code/model/training/event_conditioned_eval_support.py`
     - `datasetprocess/final_code/model/training/protocol_allphase_control_v2_context_full2s/protocol_config.json`
     - `reports/project_progress_master.md`
     - `reports/step4_decision_summary_20260408.md`
  2. Audited the training script for every user-specified fairness lock:
     - manifest / split source
     - seed and deterministic setup
     - init checkpoint handling
     - epochs / min_epochs / patience
     - batch size and learning rate
     - teacher forcing
     - event loss weight
     - `conditioning_mode`
     - `structure_width`
     - `gate_temperature`
     - `event_residual_scale`
     - selection logic and checkpoint saving
  3. Audited the decoder path to verify what actually changes between `conditioning_mode=baseline` and `conditioning_mode=structured_v2`.
  4. Compared the latest formal baseline and conditioned-v2 run summaries to determine whether the current on-record pair is already a fair matched-schedule control.
- **Why it was done**:
  - The project priority was reset on 2026-04-08 toward a matched-schedule `conditioning_mode` single-variable comparison after both minimal repair probes (`Step 4 tail amplitude penalty` and `gate_temperature=0.050`) were marked No-Go.
  - Before any new execution, the repo needed a precise read-only answer on whether the current active script can already support the requested four-cell fairness matrix and where pseudo-matching risks remain.
- **What was found**:
  1. The active script can support the requested control matrix without source edits if execution is disciplined carefully.
     - `conditioning_mode` and `selection_mode` are already exposed as CLI arguments.
     - the script already saves `best_model_legacy.pt`, `best_model_structure.pt`, and `selection_compare`, so both selection criteria can be read from the same training trajectory if desired.
  2. The current formal baseline vs conditioned-v2 pair on record is **not** a fair matched-schedule single-variable comparison.
     - baseline formal used `selection_mode=legacy_rmse`, `init_checkpoint=null`, `epochs=12`, `min_epochs=6`, `patience=4`, `lr=0.001`, `conditioning_mode=baseline`, `event_residual_scale=1.0`
     - conditioned-v2 formal used `selection_mode=structure_aware_primary`, `init_checkpoint=<baseline best_model.pt>`, `epochs=3`, `min_epochs=3`, `patience=2`, `lr=0.0003`, `conditioning_mode=structured_v2`, `event_residual_scale=1.2`
     - therefore the currently discussed formal baseline and formal conditioned-v2 runs differ in much more than `conditioning_mode`
  3. At source level, `baseline` and `structured_v2` do **not** mean "no conditioning" versus "conditioning".
     - both modes always use the event-condition embedding path (`event_to_tgt`, `event_to_film`)
     - `structured_v2` additionally activates explicit structure tracks plus three extra decoder injections: `structure_to_tgt`, `structure_to_film`, and `structure_to_steer`
     - the fair question this script can answer is therefore: whether explicit `structured_v2` conditioning adds net value over the baseline event-conditioned decoder under matched schedule, not whether conditioning in the broadest sense is valuable versus no conditioning
  4. The biggest pseudo-matching risk is early stopping.
     - `selection_mode` does not only relabel reporting; it changes `active_key`
     - `active_key` drives `best_ckpt` updates and `bad_epochs`
     - the break condition `epoch >= min_epochs and bad_epochs >= patience` means two apparently matched runs can still stop at different actual epochs
     - to enforce a truly matched schedule without code edits, execution should neutralize early-stop divergence, for example by setting `min_epochs=epochs` or otherwise making `patience` non-binding
  5. The second major pseudo-matching risk is init-checkpoint drift hidden by `strict=False`.
     - the script loads init weights with `model.load_state_dict(..., strict=False)` and does not print missing / unexpected keys
     - if the chosen init checkpoint is not from the same active architecture, one branch could silently start with some active weights not actually matched
     - this risk is low only if the same init checkpoint comes from this exact script family
  6. Other fairness-sensitive points are cleaner but still need explicit locking in execution:
     - manifest / split are fixed if the same manifest path is reused
     - deterministic seeds and dataloader generators are already wired
     - `teacher_forcing_ratio` is stochastic but matched if seed and schedule are matched
     - `W_TAIL_AMP=0.3` is hard-coded, so it stays fixed automatically for all cells
     - `structure_width`, `gate_temperature`, and `event_residual_scale` can be kept numerically fixed, but only `structured_v2` actually consumes those structure-specific pathways in forward
  7. Execution strategy judgment:
     - because the needed controls are now clear and the purpose is fairness closure rather than speculative knob search, this is closer to a formal control run than to another hypothesis smoke test
     - however, a very short pre-flight manifest / init-checkpoint / command audit is still warranted before launch to avoid wasting the formal run on a pseudo-matched setup mistake
- **Recommended next step**:
  1. Do not treat the current 2026-04-08 formal baseline vs formal conditioned-v2 pair as the fairness answer.
  2. If execution proceeds, lock one common init checkpoint across all cells, keep the manifest fixed, and neutralize early-stop divergence so the actual optimization schedule matches.
  3. Prefer a pairwise execution order that minimizes attribution risk:
     - first `legacy_rmse`: baseline vs `structured_v2`
     - then `structure_aware_primary`: baseline vs `structured_v2`
  4. Use smoke only as a command / config pre-flight if there is still uncertainty about pathing or checkpoint compatibility; otherwise the fair-comparison objective is strong enough to go straight to formal matched runs.

### Supplemental Record: 2026-04-09, Read-Only Review of the Two-Run Matched-Schedule `conditioning_mode` Formal Plan
- **Executor**: Codex
- **What was done**:
  1. Read `CLAUDE.md` first, then performed a strictly read-only review of the proposed two-command formal plan for:
     - Run 1: `conditioning_mode=baseline`
     - Run 2: `conditioning_mode=structured_v2`
  2. Inspected the active path and the exact source files that control fairness, schedule matching, checkpoint selection, and evaluation output:
     - `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
     - `datasetprocess/final_code/model/training/conditioned_trajectory_head.py`
     - `datasetprocess/final_code/model/training/event_conditioned_baseline_model.py`
     - `datasetprocess/final_code/model/training/event_conditioned_eval_support.py`
     - `datasetprocess/final_code/model/training/future_steer_speed_subjectsplit_masked.py`
     - `datasetprocess/final_code/model/training/protocol_allphase_control_v2_context_full2s/protocol_config.json`
  3. Verified whether the script can answer both reporting criteria from each single run without duplicating training, and enumerated the remaining ways a run can look matched while still being operationally unmatched.
- **Why it was done**:
  - The project needed a precise read-only answer on whether the fairness-closure experiment should be executed as:
    - two matched runs with both criteria read from each run's `selection_compare`, or
    - four separate runs split by both `conditioning_mode` and `selection_mode`
  - The goal was to avoid wasting the formal budget on duplicated training or on a pseudo-matched setup that would not support a clean thesis claim.
- **What was found**:
  1. The proposed two-run plan is sufficient for the stated question if both runs stay on the same code/data snapshot and are read from `selection_compare`.
     - each run always writes `best_model_legacy.pt`, `best_model_structure.pt`, and `best_model.pt`
     - each run then re-evaluates those checkpoints and stores the results in both `selection_comparison.csv` and `run_summary.json -> selection_compare`
     - therefore one run per `conditioning_mode` is enough to recover both the `legacy_rmse` and `structure_aware_primary` readouts
  2. For this specific script, the proposed commands are close to a genuine matched-schedule single-variable control:
     - same manifest
     - same seed
     - same device
     - same epochs / `min_epochs`
     - same batch size / learning rate
     - same `event_loss_weight`
     - same `teacher_forcing_ratio`
     - same `structure_width` / `gate_temperature` / `event_residual_scale`
     - no warm-start chaining
     - no `init_checkpoint`
  3. The script structure makes the two-run plan cleaner than a four-run matrix for this question.
     - with `min_epochs=epochs`, the optimizer schedule is already forced to run full length
     - changing `selection_mode` mainly changes which checkpoint is labeled active
     - if the fairness answer is read from `selection_compare` rather than from `final_test_metrics`, the extra two runs are mostly redundant
  4. The biggest remaining pseudo-matching risks are not in the two commands' visible fields, but in hidden defaults and hidden external inputs.
     - hidden defaults not explicitly pinned include `selection_mode`, `weight_decay`, `grad_clip`, `d_model`, `nhead`, `enc_layers`, `dec_layers`, `ffn_dim`, `dropout`, `event_embed_dim`, `event_bin_size`, `use_privileged_teacher`, and smoke-test flags
     - hidden external inputs remain present because evaluation metadata enrichment reads `protocol_d3_response_aligned_extended_v1/sample_manifest.csv`, and sample construction silently drops rows if `_make_sample(...)` raises
  5. Operationally, the fairness claim is only fully clean if both finished runs also agree on:
     - `sample_manifest_used.csv`
     - `dropped_samples`
     - the absence of accidental default drift or code edits between the two launches
  6. Determinism is strong but not absolute.
     - the script seeds Python / NumPy / Torch, fixes dataloader generators, and uses `num_workers=0`
     - but `torch.use_deterministic_algorithms(True, warn_only=True)` still allows warning-level nondeterministic fallbacks, so the same CUDA device and software stack should be preserved
  7. The decoder is implemented in a way that actually helps fairness interpretation.
     - the structure-specific modules are instantiated in both modes
     - only `structured_v2` activates the extra structure-conditioned forward-path injections
     - this means parameter count stays matched while the active forward behavior differs in the intended place
- **Recommended next step**:
  1. Use the two-run matched-schedule plan, not a four-run matrix, for the formal fairness answer about whether `structured_v2` still has net value over the baseline-conditioned decoder.
  2. Before launch, run a minimal non-training pre-flight only:
     - confirm the target environment can import the script and parse flags (`--help`)
     - confirm CUDA is visible in that environment
     - confirm both the main manifest and the hidden D3 manifest used by eval metadata are readable
  3. After execution, read only the `legacy` and `structure` rows from each run's `selection_compare` / `selection_comparison.csv` when making the fair-comparison claim.
  4. If the commands are copied into notes or a runbook, explicitly pin the omitted defaults as well so the matched-schedule contract remains stable under future script drift.

### Supplemental Record: 2026-04-09, Matched-Schedule `conditioning_mode` Formal Two-Run Execution and Fairness Readout
- **Executor**: Codex
- **What was done**:
  1. Read `CLAUDE.md` first, confirmed the active training entry is `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py`, and confirmed the script's output contract:
     - run root: `F:\data_set_process\data_process\tmp\event_conditioned_runs`
     - fairness readout sources: `selection_comparison.csv` and `run_summary.json -> selection_compare`
     - run-level manifest audit: `sample_manifest_used.csv`
     - dropped-row audit: `run_summary.json -> dropped_samples`
  2. Confirmed GPU visibility in the requested environment with `D:\ProgramData\anaconda3\envs\predict_2\python.exe`:
     - `torch.cuda.is_available() == True`
     - device: `NVIDIA GeForce RTX 2060`
  3. Executed exactly two formal runs, with no source edits, no `init_checkpoint`, and no warm-start chaining. Shared config was kept fixed across both runs:
     - manifest: `F:\data_set_process\data_process\datasetprocess\final_code\model\training\protocol_allphase_control_v2_context_full2s\sample_manifest.csv`
     - seed: `2026`
     - device: `cuda`
     - epochs/min_epochs: `12/12`
     - patience: `99`
     - batch_size: `64`
     - lr: `0.001`
     - weight_decay: `0.0`
     - grad_clip: `1.0`
     - event_loss_weight: `0.50`
     - teacher_forcing_ratio: `1.0`
     - selection_mode: `legacy_rmse`
     - d_model/nhead/enc_layers/dec_layers/ffn_dim/dropout: `128/2/2/2/256/0.1`
     - event_embed_dim/event_bin_size: `96/20`
     - structure_width/gate_temperature/event_residual_scale: `0.065/0.040/1.0`
  4. Run 1 executed as:
     - `run_prefix=EXP_EVENT_CONDITIONED_MATCHED_BASELINE`
     - `conditioning_mode=baseline`
     - output: `F:\data_set_process\data_process\tmp\event_conditioned_runs\EXP_EVENT_CONDITIONED_MATCHED_BASELINE_20260409_005715`
  5. Run 2 executed as:
     - `run_prefix=EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2`
     - `conditioning_mode=structured_v2`
     - output: `F:\data_set_process\data_process\tmp\event_conditioned_runs\EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_20260409_010302`
  6. After both runs finished, performed a strict fairness audit:
     - compared `run_summary.json -> config` between the two runs
     - verified `init_checkpoint=null` on both sides
     - verified `config` matches exactly after excluding only `run_prefix` and `conditioning_mode`
     - verified top-level `run_summary.json` metadata also matches after excluding metric-bearing fields and run-specific path fields
     - verified `sample_manifest_used.csv` row counts and SHA-256 hashes
     - verified `dropped_samples`
     - extracted the A/B/C/D four-cell answer only from each run's `selection_compare` rows (`legacy`, `structure`)
- **Why it was done**:
  - The project needed a fresh formal fairness answer for matched-schedule `conditioning_mode` comparison on 2026-04-09.
  - The user explicitly ruled out reusing the 2026-04-08 formal pair, ruled out source edits, and ruled out checkpoint-based or warm-start-based coupling.
  - The thesis/model question was whether `structured_v2` still deserves continuation when both runs are truly matched except for `conditioning_mode` and `run_prefix`.
- **What was found**:
  1. The requested execution constraints were satisfied.
     - correct interpreter used: `D:\ProgramData\anaconda3\envs\predict_2\python.exe`
     - GPU confirmed and used through `device=cuda`
     - exactly two runs were launched
     - no `init_checkpoint`
     - no warm-start chaining
     - no stderr output on either run
  2. The two runs are operationally matched at the config level.
     - `run_summary.json -> config` shows no differences after excluding only `run_prefix` and `conditioning_mode`
     - `init_checkpoint` is `null` on both sides
  3. The two runs are also matched at the used-sample level.
     - `sample_manifest_used.csv` rows: `6238` vs `6238`
     - SHA-256: `0b86eb064d11c3c8252211acbf999b8ddcab8e568ab556a80164a128084b5a24` on both sides
     - `dropped_samples`: `0` vs `0`
  4. The A/B/C/D fairness grid from `selection_compare` is:

     | Cell | Run | Selection row | overall steer RMSE | overall tail RMSE | peak_time_abs_err_s | boundary_shift_abs_err | turning_count_abs_err | selection_score |
     | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
     | A | baseline run | legacy | 0.391229 | 0.387507 | 0.598790 | 0.535222 | 1.951613 | 0.879389 |
     | B | baseline run | structure | 0.391229 | 0.387507 | 0.598790 | 0.535222 | 1.951613 | 0.879389 |
     | C | structured_v2 run | legacy | 0.397927 | 0.379108 | 0.564516 | 0.967678 | 1.391129 | 0.906284 |
     | D | structured_v2 run | structure | 0.397927 | 0.379108 | 0.564516 | 0.967678 | 1.391129 | 0.906284 |

  5. Net effect of `structured_v2` versus baseline in this matched pair is mixed rather than cleanly dominant.
     - improved: `overall_tail_rmse`, `peak_time_abs_err_s`, `turning_count_abs_err`, `selection_score`
     - worsened: `overall_steer_rmse`
     - materially worsened: `boundary_shift_abs_err` (`0.535222 -> 0.967678`)
     - within each individual run, the `legacy` and `structure` rows are identical in this pair, so the fair answer is driven entirely by run-to-run differences, not by a split between the two selection rows
  6. Thesis-facing interpretation:
     - `structured_v2` still shows signal on structure-oriented metrics and on the composite `selection_score`
     - however, this exact formal pair does not support claiming an unqualified win over baseline because the boundary-shift regression is too large to ignore
- **Recommended next step**:
  1. Treat this 2026-04-09 pair as the valid fresh fairness reference for the matched-schedule `conditioning_mode` question.
  2. Continue `structured_v2` only as a conditional research branch, not as a default replacement for baseline yet.
  3. Make the next step a focused failure analysis on the `boundary_shift_abs_err` regression under `structured_v2`, using this exact run pair as the anchor comparison.
  4. Do not spend the next budget on another broad two-run repeat until the boundary-shift failure mode is localized; the current bottleneck is diagnosis, not another blind re-run.

### Supplemental Record: 2026-04-09, Formal Conclusion Consolidation After the Matched-Schedule Fairness Answer
- **Executor**: Claude
- **What was done**:
  1. Consolidated the 2026-04-09 matched-schedule two-run result into a thesis-facing formal conclusion instead of leaving it only as raw metric tables.
  2. Reframed the branch status using the new fairness reference only, explicitly excluding the older 2026-04-08 formal pair from the fairness answer.
  3. Converted the result into a simple decision statement for future sessions: what `structured_v2` has already proven, what it has not proven, and what should happen next.
- **Why it was done**:
  - After obtaining the first fair matched-schedule answer, the project needed a stable wording that future work could build on without reopening the same baseline-vs-structured confusion.
  - The user explicitly asked for a simpler, plain-language summary of what the model is trying to achieve and what stage the project is currently in.
- **What was found**:
  1. `structured_v2` should not be described as either "already winning" or "not useful".
  2. The fair answer is narrower and more accurate: `structured_v2` keeps positive signal on tail, peak-time, turning, and composite structure-oriented score, but it still fails too visibly on `boundary_shift_abs_err` to justify replacing the baseline-conditioned decoder.
  3. The right current branch label is therefore: **conditionally valuable research branch, not default mainline replacement**.
  4. This also means the project has left the broad diagnosis stage; the dominant unresolved bottleneck is now specifically the boundary-shift failure under a branch that still has partial value elsewhere.
- **Recommended next step**:
  1. Use the 2026-04-09 matched pair as the new default reference whenever discussing the net value of `structured_v2`.
  2. Move immediately to boundary-centered failure analysis instead of broad reruns or another general-purpose knob search.
  3. Keep the thesis narrative split into two parts: retained structured benefit vs unresolved boundary regression.

### Supplemental Record: 2026-04-09, Boundary-Failure-Analysis Direction Set as the Next Execution Priority
- **Executor**: Claude
- **What was done**:
  1. Defined the next-stage analysis target as the `boundary_shift_abs_err` regression observed in the matched baseline vs matched `structured_v2` pair.
  2. Narrowed the objective from generic "why conditioned does not fully win" to a focused boundary-failure question anchored on the fair 2026-04-09 run pair.
  3. Structured the planned analysis outputs into: morphology-level split tables, boundary-local error modes, and a small representative-case panel.
- **Why it was done**:
  - The fairness question has now been answered cleanly enough; repeating broad comparison runs would add little value before the main regression is localized.
  - The user agreed that the next step should be framed in simple, practical language and should avoid drifting back into diffuse hypothesis search.
- **What was found**:
  1. The next useful unit of work is no longer a training probe but a read-only failure analysis anchored on the matched pair.
  2. The primary slices to inspect next are `single_lobe` and `reverse_correction`, because these remain the most plausible carriers of the boundary regression.
  3. The most important unresolved distinction is whether the observed boundary failure is dominated by early/late timing shift, local overshoot, or reversal/recovery misplacement.
- **Recommended next step**:
  1. Prepare and run a read-only boundary failure analysis against the matched pair.
  2. Extract morphology-sliced boundary deltas first, then add local-window diagnostics and representative-case visualization.
  3. Only after this analysis should the project revisit loss changes or structure-specific tuning.

### Supplemental Record: 2026-04-09, Read-Only Review of Why `structured_v2` Improves Tail/Peak/Turning but Worsens `boundary_shift_abs_err`
- **Executor**: Codex
- **What was done**:
  1. Performed a strictly read-only review starting from `CLAUDE.md` and this master progress log, then inspected the active code path for metric definition and evaluation flow:
     - `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
     - `datasetprocess/final_code/model/training/event_conditioned_eval_support.py`
     - `datasetprocess/final_code/model/training/conditioned_trajectory_head.py`
     - `datasetprocess/final_code/model/training/event_conditioned_baseline_model.py`
     - `datasetprocess/final_code/model/training/plot_event_conditioned_trajectory.py`
     - the downstream metric-definition file actually used by these entry points: `datasetprocess/final_code/model/training/baseline_eval_primary_aux.py`
  2. Inspected the two 2026-04-09 matched run folders read-only:
     - `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_20260409_005715`
     - `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_20260409_010302`
  3. Verified which files actually exist in those run roots and limited the review to real artifacts:
     - `run_summary.json`
     - `selection_comparison.csv`
     - `sample_manifest_used.csv`
     - `metrics.json`
     - `loss_history.csv`
     - no case-level exported CSV, prediction NPZ, figure panel, or per-sample error dump exists inside these two run folders
- **Why it was done**:
  - The project needed one narrow answer only: why `structured_v2` can improve tail / peak / turning metrics under a fair matched-schedule comparison while still causing a large regression in `boundary_shift_abs_err`.
  - The user explicitly required a no-training, no-source-edit, no-new-script review that stays fairness-safe with respect to split integrity, leakage, and future-label integrity.
- **What was found**:
  1. `boundary_shift_abs_err` is not an event-matching metric and is not selected by any matching rule inside these runs.
     - It is computed in `baseline_eval_primary_aux.compute_trajectory_sample_metrics(...)` from the predicted and true absolute steering trajectories after block downsampling and finite differencing.
     - The metric uses a fixed tail boundary at `TAIL_START_SEC = 1.5` s, then compares the change in mean derivative before vs after that fixed boundary:
       - `shift_true = mean(post_slice_true) - mean(pre_slice_true)`
       - `shift_pred = mean(post_slice_pred) - mean(pre_slice_pred)`
       - `boundary_shift_abs_err = abs(shift_pred - shift_true)`
     - Therefore it measures whether the predicted slope transition across the fixed tail boundary has the same local “jump” magnitude as the true trace, not whether the predicted peak or reversal lands at the same timestamp.
  2. `boundary_shift_abs_err`, `turning_count_abs_err`, and `peak_time_abs_err_s` share the same upstream sample-evaluation chain but not the same downstream extractor.
     - Shared upstream chain:
       - same absolute steering reconstruction
       - same block downsampling
       - same derivative sequence `d_true / d_pred`
       - same sample loop in `compute_trajectory_sample_metrics(...)`
     - Divergent downstream logic:
       - `boundary_shift_abs_err` uses fixed-window derivative mean contrast around the fixed 1.5 s boundary
       - `turning_count_abs_err` uses `compress_reversals(d_true, threshold)` / `compress_reversals(d_pred, threshold)`
       - `peak_time_abs_err_s` uses `argmax(abs(true_ds))` / `argmax(abs(pred_ds))`
     - So improvement in peak timing and reversal count can coexist with worse boundary continuity because they are not scored by the same local criterion.
  3. The matched-run evidence argues against “matching rule side effect” as the main explanation for this pair.
     - In both run folders, `selection_comparison.csv` shows `legacy`, `structure`, and `active` rows are identical within each run.
     - `run_summary.json` also shows `best_epoch = best_legacy_epoch = best_structure_epoch = 12` for both runs.
     - `sample_manifest_used.csv` is present and the matched-run notes already confirmed identical row count / hash with `dropped_samples = 0`.
     - So this regression is much more consistent with prediction-waveform behavior than with selection-row relabeling or split mismatch.
  4. The most plausible mechanism is a boundary-local shape effect introduced by the explicit structured branch.
     - `structured_v2` adds three extra decoder injections absent from `baseline`:
       - structure-conditioned target bias
       - structure-conditioned FiLM on decoder hidden states
       - direct steer residual from structure inputs
     - Those structure inputs are built from smooth Gaussian / sigmoid tracks tied to predicted turn / peak / reversal summary indices and a tail gate.
     - This design is well suited to improving coarse event organization such as “where the main peak is” and “whether a reversal exists”, but it can also soften or spread the local slope transition around the fixed 1.5 s boundary.
     - Because `boundary_shift_abs_err` looks only at the derivative jump across that fixed boundary, a smoother / wider / slightly delayed transition can score much worse there even when global tail RMSE, peak timing, and turning count improve.
  5. The current run outputs are insufficient to prove the exact per-case morphology carrier.
     - The run folders do not contain per-sample metric tables, predicted sequences, or exported case panels.
     - That means the review can localize the likely mechanism class from code plus aggregate metrics, but cannot truthfully claim which exact samples or morphology slices dominate the regression without additional read-only extraction from existing checkpoints / manifests.
- **Recommended next step**:
  1. Treat the main working hypothesis as: `structured_v2` improves coarse event placement but distorts the local slope jump at the fixed 1.5 s tail boundary.
  2. Keep the fairness interpretation narrow: this is more likely a waveform-shape issue than a matching-rule artifact in the current 2026-04-09 pair.
  3. If continuing read-only analysis, prioritize recovering per-sample boundary metrics from existing outputs or a no-training checkpoint evaluation path before proposing new training.
  4. If only one or two tiny follow-up experiments are allowed later, prefer clean ablations on the structure-specific decoder injections or structure-track sharpness rather than broad sweeps.
- **Claude verification note**:
  - Claude locally re-checked the key code anchors behind this review and found them consistent with the returned mechanism summary, especially the shared upstream evaluation chain in `event_conditioned_eval_support.py:127` / `:169`, the fixed-boundary morphology metric definitions in `baseline_eval_primary_aux.py:29-67`, the structured-track construction in `conditioned_trajectory_head.py:73-112`, the three structured injections at `conditioned_trajectory_head.py:131-155`, and the selection comparison export loop in `run_event_conditioned_trajectory_baseline.py:603-658`.
  - Based on that verification, the current working interpretation remains: this is more likely a boundary-local continuity problem in the predicted waveform than a fairness or selection artifact.

### Supplemental Record: 2026-04-09, Claude Consolidation of the Boundary-Failure Review Into the Next Minimal Action
- **Executor**: Claude
- **What was done**:
  1. Read back the Codex review result and locally re-opened the cited source files to confirm the key claim chain instead of forwarding the review blindly.
  2. Updated the top-level current-status section, date index, and topic index of this master log so a later session can recover the new state without rereading the whole day.
  3. Converted the review into one practical next-action frame: stop broad knob search, prioritize boundary failure analysis, and only then consider a tiny ablation.
- **Why it was done**:
  - The project had already reached a fair formal answer on whether `structured_v2` still has net value; what was missing was a tight mechanism interpretation usable for the next decision.
  - The user also prefers concise, action-oriented consolidation rather than another diffuse theory list.
- **What was found**:
  1. The current best interpretation is: `structured_v2` is helping coarse event organization while hurting local continuity across the fixed 1.5 s tail boundary.
  2. The current evidence does not support blaming split mismatch, leakage, selection-row switching, or generic model instability.
  3. The cleanest next experimental suspicion is the structure-specific steer residual branch, with `structure_width` / `gate_temperature` behind it as secondary sharpness controls.
- **Recommended next step**:
  1. Treat boundary failure analysis as the active mainline task.
  2. If further analysis remains read-only, recover per-sample boundary behavior first.
  3. If one minimal new experiment is later approved, test the `structure_to_steer` ablation before reopening wider hyperparameter search.
  4. Keep thesis-facing wording unchanged for now: `structured_v2` remains worth continuing, but it is not yet the replacement for baseline.

### Supplemental Record: 2026-04-09, Formal Single-Variable Test of `event_residual_scale=0.0` Under Matched Structured V2 Schedule
- **Executor**: Claude
- **Why it was done**:
  - After the read-only mechanism review, the most direct actionable hypothesis was that the explicit `structure_to_steer` residual branch might be the main cause of the boundary-local continuity regression.
  - The user explicitly stated that once Claude/Codex had analyzed things clearly, the work could proceed more aggressively instead of always stopping at the smallest conservative plan.
  - Because the maintained code path already exposes `--event-residual-scale`, this could be tested as a fairness-safe single-variable formal run without any source edit.
- **What was done**:
  1. Re-checked the active runner and model path and confirmed that `event_residual_scale` only gates the explicit structure steer residual while keeping the other `structured_v2` injections alive:
     - CLI exposure in `run_event_conditioned_trajectory_baseline.py`
     - pass-through in `event_conditioned_baseline_model.py`
     - actual residual gating in `conditioned_trajectory_head.py`
  2. Ran a short pre-flight and confirmed:
     - script parses correctly
     - CUDA is available
     - GPU is `NVIDIA GeForce RTX 2060`
     - manifest exists
     - matched structured_v2 reference run summary is readable
  3. Launched a formal single-variable training run:
     - run root: `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_NORESID_20260409_094418`
     - same manifest / seed / device / epochs / min_epochs / patience / batch_size / lr / weight_decay / grad_clip / event_loss_weight / teacher_forcing_ratio / selection_mode / architecture dims / `conditioning_mode=structured_v2` / `structure_width` / `gate_temperature`
     - only changed `event_residual_scale: 1.0 -> 0.0`
  4. Verified fairness after the run:
     - `dropped_samples = 0`
     - `sample_manifest_used.csv` head matches the matched structured_v2 reference and no sampling drift is visible
     - config diff is limited to `run_prefix`, manifest slash formatting, and `event_residual_scale`
- **What was found**:
  1. The residual-off run did **not** recover the baseline behavior.
  2. Relative to matched structured_v2, turning off the steer residual gave only a partial boundary improvement:
     - `boundary_shift_abs_err`: `0.967678 -> 0.900528` (improves, but remains far worse than baseline `0.535222`)
  3. At the same time, overall and tail performance regressed noticeably:
     - `overall steer RMSE`: `0.530886 -> 0.564264` vs baseline `0.507124`
     - `tail RMSE`: `0.379108 -> 0.412240` vs baseline `0.387507`
     - `selection_score`: `0.906284 -> 0.934958` compared against baseline `0.879389`, meaning the no-residual variant also loses part of the structured_v2 advantage profile
  4. Peak / turning behavior also did not become clearly better than the matched structured_v2 run:
     - `peak_time_abs_err_s`: `0.564516 -> 0.537097` (slight gain)
     - `turning_count_abs_err`: `1.391129 -> 1.346774` (slight gain)
     - but the cost in RMSE / tail quality is too large to call this a clean repair
  5. The working conclusion is therefore tighter now:
     - the explicit `structure_to_steer` residual is **not** the sole root cause of the boundary failure
     - and simply zeroing it out is **not** a viable default fix
- **Recommended next step**:
  1. Move the main suspicion upstream from `structure_to_steer` alone to the broader structured injection stack:
     - `structure_to_tgt`
     - `structure_to_film`
     - structure-track sharpness controlled by `structure_width` / `gate_temperature`
  2. Do not adopt the no-residual variant as the new default experimental branch.
  3. If the next run is still a single-variable intervention, prefer a sharpness-oriented ablation or an upstream structured-path ablation rather than repeating residual-scale variants.
  4. Keep the thesis-facing interpretation stable: `structured_v2` still has net research value, but the active failure mode is not repaired by removing only the direct structure steer residual.

### Three-run comparison summary (2026-04-09)
| Run | overall steer RMSE | tail RMSE | peak_time_abs_err_s | boundary_shift_abs_err | turning_count_abs_err | selection_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.507124 | 0.387507 | 0.598790 | 0.535222 | 1.951613 | 0.879389 |
| structured_v2 | 0.530886 | 0.379108 | 0.564516 | 0.967678 | 1.391129 | 0.906284 |
| structured_v2_noresid | 0.564264 | 0.412240 | 0.537097 | 0.900528 | 1.346774 | 0.934958 |

### Key file anchors for the no-residual test
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:393`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:409`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:736`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_conditioned_baseline_model.py:165`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_conditioned_baseline_model.py:178`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:131`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:147`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:153`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_20260409_010302/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_NORESID_20260409_094418/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_20260409_005715/run_summary.json`

### Supplemental Record: 2026-04-09, Strict Code Review of the Three-Version Event-Conditioned Comparison Path
- **Executor**: 混合协作（Claude + Codex）
- **Why it was done**:
  - 在继续下一轮实验前，用户明确要求先严格审查 baseline / structured_v2 / structured_v2_noresid 共用代码路径，避免后续结论其实是由代码逻辑问题造成。
- **What was done**:
  1. Claude 本地复查了训练、模型、评估与指标链：
     - `run_event_conditioned_trajectory_baseline.py`
     - `event_conditioned_baseline_model.py`
     - `conditioned_trajectory_head.py`
     - `event_head.py`
     - `event_targets.py`
     - `baseline_eval_primary_aux.py`
     - `eval_event_conditioned_trajectory.py`
  2. Codex 只读严格审查了训练/模型逻辑，重点检查 baseline 与 structured_v2 的真实差异、noresid 的真实含义、teacher forcing 非对称、selection_mode / best ckpt 路径与默认值陷阱。
- **What was found**:
  1. 当前三版本比较路径里没有看到会直接推翻经验排序的致命实现 bug，但存在会显著影响“机制归因口径”的高风险点。
  2. 最关键的问题是：`event_residual_scale=0.0` 只会把 `structure_to_steer` 残差乘成 0，**不会**关掉 `structured_v2` 的其它结构注入（`structure_to_tgt` 与 `structure_to_film` 仍然完整生效）。因此 `structured_v2_noresid` 不能被解释成“去掉 structured 注入后的干净对照”，只能解释成“只关掉显式 steer residual 的 structured_v2 变体”。
  3. `teacher_forcing_ratio` 默认是 `1.0`，训练时 trajectory head 始终吃 GT event summary，而验证/测试与独立 eval 都固定吃 predicted summary。这会带来明显 train/infer gap，而且 structured_v2 因为会把 summary 进一步变成 structure tracks，理论上比 baseline 更容易从这种 teacher forcing 里受益。
  4. `selection_compare` / `best_model.pt` 逻辑本身没坏，但强结论必须满足：三版本 `selection_mode` 一致，且不要把 `best_model.pt` 误读成永远按 RMSE 选出来的模型。当前这轮 matched pair 与 noresid run 都使用 `selection_mode=legacy_rmse`，所以经验排序仍可保留为弱结论。
  5. Claude 另外确认了一个实现口径问题：当前 runner 里 tail amplitude penalty 是硬编码常开项，因此这三版比较都发生在“带 Step4 tail penalty 的统一 runner”之上，而不是完全原始未改动的 event-conditioned runner。
- **Recommended next step**:
  1. 论文/汇报口径立即收紧：不要再把 noresid 说成“去掉 structured 注入”的版本，只能说“去掉显式 structure steer residual 的版本”。
  2. 保留当前三版本的经验排序，但把它降级为“在当前统一 runner、统一 teacher forcing、统一 selection_mode 下的经验结果”，不要上升为更强的纯机制结论。
  3. 如果后续还要做更强机制归因，优先检查更上游的 structured injection（`structure_to_tgt` / `structure_to_film`）或先处理 teacher forcing 非对称，再谈 residual 分支归因。

### Supplemental Record: 2026-04-09, Re-alignment of What the Three Existing Runs Actually Support After Strict Code Review
- **Executor**: Claude
- **Why it was done**:
  - After the strict code review, the project needed a clean restatement of what the existing three runs really compare, so later decisions are not built on overstated mechanism claims.
- **What was done**:
  1. Re-opened the three concrete run configs and selection outputs:
     - `EXP_EVENT_CONDITIONED_MATCHED_BASELINE_20260409_005715`
     - `EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_20260409_010302`
     - `EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_NORESID_20260409_094418`
  2. Re-checked that all three runs share the same:
     - manifest
     - seed `2026`
     - device `cuda`
     - epochs/min_epochs/patience
     - batch size / lr / weight decay / grad clip
     - `event_loss_weight=0.5`
     - `teacher_forcing_ratio=1.0`
     - `selection_mode=legacy_rmse`
     - model dims
     - `structure_width=0.065`
     - `gate_temperature=0.04`
     - and the same current runner, which still includes the always-on Step4 tail amplitude penalty
  3. Confirmed the only intended config differences are:
     - baseline vs structured_v2: `conditioning_mode`
     - structured_v2 vs structured_v2_noresid: `event_residual_scale`
  4. Re-checked `selection_comparison.csv` in all three runs and confirmed `legacy / structure / active` are identical within each run, so these results should be read as single-checkpoint outcomes rather than three materially different best-model variants.
- **What was found**:
  1. The current three-run comparison still supports a valid empirical ranking under the current shared training/eval regime.
  2. What it supports is **not**:
     - a clean original baseline vs original structured_v2 comparison
     - nor a clean “structured_v2 with all structured injection removed” comparison
  3. What it **does** support is:
     - baseline under the current Step4-modified runner
     - structured_v2 under the same runner
     - structured_v2 with only the explicit steer residual disabled under the same runner
  4. Therefore the strongest safe conclusions are:
     - `structured_v2` still carries useful signal under the current shared runner
     - disabling only the explicit steer residual does not repair the boundary problem and also harms overall/tail performance
     - the boundary failure cannot be reduced to that one residual branch alone
  5. Conclusions that must be downgraded are:
     - “noresid proves the whole structured branch is not the cause”
     - “the residual branch is the main mechanism”
     - “these runs isolate pure architecture effects independent of teacher-forcing asymmetry”
- **Recommended next step**:
  1. Use the existing three runs only as bounded empirical evidence under the current shared runner.
  2. Keep mechanism language narrow and explicit in all later summaries.
  3. If the next experiment is meant to strengthen mechanism inference, prefer either an upstream structured-path ablation or a teacher-forcing-symmetry cleanup rather than another loose interpretation of the current three runs.

### Safe conclusion matrix for the current three runs
| Claim | Status |
| --- | --- |
| Under the current shared runner, `structured_v2` improves some structure-oriented metrics while worsening boundary continuity | Keep |
| Under the current shared runner, turning off only the explicit structure steer residual does not fix the boundary failure and hurts overall/tail performance | Keep |
| `structured_v2_noresid` is a full “remove structured injection” control | Downgrade / reject |
| Current three runs isolate pure residual-branch causality | Downgrade / reject |
| Current three runs remain empirically fair with respect to manifest/seed/selection_mode and current runner settings | Keep |

### Config anchors for the re-aligned interpretation
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_20260409_005715/run_summary.json:257`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_20260409_010302/run_summary.json:257`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_NORESID_20260409_094418/run_summary.json:257`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_20260409_005715/selection_comparison.csv:1`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_20260409_010302/selection_comparison.csv:1`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_NORESID_20260409_094418/selection_comparison.csv:1`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:471`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:717`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:131`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:147`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:152`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_head.py:152`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_head.py:157`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py:605`

### Pending note
- split / leakage reviewer status should still be merged if it later returns something stronger than the current code-review findings.

### Supplemental Record: 2026-04-09, Second Hard-Judgment Experiment — `structured_v2` with TF0 and Sharper Gate (`gate_temperature=0.020`)
- **Executor**: Claude
- **Why it was done**:
  - After the TF0 pair showed that teacher-forcing asymmetry was a major confounder, the next priority was to test the remaining main mechanism hypothesis directly: whether sharper structure gating can reduce the boundary regression without destroying the structure-oriented gains.
- **What was done**:
  1. Ran a formal single-variable experiment on top of the TF0 setting:
     - run root: `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_GATE002_20260409_104119`
     - kept `conditioning_mode=structured_v2`, `teacher_forcing_ratio=0.0`, `event_residual_scale=1.0`, manifest, seed, device, epochs, optimizer, model dims, `structure_width=0.065`, and `selection_mode=legacy_rmse` fixed
     - changed only `gate_temperature: 0.040 -> 0.020`
  2. Compared the resulting run against:
     - `structured_v2_TF0`
     - `baseline_TF0`
- **What was found**:
  1. Lowering `gate_temperature` from `0.040` to `0.020` produced a modest but real improvement over the original TF0 structured_v2 run:
     - `steer_rmse`: `0.499995 -> 0.498674`
     - `boundary_shift_abs_err`: `0.667954 -> 0.623819`
     - `turning_count_abs_err`: `1.548387 -> 1.536290`
  2. But the sharper-gate run still does not cleanly beat the TF0 baseline on the boundary metric:
     - baseline_TF0 `boundary_shift_abs_err = 0.550237`
     - structured_v2_TF0_GATE002 `boundary_shift_abs_err = 0.623819`
  3. It also gives back some of the TF0 structured_v2 tail advantage:
     - `rmse_tail_abs_steer`: `0.392474 -> 0.407465`, which is worse than the plain TF0 structured_v2 run and still better than baseline_TF0 `0.458622`
  4. The overall meaning is now sharper:
     - teacher forcing asymmetry was indeed a major confounder
     - gate sharpness also matters and can move the boundary metric in the expected direction
     - but sharper gating alone does not fully close the boundary gap to baseline
- **Recommended next step**:
  1. Stop treating this as a hidden implementation bug hunt; the remaining issue now looks like a genuine model trade-off inside the structured path.
  2. If continuing this line, the next cleanest move is not more random knob search but either:
     - one more narrow sharpness/structure-track variant, or
     - a direct upstream structure-path ablation beyond the currently exposed CLI knobs.
  3. If the project goal is faster thesis progress rather than deeper branch rescue, it is also now reasonable to freeze the conclusion as: TF0 improves fairness of the comparison, sharper gating helps somewhat, but structured_v2 still does not cleanly dominate baseline on boundary continuity.

### TF0 comparison snapshot after the gate experiment
| Run | overall steer RMSE | tail RMSE | peak_time_abs_err_s | boundary_shift_abs_err | turning_count_abs_err | selection_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_TF0 | 0.510060 | 0.458622 | 0.718347 | 0.550237 | 1.838710 | 0.964632 |
| structured_v2_TF0 | 0.499995 | 0.392474 | 0.528629 | 0.667954 | 1.548387 | 0.877028 |
| structured_v2_TF0_GATE002 | 0.498674 | 0.407465 | 0.537702 | 0.623819 | 1.536290 | 0.890966 |

### Key anchors for the gate-temperature result
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_TF0_20260409_102438/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_20260409_101029/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_GATE002_20260409_104119/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_GATE002_20260409_104119/selection_comparison.csv`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:734`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:68`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:86`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:90`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:94`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:97`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:147`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:152`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:717`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:734`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:736`

### Current practical reading
- The structured branch is still not a clean default replacement for baseline.
- Under TF0, it does preserve useful structure-oriented gains.
- Sharper gating moves the boundary metric in the right direction, but not enough to claim closure.
- The remaining gap now looks more like a real architecture trade-off than a simple hidden bug.

### Next decision point
- Either stop this branch with a bounded negative/conditional conclusion,
- or allow one more tightly scoped structure-path experiment before freezing the result.

### Supplemental Record: 2026-04-09, Formal Freeze / Closeout of the Current `structured_v2` Rescue Branch
- **Executor**: Claude
- **Why it was done**:
  - After the fairness closure, no-residual test, strict code review, TF0 paired control, and sharper-gate follow-up, the branch had accumulated enough evidence to make a stop/continue decision without further small-step probing.
  - The user explicitly chose to close the branch rather than continue spending time on incremental rescue attempts.
- **What was done**:
  1. Consolidated the entire 2026-04-09 sequence into a final branch-level judgment instead of leaving the state spread across multiple intermediate notes.
  2. Froze the current interpretation of what has been learned from:
     - matched baseline vs structured_v2
     - structured_v2_noresid
     - structured_v2_TF0
     - baseline_TF0
     - structured_v2_TF0 with `gate_temperature=0.020`
- **What was found**:
  1. `structured_v2` is **not** a dead branch. Under fairer TF0 conditions, it still preserves meaningful structure-oriented gains over baseline in several metrics.
  2. But the current branch also does **not** meet the standard for becoming the new default model.
  3. The key blocking issue remains boundary continuity:
     - the original matched structured_v2 run worsened `boundary_shift_abs_err` sharply
     - removing only the explicit steer residual did not repair that problem
     - teacher-forcing symmetry improved the fairness of the comparison and materially changed the observed ranking
     - sharper gating moved boundary in the right direction but still did not close the gap to baseline
  4. The branch therefore ends in a bounded, negative/conditional state:
     - **research value retained**
     - **default replacement claim not supported**
     - **rescue not completed**
- **Final branch verdict**:
  - Keep as a documented research branch with partial gains and unresolved boundary trade-off.
  - Do **not** promote to default baseline replacement.
  - Do **not** continue routine knob-scanning on this branch in the current cycle.
- **Recommended next step**:
  1. Shift the main project back to a cleaner, more stable baseline-facing path for actual model progress.
  2. If this branch is revisited later, do it only under a new, clearly scoped mechanism study (for example: explicit upstream structured-path ablation or architecture redesign), not as another round of small hyperparameter rescue.
  3. In thesis/paper language, describe this branch as:
     - structured conditioning shows real signal,
     - but current implementation exhibits unresolved boundary-local continuity trade-offs,
     - so it remains exploratory rather than production/default.

### Final closeout table for the current branch
| Question | Final answer |
| --- | --- |
| Does `structured_v2` still have signal? | Yes |
| Does it cleanly beat baseline under current evidence? | No |
| Was the problem just the explicit steer residual? | No |
| Was teacher forcing asymmetry a major confounder? | Yes |
| Did sharper gating fully solve the branch? | No |
| Should this branch remain the active default mainline? | No |

### Current status update after closeout
- The current `structured_v2` rescue branch is now considered closed.
- Future work in this area should require a fresh, explicitly justified mechanism proposal rather than continuation by inertia.
- The project's practical model-progress focus should move back to cleaner baseline-centered work unless a new branch is opened intentionally.

### Key evidence anchors for the closeout
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_20260409_005715/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_20260409_010302/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_NORESID_20260409_094418/run_summary.json`

### 补充记录（2026-04-13，道路基准更新）
- 重新核对 SILAB 道路拼接来源后，确认本轮实验实际入口为 `G:/GZSY2026/zx/projects/exercise1.cfg` 中的 `("大货车紧急变道", section1.Port1)`，当前主线路径应按 `section1 -> longstraight -> section2 -> curve1 -> section3 -> fix_road -> section4 -> curve2 -> section5 -> stop -> section6 -> mu1 -> section7 -> curve3 -> section8 -> zd -> section9` 理解，其中 `zd` 为带分支的闸道模块，主线使用 `Port1 -> Port2`，分支为 `Port4 -> Port3`。
- 更正了此前对“旧版道路信息来源”的错误判断：之前误把旧道路理解成 `leave_highroad`；本次回查后确认，旧项目里真正落地使用的道路文件是 `F:/data_set_process/data_process/datasetprocess/多模态数据/被试数据集合/道路信息/full_centerline_layout.csv`，其上游中间结果来自 `F:/data_set_process/data_process/datasetprocess/多模态数据/被试数据集合/道路中心线融合结果_聚类版/road_centerline_main_route.csv`，对应旧版查看/核对文件为 `F:/data_set_process/data_process/datasetprocess/多模态数据/被试数据集合/道路信息/proving_ground_full_path_v3_tangent.csv`。
- 对比结果上，旧版 `full_centerline_layout.csv` 的主线在 `zd` 处结束，不包含当前 `exercise1` 中继续延伸到 `section9` 的部分；同时 `curve3`、`zd`、`middle_section`、`fix_road`、`longstraight` 等模块与当前版本内容已不一致，因此不能再把旧版中心线当作当前真实道路基准。
- 已完成一次基于真实车辆轨迹的随机抽样核对：从清洗后的真实轨迹文件中固定随机种子 `20260413` 抽取 6 组样本，对旧版道路与当前 `exercise1` 道路分别做最佳滑窗匹配、刚体对齐并比较 RMSE。结果为当前道路 `6/6` 样本全部优于旧版道路，旧版平均 RMSE `690.9 m`，当前道路平均 RMSE `43.5 m`。
- 因此本项目后续涉及“真实道路轨迹”“道路中心线”“道路基准路径”时，统一以当前 `exercise1` 对应道路版本为准，不再默认沿用旧项目中的 `full_centerline_layout.csv`。
- 本次结论的支撑材料已输出至 `G:/GZSY2026/zx/projects/`，重点包括：
  - `actual_random_trajectory_match_report.txt`
  - `actual_random_trajectory_match_comparison.png`
  - `exercise1_old_full_layout_overlay_report.txt`
  - `exercise1_truck_lane_change_centerline.svg`
  - `exercise1_truck_lane_change_trajectory.svg`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_20260409_101029/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_TF0_20260409_102438/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_GATE002_20260409_104119/run_summary.json`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:717`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:734`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:736`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:131`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:147`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:152`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_head.py:152`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py:605`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:367`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:369`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:386`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:388`

### End state for this cycle
- Stop here.
- Treat the current evidence as sufficient for branch closeout.
- Do not spend more GPU budget on this line unless a new mechanism hypothesis is approved.

### 补充记录（2026-04-13，protocol-safe full-run 基线评估闭环缺口确认）
- 执行主体：Claude
- Why：用户已把当前目标明确收口到“让 maintained 主线在 protocol-safe full test 上更可信地预测极限工况下驾驶员行为和车辆状态趋势”，而最终验收不再只看 RMSE，还要求趋势相关性、主响应峰值时间误差和明确方向响应样本上的反向预测率；因此需要先确认当前 full run 是否已经具备这些证据，避免在缺关键评估闭环的情况下盲目调参或重训。
- What was checked：
  1. 读取项目规则与目标驱动文档：`CLAUDE.md`、`reports/goal_driven_autonomous_workflow.md`、`reports/goal_driven_target_template.md`。
  2. 回看 `reports/project_progress_master.md` 顶部速览与 2026-04-13 最近相关记录，确认当前 maintained 主线已经完成 protocol-safe split 落地，现有 full run 基线为 `tmp/protocol_safe_runs/TRAIN_V5_8_PROTOCOL_SAFE_20260413_174639`。
  3. 读取该 run 的 `run_config.json`、`training_summary.json`、`figures/test_metrics.json`、`figures/test_metrics_by_reversal.json`、`figures/test_state_dump.csv`，核对当前已产出的 test 证据。
  4. 只读检查 active 主线脚本 `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 中的 `evaluate_and_plot()`、dataset `__getitem__()`、run 结束后的评估导出路径，确认脚本当前到底输出了哪些 test-level / event-level 信息。
- What was found：
  1. 当前 protocol-safe full run 的基础 test 指标已清楚：`rmse_steer=0.6557`、`rmse_yawrate=0.1339`、`rmse_ay=2.1182`，且 split 审计、run_config、reversal metrics、state dump 都已存在。
  2. 现有评估闭环仍缺用户这次明确要求的三类关键验收证据：
     - steer 趋势相关性（至少 event-level 分布统计，如 median / P10 / P90）
     - 主响应峰值时间误差（至少 median / P90）
     - 明确方向响应样本上的反向预测率
  3. 当前 `evaluate_and_plot()` 只保存了整体 RMSE/MAE、按 road type 的 RMSE、按 reversal label 的分类指标，以及 `test_state_dump.csv` 中的 latent / reversal / event RMSE / GT peak 幅值；并没有直接导出 `pred` / `true` 全序列级指标，也没有输出上述趋势/延时/反向预测统计。
  4. 当前 `test_state_dump.csv` 里只有 dataset 内部 `idx`，而 `selected_samples_with_split.csv` 有 `sample_key/subject_id/event_idx/anchor_idx/protocol_split_applied` 等元信息；两者原则上可通过 test subset 内部顺序或补充导出字段对齐，但当前脚本尚未把这层映射直接整理成一个完整的 test-event 审计表。
  5. 因此下一步最合理的切片不是先盲目调 loss 或直接重训，而是先把 evaluation 闭环补齐，并基于当前 full run checkpoint 重跑一次完整 test 评估，把这三类验收指标补出来，形成真正可用于后续调参/重训对比的基线。
- Recommended next step：
  1. 在 maintained 主线脚本里最小扩展 test 导出：保留现有 protocol/split/anchor/horizon 不变，只新增趋势相关性、主峰值时间误差、反向预测率及必要的 event-level 审计字段/汇总文件。
  2. 优先复用现有 full run `TRAIN_V5_8_PROTOCOL_SAFE_20260413_174639` 的 best checkpoint 做一次“只重评估不重训练”的完整基线补打，先拿到当前真实 baseline 的完整验收面板。
  3. 只有在完整基线显示主要短板位于趋势、延时或反向预测某一项之后，再决定下一轮改动应优先打哪一类机制，而不是先凭直觉改 loss。

### Supplemental Record: 2026-04-09, Next Mainline Reframed to Stay Compatible with Future Physiological Signal Streaming
- **Executor**: Claude
- **Why it was done**:
  - After closing the current `structured_v2` rescue branch, the user clarified an additional long-range requirement: the future model should be able to accept physiological signal streams later, because the dataset already contains physiological channels and the project should not lock itself into a vehicle-only dead end.
- **What was done**:
  1. Reframed the post-closeout mainline from a purely vehicle/event-only baseline refresh into a **multimodal-ready baseline-conditioned reaction-state-aware** direction.
  2. Explicitly separated:
     - what must be validated now using the current stable vehicle/event path
     - what must be reserved architecturally so a future online physiological branch can be added without redesigning the whole model
- **What was found**:
  1. The immediate next stage should still avoid jumping straight into a heavy multimodal system, because that would mix in alignment, missingness, latency, online-visibility, and leakage complexity before the response-state task itself is stabilized.
  2. But the next-stage mainline also should not be designed as a vehicle-only dead structure, because that would force a later rewrite when physiological streams are introduced.
  3. Therefore the most practical framing is:
     - use the current vehicle/event path to validate the response-state modeling problem first
     - keep the encoder / state-token / conditioning interface modular enough that a future physiological stream encoder can be attached cleanly
- **Recommended next step**:
  1. Define the new mainline as a **multimodal-ready baseline-conditioned reaction-state-aware model**.
  2. In the first implementation wave, prioritize modular interfaces over immediate multimodal complexity:
     - pluggable input branches
     - reusable state token / condition token
     - decoder that does not assume only one modality forever
  3. Only after the response-state target design is stable should the project begin a physiology-stream integration phase.

### Updated mainline wording after this clarification
- Near-term goal: build a stable baseline-conditioned response-state modeling path on vehicle/event inputs.
- Mid-term extension goal: add physiological signal streaming as an additional online branch, not as an afterthought rewrite.
- Guardrail: do not let future multimodal ambition derail the current effort to define the right response-state target first.

### Practical design guardrails implied by the new requirement
| Layer | Current-cycle expectation | Future physiology-ready expectation |
| --- | --- | --- |
| History encoder | can start from vehicle/event inputs | should allow adding a physio encoder branch later |
| State / condition representation | should already be explicit and reusable | should be able to concatenate / fuse physiology-derived state later |
| Decoder | should focus on response-state-aware prediction | should not hard-code a single-modality assumption |
| Evaluation | first validate on current stable path | later compare whether physiology helps onset / peak / amplitude / recovery |

### Current practical reading after the physiology clarification
- The project should **not** jump straight into full multimodal training next.
- The project **should** redesign the next baseline-centered mainline so that future physiology integration is natural rather than disruptive.
- This keeps the work aligned with the real long-term goal: driver-response modeling useful for later intervention / shared-control support, not just one more vehicle-only regression model.

### Immediate next implementation focus after closeout
1. Define the response-state-aware baseline-conditioned mainline.
2. Make its interfaces multimodal-ready by construction.
3. Keep first validation on the cleaner vehicle/event path.
4. Plan physiology-stream integration as the next phase, not as the current restart point.

### End state after this clarification
- `structured_v2` rescue branch remains closed.
- The next active line is now explicitly both:
  - closer to the research goal of real driver-response modeling
  - and architecturally compatible with future physiological stream integration.
- This is now the preferred framing for the next implementation cycle.

- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:717`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:128`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_head.py:147`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_speed_subjectsplit_masked.py:16`
- `F:/data_set_process/data_process/CLAUDE.md`
- `F:/data_set_process/data_process/reports/project_progress_master.md`
- `F:/data_set_process/data_process/reports/gptpro_structured_v2_closeout_pack_20260409.zip`

### End state for this cycle
- Stop here.
- Treat the current evidence as sufficient for branch closeout.
- Do not spend more GPU budget on this line unless a new mechanism hypothesis is approved.

- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_TF0_20260409_102438/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_20260409_101029/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_GATE002_20260409_104119/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_TF0_GATE002_20260409_104119/selection_comparison.csv`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:734`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:68`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:86`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:90`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:94`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:97`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:147`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:152`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:717`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:734`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:736`

### Current practical reading
- The structured branch is still not a clean default replacement for baseline.
- Under TF0, it does preserve useful structure-oriented gains.
- Sharper gating moves the boundary metric in the right direction, but not enough to claim closure.
- The remaining gap now looks more like a real architecture trade-off than a simple hidden bug.

### Next decision point
- Either stop this branch with a bounded negative/conditional conclusion,
- or allow one more tightly scoped structure-path experiment before freezing the result.

### Key file anchors for the strict review
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:448`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:461`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:471`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:560`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:603`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:717`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:736`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_conditioned_baseline_model.py:188`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:131`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:147`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/conditioned_trajectory_head.py:152`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/event_head.py:152`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py:551`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py:605`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:359`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:367`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:369`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:376`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:386`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/baseline_eval_primary_aux.py:388`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py:599`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py:605`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:49`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:51`
- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py:479`

### Pending note
- split / leakage reviewer 仍在后台，若它补出更硬的风险点，需要再合并进最终判断。

- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_BASELINE_20260409_005715/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_20260409_010302/run_summary.json`
- `F:/data_set_process/data_process/tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_MATCHED_STRUCTV2_NORESID_20260409_094418/run_summary.json`

### 补充记录（2026-04-14，按用户要求仅回退训练相关文件到昨晚版，并诊断“长值固定不变”问题）
- 执行主体：Claude
- Why：用户明确指出此前目标驱动推进后期已经偏离真实目标，当前最重要的问题不是继续扩展 response-state-aware / teacher-state 工程链路，而是模型预测出现“长值固定不变/后段发平”的失败现象；同时用户已经不希望做仓库级大回退，而是优先按最小破坏面只回退训练相关文件到“昨天晚上的版本”，再基于该版本判断后续方向。
- What was done：
  1. 先回看 `reports/project_progress_master.md`、项目 memory 与当前 maintained 主训练脚本，确认近期主线已经逐步转向 `response-state-aware + pca_latent teacher-state`，并且项目日志里已明确提醒：maintained 主线仍未落实 protocol 要求的 `subject-level fixed split`，现有 smoke/短跑结果不能直接当正式可信证据。
  2. 因为用户指定的是“昨晚版”而不是明确 git commit，先做 git/history 与 run-artifact 双向追查，确认这不是 git 中已提交的稳定版本，而是保存在实验输出目录里的脚本副本；随后按用户选择的最小范围方案，只回退训练相关文件，不碰其余工作区改动。
  3. 在覆盖前对 active maintained 文件做本地安全备份：
     - `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py.backup_20260414_before_revert`
  4. 将 maintained 主训练脚本恢复为昨晚实验目录中的脚本副本：
     - 源：`F:\数据集处理\data_process\datasetprocess\多模态数据\程序运行结果\TRAIN_V5_4_STATECOND_REV_20260413_144211\future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
     - 目标：`datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
  5. 回退后继续只读诊断该“昨晚版”到底是不是更早、更简单的旧基线，还是其实仍属于最近这条 response-state-aware 主线，并围绕其 run artifacts 做针对性核查：
     - `run_config.json`
     - `teacher_state_meta.json`
     - `loss_history.csv`
     - `figures/test_metrics.json`
     - `figures/test_metrics_by_reversal.json`
     - `figures/test_metrics_by_roadtype.json`
     - `figures/test_state_meta.json`
     - 多张 `pred_vs_gt_example_*.png`
- What was found：
  1. **“昨晚版”并不是更早的纯 baseline，而仍是 V6.0 家族的 response-state-aware 版本。**
     - `MODEL_VER = v5_8_response_state_v1`
     - `ENABLE_RESPONSE_STATE_V1 = true`
     - `ENABLE_STATE_DISTILL = true`
     - `ENABLE_REVERSAL_AUX = true`
     - `ENABLE_PEAKTIME_AUX = true`
     - `ENABLE_PEAKINTENSITY_AUX = true`
     - `TEACHER_STATE_MODE = pca_latent`
     - `TEACHER_STATE_DIM = 4`
     这说明即便回到昨晚，代码仍未退回到“完全没有 teacher-state / response-state machinery”的旧线，而只是回到这一新主线的较早可运行版本。
  2. **该版本在 800/200、2 epoch、cuda 的短跑上训练闭环是正常的。**
     - `loss_history.csv` 显示 train `2.1259 -> 1.5337`，val/test loss `1.5313 -> 1.4311`
     - `figures/test_metrics.json` 显示：`rmse_steer=0.7672`、`rmse_yawrate=0.1661`、`rmse_ay=2.9883`
     也就是说问题不像“完全没学到”或训练直接炸掉，更像是学到了过平滑/过保守的响应形态。
  3. **用户描述的“长值固定不变”现象与诊断结果一致，主要表现为后段预测被明显拉平/发钝，而不是简单整体偏移。**
     - 从多张 `pred_vs_gt_example_*.png` 观察，预测序列在后段常出现持续平滑、波动幅度收缩、向近常值趋势靠拢的形态；
     - 这与用户口头描述的“预测结果都是长值固定不变”是同一类失败模式，说明此前用户的主观判断是有 artifact 支撑的，不是误报。
  4. **当前最尖锐的辅助分支失败点不是 PCA latent 本身跑不通，而是 strong reversal 分支几乎塌掉。**
     - `figures/test_metrics_by_reversal.json`：`used_label=strong`，`rate_used=0.045`
     - 强反转：`tp=0, fp=60, fn=9, f1=0.0`
     - 弱反转：`tp=29, fp=31, fn=66, f1≈0.374`
     这更像“稀有强反转标签 + 当前损失配置/阈值组合不稳”，会把模型往过保守、少做尖锐结构变化的方向推。
  5. **因此，当前失败更像‘新主线自身已有可跑通闭环，但响应形态学上过于平滑，且强反转监督失真’，而不是单纯因为用户回退错了一个完全不可用版本。**
- Current practical judgment：
  1. 这次“回到昨晚版”已经完成了用户要求的最小回退，但它**没有**把主线退回到一个“干净旧 baseline”；它仍然处在 response-state-aware / pca-latent 这条新分支内部。
  2. 所以如果后续目标是彻底摆脱最近这条新分支带来的过平滑/长值固定问题，仅停留在“昨晚版”并不够，原则上还需要继续向更早、更简单的版本回退，或在当前版本里显式关掉/拆掉 response-state 相关支路。
  3. 如果后续目标是保留最近新主线的大体结构，只修复当前最影响观感和可信度的失败模式，那么首要怀疑对象应收口到：
     - strong reversal 标签/阈值与采样稀疏度
     - 反转辅助损失对主轨迹形态的保守化牵引
     - 以及 teacher-state / aux 分支是否把后段预测过度收缩到平滑均值轨道
- Recommended next step：
  1. 把后续主动作收口成二选一：
     - **路径 A：继续回退**到更早、真正不带 `response_state_v1 + pca_latent` 的训练版本，重新建立一个更干净的可比基线；
     - **路径 B：留在当前昨晚版**，但优先做“去塌缩/去长值固定”的定向修复，而不是继续往多模态 readiness 方向加东西。
  2. 若走路径 B，最小优先级应是先削弱或关闭 strong reversal 相关牵引，再检查预测后段是否恢复幅度和结构变化；不要先继续堆新 teacher-state 设计。
  3. 在正式可信比较前，仍要记住这条 maintained 主线当前脚本本身存在 split protocol 不安全问题；后续任何“变好了/变坏了”的结论，最多先作为方向性诊断，不应直接当论文级证据。

### 补充记录（2026-04-14，回退后第一轮最小修复：接通反打样本加权并强化局部 reversal 区域）
- 执行主体：Claude
- Why：用户明确要求“回退之后继续来解决问题”，因此本轮不再继续讨论是否再回退，而是直接基于已恢复的昨晚版训练脚本，做最小、定向、可验证的 anti-flattening 修复。只读排查后确认，当前脚本里本来已经有多处用于保留反打/形态结构的机制，但其中关键的一条——`REV_SAMPLE_WEIGHT`——并没有真正作用到 active regression loss；同时 train 侧的局部 steer 加权还把 reversal 序列直接置零，导致 hard reversal case 容易继续被均值化。
- What was changed：
  1. 在 `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 中，保留原模型结构、split、horizon、anchor 和 teacher-state 逻辑不变，只修改损失侧。
  2. 将 `W_STEER_REV` 从 `0.0` 调整到 `0.35`，作为第一轮温和增强，而不是直接大幅打开更多新损失项。
  3. 新增一组最小 helper，把原本 batch-aggregate 的回归损失改造成可按样本加权的 per-sample reduction：
     - `weighted_mean_per_sample(...)`
     - `mse_per_sample(...)`
     - `weighted_mse_loss_per_sample(...)`
     - `weighted_channel_task_loss(...)`
     - `weighted_steer_local_mse(...)`
     - `build_reversal_sample_weight(...)`
  4. 将 `REV_SAMPLE_WEIGHT` 真正接入 active regression objective：
     - 主任务 steer/yaw/ay 序列 MSE
     - amplitude loss
     - 一阶差分 loss
     - 二阶差分 loss
     - `loss_steer_wt`
     现在反打正样本会在这些回归项中获得更高权重，而不再只是配置里声明、实际未生效。
  5. 新增 `compute_active_task_losses(...)`、`compute_reversal_shape_losses(...)`、`compute_total_task_loss(...)`，把 train / val 的 active 目标统一收口成同一套逻辑，避免训练目标与选择目标再次错位。
  6. train 侧不再把局部 reversal 序列写成全零，而是和 val 一样使用 GT soft reversal 序列参与局部 steer 加权；这样 `W_STEER_REV` 才真正能作用到 tail/reversal 区域，而不是只有名义开关。
- What stayed unchanged：
  1. `LAMBDA_REV`、`LAMBDA_STATE`、`W_REVSEQ`、`W_PEAKTIME` 本轮都没有进一步扩大，避免一次同时打开太多支路，把问题重新变复杂。
  2. strong/weak reversal 标签定义本轮未改，teacher-state 构造也未改；本轮只修“已有机制未真正接通”的问题。
- Verification so far：
  1. 已使用 `D:/ProgramData/anaconda3/envs/predict_2/python.exe -m py_compile` 对修改后的训练脚本做语法检查并通过。
  2. 随后用同一 `predict_2` Python 启动脚本做运行层面的第一轮观察，任务输出已进入后台文件，等待继续读取结果确认是否成功进入训练闭环。
- Current practical judgment：
  1. 这一轮不是“加新结构”，而是把脚本原本宣称要做、但实际上没做成的 reversal-focused weighting 真正接通。
  2. 如果当前 flattening 的主因确实是 hard reversal case 在回归目标里被均值化，那么这一步应比直接开更多 aux head 更干净，也更符合当前用户要求的最小修复路线。
  3. 若这一步后可视化里尾段仍明显发平，再考虑第二轮只小幅打开 `W_REVSEQ` / `W_PEAKTIME`，而不是立刻动 `LAMBDA_REV` 或更换标签定义。
- Recommended next step：
  1. 先读取当前后台运行输出，确认脚本是否已顺利进入数据扫描、构样本、训练与评估导出。
  2. 若运行正常，优先对比新的 `pred_vs_gt_example_*.png` 与 `test_metrics_by_reversal.json`，重点看 strong/weak reversal 样本的 tail 是否不再过早水平化。
  3. 只有当这一步仍不足以改善后段发平时，再进入第二轮最小加法：小幅打开 `W_REVSEQ` / `W_PEAKTIME`。
