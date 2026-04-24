# GPT Pro Handoff Pack

日期：`2026-04-17`

这个包是给 GPT Pro 的“当前主线续推包”。目标不是让它重新从 4 月中旬的全部历史里摸索问题，而是让它直接接住最新主线和最新 no-go 证据，继续判断下一步最高 EV 的推进方向。

## 目标

让 GPT Pro 基于当前 `protocol-safe`、`subject-level fixed split`、可复现的主线证据，回答：

- 当前还值不值得继续沿 gate 路线推进
- 如果继续，下一刀应该改 `activation`、`objective routing` 还是更结构化的 late residual target
- 如果不继续，什么更激进但仍有证据约束的新切片更值得直接做

## 当前固定状态

请把下面 4 条当成已经锁定的当前状态，不要再从旧结论重来：

1. 当前公平 baseline：`TRAIN_V5_4_STATECOND_REV_20260416_103752`
2. 当前 strongest allowed mainline：`TRAIN_V5_4_STATECOND_REV_20260416_220918`
3. 最新 gate-route no-go：
   - `2026-04-17 003953`：hard-late fine
   - `2026-04-17 011755`：raw `rev_logit` late rev gate
   - `2026-04-17 100716`：dedicated strong-pos gate
4. 最新新增证据：gate source 已显著变准，但当前 activation 仍会把大盘和目标 bucket 一起带坏

## 当前主问题

现在最值得问的已经不是“要不要继续扫旧 scalar”，也不是“teacher-state 是否有问题”。

真正剩下的问题更窄：

- `220918` 已经是当前 overall/head/tail/late-peak 最强主线
- 但它在 `strong_pos / reversal exact-match` 上仍有缺口
- `011755` 说明 raw `rev_logit` 作为 gate source 不够定向
- `100716` 说明 dedicated gate source 可以把 separability 明显做对，但当前乘法式 full late fine activation 仍然不对

也就是说，现在更像是：

- `source` 已经不再是一阶瓶颈
- `activation / routing` 才是更高 EV 的下一刀

## 建议 GPT Pro 的阅读顺序

1. `MODEL_STATE_SUMMARY.md`
2. `evidence/20260417_gate_source_activation_diagnosis.md`
3. `evidence/20260417_hardlate003953_from_daily_log.md`
4. `GPTPRO_PROMPT_CN.md`
5. `context/daily_2026-04-16.md`
6. `context/experiment_registry.md`
7. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
8. `code/recalc_v58_checkpoint_with_current_metrics.py`
9. `protocol/protocol_config.json`
10. `evidence/20260416_fair_baseline103752_summary.json`
11. `evidence/20260416_mainline220918_summary.json`
12. `evidence/20260417_revgate011755_summary.json`
13. `evidence/20260417_strongposgate100716_summary.json`
13. 必要时再看对应 `run_config.json`、`cases.csv` 和 `test_state_dump.csv`

## 包内结构

- `code/`
  - 当前 active 训练脚本
  - 当前重算指标脚本
- `protocol/`
  - 当前主协议配置
  - 当前 frozen subject split
- `context/`
  - 当前进展锚点快照
  - handoff prompt
- `evidence/`
  - old-good formal reference
  - fair baseline `103752`
  - strongest mainline `220918`
  - raw late rev gate `011755`
  - dedicated strong-pos gate `100716`
  - gate source vs activation diagnosis note
  - hard-late fine `003953` 的日志摘要说明

## 关于 active 代码

当前 active script 保留了以下路径，但并不意味着它们都应作为默认主线：

- `coarse-fine`
- `phase-adaptive trend`
- `late rev gate`
- `dedicated strong-pos gate`
- `hard-late fine`

当前默认判断仍是：

- `220918` 对应的 `coarse-fine + phase-adaptive` 是 strongest allowed mainline
- `011755` 和 `100716` 都是 `no-go`
- 后续如果继续 gate 路线，应该优先改 activation / routing，而不是回头继续扫旧 scalar

## 对 GPT Pro 的明确要求

- 不要回到 `W_STEER_RATE` / `W_REVSEQ` / `W_STEER_REV` 这类旧 scalar 重扫
- 不要重开 `teacher-state` 或 random split 争论
- 可以更激进，但必须明确：
  - 为什么它比继续小修小补更值
  - 它预期会拉动哪些 guardrail 指标
  - 风险是什么
  - 应该先 smoke 还是直接 full-regime

## 红线

- 不要破坏 `subject-level fixed split`
- 不要把 smoke/random split 当正式证据
- 不要只盯总体 `rmse_steer`
- 不要把 `011755 / 100716` 的 no-go 当作“gate 路线已完全无价值”的充分证据，除非你能解释为什么 source 已修对但 activation 仍应被整体放弃
