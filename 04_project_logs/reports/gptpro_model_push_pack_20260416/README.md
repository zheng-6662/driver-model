# GPT Pro Handoff Pack

日期：`2026-04-16`

这个包是为了让 GPT Pro 在最短时间内进入当前主问题，而不是重新从整个仓库里摸索上下文。

## 目标

让 GPT Pro 基于当前 `protocol-safe`、公平、可复现的主线证据，提出下一步更激进但仍然有证据约束的推进方案，尽快解决当前模型的核心问题，而不是回到已经证伪或低决策价值的方向。

## 当前主问题

当前公平 baseline 已经不是“整体更晚、更弱、更平”的失败状态。`2026-04-16 repaired clean full baseline` 证明了 earlier broad regression 主要来自 regime 漂移。

现在真正剩下的问题更窄：

- 整条 `2s` 轨迹的趋势相似性还不够强
- `coarse segment` 方向一致性仍偏弱
- 某些 trend-oriented 改动会把 `head` 拉过头，或交还 `tail/peak/reversal` 的优势

## 当前公平基线

请把下面这个 run 当作当前公平出发点，而不是 4/15 的 smoke：

- `tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260416_103752`

它对应的 summary 已放在 `evidence/` 下。

## 已经得到的关键判断

1. 不要再把 `2026-04-15` smoke 当成 maintained mainline 退化证据。
2. `teacher-state` 不是当前 regression 的第一嫌疑项。
3. `W_STEER_REV` / `W_REVSEQ` 不是当前主问题的一阶解释。
4. `W_STEER_RATE=1.25` 能补 head，但代价太大，不适合当默认主线。
5. `W_TREND=0.10` 的 pooled-level 版本，是目前最接近主目标的方向。
6. 直接切到 `direction-aware coarse-delta` 主导的 trend loss，不是更好的默认主线。

## 建议 GPT Pro 的阅读顺序

1. `MODEL_STATE_SUMMARY.md`
2. `GPTPRO_PROMPT_CN.md`
3. `context/project_progress_hub.md`
4. `context/daily_2026-04-16.md`
5. `context/experiment_registry.md`
6. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
7. `protocol/protocol_config.json`
8. `evidence/` 下的各 run summary / cases / run_config

## 包内结构

- `code/`
  - 当前 active 训练脚本
  - 当前重算指标脚本
- `protocol/`
  - 当前主协议配置
  - 当前 frozen subject split
- `context/`
  - 当前中枢页
  - 今日日志
  - 实验登记表
- `evidence/`
  - old-good full baseline
  - repaired clean full baseline
  - `W_STEER_RATE=1.25` full run
  - `W_TREND=0.10` pooled-level full run
  - `direction-aware coarse-delta` trend full run

## 当前 active 代码状态

当前 active script 已恢复为更稳的 trend loss 默认：

- `TREND_LOSS_MODE = pooled_level_mse_v1`

同时保留了可切换实验模式：

- `TREND_LOSS_MODE = pooled_delta_direction_v1`

也就是说，GPT Pro 如果要建议继续沿 trend objective 推进，可以基于当前 active 脚本直接提“如何更合理地保留 pooled-level 主目标，再小幅叠加 delta / direction 约束”，而不需要再从零设计接口。

## 对 GPT Pro 的明确要求

- 不要保守地只给泛泛建议。
- 请给出能直接推进模型的、代码级可落地的下一步方案。
- 允许提出比当前更激进的建模切片，但必须明确：
  - 为什么它比继续扫旧 scalar 更值
  - 它预期会推动哪些指标
  - 风险是什么
  - 是该先 smoke 还是直接 full-regime

## 红线

- 不要回到 `smoke/random split` 去下正式判断
- 不要破坏 `subject-level fixed split`
- 不要引入明显 leakage
- 不要把“旧 broad regression 假说”当成现在的主问题
- 不要建议只看总体 RMSE 而忽略 head / tail / peak / reversal / trend

