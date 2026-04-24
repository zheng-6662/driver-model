# Current Model State Summary

## 一句话判断

当前应从 `2026-04-16 220918` 这条 `coarse-fine + phase-adaptive` strongest allowed mainline 出发；`2026-04-17 100716` 已经证明 dedicated strong-pos gate source 本身有效，但当前 gate activation 仍然错误，因此下一步应优先思考 `activation / routing`，而不是回去重扫旧 scalar。

## 当前最重要的 run

| role | run | conclusion |
| --- | --- | --- |
| old-good reference | `2026-04-13 174639` | 可继续当“自然感较好”的正式参考 |
| fair baseline | `2026-04-16 103752` | 当前公平、稳的 protocol-safe baseline |
| strongest allowed mainline | `2026-04-16 220918` | 当前 overall/head/tail/late-peak 最强主线 |
| no-go | `2026-04-17 003953` | hard-late fine 能补 hard bucket，但明显破坏大盘 |
| no-go | `2026-04-17 011755` | raw `rev_logit` late gate 不够定向，没守住 `220918` |
| no-go | `2026-04-17 100716` | dedicated strong-pos gate 把 source 做准了，但 activation 仍错误 |

## 当前关键对照

| run | `rmse_steer` | `late_peak_recall` | `coarse_sign_match` | `reversal_exact_match` | `strong_pos.tail_amp_ratio` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2026-04-16 baseline 103752` | `0.6328` | `0.5855` | `0.5579` | `0.4205` | `0.6342` |
| `2026-04-16 mainline 220918` | `0.5697` | `0.5940` | `0.5728` | `0.3958` | `0.3732` |
| `2026-04-17 hard-late 003953` | `0.6273` | `0.4444` | `0.5687` | `0.5530` | `0.4193` |
| `2026-04-17 rev-gate 011755` | `0.6280` | `0.3718` | `0.5723` | `0.4830` | `0.3700` |
| `2026-04-17 strong-pos gate 100716` | `0.6491` | `0.5128` | `0.5756` | `0.4470` | `0.3601` |

## 最新新增证据

基于 `test_state_dump.csv` 的直接检查，当前已经可以把 `source` 和 `activation` 分开看：

- `220918` 里的旧 `rev_prob` 对 `rev_gt_strong` 的 AUC 约为 `0.4725`
- `100716` 里的新 `strong_pos_gate_prob` 对 `rev_gt_strong` 的 AUC 约为 `0.7044`

这说明 dedicated gate source 本身明显优于 raw `rev_logit`。但尽管 source separability 大幅提升，`100716` 依然比 `220918` 更差：

- `rmse_steer: 0.5697 -> 0.6491`
- `head_amp_ratio: 1.5739 -> 2.2357`
- `tail_rmse_steer: 0.6826 -> 0.8068`
- `late_peak_recall: 0.5940 -> 0.5128`
- `first_reversal_time_mae_sec: 0.6206 -> 0.8016`
- `strong_pos.tail_amp_ratio: 0.3732 -> 0.3601`

所以当前更强的判断是：

- `source` 不再是一阶瓶颈
- 当前问题更像是 `activation / routing` 错位

## 当前固定结论

1. `2026-04-16 103752` 仍是 fair baseline。
2. `2026-04-16 220918` 仍是 strongest allowed mainline。
3. 不要把 `2026-04-17 003953 / 011755 / 100716` 当成可升级版本。
4. 不要回去重扫 `W_STEER_RATE` / `W_REVSEQ` / `W_STEER_REV`。
5. 不要重开 `teacher-state` 或 random split 争论。
6. 如果继续 gate 路线，应该优先思考：
   - tail-only gate
   - additive gate
   - training-only gate
   - gate-driven loss / target routing
   而不是继续做乘法式 full late fine gain

## 当前最值得 GPT Pro 回答的问题

1. 既然 dedicated gate source 已经显著变准，下一刀该怎样改 activation，才能尽量保住 `220918` 的 overall/head/tail/late-peak 收益？
2. gate 是否更适合作为 `loss / target / residual selection` 的路由信号，而不是直接作为前向乘法增益？
3. 如果继续修 activation 仍然 EV 不高，是否应该直接切到更结构化的 late residual / phase-specific decoder 方案？

## 明确不建议 GPT Pro 重复投入的方向

- 回到旧 scalar sweep
- 把 `011755` 的 raw `rev_logit` late gate 再做温和调参
- 把 `100716` 的 source 再继续改得更复杂，却保持同一类乘法式 full late fine activation
- 用 smoke/random split 结果下正式判断
