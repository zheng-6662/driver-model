# Current Model State Summary

## 一句话判断

当前应该从 `2026-04-16 repaired clean full baseline` 出发，围绕 `trend objective` 做更高决策价值的推进，而不是再回去盲扫旧的 loss scalar。

## 当前最重要的 run

| role | run | conclusion |
| --- | --- | --- |
| old-good reference | `2026-04-13 174639` | 可作为“自然感更好”的正式参考 |
| fair baseline | `2026-04-16 103752` | 当前最公平、最稳的 maintained baseline |
| no-go | `2026-04-16 152722` | `W_STEER_RATE=1.25` 补 head，但伤 tail/peak/reversal |
| proceed | `2026-04-16 163449` | pooled-level `W_TREND=0.10` 是目前最接近主目标的方向 |
| no-go | `2026-04-16 200855` | direction-aware coarse-delta sign 只涨一点，但整体更差 |

## 关键指标对照

| run | `rmse_steer` | `head_amp_ratio` | `onset_delay_mae` | `tail_flatness` | `late_peak_recall` | `trend_corr_mean` | `sign_match_rate` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2026-04-13 old-good` | `0.6557` | `1.6826` | `0.1378` | `0.4356` | `0.4359` | `0.6325` | `0.5530` |
| `2026-04-16 baseline 103752` | `0.6328` | `1.5856` | `0.1708` | `0.3883` | `0.5855` | `0.6517` | `0.5579` |
| `2026-04-16 W_STEER_RATE=1.25` | `0.6513` | `1.6871` | `0.1450` | `0.4659` | `0.4701` | `n/a` | `n/a` |
| `2026-04-16 W_TREND=0.10 pooled` | `0.6224` | `1.5936` | `0.1350` | `0.4223` | `0.5769` | `0.6520` | `0.5550` |
| `2026-04-16 trend delta+direction` | `0.6463` | `1.8824` | `0.1453` | `0.4034` | `0.5556` | `0.6322` | `0.5614` |

## 当前收口结论

- repaired baseline `103752` 已经推翻了“当前 maintained 主线整体更晚、更弱、更平”的旧结论
- `W_STEER_RATE=1.25` 说明“补 head”不是没戏，但这条 scalar 太粗，副作用太大
- `W_TREND=0.10` 的 pooled-level 版本，是当前最值得继续投资源的方向
- 但它还没有真正解决“coarse segment 方向一致性”
- 直接改成 `delta+direction` 主导，并没有比 pooled-level 更好

## 当前最值得 GPT Pro 回答的问题

1. 既然 pooled-level `W_TREND` 已证明方向成立，下一刀该如何升级，而不把 head/tail balance 搞坏？
2. 是应该做“pooled-level 主项 + very small residual delta/direction 项”，还是该上更结构化的 trajectory/phase objective？
3. 有没有比继续调 loss 更值的架构切片，能直接命中“整条趋势像 GT、但局部结构不塌”的问题？

## 明确不建议 GPT Pro 重复投入的方向

- 继续把 `W_STEER_REV` / `W_REVSEQ` 当成当前主问题的一阶突破口
- 把 `W_STEER_RATE` 往更大方向推
- 直接把 `direction-aware coarse-delta` 留成新默认
- 回到 smoke/random split 上做正式判断

