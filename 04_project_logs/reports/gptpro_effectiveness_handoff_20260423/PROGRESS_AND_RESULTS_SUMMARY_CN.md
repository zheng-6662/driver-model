# 当前进展与本轮结果总结

## 1. 一句话结论

到 `2026-04-22` 为止，这一轮 effectiveness 主线已经正式闭环。
当前没有产生新的正式冠军模型，项目仍以 `baseline_fixed_input` full `best_by_structured` 作为当前最优非塌陷 `2.0s` base。
Run A 继续作为 response-structure 锚点，`H10` 只保留为 horizon ceiling 证据，不升为主线。

## 2. 目前项目走到哪里了

### 2.1 先前已经锁定的基础状态

- active training script:
  - `future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
- active input version:
  - `fixed_v20260421`
- locked guardrails:
  - protocol-safe split
  - stable GPU path
  - pooled training
  - no scene-specific models
  - anchor policy unchanged

### 2.2 2026-04-21 已完成的关键进展

- lane / speed 读取问题已修复。
- D input ablation 已经闭环。
- `baseline_fixed_input` 成为新的 fit / tail numeric anchor。
- Run A 保留为 structure anchor。
- `bridge_50_50`、`bridge_schedule_B_to_A`、`plus_pedals` 只作为历史 frontier evidence，不再是当前 keeper。

### 2.3 2026-04-22 这一轮真正做了什么

这一轮不是继续追 bridge，而是切换到新的 effectiveness 问题：

1. 先用 D0 重新定义跨 horizon 的公平比较口径。
2. 检查 `1.5s` 是否能成为可推广 base。
3. 检查 `2.0s` 是否只是优化器没调好。
4. 在当前 best base 上验证轻量正则和轻量容量是否足够救主线。

## 3. 当前正式认定的 keeper

| 角色 | 当前版本 | 说明 |
| --- | --- | --- |
| fit / tail keeper | `baseline_fixed_input` full `best_by_structured` | 当前正式最优非塌陷主 base |
| structure anchor | Run A full `best_by_structured` | 仍然只作为结构参考锚点 |
| diagnostic ceiling | `H10` full `best_by_structured` | 只证明 `1.0s` 更容易，不是主线 |

## 4. D0 先回答了什么

D0 的核心作用不是训练，而是先把“怎么比较 tail”这件事修正清楚。

### 4.1 D0 锁定的 baseline 绝对时间窗指标

- `prefix_1p0s.rmse_steer = 0.4291`
- `prefix_1p5s.rmse_steer = 0.4906`
- `full_horizon.rmse_steer = 0.5559`
- `abs_tail_last_0p5s.rmse_steer = 0.7171`

### 4.2 D0 最重要的解释

旧的 `tail_rmse_steer` 是按 horizon 最后 `25%` 切的。
这会导致：

- `2.0s` tail 实际看的是最后 `100` 步
- `1.5s` tail 实际看的是最后 `75` 步
- `1.0s` tail 实际看的是最后 `50` 步

所以短 horizon 会天然占便宜。
从这一轮开始，跨 horizon 的 tail 对比必须优先看：

- `abs_tail_last_0p5s.rmse_steer`

## 5. 本轮 full runs 结果，用通俗话总结

| Run | 改了什么 | 好消息 | 坏消息 | 最终判定 |
| --- | --- | --- | --- | --- |
| `H15` | `2.0s -> 1.5s`，其他基本不动 | 整体 RMSE 和绝对 tail 都明显变好 | `strong_pos` 发生硬塌陷 | 不可晋级，不跑 `OPT_A_H15 / OPT_B_H15` |
| `OPT_A_20` | `2.0s` 上只换优化 bundle | 第一秒略有改善 | overall、绝对 tail、late peak 都变差 | 不能证明 `2.0s` 只是优化没调好 |
| `H10` | `2.0s -> 1.0s` | 明显证明短 horizon 更容易 | 只能做 ceiling 证据，且 strong-pos 样本很少 | 不升主线 |
| `CAP_192_BEST` | 在当前 winner 上做 width-only bump | 无明显核心收益 | fit/tail 变差 | 不推广 |
| `OPT_C_BEST` | 当前 winner 上加更强 weight decay | guardrail 稍稳一点 | 还是没赢 baseline anchor | 不推广 |
| `WINNER_CONFIRM` | 原 winner 完整重复一遍 | 证明 baseline 路线稳定存在 | 没有超过原始 keeper | 只做确认，不换 keeper |

## 6. 关键数字，不看术语时应该怎么理解

### 6.1 当前正式 keeper：`baseline_fixed_input`

- `rmse_steer = 0.5559`
- `abs_tail_last_0p5s.rmse_steer = 0.7171`
- `late_peak_recall = 0.6496`
- `strong_pos.tail_amp_ratio_pred_over_gt = 1.3490`
- `strong_pos.tail_flatness_rate = 0.2105`

可以把它理解成：

- 不是最激进、也不是最漂亮的一版
- 但它是当前最稳、没跨过 hard collapse 红线的一版
- 所以它继续留任

### 6.2 `H15` 为什么“看起来很强但不能升”

- `rmse_steer = 0.4930`
- `abs_tail_last_0p5s.rmse_steer = 0.6022`
- `late_peak_recall = 0.6355`

这些数字都说明 `1.5s` 确实有潜力。
但同时：

- `strong_pos.tail_amp_ratio_pred_over_gt = 0.2687`
- `strong_pos.tail_flatness_rate = 1.0000`

这表示强正样本的尾段幅值被压得太低，而且形状几乎被压平。
也就是说，模型为了拿到更好看的整体 fit，牺牲了最危险、最关键那部分响应形态。
所以这条线现在属于“有真实信号，但还不能直接上位”。

### 6.3 `OPT_A_20` 为什么证明不了 `2.0s` 可被简单救回

- `rmse_steer = 0.5887`
- `abs_tail_last_0p5s.rmse_steer = 0.7698`
- `late_peak_recall = 0.5726`

翻成白话就是：

- 换了更“像样”的优化器、scheduler、warmup、clip 以后
- `2.0s` 主任务并没有被救活
- 问题不像是“只是训练技巧差一点”
- 更像是当前目标定义或结构能力本身不够

### 6.4 `H10` 给出的启发是什么

- `rmse_steer = 0.4370`
- `abs_tail_last_0p5s.rmse_steer = 0.5272`
- `late_peak_recall = 0.7880`

这说明：

- 如果 horizon 只看 `1.0s`，任务明显容易很多
- 当前模型并不是完全不会学驾驶员响应
- 真正困难的是：当 horizon 拉到 `1.5s` 或 `2.0s` 时，怎样既保住整体拟合，又不把强正样本尾段做塌

## 7. 这一轮已经真正排除掉了什么

以下方向已经有比较明确的负结论，不适合让 GPT Pro 再原地重复：

- 不要把 bridge / gate / loss 重新拉回主线
- 不要把 `H10` 当默认下一版主模型
- 不要继续靠 plain optimizer sweep 期待 `2.0s` 自动变好
- 不要默认认为 mild width bump 能解决当前问题
- 不要继续重复 `OPT_A_H15 / OPT_B_H15`
- 不要改 split、input pipeline、anchor policy、stable GPU path

## 8. 这一轮留下的真正悬念是什么

现在最值得判断的其实只剩两条：

### 路线 A：继续押 `1.5s`，但专门做 anti-collapse

核心逻辑：

- `H15` 的整体指标是真有进步的
- 问题不是“没信号”，而是 `strong_pos` 尾段塌了
- 所以下一步可以考虑更明确地保尾段、保幅值、保 strong-pos 的目标设计

这条路线的优点：

- 建立在已经出现的真实增益上
- 不需要完全推翻当前管线

这条路线的风险：

- 如果只是继续小修 loss，可能仍然只是在局部补洞
- 可能修住了 collapse，却把 `H15` 原本的整体收益又吃回去

### 路线 B：承认当前 `2.0s -> 1.5s/2.0s` gap 更像结构问题

核心逻辑：

- `H10` ceiling 很高
- `H15` 有收益但一到 harder regime 就塌
- `OPT_A_20`、`OPT_C_BEST`、`CAP_192_BEST` 都没能从现有结构上把问题补回来

这说明：

- 与其继续做温和 sweep
- 不如让 GPT Pro 判断是否应该直接提出一个更强的新 architecture slice

这条路线的优点：

- 更符合目前证据链
- 可能比继续补 loss 更高杠杆

这条路线的风险：

- 改动更大
- 必须给出最小可验证版本，不能空谈

## 9. 我希望 GPT Pro 重点回答什么

### 9.1 如果只允许再烧 `1` 到 `2` 个 full GPU run

最应该投到哪里：

- `H15 anti-collapse`
- 还是一个最小的新架构切片

### 9.2 如果继续做 `H15`，应该具体怎么防塌

请它不要只说“加一些 regularization”。
而是要明确回答：

- 是改 loss
- 改 target
- 改 decoder position / future token handling
- 还是加一个专门保护 strong-pos tail 的 objective / branch

### 9.3 如果不继续补 `H15`

那最值得切入的新结构应该是什么，为什么它比继续扫优化器和轻量容量更值。

## 10. 红线和 guardrails

无论 GPT Pro 给什么建议，都必须满足：

- 保持 protocol-safe split
- 保持 pooled training
- 不做 scene-specific 模型
- 不回退 stable GPU path
- 不改 `fixed_v20260421` 输入版本
- 不把 bridge/gate/loss 重新作为主线
- 不把 `H10` 直接升为默认主线

## 11. 给 GPT Pro 的简短判断框架

如果它的建议本质上只是：

- 再扫几个 optimizer
- 再加一点 width/depth
- 再跑几个弱正则组合

那大概率不够好。

更有价值的回答应该是：

1. 明确指出当前真正卡点是哪里。
2. 说明为什么 `H15` 有真实信号但会塌。
3. 只给少量、但高杠杆的下一步方案。
4. 明确到代码级，告诉我们最小可执行改法。
