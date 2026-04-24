# 给 GPT Pro 的直接代码生成提示词

你现在拿到的是一个已经推进到 `2026-04-23` 的驾驶员转向响应预测项目 codegen pack。
这次不是让你只做高层分析，而是希望你基于包内**当前源码**直接提出并尽量产出**最小可落地代码实现**。

## 你的角色

请把自己当成一个愿意直接把下一步结构切片落到代码上的高级研究搭档，而不是只给方向建议的 reviewer。
你可以先给简短判断，但最终目标必须回到：

- 改哪些文件
- 加哪些配置
- 改哪些 forward / loss / train / eval 接线
- 如何形成最小可运行版本

## 你必须先接受的当前事实

1. 当前正式 fit / tail keeper 仍然是：
   - `baseline_fixed_input` full `best_by_structured`
2. Run A 仍然只是 response-structure anchor。
3. 旧 `H15` 的收益是真收益，不是 tail fraction 假象：
   - `rmse_steer: 0.5559 -> 0.4930`
   - `abs_tail_last_0p5s.rmse_steer: 0.7171 -> 0.6022`
4. 旧 `H15` 不能 promote 的根因是它把 `strong_pos` late tail 压塌了：
   - `strong_pos.tail_amp_ratio_pred_over_gt = 0.2687`
   - `strong_pos.tail_flatness_rate = 1.0000`
5. 新 `H15_AC_CF_HLF_v1` 已经完整跑完，并且仍然失败：
   - `best_by_loss`
     - `rmse_steer=0.5063`
     - `abs_tail_last_0p5s.rmse_steer=0.6323`
     - `late_peak_recall=0.5786`
     - `strong_pos.tail_amp_ratio_pred_over_gt=0.5141`
     - `strong_pos.tail_flatness_rate=0.3750`
   - `best_by_structured`
     - `rmse_steer=0.5231`
     - `abs_tail_last_0p5s.rmse_steer=0.6472`
     - `late_peak_recall=0.5251`
     - `strong_pos.tail_amp_ratio_pred_over_gt=0.3304`
     - `strong_pos.tail_flatness_rate=0.7500`
6. 对 `best_by_loss` 又做了 `8` 个 `strong_pos` representative plots 的人工复核，结果仍然支持 no-go，而不是 success。
7. 因此，下一步不该再烧预算在：
   - optimizer sweep
   - width sweep
   - broad bridge / gate / loss matrix
8. 当前最合理的下一刀是：
   - `H15_LATE_RESIDUAL_HEAD_v1`

## 你应该优先阅读的文件

1. `PROJECT_STATUS_AND_CODEGEN_BRIEF_CN.md`
2. `context/current-state.md`
3. `context/daily_2026-04-23.md`
4. `context/decision_log.md`
5. `context/experiment_registry.md`
6. `evidence/gptpro_effectiveness_review_20260423.md`
7. `evidence/h15_ac_cf_hlf_v1_summary.md`
8. `evidence/baseline_fixed_input_recalc_best_by_structured_summary.json`
9. `evidence/h15_recalc_best_by_structured_summary.json`
10. `evidence/h15_ac_cf_hlf_v1_recalc_best_by_loss_summary.json`
11. `evidence/h15_ac_cf_hlf_v1_recalc_best_by_structured_summary.json`
12. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
13. `code/v58_modular/README.md`
14. `code/v58_modular/config.py`
15. `code/v58_modular/modeling.py`
16. `code/v58_modular/losses.py`
17. `code/v58_modular/train.py`
18. `code/v58_modular/evaluation.py`
19. `code/recalc_v58_checkpoint_with_current_metrics.py`

## 这次希望你直接完成的任务

请围绕 `H15_LATE_RESIDUAL_HEAD_v1`，给出**最小可运行、最小可验证**的代码实现方案。

核心目标不是重写系统，而是：

- 保留当前 `1.5s` 线的真实 fit / tail 收益
- 针对 `t >= 1.0s` 的 late slice，增加一个更直接的 residual capacity
- 尽量减少对 `0 ~ 1.0s` 这段已经相对健康区域的副作用

## 工程约束

- 不要改 split
- 不要改 `fixed_v20260421`
- 不要改 stable GPU path
- 不要做 scene-specific 模型
- 不要把 bridge / gate / loss 重新拉回主线
- 不要把 substantial 新逻辑堆回 wrapper
- 优先只改：
  - `code/v58_modular/config.py`
  - `code/v58_modular/modeling.py`
  - `code/v58_modular/losses.py`
  - `code/v58_modular/train.py`
  - `code/v58_modular/evaluation.py`
- wrapper `future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py` 只允许做必要薄接线，不要重新做成第二个单体脚本

## 我希望你的输出按下面顺序来

### 1. 先给出 3 到 6 条最尖锐诊断

请直接回答：

- 为什么 `H15_AC_CF_HLF_v1` 说明 “objective anti-collapse” 还不够
- 为什么问题更像 late-slice representation / routing，而不是再调 optimizer
- 为什么 minimal late residual head 比继续扫 scalar 更高 EV

### 2. 给出你建议的最小结构版本

请明确说明：

- late residual head 应该接在现有 forward 的哪里
- 它负责预测什么
- 它如何只作用于 `t >= 1.0s`
- 它如何与现有 coarse-fine 主输出组合
- 为什么这个版本是最小可验证版本

### 3. 直接给文件级修改方案

至少逐文件回答：

- `config.py`
  - 新增哪些 env flags / defaults
- `modeling.py`
  - 新增哪些 module / output fields / forward logic
- `losses.py`
  - 是否需要对 late residual head 增加专属损失
  - 如果需要，公式或伪代码是什么
- `train.py`
  - 训练时如何接线、如何记录 config、如何保证旧路径兼容
- `evaluation.py`
  - 是否需要导出额外 late-slice 诊断，帮助我们判断该 head 是否真在修 `strong_pos`

### 4. 尽量直接产出代码

如果你的工作流允许，请直接给出 patch 级实现思路，或者按文件贴出关键代码段，而不是只写概念说明。
如果你认为某处需要新增开关，建议命名要清楚，例如类似：

- `DRIVER_MODEL_LATE_RESIDUAL_HEAD=1`
- `DRIVER_MODEL_LATE_RESIDUAL_START_SEC=1.0`
- `DRIVER_MODEL_W_LATE_RESIDUAL=...`

这只是命名风格示例，不是硬性规定。

### 5. 给出第一个 full run 的最小建议配置

请直接给出一个 `H15_LATE_RESIDUAL_HEAD_v1` 的 env block。
要求：

- 基于当前 `H15_AC_CF_HLF_v1` 闭环继续
- 不要顺手再带 optimizer / width sweep
- 不要开第二套复杂 matrix
- 让我们能够直接据此做**唯一的一次** full run

### 6. 说明最大风险

请明确指出：

- 这个 late residual head 可能如何失败
- 它最可能把哪些指标拉坏
- 如果失败，最可能的失败形态是什么

## 红线

- 不要建议重新扫 optimizer
- 不要建议重新扫 width
- 不要建议重开 bridge / gate / loss matrix
- 不要建议回到 random split
- 不要只盯 overall RMSE
- 不要把 `H10` 直接升为默认主线
- 不要输出需要大规模重写系统的“理想化大方案”

## 我的真实需求

我现在要的不是“再 brainstorm 三轮”，而是**一个高决策价值、代码级最小落地的下一步**。
如果你判断 `H15_LATE_RESIDUAL_HEAD_v1` 是对的，请尽量把实现回答到可以直接开改的程度。
