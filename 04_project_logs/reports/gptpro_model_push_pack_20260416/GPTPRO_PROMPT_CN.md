# GPT Pro Prompt

你现在拿到的是一个已经做过多轮公平验证的驾驶员短时反应预测项目 handoff 包。请不要从泛泛的“可以试试更多损失/更多层数/更多数据增强”开始，而要直接基于证据给出高决策价值的下一步推进方案。

## 你的角色

请把自己当成一个愿意快速推进模型的高级研究搭档，而不是保守 reviewer。允许你提出激进方案，但你必须把它约束在当前证据和公平口径之内。

## 项目目标

目标不是单纯降低总体 RMSE，而是让当前 maintained 主线在 `protocol-safe`、`subject-level fixed split` 下，更可信地预测极限工况下驾驶员未来 `2s` 的真实反应轨迹，尤其希望：

- 整条 `2s` steer 趋势更像 GT
- coarse segment 的方向一致性更高
- 不要为了补某一项而明显破坏 head / tail / peak / reversal 的整体平衡

## 当前必须接受的前提

1. 当前公平 baseline 是 `2026-04-16 103752`
2. `2026-04-15 smoke` 不能再当 maintained 主线退化证据
3. `teacher-state` 不是当前第一嫌疑项
4. `W_STEER_REV` / `W_REVSEQ` 不是当前最有决策价值的主突破口
5. `W_STEER_RATE=1.25` 不是合格升级，因为它补了 head，但明显伤了 tail/peak/reversal
6. `W_TREND=0.10` 的 pooled-level 版本是当前最接近主目标的正方向
7. 直接切到 `direction-aware coarse-delta` 主导，不是更好的默认主线

## 你要先读哪些文件

请按这个顺序读：

1. `MODEL_STATE_SUMMARY.md`
2. `context/daily_2026-04-16.md`
3. `context/experiment_registry.md`
4. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
5. `protocol/protocol_config.json`
6. `evidence/20260416_repaired_baseline_summary.json`
7. `evidence/20260416_wtrend010_summary.json`
8. `evidence/20260416_wtrenddirdelta010_summary.json`
9. 必要时再看对应 `cases.csv` 与 `run_config.json`

## 我希望你输出什么

请直接输出以下内容，按顺序来，不要只给高层空话：

### 1. 你对当前主问题的最尖锐诊断

- 用 3 到 6 条结论说清楚：
  - 当前模型真正卡在哪里
  - 为什么 pooled-level trend loss 有效但还不够
  - 为什么 direction-aware delta 版本失败
  - 你认为“下一刀”应该打在 loss、target、decoder、conditioning，还是更结构化的轨迹目标上

### 2. 你推荐的 3 个下一步实验

对每个实验都必须给出：

- 实验名称
- 核心假设
- 为什么比继续扫旧 scalar 更值
- 需要改哪些代码位置
- 具体改动是什么
- 预期会拉动哪些指标
- 最大风险是什么
- 应该先 smoke 还是直接 full-regime

### 3. 你最推荐的一条主线

请只选一条作为主推方案，并解释：

- 为什么它的 expected value 最高
- 为什么它比另外两条更适合现在直接推进
- 如果只允许我再烧 1 到 2 轮 full GPU，你会怎么排顺序

### 4. 请给出“直接可执行”的代码级建议

不要只写概念，请尽量具体到：

- 建议新增/修改哪些函数
- 大致在训练脚本的哪一类位置改
- loss / target / objective 的公式或伪代码
- 如果你建议的是结构改动，也请说清楚最小落地版本怎么做

### 5. 如果你认为应该更激进，也请直说

如果你认为现在继续修 loss 已经不够，应该直接切一个更强的新建模切片，也可以明确提出，但要满足：

- 仍然保持 protocol-safe 公平口径
- 说明为什么这个架构跳转比继续小修小补更值
- 给出最小可验证版本，而不是抽象口号

## 重要约束

- 不要建议回到 random split / smoke 结果去下正式结论
- 不要建议只盯总体 RMSE
- 不要建议删除现有文件或推翻当前 protocol-safe 闭环
- 可以大胆，但请基于当前证据链大胆，而不是脱离证据乱开枝

## 我的真实需求

我现在的目标是：**尽快解决当前真正存在的问题，快速推进模型，不需要太保守。**

所以如果你判断某条路虽然更激进，但比继续温和扫参更值得，请明确说出来，并给出最小可执行版本。

