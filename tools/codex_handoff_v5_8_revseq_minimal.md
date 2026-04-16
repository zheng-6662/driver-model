# Codex Handoff: maintained v5.8 第二阶段最小 loss 微调

## 任务目标

请在 **不改数据划分、事件锚点、future horizon、标签定义、teacher-state 模式** 的前提下，对 maintained 主训练脚本做一轮 **问题再诊断 + 最小 loss 微调建议 + 单轮运行验证**。当前问题已经不应再只描述为 tail 问题，而应收口为：

- early-response onset too flat
- whole-trajectory temporal dynamics over-smoothed
- tail amplitude collapse
- late-peak miss
- strong-reversal tail shrinkage

也就是说，这一版模型不是只在后段发平，而是 **前段启动不足、整体时间结构偏保守、后段继续收缩**。目标不是泛泛“提分”，而是：

1. 先判断当前最小改动是否仍应只动 `W_REVSEQ`
2. 明确前段过平是否需要进入固定评估闭环
3. 在保持 protocol-safe 前提下，优先改善时间结构，而不是只盯总体 `rmse_steer`

如果你认为只做 `W_REVSEQ` 微调不足以覆盖当前问题，请先给出思考结论，再决定是否需要 very small 的 onset/head 约束或评估补充。

---

## 当前基线证据

参考这轮 smoke 运行目录：

- `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726`

关键诊断（已由新增评估闭环稳定导出）：

- `tail_flatness_rate = 0.961`
- `tail_amp_ratio_pred_over_gt = 0.156`
- `late_peak_recall = 0.145`
- `strong_pos.tail_amp_ratio_pred_over_gt = 0.065`

这说明当前主问题不是“完全不会动”，而是：尾段仍严重收缩、后半段峰值保不住、strong reversal hard case 尾段几乎被压扁。

另外，基于最新人工看图反馈（同一轮 `2026-04-15 11:57` smoke 图），需要补充一个此前指标尚未单独覆盖的问题：

- 前段也明显过平，预测启动偏弱，早期响应幅值不足

因此 Codex 需要把这轮问题视为 **head + tail 两端都被压缩，且整体时间动态偏保守**，而不是只把它当作单纯 tail flattening。

请特别思考：

- 现有 tail / peak / reversal 指标是否足以解释前段问题
- 是否应新增 head/onset 评估指标（如 `head_amp_ratio_pred_over_gt`、`head_flatness_rate`、`response_onset_delay`、`early_slope_ratio_pred_over_gt`）
- 在训练最小改动上，应先只做 `W_REVSEQ` 归因，还是需要一个 very small 的 early-response 约束配套

---

## Codex 本轮思考重点

在真正修改前，请先明确回答以下问题：

1. 当前最小训练改动是否仍应优先只动 `W_REVSEQ`，还是这会过度聚焦后段问题？
2. “前段也过平”是否说明固定评估闭环里缺少 head/onset 指标？
3. 如果要继续坚持最小干预，最合理的顺序是：
   - 先补评估，不动训练
   - 先只动 `W_REVSEQ`
   - 先动 `W_REVSEQ`，再 very small 地补一个 onset/head 约束
4. 你推荐的最小方案为什么最利于归因，而不是只看起来更全面？

请把“可解释、可归因、尽量少改 active source”作为第一原则。

## 推荐改动（按优先级）

### 训练侧默认最小方案：先只动 `W_REVSEQ`

在以下文件中修改：

- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`

当前脚本里：

- `W_REVSEQ = 0.0`
- `W_PEAKTIME = 0.05`

### 单轮推荐方案

只做一档最小改动：

- `W_REVSEQ: 0.00 -> 0.05`

### 本轮不要动

- 不要改模型结构
- 不要改输入特征
- 不要改 protocol / split / anchor / future length
- 不要改 teacher-state mode
- 不要同时引入多处大权重调整

### 可选但不优先

如果你判断只开 `W_REVSEQ` 过于保守，可以 very small 地附带：

- `W_PEAKTIME: 0.05 -> 0.08`

但默认仍建议 **先只动 `W_REVSEQ`**，这样更利于归因。

---

## 关键文件

### 需要读取 / 修改

- `F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`

### 需要读取的基线结果

- `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726/figures/test_metrics.json`
- `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726/figures/test_metrics_tail.json`
- `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726/figures/test_metrics_peak.json`
- `F:/data_set_process/data_process/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726/figures/test_metrics_reversal_structure.json`

### 日志协议参考

- `F:/data_set_process/data_process/reports/progress/ai_recording_protocol.md`
- `F:/data_set_process/data_process/reports/progress/daily/2026-04-15.md`
- `F:/data_set_process/data_process/reports/progress/experiment_registry.md`

**注意：不要编辑 tmp 运行目录里的脚本副本。只改 active source。**

---

## 建议运行命令

使用本机实际可用解释器路径，不要依赖 shell 侧 `conda run`：

```bash
CUDA_VISIBLE_DEVICES=0 DRIVER_MODEL_RESULT_ROOT="F:/data_set_process/data_process/tmp/protocol_safe_runs" DRIVER_MODEL_SMOKE=1 DRIVER_MODEL_SMOKE_MAX_SAMPLES=512 DRIVER_MODEL_SMOKE_EPOCHS=2 DRIVER_MODEL_SMOKE_BATCH_SIZE=64 "D:/ProgramData/anaconda3/envs/predict_2/python.exe" "F:/data_set_process/data_process/datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py"
```

如 smoke 指标方向正确，再决定是否补正式 run。

---

## 验收标准

优先看新增 JSON 字段，而不是只看总体 RMSE。

同时请注意：如果本轮你判断“前段过平”已经足够明确，而现有 JSON 无法量化它，那么一个合格输出不一定只能是训练结果，也可以是：

- 明确说明现有评估闭环对 head/onset 问题覆盖不足
- 给出最小新增评估指标方案
- 说明为什么当前应先补评估再做训练归因，或为什么相反

也就是说，本轮验收不只接受“跑出一个新 smoke”，也接受“先把问题定义和评估缺口收清楚”的高质量结论。

### head / onset（如你决定补评估）
建议优先看：

- `head_amp_ratio_pred_over_gt`
- `head_flatness_rate`
- `response_onset_delay`
- `early_slope_ratio_pred_over_gt`

不要求四个都实现，但至少要给出是否值得补、为什么、以及最小实现路径。

### tail
看：

- `test_metrics_tail.json`
  - `tail_amp_ratio_pred_over_gt` 要 **高于** `0.156`
  - `tail_flatness_rate` 要 **低于** `0.961`

### peak
看：

- `test_metrics_peak.json`
  - `late_peak_recall` 要 **高于** `0.145`

### reversal structure
看：

- `test_metrics_reversal_structure.json`
  - `strong_pos.tail_amp_ratio_pred_over_gt` 要 **高于** `0.065`
  - `strong_pos.tail_flatness_rate` 最好不要恶化

### overall regression
看：

- `test_metrics.json`
  - `rmse_steer` 不应明显恶化
  - 如果结构指标明显改善、RMSE 仅轻微波动，可接受

### tail
看：

- `test_metrics_tail.json`
  - `tail_amp_ratio_pred_over_gt` 要 **高于** `0.156`
  - `tail_flatness_rate` 要 **低于** `0.961`

### peak
看：

- `test_metrics_peak.json`
  - `late_peak_recall` 要 **高于** `0.145`

### reversal structure
看：

- `test_metrics_reversal_structure.json`
  - `strong_pos.tail_amp_ratio_pred_over_gt` 要 **高于** `0.065`
  - `strong_pos.tail_flatness_rate` 最好不要恶化

### overall regression
看：

- `test_metrics.json`
  - `rmse_steer` 不应明显恶化
  - 如果结构指标明显改善、RMSE 仅轻微波动，可接受

---

## 日志记录要求（必须遵守新协议）

### 不要默认写回

- `F:/data_set_process/data_process/reports/project_progress_master.md`

### 本轮必须先写

- `F:/data_set_process/data_process/reports/progress/daily/2026-04-15.md`

### 如果形成新 run 结论，再补

- `F:/data_set_process/data_process/reports/progress/experiment_registry.md`

### 记录最低要求

- 执行主体
- Why
- 专业结论
- 白话解释
- 做了什么
- 产物 / 链接
- 下一步

不要重复手抄 `run_summary.json` 或 JSON 里一长串自动可得指标；人工记录重点写判断与白话解释。

---

## 单轮最推荐方案总结

如果只做一轮，请优先：

- **只改 `W_REVSEQ: 0.00 -> 0.05`**
- 跑一轮 smoke
- 用新增的 tail / peak / reversal 结构指标做对照验收

这是一轮最符合“最小干预、直指当前 hard case、便于归因”的方案。
