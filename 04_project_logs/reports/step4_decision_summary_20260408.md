# Step 4 决策摘要：基于归因证据链的最小 loss 修改方案
**日期**：2026-04-08  
**执行主体**：Claude  
**依赖输入**：Task 1-3 的归因报告、approximate 分析 CSV、重建预测序列

---

## 一、证据链总结（三轮只读分析收口）

### 1.1 Q1_fast tail 退化

| 假设 | 证据 | 结论 |
|---|---|---|
| anchor context 信号系统性偏高 → 导致 tail 幅值失配 | Task 1：Q1_fast 仅 steer_rate 略高 +0.13，其余三维均低；与 delta_rmse_tail 相关性 max \|r\|=0.28 | **不支持**"uniform 四维偏高"解释，但不排除 steer_rate 局部贡献 |
| boundary/peak timing 恶化是主因 | Task 1/2：delta_boundary_shift 在 Q1_fast 仅 +0.032（非 Q1_fast 是 +0.084）；event-level 时间对齐 conditioned 仍改善 | **否定**，boundary/timing 不是主因 |
| tail shape/amplitude 失配是主因 | Task 2：Q1_fast×single_lobe 的 tail delta=+0.056，amplitude 相关 r=0.72，远强于 boundary shift r=-0.15；Task 1：shape_corr \|r\|=0.62，peak_abs_amp_err \|r\|=0.60 | **支持**，tail amplitude/shape 失配是主因 |

**Q1_fast tail 退化机制结论**：context uniform broadcast 将 anchor 处的强 steer_rate 信号不衰减地扩散至全部 400 步，导致 tail 段幅值预测偏高/失真。问题核心是幅值（amplitude），而非时间对齐。

---

### 1.2 boundary_shift 恶化

| 假设 | 证据 | 结论 |
|---|---|---|
| 边界处斜率被平滑（flattening） | Task 2：single_lobe 的 delta_boundary_slope=+0.024 远弱于 delta_boundary_shift=+0.182；Q1_fast×single_lobe 的 delta_slope=-0.023（反而改善） | **否定**，不是斜率平滑主因 |
| 边界时间位移（shift forward/backward） | Task 2：single_lobe delta_boundary_shift=+0.182，reverse_correction=+0.107，时移是主要方向 | **支持**，以时间位移为主 |
| 由单一被试主导 | Task 1：三位被试 single_lobe 和 reverse_correction 均呈正向恶化，cwh 最重但 gf/tyy 也明显 | **否定**，morphology 主导，subject 影响幅度 |

**boundary_shift 恶化机制结论**：MSE loss 对 single_lobe 尖锐边界产生时间位移式 hedging 效应；conditioned context 在边界附近加剧不确定性但不改变斜率。peak timing / reversal probability loss 当前被禁用（W_PEAKTIME=0.0，W_REVSEQ=0.0），缺少对边界结构的显式约束。

---

## 二、Step 4 实验设计决策

### 2.1 候选修改方案对比

| 方案 | 针对问题 | 改动范围 | 可比性风险 | 当前证据支持度 |
|---|---|---|---|---|
| A：tail amplitude penalty（在 loss 中对 tail 段 abs(pred-true) 加权） | Q1_fast tail amplitude 失配 | 只加一个额外 loss 项，不改模型结构 | 低，不改 anchor/split/protocol | **最强**（r=0.72） |
| B：启用 W_PEAKTIME（peak timing loss） | boundary 时间位移 | 改 loss 权重，启用已有 loss 计算路径 | 低，代码路径已有只是禁用 | 中，间接支持 |
| C：context time-decay（broadcast 改为距离衰减） | uniform broadcast 根因 | 改模型结构（ConditionedTrajectoryHead） | 中，需重新比较公平性 | 中（机制合理但未直接量化） |
| D：A + B 同时启用 | 两类问题 | 两个 loss 权重 | 低，但难以区分各自贡献 | **不推荐**，violates single-change principle |

### 2.2 决策

**选择方案 A：tail amplitude penalty，作为 Step 4 唯一修改变量。**

理由：
1. amplitude 相关性是三轮分析中最强的单一信号（r=0.72），而非间接推断；
2. 只在 loss 层添加权重，不改模型结构、不改 context 构建、不改任何 protocol/split；
3. 与当前 formal run 的可比性最高，结果能直接对打 conditioned v2 原始版本；
4. 如果方案 A 有效，再考虑叠加方案 B；如果无效，再考虑方案 C（更根本的 context 结构改动）。

---

## 三、最小实验设计

### 3.1 修改内容

**只改训练脚本的 loss 计算部分**，新增 tail amplitude penalty：

```python
# 在 loss 函数中，对 tail 段（t >= 200）额外惩罚 amplitude 误差
TAIL_START = 200          # step index（对应 ~1s 后）
W_TAIL_AMP = 0.3          # 待调超参，建议先试 0.2 / 0.3 / 0.5

tail_mask = mask[..., TAIL_START:]                         # (B, T_tail)
tail_pred = pred[..., TAIL_START:, 0]                      # steer channel
tail_true = true[..., TAIL_START:, 0]

tail_amp_loss = (tail_pred.abs() - tail_true.abs()).abs()  # amplitude 误差
tail_amp_loss = (tail_amp_loss * tail_mask).sum() / (tail_mask.sum() + 1e-8)

total_loss = existing_loss + W_TAIL_AMP * tail_amp_loss
```

> 注意：上述为概念设计，实际 channel index 和 mask 形状需对照 `run_event_conditioned_trajectory_baseline.py` 确认。

### 3.2 实验配置

| 项目 | 值 |
|---|---|
| 基准 | 重建的 conditioned v2（Task 3 产出，test RMSE 0.4973） |
| 对比 | 方案 A 修改版（相同 checkpoint 初始化，相同 split/seed） |
| Epoch | 3（与 Task 3 conditioned v2 重建保持一致） |
| 评估指标 | test rmse_tail_abs_steer、delta_rmse_tail_Q1_fast、delta_boundary_shift_single_lobe |
| 预测序列 | 必须保存（与 Task 3 格式一致） |

### 3.3 Go/No-Go 标准

**Go（继续方向 A）**：
- Q1_fast 的 delta_rmse_tail_abs_steer 均值 ≤ 0（从 +0.0155 转为不恶化）
- overall test rmse_2s 不显著劣化（容忍 ≤ +0.01 vs 基准）

**No-Go（放弃方向 A，转向方案 C）**：
- Q1_fast tail 无改善，且 overall RMSE 反而升高
- 或者方案 A 只改善了 overall tail 而 Q1_fast 切片无变化（说明 amplitude loss 不是精确命中）

---

## 四、当前明确不做的事

- 不同时修改 W_PEAKTIME 或 context 构建（确保单变量对照）
- 不改 protocol_config、split、anchor、horizon
- 不在还未收口 per-timestep 精确曲线的情况下直接上更复杂结构改动
- 不让 multi-hypothesis 接管主线

---

## 五、下一个可选步骤（Step 4 之前可并行）

精确 per-timestep 误差曲线（低风险，只读）：
- 使用 Task 3 产出的 NPZ 文件，对 Q1_fast/non-Q1_fast 分组画逐时间步 MAE 曲线
- 目的：确认退化在 step≥200 集中，验证 tail_start=200 的切分是否合适
- 产出：一张图，不改任何代码
- 这一步可以先于 Step 4 执行，也可以与 Step 4 并行

---

## 六、参考文件

| 文件 | 用途 |
|---|---|
| `reports/attribution_master_table.csv` | 749 条 sample-level 归因宽表 |
| `reports/conditioned_v2_fast_boundary_attribution_20260408.md` | Task 1+2 归因报告 |
| `reports/approx_timestep_boundary_analysis_20260408.md` | Task 2 近似 boundary 分析 |
| `reports/context_value_range_by_latency_bucket_20260408.md` | Task 1 context 值域报告 |
| `reports/conditioned_v2_prediction_sequences.npz` | Task 3 预测序列（749×400×2） |
| `reports/baseline_prediction_sequences.npz` | Task 3 baseline 序列 |
| `datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py` | 训练脚本（已从 git 恢复） |
