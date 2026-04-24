# H15_LATE_RESIDUAL_HEAD_v1 实施说明

## 这次改动的判断

`H15_AC_CF_HLF_v1` 已经证明：

- 纯 objective anti-collapse 有效，但修复力度仍然不足；
- 问题更像 `t >= 1.0s` 的 late-slice 表达 / 路由不够，而不是再调 optimizer；
- 因此更高 EV 的最小下一刀，是在现有 coarse-fine 主干上增加一个只负责 late slice 的 steer residual head。

## 这次实际落下去的代码

已完成的文件级修改：

- `code/v58_modular/config.py`
  - 新增：
    - `DRIVER_MODEL_LATE_RESIDUAL_HEAD`
    - `DRIVER_MODEL_LATE_RESIDUAL_START_SEC`
    - `DRIVER_MODEL_W_LATE_RESIDUAL`
- `code/v58_modular/modeling.py`
  - 新增最小 late residual steer head
  - 输出仅在 `t >= LATE_RESIDUAL_START_SEC` 生效
  - 不改前段 `0 ~ start_sec`
  - 与现有主输出按加法组合
- `code/v58_modular/losses.py`
  - 新增 late residual 专属损失
  - 目标是拟合 `GT - base_path_detached`
  - 默认只在 hard late sample / late slice 上生效
- `code/v58_modular/train.py`
  - 新增 head 接线、日志、history 字段、run_config / checkpoint config 导出
- `code/v58_modular/evaluation.py`
  - 新增 `test_late_residual_metrics.json`
  - 在 `test_state_dump.csv` 中追加 late residual 诊断列
- `code/recalc_v58_checkpoint_with_current_metrics.py`
  - 新增 late residual env / instantiate 兼容

## 设计细节

### 1. head 接在哪里

接在 decoder 输出 `out` 上，作为一个最小的单独投影头：

- `decoder out -> late_residual_proj -> (B, T, 1)`

### 2. 它预测什么

它预测的是 steer 的**附加 late residual**，不是整条 steer 主轨迹。

### 3. 如何只作用于 `t >= 1.0s`

通过一个二值 late mask：

- `t < start_sec` 时输出强制为 `0`
- `t >= start_sec` 时 residual 允许生效

### 4. 如何与现有 coarse-fine 主输出组合

- 现有 base steer：`coarse_up + fine`
- 新的最终 steer：`base_steer + late_residual`

如果 coarse-fine 未开启，代码也能兼容，但本分支的推荐使用方式仍然是：

- `STEER_COARSE_FINE=1`

### 5. 为什么 loss 这样接

late residual 的专属目标是：

- `late_target = GT_steer - base_steer_detached`

这样做的目的不是把 base path 再一起拉着乱跑，而是让新增 head 专门补 base path 在 late tail 上留下的缺口。

## 推荐的第一版 full run env

见：

- `configs/h15_late_residual_head_v1_env_block.txt`

建议直接在 `H15_AC_CF_HLF_v1` 的基础上增加 late residual head，不顺手再带 optimizer / width sweep。

## 我做过的验证

这次没有执行真实训练或真实数据集重算，但已做下面这些可落地验证：

- `py_compile` 通过
- baseline / late-residual 两种模型随机前向通过
- `compute_total_task_loss()` 新返回路径通过
- recalc 的 instantiate 路径兼容旧配置与新配置

## 需要你注意的最大风险

这个 head 最可能的失败形态不是“完全没作用”，而是：

- late tail 被补过头，导致 tail RMSE 或 peak timing 变差；
- head 学成“普遍加幅值”，对 non-strong 样本带来不必要扰动；
- strong_pos 的 amplitude 上来了一些，但 `late_peak_recall` 没同步修住。

所以第一轮 full run 最值得盯的不是 only overall RMSE，而是：

- `strong_pos.tail_amp_ratio_pred_over_gt`
- `strong_pos.tail_flatness_rate`
- `abs_tail_last_0p5s.rmse_steer`
- `late_peak_recall`
- 新导出的 `test_late_residual_metrics.json`
