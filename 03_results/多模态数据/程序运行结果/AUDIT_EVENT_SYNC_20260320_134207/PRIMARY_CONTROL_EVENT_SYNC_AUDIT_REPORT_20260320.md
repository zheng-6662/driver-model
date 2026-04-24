# Primary Control 事件-同步-标签联合审计报告

生成时间：2026-03-20  
审计范围：当前 frozen primary-control 主线的事件定义、anchor、future support、同步与符号一致性  
数据输出目录：`F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207`

## 0. 先给结论

当前主线里，**late steer 问题更像是“任务定义/anchor 对齐/future support 不足”的组合问题，而不是纯粹的时序同步问题**。  
更具体地说：

1. **当前 2s future 没有被充分监督。**
   - 全部 883 个 usable 样本里，完整覆盖 2.0s future 的只有 **105 个，占 11.9%**。
   - `1.0–2.0s` 的 point-level 有效监督比例只有 **30.4%**。
   - 当前事件最大时长本身被检测脚本限制在约 **3.0s**，而 anchor 平均落在事件的 **47.3%** 处，所以从结构上就很难留下完整 2s 的后段监督。

2. **当前 anchor 语义不一致，尤其是 curve 与 non-curve。**
   - `non-curve` 样本全部用“**steer_rate 首个达到 80% 峰值的点**”当 anchor。
   - `curve` 样本全部用“**roll 峰值点**”当 anchor。
   - 这导致 curve anchor 往往落在 **pre-response / near-risk-peak / late-adjustment**，而 non-curve anchor 主要落在 **response-onset / onset-to-risk-peak**。
   - 换句话说：**两类 scene 的 anchor 并没有在研究同一控制阶段。**

3. **事件 end 本身大体不算最糟的问题，但不是完全没问题。**
   - 相对“最后一次显著控制活动”来说，`event_end - control_end` 的中位数只有 **5 ms**，90 分位约 **275 ms**，说明大多数事件 end 与控制结束还是接近的。
   - 但 curve 中存在明显尾段案例，最极端样本 `event_end` 比控制活动结束还晚 **1.475 s**，这会把大量“车辆响应尾巴”一起包进事件。

4. **同步问题不是主因，但管线里确实有两个必须先修的硬错误。**
   - `steer -> yawrate` 的最佳相关时滞中位数约 **25 ms**，`steer -> ay` 约 **125 ms**；相关符号高度稳定，分别有 **94.7% / 93.0%** 的事件表现为固定负相关方向。  
     这说明**相对时序和左右符号总体是可解释的，不像存在系统性错位**。
   - 但同时存在两个非常关键的处理错误：
     - **speed 单位被重复除以 3.6**：当前训练代码把已经是 `m/s` 的 `zx|vx` 再除了一次 `3.6`。
     - **事件检测中的 `LTR_peak` 来源在 recording 之间不一致**：`55` 个 recording 的 `LTR_peak` 与 `ay` 对齐，`27` 个 recording 则与 `ayaw` 对齐。  
       这会直接影响 `event_level` 和 episode 内 `primary` 的选择。

因此，当前最值得优先修的不是先换更复杂模型，而是：

1. **先修 event/LTR 与 speed 单位的硬错误。**
2. **再统一 anchor 到“驾驶员开始控制/初始修正”的语义。**
3. **然后再决定 future horizon 是保留 2s 还是拆成更合理的双阶段评估。**

---

## 1. 我检查了哪些代码与数据

### 1.1 主线训练与协议

- `F:\data_set_process\data_process\datasetprocess\final_code\model\training\run_exp2_clean_baseline.py`
- `F:\data_set_process\data_process\datasetprocess\final_code\model\training\future_steer_speed_subjectsplit_masked.py`
- `F:\data_set_process\data_process\datasetprocess\final_code\model\training\protocol_primary_control_v1\protocol_config.json`
- `F:\data_set_process\data_process\datasetprocess\final_code\model\training\protocol_primary_control_v1\sample_manifest.csv`
- `F:\data_set_process\data_process\datasetprocess\final_code\model\training\README_PRIMARY_CONTROL_WORKFLOW_20260318.md`

### 1.2 事件生成逻辑

- `F:\data_set_process\data_process\datasetprocess\多模态数据\数据处理代码\车辆\事件检测.py`

### 1.3 原始数据

- `F:\data_set_process\data_process\datasetprocess\多模态数据\被试数据集合\*\vehicle\*_vehicle_aligned_cleaned.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\被试数据集合\*\event\*_vehicle_aligned_cleaned_events_v312.csv`
- 共扫描 `91` 个 vehicle recording，其中 `84` 个 recording 进入 usable 样本池，`5` 个 recording 缺少事件文件或无法用于当前审计。

---

## 2. 当前代码逻辑梳理

### 2.1 事件 start / end 从哪里来

当前训练样本并不自行定义事件边界，而是直接读取 `_events_v312.csv` 中的 `start_s / end_s`。

`事件检测.py` 的核心逻辑是：

1. 用 `|steer_rate| > 4 rad/s` 触发事件。
2. 触发后向前/向后扩展，直到 steering 进入稳定段，或者达到最大时长上限。
3. 对每个事件计算 `roll_peak / LTR_peak / steer_rate_peak / yaw_rate_peak / vx_mean` 等统计量。
4. 以这些统计量把事件分成 `weak / medium_active / strong_active / extreme_active`。
5. 以 `MIN_EP_GAP = 5s` 划 episode。
6. 在每个 episode 里把 `LTR_peak` 最大的事件标成 `primary`。

因此，当前训练集的事件含义是：

- **先由 steer-rate 触发**
- **再按 steering 稳定性扩展边界**
- **再按 LTR_peak 决定 primary**

### 2.2 训练样本里的 anchor 怎么定义

`future_steer_speed_subjectsplit_masked.py` 并不是拿 `event_start` 当 anchor，而是：

- 先过滤出 `event_level ∈ {medium_active, strong_active, extreme_active}` 且 `phase_type == primary` 的事件。
- 再在事件内部找一个“当前时刻”作为 anchor。

具体规则：

1. **curve**
   - `is_curve == True` 时，anchor = 事件内 `|roll|` 最大点。

2. **non-curve**
   - 先算 `steer_rate = gradient(steer)`。
   - 在事件段里找第一个满足 `|steer_rate| >= 0.8 * max(|steer_rate|)` 的点。
   - 若 `steer_rate` 几乎无峰值，则退化为 `|roll|` 峰值点。

当前 usable 样本的 anchor 来源统计：

| anchor_source | count | ratio |
|---|---:|---:|
| noncurve_steer_rate_first80 | 656 | 74.3% |
| curve_roll_peak | 227 | 25.7% |

当前 usable 样本里 **没有 lat-trigger 样本**，全部都是 `trigger_type = steer`。  
所以这次审计里 scene 差异的主来源不是 `trigger_type`，而是 **curve / non-curve 的 anchor 规则差异**。

### 2.3 valid future length / event mask 怎么构造

当前训练代码的 future 标签与 mask 构造是：

- `valid_future_len = min(FUTURE_LEN, event_end_idx - (anchor_idx + 1))`
- 也就是：**从 anchor 后一个采样点开始，到事件结束之前，还剩多少未来点**。
- 然后创建长度 `400`（2.0s）的 mask，前 `valid_future_len` 个点为 1，其余为 0。

这意味着：

- future 标签本质上是“**anchor 之后，事件内部还剩多少真实监督**”
- 并不是“无条件监督未来 2s”
- 所以 `1.0–2.0s` 的误差解释必须强依赖 mask

### 2.4 输出标签是什么

当前输出是：

- `y_steer[t] = steer(anchor + t) - steer(anchor)`
- `y_speed[t] = speed(anchor + t) - speed(anchor)`

也就是相对 anchor 的 delta 序列。

---

## 3. 审计统计结果

### 3.1 原始事件到 usable 样本

| 项目 | 数值 |
|---|---:|
| raw primary strong events | 917 |
| usable samples | 883 |
| usable ratio | 96.3% |
| drop: no_future_inside_event | 33 |
| drop: insufficient_recording_tail | 1 |

这一段说明：**主问题不是大量样本在建样本阶段被丢弃，而是留下来的样本本身 future 支持不够。**

### 3.2 当前 anchor 落在事件的什么位置

| 指标 | 数值 |
|---|---:|
| anchor_pct mean | 47.3% |
| anchor_pct median | 47.4% |
| p25 | 28.7% |
| p75 | 67.2% |
| p90 | 81.9% |

按“相对 first-response / risk-peak / control-end”的 phase 统计：

| anchor_phase | count | ratio |
|---|---:|---:|
| between_onset_and_risk_peak | 308 | 34.9% |
| late_adjustment | 242 | 27.4% |
| response_onset | 167 | 18.9% |
| pre_response | 79 | 8.9% |
| near_risk_peak | 74 | 8.4% |
| settling_tail | 13 | 1.5% |

更关键的是，**curve / non-curve 完全不是同一种分布**：

| scene | pre_response | response_onset | between_onset_and_risk_peak | near_risk_peak | late_adjustment | settling_tail |
|---|---:|---:|---:|---:|---:|---:|
| curve | 79 | 17 | 16 | 51 | 54 | 10 |
| non_curve | 0 | 150 | 292 | 23 | 188 | 3 |

这张表基本已经说明：

- `non-curve` anchor 主要像“**控制开始后不久到风险峰值之前**”
- `curve` anchor 很多时候像“**风险已到 / 甚至还没真正开始 steering response**”

所以**当前 anchor 语义不统一**是明确成立的。

### 3.3 当前 2s future 到底监督了多少

全局统计：

| horizon | full covered ratio | point valid ratio |
|---|---:|---:|
| 0–0.5s | 84.8% | 93.6% |
| 0.5–1.0s | 60.2% | 72.1% |
| 1.0–2.0s | 11.9% | 30.4% |

`valid_future_s` 分布：

| 指标 | 数值 |
|---|---:|
| mean | 1.133 s |
| median | 1.195 s |
| p25 | 0.690 s |
| p75 | 1.455 s |
| full 2.0s ratio | 11.9% |

按 anchor phase 看 future support，更能看到问题来源：

| anchor_phase | mean valid future | full 2s ratio | 1.0–2.0s point valid ratio |
|---|---:|---:|---:|
| pre_response | 1.619 s | 17.7% | 63.9% |
| response_onset | 1.268 s | 16.2% | 37.6% |
| between_onset_and_risk_peak | 1.318 s | 18.5% | 39.3% |
| late_adjustment | 0.822 s | 2.1% | 10.9% |
| near_risk_peak | 0.751 s | 2.7% | 10.8% |
| settling_tail | 0.053 s | 0.0% | 0.0% |

这说明：

- **不是所有样本都没 future**，而是“anchor 越晚，future 越塌”。
- 当前 late steer 的评估困难，很大程度是因为一大批样本的 anchor 已经落在 **late-adjustment / near-risk-peak**。

### 3.4 不同 scene 的 support

| scene | mean valid future | full 2s ratio | 0–0.5s point valid | 0.5–1.0s point valid | 1.0–2.0s point valid |
|---|---:|---:|---:|---:|---:|
| curve | 1.067 s | 9.3% | 87.6% | 63.4% | 31.2% |
| non_curve | 1.156 s | 12.8% | 95.7% | 75.2% | 30.2% |

curve 的 future support 稍弱，但真正致命的不是单纯 curve 样本少，而是 **curve anchor 经常不在“控制开始阶段”**。

### 3.5 当前 event end 是否等于控制调整结束

用“最后一次显著控制活动”作为粗略控制结束点后，`event_end - control_end` 的统计为：

| 指标 | 数值 |
|---|---:|
| mean | 62.5 ms |
| median | 5 ms |
| p75 | 10 ms |
| p90 | 275 ms |
| max | 1.475 s |

结论：

- **大多数事件的 end 并不离谱**，与控制结束相差不大；
- 但 curve 中存在明显尾段事件，事件后半已经主要是车辆动态尾响应，而不是新的控制输入。

所以：

- `event_end` 不是当前第一优先级的大问题；
- 但对于 curve，仍然建议后续单独复核“是否需要更贴近控制结束”的边界定义。

---

## 4. 同步、单位、派生特征与符号检查

### 4.1 steer 与 yawrate / ay 的相对时滞

最佳相关时滞统计：

| 指标 | 中位数 |
|---|---:|
| steer -> yawrate | 25 ms |
| steer -> ay | 125 ms |

按 scene：

| scene | steer->yaw median lag | steer->ay median lag |
|---|---:|---:|
| curve | 45 ms |
| non_curve | 15 ms |

相关符号稳定性：

| metric | ratio |
|---|---:|
| steer_yaw_negative_corr | 94.7% |
| steer_ay_negative_corr | 93.0% |

这表明：

- 当前 steer 与 yawrate / ay 的相对时序 **总体物理可解释**
- 左右符号关系也 **基本稳定**
- 因而“**大范围同步错位**”不像是当前 late steer 的主因

### 4.2 steer_rate 是否容易受噪声影响

`steer_rate` 由 `np.gradient(steer)` 直接得到，没有任何平滑。

统计结果：

| 指标 | 数值 |
|---|---:|
| steer_rate_noise_ratio median | 0.836 |
| `|smooth_anchor_shift| > 100 ms` | 170 / 883 |
| `|smooth_anchor_shift| > 500 ms` | 76 / 883 |

分 scene 看：

| scene | median | `>100ms` | `>500ms` |
|---|---:|---:|---:|
| non_curve | 20 ms | 169 | 75 |
| curve | 0 ms | 1 | 1 |

结论：

- 对 curve 来说，anchor 基本不受 `steer_rate` 噪声影响，因为它根本不用 `steer_rate` 定位。
- 对 non-curve 来说，绝大多数样本的 shift 不大，但**存在明显长尾**。
- 也就是说：`steer_rate` 噪声不是全局主问题，但对一部分 non-curve 样本会显著改变 anchor 位置。

### 4.3 speed 单位是否一致

这是本次审计里最明确的硬错误之一。

当前训练代码优先选 `zx|vx` 作为 speed，然后无条件做：

```python
df_feat[col_speed] = df_feat[col_speed] / 3.6
```

但审计结果显示：

| 指标 | 数值 |
|---|---:|
| median(`v_km/h / speed_raw`) | 3.6002 |

这说明：

- 当前 `zx|vx` **已经是 m/s**
- 再除一次 `3.6` 后，就变成了 **错误缩小 3.6 倍** 的速度值

因此当前 speed 相关问题包括：

1. `y_speed` 标签的数值尺度错了；
2. 损失里 steer / speed 的相对权重被改变；
3. 报告里的 `speed_ms` / `speed_kmh` 命名已经不再对应真实单位。

这是一个 **必须先修** 的管线错误。

### 4.4 ay / LTR_est / event_level 是否一致

这是本次审计里第二个必须优先修的硬错误。

事件检测脚本里 `LTR_peak` 的来源依赖：

```python
ay_col = find_col(df, ["ay"])
```

而 `find_col` 是“只要列名里包含 `ay` 就算匹配”。  
这会让 `zx|ayaw` 和 `zx|ay` 都可能被命中；如果不同 recording 的列顺序不同，取到的列也会不同。

审计结果显示：

- **55 个 recording / 590 个 usable event** 的 `LTR_peak` 更接近 `zx|ay`
- **27 个 recording / 293 个 usable event** 的 `LTR_peak` 更接近 `zx|ayaw`

更直观的证据：

- 对一批 extreme 事件，`LTR_peak_event_file` 与 `0.11243 * max(|zx|ayaw|)` 的误差是 `1e-6` 量级；
- 同一批事件若用 `0.11243 * max(|zx|ay|)` 去复算，误差却能到 `0.7 ~ 1.0+`。

这意味着：

- 当前 `_events_v312.csv` 中的 `LTR_peak`
- 以及基于 `LTR_peak` 的 `event_level`
- 和 episode 内 `primary` 的选择

**在 recording 之间并不使用同一物理信号。**

这不是小瑕疵，而是会直接影响：

1. 哪些事件被判成 `medium/strong/extreme`
2. 哪个事件被当成 `primary`
3. 当前主任务的入样本集合

因此这属于 **事件定义问题 + 派生特征问题** 的叠加。

### 4.5 曲率方向 / steering 方向

在 curve-only 事件里：

| 指标 | ratio |
|---|---:|
| curve_sign_matches_direction | 78.9% |
| curve_sign_opposes_direction | 21.1% |

说明当前曲率方向与 steering 方向之间并不是随机关系，**总体是有稳定对应的**。  
因此符号体系没有表现出“完全乱掉”的特征；真正更危险的是 **LTR 来源混乱** 与 **speed 单位错误**。

---

## 5. 抽样图与案例分析

完整 48 条抽样案例见：

- `event_anchor/sample_gallery.md`
- `event_anchor/figures/`

这里挑 4 个最有代表性的案例放进报告。

### 5.1 Case 02：curve 样本，anchor 早于 steering response

`lxy / curve / strong_active / recentering / full 2s`

- anchor = `0.195s`（事件 6.5%）
- first_response = `0.365s`
- trigger = `0.390s`
- anchor phase = `pre_response`

这说明 curve anchor 可能落在“车辆动态/roll 已经高，但驾驶员真正大动作还没开始”的位置。

![case02](event_anchor/figures/case_02_lxy_curve_strong_active_recentering.png)

### 5.2 Case 01：non-curve 样本，anchor 落在 response 与 risk peak 之间

`hzh / non_curve / medium_active / multi_correction / 1.78s future`

- anchor = `1.225s`
- first_response = `0.010s`
- risk_peak = `1.945s`
- anchor phase = `between_onset_and_risk_peak`

这更符合“驾驶员已经开始修正，模型预测后续控制”的语义。

![case01](event_anchor/figures/case_01_hzh_non_curve_medium_active_multi_correction.png)

### 5.3 Case 17：curve 样本，anchor 落在 late adjustment，只剩 0.19s future

`yyl / curve / medium_active / insufficient_late_support`

- anchor = `1.170s`（事件 85.7%）
- valid_future = `0.190s`
- anchor phase = `late_adjustment`

这类样本几乎不可能为 `1–2s` 的 future 提供有效监督。

![case17](event_anchor/figures/case_17_yyl_curve_medium_active_insufficient_late_support.png)

### 5.4 Case 19：curve 样本，anchor 在事件起点，控制很快结束，但事件尾巴很长

`rjy / curve / extreme_active / reverse_correction`

- anchor = `0.000s`
- first_response = `0.270s`
- control_end = `0.325s`
- event_end = `1.800s`

这说明某些 curve 事件在训练视角里已经包含了大量“动态尾响应”，而不是同质的控制调整过程。

![case19](event_anchor/figures/case_19_rjy_curve_extreme_active_reverse_correction.png)

### 5.5 关于 non-curve 的导数敏感性

另一个值得注意的案例是：

- `event_anchor/figures/case_03_rjy_non_curve_medium_active_multi_correction.png`

该样本的 `smooth_anchor_shift_ms = 1135 ms`，说明当前 non-curve 的“首个 80% steer-rate 峰值”规则对局部尖峰存在敏感长尾。  
这不是全局主问题，但足以说明 `steer_rate` 最好不要继续完全裸导使用。

---

## 6. 问题归因：到底最可疑的是哪一类问题

### 6.1 我认为最可疑的问题排序

#### 第一位：anchor 对齐问题

理由：

- curve / non-curve anchor 规则完全不同；
- curve 中大量 anchor 落在 `pre_response / near_risk_peak / late_adjustment`；
- non-curve 则主要落在 `response_onset / onset-to-risk-peak`。

这直接破坏了“当前样本都在研究同一控制阶段”的前提。

#### 第二位：future support 问题

理由：

- 2s 完整监督比例只有 11.9%；
- `1.0–2.0s` point valid ratio 只有 30.4%；
- anchor 越晚，support 越差；
- 事件最长约 3s，本身就很难支撑“中后段 anchor + 完整 2s future”。

#### 第三位：事件定义 / 标签问题

理由：

- `LTR_peak -> event_level -> primary` 的链条在 recording 之间并不稳定；
- 有 27 个 recording 的 `LTR_peak` 实际更像由 `ayaw` 而不是 `ay` 得到；
- 这会把“哪些事件进入主任务”这个问题本身变得不一致。

#### 第四位：同步 / 派生特征问题

理由：

- 大范围时序错位的证据不强；
- 但 `steer_rate` 的长尾敏感性存在；
- speed 单位误缩放是硬错误，必须修；
- 因此它不是解释 late steer 的首因，但确实是影响结果可信度的高优先级 bug。

### 6.2 哪些必须先修

**必须先修：**

1. `speed` 单位错误  
   当前 `zx|vx` 已是 `m/s`，不能再除 `3.6`。

2. `LTR_peak` 来源不一致  
   事件检测脚本中不能再用模糊匹配 `find_col(df, ["ay"])`；必须显式绑定 `zx|ay`。

3. `anchor` 语义统一  
   不要再让 curve 用 `roll_peak`、non-curve 用 `steer_rate_first80`。  
   下一版应该优先统一到“驾驶员首次明确控制响应”附近。

**可以后面再说：**

1. 是否细化 event end  
   当前整体上不是首要瓶颈。

2. 是否引入更复杂模型  
   在当前标签/anchor/单位未修之前，不建议先堆模型。

3. 是否大规模加新特征  
   现在更该先把定义和监督逻辑理顺。

---

## 7. 下一步最值得做的 1–2 个修正实验

### 实验 1：修复事件生成与单位，但不换模型

目标：

1. 在事件检测脚本里把 `LTR_peak` 显式改为 `zx|ay * 0.11243`
2. 重新导出 `_events_v312` 的修正版
3. 在训练样本构造里修掉 `speed / 3.6` 的重复缩放
4. 其他结构、split、loss、模型全部不动

这个实验能回答：

- 当前很多结论里，有多少是“脏标签/脏单位”造成的
- 修正后事件 level 与 primary 是否明显稳定

### 实验 2：统一 anchor 到 response-onset 语义

目标：

1. curve / non-curve 都先用同一套 response-onset 规则
2. 先不要改模型与 loss
3. 只比较：
   - anchor 分布是否更一致
   - mean valid future 是否改善
   - `1.0–2.0s` support 是否明显上升
   - late steer 是否更可解释

推荐的候选 anchor：

- 事件内 smoothed `|steer_rate|` 首次超过 `30% * peak` 且持续若干采样点

这样至少能保证：

- 研究对象更接近“驾驶员开始如何调整”
- curve / non-curve 的样本语义一致

---

## 8. 审计产物清单

### 8.1 新增脚本

- `F:\data_set_process\data_process\datasetprocess\final_code\model\training\audit_primary_control_common.py`
- `F:\data_set_process\data_process\datasetprocess\final_code\model\training\audit_event_anchor_semantics.py`
- `F:\data_set_process\data_process\datasetprocess\final_code\model\training\audit_future_support_and_sync.py`

### 8.2 主要输出

#### 事件 / anchor / future

- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\event_anchor\event_anchor_manifest.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\event_anchor\usable_event_anchor_manifest.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\event_anchor\event_anchor_summary.json`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\event_anchor\future_support_by_scene.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\event_anchor\future_support_by_anchor_phase.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\event_anchor\sampled_cases.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\event_anchor\sample_gallery.md`

#### 同步 / 单位 / 符号

- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\sync_audit\sync_event_metrics.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\sync_audit\speed_unit_audit_by_file.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\sync_audit\lag_summary_by_scene.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\sync_audit\sign_consistency_summary.csv`
- `F:\data_set_process\data_process\datasetprocess\多模态数据\程序运行结果\AUDIT_EVENT_SYNC_20260320_134207\sync_audit\sync_summary.json`

---

## 9. 最终判断

如果只允许用一句话总结：

> 当前 late steer 的主要问题，不是先换模型，而是 **当前事件定义里 LTR/强度标签不完全可靠，anchor 在不同 scene 上对齐到的控制阶段不一致，而且 2s future 在大多数样本上根本没有被完整监督。**

所以真正建议的顺序是：

1. 先修 `speed` 和 `LTR` 这两个硬错误  
2. 再统一 anchor 到 response-onset  
3. 然后才谈 2s horizon 是否继续保留、以及模型升级

