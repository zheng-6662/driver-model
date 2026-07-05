# 2026-06-30 所有记录在案方法路线系统审计

## 审计范围

本报告用于回答“下一步之前，先系统审计所有记录在案的方法，再决定下一步”的问题。审计范围包括：

- 旧实验登记：`04_project_logs/reports/progress/experiment_registry.md`
- 两个月路线重建：`03_baselines/v220_two_month_route_reconstruction_20260622`
- 两个月经验复盘：`03_baselines/v229_two_month_lessons_failure_taxonomy_20260623`
- 当前重建主线：`v221` 到 `v248`
- 旧阶段报告：`09_reports/stage03`、`stage06`、`stage07` 等车辆-only、多候选、响应分解、selector 和样本重建报告

审计口径不是“哪个版本号最大”，而是每个方法族是否真正服务于用户目标：**用模型方法预测行为轨迹**。因此重点看：

1. 该方法是否提高可部署预测能力，而不是只提高 oracle 上限。
2. 该方法是否稳定改善 hard case，同时不伤 normal 样本。
3. 该方法是否需要 test 后验信息、人工标签或真实未来轨迹才能发挥作用。
4. 该方法是否已经被前面实验反复证明存在结构性缺陷。
5. 该方法是否会把任务从“行为预测方法提升”带偏成“失败机制写作/删样本/包装论文”。

---

## 总体结论

截至 v248，所有路线反复指向同一个瓶颈：

> 普通样本和方向性预测已经不是最大问题；真正阻塞是强变化样本的轨迹形状建模，包括幅值、斜率、峰谷、回正、反打和多阶段修正。候选池、锚点、oracle selector、响应分解都能解释一部分现象，但都没有稳定转成可部署预测能力。

因此下一步不应继续：

- v222a gate / no-harm gate / learned selector
- 删除差样本
- 轻量 residual 叠加
- 单纯 anchor selector
- response type 先硬分类再预测
- best-of-K / oracle candidate 当作结果
- 论文失败机制包装
- 继续盲目扩大模型而不改变 shape loss / shape target

下一步应进入一个明确的 `v249_shape_aware_curve_model`：在 v241/v248 证据基础上，直接针对轨迹形状误差建模。

---

## 方法族审计

### 1. 端到端完整曲线回归 / loss 堆叠 / 结构堆叠

代表路线：

- `structured_v2`
- `coarse-fine`
- `phase-adaptive`
- reversal / tail / first-reversal loss
- hard-late fine
- late residual head
- mainline tail calibration
- G13 / G14 前的完整曲线主模型

有效证据：

- 早期 `coarse-fine + phase-adaptive` 曾达到历史较强状态，例如 `rmse_steer=0.5697`，late peak 有改善。
- tail、peak、reversal 指标化后，能够把“预测曲线变平”“尾段塌陷”“反打没抓住”这些视觉问题量化出来。
- 完整曲线预测仍是论文/任务的核心形式，不能彻底退化成二分类。

主要问题：

- 单纯加 loss 往往是“救一个指标、伤另一个指标”：例如 late peak 提升但 tail amplitude / strong-pos tail collapse。
- 强反应、极端峰值、尾段、反转多修正之间存在目标冲突，普通 MSE 会把强变化样本压成平均曲线。
- 盲目结构堆叠没有解决核心：模型仍倾向输出保守、平滑、小幅轨迹。

审计裁决：

- **保留经验，不继续原样堆 loss。**
- 有用的是“曲线形状指标体系”和“coarse/fine 思想”；无效的是继续无约束堆模块。
- 下一步可以继承其思想，但必须变成明确 shape-aware 目标，而不是再泛泛调 loss。

---

### 2. 可复现性、协议安全、formal baseline 冻结

代表路线：

- `220918` reproducibility audit
- strict deterministic / warn-only rerun
- manual coarse upsample
- protocol-safe baseline
- v221 formal leaderboard
- v225/v226 formal route reconstruction + CI
- v228 final paper artifact freeze

有效证据：

- 修复了大量“结果是不是可信”的基础问题：split、checkpoint、metric、ZIP、leakage、forbidden feature。
- v221 明确 formal headline：loose pool `avg_joint_focus`，strict pool `peak_floor_090`。
- v225/v226 让旧 formal 指标可复现，给出 CI 和 tail error concentration。

主要问题：

- 这条线解决的是“结果可信”和“能不能写作”，不是继续提升模型。
- v228/v230 之后如果继续停留在 paper packaging，会远离用户当前目标。

审计裁决：

- **作为 guardrail 继续保留，但不作为下一步研究主线。**
- 后续任何 v249 都应继承它的纪律：train/val/test 分离、禁止 test 调参、保留 no-leak 检查。

---

### 3. 生理 / EEG / 连续风格融合

代表路线：

- E15/E16 单信号
- E17 语义状态
- E18 去人工权重表示
- E19 多信号融合
- G13H/G13I 生理/状态类候选

有效证据：

- 肌电或部分状态信号有时有辅助价值。
- 信号语义化比直接原始输入更合理。
- 这些信号可用于机制解释或状态审计。

主要问题：

- 没有稳定超过强车辆-only/旧主线。
- 多信号融合容易加噪声。
- 在车辆-only 问题还没有闭环时，直接把生理/EEG塞进曲线主模型，会把问题变复杂但不一定提升行为预测。

审计裁决：

- **暂不作为 v249 主输入扩展。**
- 只有在 vehicle-only shape-aware 模型建立后，才考虑把生理/EEG作为辅助状态或不确定性解释变量。

---

### 4. 相似历史检索 / 原型 / 训练集响应参考

代表路线：

- G14 相似历史事件诊断
- G15 相似历史事件检索 + residual
- G16 响应类型原型

有效证据：

- 检索/原型上限说明训练集中存在相似响应，问题不完全是“数据里没有答案”。
- G15B 整体 RMSE 曾很好看，例如 test RMSE `0.3980`。
- G16 的 oracle/prototype 思路证明“响应类型/原型”具有解释价值。

主要问题：

- 困难样本 G11 / hard case 仍差。
- 物理指标、tail、selection score 没同步过关。
- 本质还是“怎么可靠选原型/相似样本”的问题；一旦选择错，整条轨迹被带错。

审计裁决：

- **作为诊断上限保留，不直接重启为主线。**
- 不建议回到“先选原型再预测”的硬流程。
- 可以把“原型形状参数”转化为辅助 shape target，而不是部署时硬选一个原型。

---

### 5. 多候选 / best-of-K / oracle candidate

代表路线：

- Stage 6e 多候选 oracle gap
- Stage 7a 非 oracle selection protocol
- Stage 7b/7d non-oracle selector
- top-K Transformer branch
- keypoint / segment candidate
- v222a candidate cache

有效证据：

- oracle upper bound 很强。例如 Stage 6e broad oracle test RMSE `0.375182`，比 RBF/KNN `0.533667` 明显好。
- v222a oracle safe gate 也有明显上限，loose test oracle tail gain `0.105286`，strict test oracle tail gain `0.106719`。
- 说明候选空间中常有更好的曲线。

主要问题：

- 可部署 selector 学不稳。Stage 7d 最后 selected policy 回到 `always_rbf_reference`，test RMSE 仍 `0.533667`。
- v222a learned gate 在 locked test 不稳：loose pool RMSE/tail 变差，strict pool under reduction 变差。
- selector_failed_rate 长期在约 `0.41` 左右，candidate_missing_rate 只有约 `0.03`。也就是说问题不是没有候选，而是当前输入下选不准。

审计裁决：

- **停止把 oracle/best-of-K 当主路线。**
- 下一步不应继续训练更复杂 selector。
- 多候选思想可以转为“训练时辅助 shape diversity / shape parameter supervision”，不能作为部署时硬选择器。

---

### 6. response type / response-factorized / 机制分解

代表路线：

- Stage 3 response decomposition labels
- Stage 7f response-factorized candidates
- G16 response type prototype
- 反转/多修正/late peak/幅值桶等机制标签

有效证据：

- 方向因子较强：Stage 7f direction test accuracy `0.925`，balanced accuracy `0.919`。
- peak timing 有一定信号：test balanced accuracy `0.662`。
- 反转/多修正、tail、amplitude 等标签能帮助解释坏样本。

主要问题：

- 幅值和尾段不稳：amplitude balanced accuracy 约 `0.496`，tail balanced accuracy 约 `0.380`。
- 用户已指出：先判断响应类型再预测，本质上会产生错误传播；类型一旦错，轨迹整体错。
- 早期也尝试过响应类型路线，没有形成稳定主线。

审计裁决：

- **不能作为前置硬分类主线。**
- 可以作为 v249 的辅助多任务头：预测幅值、峰值时间、转折、尾段状态，但主轨迹仍连续输出，不能用 hard route 决定整条曲线。

---

### 7. gate / router / no-harm / reliability selector

代表路线：

- Stage 6c selector feature revision
- Stage 6d reliability gate
- W2/W3 多专家与 router
- v222a no-harm gate
- v222a bounded residual + learned gate

有效证据：

- 局部可以改善某些物理指标，例如 wrong-side 或 large recall。
- no-harm 框架本身有价值：它让 normal 样本不被 hard case 优化牺牲。

主要问题：

- 反复出现“val 通过、locked test 不稳”。
- selector/gate 往往用当前窗口信息判断未来响应，输入信号不足时必然不稳。
- learned gate 容易在改善 underestimation 的同时伤 RMSE/tail，或者保护 RMSE/tail 但 under rate 变差。

审计裁决：

- **停止作为主要提升路线。**
- no-harm 仍作为模型选择约束保留。
- 不能再把 v222a gate/router/selector 作为 v249 起点。

---

### 8. 删除样本 / 过滤 hard sample

代表路线：

- v235 删除 observe_later_like 后重训

有效证据：

- 删除后测试集指标明显好看：loose old full test RMSE `0.555940`，保留子集旧模型 `0.482685`，重训 `0.474318`。
- 说明 observe_later_like 确实是一类难样本，需要单独分析。

主要问题：

- 指标提升主要来自测试集变容易，不是模型能力真正增强。
- removed holdout 仍然很难：loose `0.868780`，strict `0.845273`。
- 这会把用户真正关心的难行为样本从任务里拿掉。

审计裁决：

- **不能作为方法提升路线。**
- 只保留为难样本分层诊断证据。

---

### 9. 样本重建 / 锚点修正 / 观察层

代表路线：

- v231 最差样本锚点上下文
- v232 过晚锚点重锚定候选
- v233 自适应锚点/观察策略
- v234 短观察后预测层
- v245 差样本锚点后移效果审查
- v246 oracle best anchor + selector
- v247 50ms fine-grid best anchor
- v248 best-anchor 后 residual shape audit

有效证据：

- 一部分差样本确实是锚点/可观测性问题。v232 找到 11 个重锚定候选，v245/v246/v247 证明后移/最佳锚点有上限收益。
- v247 fine grid 证明细粒度 best anchor 成立：test/all 0ms `0.475` 到 best `0.253`；bad_top10 `1.198` 到 `0.616`。
- v248 进一步确认：best anchor 后仍差的样本，range/excursion/slope ratio 明显偏低。

主要问题：

- v247 的 oracle best anchor 不能部署；selector 仍弱于简单 wait-latest 或不稳定。
- v248 已经证明：锚点不是当前主要矛盾。bad_top10 即使用 best anchor 仍 `0.616`，`47.4%` 仍高于 `0.65`。
- 继续做 anchor selector 会偏离当前核心瓶颈。

审计裁决：

- **锚点线阶段性收口。**
- 保留 best-anchor 作为 upper bound 和 casebook 解释，不继续把 anchor selector 当主线。
- 下一步转向 shape modeling。

---

### 10. rolling/reanchor 数据集与 original_remaining 任务构造

代表路线：

- v236 rolling/reanchor dataset + joint Ridge
- v237 rolling target/phase audit
- v238 task/model redesign

有效证据：

- v236 构造了 `1167` 个唯一事件、`7002` 个 rolling 样本。
- v237 证明 v236 的一部分失败来自 receding horizon 后移混入新行为阶段，`original_remaining` 口径更合理。
- v238 接受 `original_remaining masked point-level target`：delay 后只预测 original anchor+2s 剩余重叠段。

主要问题：

- v236 joint Ridge 太弱，0ms 不如旧 formal。
- v238 小 MLP 改善 observe/strong，但伤 normal，1000ms late delay 不稳。

审计裁决：

- **任务构造应保留。**
- v249 应继续使用 original_remaining / masked target / rolling observation 的纪律。
- 不能回到 v236 receding_2s 直接预测新 2s horizon 的混杂口径。

---

### 11. attention / TCN / stronger temporal model

代表路线：

- v239 light temporal attention
- v240 locked attention audit
- v241 TCN + multi-head query attention

有效证据：

- v239 解决了 v238 伤 normal 的问题，attention 对 observe_later_like 和 normal 都稳定改善。
- v241 是目前最强可用候选：相对 v239，test 中 observe_later_like、normal_predictable、strong_steer 六个 delay 的 tail RMSE 均改善。
- v241 保留了连续预测，不是 hard response route。

主要问题：

- v241 仍有逐样本回退：all test 中 `368/1104` 条 tail 回退，strong_400_1000 中 `47/160` 条回退。
- v248 说明 v241 的 hard case 主要是轨迹太平、幅值/斜率不足、转折不足。

审计裁决：

- **v241 是下一步最合理的 backbone。**
- 不建议推翻 v241 从零开始；应在 v241 表征上加入 shape-aware 目标或 shape residual correction。

---

### 12. 联合曲线 decoder

代表路线：

- v242 joint curve decoder

有效证据：

- 一次输出整条 21 点曲线的思路是对的，曲线连续性目标也符合当前问题。
- v242 相对 v236 有效。

主要问题：

- v242 没超过 v241：normal 全 delay 变差，strong 多数 delay 变差；all test 中 `588/1104` 条相对 v241 回退。
- 简单 joint decoder + smooth loss 可能进一步平滑曲线，反而不利于强变化。

审计裁决：

- **不直接重启 v242 架构。**
- 可吸收“一次建模整条曲线”的思想，但必须加入 peak/slope/turning 等 shape 约束，不能只加 smooth。

---

### 13. guarded fine-tune / hard sample weighting / light residual

代表路线：

- v243 v241 guarded fine-tune
- v244 hard36 vs hard24 audit
- v222a light residual

有效证据：

- 在 v241 backbone 上微调确实能产生小幅增益。
- hard24 在 locked test 上比 hard36 更稳，说明强约束/保守策略有价值。

主要问题：

- 增益极小，且 validation-selected hard36 在 observe/strong test bucket 迁移不稳。
- hard24 缺少完整 granular artifact，不能直接升级。
- 轻量 residual 线此前已经被 v222a closeout 证明不适合作为 formal 主线。

审计裁决：

- **不作为下一步主线。**
- v249 可以借鉴 guard loss/no-harm，但不能只是再调 hard weight 或 residual。

---

### 14. 失败案例复核 / 论文包装

代表路线：

- v229 two-month lessons
- v230 failure case manual review
- v228 final paper artifact freeze

有效证据：

- 这些包帮助我们识别主要失败桶：强反应低估、极端峰值、尾段、反转、多次修正。
- 对老师沟通、论文边界、casebook 很有用。

主要问题：

- 用户已经明确：目标是方法提升，不是失败机制分析论文。
- 如果继续写 case section，而不转回模型方法，会偏离目标。

审计裁决：

- **作为证据材料保留，但停止把它作为主线。**
- 下一步必须回到预测模型方法。

---

## 哪些方向已经离目标越来越远

1. **删样本路线**：指标会变好，但目标变窄，不能预测用户真正关心的强变化行为。
2. **oracle/best-of-K 路线**：上限很好看，但部署时选不准，容易制造假希望。
3. **gate/router/selector 反复调参**：多次证明 val 到 test 不稳，尤其 current-window 输入不足时。
4. **response type 前置硬分类**：一旦类型错，轨迹整体错；用户也指出此前已尝试过。
5. **单纯锚点 selector**：v248 已证明锚点不是剩余主要矛盾。
6. **继续 paper failure 包装**：对论文有用，但不是模型提升。
7. **无 shape 目标的更大模型**：会继续输出平均化、平滑化曲线。
8. **直接融合生理/EEG**：在 vehicle-only 主问题未闭环前容易加噪声。

---

## 下一步建议：v249_shape_aware_curve_model

### 基本原则

v249 不应推翻 v241，也不应回到 v222a/v247 selector。它应该是：

> v241 backbone + shape-aware trajectory objective。

也就是仍然做行为轨迹预测，但让模型明确学习真实曲线的形状，而不是只靠点均方误差。

### 推荐结构

1. **主干**：沿用 v241 的 TCN + multi-head query attention。
2. **主输出**：继续输出 21 点 future steering delta，保持 `original_remaining` masked target。
3. **辅助 shape heads**：
   - peak magnitude：最大绝对转向幅值
   - peak time：峰值出现时间
   - trough / rebound：谷值、回正幅度
   - slope energy：最大斜率、平均斜率
   - tail level：1.0-2.0s 尾段均值/终点偏移
   - reversal / zero-cross soft label：作为辅助，不做 hard route
4. **shape loss**：
   - point masked MSE
   - peak amplitude loss
   - slope / first-difference loss
   - tail loss
   - turning-point / curvature loss
   - normal no-harm guard

### 为什么这比继续调锚点更合理

v248 显示 best anchor 后仍差组：

- `mean_best_anchor_rmse=0.828`
- `range_ratio=0.466`
- `excursion_ratio=0.438`
- `slope_ratio=0.405`

这不是“曲线整体平移一下就好”的问题，而是模型没有输出足够大的幅值、足够快的变化和正确的转折结构。

### v249 的成败标准

建议 validation 先看，不用 test 调参：

- normal_predictable 不伤：normal tail/sample delta 不能明显为正。
- strong_steer 改善：strong tail、peak ratio、slope ratio 至少有稳定改善。
- bad_top10 / still_bad 组改善：不只看 RMSE，也看 range_ratio、slope_ratio、turning error。
- 不能只靠 under-rate 降低换来普通样本 false-large。
- 如果 v249 只降低平滑 loss，却让曲线更平，直接 no-go。

### 推荐的第一版 v249

第一版不要做太复杂：

- v241 checkpoint 初始化。
- 加 shape auxiliary heads 和 shape loss。
- 训练 2-3 个 loss weight 配置，全部 validation-only 选择。
- locked test 只用于最终审查。
- 输出必须包括：
  - by bucket metrics
  - per-sample delta vs v241
  - shape decomposition vs v248
  - bad_top10 / still_bad casebook
  - normal no-harm check

---

## 最终裁决

最应该继承：

- v238 的 `original_remaining masked target`
- v239/v241 的 attention/TCN temporal backbone
- v240/v244 的 locked audit 和 no-harm 纪律
- v248 的 residual shape decomposition 指标

最应该停止：

- v222a gate / bounded residual / no-harm selector 主线
- 删除样本
- 单纯 anchor selector
- response type hard routing
- oracle/best-of-K 作为结果
- 论文失败机制包装主线

下一步最合理路线：

> `v249_shape_aware_curve_model`：以 v241 为 backbone，以 v248 发现的 amplitude/slope/turning residual 为直接优化对象，继续保持 original_remaining 任务构造和 validation-only 模型选择。

