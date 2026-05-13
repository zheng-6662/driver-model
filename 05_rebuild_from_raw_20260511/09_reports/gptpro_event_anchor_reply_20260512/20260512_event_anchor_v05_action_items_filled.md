# GPTPro 回复转化后的执行任务

生成时间：2026-05-12

## 立即执行

1. 冻结 v0.5，不把 314 个高置信候选直接作为训练样本。
2. 建立 v0.6 事件复核标注规则。
3. 对现有 56 张代表性复核图生成复核表，字段包括：
   - `review_status`
   - `confidence_tier`
   - `is_primary_training_candidate`
   - `is_response_confirm_only`
   - `is_continuous_episode`
   - `coordinate_continuity_ok`
   - `pre_window_clean`
   - `post_response_confirmed`
   - `ego_relevance`
   - `review_note_cn`
4. 生成 v0.6 四类输出表：
   - `primary_training_events_v0_6`
   - `manual_review_events_v0_6`
   - `response_confirm_only_v0_6`
   - `holdout_or_excluded_v0_6`

## v0.6 自动规则优先级

1. `response_confirm_only`：横向加速度峰值、横摆角速度峰值、横滚速率峰值、横向偏移变化峰值、制动/方向盘峰值类候选。
2. 暂缓：`middle_section` 作为连续任务，不进入第一版单点事件训练。
3. 暂缓：`longstraight`、`stop`、`curve3`、`zd` 不进入第一版训练，只进诊断/复核池。
4. 第一版候选：`curve1/curve2` 的道路几何候选、`differentmu_road` 的 raw μ 候选、人工复核通过的 `fix_road` 显式变道候选。
5. 旧锚点接近程度只作解释字段，不作为核心评分。

## 后续可执行阶段

### 阶段 A：v0.6 复核表和初版样本清单

- 输入：v0.5 全部评分表、高置信复核表、代表性复核图索引。
- 输出：v0.6 四类事件表、分场景统计、中文报告。

### 阶段 B：核心场景车辆/道路-only baseline

- 只使用 v0.6 通过的核心场景。
- 不加入连续风格、生理、脑电。
- 重点指标包括方向符号、峰值幅值、大幅响应召回、time-to-peak、no-response false positive、分事件类型误差。

### 阶段 C：错误回看

- 如果车辆/道路-only baseline 仍然有明显错侧、幅值不足或时间错位，优先检查：
  - 左/右事件方向是否混合；
  - 曲率方向是否输入；
  - 车道左右关系是否一致；
  - 方向盘符号是否跨记录一致；
  - 横向偏移是否有坐标跳变；
  - t0 是否偏早或偏晚。

### 阶段 D：扩展任务

- `middle_section` 单独做连续任务/episode 模型。
- `longstraight`、`stop` 等先完成被试相关性/TTC/暴露点复核。
- 核心基线稳定后，再回到连续驾驶风格和生理数据增量验证。
