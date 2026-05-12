# 阶段 2 用户查看版总结：事件锚点与样本清单重建

## 最新修正：车辆失稳事件不再靠用户逐条人工标注

用户指出之前 404 个样本其实是弯道样本，不是车辆失稳样本；随后又指出 1227 个失稳候选如果逐条人工看，工作量太大。因此阶段 2 当前主线已经改成：从旧项目日志中找道路事件设定，用道路设定和旧 v400 事件上下文作为辅助先验，再结合原始车辆动态证据自动判定车辆失稳事件。

当前已生成 `vehicle_instability_road_guided_v0_1`：

- 全量失稳候选：1227 个；
- 自动/已确认采用：701 个；
- 中间复核：177 个；
- 低证据剔除：349 个。

这 701 个不是“人工真值”，但可以作为下一步车辆失稳样本 manifest 的主输入。道路模块只提供场景先验，不能单独证明失稳；事件成立仍然要看 `ay`、`roll_rate`、横摆角速度、横向偏移和事件后方向盘响应等车辆动态证据。

建议优先查看：

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/road_guided_instability_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_road_guided_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_auto_accepted_events_v0_1.csv`

下一步应基于这 701 个采用候选生成车辆失稳版样本清单和处理后车辆窗口，然后重新做无学习/纯车辆基线。之前基于 404 个弯道样本的阶段 3 结果只保留为历史诊断材料。

## 进一步修正：已经对全部原始车辆 CSV 重新筛选

用户进一步要求按上述标准对所有原始数据重新筛选一遍。因此现在不再只依赖已有候选表，而是直接读取 `原始车辆数据/<被试名>/*.csv` 下全部 91 个原始车辆文件重新扫描。

当前全量重筛版本是 `vehicle_instability_all_raw_rescreen_v0_1`：

- 原始车辆 CSV：91 个；
- 可读取：91 个；
- 非方向盘动态种子：4581 个；
- 合并后失稳候选：1991 个；
- 高置信主清单：908 个；
- 自动/已确认采用扩展清单：1348 个；
- 中间复核：269 个；
- 低证据剔除：374 个。

建议后续正式样本构建先使用 908 个高置信主清单；1348 个扩展采用清单可以作为敏感性对照或后续扩展，不建议一开始全部并入正式主训练集。

建议优先查看：

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/all_raw_vehicle_instability_rescreen_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_all_raw_vehicle_instability_rescreen_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_primary_high_confidence_v0_1.csv`

## 2026-05-12 重要修正：主线改为车辆失稳事件

用户已明确指出，之前自动审阅得到的 404 个样本都是弯道/道路曲率样本，而不是车辆失稳样本。这个修正已经纳入阶段 2：

- 404 个弯道候选不再作为主事件样本，只保留为道路上下文参考。
- 新主线改为 `vehicle_instability_onset_codex_v0_1`。
- 新锚点来自非方向盘车辆动态异常：`ay` 和 `roll_rate`。
- `steer_rate` 不再用于定义失稳开始，因为它属于驾驶员方向盘动作结果。
- 当前车辆失稳候选为 1227 个，其中自动高/中置信采用 358 个，需要复核 462 个。

推荐优先查看：

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/instability_event_review_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`

更新时间：2026-05-12

## 这个阶段为什么做

阶段 1 已经证明原始车辆、生理、脑电文件能互相对应，但旧流程的事件锚点不能默认相信。阶段 2 的目的就是重新把“一个样本从哪里来、什么时候开始预测、用哪段输入、预测哪段未来”写清楚。

## 这个阶段检查了什么

- 检查旧事件表能否和原始车辆文件按被试、记录时间对应。
- 找到旧项目中的道路设计/道路信息记录，并生成道路设计清单。
- 从原始车辆曲率信号找道路事件候选。
- 从原始车辆动态信号找响应 onset 候选。
- 给每个候选样本生成 1 秒、2 秒、3 秒和早期观察 0.5 秒四种窗口。
- 给每个样本写入车辆、生理、脑电是否覆盖输入窗口，以及车辆是否覆盖标签窗口。
- 生成随机切分、按记录切分、按被试切分三种 split 方案。

## 目前发现了什么

- 候选事件总数：11619。
- 样本窗口行数：46476。
- 道路设计目录文件数：49，其中含曲率信息的 CSV 为 8 个。
- 旧 v400 primary 事件候选：1461。
- 低泄漏道路曲率候选窗口行：1077。
- 已为低泄漏道路曲率候选生成第一版处理后车辆窗口数据：3 个 NPZ，每个窗口 359 个样本，车辆特征 9 个。
- 旧锚点和原始动态响应锚点在 1 秒内匹配的数量为 2817，说明旧事件大多能在原始车辆响应中找到对应迹象。
- 旧锚点和道路曲率锚点在 1 秒内匹配的数量为 169，说明道路曲率只能解释一部分事件。

## 哪些结果可信

- 每个样本行都能追溯到原始车辆文件、SHA256、被试、记录时间和窗口绝对时间。
- 每个样本都有明确的 `anchor_source` 和 `leakage_flags`。
- split 表已经保证同一事件的不同窗口不会分到不同训练/测试集合。
- 处理后车辆窗口只来自 `raw_road_curvature_onset`，未覆盖旧 v400 和 raw dynamic，因此适合作为保守车辆基线起点。

## 哪些结果还不能下结论

- 不能把旧 v400 事件锚点当成最终真相。
- 不能把 raw dynamic onset 当作无泄漏事件触发锚点，因为它来自车辆响应本身。
- 不能说生理数据有效；这里只是记录生理/脑电窗口是否覆盖。
- 不能直接训练最终模型；阶段 3 只能从强车辆基线和保守样本子集开始。

## 下一阶段是否可以继续

可以继续到阶段 3 的准备工作，但要分两条线：

1. 用 `raw_road_curvature_onset` 做低泄漏保守车辆基线。
2. 用 old v400 和 raw dynamic 做历史对照/上限分析，不能混作主结论。

## 推荐优先查看

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/anchor_source_inventory.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/split_table.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_v0_2_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/processed_vehicle_windows_v0_2_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/figures/stage02_candidate_counts_by_source.png`
