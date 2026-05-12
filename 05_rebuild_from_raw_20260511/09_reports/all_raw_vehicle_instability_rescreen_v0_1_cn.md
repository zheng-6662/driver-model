# 全部原始车辆数据失稳样本重筛 v0.1

生成时间：2026-05-12 16:25:03

## 为什么做

用户希望按“道路设定引导 + 原始车辆动态证据”的标准，对所有原始数据重新筛选样本，而不是只在已有候选表上继续人工标注。

本版本直接从 `F:/data_set_process/data_process/01_datasets/数据预处理/原始车辆数据/<被试名>/*.csv` 读取 91 个原始车辆 CSV，重新扫描 `ay` 和 `roll_rate` 非方向盘动态异常，再叠加旧 v400 事件上下文和道路模块先验进行判定。

## 筛选原则

- 主锚点只来自非方向盘车辆动态：`ay` 和 `roll_rate`。
- `steer_rate` 不作为失稳锚点，方向盘只作为事件后响应证据。
- 弯道只作为上下文，不等于失稳。
- `mu1/differentmu_road`、`fix_road`、`stop`、`zd` 等道路模块只作为场景先验，不能单独确认失稳。
- 旧 `events_v400_context` 只作为旧事件上下文，不作为新真值。

## 当前结果

原始车辆 CSV 数：91

可读取车辆 CSV 数：91

重筛候选总数：1991

自动/已确认采用：1348

高置信主清单：908

中间复核：269

低证据剔除：374

按最终建议统计：

```text
road_guided_recommended_decision
hybrid_accept_high                  885
hybrid_accept_medium                440
hybrid_reject_low_evidence          374
hybrid_review_conflict_or_medium    269
manual_confirmed_accept              23
```

按被试候选数统计：

```text
subject
byx    143
cwh    100
gf      86
gzj    156
hzh    179
jy      93
lx      59
lxy     93
rjy    116
txj    135
tyy    102
xst     26
yyl    106
yzy    128
zdq     87
zt      28
zx     236
zxy    118
```

文件读取状态：

```text
read_status
ok    91
```

## 输出

- 全量候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_candidates_v0_1.csv`
- 自动采用：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_auto_accepted_v0_1.csv`
- 高置信主清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_primary_high_confidence_v0_1.csv`
- 中间复核：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_review_queue_v0_1.csv`
- 低证据剔除：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_rejected_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_summary_v0_1.csv`
- 文件状态：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_rescreen_file_status_v0_1.csv`
- 道路模块交叉表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_module_summary_v0_1.csv`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_all_raw_vehicle_instability_rescreen_v0_1_cn.md`

## 下一步

下一步应该用自动采用表生成正式车辆失稳版 `samples_master` 和处理后车辆窗口。之前的 404 个弯道样本、以及旧的道路曲率阶段 3 模型，仍然只能作为历史诊断材料。
