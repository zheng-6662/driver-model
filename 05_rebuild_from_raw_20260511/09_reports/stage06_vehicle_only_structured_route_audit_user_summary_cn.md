# 阶段 6 用户查看版：车辆-only结构化路线审计 v0.1

## 这个阶段为什么做

阶段 4 已经说明当前连续风格路线不能升级主线，生理/EEG 也还不能进入有效性结论。因此现在必须先把车辆-only结构化路线重新收口：哪些车辆模型能作为主参照，哪些只是失败/诊断候选，下一步应该继续哪条结构化路线。

## 这个阶段检查了什么

- 只检查 B 轨道 270 个严格核心 3 秒响应样本上的已有车辆-only结果。
- 汇总 RBF、direct Transformer、响应分解 Transformer、关键点+残差、多假设/top-K、选择器/可靠性候选。
- 所有比较都以当前 RBF KRR 为参照。
- 没有使用生理、脑电、连续风格、驾驶员 ID、服务器或服务器密码文件。

## 目前发现了什么

- RBF 仍是当前最稳的车辆-only主参照：test RMSE=0.533667，错侧率=0.225，大幅响应召回=0.750。
- 响应分解 Transformer v0.1 不能升级：test RMSE=0.602174，比 RBF 差 +0.068507，大幅响应召回和尾段也更差。
- `selector_logreg_rbf_keypoint_no_subject` 是弱候选：RMSE 基本持平，方向/大幅响应/困难样本有一些改善，但尾段和零线穿越等指标仍退化，不能直接定为主线。
- oracle/best-of-K 上限很强：最佳 oracle RMSE=0.415652，说明多候选空间有潜力，但 oracle 不是可部署方法，不能当最终结论。

## 哪些结果可信

可信的是已有车辆-only候选在同一 B 轨道 test 集上的相对表现，以及“当前结构化 Transformer v0.1 不该升级主线”这个判断。

## 哪些还不能下结论

不能说车辆-only问题已经解决；也不能说生理/EEG有效。当前只能说，多假设/关键点选择器方向有研究信号，但还没有形成稳定可部署选择策略。

## 下一阶段是否可以继续

可以继续，但下一步应是 Stage 6b：RBF + 关键点/多假设候选的可部署选择器、可靠性门控和坏样本复盘，而不是直接加入生理/EEG。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_route_gate_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_candidate_scorecard.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/figures/vehicle_structured_route_rmse_summary.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/figures/vehicle_structured_route_delta_vs_rbf.png`
