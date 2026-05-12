# R2E-Steering 项目总进度看板

更新时间：2026-05-12 12:02:38

## 当前阶段

阶段 3：低泄漏道路曲率子集的无学习基线、v0.2 ridge、v0.3 无被试 ID 车辆基线、v0.4 RBF KRR 候选强车辆模型卡已完成。当前仍处在阶段 3 内，需要先复核 v0.4 轨迹图、长窗口和结构化物理错误，再决定是否进入更强车辆时序/响应分解模型。

## 当前正在做什么

核验阶段 3 v0.4 模型卡和轨迹图，准备把 `rbf_krr_vehicle_no_subject` 作为后续风格/生理增量验证前的当前强车辆参照候选。

## 已完成什么

- 阶段 0 旧流程冻结说明已生成。
- 新流程目录结构已建立。
- 三个原始目录下被试名文件夹内 CSV 清单和哈希已生成。
- 原始车辆/生理/脑电深度审计表已生成。
- 阶段 1 用户查看版中文总结已生成。
- 阶段 0/1 完成审计清单已生成。
- 阶段 2 候选事件清单、样本清单、split 表、道路设计清单和数据版本卡已生成。
- 低泄漏道路曲率候选的处理后车辆窗口 v0.2 已生成，原始 CSV 未修改。
- 阶段 3 无学习基线已完成：零响应、保持当前值、250ms 历史趋势外推、训练集平均轨迹、同类事件训练集平均轨迹。
- 阶段 3 纯车辆 ridge 基线已完成：只使用车辆历史窗口统计特征和事件元信息；标准化与 alpha 选择只在 train/val 内完成。
- 阶段 3 指标表、固定预测图、坏样本图和用户查看版总结已生成。
- 阶段 3 v0.3 方法修正已完成：发现 v0.2 `ridge_vehicle_summary` 包含 `subject` one-hot，现已降级为驾驶员 ID 控制参考；新增无被试 ID 的 ridge、kNN、RBF KRR 纯车辆基线。
- 阶段 3 v0.3 坏样本诊断表、错误桶、小样本过拟合测试和模型对照图已生成。
- 阶段 3 v0.4 RBF KRR 候选模型卡已生成：固定样本轨迹图、坏样本轨迹图、pre3 长窗口轨迹图、分被试表、分响应组表和中文解释。

## 正在运行什么任务

当前没有后台审计或训练任务在运行。

## 服务器是否在运行

阶段 3 基线评估在本地完成；未使用服务器；未读取服务器密码文件。服务器状态未主动检查。

## 最近一次结果

- 阶段 1 纳入审计 CSV：258；车辆/生理/脑电：91/82/85
- 候选事件：11619，其中 old v400 6247、raw road curvature 359、raw vehicle dynamic 5013
- `samples_master.csv/jsonl` 行数：46476
- 道路设计记录：49 个文件，其中多个 CSV 含 curvature/kappa 信息
- 低泄漏道路曲率候选处理后车辆窗口：3 个 NPZ，样本数均为 359，特征数 9
- 处理窗口形状：pre1 输入 `(359,201,9)` 标签 `(359,401)`；pre2 输入 `(359,401,9)` 标签 `(359,401)`；pre3 输入 `(359,601,9)` 标签 `(359,601)`
- 阶段 3 v0.2 指标行数：162；逐样本指标行数：19386；ridge 模型信息行数：9
- 阶段 3 v0.3 新增指标行数：81；逐样本指标行数：9693；新模型包括 `ridge_vehicle_no_subject`、`knn_vehicle_no_subject`、`rbf_krr_vehicle_no_subject`
- pre2 + session-level test 当前最好的无被试 ID 纯车辆模型为 `rbf_krr_vehicle_no_subject`：RMSE 0.382337，主峰方向准确率 0.820896，错侧率 0.179104，严重幅值不足率 0.283582，困难样本 top20 RMSE 0.642092
- pre3 + session-level test 的 `rbf_krr_vehicle_no_subject`：RMSE 0.466957，主峰方向准确率 0.791045，错侧率 0.208955，严重幅值不足率 0.358209，困难样本 top20 RMSE 0.832563，反向修正 exact rate 0.0
- v0.2 `ridge_vehicle_summary` 包含 subject one-hot，RMSE 0.422204，只能作为驾驶员 ID 控制参考，不再作为最终纯车辆主结论。
- 小样本过拟合测试中，RBF KRR 在 8/16/32/64/128 个训练峰值样本上的子集训练 RMSE 均约 0.000002，但测试 RMSE 仍为 0.511289 到 0.447567，提示容量能记住小样本，主要风险在泛化、输入信息和事件覆盖。
- v0.4 模型卡新增 4 张轨迹图、2 行候选模型汇总、20 行分被试表、34 行响应组表、48 个画图样本记录。

## 当前最大风险

old v400 仍只能作历史参考；raw vehicle dynamic 锚点来自车辆响应，存在响应结果泄漏风险；raw road curvature 候选较低泄漏但只覆盖 359 个道路曲率事件，不能代表全部事件类型。阶段 3 v0.4 显示 RBF KRR 可以作为当前强车辆参照候选，但 pre3 长窗口、反向修正 exact rate 和大幅响应幅值不足仍未解决，不能据此进入风格/生理有效性结论。

## 下一步准备做什么

1. 人工复核 v0.4 固定轨迹图和坏样本轨迹图，确认指标能否解释具体物理错误。
2. 补充响应关键点/分解车辆模型，重点处理 pre3 长窗口、反向修正和大幅响应幅值不足。
3. 继续审查道路设计文件能否进一步提供更精确、低泄漏的道路事件锚点。
4. 在强车辆基线稳定之前，不进入连续风格或生理有效性验证。

## 用户可以优先查看哪些文件

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_user_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_v0_2_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/processed_vehicle_windows_v0_2_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_user_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_baseline_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_diagnostics_v0_3_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_rbf_krr_candidate_model_card_v0_4_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre2_session_v0_4.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre2_session_v0_4.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_vehicle_model_comparison_v0_3.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_model_rmse_comparison_v0_3.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_baseline_metrics.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_fixed_predictions_pre2_session_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_bad_samples_pre2_session_test_ridge.png`
