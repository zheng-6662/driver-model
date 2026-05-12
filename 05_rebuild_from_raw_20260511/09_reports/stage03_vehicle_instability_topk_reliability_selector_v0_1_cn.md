# 阶段 3 技术报告：top-K 可靠性选择/回退 v0.1

## 范围

- 轨道：`B_response3s_strict_core`。
- 输入：事件前车辆历史、因果可得道路/事件上下文、候选轨迹自身的预测形态、top-K 分支概率。
- 不使用：subject ID、生理、脑电、连续风格、服务器、服务器密码文件。
- 训练协议：选择器仅 train 拟合；回退阈值仅 val 固定；test 只报告。

## validation 选择

| selector_model | val_rmse | test_rmse | test_wrong_side_rate | test_large_response_recall | test_difficult_top20_rmse |
| --- | --- | --- | --- | --- | --- |
| topk_top1_rbf_fallback_logreg_no_subject | 0.571482 | 0.542071 | 0.225000 | 0.750000 | 0.678907 |
| topk_rbf_branch_logreg_selector_no_subject | 0.603475 | 0.576630 | 0.150000 | 0.625000 | 0.640533 |
| topk_branch_logreg_selector_no_subject | 0.672280 | 0.610735 | 0.125000 | 0.625000 | 0.769374 |

## test 指标

| 模型 | RMSE | 错侧率 | 大幅召回 | 困难 top20 RMSE |
|---|---:|---:|---:|---:|
| RBF | 0.533667 | 0.225 | 0.750 | 0.678907 |
| top-1 | 0.587865 | 0.100 | 0.750 | 0.717094 |
| branch selector | 0.610735 | 0.125 | 0.625 | 0.769374 |
| candidate selector | 0.576630 | 0.150 | 0.625 | 0.640533 |
| top1-RBF fallback | 0.542071 | 0.225 | 0.750 | 0.678907 |
| best-of-3 oracle | 0.477526 | 0.025 | 0.875 | 0.634182 |
| best-of-RBF+topK oracle | 0.415652 | 0.075 | 0.875 | 0.604369 |

## 选择器信息

- validation 选中策略：`topk_top1_rbf_fallback_logreg_no_subject`。
- top1-RBF fallback 阈值：0.05。
- branch selector 特征数：31。
- candidate/fallback 特征数：34。

## 结论

本轮用于判断 top-K 的问题是不是“候选覆盖有潜力但选择机制不足”。本轮可部署选择策略结论：validation 选中的策略没有超过 RBF，test RMSE 比 RBF 高 0.008405；本轮不能升级为强车辆基线。 oracle 只能说明上限空间，不能作为结论性能。
