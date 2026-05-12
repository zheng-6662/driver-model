# 阶段 4 用户查看版总结：连续驾驶风格协议与候选特征 v0.1

生成时间：2026-05-13 05:05:54

## 这个阶段为什么做

阶段 3 已经把 RBF/KRR 车辆-only 模型固定成“有限主参照”。它可以作为后续比较底线，但错侧、反向修正和困难样本还没有解决。所以阶段 4 不能直接说连续风格有效，必须先把风格特征的来源、泄漏边界、标准化方式和置乱对照讲清楚。

## 这个阶段检查了什么

本轮只处理车辆原始数据中的事件前连续历史。主规则是：风格窗口最晚只能到事件锚点前 3 秒，也就是排除 `[-3, 0]` 的直接车辆输入窗口，并完全不接触 `[0, 3]` 的方向盘响应标签窗口。

同时生成了：

- 候选风格特征表；
- train-only 标准化参数；
- 道路/被试耦合审计；
- 被试内、跨被试、跨 session、道路平衡置乱和驾驶员 ID 对照协议。

## 目前发现了什么

本轮纳入 B 轨道严格核心样本 270 个。候选窗口可用性如下：

- `last120_guard3`：267/270 可用，比例 98.9%
- `last30_guard3`：270/270 可用，比例 100.0%
- `last60_guard3`：268/270 可用，比例 99.3%
- `prefix_until_guard3`：268/270 可用，比例 99.3%

按 session-level split 看：

- `test`：样本 40，被试 8，session 12，至少一个风格窗口可用 40 (100.0%)
- `train`：样本 188，被试 18，session 54，至少一个风格窗口可用 188 (100.0%)
- `val`：样本 42，被试 7，session 10，至少一个风格窗口可用 42 (100.0%)

train-only 标准化已准备：436/440 个数值特征有可用训练集均值和标准差。

## 哪些结果可信

可信的是“处理规则”和“候选特征表”：这些特征只来自事件前 3 秒以前的原始车辆历史；标准化参数只从训练集拟合；脚本没有读取服务器密码，也没有使用生理或脑电数据。

## 哪些结果还不能下结论

现在还不能说连续驾驶风格有效。原因是还没有把这些风格特征接入固定 RBF 参照后的模型，也没有完成置乱对照、分被试验证、分道路验证和物理错误指标比较。

## 下一阶段是否可以继续

可以继续做阶段 4 的探索性验证，但只能按固定 RBF 主参照比较，并必须报告置乱对照和物理指标。生理/脑电仍然不能进入有效性结论。

## 推荐优先查看

- `04_style/stage04_continuous_style_protocol_v0_1/tables/style_protocol_gate_table.csv`
- `04_style/stage04_continuous_style_protocol_v0_1/tables/style_feature_candidate_wide.csv`
- `04_style/stage04_continuous_style_protocol_v0_1/tables/style_train_only_scaler_session_split.csv`
- `04_style/stage04_continuous_style_protocol_v0_1/figures/style_feature_availability_by_window.png`
- `04_style/stage04_continuous_style_protocol_v0_1/figures/style_subject_road_coupling_heatmap.png`
