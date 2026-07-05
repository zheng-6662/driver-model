# GPTPro 外部审查请求：生理数据如何真正补足锚点前车辆信息不足

你现在作为外部审查者，请只基于下面的本地实验事实给下一步方法建议。请不要建议回到 v222a gate、删除样本、轻量 residual、单纯锚点后移，也不要把任务改成失败机制论文。目标是预测驾驶行为轨迹，并让当前差样本有本质改善。

## 当前任务

- 目标：根据事件锚点前后的可用信息，预测后续 steering 轨迹。
- 当前主要矛盾：锚点前车辆历史序列中可分辨信息不足，很多样本前段非常相似，但锚点后行为差异很大。
- 用户希望引入生理状态和驾驶风格：生理数据不是简单拼接，而应提供驾驶员/状态层面的区别信息。

## 已做路线摘要

- v241：较好的基础连续轨迹预测模型。
- v247 左右：细粒度重锚定和 best anchor 对差样本有改善，但用户看图后判断锚点不是主要问题，性能提升不足。
- v276：生理辅助 candidate gain model。test diagnostic 有少量空间，但 validation 选不出可部署规则。
- v277：加入驾驶风格和生理状态特征。没有带来稳定提升。
- v278：listwise candidate rank loss。vehicle-only test diagnostic bad_top10 为 0.6832，比 fixed wait-latest 0.6950 好一点；但 bio 组没有赢过 vehicle-only，val 仍选不出可部署提升。
- v279：不让生理直接选轨迹，而是用生理判断 v278 vehicle listwise top candidate 是否可信。test diagnostic 到 0.6791，但不是 deployable；生理可靠性没有赢过 vehicle 可靠性。可靠性模型在 val 上明显过度乐观。
- v280：修正 v279 的 in-sample top 偏差，train top 改为 recording-group OOF listwise ranker，val/test 用 full-train ranker。结果 diagnostic 退到 0.6891，仍不是 deployable，生理仍不赢 vehicle。

## 关键数字

- fixed wait-latest test bad_top10：0.695048。
- v278 best test diagnostic：0.683215；deployable best 仍 0.695048。
- v279 best test diagnostic：0.679116；deployable best 仍 0.695048；bio_beats_vehicle_reliability=False。
- v280 best test diagnostic：0.689064；deployable best 仍 0.695048；bio_beats_vehicle_reliability=False。
- v279 第一候选真实收益率：
  - train good_rate 0.588
  - val good_rate 0.246
  - test good_rate 0.255
- v280 OOF 后：
  - train good_rate 0.591
  - val good_rate 0.223
  - test good_rate 0.255
- v280 可靠性模型 val_gain_corr：
  - vehicle 0.032
  - vehicle+bio_pair -0.019
  - vehicle+bio_state -0.064
  - vehicle+style+bio_state -0.093

## 当前判断

1. 锚点不是主要矛盾。
2. 只靠车辆候选选择和阈值过滤，提升很小且无法通过 validation 部署。
3. 生理作为 pair distance、event state、candidate reliability filter 都没有体现稳定增量。
4. 当前可能需要重新设计“生理数据进入模型的方式”，而不是继续在候选过滤上调阈值。

## 希望你回答

请给出一个专业、可执行的下一步建模方案，重点回答：

1. 如果生理数据确实要提供“驾驶员状态/反应倾向”信息，它应该以什么任务形式进入模型？
2. 现在应不应该从 candidate/rerank/reliability filter 转向 latent state / representation learning / multimodal sequence fusion？
3. 在样本量只有约 1167 个事件、train/val/test 存在明显分布差异时，如何避免生理特征过拟合？
4. 给出 2 到 3 个优先级最高的实验版本，要求能用现有数据落地，并说明每个实验的输入、输出、损失函数、验证方式和失败判据。
5. 哪些方向应该明确停止？

请用中文回答，尽量具体到可以马上写代码实现。
