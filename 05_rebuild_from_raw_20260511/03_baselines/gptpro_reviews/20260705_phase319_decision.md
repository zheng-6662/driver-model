# 本地高级模型第319版决策记录

- 提问文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase319_prompt.md`
- 回复文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase319_response.md`
- 说明：浏览器自动抓取曾误抓视频页内容，最终以用户粘贴的完整回复文本为准。

## 采纳内容

- 第319版固定为“双通道配额激活门控”，不重训第316版主预测模型，不扩第317版候选库。
- 搜索目标从“安全项通过数”改为“最低激活约束 + 分组风险预算 + 加权收益目标”。
- 明确禁止全不改：整体、强方向盘、困难代理样本必须有最低校正覆盖率。
- 拆成普通样本保护通道和强方向盘/困难代理激活通道。
- 普通样本只设置校正上限，不设置最低校正率。
- 强方向盘和困难代理样本使用较松门槛与固定比例候选激活。
- 引入困难代理分数，用部署时可用特征预测第316版容易错、候选库又有潜力的样本。
- 保留残差融合，但允许强方向盘/困难通道在高置信时使用全量候选，普通通道只允许半融合或不改。

## 暂缓内容

- 暂不优先修第318版第一段或第二段模型本身，因为第318版验证集一个样本都没改，还没有进入“候选选得准不准”的阶段。
- 暂不扩候选库，因为第317版和第318版都显示候选最优上限仍然明显存在。
- 暂不使用验证集或测试集真实困难标签作为门控输入；困难前20和困难前10只作为训练监督、内部搜索约束和验证评价。

## 本地证据

- 第318版脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v318_conservative_two_stage_gate_20260705.py`
- 第318版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v318_conservative_two_stage_gate_20260705\reports\v318_conservative_two_stage_gate_cn.md`
- 第318版验证门槛表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v318_conservative_two_stage_gate_20260705\tables\v318_validation_gate_check.csv`
- 第318版阈值表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v318_conservative_two_stage_gate_20260705\tables\v318_selected_policy_thresholds.csv`
