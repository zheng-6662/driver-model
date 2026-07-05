# 本地高级模型决策记录

- 回复文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase320_response.md`
- 提问文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase320_prompt.md`
- 浏览器截图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase320_after_reply_20260705_123514.png`

## 接受

- 接受“第319版仍然全不改时，应改成排序配额选择，绝对阈值只拦截极端风险”的建议。
- 接受“候选选择不能继续只依赖候选收益回归绝对值，应加入候选正收益概率/排序信号”的建议。
- 接受“普通样本保护和强方向盘/困难代理激活分开处理”的建议。
- 已落地为第320版：`stage03_v320_rank_budget_repair_gate_20260705.py`。

## 暂缓或拒绝

- 暂不扩候选库，也暂不重训第316版主预测模型。
- 暂不把困难前20/困难前10作为验证或部署门控输入；它们仍只用于训练监督和验证/测试后评价。
- 暂不把第320版写成最终解决方案，因为虽然验证预算通过，测试困难前20和困难前10仍有小幅负收益。

## 本地证据

- 第319版失败证据：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v319_dual_channel_quota_gate_20260705\reports\v319_dual_channel_quota_gate_cn.md`
- 第319版训练折外搜索：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v319_dual_channel_quota_gate_20260705\tables\v319_train_oof_quota_search.csv`
- 第320版脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v320_rank_budget_repair_gate_20260705.py`
- 第320版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v320_rank_budget_repair_gate_20260705\reports\v320_rank_budget_repair_gate_cn.md`
- 第320版守卫：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v320_rank_budget_repair_gate_20260705\logs\guardrail_check.json`
