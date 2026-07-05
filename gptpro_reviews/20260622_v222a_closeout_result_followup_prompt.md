刚才思考停止了。请基于上一条 v222a closeout 结果，直接给下一轮 bounded 指令。

请只回答以下内容：

1. 是否接受 closeout 诊断：
   - 当前主要失败是否是 selector/gate 泛化不稳，而不是 candidate pool 大面积缺曲线？
2. 下一步唯一允许 Codex 执行的 bounded local step 是什么？
3. 需要产出的 exact files。
4. pass/fail criteria。
5. stop conditions。
6. 必须避免做什么。

关键事实摘要：
- loose baseline `avg_joint_focus` locked test RMSE/tail = `0.544884 / 0.629752`。
- strict baseline `peak_floor_090` locked test RMSE/tail = `0.571770 / 0.658306`。
- v222a noharm/gate: loose locked test fail，RMSE/tail 变差；strict RMSE/tail 安全但 under 变差。
- combined selector_failed_rate = `0.410615`。
- combined candidate_missing_rate = `0.027933`。
- high-tail candidate_missing_rate = `0.126582`。
- high-tail oracle clear gain rate = `0.911392`。
- future_route_decision: `v222b_allowed=False`, `v223_allowed=False`。
- closeout pack/report/script/zip/leakage/forbidden scan 均已通过本地验证。

限制：
- 不要要求 test retuning、新 tau、v222a_gate_v2、multi-router、v222b/v223 训练。
- 不要把 oracle/true-label/diagnostic-only row 写入 formal leaderboard/gate/usage/selected configs。
- 如果下一步只是 audit/report/route-reconstruction，也请明确边界和验收条件。
