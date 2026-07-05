# v236 Rolling Reanchor Joint Prediction 报告

## 结论边界

- 本轮是新的 rolling/reanchor 训练任务，不再继续 v222a gate、删除样本或 light residual 路线。
- 每个事件生成 0/200/400/600/800/1000ms 多个 observation time；同一 event_uid 的所有 delay 保持在同一 split。
- 小基线是 joint Ridge，多输出预测未来 2 秒 steering delta、steering rate、roll delta、roll rate、ay、yaw rate。
- alpha 只按 validation 选择，test 只在模型固定后报告。

## 数据集

- rolling 样本数：7002
- 唯一事件数：1167
- observe_later_like rolling 样本数：726
- strict subset rolling 样本数：5778

## 模型选择

- selected alpha=`1000`，val score=1.299386，val sample RMSE=0.809377，val tail=0.923287

## Test by delay

- delay=0ms: n=184，sample_RMSE=0.641212，tail_mean=0.777846，strong_under=0.596899
- delay=1000ms: n=184，sample_RMSE=0.572341，tail_mean=0.670060，strong_under=0.194175
- delay=200ms: n=184，sample_RMSE=0.572148，tail_mean=0.665324，strong_under=0.246479
- delay=400ms: n=184，sample_RMSE=0.578632，tail_mean=0.661844，strong_under=0.161972
- delay=600ms: n=184，sample_RMSE=0.585781，tail_mean=0.673953，strong_under=0.195122
- delay=800ms: n=184，sample_RMSE=0.540708，tail_mean=0.613473，strong_under=0.250000

## observe_later_like improvement

- delay=0ms: n=27，tail_mean=1.100397，delta_tail_vs_0ms=+0.000000，sample_RMSE=0.891287
- delay=1000ms: n=27，tail_mean=1.668307，delta_tail_vs_0ms=+0.567910，sample_RMSE=1.350375
- delay=200ms: n=27，tail_mean=1.060875，delta_tail_vs_0ms=-0.039522，sample_RMSE=0.865870
- delay=400ms: n=27，tail_mean=1.282036，delta_tail_vs_0ms=+0.181639，sample_RMSE=1.049436
- delay=600ms: n=27，tail_mean=1.294466，delta_tail_vs_0ms=+0.194069，sample_RMSE=1.063482
- delay=800ms: n=27，tail_mean=1.110988，delta_tail_vs_0ms=+0.010591，sample_RMSE=0.987043

## strong event improvement

- delay=0ms: tail_mean=0.961224，strong_under=0.762500，delta_tail_vs_0ms=+0.000000
- delay=1000ms: tail_mean=0.702697，strong_under=0.238806，delta_tail_vs_0ms=-0.258528
- delay=200ms: tail_mean=0.852199，strong_under=0.362500，delta_tail_vs_0ms=-0.109026
- delay=400ms: tail_mean=0.778759，strong_under=0.227848，delta_tail_vs_0ms=-0.182465
- delay=600ms: tail_mean=0.764700，strong_under=0.250000，delta_tail_vs_0ms=-0.196525
- delay=800ms: tail_mean=0.695632，strong_under=0.338462，delta_tail_vs_0ms=-0.265592

## normal no-harm

- delay=0ms: sample_RMSE=0.533926，delta_vs_0ms=+0.000000，status=pass
- delay=1000ms: sample_RMSE=0.392329，delta_vs_0ms=-0.141597，status=pass
- delay=200ms: sample_RMSE=0.450022，delta_vs_0ms=-0.083904，status=pass
- delay=400ms: sample_RMSE=0.428889，delta_vs_0ms=-0.105036，status=pass
- delay=600ms: sample_RMSE=0.446071，delta_vs_0ms=-0.087855，status=pass
- delay=800ms: sample_RMSE=0.425256，delta_vs_0ms=-0.108670，status=pass

## Old 0ms formal reference

- all: old_RMSE=0.468061，v236_0ms_RMSE=0.641212，delta=+0.173150；old_tail=0.522808，v236_tail=0.777846
- observe_later_like: old_RMSE=0.785293，v236_0ms_RMSE=0.891287，delta=+0.105994；old_tail=0.931254，v236_tail=1.100397
- normal_predictable: old_RMSE=0.343042，v236_0ms_RMSE=0.533926，delta=+0.190884；old_tail=0.374754，v236_tail=0.638458
- strong_steer: old_RMSE=0.611697，v236_0ms_RMSE=0.783853，delta=+0.172156；old_tail=0.692971，v236_tail=0.961224

## Guardrail

- guardrail status：`pass`
- 未删除 observe_later_like；未创建 gate/router/selector；未改变 formal headline；未使用 mixed-delay 指标作为正式 headline。

## 输出

- `tables/v236_rolling_sample_manifest.csv`
- `tables/v236_delay_sample_counts.csv`
- `tables/v236_train_val_test_event_split_check.csv`
- `tables/v236_baseline_metrics_by_delay.csv`
- `tables/v236_baseline_metrics_by_delay_and_bucket.csv`
- `tables/v236_observe_later_improvement_curve.csv`
- `tables/v236_strong_event_improvement_curve.csv`
- `tables/v236_normal_sample_noharm_check.csv`
- `tables/v236_metric_vs_old_0ms_formal_reference.csv`
- ZIP：`v236_rolling_reanchor_dataset_and_baseline_pack.zip`
