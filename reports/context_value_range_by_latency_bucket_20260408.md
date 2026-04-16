# Context / Anchor Signal Value Range Analysis

## Scope
- Manifest: `F:\data_set_process\data_process\datasetprocess\final_code\model\training\protocol_allphase_control_v2_context_full2s\sample_manifest.csv`
- Attribution table: `F:\data_set_process\data_process\reports\attribution_master_table.csv`
- Test samples analyzed: `749`
- Unique vehicle files loaded: `15`
- Anchor signals analyzed as absolute magnitudes for value-range comparison: `abs(steer)`, `abs(steer_rate)`, `abs(ay)`, `abs(yawrate)`.
- Raw signed values were still extracted from vehicle CSVs to keep the anchor reading traceable to source rows.

## Extraction QC
- Extraction method counts: anchor_idx=749
- Absolute anchor-time error: mean `0.000000` s, median `0.000000` s, max `0.000000` s.

## Sample Counts By Latency Bucket
| latency_proxy_bucket | n_samples |
| --- | --- |
| Q2 | 368 |
| Q1_fast | 188 |
| Q4_slow | 183 |
| Q3 | 10 |

## Latency Bucket Mean Comparison
| signal | Q1_fast | Q2 | Q3 | Q4_slow |
| --- | --- | --- | --- | --- |
| abs(ay) | 1.0566 | 1.7104 | 2.9274 | 1.8928 |
| abs(steer) | 0.4238 | 0.4832 | 0.6529 | 0.4979 |
| abs(steer_rate) | 2.7040 | 2.5479 | 2.7332 | 2.6111 |
| abs(yawrate) | 0.0517 | 0.0744 | 0.1328 | 0.0865 |

## Q1_fast vs non_Q1_fast
| signal | q1_n | non_q1_n | q1_mean | non_q1_mean | mean_diff_q1_minus_non_q1 | q1_median | non_q1_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abs(steer) | 188 | 561 | 0.4238 | 0.4910 | -0.0672 | 0.2060 | 0.3084 |
| abs(steer_rate) | 188 | 561 | 2.7040 | 2.5718 | 0.1322 | 2.2253 | 2.2166 |
| abs(ay) | 188 | 561 | 1.0566 | 1.7916 | -0.7351 | 0.4581 | 1.1148 |
| abs(yawrate) | 188 | 561 | 0.0517 | 0.0794 | -0.0277 | 0.0235 | 0.0433 |

### Direct Findings
- `abs(steer)`: Q1_fast mean 0.4238 vs non_Q1_fast 0.4910 (lower by -0.0672).
- `abs(steer_rate)`: Q1_fast mean 2.7040 vs non_Q1_fast 2.5718 (higher by 0.1322).
- `abs(ay)`: Q1_fast mean 1.0566 vs non_Q1_fast 1.7916 (lower by -0.7351).
- `abs(yawrate)`: Q1_fast mean 0.0517 vs non_Q1_fast 0.0794 (lower by -0.0277).

## Secondary Grouping: Latency Bucket x Morphology
| latency_proxy_bucket | eval_morphology_label | signal | n_samples | mean | median |
| --- | --- | --- | --- | --- | --- |
| Q1_fast | multi_correction | abs(steer) | 91 | 0.4061 | 0.2028 |
| Q1_fast | multi_correction | abs(steer_rate) | 91 | 2.5702 | 2.2340 |
| Q1_fast | multi_correction | abs(ay) | 91 | 0.9633 | 0.4137 |
| Q1_fast | multi_correction | abs(yawrate) | 91 | 0.0496 | 0.0188 |
| Q1_fast | reverse_correction | abs(steer) | 71 | 0.3942 | 0.1683 |
| Q1_fast | reverse_correction | abs(steer_rate) | 71 | 2.8048 | 2.2166 |
| Q1_fast | reverse_correction | abs(ay) | 71 | 0.9176 | 0.4280 |
| Q1_fast | reverse_correction | abs(yawrate) | 71 | 0.0460 | 0.0228 |
| Q1_fast | single_lobe | abs(steer) | 26 | 0.5667 | 0.3760 |
| Q1_fast | single_lobe | abs(steer_rate) | 26 | 2.8972 | 2.2253 |
| Q1_fast | single_lobe | abs(ay) | 26 | 1.7624 | 1.2710 |
| Q1_fast | single_lobe | abs(yawrate) | 26 | 0.0747 | 0.0372 |
| Q2 | multi_correction | abs(steer) | 201 | 0.4393 | 0.2665 |
| Q2 | multi_correction | abs(steer_rate) | 201 | 2.5906 | 2.2689 |
| Q2 | multi_correction | abs(ay) | 201 | 1.3918 | 0.9062 |
| Q2 | multi_correction | abs(yawrate) | 201 | 0.0630 | 0.0340 |
| Q2 | reverse_correction | abs(steer) | 112 | 0.4597 | 0.3062 |
| Q2 | reverse_correction | abs(steer_rate) | 112 | 2.6283 | 2.1991 |
| Q2 | reverse_correction | abs(ay) | 112 | 1.9373 | 1.2839 |
| Q2 | reverse_correction | abs(yawrate) | 112 | 0.0835 | 0.0509 |
| Q2 | single_lobe | abs(steer) | 55 | 0.6916 | 0.5116 |
| Q2 | single_lobe | abs(steer_rate) | 55 | 2.2280 | 2.0770 |
| Q2 | single_lobe | abs(ay) | 55 | 2.4126 | 1.4253 |
| Q2 | single_lobe | abs(yawrate) | 55 | 0.0975 | 0.0552 |
| Q3 | multi_correction | abs(steer) | 6 | 0.5208 | 0.3647 |
| Q3 | multi_correction | abs(steer_rate) | 6 | 2.8042 | 2.3474 |
| Q3 | multi_correction | abs(ay) | 6 | 2.3038 | 2.0342 |
| Q3 | multi_correction | abs(yawrate) | 6 | 0.0920 | 0.0706 |
| Q3 | reverse_correction | abs(steer) | 2 | 0.0463 | 0.0463 |
| Q3 | reverse_correction | abs(steer_rate) | 2 | 2.3213 | 2.3213 |
| Q3 | reverse_correction | abs(ay) | 2 | 1.6910 | 1.6910 |
| Q3 | reverse_correction | abs(yawrate) | 2 | 0.0583 | 0.0583 |
| Q3 | single_lobe | abs(steer) | 2 | 1.6558 | 1.6558 |
| Q3 | single_lobe | abs(steer_rate) | 2 | 2.9320 | 2.9320 |
| Q3 | single_lobe | abs(ay) | 2 | 6.0344 | 6.0344 |
| Q3 | single_lobe | abs(yawrate) | 2 | 0.3297 | 0.3297 |
| Q4_slow | multi_correction | abs(steer) | 93 | 0.4080 | 0.2937 |
| Q4_slow | multi_correction | abs(steer_rate) | 93 | 2.5904 | 2.1820 |
| Q4_slow | multi_correction | abs(ay) | 93 | 1.4669 | 1.0056 |
| Q4_slow | multi_correction | abs(yawrate) | 93 | 0.0660 | 0.0385 |
| Q4_slow | reverse_correction | abs(steer) | 62 | 0.4857 | 0.3111 |
| Q4_slow | reverse_correction | abs(steer_rate) | 62 | 2.7573 | 2.3038 |
| Q4_slow | reverse_correction | abs(ay) | 62 | 1.9667 | 1.2024 |
| Q4_slow | reverse_correction | abs(yawrate) | 62 | 0.0985 | 0.0414 |
| Q4_slow | single_lobe | abs(steer) | 28 | 0.8236 | 0.4903 |
| Q4_slow | single_lobe | abs(steer_rate) | 28 | 2.3562 | 2.0944 |
| Q4_slow | single_lobe | abs(ay) | 28 | 3.1438 | 2.0126 |
| Q4_slow | single_lobe | abs(yawrate) | 28 | 0.1285 | 0.0680 |

## Pearson Correlation With delta_rmse_tail_abs_steer
| subset | signal | n_samples | pearson_r |
| --- | --- | --- | --- |
| all_test | abs(steer) | 749 | 0.0551 |
| all_test | abs(steer_rate) | 749 | -0.1304 |
| all_test | abs(ay) | 749 | -0.0879 |
| all_test | abs(yawrate) | 749 | -0.0655 |
| Q1_fast_only | abs(steer) | 188 | 0.2759 |
| Q1_fast_only | abs(steer_rate) | 188 | -0.1846 |
| Q1_fast_only | abs(ay) | 188 | -0.0465 |
| Q1_fast_only | abs(yawrate) | 188 | 0.0912 |

### Correlation Ranking (All Test Samples)
- `abs(steer)`: Pearson r = 0.0551 on 749 test samples.
- `abs(yawrate)`: Pearson r = -0.0655 on 749 test samples.
- `abs(ay)`: Pearson r = -0.0879 on 749 test samples.
- `abs(steer_rate)`: Pearson r = -0.1304 on 749 test samples.

### Correlation Ranking (Q1_fast Only)
- `abs(steer)`: Pearson r = 0.2759 on 188 Q1_fast samples.
- `abs(yawrate)`: Pearson r = 0.0912 on 188 Q1_fast samples.
- `abs(ay)`: Pearson r = -0.0465 on 188 Q1_fast samples.
- `abs(steer_rate)`: Pearson r = -0.1846 on 188 Q1_fast samples.

## Key Takeaways
- The broad hypothesis is not supported as a uniform pattern: only `abs(steer_rate)` is higher in `Q1_fast` than `non_Q1_fast` (+0.1322 by mean), while `abs(ay)` shows the largest negative gap (-0.7351).
- Across all 749 test samples, anchor-signal correlations with `delta_rmse_tail_abs_steer` are weak overall (max |r| = 0.1304).
- Within the 188 `Q1_fast` samples, the strongest single relationship is `abs(steer)` with |r| = 0.2759, which is still only moderate.
- Current evidence therefore supports, at most, a narrow `steer_rate`-intensity difference rather than a consistent four-signal anchor-value elevation in `Q1_fast`.

## Interpretation
- If `Q1_fast` rows are consistently higher across these anchor magnitudes, that supports the hypothesis that stronger anchor-state context is part of the tail mismatch mechanism.
- If correlations remain weak even when Q1_fast group means are elevated, then anchor magnitude alone is likely insufficient and the remaining explanation shifts toward conditioning structure or downstream temporal broadcast effects.
- The CSV output contains the full `latency_proxy_bucket`, `Q1_fast vs non_Q1_fast`, `latency_proxy_bucket x eval_morphology_label`, and Pearson-correlation tables in a single long-format file.
