# Approximate Timestep & Boundary Analysis Using Existing CSVs

## Scope
- Attribution master table: `F:\data_set_process\data_process\reports\attribution_master_table.csv`
- Baseline sample metrics: `F:\data_set_process\data_process\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\baseline_trajectory_sample_metrics.csv`
- Conditioned sample metrics: `F:\data_set_process\data_process\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\conditioned_trajectory_sample_metrics.csv`
- Sample-level comparison: `F:\data_set_process\data_process\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\sample_level_comparison.csv`
- Samples analyzed: `749`
- Latency buckets: Q1_fast, Q2, Q3, Q4_slow

## Input Consistency Check
- The baseline / conditioned sample-metric CSVs were aligned against `sample_level_comparison.csv` on `sample_key` before analysis.
- rmse_pre_tail_abs_steer: baseline max abs diff=0, conditioned max abs diff=0
- rmse_tail_abs_steer: baseline max abs diff=0, conditioned max abs diff=0
- tail_slope_abs_err: baseline max abs diff=0, conditioned max abs diff=0
- boundary_slope_abs_err: baseline max abs diff=0, conditioned max abs diff=0
- boundary_shift_abs_err: baseline max abs diff=0, conditioned max abs diff=0
- peak_abs_amp_err: baseline max abs diff=0, conditioned max abs diff=0
- shape_corr: baseline max abs diff=0, conditioned max abs diff=0

## Key Answers
- `Q1_fast` degradation is **not purely tail-concentrated overall**. Its mean front RMSE delta is `0.0299` while its mean tail RMSE delta is `0.0155`, and the mean tail/front ratio shifts from `1.2623` to `1.2207` (`delta=-0.0417`).
- The strongest tail-focused worsening sits in `Q1_fast x single_lobe`: front delta `0.0263`, tail delta `0.0562`, ratio delta `0.0461`, tail-driven share `0.6538`.
- `single_lobe` boundary worsening looks **time-shift dominant, not slope-flattening dominant**: mean `delta_boundary_shift_abs_err=0.1821` versus mean `delta_boundary_slope_abs_err=0.0244`.
- `reverse_correction` shows the same direction, only weaker: mean `delta_boundary_shift_abs_err=0.1065` versus mean `delta_boundary_slope_abs_err=0.0333`.
- In `Q1_fast x single_lobe`, the worst cases are more **amplitude-driven** than boundary-driven: `corr(delta_tail_rmse, peak_abs_amp_err_conditioned)=0.7195` versus `corr(delta_tail_rmse, boundary_shift_abs_err_conditioned)=-0.1544`.

## Part A: Front vs Tail Error Concentration
| latency_proxy_bucket | n_samples | baseline_front_rmse_mean | baseline_tail_rmse_mean | conditioned_front_rmse_mean | conditioned_tail_rmse_mean | baseline_ratio_mean | conditioned_ratio_mean | delta_ratio_mean | delta_front_rmse_mean | delta_tail_rmse_mean | mean_focus_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1_fast | 188 | 0.3427 | 0.3752 | 0.3726 | 0.3907 | 1.2623 | 1.2207 | -0.0417 | 0.0299 | 0.0155 | mixed-with-front-heavier |
| Q2 | 368 | 0.4031 | 0.3984 | 0.3919 | 0.3711 | 1.2337 | 1.2057 | -0.0280 | -0.0112 | -0.0273 | net-improved-front-heavier |
| Q3 | 10 | 0.6266 | 0.5480 | 0.5377 | 0.4121 | 0.8227 | 0.7676 | -0.0551 | -0.0889 | -0.1359 | net-improved-front-heavier |
| Q4_slow | 183 | 0.4000 | 0.4114 | 0.3768 | 0.3679 | 1.2218 | 1.1659 | -0.0559 | -0.0232 | -0.0435 | net-improved-front-heavier |

### Morphology x Latency Interaction
| eval_morphology_label | latency_proxy_bucket | n_samples | delta_front_rmse_mean | delta_tail_rmse_mean | delta_ratio_mean | tail_driven_share | front_driven_share | mean_focus_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi_correction | Q1_fast | 91 | 0.0665 | 0.0246 | -0.1757 | 0.3516 | 0.6484 | mixed-with-front-heavier |
| multi_correction | Q2 | 201 | 0.0006 | -0.0254 | -0.0788 | 0.4677 | 0.5323 | front-heavier |
| multi_correction | Q3 | 6 | -0.0397 | -0.0458 | 0.0081 | 0.3333 | 0.6667 | net-improved-front-heavier |
| multi_correction | Q4_slow | 93 | -0.0160 | -0.0543 | -0.1410 | 0.3656 | 0.6344 | net-improved-front-heavier |
| reverse_correction | Q1_fast | 71 | -0.0157 | -0.0110 | 0.0980 | 0.5211 | 0.4789 | net-improved-but-tail-relatively-heavier |
| reverse_correction | Q2 | 112 | -0.0297 | -0.0322 | 0.0641 | 0.4554 | 0.5446 | net-improved-front-heavier |
| reverse_correction | Q3 | 2 | -0.0865 | -0.1621 | -0.0738 | 0.5000 | 0.5000 | net-improved-front-heavier |
| reverse_correction | Q4_slow | 62 | -0.0424 | -0.0374 | 0.0451 | 0.3710 | 0.6290 | net-improved-but-tail-relatively-heavier |
| single_lobe | Q1_fast | 26 | 0.0263 | 0.0562 | 0.0461 | 0.6538 | 0.3462 | tail-concentrated |
| single_lobe | Q2 | 55 | -0.0166 | -0.0244 | -0.0297 | 0.4727 | 0.5273 | net-improved-front-heavier |
| single_lobe | Q3 | 2 | -0.2387 | -0.3796 | -0.2260 | 0.0000 | 1.0000 | net-improved-front-heavier |
| single_lobe | Q4_slow | 28 | -0.0047 | -0.0216 | 0.0031 | 0.5000 | 0.5000 | net-improved-front-heavier |

### Interpretation
- `Q1_fast` overall is best described as **mixed with front heavier**, not as a tail-only failure: `delta_front_rmse_mean=0.0299` is larger than `delta_tail_rmse_mean=0.0155`, and the average tail/front ratio declines.
- `Q1_fast x reverse_correction` does not support a worsening story at all on mean RMSE: front delta `-0.0157`, tail delta `-0.0110`.
- The localized tail-worsening story is concentrated in `Q1_fast x single_lobe`, where the tail delta exceeds the front delta by `0.0298` on average.

## Part B: Boundary Smoothing vs Shifting
| eval_morphology_label | n_samples | boundary_slope_baseline_mean | boundary_slope_conditioned_mean | boundary_shift_baseline_mean | boundary_shift_conditioned_mean | delta_boundary_slope_mean | delta_boundary_shift_mean | delta_peak_amp_mean | mechanism_guess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi_correction | 391 | 0.7874 | 0.8167 | 0.7085 | 0.7254 | 0.0293 | 0.0169 | 0.0204 | no clear worsening |
| reverse_correction | 247 | 0.7818 | 0.8151 | 0.6317 | 0.7382 | 0.0333 | 0.1065 | 0.0151 | time-shift dominant |
| single_lobe | 111 | 0.5347 | 0.5591 | 0.4706 | 0.6527 | 0.0244 | 0.1821 | 0.0084 | time-shift dominant |

### Boundary-Mode Counts
| eval_morphology_label | boundary_mode | n_samples | share_within_morphology |
| --- | --- | --- | --- |
| reverse_correction | time-shift | 81 | 0.32793522267206476 |
| reverse_correction | no-clear-worsening | 68 | 0.27530364372469635 |
| reverse_correction | shift-plus-smoothing | 67 | 0.27125506072874495 |
| reverse_correction | smoothing-only | 31 | 0.12550607287449392 |
| single_lobe | shift-plus-smoothing | 43 | 0.38738738738738737 |
| single_lobe | time-shift | 30 | 0.2702702702702703 |
| single_lobe | no-clear-worsening | 25 | 0.22522522522522523 |
| single_lobe | smoothing-only | 13 | 0.11711711711711711 |

### Boundary Correlations
| eval_morphology_label | metric_x | metric_y | pearson_r |
| --- | --- | --- | --- |
| single_lobe | boundary_shift_abs_err_conditioned | boundary_slope_abs_err_conditioned | 0.6281 |
| single_lobe | boundary_shift_abs_err_conditioned | peak_abs_amp_err_conditioned | 0.1847 |
| single_lobe | delta_boundary_shift_abs_err | delta_boundary_slope_abs_err | 0.3456 |
| single_lobe | delta_boundary_shift_abs_err | delta_peak_abs_amp_err | -0.0961 |
| single_lobe | delta_rmse_tail_abs_steer | peak_abs_amp_err_conditioned | 0.3545 |
| single_lobe | delta_rmse_tail_abs_steer | boundary_shift_abs_err_conditioned | -0.1104 |
| reverse_correction | boundary_shift_abs_err_conditioned | boundary_slope_abs_err_conditioned | 0.3791 |
| reverse_correction | boundary_shift_abs_err_conditioned | peak_abs_amp_err_conditioned | 0.3924 |
| reverse_correction | delta_boundary_shift_abs_err | delta_boundary_slope_abs_err | 0.3428 |
| reverse_correction | delta_boundary_shift_abs_err | delta_peak_abs_amp_err | 0.0086 |
| reverse_correction | delta_rmse_tail_abs_steer | peak_abs_amp_err_conditioned | 0.0672 |
| reverse_correction | delta_rmse_tail_abs_steer | boundary_shift_abs_err_conditioned | 0.0187 |

### Interpretation
- `single_lobe` shows a large mean shift increase (`0.1821`) with only a small mean slope increase (`0.0244`), so the average picture is time-shift dominant.
- `reverse_correction` shows the same sign pattern: shift increase `0.1065` exceeds slope increase `0.0333`.
- Both morphologies still contain a substantial `shift-plus-smoothing` subset, so the result is better read as `time-shift dominant with a mixed secondary smoothing component`, not as a perfectly pure mode.

## Part C: Q1_fast x single_lobe Scatter
- Boundary scatter figure: `F:\data_set_process\data_process\reports\approx_boundary_slope_shift_scatter_20260408.png`
- Q1_fast x single_lobe amplitude-vs-boundary scatter: `F:\data_set_process\data_process\reports\approx_q1fast_single_lobe_amp_boundary_scatter_20260408.png`

| sample_label | subj | peak_abs_amp_err_conditioned | boundary_shift_abs_err_conditioned | boundary_slope_abs_err_conditioned | shape_corr_conditioned | delta_rmse_tail_abs_steer | dominant_dimension |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tyy#65 | tyy | 1.9472 | 0.2607 | 0.9473 | -0.7474 | 1.5254 | amplitude |
| tyy#74 | tyy | 1.2804 | 0.4599 | 0.2577 | 0.7761 | 0.4232 | amplitude |
| cwh#32 | cwh | 0.1446 | 0.4225 | 0.5340 | 0.7587 | 0.3246 | boundary_slope |
| tyy#68 | tyy | 1.1826 | 0.2476 | 0.2278 | -0.9178 | 0.3214 | amplitude |
| tyy#67 | tyy | 0.2800 | 0.0278 | 0.2380 | 0.8379 | 0.2656 | amplitude |
| cwh#4 | cwh | 0.4130 | 1.0225 | 1.3778 | 0.7845 | 0.2255 | boundary_slope |

### Interpretation
- The subset mean `delta_boundary_shift_abs_err` is positive (`0.0584`), but mean `delta_boundary_slope_abs_err` is slightly negative (`-0.0232`), which argues against slope flattening as the main driver in this exact intersection.
- The strongest relationship to tail degradation is amplitude error, not boundary error: `corr_tail_rmse_vs_amp=0.7195`, `corr_tail_rmse_vs_boundary_shift=-0.1544`, `corr_tail_rmse_vs_boundary_slope=0.1864`.
- The worst individual rows are mixed, but the largest tail-RMSE failures are led by high amplitude error more often than by exceptionally large boundary shift.

## Bottom Line
- Existing CSV metrics are sufficient to approximate the missing raw-sequence analysis well enough to answer the immediate diagnostic questions.
- The global `Q1_fast` issue is broader than a tail-only phenomenon, but the `Q1_fast x single_lobe` slice is genuinely tail-heavier.
- Boundary worsening is better described as time-shift dominant than slope-flattening dominant, especially in `Q1_fast x single_lobe`.
- For the worst `Q1_fast x single_lobe` rows, amplitude mismatch is the strongest companion of tail degradation, so boundary-only fixes would likely miss the most severe failures.
