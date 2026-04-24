# Sample Event Gallery

- Total sampled cases: 48
- Difficulty proxy: current dataset has no explicit easy/hard field, so `event_level` is used as the closest severity proxy.

## Case 01 | hzh | non_curve | medium_active | multi_correction

- file: `Entity_Recording_2025_09_26_21_17_02_vehicle_aligned_cleaned.csv` | event_idx: `43` | trigger: `steer` | split: `val`
- anchor_phase: `between_onset_and_risk_peak` | anchor_pct: `40.7%` | valid_future_s: `1.780`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-01](figures/case_01_hzh_non_curve_medium_active_multi_correction.png)

## Case 02 | lxy | curve | strong_active | recentering

- file: `Entity_Recording_2025_09_28_18_06_16_vehicle_aligned_cleaned.csv` | event_idx: `87` | trigger: `steer` | split: `train`
- anchor_phase: `pre_response` | anchor_pct: `6.5%` | valid_future_s: `2.000`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-02](figures/case_02_lxy_curve_strong_active_recentering.png)

## Case 03 | rjy | non_curve | medium_active | multi_correction

- file: `Entity_Recording_2025_09_28_20_02_20_vehicle_aligned_cleaned.csv` | event_idx: `67` | trigger: `steer` | split: `train`
- anchor_phase: `late_adjustment` | anchor_pct: `15.3%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-03](figures/case_03_rjy_non_curve_medium_active_multi_correction.png)

## Case 04 | yzy | curve | medium_active | multi_correction

- file: `Entity_Recording_2025_09_27_14_37_08_vehicle_aligned_cleaned.csv` | event_idx: `71` | trigger: `steer` | split: `train`
- anchor_phase: `pre_response` | anchor_pct: `0.0%` | valid_future_s: `1.945`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-04](figures/case_04_yzy_curve_medium_active_multi_correction.png)

## Case 05 | zx | non_curve | strong_active | multi_correction

- file: `Entity_Recording_2025_09_27_17_14_07_vehicle_aligned_cleaned.csv` | event_idx: `62` | trigger: `steer` | split: `train`
- anchor_phase: `between_onset_and_risk_peak` | anchor_pct: `30.8%` | valid_future_s: `1.470`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-05](figures/case_05_zx_non_curve_strong_active_multi_correction.png)

## Case 06 | tyy | curve | medium_active | insufficient_late_support

- file: `Entity_Recording_2025_09_28_14_44_09_vehicle_aligned_cleaned.csv` | event_idx: `54` | trigger: `steer` | split: `test`
- anchor_phase: `near_risk_peak` | anchor_pct: `48.4%` | valid_future_s: `0.955`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `event_start_fallback` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-06](figures/case_06_tyy_curve_medium_active_insufficient_late_support.png)

## Case 07 | zt | curve | medium_active | monotonic_continuation

- file: `Entity_Recording_2025_09_28_11_20_08_vehicle_aligned_cleaned.csv` | event_idx: `130` | trigger: `steer` | split: `train`
- anchor_phase: `response_onset` | anchor_pct: `26.0%` | valid_future_s: `1.475`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-07](figures/case_07_zt_curve_medium_active_monotonic_continuation.png)

## Case 08 | lx | curve | strong_active | monotonic_continuation

- file: `Entity_Recording_2025_09_26_09_06_38_vehicle_aligned_cleaned.csv` | event_idx: `61` | trigger: `steer` | split: `train`
- anchor_phase: `pre_response` | anchor_pct: `-0.3%` | valid_future_s: `1.700`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-08](figures/case_08_lx_curve_strong_active_monotonic_continuation.png)

## Case 09 | gf | curve | strong_active | multi_correction

- file: `Entity_Recording_2025_09_26_10_30_12_vehicle_aligned_cleaned.csv` | event_idx: `17` | trigger: `steer` | split: `test`
- anchor_phase: `pre_response` | anchor_pct: `7.3%` | valid_future_s: `1.845`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-09](figures/case_09_gf_curve_strong_active_multi_correction.png)

## Case 10 | cwh | curve | strong_active | recentering

- file: `Entity_Recording_2025_09_26_20_06_19_vehicle_aligned_cleaned.csv` | event_idx: `7` | trigger: `steer` | split: `test`
- anchor_phase: `late_adjustment` | anchor_pct: `29.2%` | valid_future_s: `1.365`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `event_start_fallback` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-10](figures/case_10_cwh_curve_strong_active_recentering.png)

## Case 11 | zdq | curve | extreme_active | multi_correction

- file: `Entity_Recording_2025_09_26_16_03_48_vehicle_aligned_cleaned.csv` | event_idx: `65` | trigger: `steer` | split: `train`
- anchor_phase: `between_onset_and_risk_peak` | anchor_pct: `39.2%` | valid_future_s: `1.825`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-11](figures/case_11_zdq_curve_extreme_active_multi_correction.png)

## Case 12 | zxy | curve | medium_active | insufficient_late_support

- file: `Entity_Recording_2025_09_28_16_35_30_vehicle_aligned_cleaned.csv` | event_idx: `187` | trigger: `steer` | split: `train`
- anchor_phase: `near_risk_peak` | anchor_pct: `73.0%` | valid_future_s: `0.430`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `event_start_fallback` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-12](figures/case_12_zxy_curve_medium_active_insufficient_late_support.png)

## Case 13 | txj | curve | extreme_active | multi_correction

- file: `Entity_Recording_2025_09_27_08_53_44_vehicle_aligned_cleaned.csv` | event_idx: `58` | trigger: `steer` | split: `train`
- anchor_phase: `late_adjustment` | anchor_pct: `43.5%` | valid_future_s: `1.695`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-13](figures/case_13_txj_curve_extreme_active_multi_correction.png)

## Case 14 | byx | curve | extreme_active | reverse_correction

- file: `Entity_Recording_2025_09_28_17_15_52_vehicle_aligned_cleaned.csv` | event_idx: `31` | trigger: `steer` | split: `train`
- anchor_phase: `between_onset_and_risk_peak` | anchor_pct: `5.8%` | valid_future_s: `2.000`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-14](figures/case_14_byx_curve_extreme_active_reverse_correction.png)

## Case 15 | xst | curve | medium_active | monotonic_continuation

- file: `Entity_Recording_2025_09_26_11_34_18_vehicle_aligned_cleaned.csv` | event_idx: `47` | trigger: `steer` | split: `val`
- anchor_phase: `pre_response` | anchor_pct: `5.8%` | valid_future_s: `1.455`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-15](figures/case_15_xst_curve_medium_active_monotonic_continuation.png)

## Case 16 | jy | non_curve | medium_active | multi_correction

- file: `Entity_Recording_2025_09_26_18_01_40_vehicle_aligned_cleaned.csv` | event_idx: `29` | trigger: `steer` | split: `val`
- anchor_phase: `between_onset_and_risk_peak` | anchor_pct: `21.6%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-16](figures/case_16_jy_non_curve_medium_active_multi_correction.png)

## Case 17 | yyl | curve | medium_active | insufficient_late_support

- file: `Entity_Recording_2025_09_28_09_14_23_vehicle_aligned_cleaned.csv` | event_idx: `49` | trigger: `steer` | split: `train`
- anchor_phase: `late_adjustment` | anchor_pct: `85.7%` | valid_future_s: `0.190`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-17](figures/case_17_yyl_curve_medium_active_insufficient_late_support.png)

## Case 18 | gzj | curve | extreme_active | reverse_correction

- file: `Entity_Recording_2025_09_27_12_28_14_vehicle_aligned_cleaned.csv` | event_idx: `65` | trigger: `steer` | split: `train`
- anchor_phase: `between_onset_and_risk_peak` | anchor_pct: `22.1%` | valid_future_s: `2.000`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-18](figures/case_18_gzj_curve_extreme_active_reverse_correction.png)

## Case 19 | rjy | curve | extreme_active | reverse_correction

- file: `Entity_Recording_2025_09_28_19_51_44_vehicle_aligned_cleaned.csv` | event_idx: `53` | trigger: `steer` | split: `train`
- anchor_phase: `pre_response` | anchor_pct: `0.0%` | valid_future_s: `1.795`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-19](figures/case_19_rjy_curve_extreme_active_reverse_correction.png)

## Case 20 | gf | curve | medium_active | recentering

- file: `Entity_Recording_2025_09_26_10_03_00_vehicle_aligned_cleaned.csv` | event_idx: `31` | trigger: `steer` | split: `test`
- anchor_phase: `pre_response` | anchor_pct: `0.0%` | valid_future_s: `2.000`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-20](figures/case_20_gf_curve_medium_active_recentering.png)

## Case 21 | gzj | non_curve | strong_active | multi_correction

- file: `Entity_Recording_2025_09_27_12_28_14_vehicle_aligned_cleaned.csv` | event_idx: `102` | trigger: `steer` | split: `train`
- anchor_phase: `late_adjustment` | anchor_pct: `25.4%` | valid_future_s: `1.375`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-21](figures/case_21_gzj_non_curve_strong_active_multi_correction.png)

## Case 22 | byx | curve | extreme_active | recentering

- file: `Entity_Recording_2025_09_28_17_35_43_vehicle_aligned_cleaned.csv` | event_idx: `93` | trigger: `steer` | split: `train`
- anchor_phase: `response_onset` | anchor_pct: `0.0%` | valid_future_s: `2.000`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-22](figures/case_22_byx_curve_extreme_active_recentering.png)

## Case 23 | lxy | curve | medium_active | insufficient_late_support

- file: `Entity_Recording_2025_09_28_18_06_16_vehicle_aligned_cleaned.csv` | event_idx: `41` | trigger: `steer` | split: `train`
- anchor_phase: `near_risk_peak` | anchor_pct: `48.0%` | valid_future_s: `0.570`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-23](figures/case_23_lxy_curve_medium_active_insufficient_late_support.png)

## Case 24 | zx | curve | strong_active | recentering

- file: `Entity_Recording_2025_09_27_18_00_08_vehicle_aligned_cleaned.csv` | event_idx: `50` | trigger: `steer` | split: `train`
- anchor_phase: `pre_response` | anchor_pct: `0.0%` | valid_future_s: `1.940`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-24](figures/case_24_zx_curve_strong_active_recentering.png)

## Case 25 | rjy | non_curve | extreme_active | multi_correction

- file: `Entity_Recording_2025_09_28_20_15_42_vehicle_aligned_cleaned.csv` | event_idx: `51` | trigger: `steer` | split: `train`
- anchor_phase: `response_onset` | anchor_pct: `0.0%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-25](figures/case_25_rjy_non_curve_extreme_active_multi_correction.png)

## Case 26 | jy | non_curve | medium_active | reverse_correction

- file: `Entity_Recording_2025_09_26_17_17_11_vehicle_aligned_cleaned.csv` | event_idx: `23` | trigger: `steer` | split: `val`
- anchor_phase: `response_onset` | anchor_pct: `0.2%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-26](figures/case_26_jy_non_curve_medium_active_reverse_correction.png)

## Case 27 | jy | curve | medium_active | monotonic_continuation

- file: `Entity_Recording_2025_09_26_17_51_46_vehicle_aligned_cleaned.csv` | event_idx: `13` | trigger: `steer` | split: `val`
- anchor_phase: `pre_response` | anchor_pct: `-0.4%` | valid_future_s: `1.330`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `event_start_fallback` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-27](figures/case_27_jy_curve_medium_active_monotonic_continuation.png)

## Case 28 | hzh | non_curve | medium_active | recentering

- file: `Entity_Recording_2025_09_26_21_17_02_vehicle_aligned_cleaned.csv` | event_idx: `36` | trigger: `steer` | split: `val`
- anchor_phase: `between_onset_and_risk_peak` | anchor_pct: `6.3%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-28](figures/case_28_hzh_non_curve_medium_active_recentering.png)

## Case 29 | rjy | curve | strong_active | insufficient_late_support

- file: `Entity_Recording_2025_09_28_19_33_26_vehicle_aligned_cleaned.csv` | event_idx: `17` | trigger: `steer` | split: `train`
- anchor_phase: `pre_response` | anchor_pct: `23.3%` | valid_future_s: `0.885`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-29](figures/case_29_rjy_curve_strong_active_insufficient_late_support.png)

## Case 30 | yyl | curve | strong_active | monotonic_continuation

- file: `Entity_Recording_2025_09_28_09_14_23_vehicle_aligned_cleaned.csv` | event_idx: `87` | trigger: `steer` | split: `train`
- anchor_phase: `pre_response` | anchor_pct: `0.0%` | valid_future_s: `2.000`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-30](figures/case_30_yyl_curve_strong_active_monotonic_continuation.png)

## Case 31 | txj | non_curve | extreme_active | reverse_correction

- file: `Entity_Recording_2025_09_27_09_06_19_vehicle_aligned_cleaned.csv` | event_idx: `19` | trigger: `steer` | split: `train`
- anchor_phase: `response_onset` | anchor_pct: `0.0%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-31](figures/case_31_txj_non_curve_extreme_active_reverse_correction.png)

## Case 32 | jy | non_curve | strong_active | reverse_correction

- file: `Entity_Recording_2025_09_26_17_17_11_vehicle_aligned_cleaned.csv` | event_idx: `10` | trigger: `steer` | split: `val`
- anchor_phase: `response_onset` | anchor_pct: `0.0%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-32](figures/case_32_jy_non_curve_strong_active_reverse_correction.png)

## Case 33 | hzh | non_curve | extreme_active | recentering

- file: `Entity_Recording_2025_09_27_19_33_25_vehicle_aligned_cleaned.csv` | event_idx: `39` | trigger: `steer` | split: `val`
- anchor_phase: `response_onset` | anchor_pct: `0.0%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-33](figures/case_33_hzh_non_curve_extreme_active_recentering.png)

## Case 34 | cwh | non_curve | medium_active | insufficient_late_support

- file: `Entity_Recording_2025_09_26_20_06_19_vehicle_aligned_cleaned.csv` | event_idx: `32` | trigger: `steer` | split: `test`
- anchor_phase: `response_onset` | anchor_pct: `34.7%` | valid_future_s: `0.880`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-34](figures/case_34_cwh_non_curve_medium_active_insufficient_late_support.png)

## Case 35 | yyl | non_curve | strong_active | recentering

- file: `Entity_Recording_2025_09_28_09_39_01_vehicle_aligned_cleaned.csv` | event_idx: `76` | trigger: `steer` | split: `train`
- anchor_phase: `response_onset` | anchor_pct: `1.5%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-35](figures/case_35_yyl_non_curve_strong_active_recentering.png)

## Case 36 | lxy | non_curve | medium_active | monotonic_continuation

- file: `Entity_Recording_2025_09_28_18_06_16_vehicle_aligned_cleaned.csv` | event_idx: `104` | trigger: `steer` | split: `train`
- anchor_phase: `response_onset` | anchor_pct: `0.0%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_peak_fallback` | control_end_rule: `smoothed_control_activity`

![case-36](figures/case_36_lxy_non_curve_medium_active_monotonic_continuation.png)

## Case 37 | gf | curve | extreme_active | monotonic_continuation

- file: `Entity_Recording_2025_09_26_10_18_49_vehicle_aligned_cleaned.csv` | event_idx: `36` | trigger: `steer` | split: `test`
- anchor_phase: `pre_response` | anchor_pct: `0.0%` | valid_future_s: `2.000`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-37](figures/case_37_gf_curve_extreme_active_monotonic_continuation.png)

## Case 38 | yzy | non_curve | extreme_active | monotonic_continuation

- file: `Entity_Recording_2025_09_27_14_26_04_vehicle_aligned_cleaned.csv` | event_idx: `98` | trigger: `steer` | split: `train`
- anchor_phase: `response_onset` | anchor_pct: `1.5%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-38](figures/case_38_yzy_non_curve_extreme_active_monotonic_continuation.png)

## Case 39 | zdq | curve | extreme_active | insufficient_late_support

- file: `Entity_Recording_2025_09_26_15_37_30_vehicle_aligned_cleaned.csv` | event_idx: `38` | trigger: `steer` | split: `train`
- anchor_phase: `pre_response` | anchor_pct: `16.5%` | valid_future_s: `0.955`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `event_start_fallback` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-39](figures/case_39_zdq_curve_extreme_active_insufficient_late_support.png)

## Case 40 | zdq | non_curve | extreme_active | insufficient_late_support

- file: `Entity_Recording_2025_09_26_16_03_48_vehicle_aligned_cleaned.csv` | event_idx: `39` | trigger: `steer` | split: `train`
- anchor_phase: `response_onset` | anchor_pct: `26.7%` | valid_future_s: `0.805`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-40](figures/case_40_zdq_non_curve_extreme_active_insufficient_late_support.png)

## Case 41 | tyy | non_curve | strong_active | monotonic_continuation

- file: `Entity_Recording_2025_09_28_14_23_43_vehicle_aligned_cleaned.csv` | event_idx: `152` | trigger: `steer` | split: `test`
- anchor_phase: `response_onset` | anchor_pct: `5.0%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-41](figures/case_41_tyy_non_curve_strong_active_monotonic_continuation.png)

## Case 42 | zx | non_curve | strong_active | insufficient_late_support

- file: `Entity_Recording_2025_09_27_18_07_01_vehicle_aligned_cleaned.csv` | event_idx: `23` | trigger: `steer` | split: `train`
- anchor_phase: `between_onset_and_risk_peak` | anchor_pct: `14.9%` | valid_future_s: `0.710`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-42](figures/case_42_zx_non_curve_strong_active_insufficient_late_support.png)

## Case 43 | jy | curve | medium_active | recentering

- file: `Entity_Recording_2025_09_26_17_29_44_vehicle_aligned_cleaned.csv` | event_idx: `20` | trigger: `steer` | split: `val`
- anchor_phase: `pre_response` | anchor_pct: `0.0%` | valid_future_s: `1.890`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-43](figures/case_43_jy_curve_medium_active_recentering.png)

## Case 44 | jy | non_curve | strong_active | multi_correction

- file: `Entity_Recording_2025_09_26_17_29_44_vehicle_aligned_cleaned.csv` | event_idx: `4` | trigger: `steer` | split: `val`
- anchor_phase: `between_onset_and_risk_peak` | anchor_pct: `28.4%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-44](figures/case_44_jy_non_curve_strong_active_multi_correction.png)

## Case 45 | byx | curve | extreme_active | recentering

- file: `Entity_Recording_2025_09_28_17_25_18_vehicle_aligned_cleaned.csv` | event_idx: `70` | trigger: `steer` | split: `train`
- anchor_phase: `pre_response` | anchor_pct: `11.3%` | valid_future_s: `1.690`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-45](figures/case_45_byx_curve_extreme_active_recentering.png)

## Case 46 | gzj | curve | medium_active | insufficient_late_support

- file: `Entity_Recording_2025_09_27_12_17_12_vehicle_aligned_cleaned.csv` | event_idx: `111` | trigger: `steer` | split: `train`
- anchor_phase: `late_adjustment` | anchor_pct: `64.9%` | valid_future_s: `0.730`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-46](figures/case_46_gzj_curve_medium_active_insufficient_late_support.png)

## Case 47 | gf | curve | strong_active | recentering

- file: `Entity_Recording_2025_09_26_10_52_57_vehicle_aligned_cleaned.csv` | event_idx: `81` | trigger: `steer` | split: `test`
- anchor_phase: `pre_response` | anchor_pct: `4.8%` | valid_future_s: `1.875`
- anchor_source: `curve_roll_peak` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-47](figures/case_47_gf_curve_strong_active_recentering.png)

## Case 48 | zx | non_curve | extreme_active | multi_correction

- file: `Entity_Recording_2025_09_27_18_07_01_vehicle_aligned_cleaned.csv` | event_idx: `22` | trigger: `steer` | split: `train`
- anchor_phase: `response_onset` | anchor_pct: `3.3%` | valid_future_s: `2.000`
- anchor_source: `noncurve_steer_rate_first80` | scene_trigger_rule: `steer_rate_gt4` | first_response_rule: `smoothed_steer_rate_threshold` | control_end_rule: `smoothed_control_activity`

![case-48](figures/case_48_zx_non_curve_extreme_active_multi_correction.png)

