# v254b 200Hz 连续生理事件相关表征审计

## 本轮问题

v254a 证明 1Hz/10Hz 表格窗口统计没有跨驾驶员行为增量。本轮改为从清洗后 200Hz 连续层直接抽取事件相关变化，并做每个事件自身锚点前 baseline 归一化。

## 对齐覆盖

| split   |    n |   ok_rate |   uses_post_observation_rate |   baseline_rows_mean |   baseline_rows_p10 |   pre20_pre10_rows_mean |   pre10_pre5_rows_mean |   pre5_pre2_rows_mean |   pre2_0_rows_mean |
|:--------|-----:|----------:|-----------------------------:|---------------------:|--------------------:|------------------------:|-----------------------:|----------------------:|-------------------:|
| test    | 1104 |  0.896739 |                            0 |              7971.56 |                8000 |                 2000.85 |                1000.86 |               600.865 |            400.87  |
| train   | 4044 |  0.888724 |                            0 |              7959.22 |                8000 |                 2000.87 |                1000.88 |               600.893 |            400.898 |
| val     | 1854 |  1        |                            0 |              7966.88 |                8000 |                 2000.87 |                1000.88 |               600.893 |            400.899 |

## 特征块

| feature_block                  |   raw_dim |   kept_dim |   kept_physio_columns |
|:-------------------------------|----------:|-----------:|----------------------:|
| vehicle_only                   |       268 |        268 |                     0 |
| physio200_all                  |       804 |        678 |                   678 |
| physio200_norm                 |       283 |        227 |                   227 |
| physio200_curated              |       168 |        168 |                   168 |
| vehicle_plus_physio200_norm    |       551 |        495 |                   227 |
| vehicle_plus_physio200_curated |       436 |        436 |                   168 |

## 行为分类诊断

| split_protocol   | task_type      | target                    | feature_block                  | eval_split   |   n_eval |   accuracy |   macro_f1 |        auc |   vehicle_metric |   delta_macro_f1_minus_vehicle |
|:-----------------|:---------------|:--------------------------|:-------------------------------|:-------------|---------:|-----------:|-----------:|-----------:|-----------------:|-------------------------------:|
| subject_disjoint | classification | future_cluster4           | vehicle_only                   | test         |     1104 |   0.738225 |   0.715385 | nan        |         0.715385 |                    0           |
| subject_disjoint | classification | future_cluster4           | physio200_norm                 | test         |     1104 |   0.249094 |   0.224069 | nan        |         0.715385 |                   -0.491317    |
| subject_disjoint | classification | future_cluster4           | physio200_curated              | test         |     1104 |   0.240036 |   0.215067 | nan        |         0.715385 |                   -0.500318    |
| subject_disjoint | classification | future_cluster4           | vehicle_plus_physio200_curated | test         |     1104 |   0.699275 |   0.685203 | nan        |         0.715385 |                   -0.0301825   |
| subject_disjoint | classification | high_future_abs_q75       | vehicle_only                   | test         |     1104 |   0.816123 |   0.740848 |   0.826048 |         0.740848 |                    0           |
| subject_disjoint | classification | high_future_abs_q75       | physio200_norm                 | test         |     1104 |   0.601449 |   0.543952 |   0.602945 |         0.740848 |                   -0.196896    |
| subject_disjoint | classification | high_future_abs_q75       | physio200_curated              | test         |     1104 |   0.486413 |   0.44517  |   0.484989 |         0.740848 |                   -0.295679    |
| subject_disjoint | classification | high_future_abs_q75       | vehicle_plus_physio200_curated | test         |     1104 |   0.706522 |   0.616888 |   0.678417 |         0.740848 |                   -0.12396     |
| subject_disjoint | classification | high_future_range_q75     | vehicle_only                   | test         |     1104 |   0.742754 |   0.655334 |   0.682516 |         0.655334 |                    0           |
| subject_disjoint | classification | high_future_range_q75     | physio200_norm                 | test         |     1104 |   0.67663  |   0.625188 |   0.684807 |         0.655334 |                   -0.0301458   |
| subject_disjoint | classification | high_future_range_q75     | physio200_curated              | test         |     1104 |   0.488225 |   0.451683 |   0.477714 |         0.655334 |                   -0.20365     |
| subject_disjoint | classification | high_future_range_q75     | vehicle_plus_physio200_curated | test         |     1104 |   0.674819 |   0.597052 |   0.644969 |         0.655334 |                   -0.0582817   |
| subject_disjoint | classification | strong_steer_existing     | vehicle_only                   | test         |     1104 |   0.683877 |   0.677181 |   0.725531 |         0.677181 |                    0           |
| subject_disjoint | classification | strong_steer_existing     | physio200_norm                 | test         |     1104 |   0.568841 |   0.568699 |   0.599069 |         0.677181 |                   -0.108482    |
| subject_disjoint | classification | strong_steer_existing     | physio200_curated              | test         |     1104 |   0.511775 |   0.50623  |   0.530592 |         0.677181 |                   -0.170951    |
| subject_disjoint | classification | strong_steer_existing     | vehicle_plus_physio200_curated | test         |     1104 |   0.618659 |   0.613213 |   0.669758 |         0.677181 |                   -0.0639674   |
| subject_disjoint | classification | bad_top10_v250_diagnostic | vehicle_only                   | test         |     1104 |   0.827899 |   0.495769 |   0.548837 |         0.495769 |                    0           |
| subject_disjoint | classification | bad_top10_v250_diagnostic | physio200_norm                 | test         |     1104 |   0.814312 |   0.500608 |   0.557406 |         0.495769 |                    0.00483868  |
| subject_disjoint | classification | bad_top10_v250_diagnostic | physio200_curated              | test         |     1104 |   0.852355 |   0.509517 |   0.498861 |         0.495769 |                    0.0137473   |
| subject_disjoint | classification | bad_top10_v250_diagnostic | vehicle_plus_physio200_curated | test         |     1104 |   0.768116 |   0.517047 |   0.493341 |         0.495769 |                    0.0212779   |
| subject_aware    | classification | future_cluster4           | vehicle_only                   | test         |     1398 |   0.711731 |   0.705241 | nan        |         0.705241 |                    0           |
| subject_aware    | classification | future_cluster4           | physio200_norm                 | test         |     1398 |   0.262518 |   0.253615 | nan        |         0.705241 |                   -0.451625    |
| subject_aware    | classification | future_cluster4           | physio200_curated              | test         |     1398 |   0.265379 |   0.260781 | nan        |         0.705241 |                   -0.444459    |
| subject_aware    | classification | future_cluster4           | vehicle_plus_physio200_curated | test         |     1398 |   0.679542 |   0.679799 | nan        |         0.705241 |                   -0.0254411   |
| subject_aware    | classification | high_future_abs_q75       | vehicle_only                   | test         |     1398 |   0.736767 |   0.687728 |   0.759368 |         0.687728 |                    0           |
| subject_aware    | classification | high_future_abs_q75       | physio200_norm                 | test         |     1398 |   0.640916 |   0.608897 |   0.672631 |         0.687728 |                   -0.0788308   |
| subject_aware    | classification | high_future_abs_q75       | physio200_curated              | test         |     1398 |   0.586552 |   0.542905 |   0.569362 |         0.687728 |                   -0.144824    |
| subject_aware    | classification | high_future_abs_q75       | vehicle_plus_physio200_curated | test         |     1398 |   0.693848 |   0.655209 |   0.719428 |         0.687728 |                   -0.0325196   |
| subject_aware    | classification | high_future_range_q75     | vehicle_only                   | test         |     1398 |   0.676681 |   0.655323 |   0.7494   |         0.655323 |                    0           |
| subject_aware    | classification | high_future_range_q75     | physio200_norm                 | test         |     1398 |   0.585122 |   0.558712 |   0.59856  |         0.655323 |                   -0.0966111   |
| subject_aware    | classification | high_future_range_q75     | physio200_curated              | test         |     1398 |   0.54578  |   0.521809 |   0.559455 |         0.655323 |                   -0.133514    |
| subject_aware    | classification | high_future_range_q75     | vehicle_plus_physio200_curated | test         |     1398 |   0.690987 |   0.654913 |   0.715861 |         0.655323 |                   -0.000409754 |
| subject_aware    | classification | strong_steer_existing     | vehicle_only                   | test         |     1398 |   0.658798 |   0.658671 |   0.677673 |         0.658671 |                    0           |
| subject_aware    | classification | strong_steer_existing     | physio200_norm                 | test         |     1398 |   0.573677 |   0.573669 |   0.602581 |         0.658671 |                   -0.0850021   |
| subject_aware    | classification | strong_steer_existing     | physio200_curated              | test         |     1398 |   0.536481 |   0.534465 |   0.547491 |         0.658671 |                   -0.124206    |
| subject_aware    | classification | strong_steer_existing     | vehicle_plus_physio200_curated | test         |     1398 |   0.650215 |   0.649995 |   0.681738 |         0.658671 |                   -0.00867575  |
| subject_aware    | classification | bad_top10_v250_diagnostic | vehicle_only                   | test         |     1398 |   0.628755 |   0.457799 |   0.622279 |         0.457799 |                    0           |
| subject_aware    | classification | bad_top10_v250_diagnostic | physio200_norm                 | test         |     1398 |   0.801144 |   0.543794 |   0.691528 |         0.457799 |                    0.0859947   |
| subject_aware    | classification | bad_top10_v250_diagnostic | physio200_curated              | test         |     1398 |   0.72103  |   0.496122 |   0.606659 |         0.457799 |                    0.0383232   |
| subject_aware    | classification | bad_top10_v250_diagnostic | vehicle_plus_physio200_curated | test         |     1398 |   0.833333 |   0.583202 |   0.705474 |         0.457799 |                    0.125403    |

## 未来摘要回归诊断

| split_protocol   | task_type   | target          | feature_block                  | eval_split   |   n_eval |           r2 |      mae |   vehicle_metric |   delta_r2_minus_vehicle |
|:-----------------|:------------|:----------------|:-------------------------------|:-------------|---------:|-------------:|---------:|-----------------:|-------------------------:|
| subject_disjoint | regression  | future_peak_abs | vehicle_only                   | test         |     1104 |    0.116376  | 0.550292 |        0.116376  |                 0        |
| subject_disjoint | regression  | future_peak_abs | physio200_norm                 | test         |     1104 |   -1.16651   | 0.837114 |        0.116376  |                -1.28289  |
| subject_disjoint | regression  | future_peak_abs | physio200_curated              | test         |     1104 |   -0.468635  | 0.807282 |        0.116376  |                -0.585011 |
| subject_disjoint | regression  | future_peak_abs | vehicle_plus_physio200_curated | test         |     1104 |   -0.0593985 | 0.619512 |        0.116376  |                -0.175774 |
| subject_disjoint | regression  | future_range    | vehicle_only                   | test         |     1104 |   -0.0272699 | 0.641989 |       -0.0272699 |                 0        |
| subject_disjoint | regression  | future_range    | physio200_norm                 | test         |     1104 |   -1.58447   | 0.926548 |       -0.0272699 |                -1.5572   |
| subject_disjoint | regression  | future_range    | physio200_curated              | test         |     1104 |   -0.434776  | 0.89693  |       -0.0272699 |                -0.407506 |
| subject_disjoint | regression  | future_range    | vehicle_plus_physio200_curated | test         |     1104 |   -0.199398  | 0.71907  |       -0.0272699 |                -0.172128 |
| subject_disjoint | regression  | future_mean_abs | vehicle_only                   | test         |     1104 |   -0.134768  | 0.327386 |       -0.134768  |                 0        |
| subject_disjoint | regression  | future_mean_abs | physio200_norm                 | test         |     1104 |   -0.603476  | 0.498956 |       -0.134768  |                -0.468709 |
| subject_disjoint | regression  | future_mean_abs | physio200_curated              | test         |     1104 |   -0.37566   | 0.461805 |       -0.134768  |                -0.240892 |
| subject_disjoint | regression  | future_mean_abs | vehicle_plus_physio200_curated | test         |     1104 |   -0.26961   | 0.358643 |       -0.134768  |                -0.134842 |
| subject_aware    | regression  | future_peak_abs | vehicle_only                   | test         |     1398 |    0.269693  | 0.618066 |        0.269693  |                 0        |
| subject_aware    | regression  | future_peak_abs | physio200_norm                 | test         |     1398 | -191.194     | 1.87209  |        0.269693  |              -191.464    |
| subject_aware    | regression  | future_peak_abs | physio200_curated              | test         |     1398 |   -0.214177  | 0.83899  |        0.269693  |                -0.48387  |
| subject_aware    | regression  | future_peak_abs | vehicle_plus_physio200_curated | test         |     1398 |   -0.300298  | 0.750146 |        0.269693  |                -0.569991 |
| subject_aware    | regression  | future_range    | vehicle_only                   | test         |     1398 |    0.24452   | 0.700305 |        0.24452   |                 0        |
| subject_aware    | regression  | future_range    | physio200_norm                 | test         |     1398 | -130.048     | 1.89325  |        0.24452   |              -130.292    |
| subject_aware    | regression  | future_range    | physio200_curated              | test         |     1398 |   -0.171143  | 0.924268 |        0.24452   |                -0.415664 |
| subject_aware    | regression  | future_range    | vehicle_plus_physio200_curated | test         |     1398 |   -0.0743679 | 0.806761 |        0.24452   |                -0.318888 |
| subject_aware    | regression  | future_mean_abs | vehicle_only                   | test         |     1398 |    0.305774  | 0.357806 |        0.305774  |                 0        |
| subject_aware    | regression  | future_mean_abs | physio200_norm                 | test         |     1398 | -154.958     | 1.00712  |        0.305774  |              -155.264    |
| subject_aware    | regression  | future_mean_abs | physio200_curated              | test         |     1398 |   -0.212444  | 0.468564 |        0.305774  |                -0.518218 |
| subject_aware    | regression  | future_mean_abs | vehicle_plus_physio200_curated | test         |     1398 |   -0.0879845 | 0.418961 |        0.305774  |                -0.393758 |

## eta² top

| target                | feature                                       | signal         |      eta2 |
|:----------------------|:----------------------------------------------|:---------------|----------:|
| future_cluster4       | physio200_base_EDA_Tonic_valid_ratio          | EDA_Tonic      | 0.0152023 |
| future_cluster4       | physio200_pre20_pre10_EDA_Tonic_valid_ratio   | EDA_Tonic      | 0.0152023 |
| future_cluster4       | physio200_pre10_pre5_EDA_Tonic_valid_ratio    | EDA_Tonic      | 0.0152023 |
| future_cluster4       | physio200_pre5_pre2_EDA_Tonic_valid_ratio     | EDA_Tonic      | 0.0152023 |
| future_cluster4       | physio200_pre2_0_EDA_Tonic_valid_ratio        | EDA_Tonic      | 0.0152023 |
| future_cluster4       | physio200_base_EDA_Phasic_valid_ratio         | EDA_Phasic     | 0.0152023 |
| future_cluster4       | physio200_pre20_pre10_EDA_Phasic_valid_ratio  | EDA_Phasic     | 0.0152023 |
| future_cluster4       | physio200_pre10_pre5_EDA_Phasic_valid_ratio   | EDA_Phasic     | 0.0152023 |
| future_cluster4       | physio200_pre5_pre2_EDA_Phasic_valid_ratio    | EDA_Phasic     | 0.0152023 |
| future_cluster4       | physio200_pre2_0_EDA_Phasic_valid_ratio       | EDA_Phasic     | 0.0152023 |
| future_cluster4       | physio200_pre2_0_EMG_RMS_last_minus_first     | EMG_RMS        | 0.0109632 |
| future_cluster4       | physio200_pre2_0_EMG_RMS_slope                | EMG_RMS        | 0.0109599 |
| high_future_abs_q75   | physio200_base_EDA_Tonic_valid_ratio          | EDA_Tonic      | 0.0173232 |
| high_future_abs_q75   | physio200_pre20_pre10_EDA_Tonic_valid_ratio   | EDA_Tonic      | 0.0173232 |
| high_future_abs_q75   | physio200_pre10_pre5_EDA_Tonic_valid_ratio    | EDA_Tonic      | 0.0173232 |
| high_future_abs_q75   | physio200_pre5_pre2_EDA_Tonic_valid_ratio     | EDA_Tonic      | 0.0173232 |
| high_future_abs_q75   | physio200_pre2_0_EDA_Tonic_valid_ratio        | EDA_Tonic      | 0.0173232 |
| high_future_abs_q75   | physio200_base_EDA_Phasic_valid_ratio         | EDA_Phasic     | 0.0173232 |
| high_future_abs_q75   | physio200_pre20_pre10_EDA_Phasic_valid_ratio  | EDA_Phasic     | 0.0173232 |
| high_future_abs_q75   | physio200_pre10_pre5_EDA_Phasic_valid_ratio   | EDA_Phasic     | 0.0173232 |
| high_future_abs_q75   | physio200_pre5_pre2_EDA_Phasic_valid_ratio    | EDA_Phasic     | 0.0173232 |
| high_future_abs_q75   | physio200_pre2_0_EDA_Phasic_valid_ratio       | EDA_Phasic     | 0.0173232 |
| high_future_abs_q75   | physio200_pre2_0_ECG_filt200_z_p90            | ECG_filt200    | 0.0147779 |
| high_future_abs_q75   | physio200_pre20_pre10_ECG_filt200_p10         | ECG_filt200    | 0.0139972 |
| recording             | physio200_pre10_pre5_RESP_BPM_mean            | RESP_BPM       | 1         |
| recording             | physio200_pre10_pre5_RESP_BPM_abs_mean        | RESP_BPM       | 1         |
| recording             | physio200_base_RESP_Amplitude_mean            | RESP_Amplitude | 1         |
| recording             | physio200_pre10_pre5_RESP_Amplitude_mean      | RESP_Amplitude | 1         |
| recording             | physio200_pre10_pre5_RESP_Amplitude_abs_mean  | RESP_Amplitude | 1         |
| recording             | physio200_base_RESP_BPM_mean                  | RESP_BPM       | 1         |
| recording             | physio200_pre20_pre10_RESP_Amplitude_mean     | RESP_Amplitude | 1         |
| recording             | physio200_pre20_pre10_RESP_Amplitude_abs_mean | RESP_Amplitude | 1         |
| recording             | physio200_pre20_pre10_RESP_Amplitude_rms      | RESP_Amplitude | 1         |
| recording             | physio200_pre10_pre5_RESP_Amplitude_rms       | RESP_Amplitude | 1         |
| recording             | physio200_pre5_pre2_RESP_Amplitude_mean       | RESP_Amplitude | 1         |
| recording             | physio200_pre5_pre2_RESP_Amplitude_abs_mean   | RESP_Amplitude | 1         |
| strong_steer_existing | physio200_base_EDA_Tonic_valid_ratio          | EDA_Tonic      | 0.0233727 |
| strong_steer_existing | physio200_pre20_pre10_EDA_Tonic_valid_ratio   | EDA_Tonic      | 0.0233727 |
| strong_steer_existing | physio200_pre10_pre5_EDA_Tonic_valid_ratio    | EDA_Tonic      | 0.0233727 |
| strong_steer_existing | physio200_pre5_pre2_EDA_Tonic_valid_ratio     | EDA_Tonic      | 0.0233727 |
| strong_steer_existing | physio200_pre2_0_EDA_Tonic_valid_ratio        | EDA_Tonic      | 0.0233727 |
| strong_steer_existing | physio200_base_EDA_Phasic_valid_ratio         | EDA_Phasic     | 0.0233727 |
| strong_steer_existing | physio200_pre20_pre10_EDA_Phasic_valid_ratio  | EDA_Phasic     | 0.0233727 |
| strong_steer_existing | physio200_pre10_pre5_EDA_Phasic_valid_ratio   | EDA_Phasic     | 0.0233727 |
| strong_steer_existing | physio200_pre5_pre2_EDA_Phasic_valid_ratio    | EDA_Phasic     | 0.0233727 |
| strong_steer_existing | physio200_pre2_0_EDA_Phasic_valid_ratio       | EDA_Phasic     | 0.0233727 |
| strong_steer_existing | physio200_base_RESP_filt200_median            | RESP_filt200   | 0.0172459 |
| strong_steer_existing | physio200_pre5_pre2_ECG_filt200_p10           | ECG_filt200    | 0.0155704 |
| subject               | physio200_base_EDA_Tonic_valid_ratio          | EDA_Tonic      | 1         |
| subject               | physio200_pre20_pre10_EDA_Tonic_valid_ratio   | EDA_Tonic      | 1         |
| subject               | physio200_pre10_pre5_EDA_Tonic_valid_ratio    | EDA_Tonic      | 1         |
| subject               | physio200_pre5_pre2_EDA_Tonic_valid_ratio     | EDA_Tonic      | 1         |
| subject               | physio200_pre2_0_EDA_Tonic_valid_ratio        | EDA_Tonic      | 1         |
| subject               | physio200_base_EDA_Phasic_valid_ratio         | EDA_Phasic     | 1         |
| subject               | physio200_pre20_pre10_EDA_Phasic_valid_ratio  | EDA_Phasic     | 1         |
| subject               | physio200_pre10_pre5_EDA_Phasic_valid_ratio   | EDA_Phasic     | 1         |
| subject               | physio200_pre5_pre2_EDA_Phasic_valid_ratio    | EDA_Phasic     | 1         |
| subject               | physio200_pre2_0_EDA_Phasic_valid_ratio       | EDA_Phasic     | 1         |
| subject               | physio200_base_ECG_filt200_median             | ECG_filt200    | 0.949295  |
| subject               | physio200_base_ECG_filt200_scale              | ECG_filt200    | 0.931274  |

## 判读

- subject_disjoint 是当前正式泛化口径。
- subject_aware 只是诊断同一驾驶员历史样本可用时的个体化潜力。
- 如果 subject_aware 明显好而 subject_disjoint 不好，说明生理更适合个体化/校准，不适合作为跨驾驶员直接泛化特征。
- 如果 vehicle_plus_physio200_curated 仍不超过 vehicle_only，下一步应进入表示学习/时序编码，而不是继续手工统计。

## 关键图

- `figures\v254b_macro_f1_subject_disjoint_vs_subject_aware.png`
- `figures\v254b_top_eta2_physio200.png`