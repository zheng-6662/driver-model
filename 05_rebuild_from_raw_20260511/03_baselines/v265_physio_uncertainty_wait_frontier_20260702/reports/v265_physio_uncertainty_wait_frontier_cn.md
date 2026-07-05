# v265 physiology uncertainty / wait frontier

## 本轮问题

- v260-v264 说明生理不能直接改善轨迹、selector、wait gate 或 online KNN。
- v265 检查最后一个合理用途：生理是否能作为不确定性/风险校准信号，在固定等待预算下更会挑选需要 wait-latest 的样本。
- 所有风险模型只在 train 拟合，等待比例阈值只在 val 定标，test 只报告。

## 特征块

| score                        | target                         |   feature_n |   vehicle_feature_n |   bio260_feature_n |
|:-----------------------------|:-------------------------------|------------:|--------------------:|-------------------:|
| score_vehicle_gain           | target_gain_latest_vs_keep0    |          35 |                  35 |                  0 |
| score_vehicle_bio_gain       | target_gain_latest_vs_keep0    |         100 |                  35 |                 65 |
| score_bio_only_gain          | target_gain_latest_vs_keep0    |          65 |                   0 |                 65 |
| score_vehicle_keep0_risk     | target_keep0_tail_rmse         |          35 |                  35 |                  0 |
| score_vehicle_bio_keep0_risk | target_keep0_tail_rmse         |         100 |                  35 |                 65 |
| score_bio_only_keep0_risk    | target_keep0_tail_rmse         |          65 |                   0 |                 65 |
| score_vehicle_badprob        | target_bad_top10               |          35 |                  35 |                  0 |
| score_vehicle_bio_badprob    | target_bad_top10               |         100 |                  35 |                 65 |
| score_bio_only_badprob       | target_bad_top10               |          65 |                   0 |                 65 |
| score_vehicle_oracle_gap     | target_oracle_gap_after_latest |          35 |                  35 |                  0 |
| score_vehicle_bio_oracle_gap | target_oracle_gap_after_latest |         100 |                  35 |                 65 |

## 分数诊断

| split   | score                        |   n |   auc_wait_better |   auc_bad_top10 |   spearman_gain |   spearman_keep0_rmse |
|:--------|:-----------------------------|----:|------------------:|----------------:|----------------:|----------------------:|
| val     | score_vehicle_gain           | 309 |          0.507144 |        0.462404 |    -0.0320242   |          -0.0461171   |
| val     | score_vehicle_bio_gain       | 309 |          0.428218 |        0.398352 |    -0.135933    |          -0.153134    |
| val     | score_bio_only_gain          | 309 |          0.460228 |        0.400673 |    -0.0661284   |          -0.0708612   |
| val     | score_vehicle_keep0_risk     | 309 |          0.564707 |        0.532722 |     0.034657    |           0.0637572   |
| val     | score_vehicle_bio_keep0_risk | 309 |          0.562303 |        0.517986 |     0.0689125   |           0.0491131   |
| val     | score_bio_only_keep0_risk    | 309 |          0.469913 |        0.461128 |    -0.0125197   |          -0.108178    |
| val     | score_vehicle_badprob        | 309 |          0.498695 |        0.636923 |    -0.0203359   |           0.0633142   |
| val     | score_vehicle_bio_badprob    | 309 |          0.485025 |        0.555001 |    -0.00798825  |          -0.0438158   |
| val     | score_bio_only_badprob       | 309 |          0.50419  |        0.453934 |     6.50774e-05 |          -0.0511273   |
| val     | score_vehicle_oracle_gap     | 309 |          0.506251 |        0.555233 |     0.0401328   |           0.0858766   |
| val     | score_vehicle_bio_oracle_gap | 309 |          0.569103 |        0.555465 |     0.0858066   |           0.0910653   |
| test    | score_vehicle_gain           | 184 |          0.432373 |        0.452313 |    -0.0784304   |          -0.115087    |
| test    | score_vehicle_bio_gain       | 184 |          0.460686 |        0.411483 |    -0.152948    |          -0.14677     |
| test    | score_bio_only_gain          | 184 |          0.4803   |        0.573525 |    -0.00959476  |           0.0359138   |
| test    | score_vehicle_keep0_risk     | 184 |          0.447723 |        0.472408 |    -0.0935563   |           0.0116488   |
| test    | score_vehicle_bio_keep0_risk | 184 |          0.526352 |        0.386603 |    -0.0314134   |           0.000288956 |
| test    | score_bio_only_keep0_risk    | 184 |          0.494968 |        0.475598 |    -0.0474206   |          -0.00563774  |
| test    | score_vehicle_badprob        | 184 |          0.40423  |        0.393939 |    -0.146401    |          -0.0748627   |
| test    | score_vehicle_bio_badprob    | 184 |          0.508272 |        0.617544 |     0.0810811   |           0.141076    |
| test    | score_bio_only_badprob       | 184 |          0.463244 |        0.63764  |     0.0089722   |           0.0657833   |
| test    | score_vehicle_oracle_gap     | 184 |          0.496674 |        0.441467 |    -0.0684267   |          -0.0433222   |
| test    | score_vehicle_bio_oracle_gap | 184 |          0.533686 |        0.445933 |     0.017478    |           0.0556529   |

## Test bad_top10 等待前沿

| strategy                            | score                          |   target_wait_rate |   n |   selected_tail_rmse_mean |   selected_latest_rate |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   improve_rate_vs_keep0 |
|:------------------------------------|:-------------------------------|-------------------:|----:|--------------------------:|-----------------------:|----------------------------------:|-----------------------------------:|------------------------:|
| policy_keep_0ms_anchor              | policy_keep_0ms_anchor         |                0   |  19 |                  1.19771  |              0         |                         0         |                        0.502658    |               0         |
| policy_wait_to_latest_anchor        | policy_wait_to_latest_anchor   |                1   |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| oracle_best_anchor_upper_bound      | oracle_best_anchor_upper_bound |              nan   |  19 |                  0.612475 |              0.368421  |                        -0.585231  |                       -0.082573    |               1         |
| score_vehicle_gain_wr0.10           | score_vehicle_gain             |                0.1 |  19 |                  1.15253  |              0.105263  |                        -0.0451797 |                        0.457478    |               0.105263  |
| score_vehicle_gain_wr0.20           | score_vehicle_gain             |                0.2 |  19 |                  1.01215  |              0.315789  |                        -0.185555  |                        0.317102    |               0.315789  |
| score_vehicle_gain_wr0.30           | score_vehicle_gain             |                0.3 |  19 |                  0.940354 |              0.421053  |                        -0.257352  |                        0.245306    |               0.421053  |
| score_vehicle_gain_wr0.40           | score_vehicle_gain             |                0.4 |  19 |                  0.940354 |              0.421053  |                        -0.257352  |                        0.245306    |               0.421053  |
| score_vehicle_gain_wr0.50           | score_vehicle_gain             |                0.5 |  19 |                  0.940354 |              0.421053  |                        -0.257352  |                        0.245306    |               0.421053  |
| score_vehicle_gain_wr0.60           | score_vehicle_gain             |                0.6 |  19 |                  0.798315 |              0.736842  |                        -0.399391  |                        0.103267    |               0.736842  |
| score_vehicle_gain_wr0.70           | score_vehicle_gain             |                0.7 |  19 |                  0.752834 |              0.789474  |                        -0.444873  |                        0.0577852   |               0.789474  |
| score_vehicle_gain_wr0.80           | score_vehicle_gain             |                0.8 |  19 |                  0.736715 |              0.842105  |                        -0.460991  |                        0.0416669   |               0.842105  |
| score_vehicle_gain_wr0.90           | score_vehicle_gain             |                0.9 |  19 |                  0.695936 |              0.947368  |                        -0.50177   |                        0.000887764 |               0.947368  |
| score_vehicle_gain_wr1.00           | score_vehicle_gain             |                1   |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| score_vehicle_bio_gain_wr0.10       | score_vehicle_bio_gain         |                0.1 |  19 |                  1.15147  |              0.157895  |                        -0.0462412 |                        0.456417    |               0.157895  |
| score_vehicle_bio_gain_wr0.20       | score_vehicle_bio_gain         |                0.2 |  19 |                  1.09866  |              0.263158  |                        -0.0990485 |                        0.403609    |               0.263158  |
| score_vehicle_bio_gain_wr0.30       | score_vehicle_bio_gain         |                0.3 |  19 |                  1.08599  |              0.315789  |                        -0.111712  |                        0.390945    |               0.315789  |
| score_vehicle_bio_gain_wr0.40       | score_vehicle_bio_gain         |                0.4 |  19 |                  1.0653   |              0.368421  |                        -0.132409  |                        0.370249    |               0.368421  |
| score_vehicle_bio_gain_wr0.50       | score_vehicle_bio_gain         |                0.5 |  19 |                  0.954979 |              0.578947  |                        -0.242727  |                        0.259931    |               0.578947  |
| score_vehicle_bio_gain_wr0.60       | score_vehicle_bio_gain         |                0.6 |  19 |                  0.903109 |              0.631579  |                        -0.294598  |                        0.20806     |               0.631579  |
| score_vehicle_bio_gain_wr0.70       | score_vehicle_bio_gain         |                0.7 |  19 |                  0.88699  |              0.684211  |                        -0.310716  |                        0.191942    |               0.684211  |
| score_vehicle_bio_gain_wr0.80       | score_vehicle_bio_gain         |                0.8 |  19 |                  0.831406 |              0.842105  |                        -0.3663    |                        0.136358    |               0.842105  |
| score_vehicle_bio_gain_wr0.90       | score_vehicle_bio_gain         |                0.9 |  19 |                  0.711219 |              0.947368  |                        -0.486488  |                        0.0161703   |               0.947368  |
| score_vehicle_bio_gain_wr1.00       | score_vehicle_bio_gain         |                1   |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| score_bio_only_gain_wr0.10          | score_bio_only_gain            |                0.1 |  19 |                  1.11722  |              0.315789  |                        -0.0804891 |                        0.422169    |               0.315789  |
| score_bio_only_gain_wr0.20          | score_bio_only_gain            |                0.2 |  19 |                  1.09261  |              0.368421  |                        -0.105098  |                        0.39756     |               0.368421  |
| score_bio_only_gain_wr0.30          | score_bio_only_gain            |                0.3 |  19 |                  0.89066  |              0.578947  |                        -0.307046  |                        0.195612    |               0.578947  |
| score_bio_only_gain_wr0.40          | score_bio_only_gain            |                0.4 |  19 |                  0.89066  |              0.578947  |                        -0.307046  |                        0.195612    |               0.578947  |
| score_bio_only_gain_wr0.50          | score_bio_only_gain            |                0.5 |  19 |                  0.83228  |              0.684211  |                        -0.365427  |                        0.137231    |               0.684211  |
| score_bio_only_gain_wr0.60          | score_bio_only_gain            |                0.6 |  19 |                  0.789316 |              0.789474  |                        -0.40839   |                        0.0942674   |               0.789474  |
| score_bio_only_gain_wr0.70          | score_bio_only_gain            |                0.7 |  19 |                  0.789316 |              0.789474  |                        -0.40839   |                        0.0942674   |               0.789474  |
| score_bio_only_gain_wr0.80          | score_bio_only_gain            |                0.8 |  19 |                  0.743834 |              0.842105  |                        -0.453872  |                        0.0487857   |               0.842105  |
| score_bio_only_gain_wr0.90          | score_bio_only_gain            |                0.9 |  19 |                  0.711219 |              0.947368  |                        -0.486488  |                        0.0161703   |               0.947368  |
| score_bio_only_gain_wr1.00          | score_bio_only_gain            |                1   |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| score_vehicle_keep0_risk_wr0.10     | score_vehicle_keep0_risk       |                0.1 |  19 |                  1.16449  |              0.0526316 |                        -0.0332172 |                        0.469441    |               0.0526316 |
| score_vehicle_keep0_risk_wr0.20     | score_vehicle_keep0_risk       |                0.2 |  19 |                  1.11168  |              0.157895  |                        -0.0860244 |                        0.416633    |               0.157895  |
| score_vehicle_keep0_risk_wr0.30     | score_vehicle_keep0_risk       |                0.3 |  19 |                  0.991494 |              0.263158  |                        -0.206212  |                        0.296446    |               0.263158  |
| score_vehicle_keep0_risk_wr0.40     | score_vehicle_keep0_risk       |                0.4 |  19 |                  0.991494 |              0.263158  |                        -0.206212  |                        0.296446    |               0.263158  |
| score_vehicle_keep0_risk_wr0.50     | score_vehicle_keep0_risk       |                0.5 |  19 |                  0.968042 |              0.315789  |                        -0.229664  |                        0.272993    |               0.315789  |
| score_vehicle_keep0_risk_wr0.60     | score_vehicle_keep0_risk       |                0.6 |  19 |                  0.950862 |              0.421053  |                        -0.246844  |                        0.255814    |               0.421053  |
| score_vehicle_keep0_risk_wr0.70     | score_vehicle_keep0_risk       |                0.7 |  19 |                  0.794082 |              0.684211  |                        -0.403624  |                        0.0990338   |               0.684211  |
| score_vehicle_keep0_risk_wr0.80     | score_vehicle_keep0_risk       |                0.8 |  19 |                  0.781418 |              0.736842  |                        -0.416288  |                        0.0863698   |               0.736842  |
| score_vehicle_keep0_risk_wr0.90     | score_vehicle_keep0_risk       |                0.9 |  19 |                  0.707254 |              0.947368  |                        -0.490452  |                        0.0122054   |               0.947368  |
| score_vehicle_keep0_risk_wr1.00     | score_vehicle_keep0_risk       |                1   |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| score_vehicle_bio_keep0_risk_wr0.10 | score_vehicle_bio_keep0_risk   |                0.1 |  19 |                  1.16449  |              0.0526316 |                        -0.0332172 |                        0.469441    |               0.0526316 |
| score_vehicle_bio_keep0_risk_wr0.20 | score_vehicle_bio_keep0_risk   |                0.2 |  19 |                  1.1636   |              0.105263  |                        -0.0341049 |                        0.468553    |               0.105263  |
| score_vehicle_bio_keep0_risk_wr0.30 | score_vehicle_bio_keep0_risk   |                0.3 |  19 |                  1.1636   |              0.105263  |                        -0.0341049 |                        0.468553    |               0.105263  |
| score_vehicle_bio_keep0_risk_wr0.40 | score_vehicle_bio_keep0_risk   |                0.4 |  19 |                  1.1636   |              0.105263  |                        -0.0341049 |                        0.468553    |               0.105263  |
| score_vehicle_bio_keep0_risk_wr0.50 | score_vehicle_bio_keep0_risk   |                0.5 |  19 |                  1.11812  |              0.157895  |                        -0.0795867 |                        0.423071    |               0.157895  |
| score_vehicle_bio_keep0_risk_wr0.60 | score_vehicle_bio_keep0_risk   |                0.6 |  19 |                  0.94696  |              0.368421  |                        -0.250747  |                        0.251911    |               0.368421  |
| score_vehicle_bio_keep0_risk_wr0.70 | score_vehicle_bio_keep0_risk   |                0.7 |  19 |                  0.838417 |              0.736842  |                        -0.35929   |                        0.143368    |               0.736842  |
| score_vehicle_bio_keep0_risk_wr0.80 | score_vehicle_bio_keep0_risk   |                0.8 |  19 |                  0.805758 |              0.842105  |                        -0.391948  |                        0.11071     |               0.842105  |
| score_vehicle_bio_keep0_risk_wr0.90 | score_vehicle_bio_keep0_risk   |                0.9 |  19 |                  0.793094 |              0.894737  |                        -0.404612  |                        0.0980457   |               0.894737  |
| score_vehicle_bio_keep0_risk_wr1.00 | score_vehicle_bio_keep0_risk   |                1   |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| score_vehicle_badprob_wr0.10        | score_vehicle_badprob          |                0.1 |  19 |                  1.08978  |              0.105263  |                        -0.107923  |                        0.394735    |               0.105263  |
| score_vehicle_badprob_wr0.20        | score_vehicle_badprob          |                0.2 |  19 |                  1.07621  |              0.210526  |                        -0.1215    |                        0.381158    |               0.210526  |
| score_vehicle_badprob_wr0.30        | score_vehicle_badprob          |                0.3 |  19 |                  1.02429  |              0.263158  |                        -0.173419  |                        0.329239    |               0.263158  |
| score_vehicle_badprob_wr0.40        | score_vehicle_badprob          |                0.4 |  19 |                  1.02429  |              0.263158  |                        -0.173419  |                        0.329239    |               0.263158  |
| score_vehicle_badprob_wr0.50        | score_vehicle_badprob          |                0.5 |  19 |                  0.999773 |              0.368421  |                        -0.197933  |                        0.304724    |               0.368421  |
| score_vehicle_badprob_wr0.60        | score_vehicle_badprob          |                0.6 |  19 |                  0.947902 |              0.421053  |                        -0.249804  |                        0.252854    |               0.421053  |
| score_vehicle_badprob_wr0.70        | score_vehicle_badprob          |                0.7 |  19 |                  0.931784 |              0.473684  |                        -0.265922  |                        0.236736    |               0.473684  |
| score_vehicle_badprob_wr0.80        | score_vehicle_badprob          |                0.8 |  19 |                  0.866376 |              0.578947  |                        -0.33133   |                        0.171328    |               0.578947  |
| score_vehicle_badprob_wr0.90        | score_vehicle_badprob          |                0.9 |  19 |                  0.790702 |              0.789474  |                        -0.407005  |                        0.0956533   |               0.789474  |
| score_vehicle_badprob_wr1.00        | score_vehicle_badprob          |                1   |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| score_vehicle_bio_badprob_wr0.10    | score_vehicle_bio_badprob      |                0.1 |  19 |                  1.08978  |              0.105263  |                        -0.107923  |                        0.394735    |               0.105263  |
| score_vehicle_bio_badprob_wr0.20    | score_vehicle_bio_badprob      |                0.2 |  19 |                  0.99829  |              0.263158  |                        -0.199416  |                        0.303242    |               0.263158  |
| score_vehicle_bio_badprob_wr0.30    | score_vehicle_bio_badprob      |                0.3 |  19 |                  0.933277 |              0.421053  |                        -0.264429  |                        0.238229    |               0.421053  |
| score_vehicle_bio_badprob_wr0.40    | score_vehicle_bio_badprob      |                0.4 |  19 |                  0.860662 |              0.631579  |                        -0.337045  |                        0.165613    |               0.631579  |
| score_vehicle_bio_badprob_wr0.50    | score_vehicle_bio_badprob      |                0.5 |  19 |                  0.860662 |              0.631579  |                        -0.337045  |                        0.165613    |               0.631579  |
| score_vehicle_bio_badprob_wr0.60    | score_vehicle_bio_badprob      |                0.6 |  19 |                  0.860662 |              0.631579  |                        -0.337045  |                        0.165613    |               0.631579  |
| score_vehicle_bio_badprob_wr0.70    | score_vehicle_bio_badprob      |                0.7 |  19 |                  0.787273 |              0.789474  |                        -0.410433  |                        0.0922244   |               0.789474  |
| score_vehicle_bio_badprob_wr0.80    | score_vehicle_bio_badprob      |                0.8 |  19 |                  0.715745 |              0.947368  |                        -0.481962  |                        0.0206961   |               0.947368  |
| score_vehicle_bio_badprob_wr0.90    | score_vehicle_bio_badprob      |                0.9 |  19 |                  0.715745 |              0.947368  |                        -0.481962  |                        0.0206961   |               0.947368  |
| score_vehicle_bio_badprob_wr1.00    | score_vehicle_bio_badprob      |                1   |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| score_bio_only_badprob_wr0.10       | score_bio_only_badprob         |                0.1 |  19 |                  1.15228  |              0.105263  |                        -0.0454226 |                        0.457235    |               0.105263  |
| score_bio_only_badprob_wr0.20       | score_bio_only_badprob         |                0.2 |  19 |                  1.12767  |              0.157895  |                        -0.0700313 |                        0.432626    |               0.157895  |
| score_bio_only_badprob_wr0.30       | score_bio_only_badprob         |                0.3 |  19 |                  1.022    |              0.368421  |                        -0.175711  |                        0.326947    |               0.368421  |
| score_bio_only_badprob_wr0.40       | score_bio_only_badprob         |                0.4 |  19 |                  0.937788 |              0.526316  |                        -0.259919  |                        0.242739    |               0.526316  |
| score_bio_only_badprob_wr0.50       | score_bio_only_badprob         |                0.5 |  19 |                  0.885917 |              0.578947  |                        -0.311789  |                        0.190869    |               0.578947  |
| score_bio_only_badprob_wr0.60       | score_bio_only_badprob         |                0.6 |  19 |                  0.752004 |              0.789474  |                        -0.445702  |                        0.0569558   |               0.789474  |
| score_bio_only_badprob_wr0.70       | score_bio_only_badprob         |                0.7 |  19 |                  0.752004 |              0.789474  |                        -0.445702  |                        0.0569558   |               0.789474  |
| score_bio_only_badprob_wr0.80       | score_bio_only_badprob         |                0.8 |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| score_bio_only_badprob_wr0.90       | score_bio_only_badprob         |                0.9 |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |
| score_bio_only_badprob_wr1.00       | score_bio_only_badprob         |                1   |  19 |                  0.695048 |              1         |                        -0.502658  |                        0           |               1         |

## 判读

- score_bio_only_badprob: best tail=0.6950, latest_rate=1.000, target_wait_rate=0.80.
- score_bio_only_gain: best tail=0.6950, latest_rate=1.000, target_wait_rate=1.00.
- score_bio_only_keep0_risk: best tail=0.6950, latest_rate=1.000, target_wait_rate=1.00.
- score_vehicle_badprob: best tail=0.6950, latest_rate=1.000, target_wait_rate=1.00.
- score_vehicle_bio_badprob: best tail=0.6950, latest_rate=1.000, target_wait_rate=1.00.
- score_vehicle_bio_gain: best tail=0.6950, latest_rate=1.000, target_wait_rate=1.00.
- score_vehicle_bio_keep0_risk: best tail=0.6950, latest_rate=1.000, target_wait_rate=1.00.
- score_vehicle_bio_oracle_gap: best tail=0.7480, latest_rate=0.895, target_wait_rate=1.00.
- score_vehicle_gain: best tail=0.6950, latest_rate=1.000, target_wait_rate=1.00.
- score_vehicle_keep0_risk: best tail=0.6950, latest_rate=1.000, target_wait_rate=1.00.
- score_vehicle_oracle_gap: best tail=0.6950, latest_rate=1.000, target_wait_rate=1.00.
- vehicle+bio gain 相对 vehicle gain 的最佳前沿改变量为 +0.0000。
- vehicle+bio badprob 相对 vehicle badprob 的最佳前沿改变量为 +0.0000。
- 若 bio 分数不能在同等等待预算下稳定低于 vehicle 分数，则当前生理不能作为可部署风险校准增量。

## 关键图

- `figures\v265_test_badtop10_wait_frontier.png`