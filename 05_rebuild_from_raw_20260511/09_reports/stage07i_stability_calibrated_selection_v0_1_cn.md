# Stage 7i 技术报告：stability-calibrated non-oracle selection v0.1

## Scope

- Source candidates: Stage 7g.
- No new trajectory model training.
- Selection uses train/val only; test is final reporting.
- No server used. Credential file not read.

## Selected Rule

- policy=`stability_penalty_l05`
- selected_model=`segment_resid_rf_blend_25`
- test_delta_vs_rbf=-0.005620
- difficult_delta_vs_rbf=-0.029588

## Gate

```text
                  gate_item                  status                                                                                        evidence
selected_calibration_policy   stability_penalty_l05                                                        selected_model=segment_resid_rf_blend_25
         deployable_upgrade weak_candidate_continue              test RMSE delta -0.005620; difficult RMSE delta -0.029588; needs repeat validation
           mainline_upgrade               not_final single split evidence only; no repeated validation or held-out confirmation beyond current test
 stage08_physio_eeg_allowed                 blocked                                 vehicle-only selection calibration still needs robustness check
                server_used                      no                                   local diagnostic/selection run only; credential file not read
```
