# Effectiveness Follow-up Summary

- Manifest: `F:\data_set_process\data_process\04_project_logs\reports\effectiveness_followup_20260422\effectiveness_followup_manifest.json`
- Comparison table: `F:\data_set_process\data_process\04_project_logs\reports\effectiveness_followup_20260422\effectiveness_comparison_table.csv`
- D0 table: `F:\data_set_process\data_process\04_project_logs\reports\effectiveness_followup_20260422\d0_comparison_table.csv`
- Ranking table: `F:\data_set_process\data_process\04_project_logs\reports\effectiveness_followup_20260422\effectiveness_ranking_table.csv`

## Fraction Tail Bias Note

- Fraction-based `tail_rmse_steer` uses the last 25% of the horizon.
- That means the native tail window shrinks from `100` steps at `2.0s` to `75` at `1.5s` and `50` at `1.0s`.
- Cross-horizon comparisons should therefore use `abs_tail_last_0p5s.rmse_steer` as the primary tail metric.

## D0 Anchors

```csv
action_name,base_reference,selection_source,rmse_steer,tail_rmse_steer,prefix_1p0s_rmse_steer,prefix_1p5s_rmse_steer,full_horizon_rmse_steer,abs_tail_last_0p5s_rmse_steer,strong_pos_tail_amp_ratio_pred_over_gt,strong_pos_tail_flatness_rate
D0_BASELINE,baseline_fixed_input,best_by_structured,0.5559393763542175,0.7171384919281619,0.42913099553302125,0.4905734249782061,0.5559394013601239,0.7171384915440451,1.3489819546120279,0.21052631578947367
D0_RUNA,Run A,best_by_structured,35.24658203125,42.4695019213555,27.980389023611853,32.48398617422527,35.246583759855824,42.46950192385535,0.2908007811723935,0.8421052631578947
```

## Run Table

```csv
action_name,phase,mode,selection_source,future_sec,optimizer,lr,weight_decay,scheduler,rmse_steer,abs_tail_last_0p5s_rmse_steer,late_peak_recall,first_reversal_time_mae_sec,reversal_count_exact_match_rate,hard_collapse,single_guardrail_alert
D0_BASELINE,D0,,best_by_structured,,,,,,0.5559393763542175,0.7171384915440451,0.6495726495726496,0.5107232704402515,0.5151515151515151,False,False
D0_RUNA,D0,,best_by_structured,,,,,,35.24658203125,42.46950192385535,0.6495726495726496,0.4162878787878788,0.6685606060606061,True,False
H15_SMOKE,Stage1,smoke,best_by_structured,1.5,adam,0.001,0.0,none,0.8147646188735962,0.8943485435325115,0.15384615384615385,,0.6704545454545454,True,False
H15,Stage1,full,best_by_structured,1.5,adam,0.001,0.0,none,0.49303138256073,0.6021910695312034,0.6354515050167224,0.39874999999999994,0.6628787878787878,True,False
OPT_A_20,Stage1,full,best_by_structured,2.0,adamw,0.001,0.0001,cosine,0.5886715054512024,0.7697850083155989,0.5726495726495726,0.41490909090909095,0.5227272727272727,False,False
H10,Stage3,full,best_by_structured,1.0,adam,0.001,0.0,none,0.43701404333114624,0.5271683015231194,0.7880184331797235,0.1942063492063492,0.7234848484848485,True,False
OPT_C_BEST,Stage3,full,best_by_structured,2.0,adam,0.001,0.0005,none,0.5650014877319336,0.7263117430925224,0.6239316239316239,0.475505617977528,0.39204545454545453,False,False
CAP_192_BEST,Stage3,full,best_by_structured,2.0,adam,0.001,0.0,none,0.6136959195137024,0.8224012097738663,0.5854700854700855,0.48165714285714295,0.42424242424242425,False,False
WINNER_CONFIRM,Reserve,full,best_by_structured,2.0,adam,0.001,0.0,none,0.5633100867271423,0.7209796387635579,0.6581196581196581,0.5569047619047619,0.49053030303030304,False,False
```

## Provisional Ranking

- Ranking includes only full training actions recorded in this follow-up manifest.
- Recalc-only anchors such as `D0_BASELINE` are excluded from the ranking table and must still be compared separately as fixed references.

```csv
rank,action_name,future_sec,optimizer,lr,abs_tail_last_0p5s_rmse_steer,rmse_steer,late_peak_recall,strong_pos_tail_amp_ratio_pred_over_gt,strong_pos_tail_flatness_rate
1,WINNER_CONFIRM,2.0,adam,0.001,0.7209796387635579,0.5633100867271423,0.6581196581196581,1.4113137337905708,0.2631578947368421
2,OPT_C_BEST,2.0,adam,0.001,0.7263117430925224,0.5650014877319336,0.6239316239316239,0.7856163148048031,0.5789473684210527
3,OPT_A_20,2.0,adamw,0.001,0.7697850083155989,0.5886715054512024,0.5726495726495726,1.5684597734762156,0.21052631578947367
4,CAP_192_BEST,2.0,adam,0.001,0.8224012097738663,0.6136959195137024,0.5854700854700855,0.8707421919749552,0.3684210526315789
```
