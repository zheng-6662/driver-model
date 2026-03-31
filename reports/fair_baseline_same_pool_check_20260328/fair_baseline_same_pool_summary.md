# Fair baseline check on the same sample pool

## Sample pool
- manifest: `F:\data_set_process\data_process\datasetprocess\final_code\model\training\protocol_allphase_control_v2_context_full2s\sample_manifest.csv`
- total samples: **6238**
- train: **4797**
- val: **692**
- test: **749**

## Conclusion
- The formal baseline used in `EXP2_ALLPHASE_V2_CONTEXT_FULL2S_TRUE2S_SUP_20260324_224343` already uses the **same sample pool and same split counts** as the current conditioned-v2 run.
- So the fair comparison is **valid at the sample-count level**.
- What was misleading was the earlier illustrative baseline figure, not the formal evaluation pool.

## Fair comparison metrics already available on the same pool
- baseline overall 2s RMSE: **0.3807**
- conditioned v2 overall 2s RMSE: **0.3773**
- baseline tail RMSE: **0.3978**
- conditioned v2 tail RMSE: **0.3758**
- baseline turning count abs err: **1.7717**
- conditioned v2 turning count abs err: **1.5354**
- baseline interaction-slice tail RMSE: **0.4954**
- conditioned v2 interaction-slice tail RMSE: **0.4207**

## Recommendation for the PPT
- Replace the old baseline illustration with the figure copied here as `fair_same_pool_representative_samples_overview.png`
- If needed, cite the sample-pool counts above directly in the talk to remove any concern about fairness.
