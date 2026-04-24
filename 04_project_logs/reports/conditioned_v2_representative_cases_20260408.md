# conditioned v2 representative cases (2026-04-08)

This panel reuses formal-eval single-sample plots and reorders them to match the 2026-04-08 attribution findings.

## Q1_fast worst-case
- sample_key: `tyy::Entity_Recording_2025_09_28_14_23_43_vehicle_aligned_cleaned.csv::65::trigger_idx`
- latency bucket: `Q1_fast`
- morphology: `single_lobe`
- subject: `tyy`
- interaction slice: `unknown`
- delta tail RMSE: `+1.525`
- delta boundary shift: `-0.107`
- shape corr conditioned: `-0.747`
- peak abs amp err conditioned: `1.947`
- takeaway: Tail shape and amplitude both break badly; this is the clearest fast-reaction failure case.

## Shape-heavy failure
- sample_key: `tyy::Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv::55::trigger_idx`
- latency bucket: `Q2`
- morphology: `single_lobe`
- subject: `tyy`
- interaction slice: `unknown`
- delta tail RMSE: `+0.599`
- delta boundary shift: `-0.052`
- shape corr conditioned: `-0.769`
- peak abs amp err conditioned: `0.339`
- takeaway: Boundary does not worsen much, but the conditioned tail still deviates strongly in shape.

## Single-lobe boundary-heavy
- sample_key: `tyy::Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv::54::trigger_idx`
- latency bucket: `Q4_slow`
- morphology: `single_lobe`
- subject: `tyy`
- interaction slice: `interaction`
- delta tail RMSE: `+0.109`
- delta boundary shift: `+0.278`
- shape corr conditioned: `0.734`
- peak abs amp err conditioned: `0.126`
- takeaway: A single-lobe case where conditioned remains close overall but boundary shift grows noticeably.

## Reverse-correction contrast
- sample_key: `cwh::Entity_Recording_2025_09_26_19_27_21_vehicle_aligned_cleaned.csv::6::trigger_idx`
- latency bucket: `Q4_slow`
- morphology: `reverse_correction`
- subject: `cwh`
- interaction slice: `unknown`
- delta tail RMSE: `+0.067`
- delta boundary shift: `+0.080`
- shape corr conditioned: `0.973`
- peak abs amp err conditioned: `0.163`
- takeaway: Reverse-correction does worsen, but not nearly as catastrophically as the Q1_fast single-lobe failures.

## Improved control sample
- sample_key: `tyy::Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv::52::trigger_idx`
- latency bucket: `Q2`
- morphology: `multi_correction`
- subject: `tyy`
- interaction slice: `interaction`
- delta tail RMSE: `-0.445`
- delta boundary shift: `-0.760`
- shape corr conditioned: `0.012`
- peak abs amp err conditioned: `0.144`
- takeaway: A positive control: conditioned clearly helps here, so the main issue is not that the method never works.
