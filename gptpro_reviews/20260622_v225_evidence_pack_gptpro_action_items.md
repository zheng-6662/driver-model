# v225 GPTPro action items

## Required script

- `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v225_formal_route_reconstruction_evidence_pack_20260622.py`

## Required output directory

- `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/`

## Required tables

- `tables/formal_model_lock.csv`
- `tables/formal_reconstruction_metrics_overall.csv`
- `tables/formal_reconstruction_metrics_by_pool.csv`
- `tables/formal_reconstruction_metrics_by_bucket.csv`
- `tables/formal_reconstruction_metrics_by_route_event.csv`
- `tables/per_sample_formal_reconstruction_eval.csv`
- `tables/formal_failure_case_index.csv`
- `tables/diagnostic_only_v222a_closeout_summary.csv`
- `tables/excluded_diagnostic_models_audit.csv`

## Required figures

- `figures/formal_examples/loose_main_pool/`
- `figures/formal_examples/strict_main_pool/`
- `figures/worst_tail_cases/loose_main_pool/`
- `figures/worst_tail_cases/strict_main_pool/`
- `figures/strong_under_cases/loose_main_pool/`
- `figures/strong_under_cases/strict_main_pool/`
- `figures/baseline_sufficient_cases/loose_main_pool/`
- `figures/baseline_sufficient_cases/strict_main_pool/`

Minimum counts:

- `formal_examples >= 12 PNG`
- `worst_tail_cases >= 12 PNG`
- `strong_under_cases >= 8 PNG`
- `baseline_sufficient_cases >= 8 PNG`

Figure title must display:

- `pool`
- `sample_id`
- `formal_model`
- `RMSE`
- `tail RMSE`
- `under flag`

## Required report

- `reports/v225_formal_route_reconstruction_evidence_cn.md`

## Required logs

- `logs/run_manifest.json`
- `logs/leakage_guard_report.json`
- `logs/forbidden_scan_report.json`
- `logs/metric_reproduction_check.json`
- `logs/file_inventory.json`

## Required zip

- `v225_formal_route_reconstruction_evidence_pack.zip`

## Verification criteria

- `python -m py_compile` pass
- script full run pass
- ZIP `bad_file=None`
- required files missing `[]`
- locked baseline metrics reproduce within absolute tolerance `<= 1e-5`:
  - `loose_main_pool / avg_joint_focus`: RMSE `0.544884`, tail RMSE `0.629752`
  - `strict_main_pool / peak_floor_090`: RMSE `0.571770`, tail RMSE `0.658306`
- formal / diagnostic separation passes
- leakage guard all pass
- row count and sample_id consistency passes
- no duplicate sample_id within pool/split
- no missing formal prediction
- prediction shape `N x 21`
- horizon length `21`
