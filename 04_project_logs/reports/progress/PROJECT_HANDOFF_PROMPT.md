# Project Handoff Prompt

When opening a new window or thread for `F:\data_set_process\data_process`, use this one-liner:

```text
用 $data-process-model-handoff 接手 F:\data_set_process\data_process，继续这个项目；先按 skill 规定读取当前进展锚点，再直接从当前主线继续推进，不要重新分析旧结论。
```

If the skill does not auto-trigger, paste the same line again explicitly with `$data-process-model-handoff`.

Primary progress anchors:

- `reports/progress/experiment_registry.md`
- `reports/progress/daily/2026-04-16.md`
- `datasetprocess/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
- `tools/recalc_v58_checkpoint_with_current_metrics.py`
