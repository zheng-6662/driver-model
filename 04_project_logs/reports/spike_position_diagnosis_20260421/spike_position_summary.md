# Spike Position Diagnosis 2026-04-21

Generated at: `2026-04-21T21:22:24`

## Method

The primary diagnosis used existing `pred_vs_gt_example_*.png` prediction plots, because those artifacts contain the three plotted output channels (`steer_angle`, `yawrate`, `ay`).
For each plotted prediction channel, the tool extracted the orange predicted curve, subtracted a local moving-average baseline, and marked the largest absolute residual as the local spike candidate.
A channel counted as spiking when `abs(residual_px) >= 12.0`. A cross-channel spike counted as synchronized when at least two channels landed within `8.0` future-sample indices.

## Outputs

- `spike_index_hist.png`: histogram of synchronized spike indices by run.
- `cross_channel_spike_sync.csv`: per-example table with channel spike locations and recalc/sample-table context.
- `spike_position_summary.md`: this summary.

## Run Summary

| Run | prediction images | case rows | synchronized examples | histogram points | median sync index | median sync time (s) | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Run B | 8 | 528 | 5 | 13 | 126.052 | 0.631841 | prediction_plot_images |
| Run C | 8 | 528 | 8 | 19 | 167.390 | 0.839049 | prediction_plot_images |
| Run D | 8 | 528 | 6 | 16 | 138.737 | 0.695425 | prediction_plot_images |

## Interpretation

The detected local spike is strongly cross-channel-synchronized within each run: synchronized rows require at least two output channels to choose almost the same future index.
Across runs, the preferred spike band shifts rather than perfectly matching: Run B clusters earlier, Run C later, and Run D in between. That supports a shared decoder/timestep artifact hypothesis more than an isolated single-channel plotting artifact.
Because only rendered plots are available here, this is a position-level diagnosis. A waveform-level causal diagnosis would require saved raw prediction arrays or branch-level coarse/fine outputs.

## Fallbacks And Blockers

- No run had to fall back fully to recalc case tables; image-based prediction-plot diagnosis was available for B/C/D.
- Raw per-sample prediction arrays were not found in the B/C/D run folders, so the image-based diagnosis cannot separate coarse-branch, fine-branch, or decoder-token contributions.

Report directory: `F:\data_set_process\data_process\04_project_logs\reports\spike_position_diagnosis_20260421`
