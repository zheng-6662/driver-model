# GPT Pro Effectiveness Follow-up Review

Date: `2026-04-23`
Source: GPT Pro review of the closed `2026-04-22` effectiveness round
Status: Recommendation only; no training run executed from this review yet

## Headline

The first bottleneck is not optimizer tuning or mild capacity.
The highest-value next step is an explicit anti-collapse attempt on the `1.5 s` branch.

## Core diagnosis

- `H15` has real gains under the absolute-window metric:
  - `rmse_steer: 0.5559 -> 0.4930`
  - `abs_tail_last_0p5s.rmse_steer: 0.7171 -> 0.6022`
- Those gains are not a fraction-tail artifact because D0 already switched the comparison to absolute last `0.5 s`.
- `H15` fails because it buys those gains by flattening the rare but important `strong_pos` late tail:
  - `strong_pos.tail_amp_ratio_pred_over_gt = 0.2687`
  - `strong_pos.tail_flatness_rate = 1.0000`
- `OPT_A_20` and `CAP_192_BEST` already argue against spending the next budget on optimizer or width sweeps.

## Recommended mainline

### Name

`H15_AC_CF_HLF_v1`

### Meaning

- `H15`: keep the `1.5 s` horizon
- `AC`: anti-collapse
- `CF`: coarse-fine steer decomposition
- `HLF`: hard-late fine residual supervision

### Why this is the preferred next run

- It starts from the branch that already showed real overall and absolute-tail gains.
- It uses existing code paths instead of reopening a broad search.
- It directly targets the two failing guardrails:
  - `strong_pos.tail_amp_ratio_pred_over_gt`
  - `strong_pos.tail_flatness_rate`

## Proposed configuration

- `DRIVER_MODEL_FUTURE_SEC=1.5`
- `DRIVER_MODEL_STEER_COARSE_FINE=1`
- `DRIVER_MODEL_HARD_LATE_FINE=1`
- `DRIVER_MODEL_HARD_LATE_START_SEC=1.00`
- `DRIVER_MODEL_HARD_TAIL_START_SEC=1.00`
- `DRIVER_MODEL_W_HARD_LATE_FINE=0.10`
- `DRIVER_MODEL_W_FINE_DC=0.01`
- `DRIVER_MODEL_W_TREND_COARSE=0.10`
- `DRIVER_MODEL_PHASE_ADAPTIVE_TREND=0`
- `DRIVER_MODEL_STRONG_POS_GATE=0`
- `DRIVER_MODEL_W_FIRSTREV_LOCAL=0.0`

## Why the late window must move to 1.00 s

For `future_sec=1.5`, the failing guardrail is the absolute last `0.5 s`, which corresponds to `1.0 s -> 1.5 s`.
If `HARD_LATE_START_SEC=1.25` and `HARD_TAIL_START_SEC=1.50` remain unchanged, the supervision window is misaligned with the actual failing slice.

So the review explicitly recommends aligning the anti-collapse supervision to:

- `HARD_LATE_START_SEC = 1.00`
- `HARD_TAIL_START_SEC = 1.00`

## Expected success criteria

- `strong_pos.tail_amp_ratio_pred_over_gt >= 0.60`
- `strong_pos.tail_flatness_rate <= 0.60`
- `abs_tail_last_0p5s.rmse_steer <= 0.66`
- `rmse_steer <= 0.53`
- `late_peak_recall >= 0.62`

## Failure criteria

- `strong_pos.tail_amp_ratio_pred_over_gt < 0.50`, or
- `strong_pos.tail_flatness_rate > 0.80`, or
- the strong-pos collapse is reduced but the run gives back too much on:
  - `rmse_steer`
  - `abs_tail_last_0p5s.rmse_steer`
  - `late_peak_recall`

## Plan B if the anti-collapse run fails

### Name

`H15_LATE_RESIDUAL_HEAD_v1`

### Meaning

Add a minimal late residual head for steer on the decoder output, active only for `t >= 1.0 s`.

### Why this becomes the next step

- If `H15_AC_CF_HLF_v1` fails, the issue is likely not just loss balancing.
- The next likely bottleneck is the late-slice representation itself.
- In that case, it is higher-value to add a minimal late residual output path than to keep micro-sweeping loss weights.

## Current module mapping for implementation

Because the V5.8 line has already been modularized, the relevant implementation locations are now:

- `v58_modular/config.py`
  - env flags and default values
- `v58_modular/modeling.py`
  - coarse-fine forward path and any later late-residual-head addition
- `v58_modular/losses.py`
  - hard-late mask construction and coarse-fine loss composition
- `v58_modular/train.py`
  - model wiring, logging, run config export, and training/validation history
- `v58_modular/evaluation.py`
  - review outputs that confirm whether late-tail collapse is reduced

## Explicitly de-prioritized

- more optimizer sweeps
- more mild width-only sweeps
- direct promotion of `H10`
- reopening broad bridge / gate / loss matrices as the main line
