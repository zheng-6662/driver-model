# Task Definition And Event Logic

## Current Task Framing
- The task stays pooled post-trigger steering response prediction.
- This diagnostic measures how far the actual steering response onset lags behind the event trigger marker.
- Anchor logic matches the active training script and is reported as supporting context, not as a replacement target.

## Event Source Priority
- Prefer `*_events_v400_context.csv` for `road_type_anchor`, `curvature_anchor`, `trigger_type`, `phase_type`, and `trigger_idx`.
- Fall back to `*_events_v312.csv` when the v400 context file is missing.

## Anchor Logic Reused From Active Script
- Curve: `roll_peak` over the event segment.
- Straight: first `|steer_rate| >= 0.8 * max_abs(steer_rate)` within the event segment.

## Onset Logic Reused From Active Script
- Helper: `_first_threshold_crossing_idx_np`.
- Absolute threshold floor: `STEER_ONSET_THR_ABS`.
- Final threshold: `max(STEER_ONSET_THR_ABS, 0.15 * true_peak_delta)` measured on the trigger-to-response search window.

## Protocol Split
- Not enabled: join logic intentionally disabled because no unambiguous mapping rule was approved.

## Coverage
- Vehicle files scanned: `91`
- Vehicle files with usable paired events: `85`
- Missing v312 basenames: `5`
- Missing v400 basenames: `5`
- Strong events analyzed: `3737`
