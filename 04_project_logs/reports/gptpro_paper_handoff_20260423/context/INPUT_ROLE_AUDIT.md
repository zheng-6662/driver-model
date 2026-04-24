# Input Role Audit

## Status
- Task H scaffold drafted on 2026-04-21.
- This file fixes defense-safe wording before first-wave diagnostics fill the open evidence slots.

## Fixed Task Definition
- The maintained line is a pooled post-trigger steering response prediction task.
- Sample unit: one event-aligned sample built from 3.0 s pre-trigger vehicle history and a 2.0 s future response window.
- The trigger is not a scene-start marker. It is the maintained event anchor:
  - curve-like events use the local roll peak
  - straight / emergency-lane-change-like events use the earliest steer-rate point that reaches 80% of the local peak
- The primary interpretive target is future steering response after the trigger.
- Future yaw rate and future lateral acceleration remain auxiliary regression targets used to stabilize multi-task learning.
- Training is pooled across mixed subjects, mixed events, and mixed road contexts. It is not scene-specific training and not a per-scene expert system.

## Contribution Boundary Wording
Use this wording unless first-wave diagnostics force a revision:

> This work studies pooled post-trigger steering response prediction. Given a fixed pre-trigger vehicle-history window, an event-aligned trigger state, a driver-style prior, and available future road-geometry preview, the model predicts short-horizon steering response patterns and is evaluated on whether it preserves response timing, late-peak recovery, and reversal structure. The contribution is a pooled response-prediction formulation plus an evaluation and checkpoint-selection protocol for response fidelity, not a scene-specific planner and not a causal model of human cognition.

## Claims That Are In Bounds
- A single pooled model can predict short-horizon post-trigger steering response patterns better than the fixed baseline under some objectives.
- Different checkpoint-selection rules expose a real trade-off between overall / tail fit and reversal-structure fidelity.
- Road preview and anchor construction matter for response prediction quality.
- Pooled training can be audited with explicit subgroup and confound checks instead of being treated as a scene-specific result.

## Claims That Are Out Of Bounds
- "The model understands driver cognition" or any direct causal cognition claim.
- "The model solves scene-specific trajectory planning."
- "The model is trained separately for each scenario or scene family."
- "The current active script fully reproduces the historical 220918 anchor." It does not.
- "Future road geometry is unnecessary." The current mainline uses future curvature preview.
- "Runtime physio or EEG is required at inference." Teacher-side signals are training-time only.

## Input-Role Taxonomy
| Role ID | Family | Concrete fields in the current script | Time relation | Used by deployed forward pass | Used only in training or audit | Defense-safe interpretation | First-wave TODO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | History encoder source | roll, steer, yaw rate, ay, ax, speed, z, lane error, curvature, `LTR_est`, `steer_rate`, lane derivatives | 3.0 s before anchor | yes | no | core pre-trigger vehicle state history | TODO: add the final kept-column list from the feature dump |
| R2 | Anchor dynamic context | `steer_anchor`, anchor-time `steer_rate`, `ay`, `yawrate` | at trigger | yes | no | initial post-trigger state summary, not a future label | TODO: confirm whether any one field dominates keeper checkpoints |
| R3 | Driver prior | `style_id` appended into `ctx` | pre-assigned subject/style cluster | yes | no | pooled subject prior, not a scene identifier | TODO: quantify whether `style_id` behaves as helpful prior or leakage-like shortcut |
| R4 | Future road preview | speed-projected future curvature sequence, `curve_norm` | exogenous future path geometry | yes | no | conditional road-shape preview; this makes the task conditional response prediction, not blind free-roll forecasting | TODO: record exact defense wording after the preview-boundary audit |
| R5 | Teacher-side privileged signal | 4 physio window means plus 8 EEG event features mapped to teacher latent targets | event-level side information | no | yes | training-time privileged supervision only; not a runtime requirement | TODO: cite the exact state-distill setting used in the defense run |
| R6 | Response-structure supervision | `rev_gt`, `rev_gt_weak`, `rev_gt_strong`, peak and reversal derived losses | future-label side supervision | no | yes | labels used to shape training and checkpoint selection, not to drive inference | TODO: fill the final wording for strong vs weak reversal labels |
| R7 | Metadata for splitting and subgroup analysis | `subject_id`, `event_level`, road-type flag, `curve_score_event_mean_abs`, `anchor_source_applied` | audit metadata | no | yes | analysis keys used for pooling, fairness, and stratified reporting | TODO: attach subgroup audit summary once available |
| R8 | Reporting-only artifacts | `best_by_loss`, `best_by_structured`, visual issue case panels | after training | no | yes | evaluation and selection layer, not model input | TODO: decide whether defense centers best-by-structured only or a dual-checkpoint comparison |

## Input-Role Notes That Must Stay Clear In The Defense
- The task is not "predict the whole future scene from nothing." It is "predict steering response after a trigger under known local state and available road preview."
- The same pooled model sees straight and curve cases; road type influences anchor construction and analysis, not a separate expert branch.
- Future curvature preview is a legitimate conditioning signal only if the defense states that the task is conditional on known road geometry.
- Teacher-side physio and EEG information must never be described as runtime input.
- Style information must be described as a pooled subject prior, not as proof of person-specific personalization unless diagnostics later support that narrower claim.

## Confound Focus List
These confounds should later map directly into the defense tables and next-decision rules.

| Confound ID | Confound question | Why it matters | Current stance | TODO owner |
| --- | --- | --- | --- | --- |
| CF1 | Does future curvature preview make the task look more scene-aware than it really is? | Could overstate general prediction ability if not framed as conditional preview | open wording risk | TODO |
| CF2 | Does `style_id` behave like subject leakage instead of a mild prior? | Could turn pooled prediction into hidden identity lookup | open | TODO |
| CF3 | Does straight-vs-curve anchor selection create hidden two-task behavior? | Could blur whether gains come from a shared predictor or from anchor design | known risk, still allowed if stated clearly | TODO |
| CF4 | Are teacher-side signals being mistaken for runtime input? | Would misstate deployment assumptions | wording risk only | TODO |
| CF5 | Is checkpoint selection, rather than training itself, driving the apparent improvement? | Important because Run A vs Run B differs partly by best-by-structured selection | known and should be stated, not hidden | TODO |
| CF6 | Does pooled training hide subgroup collapse? | A pooled average can mask scene or subject failure pockets | open | TODO |
| CF7 | Does the reproducibility gap weaken any strong replacement claim? | Active-script full runs are not anchor-equivalent | known hard boundary | TODO |
| CF8 | Do local visual oscillations or same-location spikes undermine the main narrative? | Could indicate a failure mode not visible in headline metrics | open | TODO |

## First-Wave Fill Slots
- TODO-A: insert the exact approved wording for task A and link the artifact used to support it.
- TODO-B: insert the exact approved wording for task B and link the artifact used to support it.
- TODO-D: insert any subgroup limitation that task D makes unavoidable.
- TODO-E: insert the outcome that would justify promoting or stopping the next candidate branch.
- TODO-F: insert the approved confound or boundary finding that most directly affects claim wording.
- TODO-G: insert the approved generalization or subgroup finding that most directly affects pooled-training wording.

## First-Wave Findings Added On 2026-04-21
- A / feature-input audit:
  - The fixed baseline input set is now concretely backed by the active-script `input_qc` dump:
    - `roll`, `yawrate`, `ay`, `ax`, `speed_mps`, `z`, `lane_distance_m`, `lane_curvature`, `yaw`, `steer`, `LTR_est`, `steer_rate`, `lane_rate`, `lane_acc`, `lane_unwrap`, `lane_unwrap_rate`, `lane_unwrap_acc`
  - `zx1|lateraldistance` is the real dominant lane column in `91/91` vehicle files, so lane distance is not optional wording-wise; it is part of the repaired maintained input pipeline.
- B / trigger-response lag:
  - Trigger-to-onset lag median is `0.105 s`, mean is `0.176 s`.
  - Straight mean lag is `0.168 s`; curve mean lag is `0.195 s`.
  - Protocol split is still disabled and must be stated explicitly as "not enabled" rather than implied.
- F / checkpoint-selection confound:
  - The constrained-Pareto selector does not overturn the current structured keepers.
  - This means checkpoint selection remains part of the story, but the current Run A / Run B keeper split is not a fragile artifact of one overly permissive structured pick.
- G / local-spike failure mode:
  - B/C/D show cross-channel synchronized spike bands at fixed future positions, which supports a shared decoder / timestep-position artifact interpretation.
  - This failure mode must be acknowledged as a real qualitative risk even when headline metrics are acceptable.
- D / input-ablation closure:
  - The repaired `baseline_fixed_input` group is the maintained-line fit / tail keeper after the full D matrix closed.
  - Pedal inputs are a late-peak trade-off branch: `plus_pedals` improves `late_peak_recall` but worsens `rmse_steer` and `tail_rmse_steer` versus `baseline_fixed_input`.
  - `minus_z` improves first-reversal timing but does not beat `baseline_fixed_input` on overall fit or tail fit, so `USE_Z=1` remains the default.
- E / bridge closure:
  - `bridge_50_50` gives the strongest fit/tail numbers inside E, but it is not promoted because the strong-pos tail guardrails collapse too sharply.
  - `bridge_schedule_B_to_A` recovers some late-peak and first-reversal timing versus `bridge_50_50`, but it gives back too much fit/tail and is not promoted either.
  - The safest wording is therefore still a dual-keeper story rather than a unified single-checkpoint claim.

## Current One-Paragraph Defense Summary
At the current repo state, the safest summary is that the project models pooled post-trigger steering response rather than scene-specific trajectory planning. The model consumes pre-trigger vehicle history, anchor-state context, a style prior, and future road-geometry preview, then predicts short-horizon steering behavior while being judged on both fit and response-structure metrics. The main scientific value is therefore the boundary-correct formulation and the response-fidelity evaluation protocol, with Run A and `baseline_fixed_input` representing the two live keepers under different objectives and the stable manual-upsample line remaining the fixed baseline for any claim.
