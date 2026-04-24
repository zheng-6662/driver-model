# Thesis Defense Tables

## Status
- Task H scaffold drafted on 2026-04-21.
- These tables are ready for slide export or chapter drafting.
- Replace every `TODO` cell only after first-wave diagnostics are signed off.

## Table T1. Problem Statement And Contribution Boundary
| Item | Draft defense wording | Keep / change rule |
| --- | --- | --- |
| Problem statement | Pooled post-trigger steering response prediction | Keep unless first-wave diagnostics show a fatal boundary problem |
| Sample definition | 3.0 s history plus 2.0 s future after the maintained event anchor | Keep |
| Trigger definition | Curve-like events use roll peak; straight-like events use earliest steer-rate point reaching 80% of the local peak | Keep and state explicitly |
| Training scope | One pooled model across subjects and road contexts; not scene-specific training | May narrow later, never broaden |
| Runtime inputs | Vehicle history, anchor context, style prior, future road preview | If any later run removes one of these, say so explicitly |
| Contribution boundary | Response prediction plus evaluation and checkpoint-selection protocol; not scene-specific planning and not causal cognition modeling | Hard guardrail |
| Fixed comparison point | Stable manual-upsample control remains the baseline for any new claim | Hard guardrail |

## Table T2. Compact Input-Role Taxonomy For Defense Slides
| Role | Short slide wording | Allowed claim | Forbidden overclaim | Evidence source |
| --- | --- | --- | --- | --- |
| History window | Pre-trigger vehicle dynamics | The model uses 3.0 s of vehicle history before the trigger | "The model predicts from raw scene semantics alone" | `INPUT_ROLE_AUDIT.md` |
| Anchor context | Trigger-state summary | The model conditions on the state at the trigger | "The trigger is a future label leak" | `INPUT_ROLE_AUDIT.md` |
| Style prior | Pooled driver-style prior | Style information is a conditioning prior in pooled training | "The model is individualized per subject" | `INPUT_ROLE_AUDIT.md` |
| Road preview | Future path geometry preview | The task is conditional on known local road geometry | "The model is a blind free-roll forecast" | `INPUT_ROLE_AUDIT.md` |
| Teacher-side signal | Privileged training-only supervision | Physio and EEG can shape training but are not required at runtime | "Deployment requires physio or EEG" | `INPUT_ROLE_AUDIT.md` |
| Selection layer | Best-by-loss vs best-by-structured | Checkpoint choice is part of the method and must be reported | "Training alone explains every gain" | current-state plus 2026-04-20 log |

## Table T3. Confound And Alternative-Explanation Placeholder
| Confound candidate | Failure pattern if true | Current known status | Evidence artifact to cite | First-wave finding | Decision impact |
| --- | --- | --- | --- | --- | --- |
| Future curvature preview boundary | Improvement disappears once task is restated as conditional-on-preview | open wording risk | TODO | TODO | TODO |
| Style prior / subject shortcut | Gains come from subject identity proxy rather than response modeling | open | TODO | TODO | TODO |
| Anchor-policy split | Straight and curve anchors behave like hidden separate tasks | known risk | TODO | TODO | TODO |
| Teacher-side privileged information confusion | Defense overstates runtime inputs | wording risk only | TODO | TODO | TODO |
| Checkpoint-selection confound | Best-by-structured, not training itself, explains the visible keeper | known | TODO | TODO | TODO |
| Pooled-training subgroup collapse | Pooled average hides failure on one subgroup | open | TODO | TODO | TODO |
| Reproducibility boundary | Active-script line cannot claim anchor replacement | known hard boundary | current-state / 2026-04-20 | TODO | TODO |
| Local oscillation / spike failure mode | Headline metrics hide qualitative instability | open | visual issue case pack | TODO | TODO |

## Table T4. Current Keeper Comparison
This table is prefilled from the current repo anchors so the defense draft starts from the actual state rather than blank placeholders.

| Candidate | Status now | Selection rule | `rmse_steer` | `tail_rmse_steer` | `late_peak_recall` | `first_reversal_time_mae_sec` | `reversal_count_exact_match_rate` | Defense use |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Stable manual-upsample control | fixed baseline | historical stable control | 0.6037 | 0.7334 | 0.4274 | 0.8066 | 0.3447 | comparison anchor only |
| Run A | live keeper | best_by_structured | 0.6152 | 0.7412 | 0.6496 | 0.4163 | 0.6686 | strongest reversal and timing story |
| `baseline_fixed_input` | live keeper | best_by_structured | 0.5559 | 0.7171 | 0.6496 | 0.5107 | 0.5152 | current fit/tail keeper after full E review |
| `bridge_50_50` | mixed bridge evidence | best_by_structured | 0.5385 | 0.6846 | 0.6197 | 0.5642 | 0.4659 | fit/tail frontier only; blocked by strong-pos tail collapse |
| `bridge_schedule_B_to_A` | mixed bridge evidence | best_by_structured | 0.5819 | 0.7749 | 0.6581 | 0.4923 | 0.4848 | schedule bridge evidence only, not promoted |
| `plus_pedals` | trade-off branch | best_by_structured | 0.5663 | 0.7445 | 0.8504 | 0.4550 | 0.5303 | late-peak evidence only, not a clean replacement |
| Run B | pre-fix comparator | best_by_structured | 0.5907 | 0.7205 | 0.5470 | 0.4828 | 0.5341 | historical fit/tail comparator before input repair |

Note:
- Run C is already mapped as timing-only diagnostic support and is intentionally omitted from the main defense table.
- `plus_lat_dyn`, `plus_road_cond`, `minus_z`, and `bridge_55_45` are intentionally omitted from the main defense table because they are not promoted keepers.
- Do not replace the stable baseline row with active-script full-run checks unless the reproducibility boundary is actually closed.

## Table T5. Stratified Result Placeholder
| Slice | Stable baseline | Run A | Run B | Run D | Run E | Run F | Run G | Reading | TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All test samples | prefilled in T4 | prefilled in T4 | prefilled in T4 | prefilled in T4 | TODO | TODO | TODO | overall comparison | TODO |
| Straight-like events | TODO | TODO | TODO | TODO | TODO | TODO | TODO | pooled training must not hide straight-only failure | TODO |
| Curve-like events | TODO | TODO | TODO | TODO | TODO | TODO | TODO | pooled training must not hide curve-only failure | TODO |
| Strong reversal bucket | TODO | TODO | TODO | TODO | TODO | TODO | TODO | critical for response-structure story | TODO |
| Late-peak cases | TODO | TODO | TODO | TODO | TODO | TODO | TODO | critical for post-trigger timing story | TODO |
| Visual issue slice | TODO | TODO | TODO | TODO | TODO | TODO | TODO | same-location spikes or local oscillation check | TODO |

## Table T6. Qualitative Case Panel Placeholder
| Panel ID | Why selected | Baseline artifact | Candidate artifact | Expected talking point | Actual first-wave finding |
| --- | --- | --- | --- | --- | --- |
| Q1 | Run A shows stronger reversal structure than baseline | TODO | TODO | response timing and reversal count visibly recover | TODO |
| Q2 | Run B shows stronger overall fit than baseline | TODO | TODO | overall and tail shape improve while keeping response fidelity acceptable | TODO |
| Q3 | Visual issue case | TODO | TODO | local oscillation or same-location spike must be acknowledged | TODO |
| Q4 | Straight-like hard case | TODO | TODO | pooled model should still behave coherently outside curve-heavy cases | TODO |
| Q5 | Curve-like hard case | TODO | TODO | pooled model should not rely on hidden scene specialization | TODO |

## Table T7. Claim-Limitation Pairing
| If the evidence supports this claim | Pair it with this limitation sentence | Required supporting artifact |
| --- | --- | --- |
| Run A is the main keeper for response fidelity | Overall / tail fit is not the strongest among live candidates, so the contribution is structure-preserving prediction rather than global-error minimization | Run A vs baseline comparison plus subgroup audit |
| `baseline_fixed_input` is the main keeper for fit | Even stronger fit/tail frontier candidates can fail critical guardrails, so the contribution is balanced pooled fit improvement rather than raw-RMSE minimization | `baseline_fixed_input` vs baseline comparison plus full guardrail table |
| The pooled formulation is still the main story | This is pooled training over mixed contexts, not scene-specific optimization | input-role audit plus subgroup table |
| Road preview is kept in the task definition | The task is conditional on known local road geometry, not a blind free-future forecast | input-role audit plus confound table |
| Teacher-side privileged supervision is mentioned | Teacher-side signals are training-time only and are not required at deployment | input-role audit |

## First-Wave Result Inserts 2026-04-21

| Slot | Result to paste into defense material | Artifact |
| --- | --- | --- |
| A / input role | `zx1|lateraldistance` is present in `91/91` vehicle files, while the old training-exact lane aliases match `0/91`; the repaired fixed pipeline therefore explicitly includes `lane_distance_m` and lane derivatives. | `../feature_input_audit_20260421/feature_presence_report.md` |
| B / task timing | Trigger-to-onset median lag is `0.105 s` across `3737` strong events; curve events have slightly larger mean lag (`0.195 s`) than straight events (`0.168 s`). | `../trigger_response_lag_20260421/trigger_to_onset_summary.json` |
| B / protocol split | Protocol split is not enabled in this diagnostic because no unambiguous `Event_Dataset_v2` join was used. | `../trigger_response_lag_20260421/TASK_DEFINITION_AND_EVENT_LOGIC.md` |
| F / checkpoint selection | The diagnostic constrained-Pareto selector keeps the same structured epochs for Run A and Run B/C/D; it does not overturn the current Run A / Run B keeper split. | `../checkpoint_selection_diagnosis_20260421/pareto_summary.json` |
| G / visual failure mode | B/C/D all show cross-channel synchronized spike bands, supporting a shared decoder / timestep-position artifact explanation and requiring a qualitative-risk limitation sentence. | `../spike_position_diagnosis_20260421/spike_position_summary.md` |
| D / input-group keeper | `baseline_fixed_input` closes D as the maintained-line fit / tail keeper; it beats old Run B on `rmse_steer`, `tail_rmse_steer`, and `late_peak_recall` while preserving `USE_Z=1` as default. | `../input_group_ablation_20260421/input_ablation_comparison_table.csv` |
| D / pedals trade-off | `plus_pedals` is the D late-peak trade-off branch, not a clean replacement, because fit and tail regress versus `baseline_fixed_input`. | `../input_group_ablation_20260421/input_ablation_comparison_table.csv` |
| D / `minus_z` control | `minus_z` improves first-reversal timing but does not beat `baseline_fixed_input` on fit/tail, so `USE_Z=1` stays default. | `../input_group_ablation_20260421/input_ablation_comparison_table.csv` |
| E / bridge fit-tail frontier | `bridge_50_50` beats `baseline_fixed_input` on `rmse_steer` and `tail_rmse_steer`, but `strong_pos.tail_amp_ratio_pred_over_gt=0.4987` and `strong_pos.tail_flatness_rate=0.7368` block promotion. | `../bridge_training_20260421/bridge_comparison_table.csv` |
| E / schedule bridge trade-off | `bridge_schedule_B_to_A` lifts `late_peak_recall` and first-reversal timing versus `bridge_50_50`, and it repairs the strong-pos tail guardrails relative to `bridge_50_50`, but it gives back too much fit/tail to promote. | `../bridge_training_20260421/bridge_comparison_table.csv` |

## Updated Limitation Sentence Candidates

- Input repair limitation:
  - "All post-2026-04-21 comparisons use the repaired `fixed_v20260421` input pipeline; old active-script runs remain useful as historical anchors but should not be treated as input-equivalent."
- Trigger timing limitation:
  - "The trigger marker and response onset are close but not identical; the empirical median trigger-to-onset lag is about `0.105 s`, so the task is best described as post-trigger response prediction rather than instantaneous reaction modeling."
- Spike limitation:
  - "Some candidate runs show synchronized local spikes at fixed future positions; this is treated as a qualitative decoder/timestep artifact risk and is not hidden by aggregate metrics."
- Bridge limitation:
  - "Changing the hybrid reversal weighting and bridge schedule moves the fit/tail versus structure trade-off frontier, but the current E matrix still does not yield a single checkpoint that simultaneously replaces both Run A and the fit/tail keeper."
