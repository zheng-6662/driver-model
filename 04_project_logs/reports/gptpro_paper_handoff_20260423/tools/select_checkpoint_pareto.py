#!/usr/bin/env python
"""
Diagnostic checkpoint selector for the 2026-04-21 A/B/C/D run set.

This tool does not change or endorse a checkpoint.  It reads existing
loss_history.csv and val_structured_history.csv artifacts, applies a
conservative overall/tail gate first, then chooses the strongest structured
candidate among the gated Pareto frontier for diagnosis only.
"""
from __future__ import print_function

import argparse
import csv
import datetime as _dt
import json
import math
import os
from pathlib import Path


DEFAULT_OUTPUT = Path("04_project_logs/reports/checkpoint_selection_diagnosis_20260421")

RUN_SPECS = [
    {
        "run_id": "stable_manualup_control",
        "role": "fixed_control",
        "path": Path("03_results/tmp/gptpro_handoff_20260421_model_progress/artifacts/stable_baseline_manualup"),
        "fallback_paths": [
            Path("03_results/tmp/v220918_strict_repro_manualup_full_d/TRAIN_V5_4_STATECOND_REV_20260420_121314"),
            Path("03_results/tmp/v220918_strict_repro_manualup_full_a/TRAIN_V5_4_STATECOND_REV_20260419_223524"),
        ],
        "note": "Canonical old stable manual-upsample control. Structured val history was not logged for this older control.",
    },
    {
        "run_id": "runA_structured_full",
        "role": "run_A_full",
        "path": Path("03_results/tmp/runA_structured_full/TRAIN_V5_4_STATECOND_REV_20260420_110255"),
        "note": "Run A history was produced before the train-unit/plot-unit decoupling fix; compare its values within-run only.",
    },
    {
        "run_id": "runB_hybrid_full",
        "role": "run_B_full",
        "path": Path("03_results/tmp/runB_hybrid_full/TRAIN_V5_4_STATECOND_REV_20260420_174856"),
        "note": "Run B full hybrid reversal-weighting run.",
    },
    {
        "run_id": "runC_hybrid_localrev_full",
        "role": "run_C_full",
        "path": Path("03_results/tmp/runC_hybrid_localrev_full/TRAIN_V5_4_STATECOND_REV_20260420_181731"),
        "note": "Run C full hybrid + local first-reversal timing run.",
    },
    {
        "run_id": "runD_hybrid_localrev_late025_full",
        "role": "run_D_full",
        "path": Path("03_results/tmp/runD_hybrid_localrev_late025_full/TRAIN_V5_4_STATECOND_REV_20260420_183649"),
        "note": "Run D full Run C + late strong-reversal downweight adjustment run.",
    },
]


STRUCTURED_COLUMNS = [
    "rmse_steer",
    "tail_rmse_steer",
    "late_peak_recall",
    "first_reversal_time_mae_sec",
    "reversal_count_exact_match_rate",
    "n_eval",
    "structured_score",
]

TABLE_COLUMNS = [
    "run_id",
    "role",
    "run_dir",
    "epoch",
    "train_loss",
    "val_loss",
    "rmse_steer",
    "tail_rmse_steer",
    "late_peak_recall",
    "first_reversal_time_mae_sec",
    "reversal_count_exact_match_rate",
    "structured_score",
    "loss_rank",
    "tail_rank",
    "structured_rank",
    "rank_gate_cut",
    "passes_overall_tail_gate",
    "pareto_frontier_after_gate",
    "is_best_by_loss",
    "is_best_by_structured",
    "is_best_by_constrained_pareto",
    "filter_reason",
    "source_loss_history",
    "source_val_structured_history",
]


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_int(value):
    number = safe_float(value)
    if number is None:
        return None
    return int(round(number))


def read_csv_dicts(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def resolve_run_dir(repo_root, spec):
    candidates = [spec["path"]]
    candidates.extend(spec.get("fallback_paths", []))
    for rel_path in candidates:
        candidate = repo_root / rel_path
        if candidate.exists():
            return candidate.resolve(), str(rel_path).replace("\\", "/")
    return (repo_root / spec["path"]).resolve(), str(spec["path"]).replace("\\", "/")


def rank_map(rows, metric_name, reverse=False):
    valid = []
    for row in rows:
        value = row.get(metric_name)
        if value is not None:
            valid.append((row["epoch"], value))
    valid.sort(key=lambda item: item[1], reverse=reverse)
    ranks = {}
    for idx, item in enumerate(valid):
        ranks[item[0]] = idx + 1
    return ranks


def dominates(a, b, objective_names):
    """Return True when row a Pareto-dominates row b for lower-is-better objectives."""
    any_strict = False
    for name in objective_names:
        av = a.get(name)
        bv = b.get(name)
        if av is None or bv is None:
            return False
        if av > bv:
            return False
        if av < bv:
            any_strict = True
    return any_strict


def extract_summary_metrics(path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None

    def nested(*keys):
        cur = data
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    return {
        "summary_path": str(path),
        "rmse_steer": nested("rmse_steer"),
        "tail_rmse_steer": nested("tail_metrics", "tail_rmse_steer"),
        "late_peak_recall": nested("peak_metrics", "late_peak_recall"),
        "first_reversal_time_mae_sec": nested("reversal_structure_metrics", "first_reversal_time_mae_sec"),
        "reversal_count_exact_match_rate": nested("reversal_structure_metrics", "reversal_count_exact_match_rate"),
        "head_amp_ratio_pred_over_gt": nested("head_metrics", "head_amp_ratio_pred_over_gt"),
        "response_onset_delay_mae_sec": nested("head_metrics", "response_onset_delay_mae_sec"),
        "smooth_trend_corr_mean": nested("trend_metrics", "smooth_trend_corr_mean"),
        "coarse_segment_sign_match_rate": nested("trend_metrics", "coarse_segment_sign_match_rate"),
        "strong_pos_tail_amp_ratio_pred_over_gt": nested(
            "reversal_structure_metrics", "by_bucket", "strong_pos", "tail_amp_ratio_pred_over_gt"
        ),
        "strong_pos_tail_flatness_rate": nested(
            "reversal_structure_metrics", "by_bucket", "strong_pos", "tail_flatness_rate"
        ),
    }


def find_recalc_summaries(run_dir, run_id):
    figures = run_dir / "figures"
    out = {}
    if not figures.exists():
        return out
    for path in sorted(figures.glob("*_summary.json")):
        name = path.name
        if "best_by_loss" in name:
            out["best_by_loss"] = extract_summary_metrics(path)
        elif "best_by_structured" in name:
            out["best_by_structured"] = extract_summary_metrics(path)
        elif "recalc_repro_audit_current_metrics" in name or "current_metrics" in name:
            out["current_metrics"] = extract_summary_metrics(path)
    # Keep the function tolerant; the caller still records missing summaries.
    return dict((k, v) for k, v in out.items() if v is not None)


def analyze_run(repo_root, spec, gate_fraction):
    run_dir, resolved_rel = resolve_run_dir(repo_root, spec)
    loss_path = run_dir / "loss_history.csv"
    structured_path = run_dir / "val_structured_history.csv"

    summary = {
        "run_id": spec["run_id"],
        "role": spec["role"],
        "run_dir": str(run_dir),
        "resolved_relative_path": resolved_rel,
        "note": spec.get("note", ""),
        "has_loss_history": loss_path.exists(),
        "has_val_structured_history": structured_path.exists(),
        "n_loss_epochs": 0,
        "n_structured_epochs": 0,
        "best_by_loss_epoch": None,
        "best_by_structured_epoch": None,
        "best_by_constrained_pareto_epoch": None,
        "best_by_constrained_pareto_is_diagnostic_only": True,
        "rank_gate_fraction": gate_fraction,
        "rank_gate_cut": None,
        "n_pass_overall_tail_gate": 0,
        "pareto_frontier_epochs_after_gate": [],
        "selection_changed_from_best_by_structured": None,
        "fallbacks": [],
        "recalc_summaries": find_recalc_summaries(run_dir, spec["run_id"]),
    }

    if not loss_path.exists():
        summary["fallbacks"].append("missing_loss_history")
        return [], summary

    loss_rows_raw = read_csv_dicts(loss_path)
    summary["n_loss_epochs"] = len(loss_rows_raw)
    rows_by_epoch = {}
    for raw in loss_rows_raw:
        epoch = safe_int(raw.get("epoch"))
        if epoch is None:
            continue
        row = {
            "run_id": spec["run_id"],
            "role": spec["role"],
            "run_dir": str(run_dir),
            "epoch": epoch,
            "train_loss": safe_float(raw.get("train_loss")),
            "val_loss": safe_float(raw.get("val_loss")),
            "source_loss_history": str(loss_path),
            "source_val_structured_history": "" if not structured_path.exists() else str(structured_path),
        }
        for name in STRUCTURED_COLUMNS:
            prefixed = "val_structured_" + name
            if prefixed in raw:
                row[name] = safe_float(raw.get(prefixed))
            else:
                row[name] = None
        rows_by_epoch[epoch] = row

    if rows_by_epoch:
        loss_rank = rank_map(list(rows_by_epoch.values()), "val_loss", reverse=False)
        for row in rows_by_epoch.values():
            row["loss_rank"] = loss_rank.get(row["epoch"])
        best_loss_epoch = min(
            [row for row in rows_by_epoch.values() if row.get("val_loss") is not None],
            key=lambda row: (row["val_loss"], row["epoch"]),
        )["epoch"]
        summary["best_by_loss_epoch"] = best_loss_epoch

    structured_rows_raw = []
    if structured_path.exists():
        structured_rows_raw = read_csv_dicts(structured_path)
        summary["n_structured_epochs"] = len(structured_rows_raw)
        for raw in structured_rows_raw:
            epoch = safe_int(raw.get("epoch"))
            if epoch is None:
                continue
            if epoch not in rows_by_epoch:
                rows_by_epoch[epoch] = {
                    "run_id": spec["run_id"],
                    "role": spec["role"],
                    "run_dir": str(run_dir),
                    "epoch": epoch,
                    "train_loss": None,
                    "val_loss": None,
                    "source_loss_history": str(loss_path),
                    "source_val_structured_history": str(structured_path),
                }
            for name in STRUCTURED_COLUMNS:
                rows_by_epoch[epoch][name] = safe_float(raw.get(name))
    else:
        summary["fallbacks"].append("missing_val_structured_history; constrained Pareto is not applicable")

    rows = [rows_by_epoch[key] for key in sorted(rows_by_epoch)]
    for row in rows:
        row.setdefault("rmse_steer", None)
        row.setdefault("tail_rmse_steer", None)
        row.setdefault("late_peak_recall", None)
        row.setdefault("first_reversal_time_mae_sec", None)
        row.setdefault("reversal_count_exact_match_rate", None)
        row.setdefault("structured_score", None)
        row.setdefault("loss_rank", None)
        row["tail_rank"] = None
        row["structured_rank"] = None
        row["rank_gate_cut"] = None
        row["passes_overall_tail_gate"] = False
        row["pareto_frontier_after_gate"] = False
        row["is_best_by_loss"] = row["epoch"] == summary["best_by_loss_epoch"]
        row["is_best_by_structured"] = False
        row["is_best_by_constrained_pareto"] = False
        row["filter_reason"] = "missing_val_structured_history" if not structured_path.exists() else ""

    if not structured_path.exists():
        return rows, summary

    structured_rows = [row for row in rows if row.get("structured_score") is not None]
    if not structured_rows:
        summary["fallbacks"].append("empty_val_structured_history")
        return rows, summary

    tail_ranks = rank_map(structured_rows, "tail_rmse_steer", reverse=False)
    structured_ranks = rank_map(structured_rows, "structured_score", reverse=False)
    best_struct_epoch = min(
        structured_rows,
        key=lambda row: (row["structured_score"], row.get("tail_rmse_steer") or 1e100, row["epoch"]),
    )["epoch"]
    summary["best_by_structured_epoch"] = best_struct_epoch

    gate_cut = max(3, int(math.ceil(len(structured_rows) * gate_fraction)))
    summary["rank_gate_cut"] = gate_cut

    gated = []
    for row in rows:
        row["tail_rank"] = tail_ranks.get(row["epoch"])
        row["structured_rank"] = structured_ranks.get(row["epoch"])
        row["rank_gate_cut"] = gate_cut
        row["is_best_by_structured"] = row["epoch"] == best_struct_epoch
        if row.get("structured_score") is None:
            row["filter_reason"] = "missing_structured_metrics_for_epoch"
            continue
        if row.get("loss_rank") is None:
            row["filter_reason"] = "missing_val_loss"
            continue
        if row.get("tail_rank") is None:
            row["filter_reason"] = "missing_tail_rmse_steer"
            continue
        if int(row["loss_rank"]) > gate_cut:
            row["filter_reason"] = "reject_overall_loss_rank_gt_cut"
            continue
        if int(row["tail_rank"]) > gate_cut:
            row["filter_reason"] = "reject_tail_rmse_rank_gt_cut"
            continue
        row["passes_overall_tail_gate"] = True
        row["filter_reason"] = "kept_after_overall_tail_gate"
        gated.append(row)

    summary["n_pass_overall_tail_gate"] = len(gated)
    frontier = []
    objectives = ["val_loss", "tail_rmse_steer", "structured_score"]
    for row in gated:
        if not any(dominates(other, row, objectives) for other in gated if other is not row):
            row["pareto_frontier_after_gate"] = True
            row["filter_reason"] = "kept_on_constrained_pareto_frontier"
            frontier.append(row)

    if frontier:
        frontier.sort(
            key=lambda row: (
                row.get("structured_score") if row.get("structured_score") is not None else 1e100,
                row.get("tail_rmse_steer") if row.get("tail_rmse_steer") is not None else 1e100,
                row.get("val_loss") if row.get("val_loss") is not None else 1e100,
                row["epoch"],
            )
        )
        selected = frontier[0]
        selected["is_best_by_constrained_pareto"] = True
        summary["best_by_constrained_pareto_epoch"] = selected["epoch"]
        summary["pareto_frontier_epochs_after_gate"] = [row["epoch"] for row in frontier]
        summary["selection_changed_from_best_by_structured"] = selected["epoch"] != best_struct_epoch
    else:
        summary["fallbacks"].append("no_epochs_survived_constrained_pareto_filter")
        summary["selection_changed_from_best_by_structured"] = None

    return rows, summary


def normalize_for_csv(row):
    out = {}
    for key in TABLE_COLUMNS:
        value = row.get(key)
        if isinstance(value, bool):
            out[key] = "1" if value else "0"
        elif value is None:
            out[key] = ""
        else:
            out[key] = value
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Output report directory.")
    parser.add_argument(
        "--gate-rank-fraction",
        type=float,
        default=0.45,
        help="Within-run rank fraction kept for both val_loss and tail_rmse_steer before Pareto selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gate_rank_fraction <= 0 or args.gate_rank_fraction > 1:
        raise ValueError("--gate-rank-fraction must be in (0, 1].")

    all_rows = []
    summaries = []
    for spec in RUN_SPECS:
        rows, summary = analyze_run(repo_root, spec, args.gate_rank_fraction)
        all_rows.extend(rows)
        summaries.append(summary)

    table_path = output_dir / "pareto_epoch_table.csv"
    json_path = output_dir / "pareto_summary.json"
    write_csv(table_path, [normalize_for_csv(row) for row in all_rows], TABLE_COLUMNS)

    summary = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "pareto_epoch_table": str(table_path),
        "filter_rule": {
            "diagnostic_only": True,
            "lower_is_better_objectives": ["val_loss", "tail_rmse_steer", "structured_score"],
            "gate_rank_fraction": args.gate_rank_fraction,
            "gate_description": (
                "For each run, rank epochs by val_loss and by tail_rmse_steer. "
                "Reject epochs outside the top ceil(N * gate_rank_fraction) on either metric. "
                "On the surviving epochs, compute a Pareto frontier over val_loss, tail_rmse_steer, "
                "and structured_score, then choose the lowest structured_score on that frontier."
            ),
            "tie_break_order": ["structured_score", "tail_rmse_steer", "val_loss", "epoch"],
        },
        "run_summaries": summaries,
        "global_notes": [
            "The constrained-Pareto epoch is a diagnostic selector only; it does not replace saved checkpoints.",
            "The old stable manual-upsample control lacks val_structured_history.csv, so only best-by-loss can be reported for it.",
            "Run A history uses the pre-fix degree-scaled training/evaluation values; this report gates and ranks it only within that run.",
        ],
    }
    write_json(json_path, summary)

    print(json.dumps({"pareto_epoch_table": str(table_path), "pareto_summary": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
