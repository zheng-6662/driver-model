from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "04_project_logs" / "reports" / "effectiveness_followup_20260422"
EXPECTED_COLUMNS = [
    "action_name",
    "phase",
    "kind",
    "mode",
    "status",
    "base_reference",
    "selection_source",
    "run_dir",
    "checkpoint_path",
    "future_sec",
    "optimizer",
    "lr",
    "weight_decay",
    "scheduler",
    "warmup_epochs",
    "grad_clip_norm",
    "d_model",
    "n_head",
    "ffn_dim",
    "dropout",
    "rmse_steer",
    "tail_rmse_steer",
    "late_peak_recall",
    "first_reversal_time_mae_sec",
    "reversal_count_exact_match_rate",
    "strong_pos_tail_amp_ratio_pred_over_gt",
    "strong_pos_tail_flatness_rate",
    "prefix_1p0s_rmse_steer",
    "prefix_1p5s_rmse_steer",
    "full_horizon_rmse_steer",
    "abs_tail_last_0p5s_rmse_steer",
    "window_future_len",
    "tail_window_start_sec",
    "hard_collapse",
    "single_guardrail_alert",
]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def get_nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def resolve_preferred_summary(record: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    if record.get("kind") == "recalc_only":
        recalc = record.get("recalc", {})
        summary_path = Path(recalc.get("summary_path", ""))
        if summary_path.exists():
            return str(record.get("selection_source", "recalc_only")), load_json(summary_path)
        return None, None

    recalc = record.get("recalc", {})
    for key in ("best_by_structured", "best_by_loss"):
        payload = recalc.get(key, {})
        summary_path = Path(payload.get("summary_path", ""))
        if summary_path.exists():
            return key, load_json(summary_path)
    return None, None


def load_run_config(record: dict[str, Any]) -> dict[str, Any] | None:
    run_dir = record.get("run_dir")
    if not run_dir:
        return None
    return load_json(Path(run_dir) / "run_config.json")


def extract_env_value(record: dict[str, Any], key: str) -> Any:
    env = record.get("env_overrides", {})
    if not isinstance(env, dict):
        env = {}
    if key in env:
        return env.get(key)

    run_config = load_run_config(record)
    if not isinstance(run_config, dict):
        return None
    run_config_fallback = {
        "DRIVER_MODEL_FUTURE_SEC": "FUTURE_SEC",
        "DRIVER_MODEL_OPTIMIZER": "OPTIMIZER",
        "DRIVER_MODEL_LR": "LR",
        "DRIVER_MODEL_WEIGHT_DECAY": "WEIGHT_DECAY",
        "DRIVER_MODEL_SCHEDULER": "SCHEDULER",
        "DRIVER_MODEL_WARMUP_EPOCHS": "WARMUP_EPOCHS",
        "DRIVER_MODEL_GRAD_CLIP_NORM": "GRAD_CLIP_NORM",
        "DRIVER_MODEL_D_MODEL": "D_MODEL",
        "DRIVER_MODEL_N_HEAD": "N_HEAD",
        "DRIVER_MODEL_FFN_DIM": "FFN_DIM",
        "DRIVER_MODEL_DROPOUT": "DROPOUT",
    }
    config_key = run_config_fallback.get(key)
    if config_key is None:
        return None
    return run_config.get(config_key)


def flatten_summary(record: dict[str, Any], selection_source: str | None, summary: dict[str, Any] | None) -> dict[str, Any]:
    row = {
        "action_name": record.get("action_name"),
        "phase": record.get("phase"),
        "kind": record.get("kind"),
        "mode": record.get("mode"),
        "status": record.get("status"),
        "base_reference": record.get("base_reference"),
        "selection_source": selection_source,
        "run_dir": record.get("run_dir"),
        "checkpoint_path": record.get("checkpoint_path"),
        "future_sec": extract_env_value(record, "DRIVER_MODEL_FUTURE_SEC"),
        "optimizer": extract_env_value(record, "DRIVER_MODEL_OPTIMIZER"),
        "lr": extract_env_value(record, "DRIVER_MODEL_LR"),
        "weight_decay": extract_env_value(record, "DRIVER_MODEL_WEIGHT_DECAY"),
        "scheduler": extract_env_value(record, "DRIVER_MODEL_SCHEDULER"),
        "warmup_epochs": extract_env_value(record, "DRIVER_MODEL_WARMUP_EPOCHS"),
        "grad_clip_norm": extract_env_value(record, "DRIVER_MODEL_GRAD_CLIP_NORM"),
        "d_model": extract_env_value(record, "DRIVER_MODEL_D_MODEL"),
        "n_head": extract_env_value(record, "DRIVER_MODEL_N_HEAD"),
        "ffn_dim": extract_env_value(record, "DRIVER_MODEL_FFN_DIM"),
        "dropout": extract_env_value(record, "DRIVER_MODEL_DROPOUT"),
    }
    if summary is None:
        return row

    row.update(
        {
            "rmse_steer": summary.get("rmse_steer"),
            "tail_rmse_steer": get_nested(summary, "tail_metrics", "tail_rmse_steer"),
            "late_peak_recall": get_nested(summary, "peak_metrics", "late_peak_recall"),
            "first_reversal_time_mae_sec": get_nested(summary, "reversal_structure_metrics", "first_reversal_time_mae_sec"),
            "reversal_count_exact_match_rate": get_nested(summary, "reversal_structure_metrics", "reversal_count_exact_match_rate"),
            "strong_pos_tail_amp_ratio_pred_over_gt": get_nested(summary, "reversal_structure_metrics", "by_bucket", "strong_pos", "tail_amp_ratio_pred_over_gt"),
            "strong_pos_tail_flatness_rate": get_nested(summary, "reversal_structure_metrics", "by_bucket", "strong_pos", "tail_flatness_rate"),
            "prefix_1p0s_rmse_steer": get_nested(summary, "window_metrics", "prefix_1p0s", "rmse_steer"),
            "prefix_1p5s_rmse_steer": get_nested(summary, "window_metrics", "prefix_1p5s", "rmse_steer"),
            "full_horizon_rmse_steer": get_nested(summary, "window_metrics", "full_horizon", "rmse_steer"),
            "abs_tail_last_0p5s_rmse_steer": get_nested(summary, "window_metrics", "abs_tail_last_0p5s", "rmse_steer"),
            "window_future_len": summary.get("future_len"),
            "tail_window_start_sec": get_nested(summary, "tail_metrics", "tail_start_sec"),
        }
    )
    return row


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def mark_collapse(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    amp = pd.to_numeric(out.get("strong_pos_tail_amp_ratio_pred_over_gt"), errors="coerce")
    flat = pd.to_numeric(out.get("strong_pos_tail_flatness_rate"), errors="coerce")
    out["hard_collapse"] = (amp < 0.60) & (flat > 0.60)
    out["single_guardrail_alert"] = ((amp < 0.60) ^ (flat > 0.60))
    return out


def build_ranking(df: pd.DataFrame) -> pd.DataFrame:
    rankable = df[
        (df["kind"] == "train")
        & (df["mode"] == "full")
        & (~df["hard_collapse"].fillna(False))
        & df["abs_tail_last_0p5s_rmse_steer"].notna()
        & df["rmse_steer"].notna()
    ].copy()
    rankable = rankable.sort_values(
        by=["abs_tail_last_0p5s_rmse_steer", "rmse_steer", "late_peak_recall"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    if not rankable.empty:
        rankable.insert(0, "rank", range(1, len(rankable) + 1))
    return rankable


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```csv\n" + df.to_csv(index=False).rstrip() + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the 2026-04-22 effectiveness follow-up outputs.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    report_dir = Path(args.report_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else report_dir / "effectiveness_followup_manifest.json"
    manifest = load_json(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Failed to load manifest: {manifest_path}")

    rows: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        selection_source, summary = resolve_preferred_summary(record)
        rows.append(flatten_summary(record, selection_source, summary))

    df = pd.DataFrame(rows)
    df = ensure_columns(df, EXPECTED_COLUMNS)
    numeric_columns = [
        "future_sec",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "grad_clip_norm",
        "d_model",
        "n_head",
        "ffn_dim",
        "dropout",
        "rmse_steer",
        "tail_rmse_steer",
        "late_peak_recall",
        "first_reversal_time_mae_sec",
        "reversal_count_exact_match_rate",
        "strong_pos_tail_amp_ratio_pred_over_gt",
        "strong_pos_tail_flatness_rate",
        "prefix_1p0s_rmse_steer",
        "prefix_1p5s_rmse_steer",
        "full_horizon_rmse_steer",
        "abs_tail_last_0p5s_rmse_steer",
        "window_future_len",
        "tail_window_start_sec",
    ]
    df = to_numeric(df, numeric_columns)
    df = mark_collapse(df)
    df = ensure_columns(df, EXPECTED_COLUMNS)

    comparison_csv = report_dir / "effectiveness_comparison_table.csv"
    df.to_csv(comparison_csv, index=False, encoding="utf-8-sig")

    d0_df = df[df["phase"] == "D0"].copy()
    d0_csv = report_dir / "d0_comparison_table.csv"
    d0_df.to_csv(d0_csv, index=False, encoding="utf-8-sig")

    ranking_df = build_ranking(df)
    ranking_csv = report_dir / "effectiveness_ranking_table.csv"
    ranking_df.to_csv(ranking_csv, index=False, encoding="utf-8-sig")

    summary_lines = [
        "# Effectiveness Follow-up Summary",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Comparison table: `{comparison_csv}`",
        f"- D0 table: `{d0_csv}`",
        f"- Ranking table: `{ranking_csv}`",
        "",
        "## Fraction Tail Bias Note",
        "",
        "- Fraction-based `tail_rmse_steer` uses the last 25% of the horizon.",
        "- That means the native tail window shrinks from `100` steps at `2.0s` to `75` at `1.5s` and `50` at `1.0s`.",
        "- Cross-horizon comparisons should therefore use `abs_tail_last_0p5s.rmse_steer` as the primary tail metric.",
        "",
        "## D0 Anchors",
        "",
        dataframe_to_markdown(
            d0_df[
                [
                    "action_name",
                    "base_reference",
                    "selection_source",
                    "rmse_steer",
                    "tail_rmse_steer",
                    "prefix_1p0s_rmse_steer",
                    "prefix_1p5s_rmse_steer",
                    "full_horizon_rmse_steer",
                    "abs_tail_last_0p5s_rmse_steer",
                    "strong_pos_tail_amp_ratio_pred_over_gt",
                    "strong_pos_tail_flatness_rate",
                ]
            ]
            if not d0_df.empty
            else pd.DataFrame()
        ),
        "",
        "## Run Table",
        "",
        dataframe_to_markdown(
            df[
                [
                    "action_name",
                    "phase",
                    "mode",
                    "selection_source",
                    "future_sec",
                    "optimizer",
                    "lr",
                    "weight_decay",
                    "scheduler",
                    "rmse_steer",
                    "abs_tail_last_0p5s_rmse_steer",
                    "late_peak_recall",
                    "first_reversal_time_mae_sec",
                    "reversal_count_exact_match_rate",
                    "hard_collapse",
                    "single_guardrail_alert",
                ]
            ]
            if not df.empty
            else pd.DataFrame()
        ),
        "",
        "## Provisional Ranking",
        "",
        "- Ranking includes only full training actions recorded in this follow-up manifest.",
        "- Recalc-only anchors such as `D0_BASELINE` are excluded from the ranking table and must still be compared separately as fixed references.",
        "",
        dataframe_to_markdown(
            ranking_df[
                [
                    "rank",
                    "action_name",
                    "future_sec",
                    "optimizer",
                    "lr",
                    "abs_tail_last_0p5s_rmse_steer",
                    "rmse_steer",
                    "late_peak_recall",
                    "strong_pos_tail_amp_ratio_pred_over_gt",
                    "strong_pos_tail_flatness_rate",
                ]
            ]
            if not ranking_df.empty
            else pd.DataFrame()
        ),
    ]
    summary_md = report_dir / "effectiveness_summary.md"
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "comparison_csv": str(comparison_csv),
                "d0_csv": str(d0_csv),
                "ranking_csv": str(ranking_csv),
                "summary_md": str(summary_md),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
