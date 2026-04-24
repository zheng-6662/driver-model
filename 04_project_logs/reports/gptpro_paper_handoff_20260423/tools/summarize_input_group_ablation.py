from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "04_project_logs" / "reports" / "input_group_ablation_20260421"

KEY_METRICS = [
    "rmse_steer",
    "tail_rmse_steer",
    "late_peak_recall",
    "first_reversal_time_mae_sec",
    "reversal_count_exact_match_rate",
    "head_amp_ratio_pred_over_gt",
    "strong_pos_tail_amp_ratio_pred_over_gt",
    "strong_pos_tail_flatness_rate",
]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def resolve_summary(run_dir: Path) -> tuple[str | None, dict[str, Any] | None]:
    figures_dir = run_dir / "figures"
    structured_path = figures_dir / "recalc_best_by_structured_summary.json"
    loss_path = figures_dir / "recalc_best_by_loss_summary.json"
    if structured_path.exists():
        return "best_by_structured", load_json(structured_path)
    if loss_path.exists():
        return "best_by_loss", load_json(loss_path)
    matches = sorted(figures_dir.glob("recalc*_summary.json"))
    for match in matches:
        payload = load_json(match)
        if payload is not None:
            return match.stem, payload
    return None, None


def flatten_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    flat = {
        "rmse_steer": payload.get("rmse_steer"),
        "tail_rmse_steer": payload.get("tail_metrics", {}).get("tail_rmse_steer"),
        "late_peak_recall": payload.get("peak_metrics", {}).get("late_peak_recall"),
        "first_reversal_time_mae_sec": payload.get("reversal_structure_metrics", {}).get("first_reversal_time_mae_sec"),
        "reversal_count_exact_match_rate": payload.get("reversal_structure_metrics", {}).get("reversal_count_exact_match_rate"),
        "head_amp_ratio_pred_over_gt": payload.get("head_metrics", {}).get("head_amp_ratio_pred_over_gt"),
        "strong_pos_tail_amp_ratio_pred_over_gt": payload.get("reversal_structure_metrics", {}).get("by_bucket", {}).get("strong_pos", {}).get("tail_amp_ratio_pred_over_gt"),
        "strong_pos_tail_flatness_rate": payload.get("reversal_structure_metrics", {}).get("by_bucket", {}).get("strong_pos", {}).get("tail_flatness_rate"),
    }
    return flat


def build_commentary(df: pd.DataFrame, matrix: str) -> list[str]:
    lines: list[str] = []
    if df.empty:
        return ["No completed runs with recalc summaries were found."]
    if "baseline_fixed_input" in df["group_name"].values and matrix == "input_ablation":
        base_row = df[df["group_name"] == "baseline_fixed_input"].iloc[0]
        for _, row in df.iterrows():
            if row["group_name"] == "baseline_fixed_input":
                continue
            delta_rmse = (
                None
                if pd.isna(row["rmse_steer"]) or pd.isna(base_row["rmse_steer"])
                else float(row["rmse_steer"] - base_row["rmse_steer"])
            )
            delta_tail = (
                None
                if pd.isna(row["tail_rmse_steer"]) or pd.isna(base_row["tail_rmse_steer"])
                else float(row["tail_rmse_steer"] - base_row["tail_rmse_steer"])
            )
            lines.append(
                f"- `{row['group_name']}` vs baseline: "
                f"delta_rmse_steer={delta_rmse if delta_rmse is not None else 'n/a'}, "
                f"delta_tail_rmse_steer={delta_tail if delta_tail is not None else 'n/a'}."
            )
    elif matrix == "bridge":
        for _, row in df.iterrows():
            lines.append(
                f"- `{row['group_name']}` kept `{row['selection_source']}` with "
                f"rmse_steer={row.get('rmse_steer')}, "
                f"tail_rmse_steer={row.get('tail_rmse_steer')}, "
                f"late_peak_recall={row.get('late_peak_recall')}, "
                f"first_reversal_time_mae_sec={row.get('first_reversal_time_mae_sec')}, "
                f"strong_pos_tail_amp_ratio_pred_over_gt={row.get('strong_pos_tail_amp_ratio_pred_over_gt')}, "
                f"strong_pos_tail_flatness_rate={row.get('strong_pos_tail_flatness_rate')}."
            )
    return lines


def dataframe_to_markdown_fallback(df: pd.DataFrame) -> str:
    if df.empty:
        return "No completed runs were available."
    try:
        return df.to_markdown(index=False)
    except Exception:
        csv_preview = df.to_csv(index=False)
        return "```csv\n" + csv_preview.rstrip() + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize input-ablation or bridge-manifest outputs into a compact table.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    report_dir = Path(args.report_dir).resolve()
    if args.manifest is not None:
        manifest_path = Path(args.manifest).resolve()
    else:
        manifests = sorted(report_dir.glob("*_manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"No manifest found under {report_dir}")
        manifest_path = manifests[-1]

    manifest = load_json(manifest_path)
    if manifest is None:
        raise RuntimeError(f"Failed to load manifest: {manifest_path}")

    rows: list[dict[str, Any]] = []
    for group in manifest.get("groups", []):
        run_dir_value = group.get("run_dir")
        if not run_dir_value:
            continue
        run_dir = Path(run_dir_value)
        selection_source, payload = resolve_summary(run_dir)
        row = {
            "group_name": group.get("group_name"),
            "status": group.get("status"),
            "run_dir": str(run_dir),
            "selection_source": selection_source,
        }
        row.update(flatten_metrics(payload))
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_csv_path = report_dir / f"{manifest['matrix']}_comparison_table.csv"
    df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    lines = [
        f"# {manifest['matrix']} Summary",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Table: `{summary_csv_path}`",
        "",
        "## Compact Table",
        "",
        dataframe_to_markdown_fallback(df),
        "",
        "## Commentary",
        *build_commentary(df, manifest["matrix"]),
        "",
        "## Metrics Used",
        *[f"- `{metric}`" for metric in KEY_METRICS],
    ]
    summary_md_path = report_dir / f"{manifest['matrix']}_summary.md"
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "summary_csv_path": str(summary_csv_path),
                "summary_md_path": str(summary_md_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
