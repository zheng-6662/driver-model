# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
DEFAULT_SUMMARY_DIR = REPORTS_DIR / "style_physio_eeg_e5_distill_summary_20260508"
DEFAULT_SEED_METRICS = DEFAULT_SUMMARY_DIR / "seed_wise_metrics.csv"
DEFAULT_CASE_FILE = THIS_DIR / "shared_prediction_cases_test.csv"
DEFAULT_OUT_DIR = DEFAULT_SUMMARY_DIR / "shape_audit"

PAIRS = [("E5A", "E2"), ("E5A", "E4")]
GROUP_COLUMNS = [
    "phase_type",
    "road_type_anchor",
    "eval_morphology_label",
    "interaction_slice",
    "reversal_slice",
    "structure_slice",
    "structure_heavy",
]
METRICS = [
    "rmse_2s_abs_steer",
    "rmse_pre_tail_abs_steer",
    "rmse_tail_abs_steer",
    "peak_time_abs_err_s",
    "turning_count_abs_err",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit E5A fixed cases and subgroup deltas.")
    parser.add_argument("--seed-metrics", default=str(DEFAULT_SEED_METRICS))
    parser.add_argument("--case-file", default=str(DEFAULT_CASE_FILE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top-regressions", type=int, default=30)
    return parser.parse_args()


def _load_seed_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"seed metrics not found: {path}")
    df = pd.read_csv(path)
    required = {"experiment_id", "seed", "run_root", "sample_metrics_csv"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"seed metrics missing columns: {sorted(missing)}")
    return df


def _load_sample_metrics(seed_metrics: pd.DataFrame, experiment_id: str, seed: int) -> pd.DataFrame:
    row = seed_metrics[
        seed_metrics["experiment_id"].astype(str).eq(experiment_id)
        & seed_metrics["seed"].astype(int).eq(int(seed))
    ]
    if row.empty:
        raise ValueError(f"missing metrics row for {experiment_id} seed={seed}")
    path = Path(str(row.iloc[0]["sample_metrics_csv"]))
    if not path.exists():
        raise FileNotFoundError(f"sample metrics not found: {path}")
    df = pd.read_csv(path)
    df["experiment_id"] = experiment_id
    df["seed"] = int(seed)
    return df


def _merge_pair(seed_metrics: pd.DataFrame, left_id: str, right_id: str, seed: int) -> pd.DataFrame:
    left = _load_sample_metrics(seed_metrics, left_id, seed)
    right = _load_sample_metrics(seed_metrics, right_id, seed)
    merged = left.merge(right, on="sample_key", suffixes=(f"_{left_id}", f"_{right_id}"))
    merged["pair"] = f"{left_id}:{right_id}"
    merged["seed"] = int(seed)
    for col in METRICS:
        left_col = f"{col}_{left_id}"
        right_col = f"{col}_{right_id}"
        if left_col in merged.columns and right_col in merged.columns:
            merged[f"delta_{col}"] = merged[left_col] - merged[right_col]
    return merged


def _fixed_case_rows(seed_metrics: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seeds = sorted(seed_metrics[seed_metrics["experiment_id"].astype(str).eq("E5A")]["seed"].astype(int).unique())
    case_keys = cases["sample_key"].astype(str).tolist()
    case_tag = dict(zip(cases["sample_key"].astype(str), cases["selection_tag"].astype(str)))
    for seed in seeds:
        e2 = _load_sample_metrics(seed_metrics, "E2", seed).set_index("sample_key")
        e4 = _load_sample_metrics(seed_metrics, "E4", seed).set_index("sample_key")
        e5 = _load_sample_metrics(seed_metrics, "E5A", seed).set_index("sample_key")
        for sample_key in case_keys:
            if sample_key not in e2.index or sample_key not in e4.index or sample_key not in e5.index:
                continue
            item: dict[str, Any] = {
                "seed": int(seed),
                "sample_key": sample_key,
                "selection_tag": case_tag.get(sample_key, ""),
            }
            for exp_id, df in [("E2", e2), ("E4", e4), ("E5A", e5)]:
                for col in METRICS:
                    item[f"{exp_id}_{col}"] = float(df.loc[sample_key, col])
            for right_id in ["E2", "E4"]:
                for col in METRICS:
                    item[f"delta_E5A_{right_id}_{col}"] = item[f"E5A_{col}"] - item[f"{right_id}_{col}"]
            rows.append(item)
    return pd.DataFrame(rows)


def _group_delta_rows(merged: pd.DataFrame, left_id: str, right_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_col in GROUP_COLUMNS:
        col_name = f"{group_col}_{left_id}"
        if col_name not in merged.columns:
            continue
        for value, group in merged.groupby(col_name, dropna=False):
            item: dict[str, Any] = {
                "pair": f"{left_id}:{right_id}",
                "seed": int(group["seed"].iloc[0]),
                "group_column": group_col,
                "group_value": "" if pd.isna(value) else str(value),
                "n": int(len(group)),
            }
            for metric in METRICS:
                delta_col = f"delta_{metric}"
                if delta_col in group.columns:
                    item[f"{metric}_delta_mean"] = float(group[delta_col].mean())
                    item[f"{metric}_improved_rate"] = float((group[delta_col] < 0).mean())
            rows.append(item)
    return rows


def _top_regressions(merged: pd.DataFrame, left_id: str, right_id: str, top_n: int) -> pd.DataFrame:
    cols = [
        "pair",
        "seed",
        "sample_key",
        f"phase_type_{left_id}",
        f"road_type_anchor_{left_id}",
        f"eval_morphology_label_{left_id}",
        f"interaction_slice_{left_id}",
        f"reversal_slice_{left_id}",
        "delta_rmse_2s_abs_steer",
        "delta_rmse_tail_abs_steer",
        "delta_peak_time_abs_err_s",
    ]
    existing = [col for col in cols if col in merged.columns]
    return merged.sort_values("delta_rmse_2s_abs_steer", ascending=False).head(top_n)[existing].copy()


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _pair_seed_summary(merged_all: pd.DataFrame, pair: str) -> pd.DataFrame:
    rows = []
    view = merged_all[merged_all["pair"].astype(str).eq(pair)]
    for seed, group in view.groupby("seed"):
        item: dict[str, Any] = {"pair": pair, "seed": int(seed), "n": int(len(group))}
        for metric in METRICS:
            delta_col = f"delta_{metric}"
            item[f"{metric}_delta_mean"] = float(group[delta_col].mean())
            item[f"{metric}_improved_rate"] = float((group[delta_col] < 0).mean())
        rows.append(item)
    return pd.DataFrame(rows)


def _write_report(
    out_dir: Path,
    pair_seed_summary: pd.DataFrame,
    fixed_cases: pd.DataFrame,
    group_summary: pd.DataFrame,
    regression_samples: pd.DataFrame,
) -> None:
    e5_e2 = pair_seed_summary[pair_seed_summary["pair"].astype(str).eq("E5A:E2")]
    e5_e4 = pair_seed_summary[pair_seed_summary["pair"].astype(str).eq("E5A:E4")]
    lines = [
        "# E5A 预测形状与分组审计",
        "",
        "## 审计目的",
        "",
        "- 检查 E5A 的提升是否只来自均值指标，还是在固定展示 case 和不同场景分组里也能站住。",
        "- 找出 E5A 相对 E2 的明显退化样本，避免汇报时只展示好看的平均值。",
        "- 这不是新训练，只读取已有预测结果和 sample-level metrics。",
        "",
        "## Seed 级配对概览",
        "",
        "| pair | seed | n | 2s误差差值 | 2s改善率 | 尾段差值 | 尾段改善率 | 峰值时间差值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in pd.concat([e5_e2, e5_e4]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pair"]),
                    str(int(row["seed"])),
                    str(int(row["n"])),
                    _fmt(row["rmse_2s_abs_steer_delta_mean"]),
                    _fmt(row["rmse_2s_abs_steer_improved_rate"]),
                    _fmt(row["rmse_tail_abs_steer_delta_mean"]),
                    _fmt(row["rmse_tail_abs_steer_improved_rate"]),
                    _fmt(row["peak_time_abs_err_s_delta_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 固定展示 case",
            "",
            f"- 固定 case 数：`{fixed_cases['sample_key'].nunique() if not fixed_cases.empty else 0}`。",
            "- 详细表：`fixed_case_deltas.csv`。",
            "",
            "## 分组差异",
            "",
            "- 详细表：`group_delta_summary.csv`。",
            "- 重点看 `phase_type`、`road_type_anchor`、`eval_morphology_label`、`reversal_slice` 等分组。",
            "",
            "## 明显退化样本",
            "",
            f"- 已输出 E5A 相对 E2 的前 `{len(regression_samples)}` 个 2 秒误差退化样本：`top_regression_samples_e5a_vs_e2.csv`。",
            "- 如果后续要做老师汇报，建议从这些退化样本里挑 2-3 个反例图一起看，避免结论过满。",
            "",
            "## 当前判断",
            "",
            "- E5A 的均值优势是真实存在的，但 seed-2028 的 sample-level 2 秒误差略有回退，需要在预测图中重点检查。",
            "- 如果固定 case 和退化样本没有明显形状灾难，可以把 E5A 暂定为主候选。",
            "- 如果退化样本集中在某类场景，再考虑补 E5B 或 E6，而不是盲目继续扩版本。",
        ]
    )
    (out_dir / "shape_audit_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = _load_seed_metrics(Path(args.seed_metrics))
    cases = pd.read_csv(args.case_file)

    merged_rows: list[pd.DataFrame] = []
    group_rows: list[dict[str, Any]] = []
    regression_rows: list[pd.DataFrame] = []
    seeds = sorted(seed_metrics[seed_metrics["experiment_id"].astype(str).eq("E5A")]["seed"].astype(int).unique())
    for seed in seeds:
        for left_id, right_id in PAIRS:
            merged = _merge_pair(seed_metrics, left_id, right_id, int(seed))
            merged_rows.append(merged)
            group_rows.extend(_group_delta_rows(merged, left_id, right_id))
            if (left_id, right_id) == ("E5A", "E2"):
                regression_rows.append(_top_regressions(merged, left_id, right_id, int(args.top_regressions)))

    merged_all = pd.concat(merged_rows, ignore_index=True)
    fixed_cases = _fixed_case_rows(seed_metrics, cases)
    group_summary = pd.DataFrame(group_rows)
    regression_samples = pd.concat(regression_rows, ignore_index=True)
    pair_summary = pd.concat(
        [_pair_seed_summary(merged_all, "E5A:E2"), _pair_seed_summary(merged_all, "E5A:E4")],
        ignore_index=True,
    )

    fixed_cases.to_csv(out_dir / "fixed_case_deltas.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(out_dir / "group_delta_summary.csv", index=False, encoding="utf-8-sig")
    regression_samples.to_csv(out_dir / "top_regression_samples_e5a_vs_e2.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(out_dir / "pair_seed_delta_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(out_dir, pair_summary, fixed_cases, group_summary, regression_samples)

    print(f"shape_audit_report: {out_dir / 'shape_audit_report_cn.md'}")
    print(f"pair_seed_delta_summary: {out_dir / 'pair_seed_delta_summary.csv'}")
    print(f"fixed_case_deltas: {out_dir / 'fixed_case_deltas.csv'}")
    print(f"group_delta_summary: {out_dir / 'group_delta_summary.csv'}")
    print(f"top_regression_samples: {out_dir / 'top_regression_samples_e5a_vs_e2.csv'}")


if __name__ == "__main__":
    main()
