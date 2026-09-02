from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (20260902, 20260903, 20260904)
MODELS = ("hold", "linear", "extra_trees", "transformer", "et_transformer_residual")
POPULATIONS = (
    "all_continuous",
    "distance_v2_305",
    "low_mu_v2_70",
    "high_dynamic_not_started",
    "action_started",
    "ordinary",
    "release_v3_historical_2323",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立验证 Run84 数据、切分和结果完整性")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--snapshot-history", action="store_true")
    return parser.parse_args()


def history_files(run_root: Path) -> list[Path]:
    baseline_root = run_root.parent
    files = []
    for child in baseline_root.iterdir():
        name = child.name.lower()
        if not child.is_dir() or not name.startswith("run"):
            continue
        digits = "".join(character for character in name[3:] if character.isdigit())
        if not digits:
            continue
        run_number = int(digits[:2])
        if 57 <= run_number <= 83:
            files.extend(path for path in child.rglob("*") if path.is_file())
    return sorted(files)


def snapshot_history(run_root: Path) -> None:
    rows = []
    for path in history_files(run_root):
        stat = path.stat()
        rows.append(
            {
                "relative_path": str(path.relative_to(run_root.parent)).replace("\\", "/"),
                "size_bytes": stat.st_size,
                "modified_time_ns": stat.st_mtime_ns,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise AssertionError("未找到 Run57—Run83 历史文件")
    frame.to_csv(run_root / "history_readonly_snapshot.csv", index=False, encoding="utf-8-sig")
    print(f"HISTORY SNAPSHOT files={len(frame)}", flush=True)


def check(condition: bool, name: str, checks: list[dict], detail: object = "") -> None:
    checks.append({"check": name, "passed": bool(condition), "detail": str(detail)})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    if args.snapshot_history:
        snapshot_history(run_root)
        return 0

    checks: list[dict] = []
    manifest = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(run_root / "dataset" / "window_metadata.csv", low_memory=False)
    inventory = pd.read_csv(run_root / "tables" / "recording_inventory.csv", low_memory=False)
    mapping = pd.read_csv(run_root / "tables" / "fixed_evaluation_mapping.csv", low_memory=False)
    history = np.load(run_root / "dataset" / "history_20hz.npy", mmap_mode="r")
    targets = np.load(run_root / "dataset" / "targets_relative_20hz.npy", mmap_mode="r")
    features = np.load(run_root / "dataset" / "extratrees_features.npy", mmap_mode="r")

    check(manifest["status"] == "DATASET_COMPLETE", "数据构建状态", checks, manifest["status"])
    check(manifest["causal_input_future_support_used"] is False, "因果输入不使用未来原始支持", checks)
    check(len(inventory) == 221, "recording来源数", checks, len(inventory))
    check((inventory["legal_windows"] > 0).all(), "221条recording均有合法窗口", checks)
    expected_windows = int(manifest["windows"])
    check(len(metadata) == expected_windows, "连续查询窗口数与manifest一致", checks, len(metadata))
    check(metadata["subject_alias"].nunique() == 38, "驾驶员数", checks, metadata["subject_alias"].nunique())
    check(history.shape == (expected_windows, 40, 10), "历史数组形状", checks, history.shape)
    check(targets.shape == (expected_windows, 20, 4), "目标数组形状", checks, targets.shape)
    check(features.shape == (expected_windows, 160), "ExtraTrees特征形状", checks, features.shape)
    check(np.isfinite(history[:, :, :4]).all(), "核心历史通道有限", checks)
    check(np.isfinite(targets).all(), "目标曲线有限", checks)
    check(
        np.array_equal(metadata["window_index"].to_numpy(np.int64), np.arange(len(metadata), dtype=np.int64)),
        "窗口索引连续",
        checks,
    )
    check(metadata.groupby("time_block_id")["recording_alias"].nunique().max() == 1, "时间块不跨recording", checks)
    expected_mapping = {
        "distance_v2_305": 305,
        "low_mu_v2_70": 70,
        "release_v3_historical_2323": 2323,
    }
    check(mapping.groupby("subset").size().to_dict() == expected_mapping, "固定评价人口", checks, mapping.groupby("subset").size().to_dict())
    check(mapping["mapping_error_s"].max() <= 0.100001, "固定锚点映射误差", checks, mapping["mapping_error_s"].max())

    raw_root = run_root / "raw_results"
    fold_metrics = pd.concat(
        [pd.read_csv(raw_root / f"fold_metrics_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    subject_metrics = pd.concat(
        [pd.read_csv(raw_root / f"subject_metrics_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    split_audit = pd.concat(
        [pd.read_csv(raw_root / f"split_audit_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    scalers = pd.concat(
        [pd.read_csv(raw_root / f"scalers_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    inner_split_audit = pd.concat(
        [pd.read_csv(raw_root / f"inner_split_audit_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    training_log = pd.concat(
        [pd.read_csv(raw_root / f"training_log_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    check(set(fold_metrics["seed"]) == set(SEEDS), "三个随机种子", checks, sorted(fold_metrics["seed"].unique()))
    check(set(fold_metrics["fold"]) == {1, 2, 3}, "每种子三折", checks, sorted(fold_metrics["fold"].unique()))
    check(set(fold_metrics["model"]) == set(MODELS), "五模型齐全", checks, sorted(fold_metrics["model"].unique()))
    check(set(fold_metrics["population"]) == set(POPULATIONS), "七评价人口齐全", checks, sorted(fold_metrics["population"].unique()))
    check(split_audit["subject_overlap"].eq(0).all(), "驾驶员零泄漏", checks)
    check(split_audit["recording_overlap"].eq(0).all(), "recording零泄漏", checks)
    check(inner_split_audit["subject_overlap"].eq(0).all(), "残差内层驾驶员零泄漏", checks)
    check(inner_split_audit["recording_overlap"].eq(0).all(), "残差内层recording零泄漏", checks)
    check(len(inner_split_audit.groupby(["seed", "outer_fold"])) == 9, "九个外折均有残差内层审计", checks)
    check(inner_split_audit.groupby(["seed", "outer_fold"]).size().eq(3).all(), "每个外折三个残差内折", checks)
    for seed in SEEDS:
        split_seed = split_audit.loc[split_audit["seed"].eq(seed)]
        check(split_seed["test_windows"].sum() == len(metadata), f"seed{seed} OOF窗口完整", checks, split_seed["test_windows"].sum())
        assignments = pd.read_csv(raw_root / f"fold_assignments_seed{seed}.csv")
        check(len(assignments) == 38 and assignments["subject_alias"].nunique() == 38, f"seed{seed}驾驶员唯一外折", checks)
        for model in MODELS:
            all_rows = fold_metrics.loc[
                fold_metrics["seed"].eq(seed)
                & fold_metrics["model"].eq(model)
                & fold_metrics["population"].eq("all_continuous")
                & fold_metrics["channel"].eq("action_macro")
            ]
            check(all_rows["windows"].sum() == len(metadata), f"seed{seed} {model}全窗口评分完整", checks, all_rows["windows"].sum())
            for population, expected in [("distance_v2_305", 305), ("low_mu_v2_70", 70)]:
                rows = fold_metrics.loc[
                    fold_metrics["seed"].eq(seed)
                    & fold_metrics["model"].eq(model)
                    & fold_metrics["population"].eq(population)
                    & fold_metrics["channel"].eq("action_macro")
                ]
                check(rows["windows"].sum() == expected, f"seed{seed} {model} {population}完整", checks, rows["windows"].sum())
            release = fold_metrics.loc[
                fold_metrics["seed"].eq(seed)
                & fold_metrics["model"].eq(model)
                & fold_metrics["population"].eq("release_v3_historical_2323")
                & fold_metrics["channel"].eq("steer_deg")
            ]
            check(release["windows"].sum() == 2323, f"seed{seed} {model} release历史对照完整", checks, release["windows"].sum())
        subject_all = subject_metrics.loc[
            subject_metrics["seed"].eq(seed)
            & subject_metrics["population"].eq("all_continuous")
            & subject_metrics["channel"].eq("action_macro")
        ]
        check(
            subject_all.groupby("model")["subject_alias"].nunique().eq(38).all(),
            f"seed{seed}五模型逐驾驶员完整",
            checks,
        )

    check(np.isfinite(fold_metrics[["mae", "rmse", "endpoint_mae"]]).all().all(), "模型指标有限", checks)
    check((scalers.loc[scalers["kind"].eq("target"), "scale"] > 0).all(), "训练折目标尺度为正", checks)
    check(len(scalers.groupby(["seed", "fold"])) == 9, "九个训练折标准化记录齐全", checks)
    check(set(training_log["model"]) == {"transformer", "et_transformer_residual"}, "Transformer训练日志齐全", checks)

    required_results = [
        "main_model_comparison.csv",
        "evaluation_population_comparison.csv",
        "driver_paired_results.csv",
        "driver_benefit_summary.csv",
        "curve_example_predictions.csv",
        "personalization_decision.json",
    ]
    check(all((run_root / "results" / name).exists() for name in required_results), "主要结果表齐全", checks)
    required_review = [
        "00_FINAL_CONCLUSION_CN.md",
        "01_DATA_POPULATION_CN.md",
        "02_MAIN_MODEL_COMPARISON_CN.md",
        "03_EVALUATION_POPULATIONS_CN.md",
        "04_DRIVER_PAIRED_SUMMARY_CN.md",
        "05_SCRIPT_INVENTORY_CN.md",
        "06_CURVE_EXAMPLES_CN.md",
    ]
    check(all((run_root / "review_light" / name).exists() for name in required_review), "Review-light文档齐全", checks)
    check(len(list((run_root / "review_light" / "figures").glob("EX*_curves.png"))) == 4, "四组预测曲线示例", checks)

    snapshot_path = run_root / "history_readonly_snapshot.csv"
    check(snapshot_path.exists(), "Run57—Run83只读快照存在", checks)
    snapshot = pd.read_csv(snapshot_path)
    current_rows = []
    for path in history_files(run_root):
        stat = path.stat()
        current_rows.append(
            {
                "relative_path": str(path.relative_to(run_root.parent)).replace("\\", "/"),
                "size_bytes": stat.st_size,
                "modified_time_ns": stat.st_mtime_ns,
            }
        )
    current = pd.DataFrame(current_rows)
    comparison = snapshot.merge(current, on="relative_path", how="outer", suffixes=("_before", "_after"), indicator=True)
    unchanged = (
        comparison["_merge"].eq("both")
        & comparison["size_bytes_before"].eq(comparison["size_bytes_after"])
        & comparison["modified_time_ns_before"].eq(comparison["modified_time_ns_after"])
    )
    check(unchanged.all(), "Run57—Run83历史文件大小和修改时间未变", checks, comparison.loc[~unchanged, "relative_path"].tolist())

    result = {"status": "ALL_PASS", "checks": checks, "check_count": len(checks)}
    (run_root / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output_manifest_path = run_root / "MANIFEST.json"
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    output_manifest["status"] = "VALIDATED_COMPLETE"
    output_manifest["validation"] = "validation.json"
    output_manifest["validation_check_count"] = len(checks)
    output_manifest_path.write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT: ALL PASS -- Run84 {len(checks)} checks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
