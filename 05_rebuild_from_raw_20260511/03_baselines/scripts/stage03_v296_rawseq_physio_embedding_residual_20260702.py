from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


"""
v296: raw physiology sequence embedding for wait1 residual correction.

v295 用 v293 的手工统计特征做 wait1 residual correction，结果显示：
    1) 生理特征在 test 诊断最优时能带来一点差样本收益；
    2) 但 validation 选择出的 deployable 策略不稳定，且弱于非生理 ablation。

v296 检查一个更根本的问题：
    是不是手工统计特征太浅，原始生理窗口的时序形状其实能提供更好的残差信息？

做法：
    - 仍使用 v249 delay=1000 的 wait1 rolling prediction 作为 baseline；
    - 对每个事件读取原始 200Hz 生理记录；
    - 用事件自身 -60~-20s baseline 做 robust z；
    - 将 post0_1 的每个信号重采样到固定长度；
    - 只在 train split 上拟合 PCA，得到 raw-sequence physiology embedding；
    - 复用 v295 的 residual/gate/val-threshold/test-report 评估框架。
"""


SEED = 20260702
RESAMPLE_N = 32
PCA_COMPONENTS = [8, 16, 32]

BASELINES = Path(__file__).resolve().parents[1]
SCRIPTS = BASELINES / "scripts"
OUT = BASELINES / "v296_rawseq_physio_embedding_residual_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v296_rawseq_physio_embedding_residual_20260702_pack.zip"

V293_SCRIPT = SCRIPTS / "stage03_v293_physio_response_visibility_latency_audit_20260702.py"
V295_SCRIPT = SCRIPTS / "stage03_v295_wait1_direct_residual_physio_20260702.py"
V249_NPZ = BASELINES / "v249_shape_aware_curve_model_20260630" / "v249_shape_aware_predictions.npz"
V293_FEATURES = (
    BASELINES
    / "v293_physio_response_visibility_latency_audit_20260702"
    / "tables"
    / "v293_prepost_physio_visibility_features.csv"
)
V293_SCREEN = (
    BASELINES
    / "v293_physio_response_visibility_latency_audit_20260702"
    / "tables"
    / "v293_train_only_feature_screen.csv"
)
V295_GUARDRAIL = BASELINES / "v295_wait1_direct_residual_physio_20260702" / "logs" / "guardrail_check.json"
THIS_SCRIPT = Path(__file__).resolve()


def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V293 = import_module_from_path("stage03_v293_for_v296", V293_SCRIPT)
V295 = import_module_from_path("stage03_v295_for_v296", V295_SCRIPT)


def ensure_dirs() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for p in [TABLES, FIGURES, REPORTS, LOGS]:
        p.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def finite(values: Iterable[object]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def resample_window(times: np.ndarray, z: np.ndarray, start: float, end: float, n: int) -> Tuple[np.ndarray, float]:
    grid = np.linspace(start, end, n)
    mask = (times >= start) & (times <= end) & np.isfinite(times) & np.isfinite(z)
    if int(mask.sum()) < 3:
        return np.full(n, np.nan, dtype=float), 0.0
    t = times[mask]
    vals = z[mask]
    # 去除重复时间戳，避免 np.interp 在局部平台上行为不稳定。
    uniq_t, uniq_idx = np.unique(t, return_index=True)
    vals = vals[uniq_idx]
    if len(uniq_t) < 3:
        return np.full(n, np.nan, dtype=float), 0.0
    out = np.interp(grid, uniq_t, vals)
    return out.astype(float), float(len(uniq_t) / max(1, int((times >= start).sum() - int((times > end).sum()))))


def build_rawseq_features(data) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按事件抽取 post0_1 原始生理序列特征。"""

    df = data.df.copy().reset_index(drop=True)
    inventory = V293.V285.load_physio_inventory()
    samples = df[["event_uid", "subject", "recording", "split", "observation_s"]].copy()
    samples["session_stamp"] = samples["recording"].map(V293.V285.session_stamp_from_recording)
    rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    grouped = list(samples.groupby(["subject", "session_stamp"], sort=False))
    for group_i, ((subject, session), group) in enumerate(grouped, start=1):
        path = inventory.get((str(subject), str(session)))
        if path is None or not path.exists():
            for _, sample in group.iterrows():
                rows.append({"event_uid": sample["event_uid"], "v296_rawseq_status": "missing_physio"})
            audit_rows.append({"subject": subject, "session_stamp": session, "status": "missing_physio", "event_n": len(group)})
            continue
        print(f"[v296] rawseq group {group_i}/{len(grouped)} subject={subject} session={session}", flush=True)
        rec = V293.V285.read_physio_recording(path)
        times = pd.to_numeric(rec["t_s"], errors="coerce").to_numpy(dtype=float)
        arrays = {
            col: pd.to_numeric(rec[col], errors="coerce").to_numpy(dtype=float)
            for col in rec.columns
            if col != "t_s"
        }
        for _, sample in group.iterrows():
            obs = float(sample["observation_s"])
            b_start = max(0.0, obs + V293.BASELINE_WINDOW[0])
            b_end = max(0.0, obs + V293.BASELINE_WINDOW[1])
            baseline_idx = (times >= b_start) & (times <= b_end)
            row: Dict[str, object] = {
                "event_uid": sample["event_uid"],
                "v296_rawseq_status": "ok",
                "v296_baseline_rows": int(baseline_idx.sum()),
            }
            for signal, candidates in V293.SIGNAL_SPECS.items():
                chosen_col, raw = V293.choose_signal(arrays, candidates, baseline_idx)
                raw = np.asarray(raw, dtype=float)
                baseline = raw[baseline_idx] if len(raw) else np.array([], dtype=float)
                z = V293.robust_z(raw, baseline) if len(raw) else np.full_like(times, np.nan, dtype=float)
                vals, valid_ratio = resample_window(times, z, obs, obs + 1.0, RESAMPLE_N)
                row[f"rawseq_post0_1_{signal}_chosen_col"] = chosen_col
                row[f"rawseq_post0_1_{signal}_valid_ratio"] = valid_ratio
                for j, value in enumerate(vals):
                    row[f"rawseq_post0_1_{signal}_z_t{j:02d}"] = value
            rows.append(row)
        audit_rows.append({"subject": subject, "session_stamp": session, "status": "ok", "event_n": len(group), "physio_file": str(path)})
    raw = pd.DataFrame(rows)
    data_cols = [c for c in raw.columns if c.startswith("rawseq_post0_1_") and "_z_t" in c]
    raw["rawseq_feature_finite_rate"] = raw[data_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().mean(axis=1)
    out = df.merge(raw, on="event_uid", how="left", validate="one_to_one")
    return out, pd.DataFrame(audit_rows)


def fit_pca_embeddings(raw: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    train = raw["split"].astype(str).eq("train").to_numpy()
    raw_cols = [c for c in raw.columns if c.startswith("rawseq_post0_1_") and "_z_t" in c]
    x = raw[raw_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    blocks: Dict[str, pd.DataFrame] = {}
    audit_rows: List[Dict[str, object]] = []
    for n_comp in PCA_COMPONENTS:
        n_eff = min(n_comp, max(1, int(train.sum()) - 1), x.shape[1])
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=n_eff, random_state=SEED))
        z = pipe.fit_transform(x[train])
        full = pipe.transform(x)
        pca = pipe.named_steps["pca"]
        cols = [f"rawseq_physio_pca{n_eff}_{i:02d}" for i in range(n_eff)]
        blocks[f"rawseq_physio_pca{n_eff}"] = pd.DataFrame(full, columns=cols)
        audit_rows.append(
            {
                "embedding_block": f"rawseq_physio_pca{n_eff}",
                "raw_feature_n": int(x.shape[1]),
                "component_n": int(n_eff),
                "train_n": int(train.sum()),
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            }
        )
    return blocks, pd.DataFrame(audit_rows)


def build_blocks(data, raw: pd.DataFrame, pca_blocks: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    train = data.df["split"].astype(str).eq("train").to_numpy()
    base = V295.curve_feature_frame("base", data.baseline_eval, data.eval_grid_s).reset_index(drop=True)
    obs = V295.curve_feature_frame("obs0_1", data.observed_prefix, data.observed_grid_s).reset_index(drop=True)
    subject = V295.subject_frame(data.df.reset_index(drop=True), train).reset_index(drop=True)

    feature_blocks: Dict[str, pd.DataFrame] = {
        "base_curve_only": base,
        "base_plus_observed_vehicle_prefix": pd.concat([base, obs], axis=1),
    }
    risk_blocks: Dict[str, pd.DataFrame] = {}
    for name, emb in pca_blocks.items():
        emb = emb.reset_index(drop=True)
        feature_blocks[f"base_plus_{name}"] = pd.concat([base, emb], axis=1)
        feature_blocks[f"base_plus_{name}_subject"] = pd.concat([base, emb, subject], axis=1)
        feature_blocks[f"base_plus_vehicle_prefix_{name}_subject"] = pd.concat([base, obs, emb, subject], axis=1)
        feature_blocks[f"vehicle_prefix_{name}_subject"] = pd.concat([obs, emb, subject], axis=1)
        risk_blocks[f"{name}_subject"] = pd.concat([emb, subject], axis=1)
        risk_blocks[f"base_{name}_subject"] = pd.concat([base, emb, subject], axis=1)
        risk_blocks[f"vehicle_prefix_{name}_subject"] = pd.concat([obs, emb, subject], axis=1)
    feature_blocks = {k: v.loc[:, ~v.columns.duplicated()].copy() for k, v in feature_blocks.items()}
    risk_blocks = {k: v.loc[:, ~v.columns.duplicated()].copy() for k, v in risk_blocks.items()}
    return feature_blocks, risk_blocks


def choose_rawseq_rows(summary: pd.DataFrame) -> pd.DataFrame:
    chosen = V295.choose_rows(summary)
    extra: List[pd.Series] = []
    candidates = summary[summary["val_candidate_ok"]].copy()
    raw = candidates[candidates["feature_block"].astype(str).str.contains("rawseq_physio", na=False)].copy()
    if not raw.empty:
        row = raw.sort_values(
            ["val_bad_top10_delta_vs_baseline_mean", "val_all_delta_vs_baseline_mean", "shrinkage"],
            ascending=[True, True, True],
        ).iloc[0].copy()
        row["choice_name"] = "best_val_rawseq_physio_deployable"
        row["choice_rule"] = "val no-harm/improve, restricted to raw sequence physiology embeddings"
        extra.append(row)
    diag = summary[
        summary["feature_block"].astype(str).str.contains("rawseq_physio", na=False)
        & summary["risk_tag"].ne("no_override")
        & (summary["val_all_delta_vs_baseline_mean"] <= 0.006)
    ].copy()
    if not diag.empty:
        row = diag.sort_values("test_bad_top10_delta_vs_baseline_mean", ascending=True).iloc[0].copy()
        row["choice_name"] = "test_best_rawseq_diagnostic_not_deployable"
        row["choice_rule"] = "diagnostic only; selected by test bad_top10 within val all bound"
        extra.append(row)
    if extra:
        chosen = pd.concat([chosen, pd.DataFrame(extra)], ignore_index=True)
    if chosen.empty:
        return chosen
    return chosen.drop_duplicates(["choice_name", "selector_tag"]).reset_index(drop=True)


def build_guardrail(data, raw: pd.DataFrame, pca_audit: pd.DataFrame, chosen: pd.DataFrame, risk_audit: pd.DataFrame) -> Dict[str, object]:
    def choice(name: str):
        hit = chosen[chosen["choice_name"].eq(name)]
        return None if hit.empty else hit.iloc[0]

    raw_choice = choice("best_val_rawseq_physio_deployable") or choice("best_val_physio_deployable")
    nonphys = choice("best_val_nonphysio_ablation")
    diag = choice("test_best_rawseq_diagnostic_not_deployable")
    best_raw_bad = float(raw_choice["test_bad_top10_delta_vs_baseline_mean"]) if raw_choice is not None else math.nan
    best_raw_all = float(raw_choice["test_all_delta_vs_baseline_mean"]) if raw_choice is not None else math.nan
    best_non_bad = float(nonphys["test_bad_top10_delta_vs_baseline_mean"]) if nonphys is not None else math.nan
    route_viable = bool(
        raw_choice is not None
        and best_raw_bad <= -0.05
        and best_raw_all <= 0.005
        and float(raw_choice["val_bad_top10_delta_vs_baseline_mean"]) < 0
        and float(raw_choice["val_all_delta_vs_baseline_mean"]) <= 0.003
    )
    weak = bool(
        raw_choice is not None
        and best_raw_bad < -0.015
        and best_raw_all <= 0.01
        and float(raw_choice["val_bad_top10_delta_vs_baseline_mean"]) < 0
    )
    return {
        "pass": True,
        "event_n": int(len(data.df)),
        "wait_ms": int(V295.WAIT_MS),
        "eval_point_n": int(data.y_eval.shape[1]),
        "rawseq_window": "post0_1",
        "rawseq_resample_n": int(RESAMPLE_N),
        "rawseq_finite_rate_mean": float(pd.to_numeric(raw["rawseq_feature_finite_rate"], errors="coerce").mean()),
        "pca_blocks": pca_audit.to_dict(orient="records"),
        "uses_post_observation": True,
        "post_features_are_wait_policy_only": True,
        "test_used_for_feature_screen_model_or_threshold": False,
        "chosen_rawseq_exists": raw_choice is not None,
        "best_rawseq_test_badtop10_delta": best_raw_bad,
        "best_rawseq_test_all_delta": best_raw_all,
        "best_nonphysio_test_badtop10_delta": best_non_bad,
        "rawseq_increment_vs_nonphysio_badtop10_delta": best_raw_bad - best_non_bad if np.isfinite(best_raw_bad) and np.isfinite(best_non_bad) else math.nan,
        "best_rawseq_diagnostic_test_badtop10_delta": float(diag["test_bad_top10_delta_vs_baseline_mean"]) if diag is not None else math.nan,
        "best_risk_test_auc": float(risk_audit["test_auc"].max()) if not risk_audit.empty else math.nan,
        "route_viable_now": route_viable,
        "weak_rawseq_physio_signal_exists": weak,
        "goal_achieved_now": route_viable,
    }


def markdown_table(df: pd.DataFrame, cols: List[str], max_rows: int = 20) -> str:
    return V295.markdown_table(df, cols, max_rows)


def plot_choice_bars(chosen: pd.DataFrame) -> Path:
    path = FIGURES / "v296_chosen_selector_test_delta.png"
    if chosen.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No chosen selector", ha="center", va="center")
    else:
        data = chosen.copy()
        x = np.arange(len(data))
        fig, ax = plt.subplots(figsize=(max(11, len(data) * 2.0), 5))
        ax.bar(x - 0.18, data["test_all_delta_vs_baseline_mean"], width=0.36, label="test all")
        ax.bar(x + 0.18, data["test_bad_top10_delta_vs_baseline_mean"], width=0.36, label="test bad_top10")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(data["choice_name"].astype(str), rotation=25, ha="right")
        ax.set_ylabel("delta RMSE vs v249 wait1 baseline")
        ax.set_title("v296 raw-sequence physiology embedding selectors")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(data, raw: pd.DataFrame, pca_audit: pd.DataFrame, risk_audit: pd.DataFrame, chosen: pd.DataFrame, summary: pd.DataFrame, guardrail: Dict[str, object]) -> Path:
    lines: List[str] = []
    lines.append("# v296 raw-sequence physiology embedding residual 审计")
    lines.append("")
    lines.append("## 结论")
    if guardrail["route_viable_now"]:
        lines.append("- v296 达到 raw-sequence physiology residual 路线可用标准。")
    else:
        lines.append("- v296 仍未达到“生理数据本质性改善差样本”的标准。")
    lines.append(
        f"- best rawseq deployable: test bad_top10 delta={guardrail.get('best_rawseq_test_badtop10_delta', math.nan):.6f}, "
        f"test all delta={guardrail.get('best_rawseq_test_all_delta', math.nan):.6f}."
    )
    lines.append(
        f"- nonphysio ablation test bad_top10 delta={guardrail.get('best_nonphysio_test_badtop10_delta', math.nan):.6f}; "
        f"rawseq 相对非生理增量={guardrail.get('rawseq_increment_vs_nonphysio_badtop10_delta', math.nan):.6f}."
    )
    lines.append("")
    lines.append("## 方法")
    lines.append("- 使用原始 200Hz 生理记录；每个事件按 observation_s 对齐。")
    lines.append("- 每个信号用事件自身 -60~-20s baseline 做 robust z，然后抽取 post0_1 并重采样到 32 点。")
    lines.append("- PCA 只在 train split 上拟合；residual/gate/threshold 复用 v295 框架。")
    lines.append("")
    lines.append("## raw sequence coverage")
    lines.append(f"- rawseq finite rate mean = {guardrail['rawseq_finite_rate_mean']:.6f}")
    lines.append(markdown_table(raw.groupby(["split", "v296_rawseq_status"]).size().reset_index(name="n"), ["split", "v296_rawseq_status", "n"], 20))
    lines.append("")
    lines.append("## PCA audit")
    lines.append(markdown_table(pca_audit, ["embedding_block", "raw_feature_n", "component_n", "explained_variance_ratio_sum"], 20))
    lines.append("")
    lines.append("## chosen selectors")
    cols = [
        "choice_name",
        "feature_block",
        "model_name",
        "shrinkage",
        "risk_tag",
        "threshold",
        "risk_val_auc",
        "risk_test_auc",
        "val_all_delta_vs_baseline_mean",
        "val_bad_top10_delta_vs_baseline_mean",
        "test_all_delta_vs_baseline_mean",
        "test_bad_top10_delta_vs_baseline_mean",
        "test_bad_top10_vehicle_ambiguous_delta_vs_baseline_mean",
        "test_bad_top10_override_rate",
    ]
    lines.append(markdown_table(chosen, cols, 30))
    lines.append("")
    lines.append("## risk audit")
    lines.append(markdown_table(risk_audit.sort_values("test_auc", ascending=False), ["risk_tag", "val_auc", "test_auc", "feature_n"], 20))
    lines.append("")
    lines.append("## top validation rawseq candidates")
    top = summary[
        summary["val_candidate_ok"].astype(bool)
        & summary["feature_block"].astype(str).str.contains("rawseq_physio", na=False)
    ].sort_values("val_bad_top10_delta_vs_baseline_mean").head(30)
    lines.append(
        markdown_table(
            top,
            [
                "feature_block",
                "model_name",
                "shrinkage",
                "risk_tag",
                "risk_val_auc",
                "val_all_delta_vs_baseline_mean",
                "val_bad_top10_delta_vs_baseline_mean",
                "test_all_delta_vs_baseline_mean",
                "test_bad_top10_delta_vs_baseline_mean",
            ],
            30,
        )
    )
    lines.append("")
    lines.append("## guardrail")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    path = REPORTS / "v296_rawseq_physio_embedding_residual_cn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))
        zf.write(THIS_SCRIPT, Path("scripts") / THIS_SCRIPT.name)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"zip test failed at {bad}")


def main() -> None:
    np.random.seed(SEED)
    ensure_dirs()
    input_hashes = pd.DataFrame(
        [
            {"path": str(V249_NPZ), "sha256": file_sha256(V249_NPZ), "role": "curve truth and v249 baseline"},
            {"path": str(V293_FEATURES), "sha256": file_sha256(V293_FEATURES), "role": "event metadata and v293 features"},
            {"path": str(V293_SCREEN), "sha256": file_sha256(V293_SCREEN), "role": "upstream train-only feature screen"},
            {"path": str(V295_GUARDRAIL), "sha256": file_sha256(V295_GUARDRAIL), "role": "v295 guardrail"},
            {"path": str(V293_SCRIPT), "sha256": file_sha256(V293_SCRIPT), "role": "raw physiology reader reuse"},
            {"path": str(V295_SCRIPT), "sha256": file_sha256(V295_SCRIPT), "role": "wait1 residual evaluator reuse"},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    print("[v296] loading v295 curve data", flush=True)
    data = V295.load_curve_data()
    raw, raw_audit = build_rawseq_features(data)
    pca_blocks, pca_audit = fit_pca_embeddings(raw)
    feature_blocks, risk_blocks = build_blocks(data, raw, pca_blocks)
    write_csv(raw_audit, TABLES / "v296_rawseq_recording_audit.csv")
    write_csv(raw.drop(columns=[c for c in raw.columns if c.endswith("_chosen_col")], errors="ignore"), TABLES / "v296_rawseq_event_features.csv")
    write_csv(pca_audit, TABLES / "v296_rawseq_pca_audit.csv")
    write_csv(pd.DataFrame([{"feature_block": k, "feature_n": v.shape[1]} for k, v in feature_blocks.items()]), TABLES / "v296_feature_block_audit.csv")
    write_csv(pd.DataFrame([{"risk_block": k, "feature_n": v.shape[1]} for k, v in risk_blocks.items()]), TABLES / "v296_risk_feature_block_audit.csv")

    residual_preds = V295.fit_residual_predictions(data, feature_blocks)
    risk_scores, risk_audit = V295.fit_risk_scores(data, risk_blocks)
    summary, selected = V295.evaluate_configs(data, residual_preds, risk_scores)
    chosen = choose_rawseq_rows(summary)
    selected_chosen = V295.selector_prediction_table(data, chosen, selected)
    guardrail = build_guardrail(data, raw, pca_audit, chosen, risk_audit)

    write_csv(summary, TABLES / "v296_rawseq_residual_selector_summary.csv")
    write_csv(risk_audit, TABLES / "v296_rawseq_badtop10_risk_audit.csv")
    write_csv(chosen, TABLES / "v296_chosen_by_val.csv")
    write_csv(selected_chosen, TABLES / "v296_chosen_event_predictions.csv")
    write_json(guardrail, LOGS / "guardrail_check.json")
    plot_choice_bars(chosen)
    write_report(data, raw, pca_audit, risk_audit, chosen, summary, guardrail)

    inventory = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            inventory.append({"path": str(p), "bytes": int(p.stat().st_size)})
    write_csv(pd.DataFrame(inventory), LOGS / "file_inventory.csv")
    make_zip()
    print("[v296] done", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
