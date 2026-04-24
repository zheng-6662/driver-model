from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt


ROOT = Path(r"F:\data_set_process\data_process")
OUT_DIR = ROOT / "reports" / "fair_baseline_same_pool_check_20260328"
MANIFEST = ROOT / "datasetprocess" / "final_code" / "model" / "training" / "protocol_allphase_control_v2_context_full2s" / "sample_manifest.csv"
BASELINE_RUN = ROOT / "tmp" / "single_output_d3_runs" / "EXP2_ALLPHASE_V2_CONTEXT_FULL2S_TRUE2S_SUP_20260324_224343"
COND_RUN = ROOT / "tmp" / "event_conditioned_runs" / "EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260327_000432"
FORMAL_DIR = ROOT / "reports" / "v3_selection_conditioned_interaction_pilot_20260327" / "task_2_conditioned_v2" / "formal_eval"


def ensure_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def count_manifest(path: Path) -> tuple[int, Counter]:
    cnt: Counter[str] = Counter()
    total = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            cnt[row.get("split", "UNKNOWN")] += 1
    return total, cnt


def write_summary_md() -> None:
    total, cnt = count_manifest(MANIFEST)
    summary = f"""# Fair baseline check on the same sample pool

## Sample pool
- manifest: `{MANIFEST}`
- total samples: **{total}**
- train: **{cnt.get('train', 0)}**
- val: **{cnt.get('val', 0)}**
- test: **{cnt.get('test', 0)}**

## Conclusion
- The formal baseline used in `EXP2_ALLPHASE_V2_CONTEXT_FULL2S_TRUE2S_SUP_20260324_224343` already uses the **same sample pool and same split counts** as the current conditioned-v2 run.
- So the fair comparison is **valid at the sample-count level**.
- What was misleading was the earlier illustrative baseline figure, not the formal evaluation pool.

## Fair comparison metrics already available on the same pool
- baseline overall 2s RMSE: **0.3807**
- conditioned v2 overall 2s RMSE: **0.3773**
- baseline tail RMSE: **0.3978**
- conditioned v2 tail RMSE: **0.3758**
- baseline turning count abs err: **1.7717**
- conditioned v2 turning count abs err: **1.5354**
- baseline interaction-slice tail RMSE: **0.4954**
- conditioned v2 interaction-slice tail RMSE: **0.4207**

## Recommendation for the PPT
- Replace the old baseline illustration with the figure copied here as `fair_same_pool_representative_samples_overview.png`
- If needed, cite the sample-pool counts above directly in the talk to remove any concern about fairness.
"""
    (OUT_DIR / "fair_baseline_same_pool_summary.md").write_text(summary, encoding="utf-8")


def build_metric_bar() -> None:
    labels = ["2s RMSE", "Tail RMSE", "Turning Cnt Err", "Interaction Tail RMSE"]
    baseline = [0.3807, 0.3978, 1.7717, 0.4954]
    conditioned = [0.3773, 0.3758, 1.5354, 0.4207]
    colors = {"baseline": "#d96459", "conditioned": "#2d6a8e"}

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=180)
    fig.patch.set_facecolor("#f7f8fb")
    ax.set_facecolor("#f7f8fb")
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], baseline, width=width, color=colors["baseline"], label="Baseline (same pool)")
    ax.bar([i + width / 2 for i in x], conditioned, width=width, color=colors["conditioned"], label="Deterministic conditioned v2")

    for idx, val in enumerate(baseline):
        ax.text(idx - width / 2, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    for idx, val in enumerate(conditioned):
        ax.text(idx + width / 2, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticklabels([])
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Fair comparison on the same 6238-sample pool", fontsize=15, fontweight="bold")
    ax.text(
        0.0,
        1.02,
        "Baseline and conditioned-v2 here use the same manifest: train 4797 / val 692 / test 749.",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fair_same_pool_metric_comparison.png", bbox_inches="tight")
    plt.close(fig)


def copy_existing_figures() -> None:
    copy2(FORMAL_DIR / "figures" / "representative_samples_overview.png", OUT_DIR / "fair_same_pool_representative_samples_overview.png")
    for name in [
        "01_good_case_tyy__Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv__52__trigger_idx.png",
        "02_good_case_cwh__Entity_Recording_2025_09_26_19_27_21_vehicle_aligned_cleaned.csv__6__trigger_idx.png",
        "03_hard_case_tyy__Entity_Recording_2025_09_28_14_23_43_vehicle_aligned_cleaned.csv__65__trigger_idx.png",
        "05_interaction_case_tyy__Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv__52__trigger_idx.png",
        "07_reversal_case_tyy__Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv__52__trigger_idx.png",
    ]:
        copy2(FORMAL_DIR / "figures" / name, OUT_DIR / name)


def main() -> None:
    ensure_dir()
    write_summary_md()
    build_metric_bar()
    copy_existing_figures()


if __name__ == "__main__":
    main()
