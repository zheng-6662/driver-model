from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_ROOT = Path(
    r"F:\data_set_process\data_process\03_results\tmp\interaction_multihyp_runs\EXP_INTERACTION_MULTIHYP_PILOT_FORMAL_20260327_010459"
)
COMPARISON_CSV = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\v3_selection_conditioned_interaction_pilot_20260327\task_3_interaction_multihyp\formal_eval\interaction_sample_level_comparison.csv"
)
OUTPUT_PATH = Path(
    r"C:\Users\Administrator\Desktop\interaction_multihyp_pilot_corrected.png"
)


def _short_case_name(sample_key: str) -> str:
    subj, recording, idx, anchor = sample_key.split("::")
    recording_short = recording.replace("Entity_Recording_", "").replace(
        "_vehicle_aligned_cleaned.csv", ""
    )
    return f"{subj} | {recording_short} | #{idx}"


def main() -> None:
    pred = np.load(RUN_ROOT / "test_predictions.npz", allow_pickle=True)
    df = pd.read_csv(COMPARISON_CSV)

    pred_df = pd.DataFrame(
        {
            "sample_key": pred["sample_key"].astype(str),
            "top1_idx": pred["top1_idx"],
            "oracle_idx": pred["oracle_idx"],
        }
    )
    merged = df.merge(pred_df, on="sample_key", how="left")
    merged["top1_vs_oracle_gap"] = (
        merged["delta_top1_rmse_2s_abs_steer"] - merged["delta_oracle_rmse_2s_abs_steer"]
    )
    merged["idx_diff"] = merged["top1_idx"] != merged["oracle_idx"]

    chosen = (
        merged.loc[merged["idx_diff"]]
        .sort_values("top1_vs_oracle_gap", ascending=False)
        .head(4)
        .reset_index(drop=True)
    )

    sample_index = {k: i for i, k in enumerate(pred["sample_key"].astype(str))}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        idx = sample_index[row["sample_key"]]
        mask = pred["mask"][idx] > 0.5
        true = pred["true"][idx, mask, 0]
        top1 = pred["top1"][idx, mask, 0]
        oracle = pred["oracle"][idx, mask, 0]
        hyps = pred["hypotheses"][idx, :, :, 0][:, mask]
        t = np.arange(mask.sum())

        for hyp_idx in range(hyps.shape[0]):
            ax.plot(
                t,
                hyps[hyp_idx],
                color="#B9BDC5",
                lw=1.6,
                alpha=0.65,
                zorder=1,
            )

        ax.plot(t, true, color="#222222", lw=2.8, label="GT", zorder=4)
        ax.plot(t, oracle, color="#2A9D8F", lw=2.8, ls="--", label="Oracle", zorder=5)
        ax.plot(t, top1, color="#F4A261", lw=3.4, label="Top1", zorder=6)

        ax.set_title(
            f"{_short_case_name(row['sample_key'])}\n"
            f"Top1 idx={int(row['top1_idx'])}, Oracle idx={int(row['oracle_idx'])}, "
            f"2s RMSE gap={row['top1_vs_oracle_gap']:.3f}"
        )
        ax.set_xlabel("Future step")
        ax.set_ylabel("Steering")
        ax.grid(alpha=0.25, linestyle="--")

    handles = [
        plt.Line2D([0], [0], color="#B9BDC5", lw=2, alpha=0.7, label="Other hypotheses"),
        plt.Line2D([0], [0], color="#222222", lw=2.8, label="GT"),
        plt.Line2D([0], [0], color="#F4A261", lw=3.4, label="Top1"),
        plt.Line2D([0], [0], color="#2A9D8F", lw=2.8, ls="--", label="Oracle"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Interaction multi-hypothesis pilot: corrected representative cases\n"
        "Selected only from samples where Top1 and Oracle choose different hypotheses, so the orange Top1 curve is visible.",
        fontsize=15,
        y=1.06,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

