from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


MULTIHYP_NPZ = Path(
    r"F:\data_set_process\data_process\tmp\interaction_multihyp_runs\EXP_INTERACTION_MULTIHYP_PILOT_FORMAL_20260327_010459\test_predictions.npz"
)
TASK2_FIG_DIR = Path(
    r"F:\data_set_process\data_process\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\figures"
)
OUTPUT_PATH = Path(
    r"C:\Users\Administrator\Desktop\interaction_multihyp_with_deterministic_context.png"
)


SAMPLES = [
    {
        "sample_key": "tyy::Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv::52::trigger_idx",
        "task2_png": TASK2_FIG_DIR
        / "05_interaction_case_tyy__Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv__52__trigger_idx.png",
        "short_title": "sample #52 | multi_correction | interaction",
    },
    {
        "sample_key": "tyy::Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv::54::trigger_idx",
        "task2_png": TASK2_FIG_DIR
        / "06_interaction_case_tyy__Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv__54__trigger_idx.png",
        "short_title": "sample #54 | single_lobe | interaction",
    },
]


def _crop_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    # Trim outer white margin for tighter layout.
    mask = np.any(arr < 245, axis=2)
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = 8
    y0 = max(0, y0 - pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)
    return arr[y0 : y1 + 1, x0 : x1 + 1]


def main() -> None:
    pred = np.load(MULTIHYP_NPZ, allow_pickle=True)
    sample_keys = pred["sample_key"].astype(str).tolist()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[1.15, 1.0], hspace=0.04)

    top = subfigs[0]
    top.suptitle(
        "Deterministic conditioned v2 context (formal task-2 figures)",
        fontsize=15,
        y=0.98,
    )
    top_axes = top.subplots(1, 2)

    for ax, info in zip(top_axes, SAMPLES):
        ax.imshow(_crop_image(info["task2_png"]))
        ax.set_title(info["short_title"], pad=8)
        ax.axis("off")

    bottom = subfigs[1]
    bottom.suptitle(
        "Interaction multi-hypothesis pilot on the same samples",
        fontsize=15,
        y=0.98,
    )
    bottom_axes = bottom.subplots(1, 2)

    for ax, info in zip(bottom_axes, SAMPLES):
        idx = sample_keys.index(info["sample_key"])
        mask = pred["mask"][idx] > 0.5
        t = np.arange(mask.sum())
        true = pred["true"][idx][mask][:, 1]
        top1 = pred["top1"][idx][mask][:, 1]
        oracle = pred["oracle"][idx][mask][:, 1]
        hyps = pred["hypotheses"][idx][:, mask, 1]

        for hyp in hyps:
            ax.plot(t, hyp, color="#B9BDC5", lw=1.4, alpha=0.7, zorder=1)

        ax.plot(t, true, color="#222222", lw=2.6, label="GT", zorder=4)
        ax.plot(t, top1, color="#F4A261", lw=3.0, label="Top1", zorder=5)
        ax.plot(t, oracle, color="#2A9D8F", lw=2.5, ls="--", label="Oracle", zorder=6)

        top1_idx = int(pred["top1_idx"][idx])
        oracle_idx = int(pred["oracle_idx"][idx])
        ax.set_title(
            f"{info['short_title']}\nTop1 idx={top1_idx}, Oracle idx={oracle_idx}"
        )
        ax.set_xlabel("Future step")
        ax.set_ylabel("Pilot output")
        ax.grid(alpha=0.22, linestyle="--")

    handles = [
        plt.Line2D([0], [0], color="#B9BDC5", lw=2, alpha=0.7, label="Other hypotheses"),
        plt.Line2D([0], [0], color="#222222", lw=2.6, label="GT"),
        plt.Line2D([0], [0], color="#F4A261", lw=3.0, label="Top1"),
        plt.Line2D([0], [0], color="#2A9D8F", lw=2.5, ls="--", label="Oracle"),
    ]
    bottom.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.96))

    fig.suptitle(
        "Interaction cases with deterministic-v2 context\n"
        "Top row keeps the original deterministic conditioned-v2 formal figures; bottom row shows multi-hyp Top1/Oracle on the same samples.",
        fontsize=17,
        y=0.995,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
