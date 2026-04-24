from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(r"F:\data_set_process\data_process")
MASTER_CSV = ROOT / "reports" / "attribution_master_table.csv"
FIG_DIR = (
    ROOT
    / "reports"
    / "v3_selection_conditioned_interaction_pilot_20260327"
    / "task_2_conditioned_v2"
    / "formal_eval"
    / "figures"
)
OUTPUT_PNG = ROOT / "reports" / "conditioned_v2_representative_cases_20260408.png"
OUTPUT_MD = ROOT / "reports" / "conditioned_v2_representative_cases_20260408.md"


SAMPLES = [
    {
        "figure_name": "03_hard_case_tyy__Entity_Recording_2025_09_28_14_23_43_vehicle_aligned_cleaned.csv__65__trigger_idx.png",
        "sample_key": "tyy::Entity_Recording_2025_09_28_14_23_43_vehicle_aligned_cleaned.csv::65::trigger_idx",
        "tag": "Q1_fast worst-case",
        "takeaway": "Tail shape and amplitude both break badly; this is the clearest fast-reaction failure case.",
    },
    {
        "figure_name": "04_hard_case_tyy__Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv__55__trigger_idx.png",
        "sample_key": "tyy::Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv::55::trigger_idx",
        "tag": "Shape-heavy failure",
        "takeaway": "Boundary does not worsen much, but the conditioned tail still deviates strongly in shape.",
    },
    {
        "figure_name": "06_interaction_case_tyy__Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv__54__trigger_idx.png",
        "sample_key": "tyy::Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv::54::trigger_idx",
        "tag": "Single-lobe boundary-heavy",
        "takeaway": "A single-lobe case where conditioned remains close overall but boundary shift grows noticeably.",
    },
    {
        "figure_name": "08_reversal_case_cwh__Entity_Recording_2025_09_26_19_27_21_vehicle_aligned_cleaned.csv__6__trigger_idx.png",
        "sample_key": "cwh::Entity_Recording_2025_09_26_19_27_21_vehicle_aligned_cleaned.csv::6::trigger_idx",
        "tag": "Reverse-correction contrast",
        "takeaway": "Reverse-correction does worsen, but not nearly as catastrophically as the Q1_fast single-lobe failures.",
    },
    {
        "figure_name": "05_interaction_case_tyy__Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv__52__trigger_idx.png",
        "sample_key": "tyy::Entity_Recording_2025_09_28_14_57_17_vehicle_aligned_cleaned.csv::52::trigger_idx",
        "tag": "Improved control sample",
        "takeaway": "A positive control: conditioned clearly helps here, so the main issue is not that the method never works.",
    },
]


METRIC_COLUMNS = [
    "latency_proxy_bucket",
    "eval_morphology_label",
    "subj",
    "interaction_slice",
    "delta_rmse_tail_abs_steer",
    "delta_boundary_shift_abs_err",
    "shape_corr_conditioned",
    "peak_abs_amp_err_conditioned",
]


def crop_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    mask = np.any(arr < 245, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return arr
    pad = 8
    y0 = max(0, ys.min() - pad)
    y1 = min(arr.shape[0] - 1, ys.max() + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(arr.shape[1] - 1, xs.max() + pad)
    return arr[y0 : y1 + 1, x0 : x1 + 1]


def short_case_name(sample_key: str) -> str:
    subj, recording, idx, _anchor = sample_key.split("::")
    recording_short = (
        recording.replace("Entity_Recording_", "")
        .replace("_vehicle_aligned_cleaned.csv", "")
        .replace("_", "-")
    )
    return f"{subj} | {recording_short} | #{idx}"


def fmt_signed(value: float) -> str:
    return f"{value:+.3f}"


def build_panel(master: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(16, 17), constrained_layout=True)
    axes = axes.ravel()

    lines_for_md = [
        "# conditioned v2 representative cases (2026-04-08)",
        "",
        "This panel reuses formal-eval single-sample plots and reorders them to match the 2026-04-08 attribution findings.",
        "",
    ]

    for ax, sample in zip(axes, SAMPLES):
        row = master.loc[master["sample_key"] == sample["sample_key"], METRIC_COLUMNS + ["sample_key"]]
        if row.empty:
            raise ValueError(f"Missing sample in attribution_master_table.csv: {sample['sample_key']}")
        row = row.iloc[0]

        image = crop_image(FIG_DIR / sample["figure_name"])
        ax.imshow(image)
        ax.axis("off")

        metric_text = (
            f"{sample['tag']}\n"
            f"{short_case_name(sample['sample_key'])}\n"
            f"latency={row['latency_proxy_bucket']} | morph={row['eval_morphology_label']} | subj={row['subj']}\n"
            f"delta tail={fmt_signed(row['delta_rmse_tail_abs_steer'])} | "
            f"delta boundary={fmt_signed(row['delta_boundary_shift_abs_err'])}\n"
            f"shape corr={row['shape_corr_conditioned']:.3f} | "
            f"peak amp err={row['peak_abs_amp_err_conditioned']:.3f}\n"
            f"{sample['takeaway']}"
        )
        ax.set_title(metric_text, loc="left", pad=10)

        lines_for_md.extend(
            [
                f"## {sample['tag']}",
                f"- sample_key: `{sample['sample_key']}`",
                f"- latency bucket: `{row['latency_proxy_bucket']}`",
                f"- morphology: `{row['eval_morphology_label']}`",
                f"- subject: `{row['subj']}`",
                f"- interaction slice: `{row['interaction_slice']}`",
                f"- delta tail RMSE: `{fmt_signed(row['delta_rmse_tail_abs_steer'])}`",
                f"- delta boundary shift: `{fmt_signed(row['delta_boundary_shift_abs_err'])}`",
                f"- shape corr conditioned: `{row['shape_corr_conditioned']:.3f}`",
                f"- peak abs amp err conditioned: `{row['peak_abs_amp_err_conditioned']:.3f}`",
                f"- takeaway: {sample['takeaway']}",
                "",
            ]
        )

    axes[-1].axis("off")
    axes[-1].text(
        0.03,
        0.90,
        "How to read these panels",
        fontsize=15,
        fontweight="bold",
        transform=axes[-1].transAxes,
    )
    axes[-1].text(
        0.03,
        0.75,
        "\n".join(
            [
                "1. delta tail > 0 means conditioned is worse than baseline on the tail segment.",
                "2. delta boundary > 0 means conditioned shifts the boundary more than baseline.",
                "3. shape corr lower means the conditioned curve looks less like the true tail shape.",
                "4. peak amp err higher means the conditioned peak magnitude is off by more.",
                "5. The control sample is there to show conditioned still helps on some cases.",
            ]
        ),
        fontsize=12,
        va="top",
        transform=axes[-1].transAxes,
    )

    fig.suptitle(
        "conditioned v2 representative cases after the 2026-04-08 attribution pass\n"
        "Reordered to separate fast-reaction tail-shape failures from morphology-driven boundary failures",
        fontsize=17,
        y=1.01,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    OUTPUT_MD.write_text("\n".join(lines_for_md), encoding="utf-8")


def main() -> None:
    master = pd.read_csv(MASTER_CSV)
    build_panel(master)
    print(f"saved: {OUTPUT_PNG}")
    print(f"saved: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
