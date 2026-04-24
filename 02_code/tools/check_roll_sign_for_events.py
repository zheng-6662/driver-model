from __future__ import annotations

from pathlib import Path
import argparse

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROLL_MAT = Path(r"F:\data_set_process\roll_base.mat")
OUTPUT_DIR = Path(r"F:\data_set_process\Carsim_validation_event_extract_files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-xlsx", action="append", required=True, help="Can be used multiple times")
    return parser.parse_args()


def load_event_roll(event_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(event_xlsx, skiprows=15)
    roll_col = next(col for col in df.columns if "roll" in str(col).lower())
    out = df[["t", roll_col]].copy()
    out.columns = ["t_event_s", "roll_event_rad"]
    out = out[(out["t_event_s"] >= 0.0) & (out["t_event_s"] <= 2.0)].copy()
    out.sort_values("t_event_s", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def load_carsim_roll() -> pd.DataFrame:
    with h5py.File(ROLL_MAT, "r") as f:
        refs = f["#refs#"]
        time_s = refs["p"][()].reshape(-1)
        roll_deg = refs["z"][()].reshape(-1)
    out = pd.DataFrame(
        {
            "t_s": time_s,
            "roll_carsim_deg": roll_deg,
            "roll_carsim_rad": np.deg2rad(roll_deg),
        }
    )
    out = out[(out["t_s"] >= 0.0) & (out["t_s"] <= 2.0)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def compute_metrics(ref: np.ndarray, sig: np.ndarray) -> dict[str, float]:
    error = sig - ref
    return {
        "rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
        "mae_rad": float(np.mean(np.abs(error))),
        "max_abs_rad": float(np.max(np.abs(error))),
        "corr": float(np.corrcoef(sig, ref)[0, 1]),
    }


def evaluate_event(event_xlsx: Path, carsim_roll: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    event_df = load_event_roll(event_xlsx)
    comp_df = pd.DataFrame({"t_s": carsim_roll["t_s"]})
    comp_df["roll_event_interp_rad"] = np.interp(
        comp_df["t_s"],
        event_df["t_event_s"],
        event_df["roll_event_rad"],
    )
    comp_df["roll_carsim_rad"] = carsim_roll["roll_carsim_rad"]
    comp_df["roll_carsim_flipped_rad"] = -carsim_roll["roll_carsim_rad"]
    comp_df["error_original_rad"] = comp_df["roll_carsim_rad"] - comp_df["roll_event_interp_rad"]
    comp_df["error_flipped_rad"] = comp_df["roll_carsim_flipped_rad"] - comp_df["roll_event_interp_rad"]

    metrics_orig = compute_metrics(comp_df["roll_event_interp_rad"].to_numpy(), comp_df["roll_carsim_rad"].to_numpy())
    metrics_flip = compute_metrics(comp_df["roll_event_interp_rad"].to_numpy(), comp_df["roll_carsim_flipped_rad"].to_numpy())

    better_mode = "flipped" if metrics_flip["rmse_rad"] < metrics_orig["rmse_rad"] else "original"
    summary = "\n".join(
        [
            f"Event file: {event_xlsx}",
            f"Better mode: {better_mode}",
            f"Original RMSE(rad): {metrics_orig['rmse_rad']:.6f}",
            f"Original MAE(rad): {metrics_orig['mae_rad']:.6f}",
            f"Original MaxAbs(rad): {metrics_orig['max_abs_rad']:.6f}",
            f"Original Corr: {metrics_orig['corr']:.6f}",
            f"Flipped RMSE(rad): {metrics_flip['rmse_rad']:.6f}",
            f"Flipped MAE(rad): {metrics_flip['mae_rad']:.6f}",
            f"Flipped MaxAbs(rad): {metrics_flip['max_abs_rad']:.6f}",
            f"Flipped Corr: {metrics_flip['corr']:.6f}",
        ]
    )
    return comp_df, summary


def plot_event(event_xlsx: Path, comp_df: pd.DataFrame) -> Path:
    stem = event_xlsx.stem
    output_png = OUTPUT_DIR / f"{stem}_roll_sign_check_0to2s_rad.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(comp_df["t_s"], comp_df["roll_event_interp_rad"], label="Original roll", color="#1f77b4", linewidth=2.0)
    axes[0].plot(comp_df["t_s"], comp_df["roll_carsim_rad"], label="Carsim roll", color="#d62728", linewidth=1.3)
    axes[0].plot(comp_df["t_s"], comp_df["roll_carsim_flipped_rad"], label="Carsim roll flipped", color="#ff7f0e", linewidth=1.3)
    axes[0].set_ylabel("Roll (rad)")
    axes[0].set_title(f"Roll Sign Check: {stem}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(comp_df["t_s"], comp_df["error_original_rad"], label="Original error", color="#d62728", linewidth=1.2)
    axes[1].plot(comp_df["t_s"], comp_df["error_flipped_rad"], label="Flipped error", color="#ff7f0e", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    carsim_roll = load_carsim_roll()

    all_summaries: list[str] = []
    for event_str in args.event_xlsx:
        event_xlsx = Path(event_str)
        comp_df, summary = evaluate_event(event_xlsx, carsim_roll)
        stem = event_xlsx.stem
        csv_path = OUTPUT_DIR / f"{stem}_roll_sign_check_0to2s_rad.csv"
        txt_path = OUTPUT_DIR / f"{stem}_roll_sign_check_0to2s_rad_metrics.txt"
        comp_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        txt_path.write_text(summary, encoding="utf-8")
        png_path = plot_event(event_xlsx, comp_df)
        print(summary)
        print(f"Saved CSV: {csv_path}")
        print(f"Saved PNG: {png_path}")
        print(f"Saved TXT: {txt_path}")
        all_summaries.append(summary)

    combined_txt = OUTPUT_DIR / "roll_sign_check_summary.txt"
    combined_txt.write_text("\n\n".join(all_summaries), encoding="utf-8")
    print(f"Saved combined summary: {combined_txt}")


if __name__ == "__main__":
    main()
