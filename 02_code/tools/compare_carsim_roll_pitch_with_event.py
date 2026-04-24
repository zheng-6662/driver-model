from __future__ import annotations

from pathlib import Path
import argparse

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_EVENT_XLSX = Path(
    r"F:\data_set_process\Carsim_validation_event_extract_files\E1_cwh_P04_medium_active_2025_09_26_20_06_19.xlsx"
)
ROLL_MAT = Path(r"F:\data_set_process\roll_base.mat")
PITCH_MAT = Path(r"F:\data_set_process\pitch_base.mat")
OUTPUT_DIR = Path(r"F:\data_set_process\Carsim_validation_event_extract_files")


def load_event_pose(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=15)
    roll_col = next(col for col in df.columns if "roll" in str(col).lower())
    pitch_col = next(col for col in df.columns if "pitch" in str(col).lower())
    out = df[["t", roll_col, pitch_col]].copy()
    out.columns = ["t_event_s", "roll_event_rad", "pitch_event_rad"]
    out = out[(out["t_event_s"] >= 0.0) & (out["t_event_s"] <= 2.0)].copy()
    out.sort_values("t_event_s", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def load_carsim_signal(path: Path, signal_name: str) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        refs = f["#refs#"]
        time_s = refs["p"][()].reshape(-1)
        signal_deg = refs["z"][()].reshape(-1)
    out = pd.DataFrame(
        {
            "t_carsim_s": time_s,
            f"{signal_name}_carsim_deg": signal_deg,
            f"{signal_name}_carsim_rad": np.deg2rad(signal_deg),
        }
    )
    out = out[(out["t_carsim_s"] >= 0.0) & (out["t_carsim_s"] <= 2.0)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def build_comparison(event_df: pd.DataFrame, roll_df: pd.DataFrame, pitch_df: pd.DataFrame) -> pd.DataFrame:
    comp_df = pd.DataFrame({"t_s": roll_df["t_carsim_s"]})
    comp_df["roll_carsim_rad"] = roll_df["roll_carsim_rad"]
    comp_df["pitch_carsim_rad"] = pitch_df["pitch_carsim_rad"]
    comp_df["roll_event_interp_rad"] = np.interp(
        comp_df["t_s"],
        event_df["t_event_s"],
        event_df["roll_event_rad"],
    )
    comp_df["pitch_event_interp_rad"] = np.interp(
        comp_df["t_s"],
        event_df["t_event_s"],
        event_df["pitch_event_rad"],
    )
    comp_df["roll_error_rad"] = comp_df["roll_carsim_rad"] - comp_df["roll_event_interp_rad"]
    comp_df["pitch_error_rad"] = comp_df["pitch_carsim_rad"] - comp_df["pitch_event_interp_rad"]
    return comp_df


def metric_lines(comp_df: pd.DataFrame, signal_name: str) -> list[str]:
    error = comp_df[f"{signal_name}_error_rad"].to_numpy()
    carsim = comp_df[f"{signal_name}_carsim_rad"].to_numpy()
    event = comp_df[f"{signal_name}_event_interp_rad"].to_numpy()
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    max_abs = float(np.max(np.abs(error)))
    corr = float(np.corrcoef(carsim, event)[0, 1])
    return [
        f"{signal_name.upper()} RMSE(rad): {rmse:.6f}",
        f"{signal_name.upper()} MAE(rad): {mae:.6f}",
        f"{signal_name.upper()} MaxAbsError(rad): {max_abs:.6f}",
        f"{signal_name.upper()} Correlation: {corr:.6f}",
    ]


def write_metrics(comp_df: pd.DataFrame, event_xlsx: Path, output_txt: Path) -> str:
    lines = [
        "Roll and pitch comparison over 0-2 s",
        f"Event file: {event_xlsx}",
        f"Roll MAT: {ROLL_MAT}",
        f"Pitch MAT: {PITCH_MAT}",
        "Units used for comparison: rad",
        f"Samples compared: {len(comp_df)}",
    ]
    lines.extend(metric_lines(comp_df, "roll"))
    lines.extend(metric_lines(comp_df, "pitch"))
    text = "\n".join(lines)
    output_txt.write_text(text, encoding="utf-8")
    return text


def plot_comparison(event_df: pd.DataFrame, comp_df: pd.DataFrame, output_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")

    axes[0, 0].plot(event_df["t_event_s"], event_df["roll_event_rad"], label="Original roll", color="#1f77b4", linewidth=2.0)
    axes[0, 0].plot(comp_df["t_s"], comp_df["roll_carsim_rad"], label="Carsim roll", color="#d62728", linewidth=1.4)
    axes[0, 0].set_title("Roll Comparison (rad)")
    axes[0, 0].set_ylabel("Roll (rad)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[1, 0].plot(comp_df["t_s"], comp_df["roll_error_rad"], color="#2ca02c", linewidth=1.4)
    axes[1, 0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Roll error (rad)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(event_df["t_event_s"], event_df["pitch_event_rad"], label="Original pitch", color="#1f77b4", linewidth=2.0)
    axes[0, 1].plot(comp_df["t_s"], comp_df["pitch_carsim_rad"], label="Carsim pitch", color="#d62728", linewidth=1.4)
    axes[0, 1].set_title("Pitch Comparison (rad)")
    axes[0, 1].set_ylabel("Pitch (rad)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 1].plot(comp_df["t_s"], comp_df["pitch_error_rad"], color="#2ca02c", linewidth=1.4)
    axes[1, 1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Pitch error (rad)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def build_output_paths(event_xlsx: Path) -> tuple[Path, Path, Path]:
    stem = event_xlsx.stem
    csv_path = OUTPUT_DIR / f"{stem}_roll_pitch_comparison_0to2s_rad.csv"
    png_path = OUTPUT_DIR / f"{stem}_roll_pitch_comparison_0to2s_rad.png"
    txt_path = OUTPUT_DIR / f"{stem}_roll_pitch_comparison_0to2s_rad_metrics.txt"
    return csv_path, png_path, txt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-xlsx", type=Path, default=DEFAULT_EVENT_XLSX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_xlsx = args.event_xlsx
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv, output_png, output_txt = build_output_paths(event_xlsx)
    event_df = load_event_pose(event_xlsx)
    roll_df = load_carsim_signal(ROLL_MAT, "roll")
    pitch_df = load_carsim_signal(PITCH_MAT, "pitch")
    comp_df = build_comparison(event_df, roll_df, pitch_df)
    comp_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    metrics_text = write_metrics(comp_df, event_xlsx, output_txt)
    plot_comparison(event_df, comp_df, output_png)
    print(metrics_text)
    print(f"Saved CSV: {output_csv}")
    print(f"Saved PNG: {output_png}")
    print(f"Saved TXT: {output_txt}")


if __name__ == "__main__":
    main()
