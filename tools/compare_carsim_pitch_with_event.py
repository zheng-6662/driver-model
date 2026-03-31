from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVENT_XLSX = Path(
    r"F:\data_set_process\Carsim_validation_event_extract_files\E1_cwh_P04_medium_active_2025_09_26_20_06_19.xlsx"
)
MAT_PATH = Path(r"F:\data_set_process\pitch_base.mat")
OUTPUT_DIR = Path(r"F:\data_set_process\Carsim_validation_event_extract_files")

OUTPUT_CSV = OUTPUT_DIR / "E1_cwh_pitch_comparison_0to2s.csv"
OUTPUT_PNG = OUTPUT_DIR / "E1_cwh_pitch_comparison_0to2s.png"
OUTPUT_TXT = OUTPUT_DIR / "E1_cwh_pitch_comparison_0to2s_metrics.txt"


def load_event_pitch(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=15)
    pitch_col = next(col for col in df.columns if "pitch" in str(col).lower())
    out = df[["t", pitch_col]].copy()
    out.columns = ["t_event_s", "pitch_event_rad"]
    out = out[(out["t_event_s"] >= 0.0) & (out["t_event_s"] <= 2.0)].copy()
    out.sort_values("t_event_s", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def load_carsim_pitch(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        refs = f["#refs#"]
        time_s = refs["p"][()].reshape(-1)
        pitch_deg = refs["z"][()].reshape(-1)
    out = pd.DataFrame(
        {
            "t_carsim_s": time_s,
            "pitch_carsim_deg": pitch_deg,
            "pitch_carsim_rad": np.deg2rad(pitch_deg),
        }
    )
    out = out[(out["t_carsim_s"] >= 0.0) & (out["t_carsim_s"] <= 2.0)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def build_comparison(event_df: pd.DataFrame, carsim_df: pd.DataFrame) -> pd.DataFrame:
    interp_pitch_rad = np.interp(
        carsim_df["t_carsim_s"].to_numpy(),
        event_df["t_event_s"].to_numpy(),
        event_df["pitch_event_rad"].to_numpy(),
    )
    out = carsim_df.copy()
    out["pitch_original_interp_rad"] = interp_pitch_rad
    out["pitch_original_interp_deg"] = np.rad2deg(interp_pitch_rad)
    out["error_rad"] = out["pitch_carsim_rad"] - out["pitch_original_interp_rad"]
    out["error_deg"] = out["pitch_carsim_deg"] - out["pitch_original_interp_deg"]
    return out


def write_metrics(comp_df: pd.DataFrame) -> str:
    rmse_rad = float(np.sqrt(np.mean(np.square(comp_df["error_rad"]))))
    mae_rad = float(np.mean(np.abs(comp_df["error_rad"])))
    max_abs_rad = float(np.max(np.abs(comp_df["error_rad"])))
    corr = float(
        np.corrcoef(
            comp_df["pitch_carsim_rad"].to_numpy(),
            comp_df["pitch_original_interp_rad"].to_numpy(),
        )[0, 1]
    )
    text = "\n".join(
        [
            "Pitch comparison over 0-2 s",
            f"Event file: {EVENT_XLSX}",
            f"Carsim MAT: {MAT_PATH}",
            "Units used for comparison: rad",
            f"Samples compared: {len(comp_df)}",
            f"RMSE(rad): {rmse_rad:.6f}",
            f"MAE(rad): {mae_rad:.6f}",
            f"MaxAbsError(rad): {max_abs_rad:.6f}",
            f"Correlation: {corr:.6f}",
            f"RMSE(deg): {np.rad2deg(rmse_rad):.6f}",
            f"MAE(deg): {np.rad2deg(mae_rad):.6f}",
            f"MaxAbsError(deg): {np.rad2deg(max_abs_rad):.6f}",
        ]
    )
    OUTPUT_TXT.write_text(text, encoding="utf-8")
    return text


def plot_comparison(event_df: pd.DataFrame, comp_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(
        event_df["t_event_s"],
        event_df["pitch_event_rad"],
        label="Original pitch (rad)",
        linewidth=2.0,
        color="#1f77b4",
    )
    axes[0].plot(
        comp_df["t_carsim_s"],
        comp_df["pitch_carsim_rad"],
        label="Carsim pitch converted to rad",
        linewidth=1.5,
        color="#d62728",
        alpha=0.9,
    )
    axes[0].set_ylabel("Pitch (rad)")
    axes[0].set_title("Pitch Comparison, 0-2 s")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        comp_df["t_carsim_s"],
        comp_df["error_rad"],
        color="#2ca02c",
        linewidth=1.5,
        label="Carsim - Original",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event_df = load_event_pitch(EVENT_XLSX)
    carsim_df = load_carsim_pitch(MAT_PATH)
    comp_df = build_comparison(event_df, carsim_df)
    comp_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    metrics_text = write_metrics(comp_df)
    plot_comparison(event_df, comp_df)
    print(metrics_text)
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved PNG: {OUTPUT_PNG}")
    print(f"Saved TXT: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
