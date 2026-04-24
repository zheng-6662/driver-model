from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(r"F:\data_set_process\data_process")
DATASET_ROOT = WORKSPACE_ROOT / "datasetprocess" / "多模态数据"
SUBJECT_ROOT = DATASET_ROOT / "被试数据集合"
MANIFEST_PATH = (
    WORKSPACE_ROOT
    / "datasetprocess"
    / "final_code"
    / "model"
    / "training"
    / "protocol_d3_response_aligned_extended_v1"
    / "sample_manifest.csv"
)
REPORT_DIR = WORKSPACE_ROOT / "reports"
OUTPUT_XLSX = REPORT_DIR / "carsim_validation_event_selection.xlsx"

WINDOW_PRE_S = 2.0
WINDOW_POST_S = 2.0

EVENT_PRIORITY = {
    "medium_active": 1,
    "strong_active": 2,
    "extreme_active": 3,
}

TARGET_SLOTS = [
    ("medium_active", "traffic_interaction"),
    ("medium_active", "geometry_route"),
    ("strong_active", "traffic_interaction"),
    ("strong_active", "obstacle_reactive"),
    ("strong_active", "geometry_route"),
    ("extreme_active", "traffic_interaction"),
]

OUTPUT_COLUMNS = [
    "event_rank",
    "event_id",
    "subject",
    "event_level",
    "mechanism_tag",
    "road_type_anchor",
    "anchor_confidence",
    "window_role",
    "rel_time_s",
    "global_time_s",
    "storage_time",
    "steering_wheel",
    "roll",
    "pitch",
    "yaw",
    "v_kmh",
    "vx",
    "vy",
    "ax",
    "ay",
    "yaw_rate",
    "roll_rate",
    "pitch_rate",
    "lateraldistance",
    "road_s_ref_m",
    "mu",
    "vehicle_file",
    "event_file",
    "episode_id",
    "episode_start_s",
    "episode_end_s",
    "primary_reference_start_s",
    "primary_reference_end_s",
    "anchor_s",
    "window_center_s",
]


@dataclass(frozen=True)
class SelectedEvent:
    event_rank: int
    event_id: str
    subject: str
    file: str
    vehicle_file: Path
    event_file: Path
    episode_id: int
    event_level: str
    mechanism_tag: str
    road_type_anchor: str
    anchor_confidence: float
    anchor_s: float
    episode_start_s: float
    episode_end_s: float
    primary_reference_start_s: float
    primary_reference_end_s: float
    anchor_speed_kmh: float
    anchor_ay: float
    anchor_yawrate: float
    anchor_roll: float
    score: float
    selection_reason: str

    @property
    def window_center_s(self) -> float:
        if self.primary_reference_start_s <= self.anchor_s <= self.primary_reference_end_s:
            return self.anchor_s
        return min(
            max(self.anchor_s, self.primary_reference_start_s),
            self.primary_reference_end_s,
        )


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    return series.map(lambda x: mapping.get(str(x).strip().lower(), x)).astype(bool)


def resolve_vehicle_file(row: pd.Series) -> Path:
    return SUBJECT_ROOT / str(row["subj"]) / "vehicle" / str(row["file"])


def resolve_event_file(row: pd.Series) -> Path:
    return SUBJECT_ROOT / str(row["subj"]) / "event" / str(row["file"]).replace(
        "_vehicle_aligned_cleaned.csv",
        "_vehicle_aligned_cleaned_events_v312.csv",
    )


def build_event_id(row: pd.Series) -> str:
    timestamp = str(row["file"]).replace("Entity_Recording_", "").replace(
        "_vehicle_aligned_cleaned.csv", ""
    )
    return f"{row['subj']}_ep{int(row['episode_id']):02d}_{timestamp}"


def score_candidate(row: pd.Series) -> float:
    confidence = float(row["anchor_confidence"])
    ay_bonus = min(abs(float(row["anchor_ay"])), 4.0) / 10.0
    yaw_bonus = min(abs(float(row["anchor_yawrate"])), 0.6) / 4.0
    roll_bonus = min(abs(float(row["anchor_roll"])), 0.08) * 4.0
    return confidence + ay_bonus + yaw_bonus + roll_bonus


def select_events(manifest: pd.DataFrame) -> list[SelectedEvent]:
    selected: list[SelectedEvent] = []
    used_keys: set[tuple[str, str, int]] = set()

    for event_level, mechanism_tag in TARGET_SLOTS:
        subset = manifest[
            (manifest["primary_reference_event_level"] == event_level)
            & (manifest["mechanism_tag"] == mechanism_tag)
        ].sort_values("selection_score", ascending=False)
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            key = (str(row["subj"]), str(row["file"]), int(row["episode_id"]))
            if key in used_keys:
                continue
            used_keys.add(key)
            selected.append(
                SelectedEvent(
                    event_rank=len(selected) + 1,
                    event_id=build_event_id(row),
                    subject=str(row["subj"]),
                    file=str(row["file"]),
                    vehicle_file=resolve_vehicle_file(row),
                    event_file=resolve_event_file(row),
                    episode_id=int(row["episode_id"]),
                    event_level=str(row["primary_reference_event_level"]),
                    mechanism_tag=str(row["mechanism_tag"]),
                    road_type_anchor=str(row["road_type_anchor"]),
                    anchor_confidence=float(row["anchor_confidence"]),
                    anchor_s=float(row["anchor_s"]),
                    episode_start_s=float(row["episode_start_s"]),
                    episode_end_s=float(row["episode_end_s"]),
                    primary_reference_start_s=float(row["primary_reference_start_s"]),
                    primary_reference_end_s=float(row["primary_reference_end_s"]),
                    anchor_speed_kmh=float(row["anchor_speed_kmh"]),
                    anchor_ay=float(row["anchor_ay"]),
                    anchor_yawrate=float(row["anchor_yawrate"]),
                    anchor_roll=float(row["anchor_roll"]),
                    score=float(row["selection_score"]),
                    selection_reason=f"{event_level} + {mechanism_tag}",
                )
            )
            break

    if len(selected) < len(TARGET_SLOTS):
        remaining = manifest.sort_values(
            ["event_priority", "selection_score"], ascending=[True, False]
        )
        for _, row in remaining.iterrows():
            key = (str(row["subj"]), str(row["file"]), int(row["episode_id"]))
            if key in used_keys:
                continue
            used_keys.add(key)
            selected.append(
                SelectedEvent(
                    event_rank=len(selected) + 1,
                    event_id=build_event_id(row),
                    subject=str(row["subj"]),
                    file=str(row["file"]),
                    vehicle_file=resolve_vehicle_file(row),
                    event_file=resolve_event_file(row),
                    episode_id=int(row["episode_id"]),
                    event_level=str(row["primary_reference_event_level"]),
                    mechanism_tag=str(row["mechanism_tag"]),
                    road_type_anchor=str(row["road_type_anchor"]),
                    anchor_confidence=float(row["anchor_confidence"]),
                    anchor_s=float(row["anchor_s"]),
                    episode_start_s=float(row["episode_start_s"]),
                    episode_end_s=float(row["episode_end_s"]),
                    primary_reference_start_s=float(row["primary_reference_start_s"]),
                    primary_reference_end_s=float(row["primary_reference_end_s"]),
                    anchor_speed_kmh=float(row["anchor_speed_kmh"]),
                    anchor_ay=float(row["anchor_ay"]),
                    anchor_yawrate=float(row["anchor_yawrate"]),
                    anchor_roll=float(row["anchor_roll"]),
                    score=float(row["selection_score"]),
                    selection_reason="fallback highest-score remaining event",
                )
            )
            if len(selected) >= len(TARGET_SLOTS):
                break

    return selected


def pick_existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def extract_event_window(event: SelectedEvent) -> pd.DataFrame:
    vehicle_df = pd.read_csv(event.vehicle_file)
    center_s = event.window_center_s
    start_s = center_s - WINDOW_PRE_S
    end_s = center_s + WINDOW_POST_S
    seg = vehicle_df[(vehicle_df["t_s"] >= start_s) & (vehicle_df["t_s"] <= end_s)].copy()
    seg["event_rank"] = event.event_rank
    seg["event_id"] = event.event_id
    seg["subject"] = event.subject
    seg["event_level"] = event.event_level
    seg["mechanism_tag"] = event.mechanism_tag
    seg["road_type_anchor"] = event.road_type_anchor
    seg["anchor_confidence"] = event.anchor_confidence
    seg["window_role"] = np.where(
        (seg["t_s"] >= event.primary_reference_start_s)
        & (seg["t_s"] <= event.primary_reference_end_s),
        "primary_reference",
        np.where(
            (seg["t_s"] >= event.episode_start_s) & (seg["t_s"] <= event.episode_end_s),
            "episode_context",
            "anchor_window",
        ),
    )
    seg["rel_time_s"] = seg["t_s"] - center_s
    seg["global_time_s"] = seg["t_s"]
    seg["storage_time"] = seg.get("StorageTime")
    seg["steering_wheel"] = seg.get("zx|SteeringWheel")
    seg["roll"] = seg.get("zx|roll")
    seg["pitch"] = seg.get("zx|pitch")
    seg["yaw"] = seg.get("zx|yaw")
    seg["v_kmh"] = seg.get("zx1|v_km/h")
    seg["vx"] = seg.get("zx|vx")
    seg["vy"] = seg.get("zx|vy")
    seg["ax"] = seg.get("zx|ax")
    seg["ay"] = seg.get("zx|ay")
    seg["yaw_rate"] = seg.get("zx|vyaw")
    seg["roll_rate"] = seg.get("zx|vroll")
    seg["pitch_rate"] = seg.get("zx|vpitch")
    seg["lateraldistance"] = seg.get("zx1|lateraldistance")
    seg["road_s_ref_m"] = seg.get("road_s_ref_m")
    seg["mu"] = seg.get("zx1|mu")
    seg["vehicle_file"] = str(event.vehicle_file)
    seg["event_file"] = str(event.event_file)
    seg["episode_id"] = event.episode_id
    seg["episode_start_s"] = event.episode_start_s
    seg["episode_end_s"] = event.episode_end_s
    seg["primary_reference_start_s"] = event.primary_reference_start_s
    seg["primary_reference_end_s"] = event.primary_reference_end_s
    seg["anchor_s"] = event.anchor_s
    seg["window_center_s"] = center_s
    return seg[OUTPUT_COLUMNS]


def build_summary(selected_events: list[SelectedEvent]) -> pd.DataFrame:
    summary_rows = []
    for event in selected_events:
        summary_rows.append(
            {
                "event_rank": event.event_rank,
                "event_id": event.event_id,
                "subject": event.subject,
                "file": event.file,
                "episode_id": event.episode_id,
                "event_level": event.event_level,
                "mechanism_tag": event.mechanism_tag,
                "road_type_anchor": event.road_type_anchor,
                "anchor_confidence": event.anchor_confidence,
                "anchor_s": event.anchor_s,
                "window_center_s": event.window_center_s,
                "window_start_s": event.window_center_s - WINDOW_PRE_S,
                "window_end_s": event.window_center_s + WINDOW_POST_S,
                "episode_start_s": event.episode_start_s,
                "episode_end_s": event.episode_end_s,
                "primary_reference_start_s": event.primary_reference_start_s,
                "primary_reference_end_s": event.primary_reference_end_s,
                "anchor_speed_kmh": event.anchor_speed_kmh,
                "anchor_ay": event.anchor_ay,
                "anchor_yawrate": event.anchor_yawrate,
                "anchor_roll": event.anchor_roll,
                "selection_score": event.score,
                "selection_reason": event.selection_reason,
                "vehicle_file": str(event.vehicle_file),
                "event_file": str(event.event_file),
            }
        )
    return pd.DataFrame(summary_rows)


def autosize_worksheet(worksheet, dataframe: pd.DataFrame) -> None:
    for idx, column in enumerate(dataframe.columns):
        content_width = dataframe[column].astype(str).map(len).max() if not dataframe.empty else 0
        width = min(max(len(column), content_width) + 2, 42)
        worksheet.set_column(idx, idx, width)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["has_full_history_3s"] = normalize_bool(manifest["has_full_history_3s"])
    manifest["full_future_2s"] = normalize_bool(manifest["full_future_2s"])
    manifest["usable_sample"] = normalize_bool(manifest["usable_sample"])
    manifest = manifest[
        manifest["primary_reference_event_level"].isin(EVENT_PRIORITY)
        & manifest["has_full_history_3s"]
        & manifest["full_future_2s"]
        & manifest["usable_sample"]
        & manifest["anchor_confidence"].notna()
    ].copy()
    manifest["event_priority"] = manifest["primary_reference_event_level"].map(EVENT_PRIORITY)
    manifest["selection_score"] = manifest.apply(score_candidate, axis=1)

    selected_events = select_events(manifest)
    summary_df = build_summary(selected_events)
    sample_df = pd.concat(
        [extract_event_window(event) for event in selected_events],
        ignore_index=True,
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="event_summary", index=False)
        sample_df.to_excel(writer, sheet_name="event_samples", index=False)

        workbook = writer.book
        float_fmt = workbook.add_format({"num_format": "0.000"})
        center_fmt = workbook.add_format({"align": "center"})
        header_fmt = workbook.add_format(
            {"bold": True, "bg_color": "#D9EAF7", "border": 1, "text_wrap": True}
        )

        for sheet_name, dataframe in {
            "event_summary": summary_df,
            "event_samples": sample_df,
        }.items():
            worksheet = writer.sheets[sheet_name]
            autosize_worksheet(worksheet, dataframe)
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, max(len(dataframe), 1), len(dataframe.columns) - 1)
            worksheet.set_row(0, 24, header_fmt)
            if "event_rank" in dataframe.columns:
                rank_col = dataframe.columns.get_loc("event_rank")
                worksheet.set_column(rank_col, rank_col, 10, center_fmt)
            numeric_cols = pick_existing_columns(
                dataframe,
                [
                    "anchor_confidence",
                    "anchor_s",
                    "window_start_s",
                    "window_end_s",
                    "episode_start_s",
                    "episode_end_s",
                    "primary_reference_start_s",
                    "primary_reference_end_s",
                    "anchor_speed_kmh",
                    "anchor_ay",
                    "anchor_yawrate",
                    "anchor_roll",
                    "selection_score",
                    "rel_time_s",
                    "global_time_s",
                    "steering_wheel",
                    "roll",
                    "pitch",
                    "yaw",
                    "v_kmh",
                    "vx",
                    "vy",
                    "ax",
                    "ay",
                    "yaw_rate",
                    "roll_rate",
                    "pitch_rate",
                    "lateraldistance",
                    "road_s_ref_m",
                    "mu",
                ],
            )
            for column in numeric_cols:
                col_idx = dataframe.columns.get_loc(column)
                worksheet.set_column(col_idx, col_idx, None, float_fmt)

    print(f"Exported validation workbook: {OUTPUT_XLSX}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
