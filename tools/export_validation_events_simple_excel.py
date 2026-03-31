from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
OUTPUT_XLSX = Path(r"F:\data_set_process\Carsim_validation_event_extract.xlsx")
OUTPUT_DIR_SINGLE = Path(r"F:\data_set_process\Carsim_validation_event_extract_files")

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


@dataclass(frozen=True)
class SelectedEvent:
    event_rank: int
    subject: str
    file: str
    episode_id: int
    event_level: str
    mechanism_tag: str
    anchor_confidence: float
    anchor_s: float
    episode_start_s: float
    episode_end_s: float
    primary_reference_start_s: float
    primary_reference_end_s: float

    @property
    def vehicle_file(self) -> Path:
        return SUBJECT_ROOT / self.subject / "vehicle" / self.file

    @property
    def event_file(self) -> Path:
        return SUBJECT_ROOT / self.subject / "event" / self.file.replace(
            "_vehicle_aligned_cleaned.csv",
            "_vehicle_aligned_cleaned_events_v312.csv",
        )

    @property
    def event_id(self) -> str:
        timestamp = self.file.replace("Entity_Recording_", "").replace(
            "_vehicle_aligned_cleaned.csv", ""
        )
        return f"E{self.event_rank}_{self.subject}_P{self.episode_id:02d}_{self.event_level}_{timestamp}"

    @property
    def window_center_s(self) -> float:
        if self.primary_reference_start_s <= self.anchor_s <= self.primary_reference_end_s:
            return self.anchor_s
        return min(
            max(self.anchor_s, self.primary_reference_start_s),
            self.primary_reference_end_s,
        )


def normalize_bool(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return series.map(lambda x: mapping.get(str(x).strip().lower(), x)).astype(bool)


def score_candidate(row: pd.Series) -> float:
    confidence = float(row["anchor_confidence"])
    ay_bonus = min(abs(float(row["anchor_ay"])), 4.0) / 10.0
    yaw_bonus = min(abs(float(row["anchor_yawrate"])), 0.6) / 4.0
    roll_bonus = min(abs(float(row["anchor_roll"])), 0.08) * 4.0
    return confidence + ay_bonus + yaw_bonus + roll_bonus


def select_events(manifest: pd.DataFrame) -> list[SelectedEvent]:
    selected: list[SelectedEvent] = []
    used_keys: set[tuple[str, str, int]] = set()

    def append_from_subset(subset: pd.DataFrame) -> bool:
        for _, row in subset.iterrows():
            key = (str(row["subj"]), str(row["file"]), int(row["episode_id"]))
            if key in used_keys:
                continue
            used_keys.add(key)
            selected.append(
                SelectedEvent(
                    event_rank=len(selected) + 1,
                    subject=str(row["subj"]),
                    file=str(row["file"]),
                    episode_id=int(row["episode_id"]),
                    event_level=str(row["primary_reference_event_level"]),
                    mechanism_tag=str(row["mechanism_tag"]),
                    anchor_confidence=float(row["anchor_confidence"]),
                    anchor_s=float(row["anchor_s"]),
                    episode_start_s=float(row["episode_start_s"]),
                    episode_end_s=float(row["episode_end_s"]),
                    primary_reference_start_s=float(row["primary_reference_start_s"]),
                    primary_reference_end_s=float(row["primary_reference_end_s"]),
                )
            )
            return True
        return False

    for event_level, mechanism_tag in TARGET_SLOTS:
        subset = manifest[
            (manifest["primary_reference_event_level"] == event_level)
            & (manifest["mechanism_tag"] == mechanism_tag)
        ].sort_values("selection_score", ascending=False)
        append_from_subset(subset)

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
                    subject=str(row["subj"]),
                    file=str(row["file"]),
                    episode_id=int(row["episode_id"]),
                    event_level=str(row["primary_reference_event_level"]),
                    mechanism_tag=str(row["mechanism_tag"]),
                    anchor_confidence=float(row["anchor_confidence"]),
                    anchor_s=float(row["anchor_s"]),
                    episode_start_s=float(row["episode_start_s"]),
                    episode_end_s=float(row["episode_end_s"]),
                    primary_reference_start_s=float(row["primary_reference_start_s"]),
                    primary_reference_end_s=float(row["primary_reference_end_s"]),
                )
            )
            if len(selected) >= len(TARGET_SLOTS):
                break

    return selected


def build_meta_rows(event: SelectedEvent) -> pd.DataFrame:
    meta = [
        ("事件编号", event.event_id),
        ("被试", event.subject),
        ("事件等级", event.event_level),
        ("事件机制", event.mechanism_tag),
        ("锚点置信度", event.anchor_confidence),
        ("窗口中心时间(s)", event.window_center_s),
        ("窗口开始时间(s)", event.window_center_s - WINDOW_PRE_S),
        ("窗口结束时间(s)", event.window_center_s + WINDOW_POST_S),
        ("主事件开始时间(s)", event.primary_reference_start_s),
        ("主事件结束时间(s)", event.primary_reference_end_s),
        ("原始车辆文件", str(event.vehicle_file)),
        ("原始事件文件", str(event.event_file)),
    ]
    return pd.DataFrame(meta, columns=["字段", "值"])


def build_timeseries_rows(event: SelectedEvent) -> pd.DataFrame:
    df = pd.read_csv(event.vehicle_file)
    center_s = event.window_center_s
    start_s = center_s - WINDOW_PRE_S
    end_s = center_s + WINDOW_POST_S
    seg = df[(df["t_s"] >= start_s) & (df["t_s"] <= end_s)].copy()

    out = pd.DataFrame(
        {
            "t": np.round(seg["t_s"] - center_s, 6),
            "time_global_s": seg["t_s"],
            "方向盘角度": seg.get("zx|SteeringWheel"),
            "横滚角_roll": seg.get("zx|roll"),
            "俯仰角_pitch": seg.get("zx|pitch"),
            "横摆角_yaw": seg.get("zx|yaw"),
            "车速_km_h": seg.get("zx1|v_km/h"),
            "纵向速度_vx": seg.get("zx|vx"),
            "横向速度_vy": seg.get("zx|vy"),
            "纵向加速度_ax": seg.get("zx|ax"),
            "横向加速度_ay": seg.get("zx|ay"),
            "横摆角速度_vyaw": seg.get("zx|vyaw"),
            "横滚角速度_vroll": seg.get("zx|vroll"),
            "原始车辆文件": str(event.vehicle_file),
            "原始事件文件": str(event.event_file),
        }
    )
    return out


def autosize_sheet(worksheet, dataframe: pd.DataFrame) -> None:
    for idx, column in enumerate(dataframe.columns):
        content_width = dataframe[column].astype(str).map(len).max() if not dataframe.empty else 0
        width = min(max(len(column), content_width) + 2, 44)
        worksheet.set_column(idx, idx, width)


def export_single_event_workbook(event: SelectedEvent) -> Path:
    OUTPUT_DIR_SINGLE.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR_SINGLE / f"{event.event_id}.xlsx"
    meta_df = build_meta_rows(event)
    ts_df = build_timeseries_rows(event)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        sheet_name = "Sheet1"
        ts_df.to_excel(writer, sheet_name=sheet_name, startrow=15, index=False)
        meta_df.to_excel(writer, sheet_name=sheet_name, startrow=1, index=False)

        workbook = writer.book
        ws = writer.sheets[sheet_name]
        title_fmt = workbook.add_format({"bold": True, "font_size": 12})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        float_fmt = workbook.add_format({"num_format": "0.000000"})

        ws.write(0, 0, f"{event.event_id} 对标提取", title_fmt)
        ws.freeze_panes(16, 0)

        for col_idx, _ in enumerate(meta_df.columns):
            ws.write(1, col_idx, meta_df.columns[col_idx], header_fmt)

        for col_idx, _ in enumerate(ts_df.columns):
            ws.write(15, col_idx, ts_df.columns[col_idx], header_fmt)
            ws.set_column(col_idx, col_idx, 18, float_fmt if col_idx <= 12 else 42)

        autosize_sheet(ws, ts_df)

    return output_path


def main() -> None:
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

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        summary_rows = []
        for event in selected_events:
            summary_rows.append(
                {
                    "事件编号": event.event_id,
                    "被试": event.subject,
                    "事件等级": event.event_level,
                    "事件机制": event.mechanism_tag,
                    "锚点置信度": event.anchor_confidence,
                    "窗口中心时间(s)": event.window_center_s,
                    "原始车辆文件": str(event.vehicle_file),
                    "原始事件文件": str(event.event_file),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="事件目录", index=False)
        autosize_sheet(writer.sheets["事件目录"], summary_df)
        writer.sheets["事件目录"].freeze_panes(1, 0)

        workbook = writer.book
        title_fmt = workbook.add_format({"bold": True, "font_size": 12})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        float_fmt = workbook.add_format({"num_format": "0.000000"})

        for event in selected_events:
            sheet_name = f"E{event.event_rank}_{event.subject}_{event.event_level}"[:31]
            meta_df = build_meta_rows(event)
            ts_df = build_timeseries_rows(event)
            ts_df.to_excel(writer, sheet_name=sheet_name, startrow=15, index=False)
            meta_df.to_excel(writer, sheet_name=sheet_name, startrow=1, index=False)

            ws = writer.sheets[sheet_name]
            ws.write(0, 0, f"{event.event_id} 对标提取", title_fmt)
            ws.freeze_panes(16, 0)
            autosize_sheet(ws, ts_df)
            for col_idx, _ in enumerate(meta_df.columns):
                ws.write(1, col_idx, meta_df.columns[col_idx], header_fmt)
            for col_idx, _ in enumerate(ts_df.columns):
                ws.write(15, col_idx, ts_df.columns[col_idx], header_fmt)
                ws.set_column(col_idx, col_idx, 18, float_fmt if col_idx <= 12 else None)

    print(f"Exported simple workbook: {OUTPUT_XLSX}")
    for event in selected_events:
        print(event.event_id, event.vehicle_file)

    print(f"Exporting single-event workbooks to: {OUTPUT_DIR_SINGLE}")
    for event in selected_events:
        output_path = export_single_event_workbook(event)
        print(f"SINGLE {output_path}")


if __name__ == "__main__":
    main()
