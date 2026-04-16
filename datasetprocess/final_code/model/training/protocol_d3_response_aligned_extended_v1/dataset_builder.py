from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROTOCOL_DIR = Path(__file__).resolve().parent
PROTOCOL_CONFIG_PATH = PROTOCOL_DIR / "protocol_config.json"
SAMPLE_MANIFEST_PATH = PROTOCOL_DIR / "sample_manifest.csv"


def load_protocol_config(protocol_config_path: str | Path | None = None) -> dict:
    path = Path(protocol_config_path) if protocol_config_path else PROTOCOL_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_protocol_manifest(protocol_config_path: str | Path | None = None) -> pd.DataFrame:
    _ = load_protocol_config(protocol_config_path)
    df = pd.read_csv(SAMPLE_MANIFEST_PATH)
    return df[df["d3_included"].fillna(False)].reset_index(drop=True)

