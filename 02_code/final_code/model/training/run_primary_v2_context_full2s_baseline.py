from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from d3_training_common import run_d3_experiment
from future_steer_speed_subjectsplit_masked import _make_sample


THIS_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = THIS_DIR / "protocol_primary_control_v2_context_full2s"
DEFAULT_MANIFEST = PROTOCOL_DIR / "sample_manifest.csv"
DEFAULT_PROTOCOL_CONFIG = PROTOCOL_DIR / "protocol_config.json"


def build_sample_bundle(manifest_path: str | Path):
    meta_df = pd.read_csv(manifest_path)
    X_list = []
    y_list = []
    curve_list = []
    ctx_list = []
    mask_list = []

    for _, row in meta_df.iterrows():
        x_win, y_seq, curve_future, ctx, future_mask = _make_sample(row)
        X_list.append(x_win)
        y_list.append(y_seq)
        curve_list.append(curve_future)
        ctx_list.append(ctx)
        mask_list.append(future_mask)

    return (
        np.stack(X_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        np.stack(curve_list).astype(np.float32),
        np.stack(ctx_list).astype(np.float32),
        np.stack(mask_list).astype(np.float32),
        meta_df.reset_index(drop=True).copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL_CONFIG))
    parser.add_argument("--run-prefix", default="EXP2_PRIMARY_V2_CONTEXT_FULL2S_BASELINE")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weighting-scheme", default="baseline")
    parser.add_argument("--supervised-horizon-len", type=int, default=300)
    args = parser.parse_args()

    sample_bundle = build_sample_bundle(args.manifest)
    result = run_d3_experiment(
        run_prefix=args.run_prefix,
        conditioned=False,
        protocol_config_path=args.protocol_config,
        sample_bundle=sample_bundle,
        experiment_config={
            "epochs": int(args.epochs),
            "min_epochs": int(args.min_epochs),
            "early_stop_patience": int(args.patience),
            "weighting_scheme": str(args.weighting_scheme),
            "supervised_horizon_len": int(args.supervised_horizon_len),
        },
    )
    print(result["run_root"])
    print(result["metric_summary_path"])


if __name__ == "__main__":
    main()
