from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "final_code" / "model" / "diagnostics" / "future_steer_event_rollpeak_transformer_v5_8_diag_eval.py",
    run_name="__main__",
)
