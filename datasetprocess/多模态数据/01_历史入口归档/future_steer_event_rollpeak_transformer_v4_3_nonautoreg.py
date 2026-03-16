from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "multimodal" / "training" / "future_steer_event_rollpeak_transformer_v4_3_nonautoreg.py",
    run_name="__main__",
)
