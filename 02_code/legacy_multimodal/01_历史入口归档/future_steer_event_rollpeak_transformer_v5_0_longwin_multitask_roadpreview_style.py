from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "multimodal" / "training" / "future_steer_event_rollpeak_transformer_v5_0_longwin_multitask_roadpreview_style.py",
    run_name="__main__",
)
