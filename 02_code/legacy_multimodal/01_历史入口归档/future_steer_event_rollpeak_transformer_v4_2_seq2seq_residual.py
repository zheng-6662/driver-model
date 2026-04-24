from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "multimodal" / "training" / "future_steer_event_rollpeak_transformer_v4_2_seq2seq_residual.py",
    run_name="__main__",
)
