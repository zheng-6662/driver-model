from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "multimodal" / "dataset" / "build_event_dataset_final.py",
    run_name="__main__",
)
