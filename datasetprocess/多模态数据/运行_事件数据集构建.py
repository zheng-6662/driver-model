from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "final_code" / "dataset" / "build_event_dataset_v2_pad_mask_multipeak.py",
    run_name="__main__",
)
