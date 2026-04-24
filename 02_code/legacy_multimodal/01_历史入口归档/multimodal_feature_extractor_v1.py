from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "multimodal" / "dataset" / "multimodal_feature_extractor_v1.py",
    run_name="__main__",
)
