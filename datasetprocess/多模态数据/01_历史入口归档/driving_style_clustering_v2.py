from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "multimodal" / "analysis" / "driving_style_clustering_v2.py",
    run_name="__main__",
)
