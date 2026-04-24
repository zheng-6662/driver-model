from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "multimodal" / "analysis" / "Related_data_physiol.py",
    run_name="__main__",
)
