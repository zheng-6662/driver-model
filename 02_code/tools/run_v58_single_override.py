from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_value(raw: str):
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def parse_override(text: str):
    if "=" not in text:
        raise ValueError(f"Invalid override (expected NAME=VALUE): {text}")
    name, raw_value = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid override name: {text}")
    return name, parse_value(raw_value.strip())


def main():
    parser = argparse.ArgumentParser(description="Run the active v5.8 training script with minimal constant overrides.")
    parser.add_argument("--script-path", required=True, help="Path to the training script.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Constant override in NAME=VALUE form.")
    args = parser.parse_args()

    script_path = Path(args.script_path).resolve()
    module = load_module(script_path, f"override_module_{abs(hash(str(script_path)))}")

    applied = {}
    for item in args.overrides:
        name, value = parse_override(item)
        if not hasattr(module, name):
            raise AttributeError(f"Module does not expose attribute: {name}")
        setattr(module, name, value)
        applied[name] = value

    print(json.dumps({"script_path": str(script_path), "applied_overrides": applied}, ensure_ascii=False, indent=2))
    if not hasattr(module, "main"):
        raise RuntimeError("Target script does not expose main()")
    module.main()


if __name__ == "__main__":
    main()
