#!/usr/bin/env python3
"""Ultralytics YOLO prediction script — executed in .venv-yolo."""

import argparse
import json
import sys
from pathlib import Path

# Allow sibling-module import inside _scripts/
sys.path.insert(0, str(Path(__file__).parent))
from weight_utils import resolve_weight_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config_json)

    model_name = cfg.get("model", "yolo11n")
    task = cfg.get("task", "detect")
    source = cfg["source"]
    imgsz = cfg.get("imgsz", 640)
    class_prompts = cfg.get("class_prompts", [])

    from ultralytics import YOLO

    # Resolve weights to weights/ directory
    model_path = resolve_weight_path(f"{model_name}.pt")
    # Pass task explicitly for OBB / YOLOE models that require it
    model = YOLO(model_path, task=task) if task != "detect" else YOLO(model_path)

    # Open-vocabulary models (e.g. yoloe-26s-seg): set class prompts before inference
    if class_prompts:
        print(f"Setting class prompts: {class_prompts}")
        model.set_classes(class_prompts)

    results = model(source, imgsz=imgsz)
    for r in results:
        if hasattr(r, 'obb') and r.obb is not None:
            print(r.obb.data.tolist())
        elif r.boxes:
            print(r.boxes.data.tolist())
        else:
            print([])


if __name__ == "__main__":
    main()
