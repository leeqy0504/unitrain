#!/usr/bin/env python3
"""Ultralytics YOLO training script — executed in .venv-yolo."""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Allow sibling-module import inside _scripts/
sys.path.insert(0, str(Path(__file__).parent))
from weight_utils import resolve_weight_path  # noqa: E402


def get_timestamped_output_dir(base_dir: str = "outputs", prefix: str = "output") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_dir}/{prefix}_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config_json)

    model_name = cfg.get("model", "yolo11n")
    task = cfg.get("task", "detect")
    data_yaml = cfg.get("data_yaml", "data.yaml")
    epochs = cfg.get("epochs", 100)
    imgsz = cfg.get("imgsz", 640)
    batch = cfg.get("batch", 16)
    device = cfg.get("device", 0)
    output_dir = cfg.get("output_dir", "outputs")
    config_file = cfg.get("config_file", "")
    class_prompts = cfg.get("class_prompts", [])

    from ultralytics import YOLO

    output_dir = get_timestamped_output_dir(output_dir, "yolo")
    # 使用绝对路径，避免 Ultralytics 在其目录下创建 runs/
    output_dir = str(Path(output_dir).absolute())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Training output: {output_dir}")

    # Copy config file BEFORE training - at the parent level (safe from Ultralytics cleanup)
    if config_file and os.path.exists(config_file):
        config_dest = Path(output_dir) / Path(config_file).name
        shutil.copy2(config_file, config_dest)
        print(f"Config file copied to: {config_dest}")

    # Resolve weights to weights/ directory
    model_path = resolve_weight_path(f"{model_name}.pt")
    # Pass task explicitly for OBB / YOLOE models that require it
    model = YOLO(model_path, task=task) if task != "detect" else YOLO(model_path)

    # Open-vocabulary models (e.g. yoloe-26s-seg): set class prompts before training
    if class_prompts:
        print(f"Setting class prompts: {class_prompts}")
        model.set_classes(class_prompts)

    # 使用二级目录：project=output_dir, name="train"
    # Ultralytics 管理 train/ 子目录，配置文件在 output_dir 安全
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=output_dir,      # 时间戳目录作为 project
        name="train",            # 固定子目录名，让 Ultralytics 管理
        exist_ok=True,
    )

    # Emit structured markers for auto-eval discovery
    train_dir = Path(output_dir) / "train"
    best_weights = train_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = train_dir / "weights" / "last.pt"
    print(f"UNITRAIN_TRAIN_OUTPUT_DIR={output_dir}")
    if best_weights.exists():
        print(f"UNITRAIN_BEST_WEIGHTS={best_weights}")


if __name__ == "__main__":
    main()
