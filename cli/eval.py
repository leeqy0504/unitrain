#!/usr/bin/env python3
"""Unified evaluation entry point.

Usage:
    python cli/eval.py --config configs/rfdetr.yaml
    python cli/eval.py --config configs/rfdetr.yaml --weights outputs/.../best.pth
"""

import argparse
import sys
from pathlib import Path

from unitrain import get_runner, load_config
from unitrain.data_converter import convert_coco_dataset


def main():
    parser = argparse.ArgumentParser(description="Unified Model Evaluation")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML file")
    parser.add_argument("--weights", "-w", default="", help="Override weights path (overrides config eval.weights)")
    parser.add_argument("--split", "-s", default="", help="Override eval split (val/test)")
    parser.add_argument("--output-dir", "-o", default="", help="Override eval output directory")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation (only run framework eval)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Command-line overrides
    if args.weights:
        config.eval_weights = args.weights
    if args.split:
        config.eval_split = args.split
    if args.output_dir:
        config.eval_output_dir = args.output_dir

    # Validate weights
    if not config.eval_weights:
        print("Error: No weights specified for evaluation.")
        print("  Use --weights <path> or set eval.weights in config YAML.")
        sys.exit(1)

    weights_path = Path(config.eval_weights).expanduser()
    if not weights_path.exists():
        print(f"Error: Weights file not found: {weights_path}")
        sys.exit(1)

    print(f">>> Evaluating with {config.framework} / {config.model}")
    print(f">>> Weights: {weights_path}")

    # Auto convert data for ultralytics if needed
    if config.framework in ("ultralytics", "yolo") and config.data_format == "coco":
        task = getattr(config, "task", "detect")

        if task == "obb":
            print(f">>> OBB task: using YOLO OBB data directly from {config.data_path}")
            data_yaml = Path(config.data_path).expanduser() / "data.yaml"
            if not data_yaml.exists():
                config.to_ultralytics_yaml(data_yaml)
        else:
            data_path = Path(config.data_path).expanduser()
            yolo_data_path = data_path.parent / f"{data_path.name}_yolo"
            if not yolo_data_path.exists():
                print(f">>> Converting COCO to YOLO format: {yolo_data_path}")
                class_info = convert_coco_dataset(data_path, yolo_data_path, task=task)
                if class_info:
                    config.num_classes = class_info["nc"]
                    config.class_names = class_info["names"]

            config.data_path = str(yolo_data_path)
            data_yaml = yolo_data_path / "data.yaml"
            if not data_yaml.exists():
                config.to_ultralytics_yaml(data_yaml)
            print(f">>> Using data.yaml: {data_yaml}")

    runner = get_runner(config.framework)
    cfg_dict = config.to_dict()

    # Add data_yaml path for ultralytics
    if config.framework in ("ultralytics", "yolo"):
        cfg_dict["data_yaml"] = str(Path(config.data_path) / "data.yaml")

    # Run framework-specific evaluation
    eval_result = runner.eval(cfg_dict)

    # Generate unified report
    metrics_json = eval_result.get("metrics_json", "")
    output_dir = eval_result.get("output_dir", "")

    if not args.skip_report and metrics_json and Path(metrics_json).exists():
        print("\n>>> Generating evaluation report...")
        try:
            from unitrain.eval_report import generate_report
            outputs = generate_report(metrics_json, output_dir)
            print(f"\n>>> Evaluation report complete!")
            print(f"    Reports: {output_dir}")
        except Exception as e:
            print(f"\n>>> Warning: Report generation failed: {e}")
            print(f"    Raw metrics available at: {metrics_json}")
    else:
        if metrics_json:
            print(f"\n>>> Raw metrics saved: {metrics_json}")

    print(">>> Evaluation complete!")


if __name__ == "__main__":
    main()
