#!/usr/bin/env python3
"""RF-DETR export script — executed in .venv-rfdetr."""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow sibling-module import inside _scripts/
sys.path.insert(0, str(Path(__file__).parent))
from weight_utils import resolve_rfdetr_weight  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config_json)

    model_cls_name = cfg["model_cls"]
    export_format = cfg.get("format", "onnx")
    output_path = cfg.get("output_path", "rfdetr.onnx")
    weights = cfg.get("weights", "")
    default_weights = cfg.get("default_pretrain_weights", "")
    device = cfg.get("device", 0)

    # Set device via environment variable before importing torch/rfdetr
    if isinstance(device, str) and device.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU device")
    elif isinstance(device, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        print(f"Using GPU device: {device}")
    elif isinstance(device, str):
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f"Using GPU device(s): {device}")

    # Dynamic import
    import rfdetr
    model_cls = getattr(rfdetr, model_cls_name)

    # Resolve weights into weights/ directory
    model_kwargs = {}
    if weights:
        model_kwargs["pretrain_weights"] = resolve_rfdetr_weight(weights)
    elif default_weights:
        model_kwargs["pretrain_weights"] = resolve_rfdetr_weight(default_weights)

    try:
        model = model_cls(**model_kwargs)
    except RuntimeError as e:
        if "PytorchStreamReader" in str(e) or "failed finding central directory" in str(e):
            weight_path = model_kwargs.get("pretrain_weights", "")
            if weight_path and os.path.exists(weight_path):
                os.remove(weight_path)
            print(
                f"\n{'='*60}\n"
                f"❌ 预训练权重文件损坏，无法加载\n"
                f"   文件: {weight_path}\n"
                f"   已自动删除损坏文件，请重新运行命令\n"
                f"{'='*60}\n"
            )
        raise

    model.export(format=export_format, output_path=output_path)
    print(output_path)


if __name__ == "__main__":
    main()
