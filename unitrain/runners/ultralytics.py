"""Runner for Ultralytics YOLO models."""

import sys
from pathlib import Path
from typing import Any

from .base import BaseRunner, PROJECT_ROOT

# Known Ultralytics model names for validation hints
KNOWN_MODELS: set[str] = {
    # YOLO11
    *(f"yolo11{s}" for s in ("n", "s", "m", "l", "x")),
    *(f"yolo11{s}-seg" for s in ("n", "s", "m", "l", "x")),
    *(f"yolo11{s}-obb" for s in ("n", "s", "m", "l", "x")),
    # YOLO26
    *(f"yolo26{s}" for s in ("n", "s", "m", "l", "x")),
    *(f"yolo26{s}-seg" for s in ("n", "s", "m", "l", "x")),
    *(f"yolo26{s}-obb" for s in ("n", "s", "m", "l", "x")),
    # YOLOE (open-vocabulary / promptable)
    *(f"yoloe-11{s}-seg" for s in ("s", "m", "l")),
    *(f"yoloe-v8{s}-seg" for s in ("s", "m", "l")),
    *(f"yoloe-26{s}-seg" for s in ("n", "s", "m", "l", "x")),
    *(f"yoloe-11{s}-seg-pf" for s in ("s", "m", "l")),
    *(f"yoloe-v8{s}-seg-pf" for s in ("s", "m", "l")),
    *(f"yoloe-26{s}-seg-pf" for s in ("n", "s", "m", "l", "x")),
}


def _warn_unknown_model(model: str) -> None:
    """Print a warning if the model name is not in the known list."""
    if model not in KNOWN_MODELS:
        print(
            f"[YOLO] Warning: unknown model '{model}'. "
            f"Will attempt to load via Ultralytics. "
            f"Known models: yolo11*/yolo26* (n/s/m/l/x, -seg/-obb), yoloe-*",
            file=sys.stderr,
        )


class UltralyticsRunner(BaseRunner):
    """Runner for Ultralytics YOLO models."""

    def __init__(self):
        super().__init__(PROJECT_ROOT / ".venv-yolo")

    def train(self, config: dict[str, Any]) -> dict[str, Any] | None:
        train_cfg = config.get("train", {})
        model = config.get("model", "yolo11n")
        _warn_unknown_model(model)

        script_config = {
            "model": model,
            "task": config.get("task", "detect"),
            "data_yaml": config.get("data_yaml", "data.yaml"),
            "epochs": train_cfg.get("epochs", 100),
            "imgsz": train_cfg.get("imgsz", 640),
            "batch": train_cfg.get("batch", 16),
            "device": train_cfg.get("device", 0),
            "output_dir": train_cfg.get("output_dir", "outputs"),
            "config_file": config.get("config_file", ""),
            "class_prompts": config.get("class_prompts", []),
        }

        result = self._run_script_file("yolo_train.py", script_config)
        if result.returncode != 0:
            print(f"[YOLO] Training failed", file=sys.stderr)
            raise RuntimeError("Training failed")
        return self._parse_train_markers(result.stdout)

    def predict(self, config: dict[str, Any], source: str) -> Any:
        model = config.get("model", "yolo11n")
        _warn_unknown_model(model)

        script_config = {
            "model": model,
            "task": config.get("task", "detect"),
            "source": source,
            "imgsz": config.get("train", {}).get("imgsz", 640),
            "class_prompts": config.get("class_prompts", []),
        }

        result = self._run_script_file("yolo_predict.py", script_config)
        if result.returncode != 0:
            print(f"[YOLO] Prediction failed", file=sys.stderr)
            raise RuntimeError("Prediction failed")
        return result.stdout

    def export(self, config: dict[str, Any], format: str = "onnx") -> Path:
        model = config.get("model", "yolo11n")
        _warn_unknown_model(model)

        script_config = {
            "model": model,
            "task": config.get("task", "detect"),
            "format": format,
            "imgsz": config.get("train", {}).get("imgsz", 640),
        }

        result = self._run_script_file("yolo_export.py", script_config, stream_output=False)
        if result.returncode != 0:
            print(f"[YOLO] Export failed", file=sys.stderr)
            raise RuntimeError("Export failed")
        return Path(result.stdout.strip())

    def eval(self, config: dict[str, Any]) -> dict[str, Any]:
        eval_cfg = config.get("eval", {})
        train_cfg = config.get("train", {})

        weights = eval_cfg.get("weights", "")
        if not weights:
            print("[YOLO] Error: eval.weights must be specified", file=sys.stderr)
            raise ValueError("eval.weights is required")

        script_config = {
            "model": config.get("model", "yolo11n"),
            "task": config.get("task", "detect"),
            "weights": weights,
            "data_yaml": config.get("data_yaml", "data.yaml"),
            "imgsz": train_cfg.get("imgsz", 640),
            "batch": train_cfg.get("batch", 16),
            "device": train_cfg.get("device", 0),
            "output_dir": eval_cfg.get("output_dir", ""),
            "conf_threshold": eval_cfg.get("conf_threshold", 0.001),
            "iou_threshold": eval_cfg.get("iou_threshold", 0.5),
        }

        result = self._run_script_file("yolo_eval.py", script_config)
        if result.returncode != 0:
            print(f"[YOLO] Evaluation failed", file=sys.stderr)
            raise RuntimeError("Evaluation failed")
        print("[YOLO] Evaluation completed successfully")

        output_dir = eval_cfg.get("output_dir", "")
        if not output_dir:
            output_dir = str(Path(weights).parent.parent / "eval_results")
        return {"output_dir": output_dir, "metrics_json": str(Path(output_dir) / "eval_metrics.json")}
