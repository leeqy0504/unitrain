"""Configuration generator for unified config to framework-specific formats."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UnifiedConfig:
    """Unified configuration for all frameworks."""
    framework: str = "ultralytics"
    model: str = "yolo11n"
    task: str = "detect"

    # Data settings
    data_path: str = "data/"
    data_format: str = "coco"
    num_classes: int = 80
    class_names: list[str] = field(default_factory=list)

    # Training settings (common)
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: int | str = 0
    
    # RF-DETR specific training settings
    lr: float = 1e-4
    grad_accum_steps: int = 4
    output_dir: str = "outputs"
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    resume: str = ""

    # Predict settings
    threshold: float = 0.5
    weights: str = ""
    predict_output_dir: str = "outputs/predict"

    # Export settings
    export_format: str = "onnx"

    # Eval settings
    eval_weights: str = ""                # 待评估的权重路径
    eval_split: str = "val"               # 评估数据集划分 (val / test)
    eval_conf_threshold: float = 0.001    # 评估置信度阈值 (COCO 标准)
    eval_iou_threshold: float = 0.5       # IoU 阈值
    eval_output_dir: str = ""             # 评估输出目录 (空则自动基于权重路径)

    # Open-vocabulary settings (YOLOE etc.)
    class_prompts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dict format used by runners."""
        return {
            "framework": self.framework,
            "model": self.model,
            "task": self.task,
            "data": {"path": self.data_path, "format": self.data_format},
            "train": {
                "epochs": self.epochs,
                "imgsz": self.imgsz,
                "batch": self.batch,
                "device": self.device,
                "lr": self.lr,
                "grad_accum_steps": self.grad_accum_steps,
                "output_dir": self.output_dir,
                "early_stopping": self.early_stopping,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "resume": self.resume,
            },
            "predict": {"threshold": self.threshold, "weights": self.weights, "output_dir": self.predict_output_dir},
            "export": {"format": self.export_format},
            "eval": {
                "weights": self.eval_weights,
                "split": self.eval_split,
                "conf_threshold": self.eval_conf_threshold,
                "iou_threshold": self.eval_iou_threshold,
                "output_dir": self.eval_output_dir,
            },
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "class_prompts": self.class_prompts,
        }

    def to_ultralytics_yaml(self, output_path: Path) -> Path:
        """Generate Ultralytics data.yaml file."""
        data_yaml = {
            "path": str(Path(self.data_path).absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": self.num_classes,
            "names": self.class_names or [f"class_{i}" for i in range(self.num_classes)],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        return output_path

    def to_rfdetr_dict(self) -> dict[str, Any]:
        """Generate RF-DETR config dict (verified from local rf-detr project)."""
        return {
            "dataset_dir": self.data_path,
            "epochs": self.epochs,
            "batch_size": self.batch,
            "grad_accum_steps": self.grad_accum_steps,
            "lr": self.lr,
            "output_dir": self.output_dir,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
        }


def load_config(config_path: str | Path) -> UnifiedConfig:
    """Load config from YAML file.
    
    Supports two formats:
    1. Nested format: data.path, train.epochs, etc.
    2. Flat format: dataset_dir, epochs, batch_size, etc. (RF-DETR style)
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Support both nested and flat formats
    train_cfg = data.get("train", {})
    predict_cfg = data.get("predict", {})
    
    # Flat format support (RF-DETR style config)
    data_path = (
        data.get("dataset_dir") or 
        data.get("data", {}).get("path") or 
        "data/"
    )
    
    # Expand ~ in path if present
    if data_path.startswith("~"):
        data_path = str(Path(data_path).expanduser())
    
    return UnifiedConfig(
        framework=data.get("framework", "ultralytics"),
        model=data.get("model", "yolo11n"),
        task=data.get("task", "detect"),
        data_path=data_path,
        data_format=data.get("data", {}).get("format", "coco"),
        num_classes=data.get("data", {}).get("nc", 80),
        class_names=data.get("data", {}).get("names", []),
        # Support both nested train.epochs and flat epochs
        epochs=data.get("epochs") or train_cfg.get("epochs", 100),
        imgsz=data.get("imgsz") or train_cfg.get("imgsz", 640),
        batch=data.get("batch_size") or data.get("batch") or train_cfg.get("batch", 16),
        device=data.get("device") or train_cfg.get("device", 0),
        lr=data.get("lr") or train_cfg.get("lr", 1e-4),
        grad_accum_steps=data.get("grad_accum_steps") or train_cfg.get("grad_accum_steps", 4),
        output_dir=data.get("output_dir") or train_cfg.get("output_dir", "outputs"),
        early_stopping=data.get("early_stopping", train_cfg.get("early_stopping", False)),
        early_stopping_patience=data.get("early_stopping_patience") or train_cfg.get("early_stopping_patience", 10),
        early_stopping_min_delta=data.get("early_stopping_min_delta") or train_cfg.get("early_stopping_min_delta", 0.001),
        resume=data.get("resume") or train_cfg.get("resume", ""),
        threshold=predict_cfg.get("threshold", 0.5),
        weights=predict_cfg.get("weights", ""),
        predict_output_dir=predict_cfg.get("output_dir", "outputs/predict"),
        export_format=data.get("export", {}).get("format", "onnx"),
        # Eval settings (nested eval: block or flat)
        eval_weights=data.get("eval", {}).get("weights", "") or data.get("eval_weights", ""),
        eval_split=data.get("eval", {}).get("split", "val") or data.get("eval_split", "val"),
        eval_conf_threshold=data.get("eval", {}).get("conf_threshold", 0.001) or data.get("eval_conf_threshold", 0.001),
        eval_iou_threshold=data.get("eval", {}).get("iou_threshold", 0.5) or data.get("eval_iou_threshold", 0.5),
        eval_output_dir=data.get("eval", {}).get("output_dir", "") or data.get("eval_output_dir", ""),
        class_prompts=data.get("class_prompts", []),
    )
