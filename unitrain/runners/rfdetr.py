"""Runner for RF-DETR models."""

import sys
from pathlib import Path
from typing import Any

from .base import BaseRunner, PROJECT_ROOT, _parse_device_ids


class RFDETRRunner(BaseRunner):
    """
    Runner for RF-DETR models.

    Verified API from local rf-detr project:
    - Models: RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRSegPreview
    - train() params: dataset_dir, epochs, batch_size, grad_accum_steps, lr, output_dir,
                      early_stopping, early_stopping_patience, early_stopping_min_delta, resume
    - predict() params: image (PIL.Image), threshold
    """

    MODEL_MAP = {
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "seg": "RFDETRSegPreview",
        "seg-preview": "RFDETRSegPreview",
        "seg-nano": "RFDETRSegNano",
        "seg-small": "RFDETRSegSmall",
        "seg-medium": "RFDETRSegMedium",
        "seg-large": "RFDETRSegLarge",
        "seg-xlarge": "RFDETRSegXLarge",
        "seg-2xlarge": "RFDETRSeg2XLarge",
    }

    # Default pretrained-weight filenames for each model variant.
    # These map to keys in RF-DETR's HOSTED_MODELS / OPEN_SOURCE_MODELS dicts.
    PRETRAIN_WEIGHTS_MAP = {
        "base": "rf-detr-base.pth",
        "large": "rf-detr-large.pth",
        "nano": "rf-detr-nano.pth",
        "small": "rf-detr-small.pth",
        "medium": "rf-detr-medium.pth",
        "seg": "rf-detr-seg-preview.pt",
        "seg-preview": "rf-detr-seg-preview.pt",
        "seg-nano": "rf-detr-seg-nano.pt",
        "seg-small": "rf-detr-seg-small.pt",
        "seg-medium": "rf-detr-seg-medium.pt",
        "seg-large": "rf-detr-seg-large.pt",
        "seg-xlarge": "rf-detr-seg-xlarge.pt",
        "seg-2xlarge": "rf-detr-seg-xxlarge.pt",
    }

    RESOLUTION_MAP = {
        "nano": 384, "small": 512, "medium": 576,
        "base": 560, "large": 560, "seg": 432, "seg-preview": 432,
        "seg-nano": 312, "seg-small": 432, "seg-medium": 504,
        "seg-large": 560, "seg-xlarge": 672, "seg-2xlarge": 672,
    }

    def __init__(self):
        super().__init__(PROJECT_ROOT / ".venv-rfdetr")

    def _resolve_model_cls(self, config: dict[str, Any]) -> str:
        model_size = config.get("model", "medium")
        return self.MODEL_MAP.get(model_size, "RFDETRMedium")

    def _resolve_default_weights(self, config: dict[str, Any]) -> str:
        """Return the default pretrained-weight filename for the model variant."""
        model_size = config.get("model", "medium")
        return self.PRETRAIN_WEIGHTS_MAP.get(model_size, "")

    def train(self, config: dict[str, Any]) -> dict[str, Any] | None:
        train_cfg = config.get("train", {})
        device = train_cfg.get("device", 0)
        device_ids = _parse_device_ids(device)
        use_ddp = len(device_ids) > 1

        script_config = {
            "model_cls": self._resolve_model_cls(config),
            "default_pretrain_weights": self._resolve_default_weights(config),
            "data_path": config.get("data", {}).get("path", "data/"),
            "epochs": train_cfg.get("epochs", 100),
            "batch": train_cfg.get("batch", 4),
            "lr": train_cfg.get("lr", 1e-4),
            "grad_accum_steps": train_cfg.get("grad_accum_steps", 4),
            "device": device,
            "output_dir": train_cfg.get("output_dir", "outputs"),
            "early_stopping": train_cfg.get("early_stopping", False),
            "early_stopping_patience": train_cfg.get("early_stopping_patience", 10),
            "early_stopping_min_delta": train_cfg.get("early_stopping_min_delta", 0.001),
            "resume": train_cfg.get("resume", ""),
            "config_file": config.get("config_file", ""),
        }

        if use_ddp:
            print(f">>> Multi-GPU training: {len(device_ids)} GPUs {device_ids}")
            result = self._run_script_file_distributed(
                "rfdetr_train.py", script_config, device_ids,
            )
        else:
            result = self._run_script_file("rfdetr_train.py", script_config)
        if result.returncode != 0:
            print(f"\n[RF-DETR] Training failed with exit code {result.returncode}", file=sys.stderr)
            raise RuntimeError("Training failed")
        print("[RF-DETR] Training completed successfully")
        return self._parse_train_markers(result.stdout)

    def predict(self, config: dict[str, Any], source: str) -> Any:
        predict_cfg = config.get("predict", {})
        script_config = {
            "model_cls": self._resolve_model_cls(config),
            "default_pretrain_weights": self._resolve_default_weights(config),
            "source": source,
            "threshold": predict_cfg.get("threshold", 0.5),
            "weights": predict_cfg.get("weights", ""),
            "output_dir": predict_cfg.get("output_dir", "outputs/predict"),
            "device": config.get("train", {}).get("device", 0),
        }

        result = self._run_script_file("rfdetr_predict.py", script_config)
        if result.returncode != 0:
            print(f"[RF-DETR] Prediction failed", file=sys.stderr)
            raise RuntimeError("Prediction failed")
        return result.stdout

    def export(self, config: dict[str, Any], format: str = "onnx") -> Path:
        model_size = config.get("model", "medium")
        predict_cfg = config.get("predict", {})
        output_path = f"rfdetr_{model_size}.{format}"

        script_config = {
            "model_cls": self._resolve_model_cls(config),
            "default_pretrain_weights": self._resolve_default_weights(config),
            "format": format,
            "output_path": output_path,
            "weights": predict_cfg.get("weights", ""),
            "device": config.get("train", {}).get("device", 0),
        }

        result = self._run_script_file("rfdetr_export.py", script_config)
        if result.returncode != 0:
            print(f"[RF-DETR] Export failed", file=sys.stderr)
            raise RuntimeError("Export failed")
        return Path(output_path)

    def eval(self, config: dict[str, Any]) -> dict[str, Any]:
        eval_cfg = config.get("eval", {})
        train_cfg = config.get("train", {})
        model_size = config.get("model", "medium")
        is_seg = model_size.startswith("seg")

        weights = eval_cfg.get("weights", "")
        if not weights:
            print("[RF-DETR] Error: eval.weights must be specified", file=sys.stderr)
            raise ValueError("eval.weights is required")

        script_config = {
            "model_cls": self._resolve_model_cls(config),
            "default_pretrain_weights": self._resolve_default_weights(config),
            "data_path": config.get("data", {}).get("path", "data/"),
            "weights": weights,
            "device": train_cfg.get("device", 0),
            "batch": train_cfg.get("batch", 4),
            "output_dir": eval_cfg.get("output_dir", ""),
            "is_seg": is_seg,
        }

        result = self._run_script_file("rfdetr_eval.py", script_config)
        if result.returncode != 0:
            print(f"\n[RF-DETR] Evaluation failed with exit code {result.returncode}", file=sys.stderr)
            raise RuntimeError("Evaluation failed")
        print("[RF-DETR] Evaluation completed successfully")

        # Return the output directory where eval_metrics.json was saved
        output_dir = eval_cfg.get("output_dir", "")
        if not output_dir:
            output_dir = str(Path(weights).parent / "eval_results")
        return {"output_dir": output_dir, "metrics_json": str(Path(output_dir) / "eval_metrics.json")}
