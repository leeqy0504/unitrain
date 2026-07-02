#!/usr/bin/env python3
"""RF-DETR prediction script — executed in .venv-rfdetr."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow sibling-module import inside _scripts/
sys.path.insert(0, str(Path(__file__).parent))
from weight_utils import resolve_rfdetr_weight  # noqa: E402

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_timestamped_output_dir(base_dir: str = "outputs", prefix: str = "output") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_dir}/{prefix}_{timestamp}"


def collect_image_files(source: str) -> list[str]:
    if os.path.isdir(source):
        return sorted(
            os.path.join(source, f)
            for f in os.listdir(source)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
    return [source]


def annotate_and_save(image: Image.Image, detections, output_path: str) -> None:
    """Draw annotations using supervision and save the image."""
    try:
        import supervision as sv

        annotated_frame = np.array(image.copy())
        n_det = len(detections.xyxy) if detections.xyxy is not None and len(detections.xyxy) > 0 else 0

        if n_det > 0:
            labels = [
                f"cls_{int(c)} {s:.2f}"
                for c, s in zip(detections.class_id, detections.confidence)
            ]

            has_masks = detections.mask is not None and len(detections.mask) > 0
            if has_masks:
                annotated_frame = sv.MaskAnnotator().annotate(annotated_frame, detections)

            annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
            annotated_frame = sv.LabelAnnotator().annotate(
                annotated_frame, detections, labels=labels
            )

        Image.fromarray(annotated_frame).save(output_path)
    except ImportError:
        pass  # supervision not available, skip visualization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config_json)

    model_cls_name = cfg["model_cls"]
    source = cfg["source"]
    threshold = cfg.get("threshold", 0.5)
    weights = cfg.get("weights", "")
    default_weights = cfg.get("default_pretrain_weights", "")
    output_dir = cfg.get("output_dir", "outputs/predict")
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

    image_files = collect_image_files(source)
    print(f"Found {len(image_files)} image(s) to process")

    output_dir = get_timestamped_output_dir(output_dir, "predict")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    all_results: dict[str, list] = {}

    for idx, img_path in enumerate(image_files):
        fname = os.path.basename(img_path)
        image = Image.open(img_path)
        detections = model.predict(image, threshold=threshold)

        n_det = len(detections.xyxy) if detections.xyxy is not None and len(detections.xyxy) > 0 else 0

        det_list = []
        if n_det > 0:
            for box, score, cls_id in zip(
                detections.xyxy, detections.confidence, detections.class_id
            ):
                det_list.append(
                    {
                        "class_id": int(cls_id),
                        "confidence": float(score),
                        "bbox": [float(x) for x in box.tolist()],
                    }
                )
        all_results[fname] = det_list

        annotate_and_save(image, detections, os.path.join(output_dir, fname))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_files):
            print(f"  Processed {idx + 1}/{len(image_files)} | {fname}: {n_det} detections")

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")
    print(f"Total: {len(image_files)} images, {sum(len(v) for v in all_results.values())} detections")


if __name__ == "__main__":
    main()
