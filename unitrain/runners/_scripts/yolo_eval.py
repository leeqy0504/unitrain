#!/usr/bin/env python3
"""Ultralytics YOLO evaluation script — executed in .venv-yolo.

Loads a trained model and runs validation, then outputs a unified JSON
metrics file compatible with eval_report.py.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow sibling-module import inside _scripts/
sys.path.insert(0, str(Path(__file__).parent))
from weight_utils import resolve_weight_path  # noqa: E402


def _parse_results_csv(csv_path: str) -> dict:
    """Parse Ultralytics results.csv for training curves."""
    epochs_data = []
    if not os.path.exists(csv_path):
        return {"epochs": []}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys
            row = {k.strip(): v.strip() for k, v in row.items()}
            epoch_info = {"epoch": int(row.get("epoch", len(epochs_data)))}
            # Collect losses
            for key in row:
                lk = key.lower()
                if "loss" in lk or "map" in lk or "precision" in lk or "recall" in lk:
                    try:
                        epoch_info[key.strip()] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            epochs_data.append(epoch_info)

    return {"epochs": epochs_data}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config_json)

    model_name = cfg.get("model", "yolo11n")
    task = cfg.get("task", "detect")
    weights = cfg.get("weights", "")
    data_yaml = cfg.get("data_yaml", "data.yaml")
    imgsz = cfg.get("imgsz", 640)
    batch = cfg.get("batch", 16)
    device = cfg.get("device", 0)
    output_dir = cfg.get("output_dir", "")
    conf_threshold = cfg.get("conf_threshold", 0.001)
    iou_threshold = cfg.get("iou_threshold", 0.5)

    if not weights:
        print("[YOLO Eval] Error: no weights specified for evaluation", file=sys.stderr)
        sys.exit(1)

    # Resolve weights path
    if os.path.exists(weights):
        weights_path = weights
    elif not os.path.isabs(weights):
        weights_path = resolve_weight_path(weights)
    else:
        weights_path = weights

    if not os.path.exists(weights_path):
        print(f"[YOLO Eval] Error: weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if not output_dir:
        output_dir = str(Path(weights_path).parent.parent / "eval_results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[YOLO Eval] Output: {output_dir}")
    print(f"[YOLO Eval] Weights: {weights_path}")
    print(f"[YOLO Eval] Data: {data_yaml}")

    from ultralytics import YOLO

    # Load trained model
    model = YOLO(weights_path, task=task) if task != "detect" else YOLO(weights_path)

    print(f"[YOLO Eval] Running validation (conf={conf_threshold}, iou={iou_threshold})...")

    # Run validation
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf_threshold,
        iou=iou_threshold,
        project=output_dir,
        name="val",
        exist_ok=True,
        plots=True,  # Generate PR curves, confusion matrix, etc.
    )

    # Build unified metrics output
    unified = {
        "framework": "ultralytics",
        "model": model_name,
        "task": task,
        "weights": weights_path,
        "timestamp": datetime.now().isoformat(),
        "overall": {},
        "per_class": [],
        "coco_stats": {},
        "curves": {
            "pr_curve": {},
            "f1_confidence": {},
            "confusion_matrix": [],
        },
        "training_log": {},
    }

    # Extract overall metrics
    is_seg = task == "segment"

    # Box metrics (always available)
    box = metrics.box
    unified["overall"]["mAP50_95"] = float(box.map)
    unified["overall"]["mAP50"] = float(box.map50)
    unified["overall"]["mAP75"] = float(box.map75)
    unified["overall"]["precision"] = float(box.mp)
    unified["overall"]["recall"] = float(box.mr)

    # Compute overall F1 from precision and recall
    p, r = box.mp, box.mr
    unified["overall"]["f1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    # Per-class metrics
    class_names = metrics.names if hasattr(metrics, "names") else {}
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        for i, cls_idx in enumerate(box.ap_class_index):
            cls_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
            cls_p = float(box.p[i]) if i < len(box.p) else 0
            cls_r = float(box.r[i]) if i < len(box.r) else 0
            cls_f1 = float(2 * cls_p * cls_r / (cls_p + cls_r)) if (cls_p + cls_r) > 0 else 0
            cls_info = {
                "name": cls_name,
                "mAP50_95": float(box.all_ap[i].mean()) if i < len(box.all_ap) else 0,
                "mAP50": float(box.ap50[i]) if i < len(box.ap50) else 0,
                "precision": cls_p,
                "recall": cls_r,
                "f1": cls_f1,
            }
            unified["per_class"].append(cls_info)

    # Segment metrics (if available)
    if is_seg and hasattr(metrics, "seg"):
        seg = metrics.seg
        unified["overall"]["mask_mAP50_95"] = float(seg.map)
        unified["overall"]["mask_mAP50"] = float(seg.map50)
        unified["overall"]["mask_precision"] = float(seg.mp)
        unified["overall"]["mask_recall"] = float(seg.mr)
        mp, mr = seg.mp, seg.mr
        unified["overall"]["mask_f1"] = float(2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0

    # ------------------------------------------------------------------
    # Extract curve data from Ultralytics Metric object
    # box has: px (confidence thresholds), f1_curve, p_curve, r_curve, prec_values
    # ------------------------------------------------------------------
    try:
        px = box.px  # ndarray shape (1000,)
        f1_curve_data = box.f1_curve  # ndarray shape (nc, 1000)
        p_curve_data = box.p_curve    # ndarray shape (nc, 1000)
        r_curve_data = box.r_curve    # ndarray shape (nc, 1000)

        if px is not None and f1_curve_data is not None and len(px) > 0:
            # Downsample to ~200 points for manageable JSON size
            step = max(1, len(px) // 200)
            idx = list(range(0, len(px), step))
            if idx[-1] != len(px) - 1:
                idx.append(len(px) - 1)

            sampled_px = [float(px[i]) for i in idx]
            f1_mean = [float(f1_curve_data[:, i].mean()) for i in idx]

            f1_per_class = {}
            p_per_class = {}
            r_per_class = {}
            cls_names_ordered = []
            if hasattr(box, 'ap_class_index') and box.ap_class_index is not None:
                for ci, cls_idx in enumerate(box.ap_class_index):
                    cname = class_names.get(int(cls_idx), f"class_{cls_idx}")
                    cls_names_ordered.append(cname)
                    if ci < f1_curve_data.shape[0]:
                        f1_per_class[cname] = [float(f1_curve_data[ci, i]) for i in idx]
                    if p_curve_data is not None and ci < p_curve_data.shape[0]:
                        p_per_class[cname] = [float(p_curve_data[ci, i]) for i in idx]
                    if r_curve_data is not None and ci < r_curve_data.shape[0]:
                        r_per_class[cname] = [float(r_curve_data[ci, i]) for i in idx]

            # Find optimal threshold (max mean F1)
            full_f1_mean = f1_curve_data.mean(0)
            best_idx = int(full_f1_mean.argmax())
            best_conf = float(px[best_idx])
            best_f1 = float(full_f1_mean[best_idx])

            unified["curves"]["f1_confidence"] = {
                "confidence": sampled_px,
                "f1_per_class": f1_per_class,
                "f1_mean": f1_mean,
                "class_names": cls_names_ordered,
                "best_conf": best_conf,
                "best_f1": best_f1,
            }

            unified["curves"]["pr_curve"] = {
                "confidence": sampled_px,
                "precision_per_class": p_per_class,
                "recall_per_class": r_per_class,
            }

            print(f"[YOLO Eval] F1-Confidence curve extracted ({len(sampled_px)} points, best F1={best_f1:.4f} @ conf={best_conf:.3f})")
    except Exception as e:
        print(f"[YOLO Eval] Warning: could not extract curve data: {e}")

    # Also extract mask curves for segmentation
    if is_seg and hasattr(metrics, 'seg'):
        try:
            seg_box = metrics.seg
            seg_px = seg_box.px
            seg_f1_curve = seg_box.f1_curve
            if seg_px is not None and seg_f1_curve is not None and len(seg_px) > 0:
                step = max(1, len(seg_px) // 200)
                idx = list(range(0, len(seg_px), step))
                if idx[-1] != len(seg_px) - 1:
                    idx.append(len(seg_px) - 1)

                sampled_px = [float(seg_px[i]) for i in idx]
                f1_mean = [float(seg_f1_curve[:, i].mean()) for i in idx]
                f1_per_class = {}
                if hasattr(seg_box, 'ap_class_index') and seg_box.ap_class_index is not None:
                    for ci, cls_idx in enumerate(seg_box.ap_class_index):
                        cname = class_names.get(int(cls_idx), f"class_{cls_idx}")
                        if ci < seg_f1_curve.shape[0]:
                            f1_per_class[cname] = [float(seg_f1_curve[ci, i]) for i in idx]

                full_f1_mean = seg_f1_curve.mean(0)
                best_idx = int(full_f1_mean.argmax())

                unified["curves"]["mask_f1_confidence"] = {
                    "confidence": sampled_px,
                    "f1_per_class": f1_per_class,
                    "f1_mean": f1_mean,
                    "best_conf": float(seg_px[best_idx]),
                    "best_f1": float(full_f1_mean[best_idx]),
                }
                print(f"[YOLO Eval] Mask F1-Confidence curve extracted")
        except Exception as e:
            print(f"[YOLO Eval] Warning: could not extract mask curve data: {e}")

    # Try to pick up framework-generated plot images
    val_dir = Path(output_dir) / "val"

    # Check for PR curve plot
    pr_curve_path = val_dir / "PR_curve.png"
    if pr_curve_path.exists():
        unified["curves"]["pr_curve"]["plot_path"] = str(pr_curve_path)

    # Confusion matrix
    cm_path = val_dir / "confusion_matrix.png"
    cm_norm_path = val_dir / "confusion_matrix_normalized.png"
    if cm_path.exists():
        unified["curves"]["confusion_matrix_path"] = str(cm_path)
    if cm_norm_path.exists():
        unified["curves"]["confusion_matrix_normalized_path"] = str(cm_norm_path)

    # Parse training history (look for results.csv in training output)
    weights_parent = Path(weights_path).parent
    # For YOLO: weights are at train/weights/best.pt, results.csv at train/results.csv
    results_csv = weights_parent.parent / "results.csv"
    if not results_csv.exists():
        # Try other common locations
        for candidate in [
            weights_parent / "results.csv",
            weights_parent.parent.parent / "results.csv",
        ]:
            if candidate.exists():
                results_csv = candidate
                break

    if results_csv.exists():
        unified["training_log"] = _parse_results_csv(str(results_csv))
        unified["training_log"]["source"] = str(results_csv)

    # Copy Ultralytics-generated plots to our output directory (already in val_dir)
    # List generated plots for reference
    plot_files = []
    if val_dir.exists():
        for p in val_dir.glob("*.png"):
            plot_files.append(str(p))
    unified["plots_generated"] = plot_files

    # Save unified metrics JSON
    metrics_path = Path(output_dir) / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
    print(f"[YOLO Eval] Metrics saved: {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("YOLO Evaluation Results")
    print("=" * 60)
    ov = unified["overall"]
    print(f"  mAP@50:95 = {ov.get('mAP50_95', 'N/A'):.4f}")
    print(f"  mAP@50    = {ov.get('mAP50', 'N/A'):.4f}")
    print(f"  mAP@75    = {ov.get('mAP75', 'N/A'):.4f}")
    print(f"  Precision = {ov.get('precision', 'N/A'):.4f}")
    print(f"  Recall    = {ov.get('recall', 'N/A'):.4f}")
    print(f"  F1        = {ov.get('f1', 'N/A'):.4f}")
    if is_seg:
        print(f"  --- Mask ---")
        print(f"  Mask mAP@50:95 = {ov.get('mask_mAP50_95', 'N/A'):.4f}")
        print(f"  Mask mAP@50    = {ov.get('mask_mAP50', 'N/A'):.4f}")
        print(f"  Mask F1        = {ov.get('mask_f1', 'N/A'):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
