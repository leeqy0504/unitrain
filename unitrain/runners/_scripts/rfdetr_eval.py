#!/usr/bin/env python3
"""RF-DETR evaluation script — executed in .venv-rfdetr.

Loads a trained checkpoint and runs COCO evaluation on the validation set.
Outputs a unified JSON metrics file compatible with eval_report.py.

Strategy: monkey-patch ``rfdetr.main.evaluate`` so that when
``model.train(eval=True)`` internally calls ``evaluate()``, we capture
the full ``test_stats`` dict (which contains ``results_json``,
``coco_eval_bbox``, etc.) **and** the ``coco_evaluator`` object (which
carries ``evalImgs`` – needed for the confidence-sweep F1 curves).
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Allow sibling-module import inside _scripts/
sys.path.insert(0, str(Path(__file__).parent))
from weight_utils import resolve_rfdetr_weight  # noqa: E402


def get_timestamped_output_dir(base_dir: str = "outputs", prefix: str = "output") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_dir}/{prefix}_{timestamp}"


# ======================================================================
# Monkey-patch helpers
# ======================================================================

_captured: dict = {
    "stats": {},
    "coco_evals": {},       # {iou_type: COCOeval}
}


def _install_eval_hook() -> None:
    """Replace ``rfdetr.main.evaluate`` with a wrapper that stores results."""
    import rfdetr.main as _main  # noqa: trigger module import

    _original_evaluate = _main.evaluate

    def _hooked_evaluate(model, criterion, postprocess, data_loader,
                         base_ds, device, args=None, header="Eval"):
        stats, coco_evaluator = _original_evaluate(
            model, criterion, postprocess, data_loader,
            base_ds, device, args, header,
        )
        _captured["stats"] = dict(stats)
        if coco_evaluator is not None:
            for iou_type in ("bbox", "segm"):
                if iou_type in coco_evaluator.coco_eval:
                    _captured["coco_evals"][iou_type] = coco_evaluator.coco_eval[iou_type]
        return stats, coco_evaluator

    _main.evaluate = _hooked_evaluate


# ======================================================================
# F1-Confidence sweep  (replicates rfdetr.engine logic on raw evalImgs)
# ======================================================================

def _f1_confidence_curves(coco_eval, class_names: list[str],
                          n_points: int = 201) -> dict:
    """Build per-class + mean F1 vs confidence curves from a COCOeval object.

    Returns a dict matching the format consumed by
    ``eval_report.plot_f1_confidence_from_data``.
    """
    evalImgs = coco_eval.evalImgs
    if not evalImgs:
        return {}

    iou50_idx = int(np.argmax(np.isclose(coco_eval.params.iouThrs, 0.50)))
    cat_ids = coco_eval.params.catIds
    num_classes = len(cat_ids)
    area_rng_all = tuple(coco_eval.params.areaRng[0])

    # Build a fast lookup: (cat_id, area_rng) → {img_id: evalImg}
    lookup: dict = {}
    for e in evalImgs:
        if e is None:
            continue
        key = (e["category_id"], tuple(e["aRng"]))
        lookup.setdefault(key, {})[e["image_id"]] = e

    per_class_data: list[dict] = []
    for cid in cat_ids:
        dt_scores, dt_matches, dt_ignore = [], [], []
        total_gt = 0
        bucket = lookup.get((cid, area_rng_all), {})
        for img_id in coco_eval.params.imgIds:
            e = bucket.get(img_id)
            if e is None:
                continue
            total_gt += sum(1 for ig in e["gtIgnore"] if not ig)
            for d in range(len(e["dtIds"])):
                dt_scores.append(e["dtScores"][d])
                dt_matches.append(e["dtMatches"][iou50_idx, d])
                dt_ignore.append(e["dtIgnore"][iou50_idx, d])
        per_class_data.append({
            "scores": np.array(dt_scores),
            "matches": np.array(dt_matches),
            "ignore": np.array(dt_ignore, dtype=bool) if dt_ignore else np.array([], dtype=bool),
            "total_gt": total_gt,
        })

    conf_thresholds = np.linspace(0.0, 1.0, n_points)
    classes_with_gt = [k for k in range(num_classes) if per_class_data[k]["total_gt"] > 0]
    if not classes_with_gt:
        return {}

    # Map cat_id index → human-readable name
    cat_id_to_name = {}
    try:
        cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    except Exception:
        pass

    def _cls_name(k: int) -> str:
        cid = cat_ids[k]
        if cid in cat_id_to_name:
            return cat_id_to_name[cid]
        return class_names[k] if k < len(class_names) else f"class_{k}"

    f1_per_class: dict[str, list[float]] = {_cls_name(k): [] for k in classes_with_gt}
    f1_mean: list[float] = []

    for conf_thresh in conf_thresholds:
        class_f1s: list[float] = []
        for k in range(num_classes):
            data = per_class_data[k]
            if data["scores"].size == 0:
                f1 = 0.0
            else:
                above = data["scores"] >= conf_thresh
                valid = above & ~data["ignore"]
                vm = data["matches"][valid]
                tp = float(np.sum(vm != 0))
                fp = float(np.sum(vm == 0))
                fn = data["total_gt"] - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if k in classes_with_gt:
                f1_per_class[_cls_name(k)].append(float(f1))
            class_f1s.append(f1)

        f1_mean.append(float(np.mean([class_f1s[k] for k in classes_with_gt])))

    best_idx = int(np.argmax(f1_mean))
    return {
        "confidence": conf_thresholds.tolist(),
        "f1_per_class": f1_per_class,
        "f1_mean": f1_mean,
        "class_names": [_cls_name(k) for k in classes_with_gt],
        "best_conf": float(conf_thresholds[best_idx]),
        "best_f1": float(f1_mean[best_idx]),
    }


# ======================================================================
# COCO 12-stat extraction helpers
# ======================================================================

_STAT_KEYS = [
    "mAP50_95", "mAP50", "mAP75",
    "mAP_small", "mAP_medium", "mAP_large",
    "AR@1", "AR@10", "AR@100",
    "AR_small", "AR_medium", "AR_large",
]


def _extract_coco_12stats(stats: dict) -> dict:
    """Build ``coco_stats`` dict from captured *test_stats*."""
    result: dict = {}
    bbox_list = stats.get("coco_eval_bbox", [])
    if bbox_list and len(bbox_list) >= 12:
        result["bbox"] = dict(zip(_STAT_KEYS, [float(v) for v in bbox_list[:12]]))
    mask_list = stats.get("coco_eval_masks", [])
    if mask_list and len(mask_list) >= 12:
        result["segm"] = dict(zip(_STAT_KEYS, [float(v) for v in mask_list[:12]]))
    return result


# ======================================================================
# Per-class metrics from results_json
# ======================================================================

def _extract_per_class(stats: dict, is_seg: bool) -> tuple[dict, list[dict]]:
    """Return ``(overall_dict, per_class_list)`` from captured stats."""
    overall: dict = {}
    per_class: list[dict] = []

    rj = stats.get("results_json", {})
    rj_mask = stats.get("results_json_masks", {})

    # Use mask results_json as primary for seg models
    primary_rj = rj_mask if (is_seg and rj_mask) else rj

    if primary_rj:
        overall["precision"] = primary_rj.get("precision", 0)
        overall["recall"] = primary_rj.get("recall", 0)
        overall["f1"] = primary_rj.get("f1_score", 0)

        class_map = primary_rj.get("class_map", [])
        for cls_info in class_map:
            if cls_info.get("class") == "all":
                overall["mAP50_95"] = cls_info.get("map@50:95", 0)
                overall["mAP50"] = cls_info.get("map@50", 0)
            else:
                per_class.append({
                    "name": cls_info.get("class", ""),
                    "mAP50_95": cls_info.get("map@50:95", 0),
                    "mAP50": cls_info.get("map@50", 0),
                    "precision": cls_info.get("precision", 0),
                    "recall": cls_info.get("recall", 0),
                    "f1": cls_info.get("f1_score", 0),
                })

    # If seg model, also merge bbox per-class AP into existing entries
    if is_seg and rj and rj is not primary_rj:
        bbox_by_name: dict[str, dict] = {}
        for cls_info in rj.get("class_map", []):
            name = cls_info.get("class", "")
            if name == "all":
                overall["bbox_mAP50_95"] = cls_info.get("map@50:95", 0)
                overall["bbox_mAP50"] = cls_info.get("map@50", 0)
                overall.setdefault("bbox_f1", cls_info.get("f1_score", 0))
            else:
                bbox_by_name[name] = {
                    "bbox_mAP50_95": cls_info.get("map@50:95", 0),
                    "bbox_mAP50": cls_info.get("map@50", 0),
                }
        for pc in per_class:
            bdata = bbox_by_name.get(pc["name"])
            if bdata:
                pc.update(bdata)

    return overall, per_class


# ======================================================================
# Category names from COCO annotation
# ======================================================================

def _load_class_names(data_path: str) -> list[str]:
    candidates = [
        os.path.join(data_path, "valid", "_annotations.coco.json"),
        os.path.join(data_path, "val", "_annotations.coco.json"),
        os.path.join(data_path, "annotations", "instances_val.json"),
        os.path.join(data_path, "annotations", "val.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                coco = json.load(f)
            cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
            return [c["name"] for c in cats]
    return []


# ======================================================================
# Training log parser
# ======================================================================

def _parse_training_log(log_path: str) -> dict:
    epochs_data: list[dict] = []
    if not os.path.exists(log_path):
        return {"epochs": []}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                info: dict = {
                    "epoch": entry.get("epoch", len(epochs_data)),
                    "train_loss": entry.get("train_loss"),
                    "test_loss": entry.get("test_loss"),
                }
                for src, dst in [
                    ("test_coco_eval_bbox", ("mAP50_95", "mAP50")),
                    ("test_coco_eval_masks", ("mask_mAP50_95", "mask_mAP50")),
                    ("ema_test_coco_eval_bbox", ("ema_mAP50_95", "ema_mAP50")),
                ]:
                    arr = entry.get(src, [])
                    if arr and len(arr) >= 2:
                        info[dst[0]] = arr[0]
                        info[dst[1]] = arr[1]
                epochs_data.append(info)
            except (json.JSONDecodeError, KeyError):
                continue
    return {"epochs": epochs_data}


# ======================================================================
# New RF-DETR fallback: parse Lightning metrics.csv
# ======================================================================

def _to_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_metrics_csv(weights_path: str, output_dir: str) -> Path | None:
    candidates = [
        Path(weights_path).parent / "metrics.csv",
        Path(output_dir).parent / "metrics.csv",
        Path(output_dir) / "metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_metrics_csv(metrics_csv: Path) -> tuple[dict, list[dict], dict]:
    rows: list[dict] = []
    with open(metrics_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    eval_rows = [
        row for row in rows
        if _to_float(row.get("val/mAP_50_95")) is not None
        or _to_float(row.get("val/segm_mAP_50_95")) is not None
    ]
    if not eval_rows:
        return {}, [], {"epochs": []}

    # Prefer the best validation row, not merely the final row. For segmentation
    # runs RF-DETR checkpoints monitor segm mAP; detection runs monitor bbox mAP.
    metric_key = "val/segm_mAP_50_95"
    if all(_to_float(row.get(metric_key)) is None for row in eval_rows):
        metric_key = "val/mAP_50_95"

    best_row = max(eval_rows, key=lambda row: _to_float(row.get(metric_key)) or float("-inf"))

    overall = {
        "mAP50_95": _to_float(best_row.get(metric_key)) or 0.0,
        "mAP50": _to_float(best_row.get("val/segm_mAP_50" if metric_key.startswith("val/segm") else "val/mAP_50")) or 0.0,
        "mAP75": _to_float(best_row.get("val/mAP_75")) or 0.0,
        "AR@100": _to_float(best_row.get("val/mAR")) or 0.0,
        "precision": _to_float(best_row.get("val/precision")) or 0.0,
        "recall": _to_float(best_row.get("val/recall")) or 0.0,
        "f1": _to_float(best_row.get("val/F1")) or 0.0,
    }
    if _to_float(best_row.get("val/segm_mAP_50_95")) is not None:
        overall["bbox_mAP50_95"] = _to_float(best_row.get("val/mAP_50_95")) or 0.0
        overall["bbox_mAP50"] = _to_float(best_row.get("val/mAP_50")) or 0.0

    per_class: list[dict] = []
    for key, value in best_row.items():
        if not key.startswith("val/AP/"):
            continue
        ap = _to_float(value)
        if ap is None:
            continue
        name = key.split("val/AP/", 1)[1]
        per_class.append({
            "name": name,
            "mAP50_95": ap,
            "mAP50": ap,
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
        })

    epochs = []
    for row in eval_rows:
        epoch = _to_float(row.get("epoch"))
        item = {
            "epoch": int(epoch) if epoch is not None else len(epochs),
            "train_loss": _to_float(row.get("train/loss")),
            "test_loss": _to_float(row.get("val/loss")),
            "mAP50_95": _to_float(row.get("val/mAP_50_95")),
            "mAP50": _to_float(row.get("val/mAP_50")),
            "mask_mAP50_95": _to_float(row.get("val/segm_mAP_50_95")),
            "mask_mAP50": _to_float(row.get("val/segm_mAP_50")),
            "ema_mAP50_95": _to_float(row.get("val/ema_mAP_50_95")),
            "ema_mAP50": _to_float(row.get("val/ema_mAP_50")),
        }
        epochs.append({k: v for k, v in item.items() if v is not None})

    return overall, per_class, {"epochs": epochs}


def _build_unified_from_metrics_csv(
    metrics_csv: Path,
    *,
    model_cls_name: str,
    is_seg: bool,
    weights_path: str,
) -> dict:
    overall, per_class, training_log = _parse_metrics_csv(metrics_csv)
    coco_stats = {}
    if overall:
        coco_stats["segm" if is_seg else "bbox"] = {
            "mAP50_95": overall.get("mAP50_95", 0.0),
            "mAP50": overall.get("mAP50", 0.0),
            "mAP75": overall.get("mAP75", 0.0),
            "AR@100": overall.get("AR@100", 0.0),
        }
        if is_seg and "bbox_mAP50_95" in overall:
            coco_stats["bbox"] = {
                "mAP50_95": overall.get("bbox_mAP50_95", 0.0),
                "mAP50": overall.get("bbox_mAP50", 0.0),
            }

    return {
        "framework": "rfdetr",
        "model": model_cls_name,
        "task": "segment" if is_seg else "detect",
        "weights": weights_path,
        "timestamp": datetime.now().isoformat(),
        "overall": overall,
        "per_class": per_class,
        "coco_stats": coco_stats,
        "curves": {
            "pr_curve": {},
            "f1_confidence": {},
            "mask_f1_confidence": {},
            "confusion_matrix": [],
        },
        "training_log": training_log,
        "source": str(metrics_csv),
    }


def _write_unified_metrics(unified: dict, output_dir: str) -> Path:
    metrics_path = Path(output_dir) / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
    print(f"[RF-DETR Eval] Metrics saved: {metrics_path}")
    return metrics_path


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config_json)

    model_cls_name = cfg["model_cls"]
    data_path = cfg.get("data_path", "data/")
    weights = cfg.get("weights", "")
    device = cfg.get("device", 0)
    output_dir = cfg.get("output_dir", "")
    default_weights = cfg.get("default_pretrain_weights", "")
    is_seg = cfg.get("is_seg", False)

    if not weights:
        print("[RF-DETR Eval] Error: no weights specified", file=sys.stderr)
        sys.exit(1)

    # Resolve weights path
    weights_path = weights
    if not os.path.isabs(weights_path) and not os.path.exists(weights_path):
        resolved = resolve_rfdetr_weight(weights)
        if resolved and os.path.exists(resolved):
            weights_path = resolved
    if not os.path.exists(weights_path):
        print(f"[RF-DETR Eval] Error: weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    if not output_dir:
        output_dir = str(Path(weights_path).parent / "eval_results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[RF-DETR Eval] Output: {output_dir}")
    print(f"[RF-DETR Eval] Weights: {weights_path}")
    print(f"[RF-DETR Eval] Dataset: {data_path}")

    # Device
    if isinstance(device, str) and device.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif isinstance(device, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    elif isinstance(device, str):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(",")[0].strip()

    # ------------------------------------------------------------------
    # Import RF-DETR and install the evaluate() hook
    # ------------------------------------------------------------------
    import rfdetr

    if importlib.util.find_spec("rfdetr.main") is None:
        metrics_csv = _find_metrics_csv(weights_path, output_dir)
        if metrics_csv is None:
            print(
                "[RF-DETR Eval] Error: installed RF-DETR has no rfdetr.main "
                "and no metrics.csv was found next to the weights.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[RF-DETR Eval] New RF-DETR layout detected; reading metrics: {metrics_csv}")
        unified = _build_unified_from_metrics_csv(
            metrics_csv,
            model_cls_name=model_cls_name,
            is_seg=is_seg,
            weights_path=weights_path,
        )
        _write_unified_metrics(unified, output_dir)
        _print_summary(unified, is_seg)
        return

    _install_eval_hook()

    model_cls = getattr(rfdetr, model_cls_name)
    model_kwargs = {}
    if default_weights:
        resolved = resolve_rfdetr_weight(default_weights)
        if resolved:
            model_kwargs["pretrain_weights"] = resolved
    model = model_cls(**model_kwargs)

    print(f"[RF-DETR Eval] Running evaluation (eval=True, resume={weights_path})...")
    model.train(
        dataset_dir=data_path,
        output_dir=output_dir,
        resume=weights_path,
        eval=True,
        epochs=9999,
        batch_size=cfg.get("batch", 4),
    )

    # ------------------------------------------------------------------
    # Extract everything from captured data
    # ------------------------------------------------------------------
    stats = _captured["stats"]
    coco_evals = _captured["coco_evals"]
    class_names = _load_class_names(data_path)

    # 1. COCO 12-stats
    coco_stats = _extract_coco_12stats(stats)

    # 2. Overall + per-class from results_json
    overall, per_class = _extract_per_class(stats, is_seg)

    # Supplement overall with COCO 12-stats
    primary_key = "segm" if is_seg and "segm" in coco_stats else "bbox"
    primary_stats = coco_stats.get(primary_key, {})
    if primary_stats:
        overall.setdefault("mAP50_95", primary_stats.get("mAP50_95", 0))
        overall.setdefault("mAP50", primary_stats.get("mAP50", 0))
        overall.setdefault("mAP75", primary_stats.get("mAP75", 0))
        overall.setdefault("AR@100", primary_stats.get("AR@100", 0))
    bbox_12 = coco_stats.get("bbox", {})
    if bbox_12 and is_seg:
        overall.setdefault("bbox_mAP50_95", bbox_12.get("mAP50_95", 0))
        overall.setdefault("bbox_mAP50", bbox_12.get("mAP50", 0))

    # 3. F1-Confidence curves from coco_evaluator objects
    curves: dict = {
        "pr_curve": {},
        "f1_confidence": {},
        "mask_f1_confidence": {},
        "confusion_matrix": [],
    }

    if "bbox" in coco_evals:
        box_f1 = _f1_confidence_curves(coco_evals["bbox"], class_names)
        if box_f1:
            curves["f1_confidence"] = box_f1
            print(f"[RF-DETR Eval] Box F1-Confidence: best F1={box_f1['best_f1']:.4f} @ conf={box_f1['best_conf']:.3f}")

    if "segm" in coco_evals:
        mask_f1 = _f1_confidence_curves(coco_evals["segm"], class_names)
        if mask_f1:
            curves["mask_f1_confidence"] = mask_f1
            print(f"[RF-DETR Eval] Mask F1-Confidence: best F1={mask_f1['best_f1']:.4f} @ conf={mask_f1['best_conf']:.3f}")

    # 4. Training log from weights dir
    weights_dir = Path(weights_path).parent
    training_log_path = weights_dir / "log.txt"
    training_log = _parse_training_log(str(training_log_path)) if training_log_path.exists() else {}

    # 5. Fallback: merge P/R/F1 from training-time results.json if we
    #    didn't get per_class from captured stats (shouldn't happen, but
    #    guard against edge cases)
    if not per_class:
        results_json_path = weights_dir / "results.json"
        if results_json_path.exists():
            with open(results_json_path) as f:
                rj = json.load(f)
            class_map_raw = rj.get("class_map", [])
            if isinstance(class_map_raw, dict):
                class_map_raw = class_map_raw.get("valid", [])
            for cls_info in class_map_raw:
                name = cls_info.get("class", "")
                if name == "all":
                    overall.setdefault("precision", cls_info.get("precision", 0))
                    overall.setdefault("recall", cls_info.get("recall", 0))
                    overall.setdefault("f1", cls_info.get("f1_score", 0))
                else:
                    per_class.append({
                        "name": name,
                        "mAP50_95": cls_info.get("map@50:95", 0),
                        "mAP50": cls_info.get("map@50", 0),
                        "precision": cls_info.get("precision", 0),
                        "recall": cls_info.get("recall", 0),
                        "f1": cls_info.get("f1_score", 0),
                    })

    # ------------------------------------------------------------------
    # Build & save unified metrics
    # ------------------------------------------------------------------
    unified = {
        "framework": "rfdetr",
        "model": model_cls_name,
        "task": "segment" if is_seg else "detect",
        "weights": weights_path,
        "timestamp": datetime.now().isoformat(),
        "overall": overall,
        "per_class": per_class,
        "coco_stats": coco_stats,
        "curves": curves,
        "training_log": training_log,
    }

    _write_unified_metrics(unified, output_dir)
    _print_summary(unified, is_seg)


def _print_summary(unified: dict, is_seg: bool) -> None:
    per_class = unified.get("per_class", [])

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RF-DETR Evaluation Results")
    print("=" * 60)
    ov = unified["overall"]
    for key in ["mAP50_95", "mAP50", "mAP75", "precision", "recall", "f1", "AR@100"]:
        val = ov.get(key)
        label = key.replace("_", "@").ljust(12)
        print(f"  {label} = {val:.4f}" if val is not None else f"  {label} = N/A")
    if is_seg and "bbox_mAP50_95" in ov:
        print(f"  bbox mAP@50:95 = {ov['bbox_mAP50_95']:.4f}")
        print(f"  bbox mAP@50    = {ov['bbox_mAP50']:.4f}")
    print("-" * 60)
    if per_class:
        print("  Per-class metrics:")
        for pc in per_class:
            name = pc["name"].ljust(25)
            ap50 = pc.get("mAP50", 0)
            f1 = pc.get("f1")
            extra = f"  F1={f1:.4f}" if f1 is not None else ""
            print(f"    {name} AP@50={ap50:.4f}{extra}")
    print("=" * 60)


if __name__ == "__main__":
    main()
