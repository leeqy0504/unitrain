"""Unified evaluation report generator for UniTrain.

Framework-agnostic module that reads the unified eval_metrics.json produced
by rfdetr_eval.py / yolo_eval.py and generates:
  - 6 types of visualization charts (PNG)
  - Terminal table printout
  - JSON / CSV / Markdown report files

Dependencies: matplotlib (available in both venvs).
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# matplotlib must be imported with Agg backend for headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ============================================================================
# Plot generation
# ============================================================================

def _setup_plot_style():
    """Set a clean plot style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "figure.dpi": 150,
    })


def plot_per_class_bar(metrics: dict, output_dir: str) -> str | dict[str, str]:
    """Per-class bar chart — one file per metric, each with full (0-1) and zoomed (0.8-1) panels."""
    _setup_plot_style()
    per_class = metrics.get("per_class", [])
    if not per_class:
        return ""

    names = [c["name"] for c in per_class]
    n_classes = len(names)
    metric_keys = [
        ("mAP50", "mAP@50", "#2196F3"),
        ("f1", "F1", "#4CAF50"),
        ("precision", "Precision", "#FF9800"),
        ("recall", "Recall", "#E91E63"),
    ]

    results: dict[str, str] = {}

    for key, label, color in metric_keys:
        values = [c.get(key, 0) or 0 for c in per_class]
        # Skip metrics where no class has a meaningful value
        if all(v == 0 or v is None for v in values):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3, 0.45 * n_classes + 1)))

        for ax, (xlim_lo, xlim_hi, subtitle) in zip(
            axes,
            [(0, 1.0, f"{label} (0\u20131)"), (0.8, 1.0, f"{label} (0.8\u20131)")],
        ):
            y_pos = list(range(n_classes))
            ax.barh(y_pos, values, color=color, alpha=0.8, height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel(label)
            ax.set_xlim(xlim_lo, xlim_hi)
            ax.set_title(subtitle, fontweight="bold")
            for i, v in enumerate(values):
                # Place label inside bar if value outside zoomed range
                if v >= xlim_lo:
                    offset = (xlim_hi - xlim_lo) * 0.01
                    ax.text(min(v + offset, xlim_hi - offset * 2), i,
                            f"{v:.4f}", va="center", fontsize=8, fontweight="bold")

        fig.suptitle(f"Per-Class {label}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fname = f"per_class_{key}.png"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        results[f"Per-Class {label}"] = path

    if len(results) == 1:
        return next(iter(results.values()))
    return results


def plot_training_loss_curve(metrics: dict, output_dir: str) -> str:
    """Training loss curve (standalone file)."""
    _setup_plot_style()
    training_log = metrics.get("training_log", {})
    epochs_data = training_log.get("epochs", [])
    if not epochs_data:
        return ""

    epochs = [e.get("epoch", i) for i, e in enumerate(epochs_data)]
    results: dict[str, str] = {}

    # ── Loss plot ──
    fig, ax = plt.subplots(figsize=(10, 5))
    has_loss = False

    # RF-DETR style
    train_loss = [e.get("train_loss") for e in epochs_data]
    test_loss = [e.get("test_loss") for e in epochs_data]
    if any(v is not None for v in train_loss):
        valid_epochs = [ep for ep, v in zip(epochs, train_loss) if v is not None]
        valid_vals = [v for v in train_loss if v is not None]
        ax.plot(valid_epochs, valid_vals, label="Train Loss", color="#2196F3", linewidth=1.5)
        has_loss = True
    if any(v is not None for v in test_loss):
        valid_epochs = [ep for ep, v in zip(epochs, test_loss) if v is not None]
        valid_vals = [v for v in test_loss if v is not None]
        ax.plot(valid_epochs, valid_vals, label="Val Loss", color="#E91E63", linewidth=1.5)
        has_loss = True

    # YOLO style
    for loss_key_pattern in ["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss",
                              "val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss"]:
        vals = [e.get(loss_key_pattern) for e in epochs_data]
        if any(v is not None for v in vals):
            valid_epochs = [ep for ep, v in zip(epochs, vals) if v is not None]
            valid_vals = [v for v in vals if v is not None]
            short_name = loss_key_pattern.replace("train/", "T:").replace("val/", "V:")
            ax.plot(valid_epochs, valid_vals, label=short_name, linewidth=1.2)
            has_loss = True

    if has_loss:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss", fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        path = os.path.join(output_dir, "training_loss.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    plt.close(fig)
    return ""


def plot_overall_summary(metrics: dict, output_dir: str) -> str:
    """Overall metrics radar/summary chart."""
    _setup_plot_style()
    overall = metrics.get("overall", {})
    if not overall:
        return ""

    # Collect available metrics for radar chart
    radar_items = []
    for key, label in [
        ("mAP50_95", "mAP@50:95"),
        ("mAP50", "mAP@50"),
        ("mAP75", "mAP@75"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]:
        val = overall.get(key)
        if val is not None and isinstance(val, (int, float)):
            radar_items.append((label, float(val)))

    if len(radar_items) < 3:
        return ""

    labels = [item[0] for item in radar_items]
    values = [item[1] for item in radar_items]
    n = len(labels)

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles_plot, values_plot, alpha=0.25, color="#2196F3")
    ax.plot(angles_plot, values_plot, color="#2196F3", linewidth=2)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Overall Metrics — {metrics.get('model', '')}", pad=20, fontsize=14, fontweight="bold")

    # Annotate values
    for angle, val, label in zip(angles, values, labels):
        ax.annotate(f"{val:.3f}", xy=(angle, val), fontsize=9,
                    ha="center", va="bottom", fontweight="bold")

    path = os.path.join(output_dir, "overall_summary.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix_from_data(metrics: dict, output_dir: str) -> str:
    """Plot confusion matrix if data is available in metrics."""
    _setup_plot_style()
    cm_data = metrics.get("curves", {}).get("confusion_matrix", [])
    if not cm_data or not isinstance(cm_data, list) or len(cm_data) == 0:
        return ""

    cm = np.array(cm_data)
    n = cm.shape[0]
    class_names = [c["name"] for c in metrics.get("per_class", [])]
    if len(class_names) < n:
        class_names += [f"cls_{i}" for i in range(len(class_names), n)]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names[:n], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names[:n], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontweight="bold")

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pr_curve_from_data(metrics: dict, output_dir: str) -> str:
    """Plot PR curve if data is available. For frameworks that already
    generate the plot, return the existing path."""
    pr_curve = metrics.get("curves", {}).get("pr_curve", {})
    if pr_curve.get("plot_path") and os.path.exists(pr_curve["plot_path"]):
        return pr_curve["plot_path"]
    # If raw data is available, plot it
    # (RF-DETR doesn't provide raw PR curve points by default)
    return ""


_F1_PALETTE = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#795548", "#607D8B", "#CDDC39", "#FF5722",
]


def _draw_f1_figure(data: dict, title: str, output_path: str,
                    ylim: tuple[float, float] = (0.0, 1.0)) -> str:
    """Draw a single F1-Confidence plot and save to *output_path*."""
    _setup_plot_style()
    conf = data["confidence"]
    f1_mean = data["f1_mean"]
    f1_per_class = data.get("f1_per_class", {})
    best_conf = data.get("best_conf", 0)
    best_f1 = data.get("best_f1", 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-class lines (semi-transparent)
    for i, (cls_name, cls_f1) in enumerate(f1_per_class.items()):
        color = _F1_PALETTE[i % len(_F1_PALETTE)]
        ax.plot(conf, cls_f1, color=color, alpha=0.4, linewidth=1.0, label=cls_name)

    # All-class mean line (bold)
    ax.plot(conf, f1_mean, color="#1565C0", linewidth=2.5, label="all classes", zorder=5)

    # Mark optimal threshold
    if best_conf > 0 and ylim[0] <= best_f1 <= ylim[1]:
        ax.axvline(x=best_conf, color="red", linestyle=":", alpha=0.6)
        ax.plot(best_conf, best_f1, "o", color="red", markersize=8, zorder=6)
        # Decide annotation placement based on Y range
        y_offset = -(ylim[1] - ylim[0]) * 0.1
        ax.annotate(
            f"F1={best_f1:.4f}\n@conf={best_conf:.3f}",
            xy=(best_conf, best_f1),
            xytext=(best_conf + 0.05, best_f1 + y_offset),
            fontsize=9, fontweight="bold", color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        )

    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_f1_confidence_from_data(metrics: dict, output_dir: str) -> str | dict[str, str]:
    """Plot F1-Confidence curves — each type as a separate image file.

    Generates:
      - f1_confidence_box.png         (Y 0–1)
      - f1_confidence_box_zoomed.png  (Y 0.9–1)
      - f1_confidence_mask.png        (Y 0–1, if seg)
      - f1_confidence_mask_zoomed.png (Y 0.9–1, if seg)
    """
    curves = metrics.get("curves", {})
    f1_data = curves.get("f1_confidence", {})
    mask_f1_data = curves.get("mask_f1_confidence", {})

    has_box = bool(f1_data.get("confidence") and f1_data.get("f1_mean"))
    has_mask = bool(mask_f1_data.get("confidence") and mask_f1_data.get("f1_mean"))

    if not has_box and not has_mask:
        if f1_data.get("plot_path") and os.path.exists(f1_data["plot_path"]):
            return f1_data["plot_path"]
        return ""

    results: dict[str, str] = {}

    if has_box:
        results["F1-Confidence (Box)"] = _draw_f1_figure(
            f1_data, "F1-Confidence Curve (Box)",
            os.path.join(output_dir, "f1_confidence_box.png"),
        )
        results["F1-Confidence (Box) Zoomed"] = _draw_f1_figure(
            f1_data, "F1-Confidence Curve (Box) — Zoomed 0.9–1.0",
            os.path.join(output_dir, "f1_confidence_box_zoomed.png"),
            ylim=(0.9, 1.0),
        )

    if has_mask:
        results["F1-Confidence (Mask)"] = _draw_f1_figure(
            mask_f1_data, "F1-Confidence Curve (Mask)",
            os.path.join(output_dir, "f1_confidence_mask.png"),
        )
        results["F1-Confidence (Mask) Zoomed"] = _draw_f1_figure(
            mask_f1_data, "F1-Confidence Curve (Mask) — Zoomed 0.9–1.0",
            os.path.join(output_dir, "f1_confidence_mask_zoomed.png"),
            ylim=(0.9, 1.0),
        )

    if len(results) == 1:
        return next(iter(results.values()))
    return results


def plot_map_epochs_curve(metrics: dict, output_dir: str) -> str:
    """Standalone mAP over epochs chart."""
    _setup_plot_style()
    training_log = metrics.get("training_log", {})
    epochs_data = training_log.get("epochs", [])
    if not epochs_data:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = [e.get("epoch", i) for i, e in enumerate(epochs_data)]
    has_data = False

    # RF-DETR style
    for key, label, color, ls in [
        ("mAP50_95", "mAP@50:95", "#2196F3", "-"),
        ("mAP50", "mAP@50", "#4CAF50", "-"),
        ("ema_mAP50_95", "EMA mAP@50:95", "#FF9800", "--"),
        ("ema_mAP50", "EMA mAP@50", "#9C27B0", "--"),
        ("mask_mAP50_95", "Mask mAP@50:95", "#795548", "-."),
        ("mask_mAP50", "Mask mAP@50", "#607D8B", "-."),
    ]:
        vals = [e.get(key) for e in epochs_data]
        if any(v is not None for v in vals):
            valid_epochs = [ep for ep, v in zip(epochs, vals) if v is not None]
            valid_vals = [v for v in vals if v is not None]
            ax.plot(valid_epochs, valid_vals, label=label, color=color, linewidth=1.5, linestyle=ls)
            has_data = True

    # YOLO style
    for key_pattern, label, color, ls in [
        ("metrics/mAP50(B)", "Box mAP@50", "#2196F3", "-"),
        ("metrics/mAP50-95(B)", "Box mAP@50:95", "#4CAF50", "-"),
        ("metrics/mAP50(M)", "Mask mAP@50", "#FF9800", "--"),
        ("metrics/mAP50-95(M)", "Mask mAP@50:95", "#9C27B0", "--"),
    ]:
        vals = [e.get(key_pattern) for e in epochs_data]
        if any(v is not None for v in vals):
            valid_epochs = [ep for ep, v in zip(epochs, vals) if v is not None]
            valid_vals = [v for v in vals if v is not None]
            ax.plot(valid_epochs, valid_vals, label=label, color=color, linewidth=1.5, linestyle=ls)
            has_data = True

    if not has_data:
        plt.close(fig)
        return ""

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("mAP Over Training Epochs", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.0)

    # Mark best epoch
    for key in ["mAP50_95", "ema_mAP50_95", "metrics/mAP50-95(B)"]:
        vals = [e.get(key) for e in epochs_data]
        valid = [(ep, v) for ep, v in zip(epochs, vals) if v is not None]
        if valid:
            best_ep, best_val = max(valid, key=lambda x: x[1])
            ax.axvline(x=best_ep, color="red", linestyle=":", alpha=0.5)
            ax.annotate(f"Best: {best_val:.4f}\n(epoch {best_ep})",
                       xy=(best_ep, best_val), fontsize=8,
                       arrowprops=dict(arrowstyle="->", color="red"),
                       xytext=(best_ep + len(epochs) * 0.05, best_val - 0.05))
            break

    plt.tight_layout()
    path = os.path.join(output_dir, "mAP_epochs.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
# Report generation (table, CSV, JSON, Markdown)
# ============================================================================

def print_terminal_table(metrics: dict) -> None:
    """Print formatted metrics table to terminal."""
    overall = metrics.get("overall", {})
    per_class = metrics.get("per_class", [])
    framework = metrics.get("framework", "unknown")
    model = metrics.get("model", "unknown")
    task = metrics.get("task", "detect")

    print()
    print("=" * 80)
    print(f"  Evaluation Report — {framework} / {model} ({task})")
    print("=" * 80)

    # Overall metrics
    print(f"\n{'Overall Metrics':^80}")
    print("-" * 80)
    header = f"{'Metric':<25} {'Value':>12}"
    print(header)
    print("-" * 37)
    for key, label in [
        ("mAP50_95", "mAP@50:95"),
        ("mAP50", "mAP@50"),
        ("mAP75", "mAP@75"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("mask_mAP50_95", "Mask mAP@50:95"),
        ("mask_mAP50", "Mask mAP@50"),
        ("mask_f1", "Mask F1"),
    ]:
        val = overall.get(key)
        if val is not None:
            print(f"  {label:<23} {val:>12.4f}")

    # COCO stats
    coco_stats = metrics.get("coco_stats", {})
    for eval_type in ["bbox", "masks", "segm"]:
        stats = coco_stats.get(eval_type)
        if stats:
            print(f"\n{'COCO ' + eval_type + ' Stats':^80}")
            print("-" * 80)
            for key, val in stats.items():
                if isinstance(val, (int, float)):
                    print(f"  {key:<23} {val:>12.4f}")

    # Per-class table
    if per_class:
        # Determine which columns have data
        has_prf = any(cls.get("precision") or cls.get("recall") or cls.get("f1") for cls in per_class)

        print(f"\n{'Per-Class Metrics':^80}")
        print("-" * 80)
        if has_prf:
            header = f"  {'Class':<20} {'mAP50:95':>10} {'mAP50':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}"
        else:
            header = f"  {'Class':<20} {'mAP50:95':>10} {'mAP50':>8}"
        print(header)
        print("  " + "-" * (66 if has_prf else 40))

        def _fmt(val):
            return f"{val:>8.4f}" if val else f"{'—':>8}"

        for cls in per_class:
            name = cls.get("name", "?")[:20]
            line = (f"  {name:<20} "
                    f"{cls.get('mAP50_95', 0):>10.4f} "
                    f"{cls.get('mAP50', 0):>8.4f}")
            if has_prf:
                line += (f" {_fmt(cls.get('precision'))}"
                         f" {_fmt(cls.get('recall'))}"
                         f" {_fmt(cls.get('f1'))}")
            print(line)
        print("  " + "-" * (66 if has_prf else 40))
        # Summary row
        if len(per_class) > 1:
            avg = lambda k: sum(c.get(k, 0) or 0 for c in per_class) / len(per_class)
            line = (f"  {'AVERAGE':<20} "
                    f"{avg('mAP50_95'):>10.4f} "
                    f"{avg('mAP50'):>8.4f}")
            if has_prf:
                line += (f" {avg('precision'):>8.4f}"
                         f" {avg('recall'):>8.4f}"
                         f" {avg('f1'):>8.4f}")
            print(line)

    print()
    print("=" * 80)


def save_csv(metrics: dict, output_dir: str) -> str:
    """Save metrics as CSV."""
    path = os.path.join(output_dir, "eval_metrics.csv")
    per_class = metrics.get("per_class", [])
    overall = metrics.get("overall", {})

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "mAP50_95", "mAP50", "precision", "recall", "f1"])
        for cls in per_class:
            writer.writerow([
                cls.get("name", ""),
                f"{cls.get('mAP50_95', 0):.6f}",
                f"{cls.get('mAP50', 0):.6f}",
                f"{cls.get('precision', 0):.6f}",
                f"{cls.get('recall', 0):.6f}",
                f"{cls.get('f1', 0):.6f}",
            ])
        # Overall row
        writer.writerow([
            "OVERALL",
            f"{overall.get('mAP50_95', 0):.6f}",
            f"{overall.get('mAP50', 0):.6f}",
            f"{overall.get('precision', 0):.6f}",
            f"{overall.get('recall', 0):.6f}",
            f"{overall.get('f1', 0):.6f}",
        ])

    return path


def save_markdown(metrics: dict, output_dir: str, plot_paths: dict) -> str:
    """Generate Markdown evaluation report."""
    path = os.path.join(output_dir, "eval_report.md")
    overall = metrics.get("overall", {})
    per_class = metrics.get("per_class", [])
    framework = metrics.get("framework", "unknown")
    model = metrics.get("model", "unknown")
    task = metrics.get("task", "detect")

    lines = [
        f"# Evaluation Report",
        f"",
        f"- **Framework**: {framework}",
        f"- **Model**: {model}",
        f"- **Task**: {task}",
        f"- **Weights**: `{metrics.get('weights', 'N/A')}`",
        f"- **Timestamp**: {metrics.get('timestamp', 'N/A')}",
        f"",
        f"## Overall Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]

    for key, label in [
        ("mAP50_95", "mAP@50:95"),
        ("mAP50", "mAP@50"),
        ("mAP75", "mAP@75"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("mask_mAP50_95", "Mask mAP@50:95"),
        ("mask_mAP50", "Mask mAP@50"),
        ("mask_f1", "Mask F1"),
    ]:
        val = overall.get(key)
        if val is not None:
            lines.append(f"| {label} | {val:.4f} |")

    # Per-class table
    if per_class:
        lines += [
            "",
            "## Per-Class Metrics",
            "",
            "| Class | mAP@50:95 | mAP@50 | Precision | Recall | F1 |",
            "|-------|-----------|--------|-----------|--------|-----|",
        ]
        for cls in per_class:
            lines.append(
                f"| {cls.get('name', '?')} "
                f"| {cls.get('mAP50_95', 0):.4f} "
                f"| {cls.get('mAP50', 0):.4f} "
                f"| {cls.get('precision', 0):.4f} "
                f"| {cls.get('recall', 0):.4f} "
                f"| {cls.get('f1', 0):.4f} |"
            )

    # COCO stats
    coco_stats = metrics.get("coco_stats", {})
    for eval_type in ["bbox", "masks", "segm"]:
        stats = coco_stats.get(eval_type)
        if stats:
            lines += [
                "",
                f"## COCO {eval_type.title()} Stats",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    lines.append(f"| {k} | {v:.4f} |")

    # Plot images
    if plot_paths:
        lines += ["", "## Visualizations", ""]
        for name, p in plot_paths.items():
            if p and os.path.exists(p):
                rel_path = os.path.relpath(p, output_dir)
                lines.append(f"### {name}")
                lines.append(f"![{name}]({rel_path})")
                lines.append("")

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    return path


# ============================================================================
# Main entry point
# ============================================================================

def generate_report(metrics_json_path: str, output_dir: str | None = None) -> dict[str, str]:
    """Generate full evaluation report from a unified metrics JSON file.

    Args:
        metrics_json_path: Path to eval_metrics.json produced by an eval script.
        output_dir: Override output directory (default: same as metrics file).

    Returns:
        Dict mapping output type names to file paths.
    """
    with open(metrics_json_path) as f:
        metrics = json.load(f)

    if not output_dir:
        output_dir = str(Path(metrics_json_path).parent)

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    outputs = {}

    # 1. Terminal table
    print_terminal_table(metrics)

    # 2. Generate plots
    plot_funcs = {
        "Per-Class Metrics": plot_per_class_bar,
        "Training Loss": plot_training_loss_curve,
        "Overall Summary": plot_overall_summary,
        "mAP Over Epochs": plot_map_epochs_curve,
        "Confusion Matrix": plot_confusion_matrix_from_data,
        "PR Curve": plot_pr_curve_from_data,
        "F1-Confidence Curve": plot_f1_confidence_from_data,
    }

    plot_paths = {}
    for name, func in plot_funcs.items():
        try:
            result = func(metrics, plots_dir)
            if isinstance(result, dict):
                # Function returned multiple files
                for sub_name, sub_path in result.items():
                    if sub_path:
                        plot_paths[sub_name] = sub_path
                        outputs[f"plot_{sub_name}"] = sub_path
                        print(f"  [Plot] {sub_name}: {sub_path}")
            elif result:
                plot_paths[name] = result
                outputs[f"plot_{name}"] = result
                print(f"  [Plot] {name}: {result}")
        except Exception as e:
            print(f"  [Plot] {name}: skipped ({e})")

    # Also note any framework-generated plots
    for p in metrics.get("plots_generated", []):
        if os.path.exists(p):
            basename = Path(p).stem
            if basename not in plot_paths:
                plot_paths[basename] = p

    # Check for framework-generated curve plots
    for key in ["confusion_matrix_path", "confusion_matrix_normalized_path"]:
        p = metrics.get("curves", {}).get(key)
        if p and os.path.exists(p):
            plot_paths[key] = p

    # 3. Save JSON (already exists, but copy to output_dir if different)
    json_path = os.path.join(output_dir, "eval_metrics.json")
    if os.path.abspath(json_path) != os.path.abspath(metrics_json_path):
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    outputs["json"] = json_path

    # 4. Save CSV
    csv_path = save_csv(metrics, output_dir)
    outputs["csv"] = csv_path
    print(f"  [CSV] {csv_path}")

    # 5. Save Markdown
    md_path = save_markdown(metrics, output_dir, plot_paths)
    outputs["markdown"] = md_path
    print(f"  [Markdown] {md_path}")

    print(f"\n  All outputs saved to: {output_dir}")
    return outputs
