#!/usr/bin/env python3
"""Unified training entry point."""

import argparse
import subprocess
import sys
from pathlib import Path

from unitrain import get_runner, load_config
from unitrain.data_converter import convert_coco_dataset


def check_gpu_memory(devices: str, threshold: float = 0.2) -> dict:
    """检查指定显卡的内存占用情况。
    
    Args:
        devices: 设备字符串，如 "0", "2,3", "cuda:0"
        threshold: 内存占用阈值（0-1），超过则警告
        
    Returns:
        dict: {gpu_id: {"used": MB, "total": MB, "percent": float, "over_threshold": bool}}
    """
    # 解析设备 ID
    device_str = devices.replace("cuda:", "").replace(" ", "")
    if not device_str or device_str == "cpu":
        return {}
    
    gpu_ids = [int(x) for x in device_str.split(",") if x.isdigit()]
    if not gpu_ids:
        return {}
    
    result = {}
    try:
        # 查询显卡内存
        output = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], text=True)
        
        for line in output.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                gpu_id = int(parts[0])
                if gpu_id in gpu_ids:
                    used = float(parts[1])
                    total = float(parts[2])
                    percent = used / total if total > 0 else 0
                    result[gpu_id] = {
                        "used": used,
                        "total": total,
                        "percent": percent,
                        "over_threshold": percent > threshold
                    }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    
    return result


def prompt_clear_gpu_cache(gpu_status: dict) -> bool:
    """提示用户是否清理显卡缓存。
    
    Returns:
        bool: 是否继续训练
    """
    over_threshold = {k: v for k, v in gpu_status.items() if v["over_threshold"]}
    if not over_threshold:
        return True
    
    print("\n" + "=" * 60)
    print("⚠️  检测到以下显卡内存占用超过 20%:")
    print("-" * 60)
    for gpu_id, info in over_threshold.items():
        print(f"  GPU {gpu_id}: {info['used']:.0f} / {info['total']:.0f} MiB ({info['percent']*100:.1f}%)")
    print("-" * 60)
    
    try:
        response = input("是否清理显卡缓存后继续? [Y/n/q] (Y=清理, n=跳过, q=退出): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消")
        return False
    
    if response == 'q':
        print("已退出")
        return False
    elif response == 'n':
        print("跳过清理，继续训练...")
        return True
    else:
        # 清理缓存
        print("正在清理显卡缓存...")
        try:
            # 尝试使用 PyTorch 清理
            import torch
            for gpu_id in over_threshold.keys():
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            print("✅ PyTorch 缓存已清理")
            
            # 查找并终止占用显存的进程（仅当前用户的）
            import os
            current_user = os.getenv("USER", "")
            output = subprocess.check_output([
                "nvidia-smi", 
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
                "-i", ",".join(str(g) for g in over_threshold.keys())
            ], text=True)
            
            orphan_pids = []
            for line in output.strip().split("\n"):
                if line.strip():
                    parts = line.split(",")
                    if len(parts) >= 1:
                        pid = parts[0].strip()
                        if pid.isdigit():
                            orphan_pids.append(int(pid))
            
            if orphan_pids:
                print(f"  发现 {len(orphan_pids)} 个占用显存的进程: {orphan_pids}")
                kill_response = input("  是否终止这些进程? [y/N]: ").strip().lower()
                if kill_response == 'y':
                    for pid in orphan_pids:
                        try:
                            os.kill(pid, 9)
                            print(f"  ✅ 已终止进程 {pid}")
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            print(f"  ❌ 无权限终止进程 {pid}")
        except ImportError:
            print("  PyTorch 未安装，跳过缓存清理")
        except Exception as e:
            print(f"  清理时出错: {e}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Unified DL Training")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML file")
    parser.add_argument("--convert-data", action="store_true", help="Convert COCO to YOLO format if needed")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU memory check")
    parser.add_argument("--skip-eval", action="store_true", help="Skip auto-evaluation after training")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 检查显卡内存占用
    if not args.skip_gpu_check:
        devices = getattr(config, "device", "") or ""
        gpu_status = check_gpu_memory(str(devices), threshold=0.2)
        if gpu_status:
            if not prompt_clear_gpu_cache(gpu_status):
                sys.exit(1)
    
    print(f">>> Training with {config.framework} / {config.model}")

    # Auto convert data for ultralytics if needed
    if config.framework in ("ultralytics", "yolo") and config.data_format == "coco":
        task = getattr(config, "task", "detect")

        # OBB task: data is already in YOLO OBB format, skip COCO→YOLO conversion
        if task == "obb":
            print(f">>> OBB task: using YOLO OBB data directly from {config.data_path}")
            data_yaml = Path(config.data_path).expanduser() / "data.yaml"
            if not data_yaml.exists():
                config.to_ultralytics_yaml(data_yaml)
                print(f">>> Generated data.yaml: {data_yaml}")
        else:
            data_path = Path(config.data_path).expanduser()
            yolo_data_path = data_path.parent / f"{data_path.name}_yolo"
            if args.convert_data or not yolo_data_path.exists():
                print(f">>> Converting COCO to YOLO format: {yolo_data_path}")
                class_info = convert_coco_dataset(data_path, yolo_data_path, task=task)

                # Update config with extracted class info from COCO
                if class_info:
                    config.num_classes = class_info["nc"]
                    config.class_names = class_info["names"]
                    print(f">>> Detected {config.num_classes} classes from dataset")

            # supervision's as_yolo already generates data.yaml
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

    # Pass config file path for auto-copying to output directory
    cfg_dict["config_file"] = str(Path(args.config).resolve())

    train_info = runner.train(cfg_dict)
    print(">>> Training complete!")

    # ------------------------------------------------------------------
    # Auto-evaluation after training
    # ------------------------------------------------------------------
    if not args.skip_eval:
        _auto_eval(runner, cfg_dict, config, train_info)


def _auto_eval(runner, cfg_dict: dict, config, train_info: dict | None) -> None:
    """Run evaluation automatically after training using best weights.

    Uses the structured markers emitted by training scripts to locate the
    training output directory and best weights file, then calls
    ``runner.eval()`` followed by ``generate_report()``.
    """
    from unitrain import generate_report

    if not train_info:
        print(">>> Skipping auto-eval: could not determine training output directory")
        return

    train_output_dir = train_info.get("output_dir", "")
    best_weights = train_info.get("best_weights", "")

    if not train_output_dir or not best_weights:
        print(">>> Skipping auto-eval: training output dir or best weights not found")
        if train_output_dir:
            print(f"    output_dir = {train_output_dir}")
        if best_weights:
            print(f"    best_weights = {best_weights}")
        return

    if not Path(best_weights).exists():
        print(f">>> Skipping auto-eval: best weights file not found: {best_weights}")
        return

    eval_output_dir = str(Path(train_output_dir) / "eval")
    print(f"\n>>> Auto-evaluation: using weights {best_weights}")
    print(f">>> Eval output dir: {eval_output_dir}")

    # Build eval config by injecting eval section into cfg_dict
    eval_cfg = dict(cfg_dict)
    eval_cfg["eval"] = {
        "weights": best_weights,
        "output_dir": eval_output_dir,
        "conf_threshold": 0.001,
        "iou_threshold": 0.5,
    }

    try:
        eval_result = runner.eval(eval_cfg)
        metrics_json = eval_result.get("metrics_json", "")
        if metrics_json and Path(metrics_json).exists():
            print(f"\n>>> Generating evaluation report...")
            report_paths = generate_report(metrics_json, eval_output_dir)
            print(f">>> Evaluation report saved to: {eval_output_dir}")
            for name, path in report_paths.items():
                print(f"    {name}: {path}")
        else:
            print(f">>> Eval completed but metrics JSON not found at: {metrics_json}")
    except Exception as e:
        print(f">>> Auto-evaluation failed: {e}")
        print(">>> Training was successful. You can run eval manually with:")
        print(f"    ./run.sh eval --config <config.yaml> --weights {best_weights}")


if __name__ == "__main__":
    main()
