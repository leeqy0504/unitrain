#!/usr/bin/env python3
"""RF-DETR training script — executed in .venv-rfdetr."""

import argparse
import importlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Allow sibling-module import inside _scripts/
sys.path.insert(0, str(Path(__file__).parent))
from weight_utils import resolve_rfdetr_weight  # noqa: E402

# ---------------------------------------------------------------------------
# DDP fix: RF-DETR seg 模型的 segmentation_head 参数在 DDP 反向传播中
# 会被标记 ready 两次，触发 "Parameter marked as ready twice" 错误。
# static_graph=True 告诉 DDP 计算图在各次迭代间不变，从而跳过该检查。
# ---------------------------------------------------------------------------
import torch
import torch.nn.parallel
import torch.nn as nn

_original_ddp_init = torch.nn.parallel.DistributedDataParallel.__init__

def _patched_ddp_init(self, module, *args, **kwargs):
    kwargs.setdefault("find_unused_parameters", True)
    kwargs.setdefault("static_graph", True)
    _original_ddp_init(self, module, *args, **kwargs)

torch.nn.parallel.DistributedDataParallel.__init__ = _patched_ddp_init


# ---------------------------------------------------------------------------
# DDP state_dict compatibility fix: 处理 module. 前缀不匹配问题
# 在 DDP 模式下加载 checkpoint 时，需要确保 keys 匹配
# ---------------------------------------------------------------------------
_original_load_state_dict = nn.Module.load_state_dict

def _patched_load_state_dict(self, state_dict, strict=True, assign=False):
    """
    智能处理 DDP state_dict 的 module. 前缀：
    - 如果模型是 DDP 但 state_dict 没有 module. 前缀，添加前缀
    - 如果模型不是 DDP 但 state_dict 有 module. 前缀，移除前缀
    """
    is_ddp_model = isinstance(self, torch.nn.parallel.DistributedDataParallel)
    state_dict_has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    
    # 如果前缀不匹配，进行转换
    if is_ddp_model and not state_dict_has_module_prefix:
        # 模型是 DDP，但 state_dict 没有 module. 前缀 → 添加前缀
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif not is_ddp_model and state_dict_has_module_prefix:
        # 模型不是 DDP，但 state_dict 有 module. 前缀 → 移除前缀
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
    return _original_load_state_dict(self, state_dict, strict=strict, assign=assign)

nn.Module.load_state_dict = _patched_load_state_dict
# ---------------------------------------------------------------------------


def get_timestamped_output_dir(base_dir: str = "outputs", prefix: str = "output") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_dir}/{prefix}_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config_json)

    model_cls_name = cfg["model_cls"]
    data_path = cfg.get("data_path", "data/")
    epochs = cfg.get("epochs", 100)
    batch = cfg.get("batch", 4)
    lr = cfg.get("lr", 1e-4)
    grad_accum = cfg.get("grad_accum_steps", 4)
    device = cfg.get("device", 0)
    output_dir = cfg.get("output_dir", "outputs")
    early_stopping = cfg.get("early_stopping", False)
    early_stopping_patience = cfg.get("early_stopping_patience", 10)
    early_stopping_min_delta = cfg.get("early_stopping_min_delta", 0.001)
    resume = cfg.get("resume", "")
    default_weights = cfg.get("default_pretrain_weights", "")
    config_file = cfg.get("config_file", "")

    # Device setup
    # When launched via torchrun (multi-GPU DDP), RANK/LOCAL_RANK/WORLD_SIZE
    # are set by torchrun and CUDA_VISIBLE_DEVICES is set by the runner.
    # In that case we must NOT override CUDA_VISIBLE_DEVICES here.
    is_distributed = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if local_rank == 0:
            print(f"DDP mode: {world_size} processes, "
                  f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    elif isinstance(device, str) and device.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU device")
    elif isinstance(device, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        print(f"Using GPU device: {device}")
    elif isinstance(device, str):
        # Single-process launch with explicit device string (e.g. "2")
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f"Using GPU device(s): {device}")

    # Dynamic import of the model class from rfdetr
    import rfdetr
    model_cls = getattr(rfdetr, model_cls_name)

    # Timestamped output
    output_dir = get_timestamped_output_dir(output_dir, "rfdetr")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Training output: {output_dir}")

    # Copy config file to output directory for reproducibility
    if config_file and os.path.exists(config_file):
        config_dest = Path(output_dir) / Path(config_file).name
        shutil.copy2(config_file, config_dest)
        if not is_distributed or (is_distributed and local_rank == 0):
            print(f"Config file copied to: {config_dest}")

    # Resolve pretrained weights into weights/ directory
    model_kwargs = {}
    if default_weights:
        resolved = resolve_rfdetr_weight(default_weights)
        if resolved:
            model_kwargs["pretrain_weights"] = resolved

    try:
        model = model_cls(**model_kwargs)
    except RuntimeError as e:
        if "PytorchStreamReader" in str(e) or "failed finding central directory" in str(e):
            # 权重文件损坏，清理后提示用户
            weight_path = model_kwargs.get("pretrain_weights", "")
            if weight_path and os.path.exists(weight_path):
                os.remove(weight_path)
            print(
                f"\n{'='*60}\n"
                f"❌ 预训练权重文件损坏，无法加载\n"
                f"   文件: {weight_path}\n"
                f"   已自动删除损坏文件，请重新运行训练命令\n"
                f"   框架将重新下载权重\n"
                f"{'='*60}\n"
            )
        raise

    train_kwargs = dict(
        dataset_dir=data_path,
        epochs=epochs,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        lr=lr,
        output_dir=output_dir,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )
    if resume:
        train_kwargs["resume"] = resume

    model.train(**train_kwargs)

    # Emit structured markers for auto-eval discovery
    print(f"UNITRAIN_TRAIN_OUTPUT_DIR={output_dir}")
    # Find best weights in output directory
    for candidate in [
        Path(output_dir) / "checkpoint_best_total.pth",
        Path(output_dir) / "checkpoint_best_ema.pth",
        Path(output_dir) / "checkpoint_best_regular.pth",
        Path(output_dir) / "checkpoint.pth",
    ]:
        if candidate.exists():
            print(f"UNITRAIN_BEST_WEIGHTS={candidate}")
            break


if __name__ == "__main__":
    main()
