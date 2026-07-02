"""UniTrain - Common utilities shared across all framework runners."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any


def get_timestamped_output_dir(base_dir: str = "outputs", prefix: str = "output") -> str:
    """
    Generate an output directory path with timestamp.
    
    Args:
        base_dir: Base output directory
        prefix: Prefix for the output folder name
        
    Returns:
        Path string like "outputs/output_20260203_143025"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_dir}/{prefix}_{timestamp}"


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if not."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_latest_checkpoint(output_dir: str | Path, pattern: str = "checkpoint*.pth") -> Path | None:
    """
    Find the latest checkpoint file in a directory.
    
    Args:
        output_dir: Directory to search
        pattern: Glob pattern for checkpoint files
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    from pathlib import Path
    import glob
    
    checkpoints = list(Path(output_dir).glob(pattern))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def find_best_weights(output_dir: str | Path, patterns: list[str] = None) -> Path | None:
    """
    Find the best model weights file in an output directory.
    
    Searches for common weight file patterns used by different frameworks.
    
    Args:
        output_dir: Directory to search
        patterns: Custom patterns to search (default: common patterns)
        
    Returns:
        Path to best weights or None if not found
    """
    if patterns is None:
        patterns = ["best*.pt", "best*.pth", "model_best*.pt", "model_best*.pth"]
    
    output_path = Path(output_dir)
    for pattern in patterns:
        matches = list(output_path.glob(f"**/{pattern}"))
        if matches:
            return matches[0]
    return None


def format_detection_result(
    box: list[float],
    score: float,
    class_id: int,
    class_names: list[str] | None = None
) -> str:
    """
    Format a single detection result for display.
    
    Args:
        box: Bounding box coordinates [x1, y1, x2, y2]
        score: Confidence score
        class_id: Class ID
        class_names: Optional list of class names
        
    Returns:
        Formatted string like "person 0.95 [100.0, 200.0, 300.0, 400.0]"
    """
    if class_names and class_id < len(class_names):
        name = class_names[class_id]
    else:
        name = f"class_{class_id}"
    return f"{name} {score:.3f} {box}"


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two config dictionaries.
    
    Args:
        base: Base configuration
        override: Override values (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_device_str(device: int | str | list[int]) -> str:
    """
    Convert device specification to string format.
    
    Args:
        device: Device ID (0), string ("cuda:0", "cpu"), or list ([0, 1])
        
    Returns:
        Device string suitable for framework use
    """
    if isinstance(device, list):
        return ",".join(str(d) for d in device)
    return str(device)
