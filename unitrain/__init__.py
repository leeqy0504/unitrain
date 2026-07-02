"""UniTrain - Universal Training Framework for deep learning models."""

from .runners import RFDETRRunner, UltralyticsRunner, get_runner
from .config_gen import UnifiedConfig, load_config
from .utils import (
    get_timestamped_output_dir,
    ensure_dir,
    find_latest_checkpoint,
    find_best_weights,
    format_detection_result,
    merge_configs,
)


def generate_report(metrics_json_path: str, output_dir: str | None = None):
    """Lazy wrapper — imports eval_report only when called (needs matplotlib)."""
    from .eval_report import generate_report as _gen
    return _gen(metrics_json_path, output_dir)


__all__ = [
    "RFDETRRunner",
    "UltralyticsRunner",
    "get_runner",
    "UnifiedConfig",
    "load_config",
    # Utils
    "get_timestamped_output_dir",
    "ensure_dir",
    "find_latest_checkpoint",
    "find_best_weights",
    "format_detection_result",
    "merge_configs",
    # Eval
    "generate_report",
]
