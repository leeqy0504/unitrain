"""Unified runners for different frameworks."""

from .base import BaseRunner
from .ultralytics import UltralyticsRunner
from .rfdetr import RFDETRRunner


def get_runner(framework: str) -> BaseRunner:
    """Factory function to get the appropriate runner."""
    runners = {
        "ultralytics": UltralyticsRunner,
        "yolo": UltralyticsRunner,
        "rfdetr": RFDETRRunner,
        "rf-detr": RFDETRRunner,
    }
    runner_cls = runners.get(framework.lower())
    if not runner_cls:
        raise ValueError(f"Unknown framework: {framework}. Supported: {list(runners.keys())}")
    return runner_cls()


__all__ = [
    "BaseRunner",
    "UltralyticsRunner",
    "RFDETRRunner",
    "get_runner",
]
