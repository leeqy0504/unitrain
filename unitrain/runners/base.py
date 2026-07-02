"""Base runner interface for framework runners."""

import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..utils import get_timestamped_output_dir

PROJECT_ROOT = Path(__file__).parent.parent.parent
_SCRIPTS_DIR = Path(__file__).parent / "_scripts"


def _parse_device_ids(device) -> list[str]:
    """Parse a device specification into a list of GPU id strings.

    Examples:
        0          → ["0"]
        "2,3"      → ["2", "3"]
        "cpu"      → []
    """
    if isinstance(device, int):
        return [str(device)]
    if isinstance(device, str):
        if device.lower() == "cpu":
            return []
        return [d.strip() for d in device.split(",") if d.strip()]
    return ["0"]


class BaseRunner(ABC):
    """Abstract base class for framework runners."""

    def __init__(self, venv_path: Path):
        self.venv_path = venv_path
        self.python = venv_path / "bin" / "python"

    def _run_script_file(
        self,
        script_name: str,
        config: dict[str, Any],
        *,
        stream_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Run a standalone Python script in the framework's virtual environment.

        The config dict is passed as a JSON string via ``--config-json`` argument.

        Args:
            script_name: Filename inside ``_scripts/`` directory (e.g. "rfdetr_train.py").
            config: Configuration dict serialised as JSON for the script.
            stream_output: Whether to stream stdout in real-time.
        """
        script_path = _SCRIPTS_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        config_json = json.dumps(config, ensure_ascii=False)
        cmd = [str(self.python), "-u", str(script_path), "--config-json", config_json]

        import os
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        if stream_output:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
                bufsize=1,
                env=env,
            )
            output_lines: list[str] = []
            for line in process.stdout:
                print(line, end="", flush=True)
                output_lines.append(line)
            process.wait()
            return subprocess.CompletedProcess(
                args=process.args,
                returncode=process.returncode,
                stdout="".join(output_lines),
                stderr="",
            )
        else:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env=env,
            )

    @staticmethod
    def _parse_train_markers(stdout: str) -> dict[str, str]:
        """Parse structured UNITRAIN_* markers from training script stdout.

        Returns dict with keys ``output_dir`` and optionally ``best_weights``.
        """
        info: dict[str, str] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("UNITRAIN_TRAIN_OUTPUT_DIR="):
                info["output_dir"] = line.split("=", 1)[1]
            elif line.startswith("UNITRAIN_BEST_WEIGHTS="):
                info["best_weights"] = line.split("=", 1)[1]
        return info

    @abstractmethod
    def train(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Train a model with the given config.

        Returns a dict with at least ``output_dir`` and optionally
        ``best_weights`` paths, or *None* on failure.
        """
        pass

    @abstractmethod
    def predict(self, config: dict[str, Any], source: str) -> Any:
        """Run inference on the given source."""
        pass

    @abstractmethod
    def export(self, config: dict[str, Any], format: str) -> Path:
        """Export model to the given format."""
        pass

    @abstractmethod
    def eval(self, config: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a trained model and return metrics dict."""
        pass

    # ------------------------------------------------------------------
    # Multi-GPU (DDP via torchrun)
    # ------------------------------------------------------------------

    def _run_script_file_distributed(
        self,
        script_name: str,
        config: dict[str, Any],
        device_ids: list[str],
        *,
        stream_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Launch a training script via ``torchrun`` for multi-GPU DDP.

        ``CUDA_VISIBLE_DEVICES`` is set to *device_ids* so that torch sees
        exactly those GPUs (remapped to 0 … N-1).  ``torchrun`` spawns one
        process per GPU with ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``
        environment variables that RF-DETR's ``init_distributed_mode`` picks
        up automatically.
        """
        script_path = _SCRIPTS_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        nproc = len(device_ids)
        config_json = json.dumps(config, ensure_ascii=False)

        # Use torchrun from the framework venv
        torchrun = self.venv_path / "bin" / "torchrun"
        if not torchrun.exists():
            # Fallback: python -m torch.distributed.run
            cmd = [
                str(self.python), "-m", "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                str(script_path),
                "--config-json", config_json,
            ]
        else:
            cmd = [
                str(torchrun),
                f"--nproc_per_node={nproc}",
                str(script_path),
                "--config-json", config_json,
            ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)

        print(f"[DDP] Launching with torchrun: nproc_per_node={nproc}, "
              f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

        if stream_output:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
                bufsize=1,
                env=env,
            )
            output_lines: list[str] = []
            for line in process.stdout:
                print(line, end="", flush=True)
                output_lines.append(line)
            process.wait()
            return subprocess.CompletedProcess(
                args=process.args,
                returncode=process.returncode,
                stdout="".join(output_lines),
                stderr="",
            )
        else:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env=env,
            )
