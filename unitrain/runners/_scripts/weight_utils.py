"""Shared weight file utilities for framework scripts.

All pretrained weight files are stored in the ``weights/`` directory
under the project root.  The helpers here transparently resolve a bare
weight filename (e.g. ``rf-detr-base.pth``) to the canonical path
``weights/rf-detr-base.pth``, moving or downloading the file as needed.
"""

import os
import shutil
import zipfile

WEIGHTS_DIR = "weights"


def validate_weight_file(path: str) -> bool:
    """Check whether a weight file is a valid PyTorch checkpoint.

    PyTorch ``.pt`` / ``.pth`` files are ZIP archives internally.
    Returns ``True`` if the file looks valid, ``False`` otherwise.
    """
    if not path or not os.path.exists(path):
        return False
    try:
        size = os.path.getsize(path)
        if size < 256:  # too small to be a real checkpoint
            return False
        with open(path, "rb") as f:
            magic = f.read(2)
            if magic != b"PK":  # ZIP magic bytes
                return False
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                return False
    except Exception:
        return False
    return True


def resolve_weight_path(weight_name: str) -> str:
    """Return the path to a weight file inside ``WEIGHTS_DIR``.

    Resolution order:
    1. If *weight_name* is empty / ``None`` → return ``""``.
    2. If it is already an absolute path or starts with ``weights/`` → return as-is.
    3. If the file exists in ``weights/`` → return that path.
    4. If the file exists in the project root (CWD) → **move** it into ``weights/``.
    5. Otherwise return the ``weights/`` target path (the caller / framework
       can still attempt to download it there).
    """
    if not weight_name:
        return ""

    # Already resolved
    if os.path.isabs(weight_name) or weight_name.startswith(f"{WEIGHTS_DIR}/") or weight_name.startswith(f"{WEIGHTS_DIR}{os.sep}"):
        return weight_name

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    target = os.path.join(WEIGHTS_DIR, weight_name)

    if os.path.exists(target):
        if not validate_weight_file(target):
            print(
                f"\n[weights] ⚠️  权重文件损坏，已删除: {target}\n"
                f"[weights]    可能原因: 下载中断或磁盘写入错误\n"
                f"[weights]    框架将尝试重新下载\n"
            )
            os.remove(target)
        else:
            return target

    # Legacy: weight sitting in project root → move it
    if os.path.exists(weight_name):
        shutil.move(weight_name, target)
        print(f"[weights] Moved {weight_name} → {target}")
        return target

    # Not found locally – return target path; framework may download later.
    return target


def resolve_rfdetr_weight(weight_name: str) -> str:
    """Resolve an RF-DETR weight file, downloading if necessary.

    Wraps :func:`resolve_weight_path` and, when the file is still missing,
    attempts to download it via the RF-DETR hosted-model registry.
    """
    target = resolve_weight_path(weight_name)
    if not target or os.path.exists(target):
        return target

    # Attempt download through RF-DETR utilities
    try:
        from rfdetr.main import HOSTED_MODELS  # noqa: WPS433
        from rfdetr.util.files import download_file  # noqa: WPS433

        # The HOSTED_MODELS keys are bare filenames like "rf-detr-base.pth"
        bare_name = os.path.basename(target)
        if bare_name in HOSTED_MODELS:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            print(f"[weights] Downloading {bare_name} → {target} …")
            download_file(HOSTED_MODELS[bare_name], target)

            # Validate downloaded file
            if os.path.exists(target) and not validate_weight_file(target):
                print(
                    f"\n[weights] ❌ 下载的权重文件无效: {target}\n"
                    f"[weights]    文件可能不完整或下载链接已失效\n"
                    f"[weights]    请手动下载并放入 {WEIGHTS_DIR}/ 目录\n"
                )
                os.remove(target)
    except ImportError:
        pass

    return target
