# file: gpupath/cuda.py

from __future__ import annotations

from typing import Any
from gpupath import _native


def cuda_available() -> bool:
    """
    Return whether a CUDA-capable runtime/device is available to gpupath.
    """
    return bool(_native.cuda_available())


def cuda_info() -> dict[str, Any]:
    """
    Return structured CUDA bootstrap information from the native backend.

    Returned keys currently include:
    - cuda_available
    - status_code
    - status_name
    - status_message
    - device_count
    - runtime_version
    - driver_version
    - primary_device
    """
    info = _native.cuda_info()
    if not isinstance(info, dict):
        raise TypeError("Native CUDA info response must be a dictionary.")
    return info
