"""
file: gpupath/engine/__init__.py

Engine backends for gpupath.

This package defines the common engine interface and concrete backend
implementations. New backends, such as a future CUDA engine, should live
here and implement :class:`gpupath.engine.base.PathEngine`.
"""

from gpupath.engine.base import PathEngine
from gpupath.engine.cpu import CpuPathEngine

__all__ = ["PathEngine", "CpuPathEngine"]
