"""AnyEG's GPU modules: the interface and its two implementations."""

from .base import ExtremumGraphKernels, KernelOutput
from .original import OriginalCudaKernels
from .torch_port import TorchKernels


def make_kernels(name: str = "torch", **kwargs) -> ExtremumGraphKernels:
    """``"torch"`` (PyTorch port, GPU if available), ``"torch:cpu"``, or ``"original"``."""
    if name == "torch":
        return TorchKernels(**kwargs)
    if name.startswith("torch:"):
        return TorchKernels(device=name.split(":", 1)[1], **kwargs)
    if name in ("original", "original-cuda"):
        return OriginalCudaKernels(**kwargs)
    raise ValueError(f"unknown kernels {name!r}; use 'torch', 'torch:cpu' or 'original'")


__all__ = ["ExtremumGraphKernels", "KernelOutput", "OriginalCudaKernels", "TorchKernels", "make_kernels"]
