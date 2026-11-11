"""
The ORIGINAL AnyEG CUDA code, run through ``anyeg/native/anyeg_driver``.

This is the reference the PyTorch port is checked against. The driver calls
AnyEG's unchanged ``compute()`` (built by ``anyeg/native/build.sh``). Each run
is a separate process because ``compute()`` keeps its state in global
variables (a second call in one process would write past an array).

Exchange format (see ``anyeg_driver.cu``): the values and neighbour lists go
in as a binary file; AnyEG's result buffer comes back unchanged and is
unpacked here exactly as ``ttkExtremeGraphDev::RequestData`` unpacks it.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..neighborhood import NeighborTable
from .base import ExtremumGraphKernels, KernelOutput

DEFAULT_DRIVER = Path(__file__).resolve().parent.parent / "native" / "build" / "anyeg_driver"
_MAGIC = b"AEG1"


class OriginalCudaKernels(ExtremumGraphKernels):
    """Runs AnyEG's own ``compute()`` on the GPU."""

    name = "original-cuda"

    def __init__(self, driver: Union[str, Path] = DEFAULT_DRIVER, gpu: Optional[int] = None,
                 timeout: float = 3600.0):
        """``gpu``: CUDA device index (default: the one with the most free memory)."""
        self.driver = Path(driver)
        self.gpu = gpu
        self.timeout = timeout
        self.last_log = ""

    def __repr__(self) -> str:
        return f"OriginalCudaKernels(gpu={self.gpu})"

    @classmethod
    def is_built(cls, driver: Union[str, Path] = DEFAULT_DRIVER) -> bool:
        return Path(driver).is_file() and os.access(driver, os.X_OK)

    def run(self, values: np.ndarray, neighbors: NeighborTable) -> KernelOutput:
        if not self.is_built(self.driver):
            raise FileNotFoundError(f"{self.driver} not found; build it with anyeg/native/build.sh")
        values = np.ascontiguousarray(values, dtype=np.float32)
        n = values.shape[0]

        with tempfile.TemporaryDirectory(prefix="anyeg_") as tmp:
            src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            self._write_input(src, values, neighbors)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self._gpu_index()))
            t0 = time.perf_counter()
            proc = subprocess.run([str(self.driver), str(src), str(dst)], env=env,
                                  capture_output=True, timeout=self.timeout)
            wall = time.perf_counter() - t0
            self.last_log = proc.stdout.decode(errors="replace")
            if proc.returncode != 0:
                raise RuntimeError(f"AnyEG driver failed ({proc.returncode}): "
                                   f"{proc.stderr.decode(errors='replace')[-2000:]}")
            seconds, result = self._read_output(dst)

        return self._unpack(result, n, {"compute_call": seconds, "process": wall,
                                        **self._module_times(self.last_log)})

    # -- file exchange -------------------------------------------------------
    @staticmethod
    def _write_input(path: Path, values: np.ndarray, neighbors: NeighborTable) -> None:
        flat = neighbors.flat().astype("<i4")
        with open(path, "wb") as f:
            f.write(_MAGIC)
            f.write(np.array([values.shape[0], flat.shape[0], values.shape[0], 1, 1], dtype="<i4").tobytes())
            f.write(values.astype("<f4").tobytes())
            f.write(neighbors.counts.astype("<i4").tobytes())
            f.write(flat.tobytes())

    @staticmethod
    def _read_output(path: Path):
        raw = path.read_bytes()
        if raw[:4] != _MAGIC:
            raise RuntimeError("unexpected driver output")
        seconds = float(np.frombuffer(raw[4:12], dtype="<f8")[0])
        length = int(np.frombuffer(raw[12:16], dtype="<i4")[0])
        return seconds, np.frombuffer(raw[16:16 + 4 * length], dtype="<i4").astype(np.int64)

    @staticmethod
    def _unpack(result: np.ndarray, n: int, seconds: dict) -> KernelOutput:
        """Split AnyEG's buffer: [nCand, cand..., nExt, ext..., FatherLinks_cpy, FatherLinks, Values]."""
        i = 0
        n_sad = int(result[i]); i += 1
        saddles = result[i:i + n_sad]; i += n_sad
        n_ext = int(result[i]); i += 1
        extrema = result[i:i + n_ext]; i += n_ext
        first_link = result[i:i + n]; i += n
        final_link = result[i:i + n]
        return KernelOutput(extrema=extrema.copy(), saddles=saddles.copy(),
                            first_link=first_link.copy(), final_link=final_link.copy(),
                            num_candidates=int(seconds.pop("_candidates", -1)),
                            seconds=seconds)

    @staticmethod
    def _module_times(log: str) -> dict:
        """AnyEG prints its own timings (in Chinese); pick them out of its log."""
        names = {"第一模块": "module1_extrema", "第二模块": "module2_pointer_doubling",
                 "第三模块": "module3_candidates", "第四模块": "module4_upper_link"}
        out = {}
        for zh, key in names.items():
            m = re.search(zh + r"执行时间[:：]?\s*([0-9.eE+-]+)", log)
            if m:
                out[key] = float(m.group(1))
        cands = re.findall(r"CandidateSaddlePointNum:\s*(\d+)", log)
        out["_candidates"] = int(cands[0]) if cands else -1     # first print = after module 3
        return out

    def _gpu_index(self) -> int:
        if self.gpu is not None:
            return self.gpu
        try:
            import torch

            if torch.cuda.is_available():
                free = [(torch.cuda.mem_get_info(i)[0], -i) for i in range(torch.cuda.device_count())]
                return -max(free)[1]
        except ImportError:
            pass
        return 0
