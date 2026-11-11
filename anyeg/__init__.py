"""
anyeg: a separate pipeline that computes extremum graphs the way AnyEG does.

AnyEG (https://github.com/BuptCIAGroup/AnyEG) is a CUDA + TTK filter. This
package reproduces it without TTK or VTK:

* ``anyeg.kernels.OriginalCudaKernels`` runs AnyEG's own CUDA code (built
  from its source by ``anyeg/native/build.sh``);
* ``anyeg.kernels.TorchKernels`` is a PyTorch port of the same four GPU
  modules, checked to give identical output;
* ``anyeg.assembly`` ports AnyEG's CPU post-processing (arcs, ``.vtp`` output).

It does not import anything from ``pipeline/`` (our own method); the
``comparison`` package puts the two side by side.

    python3 -m anyeg third_party/AnyEG/data/scalar_field_v1.vti -o out.vtp
"""

from .assembly import ArcAssembler, AssemblyOptions
from .field import ScalarField
from .graph import AnyEGGraph
from .kernels import ExtremumGraphKernels, KernelOutput, OriginalCudaKernels, TorchKernels, make_kernels
from .neighborhood import (AxisGridNeighborhood, FreudenthalGridNeighborhood, KnnNeighborhood,
                           NeighborhoodBuilder, NeighborTable)
from .pipeline import AnyEGPipeline
from .vtk_io import PolyLines, read_vti, read_vtp, write_vti, write_vtp

__all__ = [
    "AnyEGGraph", "AnyEGPipeline", "ArcAssembler", "AssemblyOptions", "AxisGridNeighborhood",
    "ExtremumGraphKernels", "FreudenthalGridNeighborhood", "KernelOutput", "KnnNeighborhood",
    "NeighborhoodBuilder", "NeighborTable", "OriginalCudaKernels", "PolyLines", "ScalarField",
    "TorchKernels", "make_kernels", "read_vti", "read_vtp", "write_vti", "write_vtp",
]
