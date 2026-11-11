#!/usr/bin/env bash
# Build the ORIGINAL AnyEG CUDA kernels with a small file-based driver.
#
#   anyeg/native/build.sh            # clones AnyEG into third_party/ if needed
#
# Environment variables:
#   ANYEG_SRC  AnyEG checkout to use        (default: third_party/AnyEG)
#   NVCC       CUDA compiler                (default: $CUDA_HOME/bin/nvcc, /usr/local/cuda/bin/nvcc)
#   CUDA_ARCH  GPU architecture             (default: sm_86, RTX 30xx)
#
# The AnyEG sources are copied unchanged except for one line: helper_cuda.h
# includes helper_string.h through an absolute path in the authors' home
# directory, which is replaced by the plain file name. TTK and VTK are not
# needed, because the driver (anyeg_driver.cu) replaces the TTK filter.
# Output: anyeg/native/build/anyeg_driver
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="${ANYEG_SRC:-$ROOT/third_party/AnyEG}"
COMMIT=08068c2bed7c9ef1cdfeb1dbb3ffb0e659ae238c   # the version this repo was tested with
NVCC="${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}"
ARCH="${CUDA_ARCH:-sm_86}"
BUILD="$HERE/build"

if [ ! -f "$SRC/cudaHelper.cu" ]; then
    echo "Cloning AnyEG into $SRC"
    git clone https://github.com/BuptCIAGroup/AnyEG.git "$SRC"
    git -C "$SRC" checkout --quiet "$COMMIT"
fi

mkdir -p "$BUILD"
for f in cudaHelper.cu helper_cuda.h helper_string.h stopwatch.h; do
    cp "$SRC/$f" "$BUILD/$f"
done
sed -i -E 's|#include *[<"]/home/[^>"]*/([^/>"]+)[>"]|#include "\1"|' "$BUILD/helper_cuda.h"

echo "Compiling with $NVCC for $ARCH"
"$NVCC" -O3 -std=c++14 -arch="$ARCH" -Xcompiler -fopenmp -w -I"$BUILD" \
    "$BUILD/cudaHelper.cu" "$HERE/anyeg_driver.cu" -o "$BUILD/anyeg_driver" -lgomp
echo "Built $BUILD/anyeg_driver"
