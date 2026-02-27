#!/usr/bin/env bash
# Build the MoE CUDA kernel into a Python extension.
# Usage: source activate fi-bench && ./solution/cuda/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/kernel.cu"

# Resolve paths from the active Python/PyTorch
TORCH_DIR=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent)")
PYINC=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

OUTPUT="$SCRIPT_DIR/moe_kernel${EXT_SUFFIX}"

echo "TORCH_DIR : $TORCH_DIR"
echo "PYINC     : $PYINC"
echo "ABI       : $ABI"
echo "OUTPUT    : $OUTPUT"
echo ""

nvcc -shared -Xcompiler -fPIC \
  -I"$TORCH_DIR/include" \
  -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -I"$PYINC" \
  -L"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lc10 -lc10_cuda \
  -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
  -DTORCH_EXTENSION_NAME=moe_kernel \
  -Wno-deprecated-gpu-targets \
  -diag-suppress 177 \
  -o "$OUTPUT" \
  "$KERNEL_SRC"

echo ""
echo "Built: $OUTPUT"
