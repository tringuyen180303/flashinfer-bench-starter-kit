# Building the CUDA Kernel

## Prerequisites

- CUDA Toolkit (tested with 12.8)
- PyTorch with CUDA support (tested with 2.8.0+cu128)
- Conda environment `fi-bench`

## Quick Build

```bash
source activate fi-bench
./solution/cuda/build.sh
```

## Manual Build

### 1. Activate the environment

```bash
source activate fi-bench
```

### 2. Compile with nvcc

```bash
TORCH_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/torch

nvcc -shared -Xcompiler -fPIC \
  -I$TORCH_DIR/include \
  -I$TORCH_DIR/include/torch/csrc/api/include \
  -I$CONDA_PREFIX/include/python3.12 \
  -L$TORCH_DIR/lib \
  -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lc10 -lc10_cuda \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -DTORCH_EXTENSION_NAME=moe_kernel \
  -Wno-deprecated-gpu-targets \
  -diag-suppress 177 \
  -o solution/cuda/moe_kernel.cpython-312-x86_64-linux-gnu.so \
  solution/cuda/kernel.cu
```

**Flags explained:**

| Flag | Purpose |
|------|---------|
| `-shared -Xcompiler -fPIC` | Build a shared library (Python extension) |
| `-I...torch/include` | PyTorch C++ headers (`torch/extension.h`, ATen) |
| `-I...torch/csrc/api/include` | PyTorch C++ API headers |
| `-I...python3.12` | Python headers (for pybind11) |
| `-L...torch/lib` | Link directory for torch libraries |
| `-ltorch -ltorch_cpu -ltorch_cuda` | Core PyTorch libs |
| `-ltorch_python` | pybind11 ↔ torch::Tensor bridge (fixes `type_caster` undefined symbol) |
| `-lc10 -lc10_cuda` | ATen/c10 tensor backend |
| `-D_GLIBCXX_USE_CXX11_ABI=1` | Must match PyTorch's ABI (check with `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"`) |
| `-DTORCH_EXTENSION_NAME=moe_kernel` | Sets the Python module name for `PYBIND11_MODULE` |
| `-Wno-deprecated-gpu-targets` | Suppress old SM arch warnings |
| `-diag-suppress 177` | Suppress "unused variable" warnings for constants used in later steps |

### 3. Run

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH \
python -c "
import sys; sys.path.insert(0, 'solution/cuda')
import torch, moe_kernel

T = 4
moe_kernel.kernel(
    torch.randn(T, 256, device='cuda'),                           # routing_logits
    torch.zeros(256, dtype=torch.bfloat16, device='cuda'),        # routing_bias
    torch.zeros(T, 7168, dtype=torch.float8_e4m3fn, device='cuda'),  # hidden_states
    torch.ones(56, T, device='cuda'),                             # hidden_states_scale
    torch.zeros(32, 4096, 7168, dtype=torch.float8_e4m3fn, device='cuda'),  # gemm1_weights
    torch.ones(32, 32, 56, device='cuda'),                        # gemm1_weights_scale
    torch.zeros(32, 7168, 2048, dtype=torch.float8_e4m3fn, device='cuda'),  # gemm2_weights
    torch.ones(32, 56, 16, device='cuda'),                        # gemm2_weights_scale
    0,                                                             # local_expert_offset
    1.0,                                                           # routed_scaling_factor
    torch.zeros(T, 7168, dtype=torch.bfloat16, device='cuda'),    # output (DPS)
)
torch.cuda.synchronize()
print('OK')
"
```

## Troubleshooting

### `undefined symbol: _ZN8pybind116detail11type_caster...`

Missing `-ltorch_python`. This library provides the pybind11 type casters for `torch::Tensor`.

### `undefined symbol` with mangled C++ names

ABI mismatch. Check your PyTorch's ABI:

```bash
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

Set `-D_GLIBCXX_USE_CXX11_ABI=` to match (0 or 1).

### `ModuleNotFoundError: No module named 'moe_kernel'`

Either the `.so` isn't on `sys.path`, or the suffix is wrong. Check:

```bash
python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
```

The output `.so` filename must end with this suffix (e.g., `.cpython-312-x86_64-linux-gnu.so`).

### `error while loading shared libraries: libtorch.so`

Set `LD_LIBRARY_PATH` to include the torch lib directory:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
```

## Running via flashinfer-bench

Instead of manual nvcc, you can use the framework which compiles automatically:

```bash
export FIB_DATASET_PATH=/path/to/mlsys26-contest
python scripts/run_local.py      # local GPU
modal run scripts/run_modal.py   # Modal B200
```
