#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="${WORKSPACE_DIR}/TensorRT-Edge-LLM"

if [ ! -d "${REPO_DIR}/.git" ]; then
    echo ">>> Cloning TensorRT-Edge-LLM ..."
    git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git "${REPO_DIR}"
fi

echo ">>> Updating submodules ..."
cd "${REPO_DIR}"
git submodule update --init --recursive

echo ">>> Installing Python export pipeline ..."
uv pip install --system --break-system-packages . -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple


CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo ">>> Detected CUDA version: ${CUDA_VER}"

TRT_PACKAGE_DIR=""
for candidate in /usr /usr/local/tensorrt; do
    if [ -f "${candidate}/include/NvInfer.h" ] || \
       [ -f "${candidate}/include/x86_64-linux-gnu/NvInfer.h" ]; then
        TRT_PACKAGE_DIR="${candidate}"
        break
    fi
done
if [ -z "${TRT_PACKAGE_DIR}" ]; then
    echo "ERROR: TensorRT not found. Set TRT_PACKAGE_DIR manually."
    exit 1
fi
echo ">>> Detected TensorRT at: ${TRT_PACKAGE_DIR}"

echo ">>> Configuring CMake build (x86 GPU) ..."
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR="${TRT_PACKAGE_DIR}" \
    -DCUDA_VERSION="${CUDA_VER}"

echo ">>> Setup complete. To build C++ runtime:"
echo "    cd ${REPO_DIR}/build && make -j\$(nproc)"
