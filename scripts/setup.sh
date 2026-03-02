#!/usr/bin/env bash
# 若构建/运行失败，请先更新宿主机 NVIDIA 驱动；或改用永久 container：
#   docker run -dit --gpus all --name trt-edge-llm --restart unless-stopped \
#     --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v /path/to/trt_edgellm_docker:/workspace -w /workspace/3rdparty/TensorRT-Edge-LLM \
#     nvcr.io/nvidia/pytorch:25.12-py3 bash
#   docker exec -it trt-edge-llm bash
#   cd /workspace/3rdparty/TensorRT-Edge-LLM && python3 -m pip install -e .
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="${WORKSPACE_DIR}/3rdparty/TensorRT-Edge-LLM"

if [ ! -d "${REPO_DIR}" ]; then
    echo ">>> Initializing submodule 3rdparty/TensorRT-Edge-LLM ..."
    cd "${WORKSPACE_DIR}"
    git submodule update --init --recursive
fi
cd "${REPO_DIR}"
echo ">>> Updating submodules (TensorRT-Edge-LLM internal) ..."
git submodule update --init --recursive

# NGC 容器预装了 NVIDIA 自编译的 PyTorch（版本号形如 2.10.0a0+xxx），
# 而上游 TensorRT-Edge-LLM 的 requirements.txt / pyproject.toml 可能要求
# 不同的 torch 版本（如 torch~=2.9.1）。
# 下面的逻辑检测容器中已安装的 torch 是否满足上游约束，
# 如果不满足则自动放宽约束以复用容器中的 torch，避免耗时的重新安装。
echo ">>> Checking container torch version ..."
if python3 -c "import torch" 2>/dev/null; then
    # 提取不含 local identifier（+xxx）的版本号，如 "2.10.0a0"
    CONTAINER_TORCH=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
    echo "    Container torch: ${CONTAINER_TORCH}"

    # 用 packaging.specifiers 解析 requirements.txt 中的 torch 版本约束，
    # 判断容器中的 torch 版本（含预发布标记 a0）是否在约束范围内
    if ! python3 -c "
from packaging.specifiers import SpecifierSet
import re, sys
with open('${REPO_DIR}/requirements.txt') as f:
    for line in f:
        m = re.match(r'^torch([><=~!].+)', line.strip())
        if m:
            spec = SpecifierSet(m.group(1), prereleases=True)
            sys.exit(0 if '${CONTAINER_TORCH}' in spec else 1)
sys.exit(0)
"; then
        # 不兼容：生成新约束 torch>=当前版本,<下一个 minor 版本
        # 例如容器 torch 为 2.10.0a0 -> torch>=2.10.0a0,<2.11.0
        MAJOR_MINOR=$(echo "${CONTAINER_TORCH}" | grep -oP '^\d+\.\d+')
        MINOR=${MAJOR_MINOR#*.}
        NEXT_MINOR=$((MINOR + 1))
        NEW_CONSTRAINT="torch>=${CONTAINER_TORCH},<2.${NEXT_MINOR}.0"
        echo "    Container torch ${CONTAINER_TORCH} not compatible, patching -> ${NEW_CONSTRAINT}"
        sed -i "s|\"torch[^\"]*\"|\"${NEW_CONSTRAINT}\"|" "${REPO_DIR}/pyproject.toml"
        sed -i "s|^torch[><=~!].*|${NEW_CONSTRAINT}|" "${REPO_DIR}/requirements.txt"
    else
        echo "    Container torch ${CONTAINER_TORCH} is compatible, no patching needed"
    fi
fi

echo ">>> Installing TensorRT-Edge-LLM (Python export pipeline) ..."
cd "${REPO_DIR}"
python3 -m pip install --no-cache-dir -e . \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple


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
