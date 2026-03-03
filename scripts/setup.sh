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
SETUP_STAMP="${REPO_DIR}/.setup_done"
FORCE_SETUP="${FORCE_SETUP:-0}"

log() {
    echo ">>> $*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        die "'${cmd}' not found in PATH."
    fi
}

ensure_commands() {
    local cmds=(git python3 uv cmake nvcc)
    local cmd
    for cmd in "${cmds[@]}"; do
        require_cmd "${cmd}"
    done
}

should_skip_setup() {
    [ "${FORCE_SETUP}" != "1" ] && [ -f "${SETUP_STAMP}" ]
}

sync_submodules() {
    if [ ! -d "${REPO_DIR}/.git" ]; then
        log "Initializing submodule 3rdparty/TensorRT-Edge-LLM ..."
        cd "${WORKSPACE_DIR}"
    else
        log "Updating submodules (TensorRT-Edge-LLM internal) ..."
        cd "${REPO_DIR}"
    fi
    git submodule update --init --recursive --depth 1
}

detect_cuda_version() {
    local cuda_ver
    cuda_ver="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
    if [ -z "${cuda_ver}" ]; then
        die "Failed to parse CUDA version from 'nvcc --version'."
    fi
    echo "${cuda_ver}"
}

detect_trt_package_dir() {
    local trt_package_dir="${TRT_PACKAGE_DIR:-}"
    local candidate
    if [ -z "${trt_package_dir}" ]; then
        for candidate in /usr /usr/local/tensorrt; do
            if [ -f "${candidate}/include/NvInfer.h" ] || \
               [ -f "${candidate}/include/x86_64-linux-gnu/NvInfer.h" ]; then
                trt_package_dir="${candidate}"
                break
            fi
        done
    fi
    if [ -z "${trt_package_dir}" ]; then
        die "TensorRT not found. Set TRT_PACKAGE_DIR manually."
    fi
    echo "${trt_package_dir}"
}

setup_python_env() {
    local pip_index_url="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    local pip_extra_index_url="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"
    local venv_dir="${REPO_DIR}/.venv"

    log "Installing TensorRT-Edge-LLM (Python export pipeline) ..."
    log "Using PIP_INDEX_URL=${pip_index_url}"
    log "Using PIP_EXTRA_INDEX_URL=${pip_extra_index_url}"
    log "Creating virtual environment at ${venv_dir} (with system-site-packages)..."

    uv venv --system-site-packages "${venv_dir}"
    uv pip install --python "${venv_dir}/bin/python" -e . \
        --index-url "${pip_index_url}" --extra-index-url "${pip_extra_index_url}"
}

configure_cmake() {
    local cuda_ver="$1"
    local trt_package_dir="$2"
    log "Configuring CMake build (x86 GPU) ..."
    cmake -S "${REPO_DIR}" -B "${REPO_DIR}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR="${trt_package_dir}" \
        -DCUDA_VERSION="${cuda_ver}"
}

main() {
    ensure_commands

    if should_skip_setup; then
        log "Setup already completed at ${SETUP_STAMP}; skipping heavy initialization."
        log "Re-run with FORCE_SETUP=1 bash scripts/setup.sh to force refresh."
        return 0
    fi

    sync_submodules
    cd "${REPO_DIR}"

    setup_python_env

    local cuda_ver
    cuda_ver="$(detect_cuda_version)"
    log "Detected CUDA version: ${cuda_ver}"

    local trt_package_dir
    trt_package_dir="$(detect_trt_package_dir)"
    log "Detected TensorRT at: ${trt_package_dir}"

    configure_cmake "${cuda_ver}" "${trt_package_dir}"

    log "Setup complete. To build C++ runtime:"
    echo "    cd ${REPO_DIR}/build && make -j\$(nproc)"
    log "To use Python env:"
    echo "    source ${REPO_DIR}/.venv/bin/activate"
    touch "${SETUP_STAMP}"
}

main "$@"
