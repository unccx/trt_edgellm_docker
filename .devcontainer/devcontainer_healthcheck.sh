#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="${WORKSPACE_DIR}/3rdparty/TensorRT-Edge-LLM"
VENV_PY="${REPO_DIR}/.venv/bin/python"

check_cmd() {
    local cmd="$1"
    if command -v "${cmd}" >/dev/null 2>&1; then
        echo ">>> OK: ${cmd}=$(command -v "${cmd}")"
    else
        echo ">>> WARN: '${cmd}' not found in PATH"
    fi
}

check_cmd python3
check_cmd cmake
check_cmd nvcc

if [ -x "${VENV_PY}" ]; then
    echo ">>> OK: virtualenv python detected at ${VENV_PY}"
else
    echo ">>> WARN: virtualenv python missing at ${VENV_PY}"
    echo ">>>      Run: bash scripts/setup.sh"
fi

if [ -f /usr/include/NvInfer.h ] || [ -f /usr/include/x86_64-linux-gnu/NvInfer.h ] || \
   [ -f /usr/local/tensorrt/include/NvInfer.h ] || [ -f /usr/local/tensorrt/include/x86_64-linux-gnu/NvInfer.h ]; then
    echo ">>> OK: TensorRT headers found"
else
    echo ">>> WARN: TensorRT headers not found in default locations"
fi
