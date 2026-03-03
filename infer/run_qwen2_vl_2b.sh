#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
ENGINE_DIR="${ENGINE_DIR:-${ROOT_DIR}/models/Qwen2-VL-2B-Instruct/engines/llm}"
MULTIMODAL_ENGINE_DIR="${MULTIMODAL_ENGINE_DIR:-${ROOT_DIR}/models/Qwen2-VL-2B-Instruct/engines/visual}"
INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/infer/input_qwen2_vl_2b_example.json}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/infer/output_qwen2_vl_2b_example.json}"
INFER_BIN="${BUILD_DIR}/infer/qwen2_vl_infer"
if [[ -z "${PLUGIN_PATH:-}" ]]; then
  DEFAULT_PLUGIN_PATH="${BUILD_DIR}/edgellm/libNvInfer_edgellm_plugin.so"
  FALLBACK_PLUGIN_PATH="${ROOT_DIR}/3rdparty/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
  if [[ -f "${DEFAULT_PLUGIN_PATH}" ]]; then
    PLUGIN_PATH="${DEFAULT_PLUGIN_PATH}"
  elif [[ -f "${FALLBACK_PLUGIN_PATH}" ]]; then
    PLUGIN_PATH="${FALLBACK_PLUGIN_PATH}"
  else
    PLUGIN_PATH="${DEFAULT_PLUGIN_PATH}"
  fi
fi

if [[ ! -x "${INFER_BIN}" ]]; then
  echo "Missing binary: ${INFER_BIN}"
  echo "Build first:"
  echo "  cmake -S ${ROOT_DIR} -B ${BUILD_DIR} -DTRT_PACKAGE_DIR=<path>"
  echo "  cmake --build ${BUILD_DIR} --target infer_proj"
  exit 1
fi

if [[ ! -f "${PLUGIN_PATH}" ]]; then
  echo "Missing plugin library: ${PLUGIN_PATH}"
  exit 1
fi

export EDGELLM_PLUGIN_PATH="${PLUGIN_PATH}"

"${INFER_BIN}" \
  --engineDir="${ENGINE_DIR}" \
  --multimodalEngineDir="${MULTIMODAL_ENGINE_DIR}" \
  --inputFile="${INPUT_FILE}" \
  --outputFile="${OUTPUT_FILE}"

echo "Inference done. Output: ${OUTPUT_FILE}"
