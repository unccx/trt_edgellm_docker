cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(qwen2_vl_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EDGELLM_SOURCE_DIR
    ""
    CACHE PATH "TensorRT-Edge-LLM source directory")
set(EDGELLM_BUILD_DIR
    ""
    CACHE PATH "TensorRT-Edge-LLM build directory")
set(CUDA_VERSION
    "12.8"
    CACHE STRING "CUDA version for runtime libraries")
set(TRT_PACKAGE_DIR
    ""
    CACHE PATH "TensorRT package path (contains include/NvInfer.h and libnvinfer)")

if(EDGELLM_SOURCE_DIR STREQUAL "" OR EDGELLM_BUILD_DIR STREQUAL "")
  message(FATAL_ERROR "Please set -DEDGELLM_SOURCE_DIR and -DEDGELLM_BUILD_DIR.")
endif()

if(TRT_PACKAGE_DIR STREQUAL "")
  message(FATAL_ERROR "Please set -DTRT_PACKAGE_DIR.")
endif()

set(CUDA_DIR "/usr/local/cuda-${CUDA_VERSION}")

find_library(
  CUDART_LIB cudart
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES lib lib64)
find_library(
  CUDA_DRIVER_LIB cuda
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES lib lib64 lib/stubs)
find_library(
  NVINFER_LIB nvinfer
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES lib lib64 ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu x86_64-linux-gnu)
find_path(
  CUDA_INCLUDE_DIR cuda.h
  HINTS ${CUDA_DIR} /usr/local/cuda /usr
  PATH_SUFFIXES include targets/x86_64-linux/include)

if(NOT CUDART_LIB OR NOT CUDA_DRIVER_LIB OR NOT NVINFER_LIB)
  message(FATAL_ERROR "Failed to locate CUDA/TensorRT runtime libraries.")
endif()

if(NOT CUDA_INCLUDE_DIR)
  message(FATAL_ERROR "Failed to locate CUDA headers (cuda.h).")
endif()

set(EDGELLM_CPP_LIB_DIR "${EDGELLM_BUILD_DIR}/cpp")
set(EDGELLM_EXAMPLE_LIB_DIR "${EDGELLM_BUILD_DIR}/examples/utils")

add_library(edgellmCore STATIC IMPORTED GLOBAL)
set_target_properties(
  edgellmCore
  PROPERTIES IMPORTED_LOCATION "${EDGELLM_CPP_LIB_DIR}/libedgellmCore.a")

add_library(edgellmTokenizer STATIC IMPORTED GLOBAL)
set_target_properties(
  edgellmTokenizer
  PROPERTIES IMPORTED_LOCATION "${EDGELLM_CPP_LIB_DIR}/libedgellmTokenizer.a")

add_library(exampleUtils STATIC IMPORTED GLOBAL)
set_target_properties(
  exampleUtils
  PROPERTIES IMPORTED_LOCATION "${EDGELLM_EXAMPLE_LIB_DIR}/libexampleUtils.a")

add_executable(qwen2_vl_infer qwen2_vl_infer.cpp)

target_include_directories(
  qwen2_vl_infer PRIVATE
  "${CUDA_INCLUDE_DIR}"
  "${EDGELLM_SOURCE_DIR}/cpp"
  "${EDGELLM_SOURCE_DIR}/examples/utils"
  "${EDGELLM_SOURCE_DIR}/examples/multimodal"
  "${EDGELLM_SOURCE_DIR}/3rdParty/nlohmannJson/include"
  "${EDGELLM_SOURCE_DIR}/3rdParty/stb")

target_link_libraries(
  qwen2_vl_infer PRIVATE
  exampleUtils
  edgellmCore
  edgellmTokenizer
  ${NVINFER_LIB}
  ${CUDART_LIB}
  ${CUDA_DRIVER_LIB}
  dl
  pthread)
