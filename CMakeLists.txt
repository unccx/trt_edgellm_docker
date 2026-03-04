cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(trt_edgellm_superbuild LANGUAGES CXX)

include(ExternalProject)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

set(EDGELLM_SOURCE_DIR
    "${CMAKE_SOURCE_DIR}/3rdparty/TensorRT-Edge-LLM"
    CACHE PATH "TensorRT-Edge-LLM source directory")
set(EDGELLM_BUILD_DIR
    "${CMAKE_BINARY_DIR}/edgellm"
    CACHE PATH "TensorRT-Edge-LLM build directory")

set(INFER_SOURCE_DIR
    "${CMAKE_SOURCE_DIR}/infer"
    CACHE PATH "infer source directory")
set(INFER_BUILD_DIR
    "${CMAKE_BINARY_DIR}/infer"
    CACHE PATH "infer build directory")

set(CUDA_VERSION
    "12.8"
    CACHE STRING "CUDA version passed to TensorRT-Edge-LLM")
set(TRT_PACKAGE_DIR
    ""
    CACHE PATH "TensorRT package path (contains include/NvInfer.h and libnvinfer)")

if(TRT_PACKAGE_DIR STREQUAL "")
  message(FATAL_ERROR "Please set -DTRT_PACKAGE_DIR=<path> when invoking CMake.")
endif()

ExternalProject_Add(
  edgellm_proj
  PREFIX "${CMAKE_BINARY_DIR}/_deps/edgellm"
  SOURCE_DIR "${EDGELLM_SOURCE_DIR}"
  BINARY_DIR "${EDGELLM_BUILD_DIR}"
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}
    -DTRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}
    -DCUDA_VERSION=${CUDA_VERSION}
  BUILD_BYPRODUCTS
    "${EDGELLM_BUILD_DIR}/cpp/libedgellmCore.a"
    "${EDGELLM_BUILD_DIR}/cpp/libedgellmTokenizer.a"
    "${EDGELLM_BUILD_DIR}/examples/utils/libexampleUtils.a"
  INSTALL_COMMAND ""
)

ExternalProject_Add(
  infer_proj
  PREFIX "${CMAKE_BINARY_DIR}/_deps/infer"
  SOURCE_DIR "${INFER_SOURCE_DIR}"
  BINARY_DIR "${INFER_BUILD_DIR}"
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DEDGELLM_SOURCE_DIR=${EDGELLM_SOURCE_DIR}
    -DEDGELLM_BUILD_DIR=${EDGELLM_BUILD_DIR}
    -DTRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}
    -DCUDA_VERSION=${CUDA_VERSION}
  BUILD_BYPRODUCTS "${INFER_BUILD_DIR}/qwen2_vl_infer"
  DEPENDS edgellm_proj
  INSTALL_COMMAND ""
)

add_custom_target(
  link_compile_commands
  COMMAND
    ${CMAKE_COMMAND} -E rm -f "${EDGELLM_SOURCE_DIR}/compile_commands.json"
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    "${EDGELLM_BUILD_DIR}/compile_commands.json"
    "${EDGELLM_SOURCE_DIR}/compile_commands.json"
  COMMAND
    ${CMAKE_COMMAND} -E rm -f "${INFER_SOURCE_DIR}/compile_commands.json"
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    "${INFER_BUILD_DIR}/compile_commands.json"
    "${INFER_SOURCE_DIR}/compile_commands.json"
  DEPENDS infer_proj
  COMMENT "Linking compile_commands.json for edgellm and infer"
  VERBATIM)

add_custom_target(all_projects DEPENDS infer_proj link_compile_commands)
