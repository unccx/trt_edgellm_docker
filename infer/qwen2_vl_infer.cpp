/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"

#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

enum InferOptionId : int {
  HELP = 900,
  INPUT_FILE = 901,
  ENGINE_DIR = 902,
  MULTIMODAL_ENGINE_DIR = 903,
  OUTPUT_FILE = 904,
  DEBUG = 905
};

struct InferArgs {
  bool help{false};
  bool debug{false};
  std::string inputFile;
  std::string outputFile;
  std::string engineDir;
  std::string multimodalEngineDir;
};

void printUsage(char const *programName) {
  std::cerr << "Usage: " << programName
            << " --engineDir=<path> --multimodalEngineDir=<path> "
               "--inputFile=<path> --outputFile=<path> [--debug]"
            << std::endl;
}

bool parseArgs(InferArgs &args, int argc, char *argv[]) {
  static struct option options[] = {
      {"help", no_argument, nullptr, InferOptionId::HELP},
      {"inputFile", required_argument, nullptr, InferOptionId::INPUT_FILE},
      {"engineDir", required_argument, nullptr, InferOptionId::ENGINE_DIR},
      {"multimodalEngineDir", required_argument, nullptr,
       InferOptionId::MULTIMODAL_ENGINE_DIR},
      {"outputFile", required_argument, nullptr, InferOptionId::OUTPUT_FILE},
      {"debug", no_argument, nullptr, InferOptionId::DEBUG},
      {nullptr, 0, nullptr, 0}};

  int opt{};
  while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1) {
    switch (opt) {
    case InferOptionId::HELP:
      args.help = true;
      return true;
    case InferOptionId::INPUT_FILE:
      args.inputFile = optarg;
      break;
    case InferOptionId::ENGINE_DIR:
      args.engineDir = optarg;
      break;
    case InferOptionId::MULTIMODAL_ENGINE_DIR:
      args.multimodalEngineDir = optarg;
      break;
    case InferOptionId::OUTPUT_FILE:
      args.outputFile = optarg;
      break;
    case InferOptionId::DEBUG:
      args.debug = true;
      break;
    default:
      return false;
    }
  }

  if (args.inputFile.empty() || args.outputFile.empty() ||
      args.engineDir.empty() || args.multimodalEngineDir.empty()) {
    return false;
  }

  gLogger.setLevel(args.debug ? nvinfer1::ILogger::Severity::kVERBOSE
                              : nvinfer1::ILogger::Severity::kINFO);
  return true;
}

std::vector<rt::LLMGenerationRequest>
parseInputFile(std::filesystem::path const &inputFilePath) {
  std::vector<rt::LLMGenerationRequest> batchedRequests;

  Json inputData;
  std::ifstream inputFileStream(inputFilePath);
  check::check(inputFileStream.is_open(),
               "Failed to open input file: " + inputFilePath.string());
  inputData = Json::parse(inputFileStream);
  inputFileStream.close();

  int batchSize = inputData.value("batch_size", 1);
  check::check(batchSize > 0, "batch_size must be positive");

  float temperature = inputData.value("temperature", 1.0f);
  float topP = inputData.value("top_p", 1.0f);
  int64_t topK = inputData.value("top_k", 50);
  int64_t maxGenerateLength = inputData.value("max_generate_length", 128);
  bool applyChatTemplate = inputData.value("apply_chat_template", true);
  bool addGenerationPrompt = inputData.value("add_generation_prompt", true);
  bool enableThinking = inputData.value("enable_thinking", false);

  check::check(inputData.contains("requests") &&
                   inputData["requests"].is_array(),
               "'requests' must be an array");
  auto const &requestsArray = inputData["requests"];

  for (size_t startIdx = 0; startIdx < requestsArray.size();
       startIdx += static_cast<size_t>(batchSize)) {
    rt::LLMGenerationRequest batchRequest;
    batchRequest.temperature = temperature;
    batchRequest.topP = topP;
    batchRequest.topK = topK;
    batchRequest.maxGenerateLength = maxGenerateLength;
    batchRequest.applyChatTemplate = applyChatTemplate;
    batchRequest.addGenerationPrompt = addGenerationPrompt;
    batchRequest.enableThinking = enableThinking;

    size_t const endIdx = std::min(startIdx + static_cast<size_t>(batchSize),
                                   requestsArray.size());
    for (size_t requestIdx = startIdx; requestIdx < endIdx; ++requestIdx) {
      auto const &requestItem = requestsArray[requestIdx];
      check::check(requestItem.contains("messages") &&
                       requestItem["messages"].is_array(),
                   "Each request must contain a 'messages' array");

      rt::LLMGenerationRequest::Request request;
      for (auto const &messageJson : requestItem["messages"]) {
        check::check(messageJson.contains("role") &&
                         messageJson.contains("content"),
                     "Each message must contain 'role' and 'content'");

        rt::Message message;
        message.role = messageJson["role"].get<std::string>();
        auto const &contentJson = messageJson["content"];

        if (contentJson.is_string()) {
          rt::Message::MessageContent content;
          content.type = "text";
          content.content = contentJson.get<std::string>();
          message.contents.push_back(std::move(content));
        } else if (contentJson.is_array()) {
          for (auto const &contentItemJson : contentJson) {
            check::check(contentItemJson.contains("type"),
                         "Each content item must contain 'type'");
            rt::Message::MessageContent content;
            content.type = contentItemJson["type"].get<std::string>();
            if (content.type == "text") {
              content.content = contentItemJson["text"].get<std::string>();
            } else if (content.type == "image") {
              content.content = contentItemJson["image"].get<std::string>();
              auto image = rt::imageUtils::loadImageFromFile(content.content);
              check::check(image.buffer != nullptr,
                           "Failed to load image: " + content.content);
              request.imageBuffers.push_back(std::move(image));
            } else {
              throw std::runtime_error("Only text/image content are supported "
                                       "in infer/qwen2_vl_infer");
            }
            message.contents.push_back(std::move(content));
          }
        } else {
          throw std::runtime_error("message.content must be string or array");
        }
        request.messages.push_back(std::move(message));
      }
      batchRequest.requests.push_back(std::move(request));
    }
    batchedRequests.push_back(std::move(batchRequest));
  }

  return batchedRequests;
}

int main(int argc, char *argv[]) {
  InferArgs args;
  if (!parseArgs(args, argc, argv)) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }
  if (args.help) {
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  auto pluginHandle = loadEdgellmPluginLib();
  if (!pluginHandle) {
    LOG_ERROR("Failed to load TensorRT-Edge-LLM plugin library.");
    return EXIT_FAILURE;
  }

  std::vector<rt::LLMGenerationRequest> batchedRequests;
  try {
    batchedRequests = parseInputFile(args.inputFile);
  } catch (std::exception const &e) {
    LOG_ERROR("Failed to parse input file: %s", e.what());
    return EXIT_FAILURE;
  }
  if (batchedRequests.empty()) {
    LOG_ERROR("No valid requests found in input file.");
    return EXIT_FAILURE;
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<rt::LLMInferenceRuntime> runtime{nullptr};
  try {
    std::unordered_map<std::string, std::string> loraWeightsMap;
    runtime = std::make_unique<rt::LLMInferenceRuntime>(
        args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
  } catch (std::exception const &e) {
    LOG_ERROR("Failed to initialize runtime: %s", e.what());
    cudaStreamDestroy(stream);
    return EXIT_FAILURE;
  }

  nlohmann::json outputData;
  outputData["input_file"] = args.inputFile;
  outputData["responses"] = nlohmann::json::array();

  bool hasFailure{false};
  for (size_t requestIdx = 0; requestIdx < batchedRequests.size();
       ++requestIdx) {
    rt::LLMGenerationResponse response;
    bool const ok =
        runtime->handleRequest(batchedRequests[requestIdx], response, stream);
    if (!ok) {
      hasFailure = true;
    }

    for (size_t batchIdx = 0;
         batchIdx < batchedRequests[requestIdx].requests.size(); ++batchIdx) {
      nlohmann::json responseJson;
      responseJson["request_idx"] = requestIdx;
      responseJson["batch_idx"] = batchIdx;
      responseJson["output_text"] =
          ok ? response.outputTexts[batchIdx]
             : "TensorRT Edge LLM cannot handle this request. Fails.";
      outputData["responses"].push_back(std::move(responseJson));
    }
  }

  try {
    std::ofstream outputFile(args.outputFile);
    check::check(outputFile.is_open(),
                 "Failed to open output file: " + args.outputFile);
    outputFile << outputData.dump(4);
  } catch (std::exception const &e) {
    LOG_ERROR("Failed to write output file: %s", e.what());
    cudaStreamDestroy(stream);
    return EXIT_FAILURE;
  }

  cudaStreamDestroy(stream);
  return hasFailure ? EXIT_FAILURE : EXIT_SUCCESS;
}
