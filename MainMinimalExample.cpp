// This minimal reference is for easier initial understanding.
// See the full example for GPU tensor binding.
// The minimal sample is also hard-coded to a specific model file and input sizes.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#pragma warning(disable: 4100) // oh hush - warning : 'argv': unreferenced formal parameter

#include <iostream>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <functional>
#include <utility>
#include <string_view>
#include <optional>
#include <memory>
#include <charconv>
#include <assert.h>

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h> // Use the good old helper functions, not the huge WinRT entanglement.

#include "cpu_provider_factory.h"
#include "dml_provider_factory.h"
#include "onnxruntime_cxx_api.h"

////////////////////////////////////////////////////////////////////////////////
// Configuration

constexpr std::string_view modelFileConstant = "en_PP-OCRv3_det_infer.onnx";
constexpr uint32_t batchSize = 1;
constexpr std::array<int64_t, 4> inputShape = {64, 3, 64, 64}; // Sizes specific to Upsample4xOpset11.

////////////////////////////////////////////////////////////////////////////////
// Main execution

int main(int argc, char* argv[])
{
    if (argc > 1)
    {
        std::cout << "A file parameter was given, but the minimal example build only supports one specific model." << std::endl;
        return EXIT_FAILURE;
    }

    #ifdef _WIN32
    std::wstring wideString = std::wstring(modelFileConstant.begin(), modelFileConstant.end());
    std::basic_string<ORTCHAR_T> modelFile = std::basic_string<ORTCHAR_T>(wideString);
    #else
    std::string modelFile = modelFileConstant;
    #endif
    std::cout << "Loading " << modelFileConstant << std::endl;

    ////////////////////////////////////////
    // Get API, and setup environment.
    OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
    const OrtDmlApi* ortDmlApi;
    ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi));
    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "DirectML_Direct3D_TensorAllocation_Test"); // Note ORT_LOGGING_LEVEL_VERBOSE is useful too.

    ////////////////////////////////////////
    // Set model-specific session options.
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // For DML EP
    sessionOptions.DisableMemPattern(); // For DML EP
    ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", batchSize);
    ortApi.AddFreeDimensionOverrideByName(sessionOptions, "p2o.DynamicDimension.0", 64);
    ortApi.AddFreeDimensionOverrideByName(sessionOptions, "p2o.DynamicDimension.1", 64);
    ortApi.AddFreeDimensionOverrideByName(sessionOptions, "p2o.DynamicDimension.2", 64);
    ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, /*device index*/ 0);
    // Preferred approach above. Deprecated: OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);

    Ort::Session session(environment, modelFile.c_str(), sessionOptions);

    ////////////////////////////////////////
    // Declare tensor data for binding.
    // Just use CPU-bound resources here (see the full example for GPU tensor binding).

    std::vector<Ort::Value> inputTensors;
    size_t elementCount = size_t(std::accumulate(inputShape.begin(), inputShape.end(), int64_t(1), std::multiplies<int64_t>()));
    std::vector<float> inputTensorValues(elementCount, 42.0f);
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size()));

    std::vector<char const*> inputNames = {"x"};
    std::vector<char const*> outputNames = {"sigmoid_0.tmp_0"};

    ////////////////////////////////////////
    // Execute the model with the given inputs and named outputs.

    std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{}, inputNames.data(), inputTensors.data(), inputTensors.size(), outputNames.data(), outputNames.size());

    std::cout << "Output count: " << outputs.size() << std::endl;
    if (!outputs.empty() && outputs[0] != nullptr && outputs[0].IsTensor())
    {
        std::cout << "Output first value: " << outputs[0].GetTensorData<float>()[1] << std::endl;
    }

    return EXIT_SUCCESS;
}
