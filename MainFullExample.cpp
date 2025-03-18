#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <iostream>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <functional>
#include <utility>
#include <string_view>
#include <span>
#include <optional>
#include <memory>
#include <charconv>
#include <assert.h>

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h> // Use the good old helper functions, not the huge WinRT entanglement.

#include "onnxruntime_c_api.h"
#include "cpu_provider_factory.h"
#include "dml_provider_factory.h"
#include "onnxruntime_cxx_api.h"

#include "FloatSupport.h"

////////////////////////////////////////////////////////////////////////////////
// Configuration

constexpr bool USE_DML_EXECUTION_PROVIDER = true;
constexpr bool PASS_TENSORS_AS_D3D_RESOURCES = true && USE_DML_EXECUTION_PROVIDER;
constexpr bool EXPORT_OPTIMIZED_FILE = false;
constexpr wchar_t const* OPTIMIZED_FILENAME = L"optimized.ort";
constexpr bool FILL_INPUTS_INCREASING_SEQUENCE = true; // Fill inputs with {0,1,2...}. Otherwise just use zeros.
constexpr GraphOptimizationLevel GRAPH_OPTIMIZATION_LEVEL = GraphOptimizationLevel::ORT_ENABLE_ALL;
constexpr std::pair<char const*, int> NAMED_MODEL_DIMENSIONS[] =
{
    {"batch_size", 1},
    {"p2o.DynamicDimension.0", 64},
    {"p2o.DynamicDimension.1", 64},
    {"p2o.DynamicDimension.2", 64},
    // Add more here if the model has any.
};

// CPU can't accept D3D resources, as you'll just get an error "No requested allocator available".
static_assert(USE_DML_EXECUTION_PROVIDER == true || PASS_TENSORS_AS_D3D_RESOURCES == false);

////////////////////////////////////////////////////////////////////////////////
// Common helpers

#define THROW_IF_FAILED(hr) {HRESULT localHr = (hr); if (FAILED(localHr)) throw localHr;}
#define RETURN_IF_FAILED(hr) {HRESULT localHr = (hr); if (FAILED(localHr)) return localHr;}

#define THROW_IF_ORT_FAILED(exp) \
    { \
        OrtStatus* status = (exp); \
        if (status != nullptr) \
        { \
            throw ConvertOrtStatusToHResult(*status); \
        } \
    }

template <typename T>
using BaseType =
    std::remove_cv_t<
        std::remove_reference_t<
            std::remove_pointer_t<
                std::remove_all_extents_t<T>
            >
        >
    >;

template <typename T>
using deleting_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;


template <typename C, typename T = BaseType<decltype(*std::declval<C>().data())>>
T GetElementCount(C const& range)
{
    return std::accumulate(range.begin(), range.end(), static_cast<T>(1), std::multiplies<T>());
};


template <typename T>
struct ComPtr : public Microsoft::WRL::ComPtr<T>
{
    // Having to call .Get() dozens of times for every function call that takes a T* is ridiculous.
    operator T* () { return this->Get(); }
};

// Collection of 64-bit unsigned integer, signed integer, and float, which is the superset of the data types.
union ScalarUnion
{
    uint64_t u;
    int64_t i;
    double f;
};


////////////////////////////////////////////////////////////////////////////////
// Forward declarations

Ort::Value CreateTensorValueUsingD3DResource(
    ID3D12Device* d3dDevice,
    OrtDmlApi const& ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    std::span<const int64_t> dimensions,
    ONNXTensorElementDataType elementDataType,
    size_t elementByteSize,
    /*out opt*/ ID3D12Resource** d3dResource,
    /*out*/ void** dmlEpResourceWrapper
);

void UploadTensorData(
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocator,
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* destinationResource,
    std::span<const std::byte> sourceData
);

void DownloadTensorData(
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocatar,
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* sourceResource,
    std::span<std::byte> destinationData
);

bool BindValues(
    size_t tensorIndex,
    bool isInputTensor,
    Ort::Session& session,
    OrtDmlApi const& ortDmlApi,
    Ort::IoBinding& ioBinding,
    Ort::MemoryInfo& memoryInformation,
    ID3D12Device* d3d12Device,
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocator,
    ID3D12GraphicsCommandList* commandList,
    std::vector<Ort::Value>& tensors,
    std::vector<std::vector<std::byte>>& tensorsValues,
    std::vector<ComPtr<IUnknown>>& tensorWrappers
);

HRESULT ConvertOrtStatusToHResult(OrtStatus& status);
void PrintFirstNValues(std::span<const std::byte> data, size_t n, ONNXTensorElementDataType dataType);
void PrintTopNValues(std::span<const std::byte> data, size_t n, ONNXTensorElementDataType dataType);
void FillIntegerValues(std::span<std::byte> data, ONNXTensorElementDataType dataType, ScalarUnion value);
void GenerateValueSequence(std::span<std::byte> data, ONNXTensorElementDataType dataType);
void FormatTypedElement(void const* data, ONNXTensorElementDataType dataType, /*out*/ std::span<char> buffer);
std::string GetModuleFileName(char const* moduleName);
std::string GetTensorName(size_t index, Ort::Session const& session, bool isInput);
bool IsSupportedOnnxTensorElementDataType(ONNXTensorElementDataType dataType);
size_t ByteSizeOfOnnxTensorElementDataType(ONNXTensorElementDataType dataType);
char const* NameOfOnnxTensorElementDataType(ONNXTensorElementDataType dataType);
bool IsSignedTensorElementDataType(ONNXTensorElementDataType dataType);
ScalarUnion ReadTensorElementOfDataType(void const* data, ONNXTensorElementDataType dataType);
void WriteTensorValue(void* data, ONNXTensorElementDataType dataType, ScalarUnion);;
template <typename T> void WriteTensorElementOfDataType(void* data, ONNXTensorElementDataType dataType, T newValue);


////////////////////////////////////////////////////////////////////////////////
// Main execution


int wmain(int argc, wchar_t* argv[])
{
    if (argc <= 1)
    {
        printf(
            "Usage:\n"
            "   OnnxRuntimeDirectMLCpp.exe SomePath/SomeOnnxModel.onnx\n"
            "\n"
            "Try the included Upsample4xOpset11.onnx.\n"
        );
        return EXIT_FAILURE;
    }

    const wchar_t* modelFilePath = argv[1];

    LARGE_INTEGER startTime;
    LARGE_INTEGER d3dDeviceCreationTime;
    LARGE_INTEGER sessionCreationTime;
    LARGE_INTEGER tensorCreationTime;
    LARGE_INTEGER bindingSynchronizationTime;
    LARGE_INTEGER runStartTime;
    LARGE_INTEGER runTime;
    LARGE_INTEGER runEndTime;
    LARGE_INTEGER synchronizeOutputsTime;
    LARGE_INTEGER downloadOutputsTime;
    LARGE_INTEGER cpuFrequency;
    QueryPerformanceFrequency(&cpuFrequency);
    QueryPerformanceCounter(&startTime);

    try
    {
        ////////////////////////////////////////
        // Setup Direct3D.
        // Yeah, D3D's interface just to upload some resource data is a bit ... verbose.
        //
        // Note that if you want a specific GPU, you should call EnumAdapters.
        // Otherwise in a system with multiple GPU's (a fast discrete one and a slow
        // integrated one), you might get the slow one depending on the defaults.
        //
        // TODO: Change the adapter from nullptr to an explicit EnumAdaptersByGpu call,
        // or EnumAdapters.
        printf("Creating Direct3D device.\n");

        ComPtr<ID3D12Device> d3d12Device;
        ComPtr<ID3D12CommandQueue> commandQueue;
        ComPtr<ID3D12CommandAllocator> commandAllocator;
        ComPtr<ID3D12GraphicsCommandList> commandList;

        D3D12_COMMAND_QUEUE_DESC const commandQueueDescription =
        {
            .Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
            .Priority = 0,
            .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
            .NodeMask = 0,
        };

        THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12Device)));
        QueryPerformanceCounter(&d3dDeviceCreationTime);

        THROW_IF_FAILED(d3d12Device->CreateCommandQueue(&commandQueueDescription, IID_PPV_ARGS(&commandQueue)));
        THROW_IF_FAILED(d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));
        THROW_IF_FAILED(d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList)));

        ////////////////////////////////////////
        // Configure the model session options

        OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
        const OrtDmlApi* ortDmlApi;
        THROW_IF_ORT_FAILED(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));

        Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DirectML_Direct3D_TensorAllocation_Test"); // Note ORT_LOGGING_LEVEL_VERBOSE is useful too.
        Ort::SessionOptions sessionOptions;
        if (USE_DML_EXECUTION_PROVIDER) // The DML EP (at least for ORT 1.14) doesn't support PARALLEL or memory pattern.
        {
            sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            sessionOptions.DisableMemPattern();
        }
        sessionOptions.SetGraphOptimizationLevel(GRAPH_OPTIMIZATION_LEVEL);

        // Set any named dimensions here, if applicable:
        for (auto namedDimension : NAMED_MODEL_DIMENSIONS)
        {
            ortApi.AddFreeDimensionOverrideByName(sessionOptions, namedDimension.first, namedDimension.second); // Nop if the model has no such name.
        }

        // Test export and reload of optimized model.
        // Note this optimized model cannot be safely reloaded on a different machine or different GPU,
        // or necessarily even after installing a new driver on the same machine. All parameters must
        // match.
        if (EXPORT_OPTIMIZED_FILE)
        {
            Ort::SessionOptions sessionOptions2(sessionOptions.Clone());

            // If exporting to optimized .onnx/.ort, then be sure to disable the most aggressive optimizations which overoptimize for intention
            // of reloading the model later.
            GraphOptimizationLevel minimumOptimizationLevel = EXPORT_OPTIMIZED_FILE ? GraphOptimizationLevel::ORT_ENABLE_EXTENDED : GraphOptimizationLevel::ORT_ENABLE_ALL;
            sessionOptions2.SetGraphOptimizationLevel(std::min(GRAPH_OPTIMIZATION_LEVEL, minimumOptimizationLevel));

            sessionOptions2.SetOptimizedModelFilePath(OPTIMIZED_FILENAME);
            if (USE_DML_EXECUTION_PROVIDER)
            {
                ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions2, /*device index*/ 0);
            }
            sessionOptions.SetGraphOptimizationLevel(GRAPH_OPTIMIZATION_LEVEL);
            Ort::Session session2 = Ort::Session(ortEnvironment, modelFilePath, sessionOptions2);
            printf("Optimized version of '%S' exported to '%S'.\n", modelFilePath, OPTIMIZED_FILENAME);
            modelFilePath = OPTIMIZED_FILENAME;
        }

        if (USE_DML_EXECUTION_PROVIDER)
        {
            printf("Adding the DirectML execution provider.\n");
            ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, /*device index*/ 0);
        }
        else
        {
            // Note you may also want to add this line even if DML is being used if you're okay with CPU fallback and tire of seeing the warning.
            // Technically it's not necessary to call this, as the CPU provider will be added implicitly if no others are.
            OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, /*use_arena*/ true);
        }

        printf("DLL path ONNX Runtime: %s\n", GetModuleFileName("onnxruntime.dll").c_str());
        printf("DLL path DirectML: %s\n", GetModuleFileName("directml.dll").c_str());

        ////////////////////////////////////////
        // Load the model

        printf("Loading model '%S'.\n", modelFilePath);
        Ort::Session session(ortEnvironment, modelFilePath, sessionOptions);
        QueryPerformanceCounter(&sessionCreationTime);

        Ort::IoBinding ioBinding = Ort::IoBinding::IoBinding(session);
        const char* memoryInformationName = PASS_TENSORS_AS_D3D_RESOURCES ? "DML" : "Cpu";
        Ort::MemoryInfo memoryInformation(memoryInformationName, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
        Ort::Allocator deviceAllocator(session, memoryInformation);

        ////////////////////////////////////////
        // Create input and output tensors

        std::vector<Ort::Value> inputTensors;
        std::vector<Ort::Value> outputTensors;
        std::vector<std::vector<std::byte>> inputTensorValues; // Preserve the values since the CPU tensor just lightly wraps them.
        std::vector<std::vector<std::byte>> outputTensorValues;
        std::vector<ComPtr<IUnknown>> inputTensorWrappers; // Preserve lifetime of tensors in the Ort::Value, which doesn't seem to hold a reference.
        std::vector<ComPtr<IUnknown>> outputTensorWrappers;

        size_t const inputCount = session.GetInputCount();
        size_t const outputCount = session.GetOutputCount();

        // Loop though inputs and outputs.
        for (int bindingPass = 0; bindingPass < 2; ++bindingPass)
        {
            bool const isInputTensor = (bindingPass == 0);
            size_t const tensorCount = isInputTensor ? inputCount : outputCount;

            for (size_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex)
            {
                BindValues(
                    tensorIndex,
                    isInputTensor,
                    session,
                    *ortDmlApi,
                    ioBinding,
                    memoryInformation,
                    d3d12Device,
                    commandQueue,
                    commandAllocator,
                    commandList,
                    isInputTensor ? inputTensors : outputTensors,
                    isInputTensor ? inputTensorValues : outputTensorValues,
                    isInputTensor ? inputTensorWrappers : outputTensorWrappers
                );
            }
        }

        QueryPerformanceCounter(&tensorCreationTime);

        // Wait for any inputs to finish uploading, in case the sources were CPU tensors.
        ioBinding.SynchronizeInputs();
        QueryPerformanceCounter(&bindingSynchronizationTime);

        ////////////////////////////////////////
        // Begin execution

        Ort::RunOptions runOptions;

        printf("Beginning execution.\n");
        QueryPerformanceCounter(&runStartTime);
        session.Run(runOptions, ioBinding);
        QueryPerformanceCounter(&runTime);
        ioBinding.SynchronizeOutputs();
        QueryPerformanceCounter(&synchronizeOutputsTime);
        runEndTime = synchronizeOutputsTime;
        printf("Finished execution.\n");

        ////////////////////////////////////////
        // Read computed outputs

        size_t const outputTensorCount = outputTensors.size();
        assert(outputTensors.size() == outputTensorValues.size());

        // If GPU outputs, then read the values back from the device.
        // If CPU outputs, then the values were already written in-place to outputTensorValues by ONNX Runtime.
        if (PASS_TENSORS_AS_D3D_RESOURCES)
        {
            for (size_t i = 0; i < outputTensorCount; ++i)
            {
                assert(outputTensors[i].IsTensor());
                ComPtr<ID3D12Resource> d3dResource;
                THROW_IF_ORT_FAILED(ortDmlApi->GetD3D12ResourceFromAllocation(deviceAllocator, outputTensorWrappers[i], &d3dResource));
                DownloadTensorData(
                    commandQueue,
                    commandAllocator,
                    commandList,
                    d3dResource,
                    outputTensorValues[i]
                );
            }
        }

        QueryPerformanceCounter(&downloadOutputsTime);
        runEndTime = synchronizeOutputsTime;
        printf("Downloaded output tensor.\n");

        ////////////////////////////////////////
        // Print timings

        auto printDuration = [=](char const* message, LARGE_INTEGER nextTime, LARGE_INTEGER previousTime = {}) mutable
        {
            if (previousTime.QuadPart == 0)
            {
                previousTime = startTime;
            }
            double durationMs = static_cast<double>(nextTime.QuadPart - previousTime.QuadPart);
            durationMs /= static_cast<double>(cpuFrequency.QuadPart);
            durationMs *= 1000.0;
            printf("%s % 12.6fms\n", message, durationMs);

            startTime = nextTime;
        };
        printDuration("D3D device creation time ....", d3dDeviceCreationTime);
        printDuration("session creation time .......", sessionCreationTime);
        printDuration("tensor creation time ........", tensorCreationTime);
        printDuration("binding synchronization time ", bindingSynchronizationTime);
        printDuration("run time ....................", runTime);
        printDuration("synchronize outputs time ....", synchronizeOutputsTime);
        printDuration("run+synchronize time.........", runEndTime, runStartTime);
        printDuration("total time...................", synchronizeOutputsTime, startTime);

        ////////////////////////////////////////
        // Print output values

        size_t const inputTensorCount = inputTensors.size();
        for (size_t i = 0; i < inputTensorCount; ++i)
        {
            assert(inputTensors[i].IsTensor());
            printf("Input #%zu:\n", i);
            Ort::TensorTypeAndShapeInfo typeAndShapeInfo = inputTensors[i].GetTensorTypeAndShapeInfo();
            PrintFirstNValues(inputTensorValues[i], 10, typeAndShapeInfo.GetElementType());
        }
        for (size_t i = 0; i < outputTensorCount; ++i)
        {
            assert(outputTensors[i].IsTensor());
            printf("Output #%zu:\n", i);
            Ort::TensorTypeAndShapeInfo typeAndShapeInfo = outputTensors[i].GetTensorTypeAndShapeInfo();
            PrintFirstNValues(outputTensorValues[i], 10, typeAndShapeInfo.GetElementType());
            PrintTopNValues(outputTensorValues[i], 10, typeAndShapeInfo.GetElementType());
        }
    }
    catch (Ort::Exception const& exception)
    {
        printf("Error running model inference: %s\n", exception.what());
        return EXIT_FAILURE;
    }
    catch (std::exception const& exception)
    {
        printf("Error running model inference: %s\n", exception.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


bool BindValues(
    size_t tensorIndex,
    bool isInputTensor,
    Ort::Session& session,
    OrtDmlApi const& ortDmlApi,
    Ort::IoBinding& ioBinding,
    Ort::MemoryInfo& memoryInformation,
    ID3D12Device* d3d12Device,
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocator,
    ID3D12GraphicsCommandList* commandList,
    std::vector<Ort::Value>& tensors,
    std::vector<std::vector<std::byte>>& tensorsValues,
    std::vector<ComPtr<IUnknown>>& tensorWrappers
    )
{
    std::string tensorName = GetTensorName(tensorIndex, session, isInputTensor);
    Ort::TypeInfo typeInfo = isInputTensor ? session.GetInputTypeInfo(tensorIndex) : session.GetOutputTypeInfo(tensorIndex);
    if (typeInfo.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR)
    {
        printf("Unknown binding type for '%s'\n", tensorName.c_str());
        return false; // Can't handle this type. So skip it.
    }

    // Get the tensor shape and type.
    // Note when computing the element count that it's unsafe to call ORT's shapeInfo.GetElementCount()
    // because you may get a SafeInt overflow if there are free dimensions, which are treated as -1's.
    // So replace those with 1's first.
#if ORT_API_VERSION >= 15
    Ort::ConstTensorTypeAndShapeInfo shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
#else
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
#endif
    ONNXTensorElementDataType const tensorDataType = shapeInfo.GetElementType();
    if (!IsSupportedOnnxTensorElementDataType(tensorDataType))
    {
        printf("Unsupported tensor data type %d '%s' for '%s'\n", int32_t(tensorDataType), NameOfOnnxTensorElementDataType(tensorDataType), tensorName.c_str());
        return false; // Can't handle this type. So skip it.
    }

    std::vector<int64_t> tensorShape = shapeInfo.GetShape();
    std::for_each(tensorShape.begin(), tensorShape.end(), [](int64_t& i) {i = std::max(i, int64_t(1)); });
    size_t const tensorElementCount = static_cast<size_t>(GetElementCount(tensorShape));

    // Allocate values for tensor.
    Ort::Value tensor(nullptr);
    ComPtr<IUnknown> executionProviderTensorWrapper;
    std::vector<std::byte> tensorValues(tensorElementCount * ByteSizeOfOnnxTensorElementDataType(tensorDataType));

    // Fill input tensor with an increasing sequence.
    if (isInputTensor)
    {
        if (FILL_INPUTS_INCREASING_SEQUENCE)
        {
            GenerateValueSequence(/*out*/ tensorValues, tensorDataType); // Alternate increasing fill pattern.
        }
        else
        {
            ScalarUnion zeroValue = {};
            FillIntegerValues(/*out*/ tensorValues, tensorDataType, zeroValue);
        }
    }

    char const* inputOrOutputString = isInputTensor ? "input" : "output";
    printf("Binding %s tensor '%s', %s[%zu].\n", inputOrOutputString, tensorName.c_str(), NameOfOnnxTensorElementDataType(tensorDataType), tensorElementCount);

    if (PASS_TENSORS_AS_D3D_RESOURCES)
    {
        // Create D3D resource for input/output.
        ComPtr<ID3D12Resource> d3dResource;
        tensor = CreateTensorValueUsingD3DResource(
            d3d12Device,
            ortDmlApi,
            memoryInformation,
            tensorShape,
            tensorDataType,
            ByteSizeOfOnnxTensorElementDataType(tensorDataType),
            /*out*/ &d3dResource,
            /*out*/ IID_PPV_ARGS_Helper(executionProviderTensorWrapper.GetAddressOf())
        );

        if (isInputTensor)
        {
            // Upload it to the GPU, and wait for completion. Note a more efficient approach would enqueue and upload
            // them all at once rather than waiting for each one to finish.
            UploadTensorData(commandQueue, commandAllocator, commandList, d3dResource, tensorValues);
        }
    }
    else // CPU tensor
    {
        tensor = Ort::Value::CreateTensor(
            memoryInformation,
            reinterpret_cast<void*>(tensorValues.data()),
            tensorValues.size(),
            tensorShape.data(),
            tensorShape.size(),
            tensorDataType
        );
    }

    if (isInputTensor)
    {
        ioBinding.BindInput(tensorName.c_str(), tensor);
    }
    else // Output
    {
        ioBinding.BindOutput(tensorName.c_str(), tensor);
    }

    tensors.push_back(std::move(tensor));
    tensorsValues.push_back(std::move(tensorValues));
    tensorWrappers.push_back(std::move(executionProviderTensorWrapper));

    return true;
}


ComPtr<ID3D12Resource> CreateD3D12ResourceOfByteSize(
    ID3D12Device* d3dDevice,
    size_t resourceByteSize,
    D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT,
    D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    )
{
    resourceByteSize = std::max(resourceByteSize, size_t(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));

    // DML needs the resources' sizes to be a multiple of 4 bytes
    (resourceByteSize += 3) &= ~3;

    D3D12_HEAP_PROPERTIES const heapProperties =
    {
        .Type = heapType, // Default to D3D12_HEAP_TYPE_DEFAULT.
        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1
    };

    D3D12_RESOURCE_DESC const resourceDesc =
    {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = static_cast<uint64_t>(resourceByteSize),
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {1, 0},
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = resourceFlags // Default to D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
    };

    ComPtr<ID3D12Resource> gpuResource;
    THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        resourceState, // Default to D3D12_RESOURCE_STATE_COMMON
        nullptr,
        __uuidof(ID3D12Resource),
        /*out*/ &gpuResource
    ));

    return gpuResource;
}


ComPtr<ID3D12Resource> CreateD3D12ResourceForTensor(
    ID3D12Device* d3dDevice,
    size_t elementByteSize,
    std::span<const int64_t> tensorDimensions
    )
{
    // Try to allocate the backing memory for the caller
    auto bufferSize = GetElementCount(tensorDimensions);
    assert(bufferSize > 0);
    assert(elementByteSize > 0);
    size_t bufferByteSize = static_cast<size_t>(bufferSize * elementByteSize);

    return CreateD3D12ResourceOfByteSize(d3dDevice, bufferByteSize);
}


Ort::Value CreateTensorValueFromExistingD3DResource(
    OrtDmlApi const& ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    ID3D12Resource* d3dResource,
    std::span<const int64_t> tensorDimensions,
    ONNXTensorElementDataType elementDataType,
    /*out*/ void** dmlEpResourceWrapper // Must stay alive with Ort::Value.
    )
{
    *dmlEpResourceWrapper = nullptr;

    void* dmlAllocatorResource;
    THROW_IF_ORT_FAILED(ortDmlApi.CreateGPUAllocationFromD3DResource(d3dResource, &dmlAllocatorResource));
    auto deleter = [&](void*) {ortDmlApi.FreeGPUAllocation(dmlAllocatorResource); };
    deleting_unique_ptr<void> dmlAllocatorResourceCleanup(dmlAllocatorResource, deleter);

    size_t tensorByteSize = static_cast<size_t>(d3dResource->GetDesc().Width);
    Ort::Value newValue(
        Ort::Value::CreateTensor(
            memoryInformation,
            dmlAllocatorResource,
            tensorByteSize,
            tensorDimensions.data(),
            tensorDimensions.size(),
            elementDataType
        )
    );

    // Return values and the wrapped resource.
    // TODO: Is there some way to get Ort::Value to just own the D3DResource
    // directly so that it gets freed after execution or session destruction?
    *dmlEpResourceWrapper = dmlAllocatorResource;
    dmlAllocatorResourceCleanup.release();

    return newValue;
}


Ort::Value CreateTensorValueUsingD3DResource(
    ID3D12Device* d3dDevice,
    OrtDmlApi const& ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    std::span<const int64_t> tensorDimensions,
    ONNXTensorElementDataType elementDataType,
    size_t elementByteSize,
    /*out opt*/ ID3D12Resource** d3dResource,
    /*out*/ void** dmlEpResourceWrapper // Must stay alive with Ort::Value.
    )
{
    // Create empty resource (values don't matter because we won't read them back anyway).
    ComPtr<ID3D12Resource> localD3dResource = CreateD3D12ResourceForTensor(
        d3dDevice,
        elementByteSize,
        tensorDimensions
    );
    if (d3dResource)
    {
        localD3dResource->AddRef();
        *d3dResource = localD3dResource;
    }

    return CreateTensorValueFromExistingD3DResource(
        ortDmlApi,
        memoryInformation,
        localD3dResource,
        tensorDimensions,
        elementDataType,
        /*out*/ dmlEpResourceWrapper
    );
}


void WaitForQueueToComplete(ID3D12CommandQueue* queue)
{
    ComPtr<ID3D12Device> device;
    THROW_IF_FAILED(queue->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));
    ComPtr<ID3D12Fence> fence;
    THROW_IF_FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
    THROW_IF_FAILED(queue->Signal(fence, 1));
    THROW_IF_FAILED(fence->SetEventOnCompletion(1, nullptr));
}


void UploadTensorData(
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocator,
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* destinationResource,
    std::span<const std::byte> sourceData
    )
{
    // Get the size of the resource.
    ComPtr<ID3D12Device> d3d12Device;
    THROW_IF_FAILED(commandQueue->GetDevice(IID_PPV_ARGS(&d3d12Device)));
    D3D12_RESOURCE_DESC resourceDesc = destinationResource->GetDesc();
    assert(resourceDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
    const size_t dataSizeInBytes = static_cast<size_t>(resourceDesc.Width);

    // Create intermediate upload resource visible to both CPU and GPU.
    ComPtr<ID3D12Resource> uploadBuffer = CreateD3D12ResourceOfByteSize(d3d12Device, dataSizeInBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);

    // Copy CPU-side data to shared memory that is both CPU and GPU visible.
    size_t clampedDataByteSize = std::min(dataSizeInBytes, sourceData.size());
    std::byte* uploadBufferData = nullptr;
    THROW_IF_FAILED(uploadBuffer->Map(0, nullptr, reinterpret_cast<void**>(&uploadBufferData)));
    memcpy(uploadBufferData, sourceData.data(), clampedDataByteSize);
    uploadBuffer->Unmap(0, nullptr);

    D3D12_RESOURCE_BARRIER const resourceBarrier =
    {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition =
        {
            .pResource = destinationResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST,
            .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        },
    };

    // Issue deferred command to copy from the intermediate shared resource to the final GPU resource,
    // and then execute the commands.
    commandList->CopyResource(destinationResource, uploadBuffer);
    commandList->ResourceBarrier(1, &resourceBarrier);
    THROW_IF_FAILED(commandList->Close());
    ID3D12CommandList* commandLists[] = { commandList };
    commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
    WaitForQueueToComplete(commandQueue);

    THROW_IF_FAILED(commandAllocator->Reset());
    THROW_IF_FAILED(commandList->Reset(commandAllocator, nullptr));
}


void DownloadTensorData(
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocator,
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* sourceResource,
    std::span<std::byte> destinationData
    )
{
    // Get the size of the resource.
    ComPtr<ID3D12Device> d3d12Device;
    THROW_IF_FAILED(commandQueue->GetDevice(IID_PPV_ARGS(d3d12Device.GetAddressOf())));
    D3D12_RESOURCE_DESC resourceDesc = sourceResource->GetDesc();
    assert(resourceDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
    const size_t dataSizeInBytes = static_cast<size_t>(resourceDesc.Width);

    // Create intermediate upload resource visible to both CPU and GPU.
    ComPtr<ID3D12Resource> downloadBuffer = CreateD3D12ResourceOfByteSize(d3d12Device, dataSizeInBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);

    D3D12_RESOURCE_BARRIER const resourceBarrier =
    {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition =
        {
            .pResource = sourceResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            .StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE,
        },
    };

    // Copy GPU data into the download buffer.
    commandList->ResourceBarrier(1, &resourceBarrier);
    commandList->CopyResource(downloadBuffer, sourceResource);
    THROW_IF_FAILED(commandList->Close());
    ID3D12CommandList* commandLists[] = { commandList };
    commandQueue->ExecuteCommandLists(static_cast<uint32_t>(std::size(commandLists)), commandLists);
    WaitForQueueToComplete(commandQueue);
    THROW_IF_FAILED(commandAllocator->Reset());
    THROW_IF_FAILED(commandList->Reset(commandAllocator, nullptr));

    // Copy from shared GPU/CPU memory to ordinary system RAM.
    size_t clampedDataByteSize = std::min(dataSizeInBytes, destinationData.size());
    std::byte* sourceData = nullptr;
    D3D12_RANGE range = {0, clampedDataByteSize };
    THROW_IF_FAILED(downloadBuffer->Map(0, &range, reinterpret_cast<void**>(&sourceData)));
    memcpy(destinationData.data(), sourceData, clampedDataByteSize);
    downloadBuffer->Unmap(0, nullptr);
}


bool IsSupportedOnnxTensorElementDataType(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:   return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:        return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:        return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:      return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:     return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  return false;
    default: return 1;
    }
}


size_t ByteSizeOfOnnxTensorElementDataType(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:   return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:       return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:       return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:     return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:       return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:       return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:       return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   return 8; // 32*2
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  return 16;
    default: return 1;
    }
}


char const* NameOfOnnxTensorElementDataType(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:   return "undefined";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:        return "bool8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:       return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:        return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:      return "char8[]";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:      return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:       return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:     return "float16m10e5s1";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    return "float16m8e7s1";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:       return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:       return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:       return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   return "float32x2";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  return "float64x2";
    default: return "unknown";
    }
}


bool IsSignedTensorElementDataType(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:        return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:       return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:        return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:      return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:     return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default: return false;
    }
}


// Read the data at the given pointer as the type, expanding it to 64 bits.
ScalarUnion ReadTensorElementOfDataType(void const* data, ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:        return ScalarUnion{ .u = *reinterpret_cast<bool const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:       return ScalarUnion{ .u = *reinterpret_cast<uint8_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:        return ScalarUnion{ .i = *reinterpret_cast<int8_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:      return ScalarUnion{ .u = *reinterpret_cast<uint16_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:       return ScalarUnion{ .i = *reinterpret_cast<int16_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:     return ScalarUnion{ .f = *reinterpret_cast<float16m10e5s1_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    return ScalarUnion{ .f = *reinterpret_cast<float16m7e8s1_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:       return ScalarUnion{ .i = *reinterpret_cast<int32_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      return ScalarUnion{ .u = *reinterpret_cast<uint32_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:       return ScalarUnion{ .f = *reinterpret_cast<float32_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      return ScalarUnion{ .u = *reinterpret_cast<uint64_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:       return ScalarUnion{ .i = *reinterpret_cast<int64_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      return ScalarUnion{ .f = *reinterpret_cast<float64_t const*>(data) };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   return ScalarUnion{ .f = reinterpret_cast<std::pair<float32_t, float32_t> const*>(data)->first };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  return ScalarUnion{ .f = reinterpret_cast<std::pair<float64_t, float64_t> const*>(data)->first };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default: return ScalarUnion{ .u = 0 };
    }
}


void WriteTensorValue(
    void* data, // Must point to memory that has at least the number of bytes specified by the dataType.
    ONNXTensorElementDataType dataType,
    ScalarUnion value
)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:    *reinterpret_cast<float16m10e5s1_t*>(data) = static_cast<float>(value.f); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:   *reinterpret_cast<float16m7e8s1_t*>(data) = static_cast<float>(value.f); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:      *reinterpret_cast<float*>   (data) = static_cast<float>   (value.f); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:     *reinterpret_cast<double*>  (data) = static_cast<double>  (value.f); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:  *reinterpret_cast<float*>   (data) = static_cast<float>   (value.f); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: *reinterpret_cast<double*>  (data) = static_cast<double>  (value.f); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:       *reinterpret_cast<bool*>    (data) = static_cast<bool>    (value.u); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:      *reinterpret_cast<uint8_t*> (data) = static_cast<uint8_t> (value.u); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:     *reinterpret_cast<uint16_t*>(data) = static_cast<uint16_t>(value.u); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:     *reinterpret_cast<uint32_t*>(data) = static_cast<uint32_t>(value.u); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:     *reinterpret_cast<uint64_t*>(data) = static_cast<uint64_t>(value.u); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:       *reinterpret_cast<int8_t*>  (data) = static_cast<int8_t>  (value.i); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:      *reinterpret_cast<int16_t*> (data) = static_cast<int16_t> (value.i); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:      *reinterpret_cast<int32_t*> (data) = static_cast<int32_t> (value.i); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:      *reinterpret_cast<int64_t*> (data) = static_cast<int64_t> (value.i); break;
    default: throw std::ios::failure("Unsupported data type for tensor.");
    }
}


// Write a value to the given pointer as the type.
template <typename T>
void WriteTensorElementOfDataType(void* data, ONNXTensorElementDataType dataType, T newValue)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:        *reinterpret_cast<bool*>(data) = static_cast<bool>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:       *reinterpret_cast<uint8_t*>(data) = static_cast<uint8_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:        *reinterpret_cast<int8_t*>(data) = static_cast<int8_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:      *reinterpret_cast<uint16_t*>(data) = static_cast<uint16_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:       *reinterpret_cast<int16_t*>(data) = static_cast<int16_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:     *reinterpret_cast<float16m10e5s1_t*>(data) = static_cast<float16m10e5s1_t>(float32_t(newValue)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    *reinterpret_cast<float16m7e8s1_t*>(data) = static_cast<float16m7e8s1_t>(float32_t(newValue)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:       *reinterpret_cast<int32_t*>(data) = static_cast<int32_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      *reinterpret_cast<uint32_t*>(data) = static_cast<uint32_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:       *reinterpret_cast<float32_t*>(data) = static_cast<float>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      *reinterpret_cast<uint64_t*>(data) = static_cast<uint64_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:       *reinterpret_cast<int64_t*>(data) = static_cast<int64_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      *reinterpret_cast<float64_t*>(data) = static_cast<double>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   reinterpret_cast<std::pair<float32_t, float32_t>*>(data)->first = static_cast<float32_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  reinterpret_cast<std::pair<float64_t, float64_t>*>(data)->first = static_cast<float64_t>(newValue); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default: break; // Do nothing.
    }
}


void FormatTypedElement(void const* data, ONNXTensorElementDataType dataType, /*out*/ std::span<char> buffer)
{
    if (buffer.empty())
        return;

    ScalarUnion value = ReadTensorElementOfDataType(data, dataType);
    std::to_chars_result charsResult;
    char* dataEnd = buffer.data() + buffer.size();

    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        charsResult = std::to_chars(buffer.data(), dataEnd, value.u);
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        charsResult = std::to_chars(buffer.data(), dataEnd, value.i);
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        charsResult = std::to_chars(buffer.data(), dataEnd, value.f);
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default:
        strncpy_s(buffer.data(), buffer.size(), "unsupported", buffer.size() - 1);
        return;
    }

    // Ensure null terminator.
    if (charsResult.ptr == dataEnd)
    {
        --charsResult.ptr;
    }
    *charsResult.ptr = '\0';
}


std::string GetTensorName(size_t index, Ort::Session const& session, bool isInput)
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr name = isInput ? session.GetInputNameAllocated(index, allocator) : session.GetOutputNameAllocated(index, allocator);
    std::string returnName(name.get());
    name.release();
    return returnName;
}


std::string GetModuleFileName(char const* moduleName)
{
    HMODULE module = GetModuleHandleA(moduleName);
    if (module == nullptr)
    {
        return "";
    }

    std::string fileName(MAX_PATH + 1, '\0');
    GetModuleFileNameA(module, /*out*/ fileName.data(), MAX_PATH);
    return fileName;
}


void FillIntegerValues(std::span<std::byte> data, ONNXTensorElementDataType dataType, ScalarUnion value)
{
    size_t const bytesPerElement = ByteSizeOfOnnxTensorElementDataType(dataType);
    size_t const elementCount = data.size_bytes() / bytesPerElement;

    for (size_t i = 0, ci = elementCount; i < ci; ++i)
    {
        WriteTensorValue(&data[i * bytesPerElement], dataType, value);
    }
}


void GenerateValueSequence(std::span<std::byte> data, ONNXTensorElementDataType dataType)
{
    size_t const bytesPerElement = ByteSizeOfOnnxTensorElementDataType(dataType);
    size_t const elementCount = data.size_bytes() / bytesPerElement;

    for (size_t i = 0, ci = elementCount; i < ci; ++i)
    {
        WriteTensorElementOfDataType(&data[i * bytesPerElement], dataType, static_cast<uint32_t>(i));
    }
}


void PrintFirstNValues(std::span<const std::byte> data, size_t n, ONNXTensorElementDataType dataType)
{
    size_t const bytesPerElement = ByteSizeOfOnnxTensorElementDataType(dataType);
    size_t const elementCount = data.size_bytes() / bytesPerElement;
    n = std::min(n, elementCount);

    char numberBuffer[40];

    // Print the first 10 and top 10 results.
    printf("  First %zu/%zu results:\n", n, elementCount);
    for (size_t i = 0; i < n; ++i)
    {
        FormatTypedElement(&data[i * bytesPerElement], dataType, /*out*/ numberBuffer);
        printf("    element[%zu] = %s\n", i, numberBuffer);
    }
}


void PrintTopNValues(std::span<const std::byte> data, size_t n, ONNXTensorElementDataType dataType)
{
    size_t const bytesPerElement = ByteSizeOfOnnxTensorElementDataType(dataType);
    size_t const elementCount = data.size_bytes() / bytesPerElement;
    n = std::min(n, elementCount);

    char numberBuffer[40];

    size_t maxSortSize = 10'000;
    if (elementCount > maxSortSize)
        return;

    printf("  Top %zu/%zu results:\n", n, elementCount);

    std::vector<uint32_t> indices(elementCount, 0);
    std::iota(indices.begin(), indices.end(), 0);

    std::byte const* dataPointer = data.data();
    // Determine whether the data type is signed ahead of time so that unsigned comparisons
    // correctly place positive numbers before negative ones. All comparisons regardless of
    // data type are done bitwise (which is safe even for floating point numbers).
    uint64_t const signInversion = IsSignedTensorElementDataType(dataType) ? (uint64_t(1) << 63) : 0;

    sort(
        indices.begin(),
        indices.end(),
        [&, dataPointer, bytesPerElement, dataType](uint32_t a, uint32_t b)
        {
            ScalarUnion valueA = ReadTensorElementOfDataType(&dataPointer[a * bytesPerElement], dataType);
            ScalarUnion valueB = ReadTensorElementOfDataType(&dataPointer[b * bytesPerElement], dataType);
            valueA.u ^= signInversion;
            valueB.u ^= signInversion;
            return valueA.u > valueB.u;
        }
    );

    for (size_t i = 0; i < n; ++i)
    {
        FormatTypedElement(&data[indices[i] * bytesPerElement], dataType, /*out*/ numberBuffer);
        printf("    element[%u] = %s\n", indices[i], numberBuffer);
    }
}


// Converts an OrtStatus to an HRESULT, freeing the OrtStatus.
//
// If ORT returns a non-null OrtStatusPtr, it must be freed to avoid leaking memory.
// Unfortunately the OrtStatus object itself offers no reference counting lifetime
// management methods to easily release it (unlike all COM objects with IUnknown::Release
// or any methods to get more information, meaning you either have to ferry the OrtApi
// interface around or (for C++) use a global singleton. Since this is a simple C++ app,
// and we're not doing anything exotic like mixing multiple ONNX Runtime versions in a
// single process, just use the global singleton.
//
HRESULT ConvertOrtStatusToHResult(OrtStatus& status)
{
    // std::string errorMessage = ortApi.GetErrorMessage(&status);
    OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
    OrtErrorCode ortErrorCode = ortApi.GetErrorCode(&status);
    ortApi.ReleaseStatus(&status);

    switch (ortErrorCode)
    {
    // POSIX error codes really are inadequate to convey common errors, like even just bad file format :/.
    // Consider using a custom error domain.
    case OrtErrorCode::ORT_OK:               return S_OK;
    case OrtErrorCode::ORT_INVALID_ARGUMENT: return E_INVALIDARG;
    case OrtErrorCode::ORT_NO_SUCHFILE:      return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    case OrtErrorCode::ORT_NOT_IMPLEMENTED:  return E_NOTIMPL;
    case OrtErrorCode::ORT_FAIL:
    case OrtErrorCode::ORT_NO_MODEL:
    case OrtErrorCode::ORT_ENGINE_ERROR:
    case OrtErrorCode::ORT_RUNTIME_EXCEPTION:
    case OrtErrorCode::ORT_INVALID_PROTOBUF:
    case OrtErrorCode::ORT_MODEL_LOADED:
    case OrtErrorCode::ORT_INVALID_GRAPH:
    case OrtErrorCode::ORT_EP_FAIL:
    default:                                return E_FAIL;
    }
}
