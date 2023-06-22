# ONNX Runtime DirectML Execution Provider Sample

Dwayne Robinson 2022-04-06..2023-06-22

## What is it?
Just testing the DirectML execution provider in ONNX Runtime via D3D resources instead of CPU tensors (which incur synchronization costs). All the code is in `MainFullExample.cpp`, with `MainMinimalExample.cpp` being a much simpler introductory sample (no GPU binding and hard-coded to the given model).

## Usage
- **OS**: Windows 10+.
- **GPU**: DirectX 12 compute capable.
- **Running**: Command line app.
- **License**: [License.txt](License.txt) tldr: Do whatever you want with the code.

## Building
- Standard Visual Studio msbuild project.
- The Nuget dependencies [onnxruntime.dll 1.13](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/) and [directml.dll 1.10.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/) should automatically be copied into your build folder (and after building, it all just works 🤞).

## Related
- https://github.com/microsoft/DirectML
- https://docs.microsoft.com/en-us/windows/ai/directml/dml-intro
- https://www.nuget.org/packages/Microsoft.AI.DirectML/1.10.1
- ttps://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/
- https://github.com/microsoft/onnxruntime/
- https://onnx.ai/
