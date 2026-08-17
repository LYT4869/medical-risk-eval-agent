#pragma once

#include <cstddef>
#include <string>

namespace treesem
{
namespace config
{

enum class ModelBackend
{
    Remote,
    Onnx,
    OnnxFallback,
    Shadow,
};

std::string toString(ModelBackend backend);

struct TreeSemServerConfig
{
    int listenPort{8080};
    int ioThreadCount{2};
    std::string modelAdapterUrl{"http://127.0.0.1:18081/v1/predict"};
    long modelConnectTimeoutMs{500};
    long modelRequestTimeoutMs{5000};
    std::size_t inferenceWorkerCount{2};
    std::size_t inferenceQueueCapacity{32};
    ModelBackend modelBackend{ModelBackend::OnnxFallback};
    std::string servingBundleDirectory;

    static TreeSemServerConfig load(int argc, char* argv[]);
};

} // namespace config
} // namespace treesem
