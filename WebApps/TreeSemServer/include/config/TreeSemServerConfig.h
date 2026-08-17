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

enum class StorageBackend
{
    MySql,
    Memory,
};

std::string toString(ModelBackend backend);
std::string toString(StorageBackend backend);

struct TreeSemServerConfig
{
    int listenPort{8080};
    int ioThreadCount{2};
    std::string modelAdapterUrl{"http://127.0.0.1:18081/v1/predict"};
    long modelConnectTimeoutMs{500};
    long modelRequestTimeoutMs{5000};
    std::size_t inferenceWorkerCount{2};
    std::size_t inferenceQueueCapacity{32};
    std::size_t databaseWorkerCount{4};
    std::size_t databaseQueueCapacity{64};
    ModelBackend modelBackend{ModelBackend::OnnxFallback};
    std::string servingBundleDirectory;
    StorageBackend storageBackend{StorageBackend::MySql};
    std::string databaseHost{"127.0.0.1"};
    int databasePort{3307};
    std::string databaseName{"treesem"};
    std::string databaseUser{"treesem_app"};
    std::string databasePassword;
    std::size_t databasePoolSize{8};
    long databaseAcquireTimeoutMs{500};
    long databaseConnectTimeoutMs{1000};
    long databaseReadTimeoutMs{2000};
    long databaseWriteTimeoutMs{2000};
    long sessionTtlSeconds{3600};
    bool cookieSecure{false};

    static TreeSemServerConfig load(int argc, char* argv[]);
};

} // namespace config
} // namespace treesem
