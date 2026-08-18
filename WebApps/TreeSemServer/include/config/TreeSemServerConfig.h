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
    bool agentEnabled{true};
    std::string agentUrl{"http://127.0.0.1:8091/v1/agent/runs"};
    std::string agentServiceSecret;
    long agentConnectTimeoutMs{500};
    long agentRequestTimeoutMs{30000};
    std::size_t agentWorkerCount{4};
    std::size_t agentQueueCapacity{64};
    std::size_t agentContextMessages{12};
    std::size_t agentMessageMaxCharacters{4000};
    std::size_t agentResponseMaxBytes{1024 * 1024};
    bool knowledgeEnabled{true};
    std::string knowledgeJwtSecret;
    long knowledgeTokenTtlSeconds{120};
    bool authRequired{true};
    std::string deploymentEnvironment{"local"};
    std::string accessJwtSecret;
    std::string capabilityJwtSecret;
    long accessTokenTtlSeconds{900};
    long refreshTokenTtlSeconds{604800};
    long capabilityTokenTtlSeconds{120};
    bool refreshCookieSecure{false};
    std::string allowedOrigins{"http://127.0.0.1:3000"};
    std::size_t authWorkerCount{2};
    std::size_t authQueueCapacity{32};
    bool observabilityEnabled{true};
    bool metricsEnabled{true};
    std::string metricsBearerToken;
    double traceSampleRate{1.0};
    long slowRequestMs{1000};

    static TreeSemServerConfig load(int argc, char* argv[]);
};

} // namespace config
} // namespace treesem
