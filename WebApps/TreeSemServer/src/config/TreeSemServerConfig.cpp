#include "config/TreeSemServerConfig.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <stdexcept>

namespace treesem
{
namespace config
{
namespace
{

std::string environmentOr(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr || *value == '\0' ? fallback : value;
}

long parsePositiveLong(const std::string& value, const std::string& name)
{
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end != '\0' || parsed <= 0)
    {
        throw std::invalid_argument(name + " must be a positive integer");
    }
    return parsed;
}

int parsePort(const std::string& value)
{
    const long port = parsePositiveLong(value, "listen port");
    if (port > 65535)
    {
        throw std::invalid_argument("listen port must be in the range 1..65535");
    }
    return static_cast<int>(port);
}

std::size_t parsePositiveSize(const char* name, std::size_t fallback)
{
    const long parsed = parsePositiveLong(
        environmentOr(name, std::to_string(fallback)), name);
    return static_cast<std::size_t>(parsed);
}

ModelBackend parseBackend(const std::string& value)
{
    if (value == "remote")
    {
        return ModelBackend::Remote;
    }
    if (value == "onnx")
    {
        return ModelBackend::Onnx;
    }
    if (value == "onnx_fallback")
    {
        return ModelBackend::OnnxFallback;
    }
    if (value == "shadow")
    {
        return ModelBackend::Shadow;
    }
    throw std::invalid_argument(
        "TREESEM_MODEL_BACKEND must be remote, onnx, onnx_fallback or shadow");
}

StorageBackend parseStorageBackend(const std::string& value)
{
    if (value == "mysql") return StorageBackend::MySql;
    if (value == "memory") return StorageBackend::Memory;
    throw std::invalid_argument(
        "TREESEM_STORAGE_BACKEND must be mysql or memory");
}

bool parseBoolean(const std::string& value, const std::string& name)
{
    if (value == "true" || value == "1") return true;
    if (value == "false" || value == "0") return false;
    throw std::invalid_argument(name + " must be true or false");
}

} // namespace

std::string toString(ModelBackend backend)
{
    switch (backend)
    {
    case ModelBackend::Remote:
        return "remote";
    case ModelBackend::Onnx:
        return "onnx";
    case ModelBackend::OnnxFallback:
        return "onnx_fallback";
    case ModelBackend::Shadow:
        return "shadow";
    }
    throw std::invalid_argument("unknown model backend");
}

std::string toString(StorageBackend backend)
{
    switch (backend)
    {
    case StorageBackend::MySql: return "mysql";
    case StorageBackend::Memory: return "memory";
    }
    throw std::invalid_argument("unknown storage backend");
}

TreeSemServerConfig TreeSemServerConfig::load(int argc, char* argv[])
{
    if (argc > 2)
    {
        throw std::invalid_argument("usage: treesem_server [port]");
    }

    TreeSemServerConfig config;
    if (argc == 2)
    {
        config.listenPort = parsePort(argv[1]);
    }
    config.modelAdapterUrl = environmentOr(
        "TREESEM_MODEL_ADAPTER_URL", config.modelAdapterUrl);
    if (config.modelAdapterUrl.empty())
    {
        throw std::invalid_argument("TREESEM_MODEL_ADAPTER_URL must not be empty");
    }
    config.modelConnectTimeoutMs = parsePositiveLong(
        environmentOr(
            "TREESEM_MODEL_CONNECT_TIMEOUT_MS",
            std::to_string(config.modelConnectTimeoutMs)),
        "TREESEM_MODEL_CONNECT_TIMEOUT_MS");
    config.modelRequestTimeoutMs = parsePositiveLong(
        environmentOr(
            "TREESEM_MODEL_TIMEOUT_MS",
            std::to_string(config.modelRequestTimeoutMs)),
        "TREESEM_MODEL_TIMEOUT_MS");
    config.inferenceWorkerCount = parsePositiveSize(
        "TREESEM_INFERENCE_WORKERS", config.inferenceWorkerCount);
    config.inferenceQueueCapacity = parsePositiveSize(
        "TREESEM_INFERENCE_QUEUE_CAPACITY", config.inferenceQueueCapacity);
    config.databaseWorkerCount = parsePositiveSize(
        "TREESEM_DATABASE_WORKERS", config.databaseWorkerCount);
    config.databaseQueueCapacity = parsePositiveSize(
        "TREESEM_DATABASE_QUEUE_CAPACITY", config.databaseQueueCapacity);
    config.modelBackend = parseBackend(environmentOr(
        "TREESEM_MODEL_BACKEND", toString(config.modelBackend)));
    config.servingBundleDirectory = environmentOr(
        "TREESEM_SERVING_BUNDLE_DIR", config.servingBundleDirectory);
    config.storageBackend = parseStorageBackend(environmentOr(
        "TREESEM_STORAGE_BACKEND", toString(config.storageBackend)));
    config.databaseHost = environmentOr("TREESEM_DB_HOST", config.databaseHost);
    config.databasePort = parsePort(environmentOr(
        "TREESEM_DB_PORT", std::to_string(config.databasePort)));
    config.databaseName = environmentOr("TREESEM_DB_NAME", config.databaseName);
    config.databaseUser = environmentOr("TREESEM_DB_USER", config.databaseUser);
    config.databasePassword = environmentOr(
        "TREESEM_DB_PASSWORD", config.databasePassword);
    config.databasePoolSize = parsePositiveSize(
        "TREESEM_DB_POOL_SIZE", config.databasePoolSize);
    config.databaseAcquireTimeoutMs = parsePositiveLong(environmentOr(
        "TREESEM_DB_ACQUIRE_TIMEOUT_MS",
        std::to_string(config.databaseAcquireTimeoutMs)),
        "TREESEM_DB_ACQUIRE_TIMEOUT_MS");
    config.databaseConnectTimeoutMs = parsePositiveLong(environmentOr(
        "TREESEM_DB_CONNECT_TIMEOUT_MS",
        std::to_string(config.databaseConnectTimeoutMs)),
        "TREESEM_DB_CONNECT_TIMEOUT_MS");
    config.databaseReadTimeoutMs = parsePositiveLong(environmentOr(
        "TREESEM_DB_READ_TIMEOUT_MS",
        std::to_string(config.databaseReadTimeoutMs)),
        "TREESEM_DB_READ_TIMEOUT_MS");
    config.databaseWriteTimeoutMs = parsePositiveLong(environmentOr(
        "TREESEM_DB_WRITE_TIMEOUT_MS",
        std::to_string(config.databaseWriteTimeoutMs)),
        "TREESEM_DB_WRITE_TIMEOUT_MS");
    config.sessionTtlSeconds = parsePositiveLong(environmentOr(
        "TREESEM_SESSION_TTL_SECONDS", std::to_string(config.sessionTtlSeconds)),
        "TREESEM_SESSION_TTL_SECONDS");
    config.cookieSecure = parseBoolean(environmentOr(
        "TREESEM_COOKIE_SECURE", config.cookieSecure ? "true" : "false"),
        "TREESEM_COOKIE_SECURE");
    if (config.databaseHost.empty() || config.databaseName.empty() ||
        config.databaseUser.empty())
    {
        throw std::invalid_argument("database host, name and user must not be empty");
    }
    if (config.storageBackend == StorageBackend::MySql &&
        config.databasePassword.empty())
    {
        throw std::invalid_argument(
            "TREESEM_DB_PASSWORD is required for mysql storage");
    }
    return config;
}

} // namespace config
} // namespace treesem
