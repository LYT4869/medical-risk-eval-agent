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
    config.modelBackend = parseBackend(environmentOr(
        "TREESEM_MODEL_BACKEND", toString(config.modelBackend)));
    config.servingBundleDirectory = environmentOr(
        "TREESEM_SERVING_BUNDLE_DIR", config.servingBundleDirectory);
    return config;
}

} // namespace config
} // namespace treesem
