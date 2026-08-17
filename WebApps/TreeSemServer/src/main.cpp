#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include <muduo/base/Logging.h>

#include "http/HttpRequest.h"
#include "http/HttpResponse.h"
#include "http/HttpServer.h"
#include "client/PythonModelClient.h"
#include "service/InferenceScheduler.h"
#include "service/PythonModelProxy.h"

namespace
{

int parsePort(int argc, char* argv[])
{
    if (argc < 2)
    {
        return 8080;
    }

    const int port = std::stoi(argv[1]);
    if (port <= 0 || port > 65535)
    {
        throw std::invalid_argument("port must be in the range 1..65535");
    }
    return port;
}

void healthHandler(const http::HttpRequest& request, http::HttpResponse* response)
{
    static const std::string body =
        R"({"status":"ok","service":"treeSem-backend"})";

    response->setStatusLine(
        request.getVersion(), http::HttpResponse::k200Ok, "OK");
    response->setContentType("application/json; charset=utf-8");
    response->setBody(body);
}

std::string environmentOr(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr || *value == '\0' ? fallback : value;
}

long positiveEnvironmentLong(const char* name, long fallback)
{
    const std::string value = environmentOr(name, std::to_string(fallback));
    const long parsed = std::stol(value);
    if (parsed <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        muduo::Logger::setLogLevel(muduo::Logger::INFO);

        const int port = parsePort(argc, argv);
        treesem::client::PythonModelClientConfig clientConfig;
        clientConfig.predictUrl = environmentOr(
            "TREESEM_MODEL_ADAPTER_URL",
            "http://127.0.0.1:18081/v1/predict");
        clientConfig.connectTimeoutMs =
            positiveEnvironmentLong("TREESEM_MODEL_CONNECT_TIMEOUT_MS", 500);
        clientConfig.requestTimeoutMs =
            positiveEnvironmentLong("TREESEM_MODEL_TIMEOUT_MS", 5000);

        treesem::client::PythonModelClient modelClient(clientConfig);
        treesem::service::PythonModelProxy modelProxy(modelClient);
        treesem::service::InferenceScheduler inferenceScheduler(
            static_cast<std::size_t>(
                positiveEnvironmentLong("TREESEM_INFERENCE_WORKERS", 2)),
            static_cast<std::size_t>(
                positiveEnvironmentLong("TREESEM_INFERENCE_QUEUE_CAPACITY", 32)));

        http::HttpServer server(port, "TreeSemServer");
        server.Get("/health", healthHandler);
        const http::AsyncHttpCallback predictionHandler =
            [&inferenceScheduler, &modelProxy](http::HttpRequest request,
                                               http::AsyncResponder responder) {
                std::string body = request.getBody();
                inferenceScheduler.schedule(
                    [body = std::move(body), &modelProxy]() {
                        return modelProxy.predict(body);
                    },
                    std::move(responder));
            };
        server.PostAsync("/api/v1/predictions", predictionHandler);
        server.PostAsync("/internal/v1/predictions", predictionHandler);
        server.setThreadNum(2);

        LOG_INFO << "treeSem backend listening on port " << port;
        server.start();
    }
    catch (const std::exception& error)
    {
        std::cerr << "failed to start treeSem backend: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
