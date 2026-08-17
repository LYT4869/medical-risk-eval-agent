#include <cstdlib>
#include <exception>
#include <iostream>
#include <utility>

#include <muduo/base/Logging.h>

#include "api/HealthController.h"
#include "api/PredictionController.h"
#include "application/PredictionService.h"
#include "client/PythonModelClient.h"
#include "config/TreeSemServerConfig.h"
#include "http/HttpServer.h"
#include "infrastructure/model/RemoteTreeSemModelService.h"
#include "infrastructure/process/ProcessSignalHandler.h"
#include "service/InferenceScheduler.h"

int main(int argc, char* argv[])
{
    try
    {
        muduo::Logger::setLogLevel(muduo::Logger::INFO);

        const treesem::config::TreeSemServerConfig config =
            treesem::config::TreeSemServerConfig::load(argc, argv);
        treesem::client::PythonModelClientConfig clientConfig;
        clientConfig.predictUrl = config.modelAdapterUrl;
        clientConfig.connectTimeoutMs = config.modelConnectTimeoutMs;
        clientConfig.requestTimeoutMs = config.modelRequestTimeoutMs;

        treesem::client::PythonModelClient modelClient(clientConfig);
        treesem::infrastructure::RemoteTreeSemModelService modelService(modelClient);
        treesem::application::PredictionService predictionService(modelService);
        treesem::service::InferenceScheduler inferenceScheduler(
            config.inferenceWorkerCount,
            config.inferenceQueueCapacity);
        treesem::api::PredictionController predictionController(
            predictionService,
            inferenceScheduler);

        http::HttpServer server(config.listenPort, "TreeSemServer");
        server.Get("/health", treesem::api::healthHandler);
        const http::AsyncHttpCallback predictionHandler =
            [&predictionController](http::HttpRequest request,
                                    http::AsyncResponder responder) {
                predictionController.handle(std::move(request), std::move(responder));
            };
        server.PostAsync("/api/v1/predictions", predictionHandler);
        server.PostAsync("/internal/v1/predictions", predictionHandler);
        server.setThreadNum(config.ioThreadCount);
        LOG_INFO << "treeSem backend listening on port " << config.listenPort;
        {
            treesem::infrastructure::ProcessSignalHandler processSignals(
                server.getLoop(), [&server]() { server.stop(); });
            server.start();
        }
        inferenceScheduler.shutdown();
    }
    catch (const std::exception& error)
    {
        std::cerr << "failed to start treeSem backend: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
