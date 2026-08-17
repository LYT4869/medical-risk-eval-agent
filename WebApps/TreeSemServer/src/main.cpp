#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include <muduo/base/Logging.h>

#include "api/HealthController.h"
#include "api/PredictionController.h"
#include "application/PredictionService.h"
#include "client/PythonModelClient.h"
#include "config/TreeSemServerConfig.h"
#include "http/HttpServer.h"
#include "infrastructure/model/RemoteTreeSemModelService.h"
#include "infrastructure/model/RoutingModelServices.h"
#if defined(TREESEM_HAS_ONNXRUNTIME)
#include "infrastructure/model/OnnxTreeSemModelService.h"
#endif
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
        treesem::infrastructure::RemoteTreeSemModelService remoteModelService(
            modelClient);
        const treesem::model::IModelService* selectedModelService =
            &remoteModelService;
        std::optional<std::string> modelVersion;
        std::string primaryBackend = "remote";
        bool fallbackEnabled = false;
        std::unique_ptr<treesem::infrastructure::FallbackModelService>
            fallbackModelService;
        std::unique_ptr<treesem::infrastructure::ShadowModelService>
            shadowModelService;
#if defined(TREESEM_HAS_ONNXRUNTIME)
        std::unique_ptr<treesem::infrastructure::OnnxTreeSemModelService>
            onnxModelService;
        if (config.modelBackend != treesem::config::ModelBackend::Remote)
        {
            if (config.servingBundleDirectory.empty())
            {
                throw std::invalid_argument(
                    "TREESEM_SERVING_BUNDLE_DIR is required for ONNX backends");
            }
            onnxModelService = std::make_unique<
                treesem::infrastructure::OnnxTreeSemModelService>(
                config.servingBundleDirectory);
            selectedModelService = onnxModelService.get();
            modelVersion = onnxModelService->bundle().modelVersion();
            primaryBackend = "onnx";
            if (config.modelBackend == treesem::config::ModelBackend::OnnxFallback)
            {
                fallbackModelService = std::make_unique<
                    treesem::infrastructure::FallbackModelService>(
                    *onnxModelService, remoteModelService);
                selectedModelService = fallbackModelService.get();
                fallbackEnabled = true;
            }
            else if (config.modelBackend == treesem::config::ModelBackend::Shadow)
            {
                shadowModelService = std::make_unique<
                    treesem::infrastructure::ShadowModelService>(
                    *onnxModelService, remoteModelService);
                selectedModelService = shadowModelService.get();
            }
        }
#else
        if (config.modelBackend != treesem::config::ModelBackend::Remote)
        {
            throw std::invalid_argument(
                "this build has no ONNX Runtime; use TREESEM_MODEL_BACKEND=remote");
        }
#endif

        treesem::application::PredictionService predictionService(
            *selectedModelService);
        treesem::service::InferenceScheduler inferenceScheduler(
            config.inferenceWorkerCount,
            config.inferenceQueueCapacity);
        treesem::api::PredictionController predictionController(
            predictionService,
            inferenceScheduler);
        treesem::api::HealthController healthController({
            modelVersion,
            treesem::config::toString(config.modelBackend),
            primaryBackend,
            fallbackEnabled});

        http::HttpServer server(config.listenPort, "TreeSemServer");
        server.Get(
            "/health",
            [&healthController](const http::HttpRequest& request,
                                http::HttpResponse* response) {
                healthController.handle(request, response);
            });
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
