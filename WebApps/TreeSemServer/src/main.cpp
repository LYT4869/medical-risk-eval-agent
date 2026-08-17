#include <cstdlib>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include <muduo/base/Logging.h>

#include "api/HealthController.h"
#include "api/BusinessController.h"
#include "api/PredictionController.h"
#include "application/PredictionService.h"
#include "application/RecordServices.h"
#include "application/SessionService.h"
#include "client/PythonModelClient.h"
#include "config/TreeSemServerConfig.h"
#include "http/HttpServer.h"
#include "infrastructure/model/RemoteTreeSemModelService.h"
#include "infrastructure/model/RoutingModelServices.h"
#if defined(TREESEM_HAS_ONNXRUNTIME)
#include "infrastructure/model/OnnxTreeSemModelService.h"
#endif
#include "infrastructure/process/ProcessSignalHandler.h"
#include "infrastructure/persistence/InMemoryTreeSemStore.h"
#if defined(TREESEM_HAS_MYSQL)
#include "infrastructure/persistence/MySqlTreeSemStore.h"
#endif
#include "persistence/ITreeSemStore.h"
#include "service/BlockingTaskScheduler.h"
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

        std::unique_ptr<treesem::persistence::ITreeSemStore> store;
        if (config.storageBackend == treesem::config::StorageBackend::Memory)
        {
            store = std::make_unique<
                treesem::infrastructure::InMemoryTreeSemStore>();
        }
        else
        {
#if defined(TREESEM_HAS_MYSQL)
            store = std::make_unique<treesem::infrastructure::MySqlTreeSemStore>(
                treesem::infrastructure::MySqlStoreConfig::from(config));
#else
            throw std::invalid_argument(
                "this build has no MySQL support; use TREESEM_STORAGE_BACKEND=memory");
#endif
        }
        treesem::application::SessionService sessionService(
            *store, std::chrono::seconds(config.sessionTtlSeconds));
        treesem::application::PredictionService predictionService(
            *selectedModelService, *store, sessionService);
        treesem::application::ExplanationService explanationService(*store);
        treesem::application::HistoryService historyService(*store);
        treesem::application::ComparisonService comparisonService(*store);
        treesem::application::FeedbackService feedbackService(*store);
        treesem::service::BlockingTaskScheduler predictionScheduler(
            config.inferenceWorkerCount,
            config.inferenceQueueCapacity,
            treesem::service::predictionSchedulerErrors());
        treesem::service::BlockingTaskScheduler databaseScheduler(
            config.databaseWorkerCount,
            config.databaseQueueCapacity,
            treesem::service::databaseSchedulerErrors());
        treesem::api::PredictionController predictionController(
            predictionService, predictionScheduler, config.cookieSecure,
            config.sessionTtlSeconds);
        treesem::api::BusinessController businessController(
            sessionService, explanationService, historyService,
            comparisonService, feedbackService, *store, databaseScheduler,
            config.cookieSecure, config.sessionTtlSeconds);
        treesem::api::HealthController healthController({
            modelVersion,
            treesem::config::toString(config.modelBackend),
            primaryBackend,
            fallbackEnabled,
            treesem::config::toString(config.storageBackend),
            config.storageBackend == treesem::config::StorageBackend::MySql
                ? config.databasePoolSize : 0,
            config.sessionTtlSeconds});

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
        server.GetAsync(
            "/ready",
            [&businessController](http::HttpRequest request,
                                  http::AsyncResponder responder) {
                businessController.ready(std::move(request), std::move(responder));
            });
        const http::AsyncHttpCallback getPrediction =
            [&businessController](http::HttpRequest request,
                                  http::AsyncResponder responder) {
                businessController.getPrediction(
                    std::move(request), std::move(responder));
            };
        const http::AsyncHttpCallback getExplanation =
            [&businessController](http::HttpRequest request,
                                  http::AsyncResponder responder) {
                businessController.getExplanation(
                    std::move(request), std::move(responder));
            };
        server.addAsyncRoute(
            http::HttpRequest::kGet,
            "/api/v1/predictions/:prediction_id", getPrediction);
        server.addAsyncRoute(
            http::HttpRequest::kGet,
            "/internal/v1/predictions/:prediction_id", getPrediction);
        server.addAsyncRoute(
            http::HttpRequest::kGet,
            "/api/v1/predictions/:prediction_id/explanation", getExplanation);
        server.addAsyncRoute(
            http::HttpRequest::kGet,
            "/internal/v1/explanations/:prediction_id", getExplanation);
        const http::AsyncHttpCallback historyHandler =
            [&businessController](http::HttpRequest request,
                                  http::AsyncResponder responder) {
                businessController.getHistory(
                    std::move(request), std::move(responder));
            };
        server.GetAsync("/api/v1/sessions/current/history", historyHandler);
        server.addAsyncRoute(
            http::HttpRequest::kGet,
            "/internal/v1/sessions/:session_id/history", historyHandler);
        const http::AsyncHttpCallback comparisonHandler =
            [&businessController](http::HttpRequest request,
                                  http::AsyncResponder responder) {
                businessController.compare(std::move(request), std::move(responder));
            };
        server.PostAsync("/api/v1/comparisons", comparisonHandler);
        server.PostAsync("/internal/v1/comparisons", comparisonHandler);
        server.addAsyncRoute(
            http::HttpRequest::kPost,
            "/internal/v1/predictions/:prediction_id/feedback",
            [&businessController](http::HttpRequest request,
                                  http::AsyncResponder responder) {
                businessController.submitFeedback(
                    std::move(request), std::move(responder));
            });
        server.addAsyncRoute(
            http::HttpRequest::kGet,
            "/internal/v1/predictions/:prediction_id/feedback",
            [&businessController](http::HttpRequest request,
                                  http::AsyncResponder responder) {
                businessController.listFeedback(
                    std::move(request), std::move(responder));
            });
        server.setThreadNum(config.ioThreadCount);
        LOG_INFO << "treeSem backend listening on port " << config.listenPort;
        {
            treesem::infrastructure::ProcessSignalHandler processSignals(
                server.getLoop(), [&server]() { server.stop(); });
            server.start();
        }
        predictionScheduler.shutdown();
        databaseScheduler.shutdown();
    }
    catch (const std::exception& error)
    {
        std::cerr << "failed to start treeSem backend: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
