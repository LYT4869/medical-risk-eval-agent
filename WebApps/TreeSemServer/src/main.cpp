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
#include "api/ChatController.h"
#include "api/AuthController.h"
#include "api/DoctorController.h"
#include "application/PredictionService.h"
#include "application/RecordServices.h"
#include "application/SessionService.h"
#include "application/AgentApplicationService.h"
#include "application/AuthService.h"
#include "application/AuditService.h"
#include "client/PythonModelClient.h"
#include "client/PythonAgentClient.h"
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
#if defined(TREESEM_HAS_AUTH)
#include "security/JwtService.h"
#include "security/PasswordHasher.h"
#include "security/SecurityMiddleware.h"
#endif

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
        auto* securityStore = dynamic_cast<treesem::persistence::ISecurityStore*>(
            store.get());
        if (securityStore == nullptr)
            throw std::logic_error("configured store has no security support");
        const treesem::security::JwtService* configuredJwtService = nullptr;
        const treesem::application::AuditService* auditServicePtr = nullptr;
#if defined(TREESEM_HAS_AUTH)
        const std::string developmentAccessSecret =
            "treesem-development-access-secret-change-me";
        const std::string developmentCapabilitySecret =
            "treesem-development-capability-secret-change-me";
        const std::string developmentKnowledgeSecret =
            "treesem-development-knowledge-secret-change-me";
        treesem::security::JwtConfig jwtConfig;
        jwtConfig.accessSecret = config.authRequired
            ? config.accessJwtSecret : developmentAccessSecret;
        jwtConfig.capabilitySecret = config.authRequired
            ? config.capabilityJwtSecret : developmentCapabilitySecret;
        jwtConfig.accessTtl = std::chrono::seconds(config.accessTokenTtlSeconds);
        jwtConfig.capabilityTtl =
            std::chrono::seconds(config.capabilityTokenTtlSeconds);
        jwtConfig.knowledgeSecret = config.knowledgeJwtSecret.empty()
            ? developmentKnowledgeSecret : config.knowledgeJwtSecret;
        jwtConfig.knowledgeTtl =
            std::chrono::seconds(config.knowledgeTokenTtlSeconds);
        treesem::security::JwtService jwtService(std::move(jwtConfig));
        treesem::security::PasswordHasher passwordHasher;
        treesem::application::AuthService authService(
            *securityStore, passwordHasher, jwtService,
            std::chrono::seconds(config.accessTokenTtlSeconds),
            std::chrono::seconds(config.refreshTokenTtlSeconds));
        treesem::application::AuditService auditService(*securityStore);
        configuredJwtService = (config.authRequired || config.knowledgeEnabled)
            ? &jwtService : nullptr;
        auditServicePtr = config.authRequired ? &auditService : nullptr;
#else
        if (config.authRequired)
            throw std::invalid_argument("this build has no authentication support");
#endif
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
#if defined(TREESEM_HAS_AUTH)
        treesem::service::BlockingTaskScheduler authScheduler(
            config.authWorkerCount, config.authQueueCapacity,
            treesem::service::authenticationSchedulerErrors());
        treesem::api::AuthController authController(
            authService, sessionService, *securityStore, authScheduler,
            config.refreshCookieSecure,
            config.refreshTokenTtlSeconds, config.allowedOrigins);
#endif
        std::unique_ptr<treesem::client::PythonAgentClient> agentClient;
        std::unique_ptr<treesem::application::AgentApplicationService> agentService;
        std::unique_ptr<treesem::service::BlockingTaskScheduler> agentScheduler;
        std::unique_ptr<treesem::api::ChatController> chatController;
        if (config.agentEnabled)
        {
            treesem::client::PythonAgentClientConfig agentClientConfig;
            agentClientConfig.url = config.agentUrl;
            agentClientConfig.serviceSecret = config.agentServiceSecret;
            agentClientConfig.connectTimeoutMs = config.agentConnectTimeoutMs;
            agentClientConfig.requestTimeoutMs = config.agentRequestTimeoutMs;
            agentClientConfig.maxResponseBytes = config.agentResponseMaxBytes;
            agentClient = std::make_unique<treesem::client::PythonAgentClient>(
                std::move(agentClientConfig));
            agentService = std::make_unique<treesem::application::AgentApplicationService>(
                *agentClient, *store, sessionService, config.agentContextMessages,
                configuredJwtService,
                config.knowledgeEnabled && configuredJwtService != nullptr);
            agentScheduler = std::make_unique<treesem::service::BlockingTaskScheduler>(
                config.agentWorkerCount, config.agentQueueCapacity,
                treesem::service::agentSchedulerErrors());
            chatController = std::make_unique<treesem::api::ChatController>(
                *agentService, sessionService, *agentScheduler, databaseScheduler,
                config.cookieSecure, config.sessionTtlSeconds,
                config.agentMessageMaxCharacters, securityStore,
                configuredJwtService,
                config.authRequired, auditServicePtr);
        }
#if defined(TREESEM_HAS_AUTH)
        treesem::api::DoctorController doctorController(
            authService, sessionService, predictionService, agentService.get(),
            feedbackService, *store, *securityStore, predictionScheduler,
            agentScheduler.get(), databaseScheduler, auditService,
            config.cookieSecure,
            config.sessionTtlSeconds, config.agentMessageMaxCharacters);
#endif
        treesem::api::PredictionController predictionController(
            predictionService, predictionScheduler, config.cookieSecure,
            config.sessionTtlSeconds, securityStore, config.authRequired,
            auditServicePtr);
        treesem::api::BusinessController businessController(
            sessionService, explanationService, historyService,
            comparisonService, feedbackService, *store, databaseScheduler,
            config.cookieSecure, config.sessionTtlSeconds,
            securityStore, config.authRequired, auditServicePtr);

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
#if defined(TREESEM_HAS_AUTH)
        server.addMiddleware(std::make_shared<treesem::security::SecurityMiddleware>(
            jwtService, config.authRequired, config.allowedOrigins));
#endif
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
#if defined(TREESEM_HAS_AUTH)
        server.PostAsync("/api/v1/auth/register",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.registerPatient(std::move(request), std::move(responder));
            });
        server.PostAsync("/api/v1/auth/login",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.login(std::move(request), std::move(responder));
            });
        server.PostAsync("/api/v1/auth/refresh",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.refresh(std::move(request), std::move(responder));
            });
        server.PostAsync("/api/v1/auth/logout",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.logout(std::move(request), std::move(responder));
            });
        server.GetAsync("/api/v1/auth/me",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.me(std::move(request), std::move(responder));
            });
        server.PostAsync("/api/v1/admin/doctors",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.createDoctor(std::move(request), std::move(responder));
            });
        server.PostAsync("/api/v1/admin/doctor-patient-assignments",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.createAssignment(std::move(request), std::move(responder));
            });
        server.GetAsync("/api/v1/admin/doctor-patient-assignments",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.listAssignments(std::move(request), std::move(responder));
            });
        server.GetAsync("/api/v1/admin/audit-events",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.listAudit(std::move(request), std::move(responder));
            });
        server.addAsyncRoute(http::HttpRequest::kDelete,
            "/api/v1/admin/doctor-patient-assignments/:assignment_id",
            [&authController](http::HttpRequest request, http::AsyncResponder responder) {
                authController.revokeAssignment(std::move(request), std::move(responder));
            });
        server.addAsyncRoute(http::HttpRequest::kPost,
            "/api/v1/doctor/patients/:patient_id/predictions",
            [&doctorController](http::HttpRequest request, http::AsyncResponder responder) {
                doctorController.predict(std::move(request), std::move(responder));
            });
        server.addAsyncRoute(http::HttpRequest::kPost,
            "/api/v1/doctor/patients/:patient_id/chat",
            [&doctorController](http::HttpRequest request, http::AsyncResponder responder) {
                doctorController.chat(std::move(request), std::move(responder));
            });
        server.addAsyncRoute(http::HttpRequest::kGet,
            "/api/v1/doctor/patients/:patient_id/history",
            [&doctorController](http::HttpRequest request, http::AsyncResponder responder) {
                doctorController.history(std::move(request), std::move(responder));
            });
        server.addAsyncRoute(http::HttpRequest::kPost,
            "/api/v1/predictions/:prediction_id/feedback",
            [&doctorController](http::HttpRequest request, http::AsyncResponder responder) {
                doctorController.feedback(std::move(request), std::move(responder));
            });
#endif
        if (chatController)
        {
            server.PostAsync("/api/v1/chat",
                [&chatController](http::HttpRequest request,
                                  http::AsyncResponder responder) {
                    chatController->chat(std::move(request), std::move(responder));
                });
            server.GetAsync("/api/v1/chat/history",
                [&chatController](http::HttpRequest request,
                                  http::AsyncResponder responder) {
                    chatController->history(std::move(request), std::move(responder));
                });
        }
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
        if (agentScheduler) agentScheduler->shutdown();
#if defined(TREESEM_HAS_AUTH)
        authScheduler.shutdown();
#endif
    }
    catch (const std::exception& error)
    {
        std::cerr << "failed to start treeSem backend: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
