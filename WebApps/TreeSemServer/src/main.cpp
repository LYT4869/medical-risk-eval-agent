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
#include "api/MetricsController.h"
#include "api/BusinessController.h"
#include "api/PredictionController.h"
#include "api/ChatController.h"
#include "api/AuthController.h"
#include "api/DoctorController.h"
#if defined(TREESEM_HAS_MYSQL) && defined(TREESEM_HAS_AUTH)
#include "api/V2SimController.h"
#include "infrastructure/persistence/V2SimArtifactStore.h"
#endif
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
#include "middleware/ObservabilityMiddleware.h"
#include "observability/MetricsRegistry.h"
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
        auto metrics = std::make_shared<http::observability::MetricsRegistry>();
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
                    *onnxModelService, remoteModelService, metrics);
                selectedModelService = fallbackModelService.get();
                fallbackEnabled = true;
            }
            else if (config.modelBackend == treesem::config::ModelBackend::Shadow)
            {
                shadowModelService = std::make_unique<
                    treesem::infrastructure::ShadowModelService>(
                    *onnxModelService, remoteModelService, metrics);
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

        treesem::infrastructure::InstrumentedModelService instrumentedModelService(
            *selectedModelService, metrics);
        selectedModelService = &instrumentedModelService;

        std::unique_ptr<treesem::persistence::ITreeSemStore> store;
        if (config.storageBackend == treesem::config::StorageBackend::Memory)
        {
            store = std::make_unique<
                treesem::infrastructure::InMemoryTreeSemStore>();
        }
        else
        {
#if defined(TREESEM_HAS_MYSQL)
            auto mysqlConfig = treesem::infrastructure::MySqlStoreConfig::from(config);
            mysqlConfig.connection.metrics = metrics;
            store = std::make_unique<treesem::infrastructure::MySqlTreeSemStore>(
                std::move(mysqlConfig));
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
            treesem::service::predictionSchedulerErrors(), metrics, "prediction");
        treesem::service::BlockingTaskScheduler databaseScheduler(
            config.databaseWorkerCount,
            config.databaseQueueCapacity,
            treesem::service::databaseSchedulerErrors(), metrics, "database");
#if defined(TREESEM_HAS_AUTH)
        treesem::service::BlockingTaskScheduler authScheduler(
            config.authWorkerCount, config.authQueueCapacity,
            treesem::service::authenticationSchedulerErrors(), metrics, "authentication");
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
                treesem::service::agentSchedulerErrors(), metrics, "agent");
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
#if defined(TREESEM_HAS_MYSQL) && defined(TREESEM_HAS_AUTH)
        std::unique_ptr<treesem::infrastructure::V2SimArtifactStore> v2SimStore;
        std::unique_ptr<treesem::api::V2SimController> v2SimController;
        const char* v2SimFlag = std::getenv("TREESEM_V2_SIM_ARTIFACT_API");
        if (v2SimFlag != nullptr && std::string(v2SimFlag) == "1")
        {
            if (!config.authRequired ||
                config.storageBackend != treesem::config::StorageBackend::MySql)
                throw std::invalid_argument(
                    "V2 SIM Artifact API requires enabled JWT auth and MySQL");
            if (config.capabilityTokenTtlSeconds > 120)
                throw std::invalid_argument(
                    "V2 SIM Artifact API requires capability TTL at most 120s");
            auto v2Config = treesem::infrastructure::MySqlStoreConfig::from(config);
            v2Config.connection.poolSize = 2;
            v2SimStore = std::make_unique<
                treesem::infrastructure::V2SimArtifactStore>(
                std::move(v2Config.connection));
            v2SimController = std::make_unique<treesem::api::V2SimController>(
                authService, *securityStore, auditService, jwtService,
                *v2SimStore, databaseScheduler);
        }
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
        treesem::api::MetricsController metricsController(
            metrics, config.metricsBearerToken);

        http::HttpServer server(config.listenPort, "TreeSemServer");
        if (config.observabilityEnabled)
            server.addMiddleware(std::make_shared<http::middleware::ObservabilityMiddleware>(
                metrics, "treesem-backend", config.traceSampleRate,
                config.slowRequestMs));
#if defined(TREESEM_HAS_AUTH)
        server.addMiddleware(std::make_shared<treesem::security::SecurityMiddleware>(
            jwtService, config.authRequired, config.allowedOrigins, metrics));
#endif
        server.Get(
            "/health",
            [&healthController](const http::HttpRequest& request,
                                http::HttpResponse* response) {
                healthController.handle(request, response);
            });
        if (config.metricsEnabled)
            server.Get("/internal/metrics",
                [&metricsController](const http::HttpRequest& request,
                                     http::HttpResponse* response) {
                    metricsController.handle(request, response);
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

#if defined(TREESEM_HAS_MYSQL) && defined(TREESEM_HAS_AUTH)
        // Explicitly opt-in *synthetic* artifact HTTP routes. No real EHR use.
        if (v2SimController)
        {
            // Public doctor routes require a verified access JWT and active
            // patient assignment on every call. They never accept artifacts.
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/api/v2/doctor/patients/:patient_id/sim-runs",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->create(std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kGet,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->status(std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/confirm",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->confirm(std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/contract",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->editContract(
                        std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/capability",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->capability(std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kGet,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/artifacts/:kind/:artifact_key",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->get(std::move(request), std::move(responder));
                });

            // Fixed P001 scenario replay only. A doctor Access JWT is required;
            // Agent capability tokens cannot manufacture LIS corrections.
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m4/history",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->stageHistoricalLab(
                        std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m4/correction",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->applyOfficialSimCorrection(
                        std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kGet,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m4/state",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->correctionState(
                        std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m4/validate-current",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->checkCurrentSources(
                        std::move(request), std::move(responder));
                });

            // Initial M5 report publication: short-lived internal Agent Tool
            // capability only. Doctor revisions use separate authenticated
            // report-specific endpoints with mandatory base revision.
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/internal/v2/sim-runs/:run_id/artifacts/m5/publish",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->m5Publish(
                        std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kGet,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m5/reports/:report_id/revisions/:revision",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->m5Revision(
                        std::move(request), std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m5/reports/:report_id/messages",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->m5Message(
                        std::move(request), std::move(responder));
                });

            // Only a short-lived, signed, run-scoped Tool capability can use
            // internal artifact writes or Agent status/read requests.
            server.addAsyncRoute(http::HttpRequest::kGet,
                "/internal/v2/sim-runs/:run_id",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->internalStatus(std::move(request),
                                                   std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kPost,
                "/internal/v2/sim-runs/:run_id/artifacts",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->internalPut(std::move(request),
                                                std::move(responder));
                });
            server.addAsyncRoute(http::HttpRequest::kGet,
                "/internal/v2/sim-runs/:run_id/artifacts/:kind/:artifact_key",
                [&v2SimController](http::HttpRequest request,
                                   http::AsyncResponder responder) {
                    v2SimController->internalGet(std::move(request),
                                                std::move(responder));
                });
        }
#endif
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
