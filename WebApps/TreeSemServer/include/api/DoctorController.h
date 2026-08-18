#pragma once

#include "application/AgentApplicationService.h"
#include "application/AuditService.h"
#include "application/AuthService.h"
#include "application/PredictionService.h"
#include "application/RecordServices.h"
#include "application/SessionService.h"
#include "persistence/ISecurityStore.h"
#include "persistence/ITreeSemStore.h"
#include "service/BlockingTaskScheduler.h"

namespace treesem::api
{
class DoctorController
{
public:
    DoctorController(const application::AuthService& auth,
                     const application::SessionService& sessions,
                     const application::PredictionService& predictions,
                     const application::AgentApplicationService* agent,
                     const application::FeedbackService& feedback,
                     persistence::ITreeSemStore& store,
                     persistence::ISecurityStore& securityStore,
                     service::BlockingTaskScheduler& predictionScheduler,
                     service::BlockingTaskScheduler* agentScheduler,
                     service::BlockingTaskScheduler& databaseScheduler,
                     const application::AuditService& audit,
                     bool cookieSecure, long sessionTtlSeconds,
                     std::size_t maxMessageCharacters);
    void predict(http::HttpRequest request, http::AsyncResponder responder) const;
    void chat(http::HttpRequest request, http::AsyncResponder responder) const;
    void history(http::HttpRequest request, http::AsyncResponder responder) const;
    void feedback(http::HttpRequest request, http::AsyncResponder responder) const;
private:
    domain::ActorContext doctor(const http::HttpRequest& request,
                                const std::string& patientId) const;
    application::ResolvedSession contextSession(
        const http::HttpRequest& request,
        const domain::ActorContext& actor,
        const std::string& patientId) const;
    const application::AuthService& auth_;
    const application::SessionService& sessions_;
    const application::PredictionService& predictions_;
    const application::AgentApplicationService* agent_;
    const application::FeedbackService& feedback_;
    persistence::ITreeSemStore& store_;
    persistence::ISecurityStore& securityStore_;
    service::BlockingTaskScheduler& predictionScheduler_;
    service::BlockingTaskScheduler* agentScheduler_;
    service::BlockingTaskScheduler& databaseScheduler_;
    const application::AuditService& audit_;
    bool cookieSecure_;
    long sessionTtlSeconds_;
    std::size_t maxMessageCharacters_;
};
} // namespace treesem::api
