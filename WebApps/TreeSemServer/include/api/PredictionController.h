#pragma once

#include "application/PredictionService.h"
#include "http/AsyncHttp.h"
#include "http/HttpRequest.h"
#include "service/InferenceScheduler.h"
#include "service/BlockingTaskScheduler.h"
#include "persistence/ISecurityStore.h"
#include "application/AuditService.h"

namespace treesem
{
namespace api
{

class PredictionController
{
public:
    PredictionController(
        const application::PredictionService& predictionService,
        service::InferenceScheduler& inferenceScheduler);
    PredictionController(
        const application::PredictionService& predictionService,
        service::BlockingTaskScheduler& predictionScheduler,
        bool cookieSecure,
        long sessionTtlSeconds,
        persistence::ISecurityStore* securityStore = nullptr,
        bool authRequired = false,
        const application::AuditService* audit = nullptr);

    void handle(http::HttpRequest request, http::AsyncResponder responder) const;

private:
    const application::PredictionService& predictionService_;
    service::BlockingTaskScheduler& predictionScheduler_;
    bool persistent_{false};
    bool cookieSecure_{false};
    long sessionTtlSeconds_{3600};
    persistence::ISecurityStore* securityStore_{nullptr};
    bool authRequired_{false};
    const application::AuditService* audit_{nullptr};
};

} // namespace api
} // namespace treesem
