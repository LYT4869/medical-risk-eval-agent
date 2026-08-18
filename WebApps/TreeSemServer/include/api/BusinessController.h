#pragma once

#include <optional>

#include "api/BusinessJsonCodec.h"
#include "application/RecordServices.h"
#include "application/SessionService.h"
#include "http/AsyncHttp.h"
#include "http/HttpRequest.h"
#include "persistence/ITreeSemStore.h"
#include "service/BlockingTaskScheduler.h"
#include "persistence/ISecurityStore.h"
#include "application/AuditService.h"

namespace treesem
{
namespace api
{

class BusinessController
{
public:
    BusinessController(
        const application::SessionService& sessions,
        const application::ExplanationService& explanations,
        const application::HistoryService& history,
        const application::ComparisonService& comparisons,
        const application::FeedbackService& feedback,
        persistence::ITreeSemStore& store,
        service::BlockingTaskScheduler& databaseScheduler,
        bool cookieSecure,
        long sessionTtlSeconds,
        persistence::ISecurityStore* securityStore = nullptr,
        bool authRequired = false,
        const application::AuditService* audit = nullptr);

    void getPrediction(http::HttpRequest request, http::AsyncResponder responder) const;
    void getExplanation(http::HttpRequest request, http::AsyncResponder responder) const;
    void getHistory(http::HttpRequest request, http::AsyncResponder responder) const;
    void compare(http::HttpRequest request, http::AsyncResponder responder) const;
    void submitFeedback(http::HttpRequest request, http::AsyncResponder responder) const;
    void listFeedback(http::HttpRequest request, http::AsyncResponder responder) const;
    void ready(http::HttpRequest request, http::AsyncResponder responder) const;

private:
    application::ResolvedSession resolveRequestSession(
        const http::HttpRequest& request,
        bool allowPublicReplacement) const;
    void getPredictionImpl(http::HttpRequest request,
                           http::AsyncResponder responder,
                           bool explanationOnly) const;
    http::ResponseWriter decoratePublicSession(
        http::ResponseWriter writer,
        const http::HttpRequest& request,
        const std::optional<application::ResolvedSession>& session) const;

    const application::SessionService& sessions_;
    const application::ExplanationService& explanations_;
    const application::HistoryService& history_;
    const application::ComparisonService& comparisons_;
    const application::FeedbackService& feedback_;
    persistence::ITreeSemStore& store_;
    service::BlockingTaskScheduler& databaseScheduler_;
    bool cookieSecure_;
    long sessionTtlSeconds_;
    persistence::ISecurityStore* securityStore_;
    bool authRequired_;
    const application::AuditService* audit_;
};

} // namespace api
} // namespace treesem
