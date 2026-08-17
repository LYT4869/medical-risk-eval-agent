#include "api/BusinessController.h"

#include <utility>

#include "api/ApiException.h"
#include "api/HttpErrorMapper.h"
#include "application/BusinessException.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem
{
namespace api
{
namespace
{

bool internalRequest(const http::HttpRequest& request)
{
    return request.path().rfind("/internal/", 0) == 0;
}

template<typename Work>
http::ResponseWriter safeBusinessWork(Work&& work)
{
    try { return work(); }
    catch (const ApiException& error) { return HttpErrorMapper::from(error); }
    catch (const application::BusinessException& error)
    {
        return HttpErrorMapper::from(error);
    }
    catch (...) { return HttpErrorMapper::internalError(); }
}

} // namespace

BusinessController::BusinessController(
    const application::SessionService& sessions,
    const application::ExplanationService& explanations,
    const application::HistoryService& history,
    const application::ComparisonService& comparisons,
    const application::FeedbackService& feedback,
    persistence::ITreeSemStore& store,
    service::BlockingTaskScheduler& databaseScheduler,
    bool cookieSecure,
    long sessionTtlSeconds)
    : sessions_(sessions)
    , explanations_(explanations)
    , history_(history)
    , comparisons_(comparisons)
    , feedback_(feedback)
    , store_(store)
    , databaseScheduler_(databaseScheduler)
    , cookieSecure_(cookieSecure)
    , sessionTtlSeconds_(sessionTtlSeconds)
{}

http::ResponseWriter BusinessController::decoratePublicSession(
    http::ResponseWriter writer,
    const http::HttpRequest& request,
    const std::optional<application::ResolvedSession>& session) const
{
    if (internalRequest(request) || !session.has_value() || !session->created)
    {
        return writer;
    }
    const std::string cookie = BusinessJsonCodec::sessionCookie(
        session->session.sessionId, sessionTtlSeconds_, cookieSecure_);
    return [writer = std::move(writer), cookie](http::HttpResponse* response) {
        writer(response);
        response->addHeader("Set-Cookie", cookie);
    };
}

application::ResolvedSession BusinessController::resolveRequestSession(
    const http::HttpRequest& request,
    bool allowPublicReplacement) const
{
    const bool internal = internalRequest(request);
    const auto access = internal || !allowPublicReplacement
        ? application::SessionAccess::Internal
        : application::SessionAccess::Public;
    const auto supplied = BusinessJsonCodec::sessionId(
        request, internal ? application::SessionAccess::Internal
                          : application::SessionAccess::Public);
    if (!supplied.has_value())
    {
        throw application::BusinessException(
            application::BusinessException::Kind::InvalidSession,
            "session is required for this request");
    }
    return sessions_.resolve(supplied, access);
}

void BusinessController::getPredictionImpl(
    http::HttpRequest request,
    http::AsyncResponder responder,
    bool explanationOnly) const
{
    databaseScheduler_.schedule(
        [request = std::move(request), explanationOnly, this]() {
            std::optional<application::ResolvedSession> session;
            http::ResponseWriter writer = safeBusinessWork([&]() {
                session = resolveRequestSession(request, true);
                const std::string predictionId =
                    request.getPathParameters("prediction_id");
                const auto record = explanations_.get(
                    session->session.sessionId, predictionId);
                return HttpErrorMapper::success(explanationOnly
                    ? BusinessJsonCodec::serializeExplanation(record)
                    : BusinessJsonCodec::serializePrediction(record));
            });
            return decoratePublicSession(std::move(writer), request, session);
        },
        std::move(responder));
}

void BusinessController::getPrediction(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    getPredictionImpl(std::move(request), std::move(responder), false);
}

void BusinessController::getExplanation(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    getPredictionImpl(std::move(request), std::move(responder), true);
}

void BusinessController::getHistory(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    databaseScheduler_.schedule(
        [request = std::move(request), this]() {
            std::optional<application::ResolvedSession> resolved;
            http::ResponseWriter writer = safeBusinessWork([&]() {
                std::string sessionId;
                if (internalRequest(request))
                {
                    sessionId = request.getPathParameters("session_id");
                    const auto headerSession = BusinessJsonCodec::sessionId(
                        request, application::SessionAccess::Internal);
                    if (!headerSession.has_value())
                    {
                        throw application::BusinessException(
                            application::BusinessException::Kind::InvalidSession,
                            "internal requests require a session id");
                    }
                    if (*headerSession != sessionId)
                    {
                        throw application::BusinessException(
                            application::BusinessException::Kind::NotFound,
                            "session was not found");
                    }
                    (void)sessions_.resolve(
                        headerSession, application::SessionAccess::Internal);
                }
                else
                {
                    resolved = resolveRequestSession(request, true);
                    sessionId = resolved->session.sessionId;
                }
                const auto page = history_.list(
                    sessionId,
                    BusinessJsonCodec::parseHistoryLimit(request),
                    BusinessJsonCodec::parseHistoryCursor(request));
                return HttpErrorMapper::success(
                    BusinessJsonCodec::serializeHistory(page));
            });
            return decoratePublicSession(std::move(writer), request, resolved);
        },
        std::move(responder));
}

void BusinessController::compare(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    std::pair<std::string, std::string> ids;
    try { ids = BusinessJsonCodec::parseComparison(request.getBody()); }
    catch (const ApiException& error)
    {
        responder(HttpErrorMapper::from(error));
        return;
    }
    databaseScheduler_.schedule(
        [request = std::move(request), ids = std::move(ids), this]() {
            std::optional<application::ResolvedSession> session;
            http::ResponseWriter writer = safeBusinessWork([&]() {
                session = resolveRequestSession(request, true);
                return HttpErrorMapper::success(
                    BusinessJsonCodec::serializeComparison(comparisons_.compare(
                        session->session.sessionId, ids.first, ids.second)));
            });
            return decoratePublicSession(std::move(writer), request, session);
        },
        std::move(responder));
}

void BusinessController::submitFeedback(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    application::FeedbackInput input;
    try
    {
        input = BusinessJsonCodec::parseFeedback(
            request.getBody(), request.getHeader("Idempotency-Key"));
    }
    catch (const ApiException& error)
    {
        responder(HttpErrorMapper::from(error));
        return;
    }
    databaseScheduler_.schedule(
        [request = std::move(request), input = std::move(input), this]() {
            return safeBusinessWork([&]() {
                const auto session = resolveRequestSession(request, false);
                const auto result = feedback_.submit(
                    session.session.sessionId,
                    request.getPathParameters("prediction_id"), input);
                return HttpErrorMapper::json(
                    result.created ? http::HttpResponse::k201Created
                                   : http::HttpResponse::k200Ok,
                    result.created ? "Created" : "OK",
                    BusinessJsonCodec::serializeFeedback(result.feedback));
            });
        },
        std::move(responder));
}

void BusinessController::listFeedback(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    databaseScheduler_.schedule(
        [request = std::move(request), this]() {
            return safeBusinessWork([&]() {
                const auto session = resolveRequestSession(request, false);
                return HttpErrorMapper::success(
                    BusinessJsonCodec::serializeFeedbackList(feedback_.list(
                        session.session.sessionId,
                        request.getPathParameters("prediction_id"))));
            });
        },
        std::move(responder));
}

void BusinessController::ready(
    http::HttpRequest, http::AsyncResponder responder) const
{
    databaseScheduler_.schedule(
        [this]() {
            return safeBusinessWork([&]() {
                if (!store_.ping())
                {
                    throw application::BusinessException(
                        application::BusinessException::Kind::DatabaseUnavailable,
                        "database ping failed");
                }
                return HttpErrorMapper::success(R"({"status":"ready"})");
            });
        },
        std::move(responder));
}

} // namespace api
} // namespace treesem
