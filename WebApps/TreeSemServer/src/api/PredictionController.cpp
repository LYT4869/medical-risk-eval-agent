#include "api/PredictionController.h"

#include <utility>

#include "api/ApiException.h"
#include "api/BusinessJsonCodec.h"
#include "api/HttpErrorMapper.h"
#include "api/PredictionJsonCodec.h"
#include "model/ModelException.h"

namespace treesem
{
namespace api
{

PredictionController::PredictionController(
    const application::PredictionService& predictionService,
    service::InferenceScheduler& inferenceScheduler)
    : predictionService_(predictionService)
    , predictionScheduler_(inferenceScheduler)
{}

PredictionController::PredictionController(
    const application::PredictionService& predictionService,
    service::BlockingTaskScheduler& predictionScheduler,
    bool cookieSecure,
    long sessionTtlSeconds,
    persistence::ISecurityStore* securityStore,
    bool authRequired,
    const application::AuditService* audit)
    : predictionService_(predictionService)
    , predictionScheduler_(predictionScheduler)
    , persistent_(true)
    , cookieSecure_(cookieSecure)
    , sessionTtlSeconds_(sessionTtlSeconds)
    , securityStore_(securityStore)
    , authRequired_(authRequired)
    , audit_(audit)
{}

void PredictionController::handle(
    http::HttpRequest request,
    http::AsyncResponder responder) const
{
    model::ModelInput input;
    try
    {
        input = PredictionJsonCodec::parseInput(request.getBody());
    }
    catch (const ApiException& exception)
    {
        responder(HttpErrorMapper::from(exception));
        return;
    }
    catch (...)
    {
        responder(HttpErrorMapper::internalError());
        return;
    }

    const bool internal = request.path().rfind("/internal/", 0) == 0;
    std::optional<std::string> suppliedSession;
    if (persistent_)
    {
        suppliedSession = BusinessJsonCodec::sessionId(
            request, internal ? application::SessionAccess::Internal
                              : application::SessionAccess::Public);
    }
    predictionScheduler_.schedule(
        [input = std::move(input), suppliedSession, internal,
         actorId = request.getHeader("X-TreeSem-Actor-Id"),
         actorRole = request.getHeader("X-TreeSem-Actor-Role"),
         runId = request.getHeader("X-TreeSem-Agent-Run-Id"), this]()
            -> http::ResponseWriter {
            try
            {
                if (persistent_)
                {
                    std::optional<std::string> subject;
                    if (authRequired_)
                    {
                        const auto activeUser = securityStore_ == nullptr
                            ? std::nullopt : securityStore_->findUserById(actorId);
                        if (!activeUser.has_value() || activeUser->status != "active" ||
                            domain::toString(activeUser->role) != actorRole)
                            throw application::AuthException(
                                application::AuthException::Kind::InvalidToken,
                                "account is no longer active");
                        if (actorRole == "admin")
                        {
                            if (audit_ != nullptr)
                                audit_->record(actorId, actorRole, "",
                                    "prediction.create", "session", suppliedSession,
                                    "denied", "admin_has_no_clinical_access");
                            throw application::AuthException(
                                application::AuthException::Kind::Forbidden,
                                "administrators cannot access clinical predictions");
                        }
                        if (!suppliedSession.has_value() || actorId.empty() ||
                            securityStore_ == nullptr ||
                            !(subject = securityStore_->sessionSubjectForActor(
                                *suppliedSession, actorId)).has_value())
                        {
                            if (audit_ != nullptr)
                                audit_->record(actorId, actorRole, "",
                                    "prediction.create", "session", suppliedSession,
                                    "denied", "ownership_denied");
                            throw application::BusinessException(
                                application::BusinessException::Kind::NotFound,
                                "session is not owned by actor");
                        }
                        if (actorRole == "doctor" &&
                            !securityStore_->hasActiveAssignment(actorId, *subject))
                        {
                            if (audit_ != nullptr)
                                audit_->record(actorId, actorRole, "",
                                    "prediction.create", "session", suppliedSession,
                                    "denied", "assignment_inactive");
                            throw application::BusinessException(
                                application::BusinessException::Kind::NotFound,
                                "patient assignment was not found");
                        }
                        if (internal && (runId.empty() ||
                            !securityStore_->isAgentRunRunning(runId, *suppliedSession)))
                        {
                            if (audit_ != nullptr)
                                audit_->record(actorId, actorRole, "",
                                    "prediction.create", "session", suppliedSession,
                                    "denied", "agent_run_inactive");
                            throw application::BusinessException(
                                application::BusinessException::Kind::NotFound,
                                "agent run was not found");
                        }
                        if (audit_ != nullptr)
                            audit_->record(actorId, actorRole, "",
                                "prediction.create", "session", *suppliedSession,
                                "allowed", internal ? "agent_tool" : "authenticated");
                    }
                    auto created = predictionService_.createPrediction(
                        input,
                        suppliedSession,
                        internal ? application::SessionAccess::Internal
                                 : application::SessionAccess::Public,
                        subject, authRequired_ ? std::optional<std::string>(actorId)
                                               : std::nullopt);
                    http::ResponseWriter success = HttpErrorMapper::success(
                        BusinessJsonCodec::serializePrediction(created.first));
                    const bool setCookie = !internal && created.second.created;
                    const std::string cookie = setCookie
                        ? BusinessJsonCodec::sessionCookie(
                              created.second.session.sessionId,
                              sessionTtlSeconds_, cookieSecure_)
                        : std::string();
                    return [success = std::move(success), setCookie, cookie](
                               http::HttpResponse* response) {
                        success(response);
                        if (setCookie) response->addHeader("Set-Cookie", cookie);
                    };
                }
                model::ModelResult result = predictionService_.predict(input);
                return HttpErrorMapper::success(
                    PredictionJsonCodec::serializeResult(result));
            }
            catch (const model::ModelException& exception)
            {
                return HttpErrorMapper::from(exception);
            }
            catch (const application::BusinessException& exception)
            {
                return HttpErrorMapper::from(exception);
            }
            catch (const application::AuthException& exception)
            {
                return HttpErrorMapper::from(exception);
            }
            catch (...)
            {
                return HttpErrorMapper::internalError();
            }
        },
        std::move(responder));
}

} // namespace api
} // namespace treesem
