#include "api/DoctorController.h"

#include <nlohmann/json.hpp>

#include "api/BusinessJsonCodec.h"
#include "api/ChatJsonCodec.h"
#include "api/HttpErrorMapper.h"
#include "api/PredictionJsonCodec.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem::api
{
namespace
{
template<typename Work> http::ResponseWriter safe(Work&& work)
{
    try { return work(); }
    catch (const ApiException& error) { return HttpErrorMapper::from(error); }
    catch (const application::AuthException& error) { return HttpErrorMapper::from(error); }
    catch (const application::BusinessException& error) { return HttpErrorMapper::from(error); }
    catch (const client::AgentClientException& error) { return HttpErrorMapper::from(error); }
    catch (const model::ModelException& error) { return HttpErrorMapper::from(error); }
    catch (...) { return HttpErrorMapper::internalError(); }
}

http::ResponseWriter cookie(http::ResponseWriter writer,
                            const application::ResolvedSession& session,
                            bool secure, long ttl)
{
    const std::string value = session.created
        ? BusinessJsonCodec::sessionCookie(session.session.sessionId, ttl, secure)
        : std::string();
    return [writer = std::move(writer), value](http::HttpResponse* response) {
        writer(response); if (!value.empty()) response->addHeader("Set-Cookie", value);
        response->addHeader("Cache-Control", "no-store");
    };
}
}

DoctorController::DoctorController(
    const application::AuthService& auth, const application::SessionService& sessions,
    const application::PredictionService& predictions,
    const application::AgentApplicationService* agent,
    const application::FeedbackService& feedback, persistence::ITreeSemStore& store,
    persistence::ISecurityStore& securityStore,
    service::BlockingTaskScheduler& predictionScheduler,
    service::BlockingTaskScheduler* agentScheduler,
    service::BlockingTaskScheduler& databaseScheduler,
    const application::AuditService& audit, bool cookieSecure,
    long sessionTtlSeconds, std::size_t maxMessageCharacters)
    : auth_(auth), sessions_(sessions), predictions_(predictions), agent_(agent),
      feedback_(feedback), store_(store), securityStore_(securityStore),
      predictionScheduler_(predictionScheduler), agentScheduler_(agentScheduler),
      databaseScheduler_(databaseScheduler), audit_(audit), cookieSecure_(cookieSecure),
      sessionTtlSeconds_(sessionTtlSeconds), maxMessageCharacters_(maxMessageCharacters) {}

domain::ActorContext DoctorController::doctor(
    const http::HttpRequest& request, const std::string& patientId) const
{
    domain::ActorContext actor;
    actor.userId = request.getHeader("X-TreeSem-Actor-Id");
    actor.role = domain::parseUserRole(request.getHeader("X-TreeSem-Actor-Role"));
    actor.requestId = request.requestContext().requestId;
    actor.subjectUserId = patientId;
    const auto activeUser = securityStore_.findUserById(actor.userId);
    if (!activeUser.has_value() || activeUser->status != "active" ||
        activeUser->role != actor.role)
        throw application::AuthException(
            application::AuthException::Kind::InvalidToken,
            "account is no longer active");
    if (actor.role != domain::UserRole::Doctor)
        throw application::AuthException(application::AuthException::Kind::Forbidden,
                                         "doctor role required");
    try { auth_.authorizeSubject(actor, patientId); }
    catch (...)
    {
        audit_.record(actor.userId, domain::toString(actor.role), actor.requestId,
                      "clinical.access", "patient", patientId,
                      "denied", "assignment_or_role_denied");
        throw;
    }
    return actor;
}

application::ResolvedSession DoctorController::contextSession(
    const http::HttpRequest& request, const domain::ActorContext& actor,
    const std::string& patientId) const
{
    const auto supplied = BusinessJsonCodec::sessionId(request, application::SessionAccess::Public);
    if (supplied.has_value())
    {
        const auto subject = securityStore_.sessionSubjectForActor(*supplied, actor.userId);
        if (subject == patientId)
            return sessions_.resolve(supplied, application::SessionAccess::Internal);
    }
    auto created = sessions_.createNew();
    if (!securityStore_.bindSession(created.session.sessionId, actor.userId, patientId))
        throw application::BusinessException(application::BusinessException::Kind::PersistenceFailure,
                                             "failed to bind doctor session");
    return created;
}

void DoctorController::predict(http::HttpRequest request, http::AsyncResponder responder) const
{
    model::ModelInput input;
    try { input = PredictionJsonCodec::parseInput(request.getBody()); }
    catch (const ApiException& error) { responder(HttpErrorMapper::from(error)); return; }
    predictionScheduler_.schedule([request = std::move(request), input = std::move(input), this]() {
        return safe([&]() {
            const std::string patient = request.getPathParameters("patient_id");
            auto actor = doctor(request, patient);
            auto session = contextSession(request, actor, patient);
            audit_.record(actor.userId, domain::toString(actor.role), actor.requestId,
                          "prediction.create", "patient", patient,
                          "allowed", "active_assignment");
            auto result = predictions_.createPrediction(input, session.session.sessionId,
                application::SessionAccess::Internal, patient, actor.userId);
            return cookie(HttpErrorMapper::success(
                BusinessJsonCodec::serializePrediction(result.first)), session,
                cookieSecure_, sessionTtlSeconds_);
        });
    }, std::move(responder));
}

void DoctorController::chat(http::HttpRequest request, http::AsyncResponder responder) const
{
    if (agent_ == nullptr || agentScheduler_ == nullptr)
    {
        responder(HttpErrorMapper::json(http::HttpResponse::k503ServiceUnavailable,
            "Service Unavailable",
            R"({"error":"agent_unavailable","message":"The agent service is disabled."})"));
        return;
    }
    ParsedChatRequest parsed;
    try { parsed = ChatJsonCodec::parse(request, maxMessageCharacters_); }
    catch (const ApiException& error) { responder(HttpErrorMapper::from(error)); return; }
    agentScheduler_->schedule([request = std::move(request), parsed = std::move(parsed), this]() {
        return safe([&]() {
            const std::string patient = request.getPathParameters("patient_id");
            auto actor = doctor(request, patient);
            auto session = contextSession(request, actor, patient);
            actor.sessionId = session.session.sessionId;
            audit_.record(actor.userId, domain::toString(actor.role), actor.requestId,
                          "agent.chat", "patient", patient,
                          "allowed", "active_assignment");
            auto result = agent_->chat(parsed.message, parsed.idempotencyKey,
                session.session.sessionId, application::SessionAccess::Internal, actor,
                client::TraceCarrier{request.requestContext().requestId,
                    request.requestContext().traceId,
                    request.requestContext().spanId});
            return cookie(HttpErrorMapper::success(ChatJsonCodec::serialize(result)),
                          session, cookieSecure_, sessionTtlSeconds_);
        });
    }, std::move(responder));
}

void DoctorController::history(http::HttpRequest request, http::AsyncResponder responder) const
{
    databaseScheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const std::string patient = request.getPathParameters("patient_id");
            const auto actor = doctor(request, patient);
            audit_.record(actor.userId, domain::toString(actor.role), actor.requestId,
                          "prediction.history", "patient", patient,
                          "allowed", "active_assignment");
            return HttpErrorMapper::success(BusinessJsonCodec::serializeHistory(
                store_.listPredictionsBySubject(patient,
                    BusinessJsonCodec::parseHistoryLimit(request),
                    BusinessJsonCodec::parseHistoryCursor(request))));
        });
    }, std::move(responder));
}

void DoctorController::feedback(http::HttpRequest request, http::AsyncResponder responder) const
{
    databaseScheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const std::string predictionId = request.getPathParameters("prediction_id");
            const auto prediction = store_.findPredictionAny(predictionId);
            if (!prediction.has_value() || !prediction->subjectUserId.has_value())
                throw application::BusinessException(
                application::BusinessException::Kind::NotFound, "prediction not found");
            const std::string patient = *prediction->subjectUserId;
            auto actor = doctor(request, patient);
            audit_.record(actor.userId, domain::toString(actor.role), actor.requestId,
                          "feedback.create", "prediction", predictionId,
                          "allowed", "verified_doctor");
            const auto root = nlohmann::json::parse(request.getBody());
            application::FeedbackInput input;
            input.reviewerReference = actor.userId;
            input.reviewerVerified = true;
            input.idempotencyKey = request.getHeader("Idempotency-Key");
            input.assessment = domain::parseFeedbackAssessment(
                root.at("assessment").get<std::string>());
            if (root.contains("corrected_label")) input.correctedLabel = root.at("corrected_label").get<int>();
            if (root.contains("comment")) input.comment = root.at("comment").get<std::string>();
            const auto result = feedback_.submit(prediction->sessionId, predictionId, input);
            return HttpErrorMapper::json(result.created ? http::HttpResponse::k201Created
                                                        : http::HttpResponse::k200Ok,
                result.created ? "Created" : "OK",
                BusinessJsonCodec::serializeFeedback(result.feedback));
        });
    }, std::move(responder));
}
} // namespace treesem::api
