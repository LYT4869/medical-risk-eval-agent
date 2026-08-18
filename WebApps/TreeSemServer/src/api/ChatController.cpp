#include "api/ChatController.h"

#include "api/BusinessJsonCodec.h"
#include "api/ChatJsonCodec.h"
#include "api/HttpErrorMapper.h"
#include "application/BusinessException.h"
#include "client/IAgentClient.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem::api
{
namespace
{
template<typename Work> http::ResponseWriter safe(Work&& work)
{
    try { return work(); }
    catch (const ApiException& error) { return HttpErrorMapper::from(error); }
    catch (const application::BusinessException& error) { return HttpErrorMapper::from(error); }
    catch (const application::AuthException& error) { return HttpErrorMapper::from(error); }
    catch (const client::AgentClientException& error) { return HttpErrorMapper::from(error); }
    catch (...) { return HttpErrorMapper::internalError(); }
}
http::ResponseWriter withCookie(http::ResponseWriter writer,
                                const application::ResolvedSession& session,
                                long ttl, bool secure)
{
    const std::string cookie = session.created
        ? BusinessJsonCodec::sessionCookie(session.session.sessionId, ttl, secure)
        : std::string();
    return [writer = std::move(writer), cookie](http::HttpResponse* response) {
        writer(response);
        if (!cookie.empty()) response->addHeader("Set-Cookie", cookie);
        response->addHeader("Cache-Control", "no-store");
    };
}
}

ChatController::ChatController(
    const application::AgentApplicationService& service,
    const application::SessionService& sessions,
    service::BlockingTaskScheduler& agentScheduler,
    service::BlockingTaskScheduler& databaseScheduler,
    bool cookieSecure, long sessionTtlSeconds, std::size_t maxMessageCharacters,
    persistence::ISecurityStore* securityStore,
    const security::JwtService* jwt,
    bool authRequired,
    const application::AuditService* audit)
    : service_(service), sessions_(sessions), agentScheduler_(agentScheduler),
      databaseScheduler_(databaseScheduler), cookieSecure_(cookieSecure),
      sessionTtlSeconds_(sessionTtlSeconds), maxMessageCharacters_(maxMessageCharacters),
      securityStore_(securityStore), jwt_(jwt), authRequired_(authRequired),
      audit_(audit) {}

void ChatController::chat(http::HttpRequest request, http::AsyncResponder responder) const
{
    ParsedChatRequest parsed;
    try { parsed = ChatJsonCodec::parse(request, maxMessageCharacters_); }
    catch (const ApiException& error) { responder(HttpErrorMapper::from(error)); return; }
    const auto supplied = BusinessJsonCodec::sessionId(request, application::SessionAccess::Public);
    const std::string actorId = request.getHeader("X-TreeSem-Actor-Id");
    const std::string actorRole = request.getHeader("X-TreeSem-Actor-Role");
    const client::TraceCarrier trace{request.requestContext().requestId,
        request.requestContext().traceId, request.requestContext().spanId};
    agentScheduler_.schedule([parsed = std::move(parsed), supplied, actorId, actorRole,
                              trace, this]() {
        return safe([&]() {
            std::optional<domain::ActorContext> actor;
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
                        audit_->record(actorId, actorRole, "", "agent.chat",
                            "session", supplied, "denied",
                            "admin_has_no_clinical_access");
                    throw application::AuthException(
                        application::AuthException::Kind::Forbidden,
                        "administrators cannot access clinical chat");
                }
                if (!supplied.has_value() || actorId.empty() || securityStore_ == nullptr)
                    throw application::BusinessException(
                        application::BusinessException::Kind::NotFound, "session not found");
                const auto subject = securityStore_->sessionSubjectForActor(*supplied, actorId);
                if (!subject.has_value())
                {
                    if (audit_ != nullptr)
                        audit_->record(actorId, actorRole, "", "agent.chat",
                            "session", supplied, "denied", "ownership_denied");
                    throw application::BusinessException(
                        application::BusinessException::Kind::NotFound, "session not found");
                }
                actor = domain::ActorContext{actorId, domain::parseUserRole(actorRole), 0,
                    supplied, subject, trace.requestId};
                if (audit_ != nullptr)
                    audit_->record(actorId, actorRole, actor->requestId,
                        "agent.chat", "session", *supplied,
                        "allowed", "authenticated");
            }
            const auto result = service_.chat(parsed.message, parsed.idempotencyKey,
                supplied, application::SessionAccess::Public, actor, trace);
            return withCookie(HttpErrorMapper::success(ChatJsonCodec::serialize(result)),
                              result.session, sessionTtlSeconds_, cookieSecure_);
        });
    }, std::move(responder));
}

void ChatController::history(http::HttpRequest request, http::AsyncResponder responder) const
{
    databaseScheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const auto session = sessions_.resolve(
                BusinessJsonCodec::sessionId(request, application::SessionAccess::Public),
                application::SessionAccess::Public);
            if (authRequired_)
            {
                const std::string actor = request.getHeader("X-TreeSem-Actor-Id");
                const std::string role = request.getHeader("X-TreeSem-Actor-Role");
                const auto activeUser = securityStore_ == nullptr
                    ? std::nullopt : securityStore_->findUserById(actor);
                if (!activeUser.has_value() || activeUser->status != "active" ||
                    domain::toString(activeUser->role) != role)
                    throw application::AuthException(
                        application::AuthException::Kind::InvalidToken,
                        "account is no longer active");
                if (role == "admin")
                {
                    if (audit_ != nullptr)
                        audit_->record(actor, "admin", request.getHeader("X-Request-Id"),
                            "agent.history", "session", session.session.sessionId,
                            "denied", "admin_has_no_clinical_access");
                    throw application::AuthException(
                        application::AuthException::Kind::Forbidden,
                        "administrators cannot access clinical chat");
                }
                if (actor.empty() || securityStore_ == nullptr ||
                    !securityStore_->sessionSubjectForActor(
                        session.session.sessionId, actor).has_value())
                {
                    if (audit_ != nullptr)
                        audit_->record(actor, role,
                            request.getHeader("X-Request-Id"), "agent.history",
                            "session", session.session.sessionId,
                            "denied", "ownership_denied");
                    throw application::BusinessException(
                        application::BusinessException::Kind::NotFound, "session not found");
                }
                if (audit_ != nullptr)
                    audit_->record(actor, role,
                        request.getHeader("X-Request-Id"), "agent.history",
                        "session", session.session.sessionId, "allowed", "authenticated");
            }
            auto writer = HttpErrorMapper::success(ChatJsonCodec::serializeHistory(
                service_.history(session.session.sessionId,
                    BusinessJsonCodec::parseHistoryLimit(request),
                    ChatJsonCodec::parseCursor(request))));
            return withCookie(std::move(writer), session, sessionTtlSeconds_, cookieSecure_);
        });
    }, std::move(responder));
}
} // namespace treesem::api
