#include "api/AuthController.h"

#include <algorithm>
#include <chrono>

#include <nlohmann/json.hpp>

#include "api/BusinessJsonCodec.h"
#include "api/HttpErrorMapper.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem::api
{
namespace
{
using Json = nlohmann::json;
struct Credentials { std::string email; std::string password; std::string displayName; };

Credentials credentials(const std::string& body, bool requireName)
{
    try
    {
        const Json root = Json::parse(body);
        if (!root.is_object() || !root.contains("email") || !root.contains("password") ||
            !root.at("email").is_string() || !root.at("password").is_string() ||
            (requireName && (!root.contains("display_name") || !root.at("display_name").is_string())))
            throw std::invalid_argument("invalid auth request");
        return {root.at("email").get<std::string>(), root.at("password").get<std::string>(),
                requireName ? root.at("display_name").get<std::string>() : std::string()};
    }
    catch (...) { throw application::AuthException(
        application::AuthException::Kind::InvalidCredentials, "invalid auth request"); }
}

std::string cookieValue(const http::HttpRequest& request, const std::string& name)
{
    const std::string cookies = request.getHeader("Cookie");
    std::size_t begin = 0;
    while (begin < cookies.size())
    {
        const auto end = cookies.find(';', begin);
        std::string item = cookies.substr(begin, end == std::string::npos ? end : end - begin);
        const auto first = item.find_first_not_of(" \t");
        if (first != std::string::npos) item.erase(0, first);
        const auto equals = item.find('=');
        if (equals != std::string::npos && item.substr(0, equals) == name)
            return item.substr(equals + 1);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return {};
}

std::string bearer(const http::HttpRequest& request)
{
    const std::string header = request.getHeader("Authorization");
    return header.rfind("Bearer ", 0) == 0 ? header.substr(7) : std::string();
}

std::string userJson(const domain::UserRecord& user)
{
    return Json{{"user_id", user.userId}, {"role", domain::toString(user.role)},
                {"display_name", user.displayName}}.dump();
}

http::ResponseWriter authResponse(const application::AuthResult& result,
                                  bool secure, long refreshTtl,
                                  const std::string& sessionCookie)
{
    const std::string body = Json{{"access_token", result.accessToken},
        {"token_type", "Bearer"}, {"expires_in", result.accessExpiresIn},
        {"user", Json::parse(userJson(result.user))}}.dump();
    std::string cookie = "treeSemRefresh=" + result.refreshToken +
        "; Path=/api/v1/auth; HttpOnly; SameSite=Strict; Max-Age=" +
        std::to_string(refreshTtl);
    if (secure) cookie += "; Secure";
    return [body, cookie, sessionCookie](http::HttpResponse* response) {
        response->setStatusCode(http::HttpResponse::k200Ok);
        response->setStatusMessage("OK"); response->setContentType("application/json; charset=utf-8");
        response->setBody(body); response->addHeader("Set-Cookie", cookie);
        response->addHeader("Set-Cookie", sessionCookie);
        response->addHeader("Cache-Control", "no-store");
    };
}

template<typename Work> http::ResponseWriter safe(Work&& work)
{
    try { return work(); }
    catch (const application::AuthException& error) { return HttpErrorMapper::from(error); }
    catch (...) { return HttpErrorMapper::internalError(); }
}
}

AuthController::AuthController(
    const application::AuthService& auth,
    const application::SessionService& sessions,
    persistence::ISecurityStore& securityStore,
    service::BlockingTaskScheduler& scheduler,
    bool secure, long refreshTtl, std::string allowedOrigins)
    : auth_(auth), sessions_(sessions), securityStore_(securityStore),
      scheduler_(scheduler), secure_(secure),
      refreshTtlSeconds_(refreshTtl), allowedOrigins_(std::move(allowedOrigins)) {}

std::string AuthController::requestId(const http::HttpRequest& request) const
{
    const std::string supplied = request.getHeader("X-Request-Id");
    return infrastructure::isValidOpaqueId(supplied, "req_")
        ? supplied : infrastructure::generateOpaqueId("req_");
}

domain::ActorContext AuthController::actor(const http::HttpRequest& request) const
{
    const auto token = bearer(request);
    if (token.empty()) throw application::AuthException(
        application::AuthException::Kind::InvalidToken, "missing bearer token");
    return auth_.authenticate(token, requestId(request));
}

void AuthController::checkOrigin(const http::HttpRequest& request) const
{
    const std::string origin = request.getHeader("Origin");
    bool allowed = origin.empty();
    std::size_t begin = 0;
    while (!allowed && begin <= allowedOrigins_.size())
    {
        const auto end = allowedOrigins_.find(',', begin);
        allowed = allowedOrigins_.substr(begin, end == std::string::npos
            ? end : end - begin) == origin;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (!allowed)
        throw application::AuthException(application::AuthException::Kind::Forbidden,
                                         "origin is not allowed");
}

void AuthController::registerPatient(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const auto input = credentials(request.getBody(), true);
            auto session = sessions_.resolve(BusinessJsonCodec::sessionId(
                request, application::SessionAccess::Public), application::SessionAccess::Public);
            const auto result = auth_.registerPatient(input.email, input.password,
                input.displayName, session.session.sessionId, requestId(request));
            if (!securityStore_.sessionSubjectForActor(
                    session.session.sessionId, result.user.userId).has_value())
            {
                session = sessions_.createNew();
                if (!securityStore_.bindSession(session.session.sessionId,
                                                result.user.userId, result.user.userId))
                    throw application::AuthException(
                        application::AuthException::Kind::AuditUnavailable,
                        "failed to bind authenticated session");
            }
            return authResponse(result, secure_,
                refreshTtlSeconds_, BusinessJsonCodec::sessionCookie(
                    session.session.sessionId, sessions_.ttl().count(), secure_));
        });
    }, std::move(responder));
}

void AuthController::login(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const auto input = credentials(request.getBody(), false);
            auto session = sessions_.resolve(BusinessJsonCodec::sessionId(
                request, application::SessionAccess::Public), application::SessionAccess::Public);
            const auto result = auth_.login(input.email, input.password,
                session.session.sessionId, requestId(request));
            if (!securityStore_.sessionSubjectForActor(
                    session.session.sessionId, result.user.userId).has_value())
            {
                session = sessions_.createNew();
                if (!securityStore_.bindSession(session.session.sessionId,
                                                result.user.userId, result.user.userId))
                    throw application::AuthException(
                        application::AuthException::Kind::AuditUnavailable,
                        "failed to bind authenticated session");
            }
            return authResponse(result, secure_, refreshTtlSeconds_,
                BusinessJsonCodec::sessionCookie(session.session.sessionId,
                    sessions_.ttl().count(), secure_));
        });
    }, std::move(responder));
}

void AuthController::refresh(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            checkOrigin(request);
            const auto session = sessions_.resolve(BusinessJsonCodec::sessionId(
                request, application::SessionAccess::Public),
                application::SessionAccess::Public);
            const auto result = auth_.refresh(
                cookieValue(request, "treeSemRefresh"), requestId(request));
            if (!securityStore_.bindSession(session.session.sessionId,
                                            result.user.userId, result.user.userId))
                throw application::AuthException(
                    application::AuthException::Kind::Forbidden,
                    "session belongs to another account");
            return authResponse(result, secure_, refreshTtlSeconds_,
                BusinessJsonCodec::sessionCookie(session.session.sessionId,
                    sessions_.ttl().count(), secure_));
        });
    }, std::move(responder));
}

void AuthController::logout(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            checkOrigin(request); auth_.logout(cookieValue(request, "treeSemRefresh"), requestId(request));
            return [](http::HttpResponse* response) {
                response->setStatusCode(http::HttpResponse::k204NoContent);
                response->setStatusMessage("No Content");
                response->addHeader("Set-Cookie", "treeSemRefresh=; Path=/api/v1/auth; HttpOnly; SameSite=Strict; Max-Age=0");
                response->addHeader("Set-Cookie", "treeSemSession=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0");
                response->addHeader("Cache-Control", "no-store");
            };
        });
    }, std::move(responder));
}

void AuthController::me(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const auto context = actor(request);
            const std::string body = Json{{"user_id", context.userId},
                {"role", domain::toString(context.role)}}.dump();
            return HttpErrorMapper::success(body);
        });
    }, std::move(responder));
}

void AuthController::createDoctor(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const auto input = credentials(request.getBody(), true);
            const auto user = auth_.createUser(actor(request), input.email, input.password,
                input.displayName, domain::UserRole::Doctor);
            return HttpErrorMapper::json(http::HttpResponse::k201Created, "Created", userJson(user));
        });
    }, std::move(responder));
}

void AuthController::createAssignment(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const auto root = Json::parse(request.getBody());
            const auto assignment = auth_.assign(actor(request),
                root.at("doctor_user_id").get<std::string>(),
                root.at("patient_user_id").get<std::string>());
            return HttpErrorMapper::json(http::HttpResponse::k201Created, "Created",
                Json{{"assignment_id", assignment.assignmentId},
                     {"doctor_user_id", assignment.doctorUserId},
                     {"patient_user_id", assignment.patientUserId},
                     {"status", assignment.status}}.dump());
        });
    }, std::move(responder));
}

void AuthController::revokeAssignment(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            auth_.revokeAssignment(actor(request), request.getPathParameters("assignment_id"));
            return HttpErrorMapper::success(R"({"status":"revoked"})");
        });
    }, std::move(responder));
}

void AuthController::listAssignments(http::HttpRequest request,
                                     http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const auto context = actor(request);
            if (context.role != domain::UserRole::Admin)
                throw application::AuthException(
                    application::AuthException::Kind::Forbidden, "forbidden");
            Json items = Json::array();
            for (const auto& item : securityStore_.listAssignments())
                items.push_back({{"assignment_id", item.assignmentId},
                    {"doctor_user_id", item.doctorUserId},
                    {"patient_user_id", item.patientUserId}, {"status", item.status},
                    {"created_at", infrastructure::formatUtc(item.createdAt)}});
            return HttpErrorMapper::success(Json{{"items", std::move(items)}}.dump());
        });
    }, std::move(responder));
}

void AuthController::listAudit(http::HttpRequest request,
                               http::AsyncResponder responder) const
{
    scheduler_.schedule([request = std::move(request), this]() {
        return safe([&]() {
            const auto context = actor(request);
            if (context.role != domain::UserRole::Admin)
                throw application::AuthException(
                    application::AuthException::Kind::Forbidden, "forbidden");
            domain::AuditEvent queryEvent;
            queryEvent.eventId = infrastructure::generateOpaqueId("evt_");
            queryEvent.requestId = context.requestId;
            queryEvent.actorUserId = context.userId;
            queryEvent.actorRole = context.role;
            queryEvent.action = "audit.query";
            queryEvent.resourceType = "audit_event";
            queryEvent.outcome = "allowed";
            queryEvent.reasonCode = "admin_authorized";
            queryEvent.createdAt = std::chrono::system_clock::now();
            try { securityStore_.appendAudit(queryEvent); }
            catch (...) { throw application::AuthException(
                application::AuthException::Kind::AuditUnavailable,
                "audit query could not be audited"); }
            std::size_t limit = 50;
            const std::string rawLimit = request.getQueryParameters("limit");
            if (!rawLimit.empty())
            {
                limit = static_cast<std::size_t>(std::stoul(rawLimit));
                if (limit == 0 || limit > 100) throw application::AuthException(
                    application::AuthException::Kind::NotFound, "invalid limit");
            }
            std::optional<domain::AuditCursor> cursor;
            const std::string rawCursor = request.getQueryParameters("cursor");
            if (!rawCursor.empty())
            {
                const std::string decoded = infrastructure::base64UrlDecode(rawCursor, 256);
                const auto separator = decoded.find(':');
                if (separator == std::string::npos) throw std::invalid_argument("invalid cursor");
                cursor = domain::AuditCursor{infrastructure::fromEpochMicroseconds(
                    std::stoll(decoded.substr(0, separator))), decoded.substr(separator + 1)};
            }
            const auto page = securityStore_.listAudit(limit, cursor);
            Json items = Json::array();
            for (const auto& event : page.items)
                items.push_back({{"event_id", event.eventId}, {"request_id", event.requestId},
                    {"actor_user_id", event.actorUserId.has_value() ? Json(*event.actorUserId) : Json(nullptr)},
                    {"actor_role", event.actorRole.has_value() ? Json(domain::toString(*event.actorRole)) : Json(nullptr)},
                    {"action", event.action}, {"resource_type", event.resourceType},
                    {"resource_id", event.resourceId.has_value() ? Json(*event.resourceId) : Json(nullptr)},
                    {"outcome", event.outcome}, {"reason_code", event.reasonCode},
                    {"created_at", infrastructure::formatUtc(event.createdAt)}});
            Json next = nullptr;
            if (page.nextCursor.has_value())
                next = infrastructure::base64UrlEncode(
                    std::to_string(infrastructure::epochMicroseconds(
                        page.nextCursor->createdAt)) + ":" + page.nextCursor->eventId);
            return HttpErrorMapper::success(Json{{"items", std::move(items)},
                                                  {"next_cursor", std::move(next)}}.dump());
        });
    }, std::move(responder));
}
} // namespace treesem::api
