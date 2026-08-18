#include "security/SecurityMiddleware.h"

#include <algorithm>
#include <chrono>

#include <nlohmann/json.hpp>

#include "infrastructure/support/ValueSupport.h"

namespace treesem::security
{
namespace
{
[[noreturn]] void reject(http::HttpResponse::HttpStatusCode status,
                         const std::string& code, const std::string& message)
{
    http::HttpResponse response;
    response.setStatusCode(status);
    response.setStatusMessage(status == http::HttpResponse::k403Forbidden
        ? "Forbidden" : "Unauthorized");
    response.setContentType("application/json; charset=utf-8");
    response.setBody(nlohmann::json{{"error", code}, {"message", message}}.dump());
    response.addHeader("Cache-Control", "no-store");
    throw response;
}

std::string bearer(const http::HttpRequest& request)
{
    const std::string header = request.getHeader("Authorization");
    return header.rfind("Bearer ", 0) == 0 ? header.substr(7) : std::string();
}

std::string requiredTool(const http::HttpRequest& request)
{
    const std::string path = request.path();
    if (path == "/internal/v1/predictions" && request.method() == http::HttpRequest::kPost)
        return "predict_sample";
    if (path.find("/internal/v1/explanations/") == 0) return "get_explanation";
    if (path.find("/internal/v1/sessions/") == 0) return "get_prediction_history";
    if (path == "/internal/v1/comparisons") return "compare_predictions";
    if (path.find("/feedback") != std::string::npos) return {};
    if (path.find("/internal/v1/predictions/") == 0 &&
        request.method() == http::HttpRequest::kGet) return "get_prediction";
    return {};
}
}

SecurityMiddleware::SecurityMiddleware(
    const JwtService& jwt, bool required, std::string allowedOrigins)
    : jwt_(jwt), required_(required), allowedOrigins_(std::move(allowedOrigins)) {}

void SecurityMiddleware::before(http::HttpRequest& request)
{
    request.setHeader("X-TreeSem-Actor-Id", "");
    request.setHeader("X-TreeSem-Actor-Role", "");
    request.setHeader("X-TreeSem-Subject-Id", "");
    request.setHeader("X-TreeSem-Agent-Run-Id", "");
    const std::string origin = request.getHeader("Origin");
    if (!origin.empty())
    {
        bool allowed = false;
        std::size_t begin = 0;
        while (begin <= allowedOrigins_.size())
        {
            const auto end = allowedOrigins_.find(',', begin);
            if (allowedOrigins_.substr(begin, end == std::string::npos
                    ? end : end - begin) == origin) { allowed = true; break; }
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        if (!allowed) reject(http::HttpResponse::k403Forbidden,
            "forbidden", "The request origin is not allowed.");
    }
    if (request.method() == http::HttpRequest::kOptions)
    {
        http::HttpResponse response;
        response.setStatusCode(http::HttpResponse::k204NoContent);
        response.setStatusMessage("No Content");
        response.addHeader("Access-Control-Allow-Origin", origin);
        response.addHeader("Access-Control-Allow-Credentials", "true");
        response.addHeader("Access-Control-Allow-Methods",
                           "GET, POST, DELETE, OPTIONS");
        response.addHeader("Access-Control-Allow-Headers",
                           "Authorization, Content-Type, Idempotency-Key");
        response.addHeader("Access-Control-Max-Age", "600");
        response.addHeader("Vary", "Origin");
        throw response;
    }
    if (!required_) return;
    const std::string path = request.path();
    if (path == "/health" || path == "/ready" ||
        path.rfind("/api/v1/auth/", 0) == 0) return;
    const auto now = domain::TimePoint(std::chrono::microseconds(
        infrastructure::epochMicroseconds(std::chrono::system_clock::now())));
    const bool internal = path.rfind("/internal/", 0) == 0;
    const std::string token = bearer(request);
    if (token.empty()) reject(http::HttpResponse::k401Unauthorized,
        "authentication_required", "Authentication is required.");
    try
    {
        if (internal)
        {
            const auto capability = jwt_.verifyCapability(token, now);
            const std::string tool = requiredTool(request);
            if (tool.empty() || std::find(capability.allowedTools.begin(),
                    capability.allowedTools.end(), tool) == capability.allowedTools.end() ||
                request.getHeader("X-TreeSem-Session-Id") != capability.sessionId)
                reject(http::HttpResponse::k403Forbidden, "capability_forbidden",
                       "The capability does not permit this tool call.");
            request.setHeader("X-TreeSem-Actor-Id", capability.actorId);
            request.setHeader("X-TreeSem-Actor-Role", domain::toString(capability.actorRole));
            request.setHeader("X-TreeSem-Subject-Id", capability.subjectUserId);
            request.setHeader("X-TreeSem-Agent-Run-Id", capability.runId);
        }
        else
        {
            const auto actor = jwt_.verifyAccess(token, now);
            request.setHeader("X-TreeSem-Actor-Id", actor.userId);
            request.setHeader("X-TreeSem-Actor-Role", domain::toString(actor.role));
            request.setHeader("X-TreeSem-Subject-Id", actor.userId);
        }
    }
    catch (const http::HttpResponse&) { throw; }
    catch (...)
    {
        if (internal)
            reject(http::HttpResponse::k403Forbidden, "capability_forbidden",
                   "The internal capability is invalid, expired, or out of scope.");
        reject(http::HttpResponse::k401Unauthorized, "invalid_access_token",
               "The authentication token is invalid or expired.");
    }
}

void SecurityMiddleware::after(http::HttpResponse& response)
{
    response.addHeader("X-Content-Type-Options", "nosniff");
    response.addHeader("Referrer-Policy", "no-referrer");
    response.addHeader("Cache-Control", "no-store");
}

void SecurityMiddleware::after(
    const http::HttpRequest& request, http::HttpResponse& response)
{
    after(response);
    const std::string origin = request.getHeader("Origin");
    if (!origin.empty())
    {
        response.addHeader("Access-Control-Allow-Origin", origin);
        response.addHeader("Access-Control-Allow-Credentials", "true");
        response.addHeader("Vary", "Origin");
    }
}
} // namespace treesem::security
