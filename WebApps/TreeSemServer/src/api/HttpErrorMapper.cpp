#include "api/HttpErrorMapper.h"

#include <utility>

#include <nlohmann/json.hpp>

namespace treesem
{
namespace api
{

http::ResponseWriter HttpErrorMapper::json(
    http::HttpResponse::HttpStatusCode status,
    std::string statusMessage,
    std::string body)
{
    return [status,
            statusMessage = std::move(statusMessage),
            body = std::move(body)](http::HttpResponse* response) {
        response->setStatusCode(status);
        response->setStatusMessage(statusMessage);
        response->setContentType("application/json; charset=utf-8");
        response->setBody(body);
    };
}

http::ResponseWriter HttpErrorMapper::success(std::string body)
{
    return json(http::HttpResponse::k200Ok, "OK", std::move(body));
}

http::ResponseWriter HttpErrorMapper::error(
    http::HttpResponse::HttpStatusCode status,
    const std::string& statusMessage,
    const std::string& code,
    const std::string& safeMessage)
{
    return json(
        status,
        statusMessage,
        nlohmann::json{{"error", code}, {"message", safeMessage}}.dump());
}

http::ResponseWriter HttpErrorMapper::from(const ApiException& exception)
{
    if (exception.kind() == ApiException::Kind::InvalidJson)
    {
        return error(
            http::HttpResponse::k400BadRequest,
            "Bad Request",
            "invalid_json",
            "Request body must be valid JSON.");
    }
    return error(
        http::HttpResponse::k400BadRequest,
        "Bad Request",
        "invalid_request",
        "Prediction request does not match the API contract.");
}

http::ResponseWriter HttpErrorMapper::from(const model::ModelException& exception)
{
    switch (exception.kind())
    {
    case model::ModelException::Kind::InvalidInput:
        return error(
            http::HttpResponse::k400BadRequest,
            "Bad Request",
            "invalid_request",
            "The model rejected the prediction input.");
    case model::ModelException::Kind::Timeout:
        return error(
            http::HttpResponse::k504GatewayTimeout,
            "Gateway Timeout",
            "model_adapter_timeout",
            "The model service did not respond before the timeout.");
    case model::ModelException::Kind::InvalidResponse:
        return error(
            http::HttpResponse::k502BadGateway,
            "Bad Gateway",
            "invalid_model_adapter_response",
            "The model service returned an invalid response.");
    case model::ModelException::Kind::DownstreamFailure:
        return error(
            http::HttpResponse::k502BadGateway,
            "Bad Gateway",
            "model_adapter_failed",
            "The model service failed to execute the prediction.");
    case model::ModelException::Kind::Unavailable:
        return error(
            http::HttpResponse::k502BadGateway,
            "Bad Gateway",
            "model_adapter_unavailable",
            "The model service is unavailable.");
    case model::ModelException::Kind::InferenceFailure:
        return error(
            http::HttpResponse::k500InternalServerError,
            "Internal Server Error",
            "model_inference_failed",
            "The local model could not complete inference.");
    }
    return internalError();
}

http::ResponseWriter HttpErrorMapper::from(
    const application::BusinessException& exception)
{
    using Kind = application::BusinessException::Kind;
    switch (exception.kind())
    {
    case Kind::InvalidInput:
        return error(
            http::HttpResponse::k400BadRequest, "Bad Request",
            "invalid_request", "The business request is invalid.");
    case Kind::InvalidSession:
        return error(
            http::HttpResponse::k400BadRequest, "Bad Request",
            "invalid_session", "The session identifier is missing or invalid.");
    case Kind::NotFound:
        return error(
            http::HttpResponse::k404NotFound, "Not Found",
            "resource_not_found", "The requested resource was not found.");
    case Kind::Conflict:
        return error(
            http::HttpResponse::k409Conflict, "Conflict",
            "session_conflict", "The session state changed before the request completed.");
    case Kind::IdempotencyConflict:
        return error(
            http::HttpResponse::k409Conflict, "Conflict",
            "idempotency_conflict", "The idempotency key was reused for another request.");
    case Kind::AgentRunInProgress:
        return error(
            http::HttpResponse::k409Conflict, "Conflict",
            "agent_run_in_progress", "The agent run is still in progress.");
    case Kind::DatabaseBusy:
        return error(
            http::HttpResponse::k503ServiceUnavailable, "Service Unavailable",
            "database_busy", "The database connection pool is busy.");
    case Kind::DatabaseUnavailable:
        return error(
            http::HttpResponse::k503ServiceUnavailable, "Service Unavailable",
            "database_unavailable", "The database is unavailable.");
    case Kind::PersistenceFailure:
        return error(
            http::HttpResponse::k500InternalServerError, "Internal Server Error",
            "persistence_error", "The service could not persist the request.");
    }
    return internalError();
}

http::ResponseWriter HttpErrorMapper::from(const client::AgentClientException& exception)
{
    using Kind = client::AgentClientException::Kind;
    switch (exception.kind())
    {
    case Kind::Timeout:
        return error(http::HttpResponse::k504GatewayTimeout, "Gateway Timeout",
                     "agent_timeout", "The agent did not finish before the timeout.");
    case Kind::Unavailable:
        return error(http::HttpResponse::k502BadGateway, "Bad Gateway",
                     "agent_unavailable", "The agent service is unavailable.");
    case Kind::InvalidResponse:
        return error(http::HttpResponse::k502BadGateway, "Bad Gateway",
                     "invalid_agent_response", "The agent returned an invalid response.");
    case Kind::ExecutionFailed:
        return error(http::HttpResponse::k502BadGateway, "Bad Gateway",
                     "agent_execution_failed", "The agent could not complete the request.");
    }
    return internalError();
}

http::ResponseWriter HttpErrorMapper::from(const application::AuthException& exception)
{
    using Kind = application::AuthException::Kind;
    switch (exception.kind())
    {
    case Kind::InvalidCredentials:
        return error(http::HttpResponse::k401Unauthorized, "Unauthorized",
                     "invalid_credentials", "The email or password is invalid.");
    case Kind::InvalidToken:
        return error(http::HttpResponse::k401Unauthorized, "Unauthorized",
                     "invalid_access_token", "The authentication token is invalid or expired.");
    case Kind::InvalidRefreshToken:
        return error(http::HttpResponse::k401Unauthorized, "Unauthorized",
                     "invalid_refresh_token", "The refresh token is invalid or expired.");
    case Kind::RateLimited:
        return error(http::HttpResponse::k429TooManyRequests, "Too Many Requests",
                     "authentication_rate_limited", "Authentication is temporarily rate limited.");
    case Kind::Conflict:
        return error(http::HttpResponse::k409Conflict, "Conflict",
                     "account_conflict", "The account request conflicts with existing state.");
    case Kind::Forbidden:
        return error(http::HttpResponse::k403Forbidden, "Forbidden",
                     "forbidden", "The authenticated user cannot perform this action.");
    case Kind::NotFound:
        return error(http::HttpResponse::k404NotFound, "Not Found",
                     "resource_not_found", "The requested resource was not found.");
    case Kind::AuditUnavailable:
        return error(http::HttpResponse::k503ServiceUnavailable, "Service Unavailable",
                     "audit_unavailable", "The security audit store is unavailable.");
    }
    return internalError();
}

http::ResponseWriter HttpErrorMapper::internalError()
{
    return error(
        http::HttpResponse::k500InternalServerError,
        "Internal Server Error",
        "internal_error",
        "The server could not complete the request.");
}

} // namespace api
} // namespace treesem
