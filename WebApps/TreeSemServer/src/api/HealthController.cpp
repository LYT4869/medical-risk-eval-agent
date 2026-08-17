#include "api/HealthController.h"

#include <nlohmann/json.hpp>

namespace treesem
{
namespace api
{

HealthController::HealthController(HealthMetadata metadata)
{
    nlohmann::json body{
        {"status", "ok"},
        {"service", "treeSem-backend"},
        {"configured_backend", metadata.configuredBackend},
        {"primary_backend", metadata.primaryBackend},
        {"fallback_enabled", metadata.fallbackEnabled},
        {"storage_backend", metadata.storageBackend},
        {"database_pool_size", metadata.databasePoolSize},
        {"session_ttl_seconds", metadata.sessionTtlSeconds},
    };
    if (metadata.modelVersion.has_value())
    {
        body["model_version"] = *metadata.modelVersion;
    }
    body_ = body.dump();
}

void HealthController::handle(
    const http::HttpRequest& request,
    http::HttpResponse* response) const
{
    response->setStatusLine(
        request.getVersion(), http::HttpResponse::k200Ok, "OK");
    response->setContentType("application/json; charset=utf-8");
    response->setBody(body_);
}

} // namespace api
} // namespace treesem
