#include "api/MetricsController.h"

#include <openssl/crypto.h>
#include <stdexcept>

namespace treesem::api
{
namespace
{

bool constantTimeEqual(const std::string& left, const std::string& right)
{
    if (left.size() != right.size()) return false;
    return left.empty() || CRYPTO_memcmp(left.data(), right.data(), left.size()) == 0;
}

void unauthorized(http::HttpResponse* response)
{
    response->setStatusCode(http::HttpResponse::k401Unauthorized);
    response->setStatusMessage("Unauthorized");
    response->setContentType("application/json; charset=utf-8");
    response->addHeader("Cache-Control", "no-store");
    response->setBody(
        R"({"error":"metrics_authentication_required","message":"Metrics authentication is required."})");
}

} // namespace

MetricsController::MetricsController(
    std::shared_ptr<http::observability::MetricsRegistry> registry,
    std::string bearerToken)
    : registry_(std::move(registry)), bearerToken_(std::move(bearerToken))
{
    if (!registry_) throw std::invalid_argument("metrics registry is required");
}

void MetricsController::handle(const http::HttpRequest& request,
                               http::HttpResponse* response) const
{
    if (!bearerToken_.empty())
    {
        const std::string authorization = request.getHeader("Authorization");
        const std::string expected = "Bearer " + bearerToken_;
        if (!constantTimeEqual(authorization, expected))
        {
            unauthorized(response);
            return;
        }
    }
    response->setStatusCode(http::HttpResponse::k200Ok);
    response->setStatusMessage("OK");
    response->setContentType("text/plain; version=0.0.4; charset=utf-8");
    response->addHeader("Cache-Control", "no-store");
    response->setBody(registry_->renderPrometheus());
}

} // namespace treesem::api
