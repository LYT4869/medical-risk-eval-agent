#include <cassert>
#include <chrono>
#include <memory>
#include <regex>
#include <string>

#include "api/MetricsController.h"
#include "middleware/ObservabilityMiddleware.h"
#include "observability/MetricsRegistry.h"

int main()
{
    auto registry = std::make_shared<http::observability::MetricsRegistry>();
    http::middleware::ObservabilityMiddleware middleware(
        registry, "treesem-test", 1.0, 1000);
    http::HttpRequest request;
    const std::string method = "GET";
    request.setMethod(method.data(), method.data() + method.size());
    request.mutableRequestContext().routePattern =
        "/api/v1/predictions/:prediction_id";
    request.setHeader("X-Request-Id", "attacker-controlled");
    request.setHeader("traceparent",
                      "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01");
    middleware.before(request);
    assert(std::regex_match(request.requestContext().requestId,
                            std::regex("req_[0-9a-f]{32}")));
    assert(request.requestContext().traceId ==
           "0123456789abcdef0123456789abcdef");
    assert(request.requestContext().parentSpanId == "0123456789abcdef");

    http::HttpResponse response(false);
    response.setStatusCode(http::HttpResponse::k200Ok);
    middleware.after(request, response);
    assert(response.getHeader("X-Request-Id") == request.requestContext().requestId);
    assert(response.getHeader("X-Trace-Id") == request.requestContext().traceId);
    const std::string rendered = registry->renderPrometheus();
    assert(rendered.find("treesem_http_requests_total") != std::string::npos);
    assert(rendered.find("/api/v1/predictions/:prediction_id") != std::string::npos);
    assert(rendered.find("attacker-controlled") == std::string::npos);
    assert(rendered.find(request.requestContext().requestId) == std::string::npos);

    treesem::api::MetricsController controller(registry, "secure-token");
    http::HttpResponse denied(false);
    controller.handle(request, &denied);
    assert(denied.getStatusCode() == http::HttpResponse::k401Unauthorized);
    request.setHeader("Authorization", "Bearer secure-token");
    http::HttpResponse allowed(false);
    controller.handle(request, &allowed);
    assert(allowed.getStatusCode() == http::HttpResponse::k200Ok);
    assert(allowed.body().find("treesem_http_request_duration_seconds") !=
           std::string::npos);
}
