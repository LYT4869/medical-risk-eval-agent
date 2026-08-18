#pragma once

#include <memory>
#include <string>

#include "Middleware.h"
#include "observability/MetricsRegistry.h"

namespace http::middleware
{

class ObservabilityMiddleware final : public Middleware
{
public:
    ObservabilityMiddleware(std::shared_ptr<observability::MetricsRegistry> metrics,
                            std::string serviceName,
                            double sampleRate = 1.0,
                            long slowRequestMs = 1000);
    void before(HttpRequest& request) override;
    void after(HttpResponse& response) override;
    void after(const HttpRequest& request, HttpResponse& response) override;

private:
    std::shared_ptr<observability::MetricsRegistry> metrics_;
    std::string serviceName_;
    double sampleRate_;
    long slowRequestMs_;
};

} // namespace http::middleware
