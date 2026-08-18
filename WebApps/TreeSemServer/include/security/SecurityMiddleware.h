#pragma once

#include <memory>
#include <string>

#include "middleware/Middleware.h"
#include "observability/MetricsRegistry.h"
#include "security/JwtService.h"

namespace treesem::security
{
class SecurityMiddleware : public http::middleware::Middleware
{
public:
    SecurityMiddleware(const JwtService& jwt, bool required,
                       std::string allowedOrigins,
                       std::shared_ptr<http::observability::MetricsRegistry> metrics = {});
    void before(http::HttpRequest& request) override;
    void after(http::HttpResponse& response) override;
    void after(const http::HttpRequest& request,
               http::HttpResponse& response) override;
private:
    const JwtService& jwt_;
    bool required_;
    std::string allowedOrigins_;
    std::shared_ptr<http::observability::MetricsRegistry> metrics_;
};
} // namespace treesem::security
