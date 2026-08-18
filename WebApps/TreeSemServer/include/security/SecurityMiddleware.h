#pragma once

#include <string>

#include "middleware/Middleware.h"
#include "security/JwtService.h"

namespace treesem::security
{
class SecurityMiddleware : public http::middleware::Middleware
{
public:
    SecurityMiddleware(const JwtService& jwt, bool required,
                       std::string allowedOrigins);
    void before(http::HttpRequest& request) override;
    void after(http::HttpResponse& response) override;
    void after(const http::HttpRequest& request,
               http::HttpResponse& response) override;
private:
    const JwtService& jwt_;
    bool required_;
    std::string allowedOrigins_;
};
} // namespace treesem::security
