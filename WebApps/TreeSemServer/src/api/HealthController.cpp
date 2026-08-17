#include "api/HealthController.h"

#include <string>

namespace treesem
{
namespace api
{

void healthHandler(const http::HttpRequest& request, http::HttpResponse* response)
{
    static const std::string body =
        R"({"status":"ok","service":"treeSem-backend"})";

    response->setStatusLine(
        request.getVersion(), http::HttpResponse::k200Ok, "OK");
    response->setContentType("application/json; charset=utf-8");
    response->setBody(body);
}

} // namespace api
} // namespace treesem
