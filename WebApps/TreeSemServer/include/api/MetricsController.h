#pragma once

#include <memory>
#include <string>

#include "http/HttpRequest.h"
#include "http/HttpResponse.h"
#include "observability/MetricsRegistry.h"

namespace treesem::api
{

class MetricsController
{
public:
    MetricsController(std::shared_ptr<http::observability::MetricsRegistry> registry,
                      std::string bearerToken);
    void handle(const http::HttpRequest& request,
                http::HttpResponse* response) const;
private:
    std::shared_ptr<http::observability::MetricsRegistry> registry_;
    std::string bearerToken_;
};

} // namespace treesem::api
