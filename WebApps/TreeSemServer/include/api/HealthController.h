#pragma once

#include <optional>
#include <string>

#include "http/HttpRequest.h"
#include "http/HttpResponse.h"

namespace treesem
{
namespace api
{

struct HealthMetadata
{
    std::optional<std::string> modelVersion;
    std::string configuredBackend;
    std::string primaryBackend;
    bool fallbackEnabled{false};
};

class HealthController
{
public:
    explicit HealthController(HealthMetadata metadata);
    void handle(const http::HttpRequest& request, http::HttpResponse* response) const;

private:
    std::string body_;
};

} // namespace api
} // namespace treesem
