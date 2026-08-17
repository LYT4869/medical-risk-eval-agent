#pragma once

#include "http/HttpRequest.h"
#include "http/HttpResponse.h"

namespace treesem
{
namespace api
{

void healthHandler(const http::HttpRequest& request, http::HttpResponse* response);

} // namespace api
} // namespace treesem
