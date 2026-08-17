#pragma once

#include <functional>

#include "HttpRequest.h"
#include "HttpResponse.h"

namespace http
{

// ResponseWriter is executed by HttpServer on the connection's I/O loop.
// The worker thread only supplies the lightweight response-building function.
using ResponseWriter = std::function<void(HttpResponse*)>;

// AsyncResponder is thread-safe and one-shot when created by HttpServer.
using AsyncResponder = std::function<void(ResponseWriter)>;
using AsyncHttpCallback = std::function<void(HttpRequest, AsyncResponder)>;

} // namespace http
