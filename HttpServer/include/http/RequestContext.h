#pragma once

#include <chrono>
#include <string>

namespace http
{

struct RequestContext
{
    std::string requestId;
    std::string traceId;
    std::string spanId;
    std::string parentSpanId;
    std::string routePattern{"unmatched"};
    std::chrono::steady_clock::time_point startedAt{};
    bool sampled{true};

    bool initialized() const noexcept
    {
        return !requestId.empty() && !traceId.empty() && !spanId.empty();
    }
};

} // namespace http
