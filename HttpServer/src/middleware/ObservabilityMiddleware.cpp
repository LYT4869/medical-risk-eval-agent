#include "middleware/ObservabilityMiddleware.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <muduo/base/Logging.h>
#include <openssl/rand.h>

namespace http::middleware
{
namespace
{

bool isLowerHex(const std::string& value, std::size_t length)
{
    return value.size() == length && std::all_of(
        value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
        });
}

std::string randomHex(std::size_t bytes)
{
    std::vector<unsigned char> buffer(bytes);
    if (RAND_bytes(buffer.data(), static_cast<int>(buffer.size())) != 1)
        throw std::runtime_error("secure random generation failed");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char value : buffer) out << std::setw(2) << unsigned(value);
    return out.str();
}

bool allZeros(const std::string& value)
{
    return std::all_of(value.begin(), value.end(), [](char ch) { return ch == '0'; });
}

void parseTraceParent(const std::string& header, RequestContext& context)
{
    if (header.size() != 55 || header[2] != '-' || header[35] != '-' ||
        header[52] != '-' || header.substr(0, 2) != "00") return;
    const std::string traceId = header.substr(3, 32);
    const std::string parentSpan = header.substr(36, 16);
    const std::string flags = header.substr(53, 2);
    if (!isLowerHex(traceId, 32) || !isLowerHex(parentSpan, 16) ||
        !isLowerHex(flags, 2) || allZeros(traceId) || allZeros(parentSpan)) return;
    context.traceId = traceId;
    context.parentSpanId = parentSpan;
}

std::string methodName(HttpRequest::Method method)
{
    switch (method)
    {
    case HttpRequest::kGet: return "GET";
    case HttpRequest::kPost: return "POST";
    case HttpRequest::kHead: return "HEAD";
    case HttpRequest::kPut: return "PUT";
    case HttpRequest::kDelete: return "DELETE";
    case HttpRequest::kOptions: return "OPTIONS";
    default: return "INVALID";
    }
}

} // namespace

ObservabilityMiddleware::ObservabilityMiddleware(
    std::shared_ptr<observability::MetricsRegistry> metrics,
    std::string serviceName, double sampleRate, long slowRequestMs)
    : metrics_(std::move(metrics)), serviceName_(std::move(serviceName)),
      sampleRate_(sampleRate), slowRequestMs_(slowRequestMs)
{
    if (!metrics_ || serviceName_.empty() || sampleRate_ < 0.0 ||
        sampleRate_ > 1.0 || slowRequestMs_ <= 0)
        throw std::invalid_argument("invalid observability middleware configuration");
}

void ObservabilityMiddleware::before(HttpRequest& request)
{
    RequestContext context;
    const std::string suppliedRequestId = request.getHeader("X-Request-Id");
    context.requestId = suppliedRequestId.size() == 36 &&
        suppliedRequestId.rfind("req_", 0) == 0 &&
        isLowerHex(suppliedRequestId.substr(4), 32)
        ? suppliedRequestId : "req_" + randomHex(16);
    parseTraceParent(request.getHeader("traceparent"), context);
    if (context.traceId.empty()) context.traceId = randomHex(16);
    context.spanId = randomHex(8);
    context.routePattern = request.requestContext().routePattern;
    context.startedAt = std::chrono::steady_clock::now();
    if (sampleRate_ < 1.0)
    {
        const unsigned long sample = std::stoul(context.spanId.substr(0, 8), nullptr, 16);
        context.sampled = static_cast<double>(sample) / 4294967295.0 < sampleRate_;
    }
    request.setRequestContext(std::move(context));
    request.setHeader("X-Request-Id", request.requestContext().requestId);
    request.setHeader("X-Trace-Id", request.requestContext().traceId);
    metrics_->increment("treesem_http_requests_total",
        {{"method", methodName(request.method())},
         {"operation", request.requestContext().routePattern}});
    metrics_->addGauge("treesem_http_active_requests", {}, 1.0);
}

void ObservabilityMiddleware::after(HttpResponse&)
{}

void ObservabilityMiddleware::after(const HttpRequest& request,
                                    HttpResponse& response)
{
    const RequestContext& context = request.requestContext();
    if (!context.initialized()) return;
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - context.startedAt).count();
    const int status = static_cast<int>(response.getStatusCode());
    const std::string statusClass = std::to_string(status / 100) + "xx";
    const observability::Labels labels{
        {"method", methodName(request.method())},
        {"operation", context.routePattern},
        {"status", statusClass}};
    metrics_->increment("treesem_http_responses_total", labels);
    metrics_->observe("treesem_http_request_duration_seconds", labels, elapsed);
    metrics_->addGauge("treesem_http_active_requests", {}, -1.0);
    response.addHeader("X-Request-Id", context.requestId);
    response.addHeader("X-Trace-Id", context.traceId);
    if (context.sampled || elapsed * 1000.0 >= slowRequestMs_ || status >= 500)
    {
        LOG_INFO << "{\"event\":\"http_request\",\"service\":\""
                 << serviceName_ << "\",\"request_id\":\"" << context.requestId
                 << "\",\"trace_id\":\"" << context.traceId
                 << "\",\"span_id\":\"" << context.spanId
                 << "\",\"parent_span_id\":\"" << context.parentSpanId
                 << "\",\"operation\":\"" << context.routePattern
                 << "\",\"duration_ms\":" << elapsed * 1000.0
                 << ",\"outcome\":\"" << (status < 500 ? "success" : "error")
                 << "\",\"status\":" << status << '}';
    }
}

} // namespace http::middleware
