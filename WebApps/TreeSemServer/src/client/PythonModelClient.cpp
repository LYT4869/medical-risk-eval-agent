#include "client/PythonModelClient.h"

#include <mutex>
#include <utility>

#include <curl/curl.h>

namespace treesem
{
namespace client
{
namespace
{

struct ResponseBuffer
{
    std::string body;
    std::size_t limit;
    bool overflow{false};
};

std::size_t writeResponse(char* data,
                          std::size_t elementSize,
                          std::size_t elementCount,
                          void* context)
{
    const std::size_t bytes = elementSize * elementCount;
    auto* response = static_cast<ResponseBuffer*>(context);
    if (bytes > response->limit - response->body.size())
    {
        response->overflow = true;
        return 0;
    }
    response->body.append(data, bytes);
    return bytes;
}

void ensureCurlInitialized()
{
    static std::once_flag initialized;
    static CURLcode result = CURLE_OK;
    std::call_once(initialized, []() { result = curl_global_init(CURL_GLOBAL_DEFAULT); });
    if (result != CURLE_OK)
    {
        throw ModelAdapterException(
            ModelAdapterException::Kind::Unavailable,
            "failed to initialize libcurl");
    }
}

} // namespace

ModelAdapterException::ModelAdapterException(Kind kind, const std::string& message)
    : std::runtime_error(message)
    , kind_(kind)
{}

ModelAdapterException::Kind ModelAdapterException::kind() const noexcept
{
    return kind_;
}

PythonModelClient::PythonModelClient(PythonModelClientConfig config)
    : config_(std::move(config))
{
    if (config_.predictUrl.empty())
    {
        throw std::invalid_argument("predictUrl must not be empty");
    }
    if (config_.connectTimeoutMs <= 0 || config_.requestTimeoutMs <= 0)
    {
        throw std::invalid_argument("model adapter timeouts must be positive");
    }
    if (config_.maxResponseBytes == 0)
    {
        throw std::invalid_argument("maxResponseBytes must be positive");
    }
    ensureCurlInitialized();
}

ModelAdapterResponse PythonModelClient::predict(const std::string& requestBody) const
{
    CURL* handle = curl_easy_init();
    if (handle == nullptr)
    {
        throw ModelAdapterException(
            ModelAdapterException::Kind::Unavailable,
            "failed to create a model adapter HTTP request");
    }

    ResponseBuffer response{{}, config_.maxResponseBytes};
    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");

    curl_easy_setopt(handle, CURLOPT_URL, config_.predictUrl.c_str());
    curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(handle, CURLOPT_POST, 1L);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, requestBody.data());
    curl_easy_setopt(
        handle, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(requestBody.size()));
    curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, config_.connectTimeoutMs);
    curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, config_.requestTimeoutMs);
    curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, writeResponse);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response);

    const CURLcode result = curl_easy_perform(handle);
    long statusCode = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &statusCode);
    curl_slist_free_all(headers);
    curl_easy_cleanup(handle);

    if (response.overflow)
    {
        throw ModelAdapterException(
            ModelAdapterException::Kind::InvalidResponse,
            "model adapter response exceeded the configured limit");
    }
    if (result == CURLE_OPERATION_TIMEDOUT)
    {
        throw ModelAdapterException(
            ModelAdapterException::Kind::Timeout,
            "model adapter request timed out");
    }
    if (result != CURLE_OK)
    {
        throw ModelAdapterException(
            ModelAdapterException::Kind::Unavailable,
            curl_easy_strerror(result));
    }
    if (statusCode <= 0 || response.body.empty())
    {
        throw ModelAdapterException(
            ModelAdapterException::Kind::InvalidResponse,
            "model adapter returned an empty HTTP response");
    }
    return ModelAdapterResponse{statusCode, std::move(response.body)};
}

} // namespace client
} // namespace treesem
