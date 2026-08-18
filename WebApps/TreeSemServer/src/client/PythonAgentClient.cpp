#include "client/PythonAgentClient.h"

#include <mutex>
#include <utility>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace treesem::client
{
namespace
{
struct Buffer { std::string body; std::size_t limit; bool overflow{false}; };

std::size_t writeBody(char* data, std::size_t size, std::size_t count, void* opaque)
{
    auto* buffer = static_cast<Buffer*>(opaque);
    const std::size_t bytes = size * count;
    if (bytes > buffer->limit - buffer->body.size())
    {
        buffer->overflow = true;
        return 0;
    }
    buffer->body.append(data, bytes);
    return bytes;
}

void ensureCurl()
{
    static std::once_flag once;
    static CURLcode result = CURLE_OK;
    std::call_once(once, [] { result = curl_global_init(CURL_GLOBAL_DEFAULT); });
    if (result != CURLE_OK)
        throw AgentClientException(AgentClientException::Kind::Unavailable,
                                   "failed to initialize HTTP client");
}

nlohmann::json encodeRequest(const AgentRequest& request)
{
    nlohmann::json recent = nlohmann::json::array();
    for (const auto& message : request.recentMessages)
        recent.push_back({{"role", message.role}, {"content", message.content}});
    nlohmann::json root{
        {"run_id", request.runId}, {"session_id", request.sessionId},
        {"message", request.message}, {"recent_messages", std::move(recent)}};
    if (request.currentPredictionId.has_value())
        root["current_prediction"] = {
            {"prediction_id", *request.currentPredictionId},
            {"model_version", request.currentModelVersion.has_value()
                ? nlohmann::json(*request.currentModelVersion) : nlohmann::json(nullptr)}};
    else root["current_prediction"] = nullptr;
    root["capability_token"] = request.capabilityToken.has_value()
        ? nlohmann::json(*request.capabilityToken) : nlohmann::json(nullptr);
    return root;
}

AgentResponse decodeResponse(const std::string& body)
{
    try
    {
        const auto root = nlohmann::json::parse(body);
        if (!root.is_object() || !root.at("answer").is_string() ||
            !root.at("step_count").is_number_integer() ||
            !root.at("tools_used").is_array() ||
            !root.at("grounding_prediction_ids").is_array())
            throw std::runtime_error("invalid response shape");
        AgentResponse result;
        result.answer = root.at("answer").get<std::string>();
        result.stepCount = root.at("step_count").get<int>();
        if (result.answer.empty() || result.answer.size() > 16000 ||
            result.stepCount <= 0 || result.stepCount > 5)
            throw std::runtime_error("invalid response value");
        for (const auto& item : root.at("tools_used"))
        {
            if (!item.is_object() || !item.at("name").is_string() ||
                !item.at("status").is_string() ||
                !item.at("duration_ms").is_number_unsigned())
                throw std::runtime_error("invalid tool summary");
            result.tools.push_back({item.at("name").get<std::string>(),
                                    item.at("status").get<std::string>(),
                                    item.at("duration_ms").get<std::uint64_t>()});
        }
        for (const auto& item : root.at("grounding_prediction_ids"))
        {
            if (!item.is_string()) throw std::runtime_error("invalid grounding id");
            result.groundingPredictionIds.push_back(item.get<std::string>());
        }
        return result;
    }
    catch (const std::exception& error)
    {
        throw AgentClientException(AgentClientException::Kind::InvalidResponse,
                                   "agent returned an invalid response");
    }
}
} // namespace

PythonAgentClient::PythonAgentClient(PythonAgentClientConfig config)
    : config_(std::move(config))
{
    if (config_.url.empty() || config_.connectTimeoutMs <= 0 ||
        config_.requestTimeoutMs <= 0 || config_.maxResponseBytes == 0)
        throw std::invalid_argument("invalid Python agent client configuration");
    ensureCurl();
}

AgentResponse PythonAgentClient::run(const AgentRequest& request) const
{
    CURL* handle = curl_easy_init();
    if (!handle)
        throw AgentClientException(AgentClientException::Kind::Unavailable,
                                   "failed to create HTTP request");
    const std::string payload = encodeRequest(request).dump();
    Buffer response{{}, config_.maxResponseBytes};
    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");
    const std::string secretHeader = "X-TreeSem-Agent-Token: " + config_.serviceSecret;
    if (!config_.serviceSecret.empty())
        headers = curl_slist_append(headers, secretHeader.c_str());
    curl_easy_setopt(handle, CURLOPT_URL, config_.url.c_str());
    curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(handle, CURLOPT_POST, 1L);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, payload.data());
    curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE_LARGE,
                     static_cast<curl_off_t>(payload.size()));
    curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, config_.connectTimeoutMs);
    curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, config_.requestTimeoutMs);
    curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, writeBody);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response);
    const CURLcode result = curl_easy_perform(handle);
    long status = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(handle);
    if (response.overflow)
        throw AgentClientException(AgentClientException::Kind::InvalidResponse,
                                   "agent response exceeded limit");
    if (result == CURLE_OPERATION_TIMEDOUT)
        throw AgentClientException(AgentClientException::Kind::Timeout,
                                   "agent request timed out");
    if (result != CURLE_OK)
        throw AgentClientException(AgentClientException::Kind::Unavailable,
                                   "agent service unavailable");
    if (status == 504)
        throw AgentClientException(AgentClientException::Kind::Timeout,
                                   "agent execution timed out");
    if (status >= 500)
        throw AgentClientException(AgentClientException::Kind::ExecutionFailed,
                                   "agent execution failed");
    if (status != 200 || response.body.empty())
        throw AgentClientException(AgentClientException::Kind::InvalidResponse,
                                   "agent returned unexpected status");
    return decodeResponse(response.body);
}

} // namespace treesem::client
