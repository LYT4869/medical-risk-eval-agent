#include "api/ChatJsonCodec.h"

#include <algorithm>
#include <cctype>

#include <nlohmann/json.hpp>

#include "api/ApiException.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem::api
{
namespace
{
using Json = nlohmann::json;
[[noreturn]] void invalid() { throw ApiException(ApiException::Kind::InvalidRequest, "invalid chat request"); }
bool validKey(const std::string& value)
{
    return value.size() >= 8 && value.size() <= 64 &&
        std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '-' || c == '_' || c == '.';
        });
}
}

ParsedChatRequest ChatJsonCodec::parse(const http::HttpRequest& request,
                                       std::size_t maxCharacters)
{
    try
    {
        const Json root = Json::parse(request.getBody());
        if (!root.is_object() || root.size() != 1 || !root.contains("message") ||
            !root.at("message").is_string()) invalid();
        ParsedChatRequest result{root.at("message").get<std::string>(),
                                 request.getHeader("Idempotency-Key")};
        if (result.message.find_first_not_of(" \t\r\n") == std::string::npos ||
            result.message.size() > maxCharacters || !validKey(result.idempotencyKey)) invalid();
        return result;
    }
    catch (const ApiException&) { throw; }
    catch (const Json::parse_error&) { throw ApiException(ApiException::Kind::InvalidJson, "invalid JSON"); }
    catch (const Json::exception&) { invalid(); }
}

std::string ChatJsonCodec::serialize(const application::ChatResult& result)
{
    Json tools = Json::array();
    for (const auto& tool : result.run.tools)
        tools.push_back({{"name", tool.name}, {"status", tool.status},
                         {"duration_ms", tool.durationMs}});
    return Json{{"run_id", result.run.runId},
                {"message_id", result.finalMessage.messageId},
                {"session_id", result.run.sessionId},
                {"status", domain::toString(result.run.status)},
                {"answer", result.finalMessage.content},
                {"step_count", result.run.stepCount},
                {"tools_used", std::move(tools)},
                {"grounding_prediction_ids", result.run.groundingPredictionIds},
                {"created_at", infrastructure::formatUtc(result.finalMessage.createdAt)}}.dump();
}

std::string ChatJsonCodec::serializeHistory(const domain::ChatPage& page)
{
    Json items = Json::array();
    for (const auto& message : page.items)
        items.push_back({{"message_id", message.messageId}, {"run_id", message.runId},
                         {"role", message.role}, {"content", message.content},
                         {"created_at", infrastructure::formatUtc(message.createdAt)}});
    Json cursor = nullptr;
    if (page.nextCursor.has_value())
        cursor = infrastructure::encodeHistoryCursor(
            {page.nextCursor->createdAt, page.nextCursor->messageId});
    return Json{{"session_id", page.sessionId}, {"items", std::move(items)},
                {"next_cursor", std::move(cursor)}}.dump();
}

std::optional<domain::ChatCursor> ChatJsonCodec::parseCursor(
    const http::HttpRequest& request)
{
    const std::string raw = request.getQueryParameters("cursor");
    if (raw.empty()) return std::nullopt;
    try
    {
        const auto cursor = infrastructure::decodeHistoryCursor(raw);
        return domain::ChatCursor{cursor.createdAt, cursor.predictionId};
    }
    catch (...) { invalid(); }
}
} // namespace treesem::api
