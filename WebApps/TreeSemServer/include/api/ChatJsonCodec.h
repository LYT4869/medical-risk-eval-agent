#pragma once

#include <optional>
#include <string>

#include "application/AgentApplicationService.h"
#include "http/HttpRequest.h"

namespace treesem::api
{
struct ParsedChatRequest { std::string message; std::string idempotencyKey; };

class ChatJsonCodec
{
public:
    static ParsedChatRequest parse(const http::HttpRequest& request,
                                   std::size_t maxCharacters);
    static std::string serialize(const application::ChatResult& result);
    static std::string serializeHistory(const domain::ChatPage& page);
    static std::optional<domain::ChatCursor> parseCursor(
        const http::HttpRequest& request);
};
} // namespace treesem::api
