#pragma once

#include <optional>
#include <string>

#include "client/IAgentClient.h"
#include "application/SessionService.h"
#include "persistence/ITreeSemStore.h"
#include "security/JwtService.h"

namespace treesem::application
{

struct ChatResult
{
    domain::AgentRunRecord run;
    domain::ChatMessage finalMessage;
    ResolvedSession session;
    bool replayed{false};
};

class AgentApplicationService
{
public:
    AgentApplicationService(const client::IAgentClient& agent,
                            persistence::ITreeSemStore& store,
                            const SessionService& sessions,
                            std::size_t contextMessages,
                            const security::JwtService* jwt = nullptr);
    ChatResult chat(const std::string& message,
                    const std::string& idempotencyKey,
                    const std::optional<std::string>& suppliedSession,
                    SessionAccess access,
                    const std::optional<domain::ActorContext>& actor = std::nullopt) const;
    domain::ChatPage history(const std::string& sessionId,
                             std::size_t limit,
                             const std::optional<domain::ChatCursor>& cursor) const;
private:
    const client::IAgentClient& agent_;
    persistence::ITreeSemStore& store_;
    const SessionService& sessions_;
    std::size_t contextMessages_;
    const security::JwtService* jwt_;
};

} // namespace treesem::application
