#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain/BusinessTypes.h"

namespace treesem::client
{

struct AgentRequest
{
    std::string runId;
    std::string sessionId;
    std::string message;
    std::vector<domain::ChatMessage> recentMessages;
    std::optional<std::string> currentPredictionId;
    std::optional<std::string> currentModelVersion;
    std::optional<std::string> capabilityToken;
    std::string actorRole{"patient"};
    std::optional<std::string> knowledgeCapabilityToken;
};

struct AgentResponse
{
    std::string answer;
    int stepCount{0};
    std::vector<domain::AgentToolSummary> tools;
    std::vector<std::string> groundingPredictionIds;
    std::vector<std::string> groundingSourceIds;
    std::vector<domain::KnowledgeCitation> citations;
    std::optional<std::string> knowledgeIndexVersion;
    std::optional<domain::AgentSkillUse> skillUsed;
};

class AgentClientException : public std::runtime_error
{
public:
    enum class Kind { Timeout, Unavailable, InvalidResponse, ExecutionFailed };
    AgentClientException(Kind kind, const std::string& message)
        : std::runtime_error(message), kind_(kind) {}
    Kind kind() const noexcept { return kind_; }
private:
    Kind kind_;
};

class IAgentClient
{
public:
    virtual ~IAgentClient() = default;
    virtual AgentResponse run(const AgentRequest& request) const = 0;
};

} // namespace treesem::client
