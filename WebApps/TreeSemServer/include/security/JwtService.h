#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "domain/SecurityTypes.h"

namespace treesem::security
{
struct JwtConfig
{
    std::string accessSecret;
    std::string capabilitySecret;
    std::chrono::seconds accessTtl{900};
    std::chrono::seconds capabilityTtl{120};
    std::string knowledgeSecret;
    std::chrono::seconds knowledgeTtl{120};
};

struct KnowledgeCapabilityContext
{
    std::string actorId;
    domain::UserRole actorRole;
    std::string sessionId;
    std::string subjectUserId;
    std::string runId;
    std::vector<std::string> allowedScopes;
};

struct CapabilityContext
{
    std::string actorId;
    domain::UserRole actorRole;
    std::string sessionId;
    std::string subjectUserId;
    std::string runId;
    std::vector<std::string> allowedTools;
};

class JwtService
{
public:
    explicit JwtService(JwtConfig config);
    std::string issueAccess(const domain::UserRecord& user,
                            domain::TimePoint now) const;
    domain::ActorContext verifyAccess(const std::string& token,
                                      domain::TimePoint now) const;
    std::string issueCapability(const CapabilityContext& context,
                                domain::TimePoint now) const;
    CapabilityContext verifyCapability(const std::string& token,
                                       domain::TimePoint now) const;
    std::string issueKnowledgeCapability(
        const KnowledgeCapabilityContext& context,
        domain::TimePoint now) const;
    KnowledgeCapabilityContext verifyKnowledgeCapability(
        const std::string& token, domain::TimePoint now) const;
private:
    JwtConfig config_;
};
} // namespace treesem::security
