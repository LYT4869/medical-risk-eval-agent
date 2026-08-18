#pragma once

#include "application/AgentApplicationService.h"
#include "application/SessionService.h"
#include "http/AsyncHttp.h"
#include "service/BlockingTaskScheduler.h"
#include "persistence/ISecurityStore.h"
#include "security/JwtService.h"
#include "application/AuditService.h"

namespace treesem::api
{
class ChatController
{
public:
    ChatController(const application::AgentApplicationService& service,
                   const application::SessionService& sessions,
                   service::BlockingTaskScheduler& agentScheduler,
                   service::BlockingTaskScheduler& databaseScheduler,
                   bool cookieSecure, long sessionTtlSeconds,
                   std::size_t maxMessageCharacters,
                   persistence::ISecurityStore* securityStore = nullptr,
                   const security::JwtService* jwt = nullptr,
                   bool authRequired = false,
                   const application::AuditService* audit = nullptr);
    void chat(http::HttpRequest request, http::AsyncResponder responder) const;
    void history(http::HttpRequest request, http::AsyncResponder responder) const;
private:
    const application::AgentApplicationService& service_;
    const application::SessionService& sessions_;
    service::BlockingTaskScheduler& agentScheduler_;
    service::BlockingTaskScheduler& databaseScheduler_;
    bool cookieSecure_;
    long sessionTtlSeconds_;
    std::size_t maxMessageCharacters_;
    persistence::ISecurityStore* securityStore_;
    const security::JwtService* jwt_;
    bool authRequired_;
    const application::AuditService* audit_;
};
} // namespace treesem::api
