#pragma once

#include <string>

#include "application/AuthService.h"
#include "application/SessionService.h"
#include "http/AsyncHttp.h"
#include "service/BlockingTaskScheduler.h"

namespace treesem::api
{
class AuthController
{
public:
    AuthController(const application::AuthService& auth,
                   const application::SessionService& sessions,
                   persistence::ISecurityStore& securityStore,
                   service::BlockingTaskScheduler& scheduler,
                   bool refreshCookieSecure,
                   long refreshTtlSeconds,
                   std::string allowedOrigins);
    void registerPatient(http::HttpRequest request, http::AsyncResponder responder) const;
    void login(http::HttpRequest request, http::AsyncResponder responder) const;
    void refresh(http::HttpRequest request, http::AsyncResponder responder) const;
    void logout(http::HttpRequest request, http::AsyncResponder responder) const;
    void me(http::HttpRequest request, http::AsyncResponder responder) const;
    void createDoctor(http::HttpRequest request, http::AsyncResponder responder) const;
    void createAssignment(http::HttpRequest request, http::AsyncResponder responder) const;
    void revokeAssignment(http::HttpRequest request, http::AsyncResponder responder) const;
    void listAssignments(http::HttpRequest request, http::AsyncResponder responder) const;
    void listAudit(http::HttpRequest request, http::AsyncResponder responder) const;
private:
    domain::ActorContext actor(const http::HttpRequest& request) const;
    std::string requestId(const http::HttpRequest& request) const;
    void checkOrigin(const http::HttpRequest& request) const;
    const application::AuthService& auth_;
    const application::SessionService& sessions_;
    persistence::ISecurityStore& securityStore_;
    service::BlockingTaskScheduler& scheduler_;
    bool secure_;
    long refreshTtlSeconds_;
    std::string allowedOrigins_;
};
} // namespace treesem::api
