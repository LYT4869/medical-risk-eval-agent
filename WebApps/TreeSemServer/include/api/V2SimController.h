#pragma once

#include <utility>

#include "application/AuditService.h"
#include "application/AuthService.h"
#include "http/AsyncHttp.h"
#include "http/HttpRequest.h"
#include "infrastructure/persistence/V2SimArtifactStore.h"
#include "persistence/ISecurityStore.h"
#include "service/BlockingTaskScheduler.h"

namespace treesem::api
{
// Opt-in, authenticated SIM-only endpoint; not a production clinical Tool API.
class V2SimController
{
public:
    V2SimController(const application::AuthService& auth,
                    persistence::ISecurityStore& security,
                    const application::AuditService& audit,
                    infrastructure::V2SimArtifactStore& artifacts,
                    service::BlockingTaskScheduler& scheduler);

    void create(http::HttpRequest request, http::AsyncResponder responder) const;
    void confirm(http::HttpRequest request, http::AsyncResponder responder) const;
    void put(http::HttpRequest request, http::AsyncResponder responder) const;
    void get(http::HttpRequest request, http::AsyncResponder responder) const;

private:
    std::string authorizedDoctor(const http::HttpRequest& request) const;
    const application::AuthService& auth_;
    persistence::ISecurityStore& security_;
    const application::AuditService& audit_;
    infrastructure::V2SimArtifactStore& artifacts_;
    service::BlockingTaskScheduler& scheduler_;
};
} // namespace treesem::api
