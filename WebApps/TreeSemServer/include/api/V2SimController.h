#pragma once

#include <utility>

#include "application/AuditService.h"
#include "application/AuthService.h"
#include "http/AsyncHttp.h"
#include "http/HttpRequest.h"
#include "infrastructure/persistence/V2SimArtifactStore.h"
#include "persistence/ISecurityStore.h"
#include "service/BlockingTaskScheduler.h"
#include "security/JwtService.h"

namespace treesem::api
{
// Opt-in, authenticated SIM-only endpoint; not a production clinical Tool API.
class V2SimController
{
public:
    V2SimController(const application::AuthService& auth,
                    persistence::ISecurityStore& security,
                    const application::AuditService& audit,
                    const security::JwtService& jwt,
                    infrastructure::V2SimArtifactStore& artifacts,
                    service::BlockingTaskScheduler& scheduler);

    void create(http::HttpRequest request, http::AsyncResponder responder) const;
    void confirm(http::HttpRequest request, http::AsyncResponder responder) const;
    void editContract(http::HttpRequest request, http::AsyncResponder responder) const;
    void status(http::HttpRequest request, http::AsyncResponder responder) const;
    void capability(http::HttpRequest request, http::AsyncResponder responder) const;
    void internalStatus(http::HttpRequest request, http::AsyncResponder responder) const;
    void internalPut(http::HttpRequest request, http::AsyncResponder responder) const;
    void internalGet(http::HttpRequest request, http::AsyncResponder responder) const;
    // Doctor-only SIM fixture controls; Agent tool capabilities cannot
    // claim an independent record has become an official LIS correction.
    void stageHistoricalLab(http::HttpRequest request, http::AsyncResponder responder) const;
    void applyOfficialSimCorrection(http::HttpRequest request, http::AsyncResponder responder) const;
    void correctionState(http::HttpRequest request, http::AsyncResponder responder) const;
    void checkCurrentSources(http::HttpRequest request, http::AsyncResponder responder) const;
    void put(http::HttpRequest request, http::AsyncResponder responder) const;
    void get(http::HttpRequest request, http::AsyncResponder responder) const;

private:
    std::string authorizedDoctor(const http::HttpRequest& request) const;
    std::string authorizedInternal(const http::HttpRequest& request) const;
    const application::AuthService& auth_;
    persistence::ISecurityStore& security_;
    const application::AuditService& audit_;
    const security::JwtService& jwt_;
    infrastructure::V2SimArtifactStore& artifacts_;
    service::BlockingTaskScheduler& scheduler_;
};
} // namespace treesem::api
