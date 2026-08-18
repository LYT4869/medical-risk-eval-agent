#include "application/AuditService.h"

#include <chrono>

#include "application/AuthService.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem::application
{
void AuditService::record(
    const std::string& actorId, const std::string& actorRole,
    const std::string& requestId, const std::string& action,
    const std::string& resourceType, const std::optional<std::string>& resourceId,
    const std::string& outcome, const std::string& reasonCode) const
{
    domain::AuditEvent event;
    event.eventId = infrastructure::generateOpaqueId("evt_");
    event.requestId = infrastructure::isValidOpaqueId(requestId, "req_")
        ? requestId : infrastructure::generateOpaqueId("req_");
    if (!actorId.empty()) event.actorUserId = actorId;
    if (!actorRole.empty()) event.actorRole = domain::parseUserRole(actorRole);
    event.action = action; event.resourceType = resourceType;
    event.resourceId = resourceId; event.outcome = outcome;
    event.reasonCode = reasonCode;
    event.createdAt = domain::TimePoint(std::chrono::microseconds(
        infrastructure::epochMicroseconds(std::chrono::system_clock::now())));
    try { store_.appendAudit(event); }
    catch (...) { throw AuthException(AuthException::Kind::AuditUnavailable,
                                      "audit write failed"); }
}
} // namespace treesem::application
