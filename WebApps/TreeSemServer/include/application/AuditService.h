#pragma once

#include <optional>
#include <string>

#include "domain/SecurityTypes.h"
#include "persistence/ISecurityStore.h"

namespace treesem::application
{
class AuditService
{
public:
    explicit AuditService(persistence::ISecurityStore& store) : store_(store) {}
    void record(const std::string& actorId, const std::string& actorRole,
                const std::string& requestId, const std::string& action,
                const std::string& resourceType,
                const std::optional<std::string>& resourceId,
                const std::string& outcome,
                const std::string& reasonCode) const;
private:
    persistence::ISecurityStore& store_;
};
} // namespace treesem::application
