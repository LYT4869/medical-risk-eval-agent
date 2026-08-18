#pragma once

#include <optional>
#include <string>

#include "domain/SecurityTypes.h"

namespace treesem::persistence
{
enum class RefreshRotateResult { Rotated, NotFound, Expired, Reused };

class ISecurityStore
{
public:
    virtual ~ISecurityStore() = default;
    virtual void createUser(const domain::UserRecord& user) = 0;
    virtual std::optional<domain::UserRecord> findUserByEmail(
        const std::string& normalizedEmail) = 0;
    virtual std::optional<domain::UserRecord> findUserById(
        const std::string& userId) = 0;
    virtual bool hasAdminUser() = 0;
    virtual void recordLoginFailure(const std::string& userId,
                                    int failedCount,
                                    const std::optional<domain::TimePoint>& lockedUntil,
                                    domain::TimePoint now) = 0;
    virtual void recordLoginSuccess(const std::string& userId,
                                    domain::TimePoint now) = 0;
    virtual void createRefreshSession(const domain::RefreshSession& session) = 0;
    virtual RefreshRotateResult rotateRefreshSession(
        const std::string& oldTokenHash,
        const domain::RefreshSession& replacement,
        domain::TimePoint now,
        std::optional<domain::RefreshSession>& oldSession) = 0;
    virtual void revokeRefreshFamily(const std::string& familyId,
                                     domain::TimePoint now) = 0;
    virtual std::optional<domain::RefreshSession> findRefreshSession(
        const std::string& tokenHash) = 0;
    virtual domain::DoctorPatientAssignment saveAssignment(
        const domain::DoctorPatientAssignment& assignment) = 0;
    virtual bool revokeAssignment(const std::string& assignmentId,
                                  domain::TimePoint now) = 0;
    virtual bool hasActiveAssignment(const std::string& doctorUserId,
                                     const std::string& patientUserId) = 0;
    virtual std::vector<domain::DoctorPatientAssignment> listAssignments() = 0;
    virtual void appendAudit(const domain::AuditEvent& event) = 0;
    virtual domain::AuditPage listAudit(
        std::size_t limit,
        const std::optional<domain::AuditCursor>& cursor) = 0;
    virtual bool bindSession(const std::string& sessionId,
                             const std::string& ownerUserId,
                             const std::string& subjectUserId) = 0;
    virtual std::optional<std::string> sessionSubjectForActor(
        const std::string& sessionId,
        const std::string& actorUserId) = 0;
    virtual bool isAgentRunRunning(const std::string& runId,
                                   const std::string& sessionId) = 0;
};
} // namespace treesem::persistence
