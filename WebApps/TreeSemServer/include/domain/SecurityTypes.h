#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "domain/BusinessTypes.h"

namespace treesem::domain
{
enum class UserRole { Patient, Doctor, Admin };
std::string toString(UserRole role);
UserRole parseUserRole(const std::string& value);

struct UserRecord
{
    std::string userId;
    std::string emailNormalized;
    std::string passwordPhc;
    std::string displayName;
    UserRole role{UserRole::Patient};
    std::string status{"active"};
    int failedLoginCount{0};
    std::optional<TimePoint> lockedUntil;
    std::uint64_t tokenVersion{0};
    TimePoint createdAt;
    TimePoint updatedAt;
};

struct RefreshSession
{
    std::string refreshSessionId;
    std::string tokenFamilyId;
    std::string userId;
    std::string tokenSha256;
    TimePoint expiresAt;
    std::optional<TimePoint> consumedAt;
    std::optional<TimePoint> revokedAt;
    std::optional<std::string> replacedById;
    TimePoint createdAt;
    TimePoint lastUsedAt;
};

struct DoctorPatientAssignment
{
    std::string assignmentId;
    std::string doctorUserId;
    std::string patientUserId;
    std::string status{"active"};
    std::string createdByAdminId;
    TimePoint createdAt;
    std::optional<TimePoint> revokedAt;
};

struct AuditEvent
{
    std::string eventId;
    std::string requestId;
    std::optional<std::string> actorUserId;
    std::optional<UserRole> actorRole;
    std::string action;
    std::string resourceType;
    std::optional<std::string> resourceId;
    std::string outcome;
    std::string reasonCode;
    TimePoint createdAt;
};

struct AuditCursor { TimePoint createdAt; std::string eventId; };
struct AuditPage
{
    std::vector<AuditEvent> items;
    std::optional<AuditCursor> nextCursor;
};

struct ActorContext
{
    std::string userId;
    UserRole role{UserRole::Patient};
    std::uint64_t tokenVersion{0};
    std::optional<std::string> sessionId;
    std::optional<std::string> subjectUserId;
    std::string requestId;
};
} // namespace treesem::domain
