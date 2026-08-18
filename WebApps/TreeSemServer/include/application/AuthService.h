#pragma once

#include <chrono>
#include <optional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "persistence/ISecurityStore.h"
#include "security/JwtService.h"
#include "security/PasswordHasher.h"

namespace treesem::application
{
class AuthException : public std::runtime_error
{
public:
    enum class Kind { InvalidCredentials, InvalidToken, InvalidRefreshToken,
                      RateLimited, Conflict, Forbidden, NotFound,
                      AuditUnavailable };
    AuthException(Kind kind, const std::string& message)
        : std::runtime_error(message), kind_(kind) {}
    Kind kind() const noexcept { return kind_; }
private:
    Kind kind_;
};

struct AuthResult
{
    domain::UserRecord user;
    std::string accessToken;
    std::string refreshToken;
    long accessExpiresIn{900};
};

class AuthService
{
public:
    AuthService(persistence::ISecurityStore& store,
                const security::PasswordHasher& passwords,
                const security::JwtService& jwt,
                std::chrono::seconds accessTtl,
                std::chrono::seconds refreshTtl);
    AuthResult registerPatient(const std::string& email,
                               const std::string& password,
                               const std::string& displayName,
                               const std::optional<std::string>& sessionId,
                               const std::string& requestId) const;
    AuthResult login(const std::string& email, const std::string& password,
                     const std::optional<std::string>& sessionId,
                     const std::string& requestId) const;
    AuthResult refresh(const std::string& refreshToken,
                       const std::string& requestId) const;
    void logout(const std::string& refreshToken,
                const std::string& requestId) const;
    domain::ActorContext authenticate(const std::string& accessToken,
                                      const std::string& requestId) const;
    domain::UserRecord createUser(const domain::ActorContext& admin,
                                  const std::string& email,
                                  const std::string& password,
                                  const std::string& displayName,
                                  domain::UserRole role) const;
    domain::DoctorPatientAssignment assign(
        const domain::ActorContext& admin,
        const std::string& doctorId,
        const std::string& patientId) const;
    void revokeAssignment(const domain::ActorContext& admin,
                          const std::string& assignmentId) const;
    void authorizeSubject(const domain::ActorContext& actor,
                          const std::string& subjectUserId) const;
private:
    AuthResult issue(domain::UserRecord user,
                     const std::optional<std::string>& familyId,
                     const std::optional<std::string>& sessionId) const;
    void audit(const std::optional<domain::ActorContext>& actor,
               const std::string& requestId, const std::string& action,
               const std::string& resourceType,
               const std::optional<std::string>& resourceId,
               const std::string& outcome,
               const std::string& reason) const;
    bool unknownLoginRateLimited(const std::string& emailHash,
                                 domain::TimePoint now) const;
    struct UnknownLoginWindow
    {
        int attempts{0};
        domain::TimePoint windowStarted;
        domain::TimePoint lastSeen;
    };
    persistence::ISecurityStore& store_;
    const security::PasswordHasher& passwords_;
    const security::JwtService& jwt_;
    std::chrono::seconds accessTtl_;
    std::chrono::seconds refreshTtl_;
    std::string dummyPasswordHash_;
    mutable std::mutex unknownLoginMutex_;
    mutable std::unordered_map<std::string, UnknownLoginWindow> unknownLogins_;
};
} // namespace treesem::application
