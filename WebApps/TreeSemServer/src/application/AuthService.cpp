#include "application/AuthService.h"

#include <algorithm>
#include <cctype>

#include "infrastructure/support/ValueSupport.h"

namespace treesem::application
{
namespace
{
domain::TimePoint nowUtc()
{
    return domain::TimePoint(std::chrono::microseconds(
        infrastructure::epochMicroseconds(std::chrono::system_clock::now())));
}
std::string normalizedEmail(std::string value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    const auto end = value.find_last_not_of(" \t\r\n");
    if (begin == std::string::npos)
        throw AuthException(AuthException::Kind::InvalidCredentials, "invalid email");
    value = value.substr(begin, end - begin + 1);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value.size() > 254 || value.find('@') == std::string::npos)
        throw AuthException(AuthException::Kind::InvalidCredentials, "invalid email");
    return value;
}
}

AuthService::AuthService(
    persistence::ISecurityStore& store, const security::PasswordHasher& passwords,
    const security::JwtService& jwt, std::chrono::seconds accessTtl,
    std::chrono::seconds refreshTtl)
    : store_(store), passwords_(passwords), jwt_(jwt), accessTtl_(accessTtl),
      refreshTtl_(refreshTtl),
      dummyPasswordHash_(passwords.hash("treeSem-dummy-password-never-used")) {}

bool AuthService::unknownLoginRateLimited(
    const std::string& emailHash, domain::TimePoint now) const
{
    std::lock_guard<std::mutex> lock(unknownLoginMutex_);
    constexpr std::size_t kMaximumEntries = 4096;
    constexpr int kMaximumAttempts = 5;
    const auto window = std::chrono::minutes(15);
    auto found = unknownLogins_.find(emailHash);
    if (found == unknownLogins_.end())
    {
        if (unknownLogins_.size() >= kMaximumEntries)
        {
            const auto oldest = std::min_element(
                unknownLogins_.begin(), unknownLogins_.end(),
                [](const auto& left, const auto& right) {
                    return left.second.lastSeen < right.second.lastSeen;
                });
            if (oldest != unknownLogins_.end()) unknownLogins_.erase(oldest);
        }
        found = unknownLogins_.emplace(emailHash,
            UnknownLoginWindow{0, now, now}).first;
    }
    auto& state = found->second;
    if (now - state.windowStarted >= window)
    {
        state.attempts = 0;
        state.windowStarted = now;
    }
    ++state.attempts;
    state.lastSeen = now;
    return state.attempts >= kMaximumAttempts;
}

void AuthService::audit(
    const std::optional<domain::ActorContext>& actor,
    const std::string& requestId, const std::string& action,
    const std::string& resourceType, const std::optional<std::string>& resourceId,
    const std::string& outcome, const std::string& reason) const
{
    domain::AuditEvent event;
    event.eventId = infrastructure::generateOpaqueId("evt_");
    event.requestId = requestId;
    if (actor.has_value()) { event.actorUserId = actor->userId; event.actorRole = actor->role; }
    event.action = action; event.resourceType = resourceType; event.resourceId = resourceId;
    event.outcome = outcome; event.reasonCode = reason; event.createdAt = nowUtc();
    try { store_.appendAudit(event); }
    catch (...) { throw AuthException(AuthException::Kind::AuditUnavailable, "audit unavailable"); }
}

AuthResult AuthService::issue(
    domain::UserRecord user, const std::optional<std::string>& familyId,
    const std::optional<std::string>& sessionId) const
{
    const auto now = nowUtc();
    const std::string raw = infrastructure::secureRandomToken(32);
    domain::RefreshSession refresh;
    refresh.refreshSessionId = infrastructure::generateOpaqueId("rfs_");
    refresh.tokenFamilyId = familyId.value_or(infrastructure::generateOpaqueId("fam_"));
    refresh.userId = user.userId; refresh.tokenSha256 = infrastructure::sha256Hex(raw);
    refresh.createdAt = now; refresh.lastUsedAt = now; refresh.expiresAt = now + refreshTtl_;
    store_.createRefreshSession(refresh);
    if (sessionId.has_value()) (void)store_.bindSession(*sessionId, user.userId, user.userId);
    return {user, jwt_.issueAccess(user, now), raw, accessTtl_.count()};
}

AuthResult AuthService::registerPatient(
    const std::string& email, const std::string& password,
    const std::string& displayName, const std::optional<std::string>& sessionId,
    const std::string& requestId) const
{
    if (displayName.empty() || displayName.size() > 100)
        throw AuthException(AuthException::Kind::InvalidCredentials, "invalid display name");
    const auto now = nowUtc();
    domain::UserRecord user;
    user.userId = infrastructure::generateOpaqueId("usr_");
    user.emailNormalized = normalizedEmail(email); user.passwordPhc = passwords_.hash(password);
    user.displayName = displayName; user.createdAt = now; user.updatedAt = now;
    try { store_.createUser(user); }
    catch (...)
    {
        audit(std::nullopt, requestId, "auth.register", "account",
              infrastructure::sha256Hex(user.emailNormalized), "denied", "account_conflict");
        throw AuthException(AuthException::Kind::Conflict, "account conflict");
    }
    auto result = issue(user, std::nullopt, sessionId);
    domain::ActorContext actor{user.userId, user.role, user.tokenVersion,
        sessionId, user.userId, requestId};
    audit(actor, requestId, "auth.register", "account", user.userId,
          "allowed", "registered");
    return result;
}

AuthResult AuthService::login(
    const std::string& email, const std::string& password,
    const std::optional<std::string>& sessionId, const std::string& requestId) const
{
    const std::string normalized = normalizedEmail(email);
    auto user = store_.findUserByEmail(normalized);
    const auto now = nowUtc();
    const bool valid = user.has_value()
        ? passwords_.verify(user->passwordPhc, password)
        : passwords_.verify(dummyPasswordHash_, password);
    const std::string emailHash = infrastructure::sha256Hex(normalized);
    if (!user.has_value() && unknownLoginRateLimited(emailHash, now))
    {
        audit(std::nullopt, requestId, "auth.login", "account",
              emailHash, "denied", "rate_limited");
        throw AuthException(AuthException::Kind::RateLimited,
                            "authentication temporarily rate limited");
    }
    if (user.has_value() && user->lockedUntil.has_value() &&
        *user->lockedUntil > now)
    {
        audit(std::nullopt, requestId, "auth.login", "account",
              emailHash, "denied", "rate_limited");
        throw AuthException(AuthException::Kind::RateLimited,
                            "account temporarily locked");
    }
    if (!user.has_value() || !valid || user->status != "active")
    {
        if (user.has_value())
        {
            const int failures = user->failedLoginCount + 1;
            const auto locked = failures >= 5
                ? std::optional<domain::TimePoint>(now + std::chrono::minutes(15))
                : std::nullopt;
            store_.recordLoginFailure(user->userId, failures, locked, now);
        }
        audit(std::nullopt, requestId, "auth.login", "account",
              emailHash, "denied", "invalid_credentials");
        throw AuthException(AuthException::Kind::InvalidCredentials, "invalid credentials");
    }
    store_.recordLoginSuccess(user->userId, now);
    user->failedLoginCount = 0; user->lockedUntil.reset();
    auto result = issue(*user, std::nullopt, sessionId);
    domain::ActorContext actor{user->userId, user->role, user->tokenVersion,
        sessionId, user->userId, requestId};
    audit(actor, requestId, "auth.login", "account", user->userId,
          "allowed", "authenticated");
    return result;
}

AuthResult AuthService::refresh(const std::string& raw,
                                const std::string& requestId) const
{
    if (raw.empty() || raw.size() > 256)
        throw AuthException(AuthException::Kind::InvalidRefreshToken, "invalid refresh token");
    const auto now = nowUtc();
    const auto old = store_.findRefreshSession(infrastructure::sha256Hex(raw));
    if (!old.has_value())
        throw AuthException(AuthException::Kind::InvalidRefreshToken, "invalid refresh token");
    auto user = store_.findUserById(old->userId);
    if (!user.has_value() || user->status != "active")
        throw AuthException(AuthException::Kind::InvalidRefreshToken, "invalid refresh token");
    const std::string replacementRaw = infrastructure::secureRandomToken(32);
    domain::RefreshSession replacement;
    replacement.refreshSessionId = infrastructure::generateOpaqueId("rfs_");
    replacement.tokenFamilyId = old->tokenFamilyId; replacement.userId = old->userId;
    replacement.tokenSha256 = infrastructure::sha256Hex(replacementRaw);
    replacement.createdAt = now; replacement.lastUsedAt = now;
    replacement.expiresAt = now + refreshTtl_;
    std::optional<domain::RefreshSession> lockedOld;
    const auto rotated = store_.rotateRefreshSession(
        infrastructure::sha256Hex(raw), replacement, now, lockedOld);
    if (rotated != persistence::RefreshRotateResult::Rotated)
    {
        if (lockedOld.has_value()) store_.revokeRefreshFamily(lockedOld->tokenFamilyId, now);
        audit(std::nullopt, requestId, "auth.refresh", "refresh_family",
              old->tokenFamilyId, "denied", "refresh_reuse_or_expired");
        throw AuthException(AuthException::Kind::InvalidRefreshToken, "invalid refresh token");
    }
    domain::ActorContext actor{user->userId, user->role, user->tokenVersion,
        std::nullopt, user->userId, requestId};
    audit(actor, requestId, "auth.refresh", "refresh_family", old->tokenFamilyId,
          "allowed", "rotated");
    return {*user, jwt_.issueAccess(*user, now), replacementRaw, accessTtl_.count()};
}

void AuthService::logout(const std::string& raw,
                         const std::string& requestId) const
{
    const auto found = store_.findRefreshSession(infrastructure::sha256Hex(raw));
    if (found.has_value()) store_.revokeRefreshFamily(found->tokenFamilyId, nowUtc());
    audit(std::nullopt, requestId, "auth.logout", "refresh_family",
          found.has_value() ? std::optional<std::string>(found->tokenFamilyId) : std::nullopt,
          "allowed", "revoked");
}

domain::ActorContext AuthService::authenticate(
    const std::string& token, const std::string& requestId) const
{
    domain::ActorContext actor;
    try { actor = jwt_.verifyAccess(token, nowUtc()); }
    catch (...) { throw AuthException(AuthException::Kind::InvalidToken, "invalid access token"); }
    const auto user = store_.findUserById(actor.userId);
    if (!user.has_value() || user->status != "active" || user->tokenVersion != actor.tokenVersion)
        throw AuthException(AuthException::Kind::InvalidToken, "invalid access token");
    actor.requestId = requestId; return actor;
}

domain::UserRecord AuthService::createUser(
    const domain::ActorContext& admin, const std::string& email,
    const std::string& password, const std::string& displayName,
    domain::UserRole role) const
{
    if (admin.role != domain::UserRole::Admin || role == domain::UserRole::Admin)
        throw AuthException(AuthException::Kind::Forbidden, "forbidden");
    if (displayName.empty() || displayName.size() > 100)
        throw AuthException(AuthException::Kind::InvalidCredentials,
                            "invalid display name");
    const auto now = nowUtc();
    domain::UserRecord user{infrastructure::generateOpaqueId("usr_"),
        normalizedEmail(email), passwords_.hash(password), displayName, role,
        "active", 0, std::nullopt, 0, now, now};
    store_.createUser(user);
    audit(admin, admin.requestId, "admin.create_user", "account", user.userId,
          "allowed", domain::toString(role));
    return user;
}

domain::DoctorPatientAssignment AuthService::assign(
    const domain::ActorContext& admin, const std::string& doctorId,
    const std::string& patientId) const
{
    if (admin.role != domain::UserRole::Admin)
        throw AuthException(AuthException::Kind::Forbidden, "forbidden");
    const auto doctor = store_.findUserById(doctorId);
    const auto patient = store_.findUserById(patientId);
    if (!doctor.has_value() || doctor->role != domain::UserRole::Doctor ||
        !patient.has_value() || patient->role != domain::UserRole::Patient)
        throw AuthException(AuthException::Kind::NotFound, "account not found");
    domain::DoctorPatientAssignment assignment;
    assignment.assignmentId = infrastructure::generateOpaqueId("asn_");
    assignment.doctorUserId = doctorId; assignment.patientUserId = patientId;
    assignment.createdByAdminId = admin.userId; assignment.createdAt = nowUtc();
    assignment = store_.saveAssignment(assignment);
    audit(admin, admin.requestId, "assignment.create", "assignment",
          assignment.assignmentId, "allowed", "assigned");
    return assignment;
}

void AuthService::revokeAssignment(
    const domain::ActorContext& admin, const std::string& id) const
{
    if (admin.role != domain::UserRole::Admin)
        throw AuthException(AuthException::Kind::Forbidden, "forbidden");
    if (!store_.revokeAssignment(id, nowUtc()))
        throw AuthException(AuthException::Kind::NotFound, "assignment not found");
    audit(admin, admin.requestId, "assignment.revoke", "assignment", id,
          "allowed", "revoked");
}

void AuthService::authorizeSubject(
    const domain::ActorContext& actor, const std::string& subject) const
{
    const auto actorUser = store_.findUserById(actor.userId);
    const auto subjectUser = store_.findUserById(subject);
    if (!actorUser.has_value() || actorUser->status != "active" ||
        actorUser->role != actor.role || !subjectUser.has_value() ||
        subjectUser->status != "active" ||
        subjectUser->role != domain::UserRole::Patient)
        throw AuthException(AuthException::Kind::NotFound, "resource not found");
    if (actor.role == domain::UserRole::Patient && actor.userId == subject) return;
    if (actor.role == domain::UserRole::Doctor &&
        store_.hasActiveAssignment(actor.userId, subject)) return;
    if (actor.role == domain::UserRole::Admin)
        throw AuthException(AuthException::Kind::Forbidden, "admin has no clinical access");
    throw AuthException(AuthException::Kind::NotFound, "resource not found");
}
} // namespace treesem::application
