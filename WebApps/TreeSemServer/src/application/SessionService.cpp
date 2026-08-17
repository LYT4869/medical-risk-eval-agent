#include "application/SessionService.h"

#include <stdexcept>

#include "application/BusinessException.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem
{
namespace application
{

SessionService::SessionService(persistence::ITreeSemStore& store,
                               std::chrono::seconds ttl)
    : store_(store)
    , ttl_(ttl)
{
    if (ttl_.count() <= 0)
    {
        throw std::invalid_argument("session TTL must be positive");
    }
}

ResolvedSession SessionService::create() const
{
    for (int attempt = 0; attempt < 4; ++attempt)
    {
        const auto now = domain::TimePoint(std::chrono::microseconds(
            infrastructure::epochMicroseconds(std::chrono::system_clock::now())));
        domain::SessionRecord session;
        session.sessionId = infrastructure::generateOpaqueId("ses_");
        session.createdAt = now;
        session.lastAccessedAt = now;
        session.expiresAt = now + ttl_;
        try
        {
            store_.createSession(session);
            return {std::move(session), true};
        }
        catch (const BusinessException& error)
        {
            if (error.kind() != BusinessException::Kind::Conflict || attempt == 3)
            {
                throw;
            }
        }
    }
    throw BusinessException(
        BusinessException::Kind::PersistenceFailure,
        "unable to allocate a unique session id");
}

ResolvedSession SessionService::resolve(
    const std::optional<std::string>& suppliedSessionId,
    SessionAccess access) const
{
    if (!suppliedSessionId.has_value())
    {
        if (access == SessionAccess::Internal)
        {
            throw BusinessException(
                BusinessException::Kind::InvalidSession,
                "internal requests require a session id");
        }
        return create();
    }
    if (!infrastructure::isValidOpaqueId(*suppliedSessionId, "ses_"))
    {
        throw BusinessException(
            BusinessException::Kind::InvalidSession,
            "invalid session id");
    }
    const auto now = domain::TimePoint(std::chrono::microseconds(
        infrastructure::epochMicroseconds(std::chrono::system_clock::now())));
    auto session = store_.touchActiveSession(*suppliedSessionId, now, now + ttl_);
    if (session.has_value())
    {
        return {std::move(*session), false};
    }
    if (access == SessionAccess::Public)
    {
        return create();
    }
    throw BusinessException(
        BusinessException::Kind::NotFound,
        "session was not found");
}

std::chrono::seconds SessionService::ttl() const noexcept
{
    return ttl_;
}

} // namespace application
} // namespace treesem
