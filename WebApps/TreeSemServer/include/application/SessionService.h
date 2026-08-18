#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "domain/BusinessTypes.h"
#include "persistence/ITreeSemStore.h"

namespace treesem
{
namespace application
{

enum class SessionAccess
{
    Public,
    Internal,
};

struct ResolvedSession
{
    domain::SessionRecord session;
    bool created{false};
};

class SessionService
{
public:
    SessionService(persistence::ITreeSemStore& store,
                   std::chrono::seconds ttl);

    ResolvedSession resolve(
        const std::optional<std::string>& suppliedSessionId,
        SessionAccess access) const;
    std::chrono::seconds ttl() const noexcept;
    ResolvedSession createNew() const;

private:
    ResolvedSession create() const;

    persistence::ITreeSemStore& store_;
    std::chrono::seconds ttl_;
};

} // namespace application
} // namespace treesem
