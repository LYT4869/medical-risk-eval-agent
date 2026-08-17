#pragma once

#include <stdexcept>
#include <string>

namespace treesem
{
namespace application
{

class BusinessException : public std::runtime_error
{
public:
    enum class Kind
    {
        InvalidInput,
        InvalidSession,
        NotFound,
        Conflict,
        IdempotencyConflict,
        DatabaseBusy,
        DatabaseUnavailable,
        PersistenceFailure,
    };

    BusinessException(Kind kind, const std::string& message);
    Kind kind() const noexcept;

private:
    Kind kind_;
};

} // namespace application
} // namespace treesem
