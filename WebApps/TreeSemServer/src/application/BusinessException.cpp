#include "application/BusinessException.h"

namespace treesem
{
namespace application
{

BusinessException::BusinessException(Kind kind, const std::string& message)
    : std::runtime_error(message)
    , kind_(kind)
{}

BusinessException::Kind BusinessException::kind() const noexcept
{
    return kind_;
}

} // namespace application
} // namespace treesem
