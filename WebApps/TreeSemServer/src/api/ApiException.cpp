#include "api/ApiException.h"

namespace treesem
{
namespace api
{

ApiException::ApiException(Kind kind, const std::string& message)
    : std::runtime_error(message)
    , kind_(kind)
{}

ApiException::Kind ApiException::kind() const noexcept
{
    return kind_;
}

} // namespace api
} // namespace treesem
