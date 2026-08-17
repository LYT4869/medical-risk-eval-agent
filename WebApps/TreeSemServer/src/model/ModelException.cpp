#include "model/ModelException.h"

namespace treesem
{
namespace model
{

ModelException::ModelException(Kind kind, const std::string& message)
    : std::runtime_error(message)
    , kind_(kind)
{}

ModelException::Kind ModelException::kind() const noexcept
{
    return kind_;
}

} // namespace model
} // namespace treesem
