#pragma once

#include <stdexcept>
#include <string>

namespace treesem
{
namespace model
{

class ModelException : public std::runtime_error
{
public:
    enum class Kind
    {
        InvalidInput,
        Timeout,
        Unavailable,
        InvalidResponse,
        DownstreamFailure,
    };

    ModelException(Kind kind, const std::string& message);

    Kind kind() const noexcept;

private:
    Kind kind_;
};

} // namespace model
} // namespace treesem
