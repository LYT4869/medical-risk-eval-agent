#pragma once

#include <stdexcept>
#include <string>

namespace treesem
{
namespace api
{

class ApiException : public std::runtime_error
{
public:
    enum class Kind
    {
        InvalidJson,
        InvalidRequest,
    };

    ApiException(Kind kind, const std::string& message);
    Kind kind() const noexcept;

private:
    Kind kind_;
};

} // namespace api
} // namespace treesem
