#pragma once

#include <string>

namespace treesem::security
{
class PasswordHasher
{
public:
    std::string hash(const std::string& password) const;
    bool verify(const std::string& encoded, const std::string& password) const;
};
} // namespace treesem::security
