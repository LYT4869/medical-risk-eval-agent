#include "domain/SecurityTypes.h"

#include <stdexcept>

namespace treesem::domain
{
std::string toString(UserRole role)
{
    switch (role)
    {
    case UserRole::Patient: return "patient";
    case UserRole::Doctor: return "doctor";
    case UserRole::Admin: return "admin";
    }
    throw std::invalid_argument("unknown user role");
}

UserRole parseUserRole(const std::string& value)
{
    if (value == "patient") return UserRole::Patient;
    if (value == "doctor") return UserRole::Doctor;
    if (value == "admin") return UserRole::Admin;
    throw std::invalid_argument("invalid user role");
}
} // namespace treesem::domain
