#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "config/TreeSemServerConfig.h"
#include "domain/SecurityTypes.h"
#include "infrastructure/persistence/MySqlTreeSemStore.h"
#include "infrastructure/support/ValueSupport.h"
#include "security/PasswordHasher.h"

namespace
{
std::string normalizeEmail(std::string value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    const auto end = value.find_last_not_of(" \t\r\n");
    if (begin == std::string::npos)
        throw std::invalid_argument("bootstrap admin email is invalid");
    value = value.substr(begin, end - begin + 1);
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    if (value.size() > 254 || value.find('@') == std::string::npos)
        throw std::invalid_argument("bootstrap admin email is invalid");
    return value;
}
}

int main(int argc, char* argv[])
{
    try
    {
        if (argc != 2 || std::string(argv[1]) != "bootstrap")
            throw std::invalid_argument("usage: treesem-admin bootstrap");
        const char* email = std::getenv("TREESEM_BOOTSTRAP_ADMIN_EMAIL");
        const char* password = std::getenv("TREESEM_BOOTSTRAP_ADMIN_PASSWORD");
        const char* display = std::getenv("TREESEM_BOOTSTRAP_ADMIN_DISPLAY_NAME");
        if (email == nullptr || password == nullptr || *email == '\0' || *password == '\0')
            throw std::invalid_argument(
                "TREESEM_BOOTSTRAP_ADMIN_EMAIL and TREESEM_BOOTSTRAP_ADMIN_PASSWORD are required");
        char program[] = "treesem-admin";
        char* configArguments[] = {program, nullptr};
        const auto config = treesem::config::TreeSemServerConfig::load(1, configArguments);
        if (config.storageBackend != treesem::config::StorageBackend::MySql)
            throw std::invalid_argument("admin bootstrap requires mysql storage");
        treesem::infrastructure::MySqlTreeSemStore store(
            treesem::infrastructure::MySqlStoreConfig::from(config));
        if (store.hasAdminUser())
            throw std::runtime_error("an administrator already exists");
        treesem::security::PasswordHasher hasher;
        const auto now = treesem::domain::TimePoint(std::chrono::microseconds(
            treesem::infrastructure::epochMicroseconds(
                std::chrono::system_clock::now())));
        treesem::domain::UserRecord admin;
        admin.userId = treesem::infrastructure::generateOpaqueId("usr_");
        admin.emailNormalized = normalizeEmail(email);
        admin.passwordPhc = hasher.hash(password);
        admin.displayName = display == nullptr || *display == '\0'
            ? "treeSem Administrator" : display;
        if (admin.displayName.size() > 100)
            throw std::invalid_argument("bootstrap admin display name is too long");
        admin.role = treesem::domain::UserRole::Admin;
        admin.createdAt = now; admin.updatedAt = now;
        store.createUser(admin);
        std::cout << "administrator created: " << admin.userId << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "admin bootstrap failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
