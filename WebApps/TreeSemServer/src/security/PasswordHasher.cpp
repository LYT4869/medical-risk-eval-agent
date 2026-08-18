#include "security/PasswordHasher.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <openssl/rand.h>

extern "C" {
int argon2id_hash_encoded(std::uint32_t, std::uint32_t, std::uint32_t,
                          const void*, std::size_t, const void*, std::size_t,
                          std::size_t, char*, std::size_t);
int argon2id_verify(const char*, const void*, std::size_t);
}

namespace treesem::security
{
std::string PasswordHasher::hash(const std::string& password) const
{
    if (password.size() < 12 || password.size() > 128)
        throw std::invalid_argument("password length must be 12..128");
    std::array<unsigned char, 32> salt{};
    if (RAND_bytes(salt.data(), static_cast<int>(salt.size())) != 1)
        throw std::runtime_error("password salt generation failed");
    std::array<char, 512> encoded{};
    const int result = argon2id_hash_encoded(
        3, 65536, 1, password.data(), password.size(), salt.data(), salt.size(),
        32, encoded.data(), encoded.size());
    if (result != 0) throw std::runtime_error("Argon2id hashing failed");
    return encoded.data();
}

bool PasswordHasher::verify(const std::string& encoded,
                            const std::string& password) const
{
    if (encoded.size() > 511 || password.size() > 128) return false;
    return argon2id_verify(encoded.c_str(), password.data(), password.size()) == 0;
}
} // namespace treesem::security
