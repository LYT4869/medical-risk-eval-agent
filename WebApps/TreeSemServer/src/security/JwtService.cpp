#include "security/JwtService.h"

#include <array>
#include <stdexcept>

#include <openssl/hmac.h>
#include <nlohmann/json.hpp>

#include "infrastructure/support/ValueSupport.h"

namespace treesem::security
{
namespace
{
using Json = nlohmann::json;
std::string sign(const std::string& input, const std::string& secret)
{
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    if (!HMAC(EVP_sha256(), secret.data(), static_cast<int>(secret.size()),
              reinterpret_cast<const unsigned char*>(input.data()), input.size(),
              digest.data(), &size))
        throw std::runtime_error("JWT signing failed");
    return infrastructure::base64UrlEncode(
        std::string(reinterpret_cast<char*>(digest.data()), size));
}

std::int64_t epochSeconds(domain::TimePoint time)
{
    return std::chrono::duration_cast<std::chrono::seconds>(
        time.time_since_epoch()).count();
}

std::string issue(Json claims, const std::string& secret)
{
    const std::string header = infrastructure::base64UrlEncode(
        Json{{"alg", "HS256"}, {"typ", "JWT"}}.dump());
    const std::string payload = infrastructure::base64UrlEncode(claims.dump());
    const std::string input = header + "." + payload;
    return input + "." + sign(input, secret);
}

Json verify(const std::string& token, const std::string& secret,
            const std::string& audience, domain::TimePoint now)
{
    if (token.size() > 8192) throw std::invalid_argument("token too long");
    const auto first = token.find('.');
    const auto second = first == std::string::npos ? first : token.find('.', first + 1);
    if (first == std::string::npos || second == std::string::npos ||
        token.find('.', second + 1) != std::string::npos)
        throw std::invalid_argument("invalid JWT");
    const std::string input = token.substr(0, second);
    if (!infrastructure::constantTimeEquals(token.substr(second + 1), sign(input, secret)))
        throw std::invalid_argument("invalid JWT signature");
    const Json header = Json::parse(infrastructure::base64UrlDecode(token.substr(0, first), 1024));
    if (!header.is_object() || header.value("alg", "") != "HS256" ||
        header.value("typ", "") != "JWT") throw std::invalid_argument("invalid JWT header");
    const Json claims = Json::parse(infrastructure::base64UrlDecode(
        token.substr(first + 1, second - first - 1), 8192));
    if (claims.value("iss", "") != "treesem-backend" ||
        claims.value("aud", "") != audience || !claims.contains("sub") ||
        !claims.at("sub").is_string() || !claims.contains("exp") ||
        !claims.at("exp").is_number_integer() ||
        claims.at("exp").get<std::int64_t>() <= epochSeconds(now))
        throw std::invalid_argument("invalid JWT claims");
    return claims;
}
}

JwtService::JwtService(JwtConfig config) : config_(std::move(config))
{
    if (config_.accessSecret.size() < 32 || config_.capabilitySecret.size() < 32 ||
        config_.accessSecret == config_.capabilitySecret)
        throw std::invalid_argument("JWT secrets must be distinct and at least 32 bytes");
}

std::string JwtService::issueAccess(const domain::UserRecord& user,
                                    domain::TimePoint now) const
{
    return issue({{"iss", "treesem-backend"}, {"aud", "treesem-public"},
                  {"sub", user.userId}, {"role", domain::toString(user.role)},
                  {"ver", user.tokenVersion}, {"jti", infrastructure::generateOpaqueId("jti_")},
                  {"iat", epochSeconds(now)}, {"exp", epochSeconds(now + config_.accessTtl)}},
                 config_.accessSecret);
}

domain::ActorContext JwtService::verifyAccess(
    const std::string& token, domain::TimePoint now) const
{
    const auto claims = verify(token, config_.accessSecret, "treesem-public", now);
    domain::ActorContext actor;
    actor.userId = claims.at("sub").get<std::string>();
    actor.role = domain::parseUserRole(claims.at("role").get<std::string>());
    actor.tokenVersion = claims.at("ver").get<std::uint64_t>();
    return actor;
}

std::string JwtService::issueCapability(
    const CapabilityContext& context, domain::TimePoint now) const
{
    return issue({{"iss", "treesem-backend"}, {"aud", "treesem-internal-tools"},
                  {"sub", context.actorId}, {"role", domain::toString(context.actorRole)},
                  {"session_id", context.sessionId}, {"subject_user_id", context.subjectUserId},
                  {"run_id", context.runId}, {"tools", context.allowedTools},
                  {"jti", infrastructure::generateOpaqueId("jti_")},
                  {"iat", epochSeconds(now)}, {"exp", epochSeconds(now + config_.capabilityTtl)}},
                 config_.capabilitySecret);
}

CapabilityContext JwtService::verifyCapability(
    const std::string& token, domain::TimePoint now) const
{
    const auto claims = verify(token, config_.capabilitySecret,
                               "treesem-internal-tools", now);
    return {claims.at("sub").get<std::string>(),
            domain::parseUserRole(claims.at("role").get<std::string>()),
            claims.at("session_id").get<std::string>(),
            claims.at("subject_user_id").get<std::string>(),
            claims.at("run_id").get<std::string>(),
            claims.at("tools").get<std::vector<std::string>>()};
}
} // namespace treesem::security
