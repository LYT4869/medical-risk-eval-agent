#include "infrastructure/support/ValueSupport.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

namespace treesem
{
namespace infrastructure
{
namespace
{

std::string base64UrlEncode(const std::string& input)
{
    std::string output(4 * ((input.size() + 2) / 3), '\0');
    const int size = EVP_EncodeBlock(
        reinterpret_cast<unsigned char*>(output.data()),
        reinterpret_cast<const unsigned char*>(input.data()),
        static_cast<int>(input.size()));
    output.resize(static_cast<std::size_t>(size));
    for (char& value : output)
    {
        if (value == '+') value = '-';
        else if (value == '/') value = '_';
    }
    while (!output.empty() && output.back() == '=') output.pop_back();
    return output;
}

std::string base64UrlDecode(std::string input)
{
    if (input.empty() || input.size() > 256)
    {
        throw std::invalid_argument("invalid history cursor");
    }
    for (char& value : input)
    {
        if (value == '-') value = '+';
        else if (value == '_') value = '/';
        else if (!std::isalnum(static_cast<unsigned char>(value)))
        {
            throw std::invalid_argument("invalid history cursor");
        }
    }
    while (input.size() % 4 != 0) input.push_back('=');
    std::string output(3 * (input.size() / 4), '\0');
    const int size = EVP_DecodeBlock(
        reinterpret_cast<unsigned char*>(output.data()),
        reinterpret_cast<const unsigned char*>(input.data()),
        static_cast<int>(input.size()));
    if (size < 0)
    {
        throw std::invalid_argument("invalid history cursor");
    }
    std::size_t padding = 0;
    if (!input.empty() && input.back() == '=') ++padding;
    if (input.size() > 1 && input[input.size() - 2] == '=') ++padding;
    output.resize(static_cast<std::size_t>(size) - padding);
    return output;
}

} // namespace

std::string generateOpaqueId(const std::string& prefix)
{
    std::array<unsigned char, 16> random{};
    if (RAND_bytes(random.data(), static_cast<int>(random.size())) != 1)
    {
        throw std::runtime_error("secure random id generation failed");
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string result = prefix;
    result.reserve(prefix.size() + random.size() * 2);
    for (unsigned char value : random)
    {
        result.push_back(hex[value >> 4]);
        result.push_back(hex[value & 0x0f]);
    }
    return result;
}

std::string sha256Hex(const std::string& value)
{
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(
        reinterpret_cast<const unsigned char*>(value.data()), value.size(), digest.data());
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2);
    for (unsigned char byte : digest)
    {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
    }
    return result;
}

std::int64_t epochMicroseconds(domain::TimePoint value)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        value.time_since_epoch()).count();
}

domain::TimePoint fromEpochMicroseconds(std::int64_t value)
{
    return domain::TimePoint(std::chrono::microseconds(value));
}

std::string formatUtc(domain::TimePoint value)
{
    const std::int64_t micros = epochMicroseconds(value);
    const std::time_t seconds = static_cast<std::time_t>(micros / 1000000);
    const int fraction = static_cast<int>((micros % 1000000 + 1000000) % 1000000);
    std::tm utc{};
    gmtime_r(&seconds, &utc);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
           << std::setw(6) << std::setfill('0') << fraction << 'Z';
    return output.str();
}

domain::TimePoint parseUtc(const std::string& value)
{
    if (value.size() != 27 || value[19] != '.' || value.back() != 'Z')
    {
        throw std::invalid_argument("invalid UTC timestamp");
    }
    std::tm utc{};
    std::istringstream input(value.substr(0, 19));
    input >> std::get_time(&utc, "%Y-%m-%dT%H:%M:%S");
    if (input.fail())
    {
        throw std::invalid_argument("invalid UTC timestamp");
    }
    const std::string fraction = value.substr(20, 6);
    if (!std::all_of(fraction.begin(), fraction.end(), [](char item) {
            return std::isdigit(static_cast<unsigned char>(item));
        }))
    {
        throw std::invalid_argument("invalid UTC timestamp");
    }
    const std::time_t seconds = timegm(&utc);
    return domain::TimePoint(std::chrono::seconds(seconds) +
                             std::chrono::microseconds(std::stoll(fraction)));
}

std::string encodeHistoryCursor(const domain::HistoryCursor& cursor)
{
    return base64UrlEncode(
        std::to_string(epochMicroseconds(cursor.createdAt)) + ":" +
        cursor.predictionId);
}

domain::HistoryCursor decodeHistoryCursor(const std::string& cursor)
{
    const std::string decoded = base64UrlDecode(cursor);
    const std::size_t separator = decoded.find(':');
    if (separator == std::string::npos)
    {
        throw std::invalid_argument("invalid history cursor");
    }
    std::size_t consumed = 0;
    const std::int64_t micros = std::stoll(decoded.substr(0, separator), &consumed);
    if (consumed != separator)
    {
        throw std::invalid_argument("invalid history cursor");
    }
    const std::string id = decoded.substr(separator + 1);
    if (!isValidOpaqueId(id, "pred_"))
    {
        throw std::invalid_argument("invalid history cursor");
    }
    return {fromEpochMicroseconds(micros), id};
}

bool isValidOpaqueId(const std::string& value, const std::string& prefix)
{
    if (value.size() != prefix.size() + 32 || value.compare(0, prefix.size(), prefix) != 0)
    {
        return false;
    }
    return std::all_of(value.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
                       value.end(), [](char item) {
        return std::isdigit(static_cast<unsigned char>(item)) ||
            (item >= 'a' && item <= 'f');
    });
}

bool isValidIdempotencyKey(const std::string& value)
{
    return value.size() >= 8 && value.size() <= 64 &&
        std::all_of(value.begin(), value.end(), [](char item) {
            return std::isalnum(static_cast<unsigned char>(item)) || item == '-' ||
                item == '_' || item == '.';
        });
}

} // namespace infrastructure
} // namespace treesem
