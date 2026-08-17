#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "domain/BusinessTypes.h"

namespace treesem
{
namespace infrastructure
{

std::string generateOpaqueId(const std::string& prefix);
std::string sha256Hex(const std::string& value);
std::string formatUtc(domain::TimePoint value);
domain::TimePoint parseUtc(const std::string& value);
std::int64_t epochMicroseconds(domain::TimePoint value);
domain::TimePoint fromEpochMicroseconds(std::int64_t value);
std::string encodeHistoryCursor(const domain::HistoryCursor& cursor);
domain::HistoryCursor decodeHistoryCursor(const std::string& cursor);
bool isValidOpaqueId(const std::string& value, const std::string& prefix);
bool isValidIdempotencyKey(const std::string& value);

} // namespace infrastructure
} // namespace treesem
