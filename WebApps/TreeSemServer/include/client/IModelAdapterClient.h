#pragma once

#include <string>

namespace treesem
{
namespace client
{

struct ModelAdapterResponse
{
    long statusCode;
    std::string body;
};

class IModelAdapterClient
{
public:
    virtual ~IModelAdapterClient() = default;
    virtual ModelAdapterResponse predict(const std::string& requestBody) const = 0;
};

} // namespace client
} // namespace treesem
