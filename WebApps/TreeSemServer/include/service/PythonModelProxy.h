#pragma once

#include <string>

#include "client/IModelAdapterClient.h"
#include "http/AsyncHttp.h"

namespace treesem
{
namespace service
{

class PythonModelProxy
{
public:
    explicit PythonModelProxy(const client::IModelAdapterClient& client);
    http::ResponseWriter predict(const std::string& requestBody) const;

private:
    const client::IModelAdapterClient& client_;
};

} // namespace service
} // namespace treesem
