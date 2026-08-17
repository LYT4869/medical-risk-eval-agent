#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "client/IModelAdapterClient.h"

namespace treesem
{
namespace client
{

class ModelAdapterException : public std::runtime_error
{
public:
    enum class Kind
    {
        Timeout,
        Unavailable,
        InvalidResponse,
    };

    ModelAdapterException(Kind kind, const std::string& message);
    Kind kind() const noexcept;

private:
    Kind kind_;
};

struct PythonModelClientConfig
{
    std::string predictUrl{"http://127.0.0.1:18081/v1/predict"};
    long connectTimeoutMs{500};
    long requestTimeoutMs{5000};
    std::size_t maxResponseBytes{1024 * 1024};
};

class PythonModelClient final : public IModelAdapterClient
{
public:
    explicit PythonModelClient(PythonModelClientConfig config = {});
    ModelAdapterResponse predict(const std::string& requestBody) const override;

private:
    PythonModelClientConfig config_;
};

} // namespace client
} // namespace treesem
