#pragma once

#include <cstddef>
#include <string>

#include "client/IAgentClient.h"

namespace treesem::client
{

struct PythonAgentClientConfig
{
    std::string url{"http://127.0.0.1:8091/v1/agent/runs"};
    std::string serviceSecret;
    long connectTimeoutMs{500};
    long requestTimeoutMs{30000};
    std::size_t maxResponseBytes{1024 * 1024};
};

class PythonAgentClient final : public IAgentClient
{
public:
    explicit PythonAgentClient(PythonAgentClientConfig config);
    AgentResponse run(const AgentRequest& request) const override;
private:
    PythonAgentClientConfig config_;
};

} // namespace treesem::client
