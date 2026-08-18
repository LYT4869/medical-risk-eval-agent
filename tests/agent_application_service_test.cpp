#include <cassert>
#include <chrono>

#include "application/AgentApplicationService.h"
#include "application/SessionService.h"
#include "infrastructure/persistence/InMemoryTreeSemStore.h"

namespace
{
class FakeAgent : public treesem::client::IAgentClient
{
public:
    treesem::client::AgentResponse run(
        const treesem::client::AgentRequest& request) const override
    {
        assert(request.sessionId.rfind("ses_", 0) == 0);
        return {"Safe final answer", 2, {{"get_prediction_history", "success", 1}}, {}};
    }
};
}

int main()
{
    treesem::infrastructure::InMemoryTreeSemStore store;
    treesem::application::SessionService sessions(store, std::chrono::seconds(3600));
    FakeAgent agent;
    treesem::application::AgentApplicationService service(agent, store, sessions, 12);
    auto first = service.chat("hello", "request-key-001", std::nullopt,
        treesem::application::SessionAccess::Public);
    assert(first.finalMessage.content == "Safe final answer");
    auto replay = service.chat("hello", "request-key-001",
        first.session.session.sessionId, treesem::application::SessionAccess::Public);
    assert(replay.replayed);
    assert(replay.run.runId == first.run.runId);
    const auto history = service.history(first.run.sessionId, 20, std::nullopt);
    assert(history.items.size() == 2);
}
