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
        treesem::client::AgentResponse response;
        response.answer = "Safe final answer [cite_aaaaaaaaaaaaaaaaaaaa]";
        response.stepCount = 2;
        response.tools = {{"search_medical_knowledge", "success", 1}};
        response.groundingSourceIds = {"cite_aaaaaaaaaaaaaaaaaaaa"};
        response.citations = {{"cite_aaaaaaaaaaaaaaaaaaaa", "src_who_pph",
            "WHO guideline", "Overview", 1, "WHO", "2025-10-05",
            "https://www.who.int/example"}};
        response.knowledgeIndexVersion = "knowledge-test";
        response.skillUsed = treesem::domain::AgentSkillUse{
            "pph_evidence_education", "1.0.0", std::string(64, 'a')};
        return response;
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
    assert(first.finalMessage.content.find("Safe final answer") == 0);
    assert(first.run.groundingSourceIds.size() == 1);
    assert(first.run.citations.size() == 1);
    assert(first.run.skillUsed.has_value());
    auto replay = service.chat("hello", "request-key-001",
        first.session.session.sessionId, treesem::application::SessionAccess::Public);
    assert(replay.replayed);
    assert(replay.run.runId == first.run.runId);
    assert(replay.run.knowledgeIndexVersion == first.run.knowledgeIndexVersion);
    assert(replay.run.skillUsed->version == "1.0.0");
    const auto history = service.history(first.run.sessionId, 20, std::nullopt);
    assert(history.items.size() == 2);
}
