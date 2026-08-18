#include "application/AgentApplicationService.h"

#include <chrono>

#include "application/BusinessException.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem::application
{
namespace
{
domain::TimePoint nowUtc()
{
    return domain::TimePoint(std::chrono::microseconds(
        infrastructure::epochMicroseconds(std::chrono::system_clock::now())));
}
}

AgentApplicationService::AgentApplicationService(
    const client::IAgentClient& agent, persistence::ITreeSemStore& store,
    const SessionService& sessions, std::size_t contextMessages,
    const security::JwtService* jwt)
    : agent_(agent), store_(store), sessions_(sessions),
      contextMessages_(contextMessages), jwt_(jwt)
{
    if (contextMessages_ == 0) throw std::invalid_argument("chat context limit must be positive");
}

ChatResult AgentApplicationService::chat(
    const std::string& message, const std::string& idempotencyKey,
    const std::optional<std::string>& suppliedSession, SessionAccess access,
    const std::optional<domain::ActorContext>& actor) const
{
    ResolvedSession resolved = sessions_.resolve(suppliedSession, access);
    const auto startedAt = nowUtc();
    domain::AgentRunRecord run;
    run.runId = infrastructure::generateOpaqueId("run_");
    run.sessionId = resolved.session.sessionId;
    run.idempotencyKey = idempotencyKey;
    run.payloadSha256 = infrastructure::sha256Hex(message);
    run.startedAt = startedAt;
    if (actor.has_value())
    {
        run.actorUserId = actor->userId;
        run.subjectUserId = actor->subjectUserId.value_or(actor->userId);
    }
    domain::ChatMessage user{
        infrastructure::generateOpaqueId("msg_"), run.sessionId, run.runId,
        "user", message, startedAt, run.actorUserId, run.subjectUserId};
    const auto start = store_.startAgentRun(run, user);
    if (!start.created)
    {
        if (start.run.status == domain::AgentRunStatus::Running)
            throw BusinessException(BusinessException::Kind::AgentRunInProgress,
                                    "agent run is in progress");
        if (start.run.status == domain::AgentRunStatus::Failed ||
            !start.finalMessage.has_value())
            throw BusinessException(BusinessException::Kind::PersistenceFailure,
                                    "prior agent run failed");
        return {start.run, *start.finalMessage, std::move(resolved), true};
    }
    try
    {
        client::AgentRequest request;
        request.runId = run.runId;
        request.sessionId = run.sessionId;
        request.message = message;
        request.recentMessages = store_.recentChatMessages(run.sessionId,
                                                           contextMessages_ + 1);
        if (!request.recentMessages.empty() &&
            request.recentMessages.back().messageId == user.messageId)
            request.recentMessages.pop_back();
        request.currentPredictionId = resolved.session.currentPredictionId;
        if (actor.has_value() && jwt_ != nullptr)
        {
            security::CapabilityContext capability;
            capability.actorId = actor->userId;
            capability.actorRole = actor->role;
            capability.sessionId = run.sessionId;
            capability.subjectUserId = actor->subjectUserId.value_or(actor->userId);
            capability.runId = run.runId;
            capability.allowedTools = {"predict_sample", "get_prediction",
                "get_explanation", "get_prediction_history", "compare_predictions"};
            request.capabilityToken = jwt_->issueCapability(capability, startedAt);
        }
        if (request.currentPredictionId.has_value())
        {
            const auto prediction = store_.findPrediction(
                run.sessionId, *request.currentPredictionId);
            if (prediction.has_value())
                request.currentModelVersion = prediction->result.modelVersion;
        }
        const client::AgentResponse response = agent_.run(request);
        const auto completedAt = nowUtc();
        domain::ChatMessage assistant{
            infrastructure::generateOpaqueId("msg_"), run.sessionId, run.runId,
            "assistant", response.answer, completedAt,
            run.actorUserId, run.subjectUserId};
        run.status = domain::AgentRunStatus::Completed;
        run.stepCount = response.stepCount;
        run.tools = response.tools;
        run.groundingPredictionIds = response.groundingPredictionIds;
        run.finalMessageId = assistant.messageId;
        run.completedAt = completedAt;
        store_.completeAgentRun(run, assistant);
        return {std::move(run), std::move(assistant), std::move(resolved), false};
    }
    catch (...)
    {
        try { store_.failAgentRun(run.runId, "agent_execution_failed", nowUtc()); }
        catch (...) {}
        throw;
    }
}

domain::ChatPage AgentApplicationService::history(
    const std::string& sessionId, std::size_t limit,
    const std::optional<domain::ChatCursor>& cursor) const
{
    return store_.listChatMessages(sessionId, limit, cursor);
}

} // namespace treesem::application
