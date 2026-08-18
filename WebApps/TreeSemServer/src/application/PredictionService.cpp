#include "application/PredictionService.h"

#include <chrono>
#include <stdexcept>

#include "application/BusinessException.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem
{
namespace application
{

PredictionService::PredictionService(const model::IModelService& modelService)
    : modelService_(modelService)
{}

PredictionService::PredictionService(
    const model::IModelService& modelService,
    persistence::ITreeSemStore& store,
    const SessionService& sessionService)
    : modelService_(modelService)
    , store_(&store)
    , sessionService_(&sessionService)
{}

model::ModelResult PredictionService::predict(const model::ModelInput& input) const
{
    return modelService_.predict(input);
}

std::pair<domain::PredictionRecord, ResolvedSession>
PredictionService::createPrediction(
    const model::ModelInput& input,
    const std::optional<std::string>& suppliedSessionId,
    SessionAccess access,
    const std::optional<std::string>& subjectUserId,
    const std::optional<std::string>& createdByUserId) const
{
    if (store_ == nullptr || sessionService_ == nullptr)
    {
        throw std::logic_error("prediction persistence is not configured");
    }
    ResolvedSession resolved = sessionService_->resolve(suppliedSessionId, access);
    model::ModelResult modelResult = modelService_.predict(input);
    const auto createdAt = domain::TimePoint(std::chrono::microseconds(
        infrastructure::epochMicroseconds(std::chrono::system_clock::now())));
    domain::PredictionRecord record;
    record.sessionId = resolved.session.sessionId;
    record.subjectUserId = subjectUserId;
    record.createdByUserId = createdByUserId;
    record.result = std::move(modelResult);
    record.createdAt = createdAt;
    for (int attempt = 0; attempt < 4; ++attempt)
    {
        record.predictionId = infrastructure::generateOpaqueId("pred_");
        try
        {
            store_->savePredictionAndSetCurrent(
                record, createdAt + sessionService_->ttl());
            break;
        }
        catch (const BusinessException& error)
        {
            if (error.kind() != BusinessException::Kind::Conflict || attempt == 3)
            {
                throw;
            }
        }
    }
    resolved.session.currentPredictionId = record.predictionId;
    resolved.session.lastAccessedAt = createdAt;
    resolved.session.expiresAt = createdAt + sessionService_->ttl();
    ++resolved.session.version;
    return {std::move(record), std::move(resolved)};
}

} // namespace application
} // namespace treesem
