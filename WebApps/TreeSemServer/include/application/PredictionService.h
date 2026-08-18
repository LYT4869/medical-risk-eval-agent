#pragma once

#include "model/IModelService.h"
#include "application/SessionService.h"
#include "domain/BusinessTypes.h"
#include "persistence/ITreeSemStore.h"

namespace treesem
{
namespace application
{

class PredictionService
{
public:
    explicit PredictionService(const model::IModelService& modelService);
    PredictionService(const model::IModelService& modelService,
                      persistence::ITreeSemStore& store,
                      const SessionService& sessionService);

    model::ModelResult predict(const model::ModelInput& input) const;
    std::pair<domain::PredictionRecord, ResolvedSession> createPrediction(
        const model::ModelInput& input,
        const std::optional<std::string>& suppliedSessionId,
        SessionAccess access,
        const std::optional<std::string>& subjectUserId = std::nullopt,
        const std::optional<std::string>& createdByUserId = std::nullopt) const;

private:
    const model::IModelService& modelService_;
    persistence::ITreeSemStore* store_{nullptr};
    const SessionService* sessionService_{nullptr};
};

} // namespace application
} // namespace treesem
