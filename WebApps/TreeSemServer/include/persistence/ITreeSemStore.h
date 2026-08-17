#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "domain/BusinessTypes.h"

namespace treesem
{
namespace persistence
{

class ITreeSemStore
{
public:
    virtual ~ITreeSemStore() = default;

    virtual void createSession(const domain::SessionRecord& session) = 0;
    virtual std::optional<domain::SessionRecord> touchActiveSession(
        const std::string& sessionId,
        domain::TimePoint accessedAt,
        domain::TimePoint expiresAt) = 0;
    virtual void savePredictionAndSetCurrent(
        const domain::PredictionRecord& prediction,
        domain::TimePoint sessionExpiresAt) = 0;
    virtual std::optional<domain::PredictionRecord> findPrediction(
        const std::string& sessionId,
        const std::string& predictionId) = 0;
    virtual domain::HistoryPage listPredictions(
        const std::string& sessionId,
        std::size_t limit,
        const std::optional<domain::HistoryCursor>& cursor) = 0;
    virtual domain::FeedbackSaveResult saveFeedback(
        const domain::ClinicalFeedback& feedback) = 0;
    virtual std::vector<domain::ClinicalFeedback> listFeedback(
        const std::string& sessionId,
        const std::string& predictionId) = 0;
    virtual bool ping() = 0;
};

} // namespace persistence
} // namespace treesem
