#pragma once

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "persistence/ITreeSemStore.h"

namespace treesem
{
namespace infrastructure
{

class InMemoryTreeSemStore : public persistence::ITreeSemStore
{
public:
    void createSession(const domain::SessionRecord& session) override;
    std::optional<domain::SessionRecord> touchActiveSession(
        const std::string& sessionId,
        domain::TimePoint accessedAt,
        domain::TimePoint expiresAt) override;
    void savePredictionAndSetCurrent(
        const domain::PredictionRecord& prediction,
        domain::TimePoint sessionExpiresAt) override;
    std::optional<domain::PredictionRecord> findPrediction(
        const std::string& sessionId,
        const std::string& predictionId) override;
    domain::HistoryPage listPredictions(
        const std::string& sessionId,
        std::size_t limit,
        const std::optional<domain::HistoryCursor>& cursor) override;
    domain::FeedbackSaveResult saveFeedback(
        const domain::ClinicalFeedback& feedback) override;
    std::vector<domain::ClinicalFeedback> listFeedback(
        const std::string& sessionId,
        const std::string& predictionId) override;
    bool ping() override;

private:
    std::mutex mutex_;
    std::unordered_map<std::string, domain::SessionRecord> sessions_;
    std::unordered_map<std::string, domain::PredictionRecord> predictions_;
    std::vector<domain::ClinicalFeedback> feedback_;
};

} // namespace infrastructure
} // namespace treesem
