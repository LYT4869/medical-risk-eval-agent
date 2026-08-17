#pragma once

#include <memory>

#include "config/TreeSemServerConfig.h"
#include "infrastructure/persistence/MySqlConnectionPool.h"
#include "persistence/ITreeSemStore.h"

namespace treesem
{
namespace infrastructure
{

struct MySqlStoreConfig
{
    MySqlConnectionConfig connection;
    static MySqlStoreConfig from(const config::TreeSemServerConfig& config);
};

class MySqlTreeSemStore : public persistence::ITreeSemStore
{
public:
    explicit MySqlTreeSemStore(MySqlStoreConfig config);

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
    MySqlConnectionPool pool_;
};

} // namespace infrastructure
} // namespace treesem
