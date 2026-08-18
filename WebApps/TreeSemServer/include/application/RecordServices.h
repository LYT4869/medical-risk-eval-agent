#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "domain/BusinessTypes.h"
#include "persistence/ITreeSemStore.h"

namespace treesem
{
namespace application
{

class ExplanationService
{
public:
    explicit ExplanationService(persistence::ITreeSemStore& store);
    domain::PredictionRecord get(const std::string& sessionId,
                                 const std::string& predictionId) const;
private:
    persistence::ITreeSemStore& store_;
};

class HistoryService
{
public:
    explicit HistoryService(persistence::ITreeSemStore& store);
    domain::HistoryPage list(
        const std::string& sessionId,
        std::size_t limit,
        const std::optional<domain::HistoryCursor>& cursor) const;
private:
    persistence::ITreeSemStore& store_;
};

class ComparisonService
{
public:
    explicit ComparisonService(persistence::ITreeSemStore& store);
    domain::PredictionComparison compare(
        const std::string& sessionId,
        const std::string& predictionIdA,
        const std::string& predictionIdB) const;
private:
    persistence::ITreeSemStore& store_;
};

struct FeedbackInput
{
    std::string reviewerReference;
    domain::FeedbackAssessment assessment;
    std::optional<int> correctedLabel;
    std::optional<std::string> comment;
    std::string idempotencyKey;
    bool reviewerVerified{false};
};

class FeedbackService
{
public:
    explicit FeedbackService(persistence::ITreeSemStore& store);
    domain::FeedbackSaveResult submit(
        const std::string& sessionId,
        const std::string& predictionId,
        const FeedbackInput& input) const;
    std::vector<domain::ClinicalFeedback> list(
        const std::string& sessionId,
        const std::string& predictionId) const;
private:
    persistence::ITreeSemStore& store_;
};

} // namespace application
} // namespace treesem
