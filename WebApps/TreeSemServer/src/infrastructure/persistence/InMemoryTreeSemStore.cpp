#include "infrastructure/persistence/InMemoryTreeSemStore.h"

#include <algorithm>

#include "application/BusinessException.h"

namespace treesem
{
namespace infrastructure
{

void InMemoryTreeSemStore::createSession(const domain::SessionRecord& session)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!sessions_.emplace(session.sessionId, session).second)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict,
            "session id collision");
    }
}

std::optional<domain::SessionRecord> InMemoryTreeSemStore::touchActiveSession(
    const std::string& sessionId,
    domain::TimePoint accessedAt,
    domain::TimePoint expiresAt)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = sessions_.find(sessionId);
    if (found == sessions_.end() || found->second.status != "active" ||
        found->second.expiresAt <= accessedAt)
    {
        return std::nullopt;
    }
    found->second.lastAccessedAt = accessedAt;
    found->second.expiresAt = expiresAt;
    ++found->second.version;
    return found->second;
}

void InMemoryTreeSemStore::savePredictionAndSetCurrent(
    const domain::PredictionRecord& prediction,
    domain::TimePoint sessionExpiresAt)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto session = sessions_.find(prediction.sessionId);
    if (session == sessions_.end() || session->second.status != "active" ||
        session->second.expiresAt <= prediction.createdAt)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict,
            "session expired before prediction commit");
    }
    if (!predictions_.emplace(prediction.predictionId, prediction).second)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict,
            "prediction id collision");
    }
    session->second.currentPredictionId = prediction.predictionId;
    session->second.lastAccessedAt = prediction.createdAt;
    session->second.expiresAt = sessionExpiresAt;
    ++session->second.version;
}

std::optional<domain::PredictionRecord> InMemoryTreeSemStore::findPrediction(
    const std::string& sessionId,
    const std::string& predictionId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = predictions_.find(predictionId);
    if (found == predictions_.end() || found->second.sessionId != sessionId)
    {
        return std::nullopt;
    }
    return found->second;
}

domain::HistoryPage InMemoryTreeSemStore::listPredictions(
    const std::string& sessionId,
    std::size_t limit,
    const std::optional<domain::HistoryCursor>& cursor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<domain::PredictionRecord> records;
    for (const auto& entry : predictions_)
    {
        const auto& record = entry.second;
        const bool beforeCursor = !cursor.has_value() ||
            record.createdAt < cursor->createdAt ||
            (record.createdAt == cursor->createdAt &&
             record.predictionId < cursor->predictionId);
        if (record.sessionId == sessionId && beforeCursor)
        {
            records.push_back(record);
        }
    }
    std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
        if (left.createdAt != right.createdAt)
        {
            return left.createdAt > right.createdAt;
        }
        return left.predictionId > right.predictionId;
    });
    domain::HistoryPage page;
    page.sessionId = sessionId;
    const std::size_t count = std::min(limit, records.size());
    for (std::size_t index = 0; index < count; ++index)
    {
        const auto& record = records[index];
        page.items.push_back({
            record.predictionId,
            record.createdAt,
            record.result.prediction.label,
            record.result.prediction.positiveProbability,
            record.result.prediction.confidence,
            record.result.modelVersion,
            record.result.servingBackend});
    }
    if (records.size() > limit && !page.items.empty())
    {
        page.nextCursor = domain::HistoryCursor{
            page.items.back().createdAt, page.items.back().predictionId};
    }
    return page;
}

domain::FeedbackSaveResult InMemoryTreeSemStore::saveFeedback(
    const domain::ClinicalFeedback& feedback)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto prediction = predictions_.find(feedback.predictionId);
    if (prediction == predictions_.end() ||
        prediction->second.sessionId != feedback.sessionId)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::NotFound,
            "prediction not found");
    }
    const auto existing = std::find_if(
        feedback_.begin(), feedback_.end(), [&](const auto& item) {
            return item.sessionId == feedback.sessionId &&
                item.idempotencyKey == feedback.idempotencyKey;
        });
    if (existing != feedback_.end())
    {
        if (existing->payloadSha256 != feedback.payloadSha256)
        {
            throw application::BusinessException(
                application::BusinessException::Kind::IdempotencyConflict,
                "idempotency key was reused with another payload");
        }
        return {*existing, false};
    }
    feedback_.push_back(feedback);
    return {feedback, true};
}

std::vector<domain::ClinicalFeedback> InMemoryTreeSemStore::listFeedback(
    const std::string& sessionId,
    const std::string& predictionId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto prediction = predictions_.find(predictionId);
    if (prediction == predictions_.end() || prediction->second.sessionId != sessionId)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::NotFound,
            "prediction not found");
    }
    std::vector<domain::ClinicalFeedback> result;
    for (const auto& item : feedback_)
    {
        if (item.sessionId == sessionId && item.predictionId == predictionId)
        {
            result.push_back(item);
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.createdAt < right.createdAt;
    });
    return result;
}

bool InMemoryTreeSemStore::ping()
{
    return true;
}

} // namespace infrastructure
} // namespace treesem
