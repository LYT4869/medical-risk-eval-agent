#include "application/RecordServices.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <map>
#include <set>

#include <nlohmann/json.hpp>

#include "application/BusinessException.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem
{
namespace application
{
namespace
{

domain::PredictionRecord requirePrediction(
    persistence::ITreeSemStore& store,
    const std::string& sessionId,
    const std::string& predictionId)
{
    if (!infrastructure::isValidOpaqueId(sessionId, "ses_") ||
        !infrastructure::isValidOpaqueId(predictionId, "pred_"))
    {
        throw BusinessException(
            BusinessException::Kind::InvalidInput, "invalid resource id");
    }
    auto prediction = store.findPrediction(sessionId, predictionId);
    if (!prediction.has_value())
    {
        throw BusinessException(
            BusinessException::Kind::NotFound, "prediction was not found");
    }
    return std::move(*prediction);
}

domain::PredictionSummary summarize(const domain::PredictionRecord& record)
{
    return {
        record.predictionId,
        record.createdAt,
        record.result.prediction.label,
        record.result.prediction.positiveProbability,
        record.result.prediction.confidence,
        record.result.modelVersion,
        record.result.servingBackend};
}

bool samePath(const std::vector<model::DecisionPathStep>& left,
              const std::vector<model::DecisionPathStep>& right)
{
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (left[index].nodeId != right[index].nodeId ||
            left[index].featureIndex != right[index].featureIndex ||
            left[index].comparisonOperator != right[index].comparisonOperator ||
            left[index].leafId != right[index].leafId)
        {
            return false;
        }
    }
    return true;
}

bool blank(const std::string& value)
{
    return std::all_of(value.begin(), value.end(), [](unsigned char item) {
        return std::isspace(item);
    });
}

} // namespace

ExplanationService::ExplanationService(persistence::ITreeSemStore& store)
    : store_(store)
{}

domain::PredictionRecord ExplanationService::get(
    const std::string& sessionId,
    const std::string& predictionId) const
{
    return requirePrediction(store_, sessionId, predictionId);
}

HistoryService::HistoryService(persistence::ITreeSemStore& store)
    : store_(store)
{}

domain::HistoryPage HistoryService::list(
    const std::string& sessionId,
    std::size_t limit,
    const std::optional<domain::HistoryCursor>& cursor) const
{
    if (!infrastructure::isValidOpaqueId(sessionId, "ses_") ||
        limit == 0 || limit > 100)
    {
        throw BusinessException(
            BusinessException::Kind::InvalidInput, "invalid history request");
    }
    return store_.listPredictions(sessionId, limit, cursor);
}

ComparisonService::ComparisonService(persistence::ITreeSemStore& store)
    : store_(store)
{}

domain::PredictionComparison ComparisonService::compare(
    const std::string& sessionId,
    const std::string& predictionIdA,
    const std::string& predictionIdB) const
{
    const domain::PredictionRecord left =
        requirePrediction(store_, sessionId, predictionIdA);
    const domain::PredictionRecord right =
        requirePrediction(store_, sessionId, predictionIdB);
    domain::PredictionComparison result;
    result.predictionA = summarize(left);
    result.predictionB = summarize(right);
    result.labelChanged = left.result.prediction.label != right.result.prediction.label;
    result.modelVersionChanged = left.result.modelVersion != right.result.modelVersion;
    result.positiveProbabilityDelta =
        right.result.prediction.positiveProbability -
        left.result.prediction.positiveProbability;
    result.confidenceDelta =
        right.result.prediction.confidence - left.result.prediction.confidence;
    result.clusterChanged =
        left.result.prediction.clusterId != right.result.prediction.clusterId;
    result.treeLeafChanged =
        left.result.prediction.treeLeafId != right.result.prediction.treeLeafId;
    result.pathChanged = !samePath(left.result.decisionPath, right.result.decisionPath);

    std::map<int, const model::ImportantFeature*> leftFeatures;
    std::map<int, const model::ImportantFeature*> rightFeatures;
    std::set<int> indices;
    for (const auto& feature : left.result.importantFeatures)
    {
        leftFeatures[feature.index] = &feature;
        indices.insert(feature.index);
    }
    for (const auto& feature : right.result.importantFeatures)
    {
        rightFeatures[feature.index] = &feature;
        indices.insert(feature.index);
    }
    for (int index : indices)
    {
        const auto leftFound = leftFeatures.find(index);
        const auto rightFound = rightFeatures.find(index);
        domain::FeatureDifference difference;
        difference.index = index;
        const model::ImportantFeature* leftFeature =
            leftFound == leftFeatures.end() ? nullptr : leftFound->second;
        const model::ImportantFeature* rightFeature =
            rightFound == rightFeatures.end() ? nullptr : rightFound->second;
        difference.name = leftFeature != nullptr ? leftFeature->name : rightFeature->name;
        if (leftFeature != nullptr)
        {
            difference.standardizedValueA = leftFeature->standardizedValue;
            difference.originalValueA = leftFeature->originalValue;
        }
        if (rightFeature != nullptr)
        {
            difference.standardizedValueB = rightFeature->standardizedValue;
            difference.originalValueB = rightFeature->originalValue;
        }
        if (leftFeature != nullptr && rightFeature != nullptr)
        {
            difference.standardizedDelta =
                rightFeature->standardizedValue - leftFeature->standardizedValue;
            if (leftFeature->originalValue.has_value() &&
                rightFeature->originalValue.has_value() &&
                leftFeature->unit == rightFeature->unit)
            {
                difference.originalDelta =
                    *rightFeature->originalValue - *leftFeature->originalValue;
                difference.unit = leftFeature->unit;
            }
        }
        const bool changed = leftFeature == nullptr || rightFeature == nullptr ||
            leftFeature->name != rightFeature->name ||
            leftFeature->standardizedValue != rightFeature->standardizedValue ||
            leftFeature->originalValue != rightFeature->originalValue ||
            leftFeature->unit != rightFeature->unit;
        if (changed) result.changedFeatures.push_back(std::move(difference));
    }
    return result;
}

FeedbackService::FeedbackService(persistence::ITreeSemStore& store)
    : store_(store)
{}

domain::FeedbackSaveResult FeedbackService::submit(
    const std::string& sessionId,
    const std::string& predictionId,
    const FeedbackInput& input) const
{
    const domain::PredictionRecord prediction =
        requirePrediction(store_, sessionId, predictionId);
    if (input.reviewerReference.empty() || blank(input.reviewerReference) ||
        input.reviewerReference.size() > 128 ||
        !infrastructure::isValidIdempotencyKey(input.idempotencyKey) ||
        (input.comment.has_value() && input.comment->size() > 1000))
    {
        throw BusinessException(
            BusinessException::Kind::InvalidInput, "invalid feedback request");
    }
    const bool needsComment = input.assessment != domain::FeedbackAssessment::Agree;
    if ((needsComment && (!input.comment.has_value() || input.comment->empty() ||
                          blank(*input.comment))) ||
        (input.assessment != domain::FeedbackAssessment::Disagree &&
         input.correctedLabel.has_value()) ||
        (input.assessment == domain::FeedbackAssessment::Disagree &&
         (!input.correctedLabel.has_value() ||
          (*input.correctedLabel != 0 && *input.correctedLabel != 1) ||
          *input.correctedLabel == prediction.result.prediction.label)))
    {
        throw BusinessException(
            BusinessException::Kind::InvalidInput, "invalid feedback semantics");
    }
    const std::string canonical = nlohmann::json{
        {"prediction_id", predictionId},
        {"reviewer_reference", input.reviewerReference},
        {"assessment", domain::toString(input.assessment)},
        {"corrected_label", input.correctedLabel.has_value()
            ? nlohmann::json(*input.correctedLabel) : nlohmann::json(nullptr)},
        {"comment", input.comment.has_value()
            ? nlohmann::json(*input.comment) : nlohmann::json(nullptr)}}.dump();
    const auto now = domain::TimePoint(std::chrono::microseconds(
        infrastructure::epochMicroseconds(std::chrono::system_clock::now())));
    domain::ClinicalFeedback feedback{
        infrastructure::generateOpaqueId("fb_"),
        predictionId,
        sessionId,
        input.reviewerReference,
        false,
        input.assessment,
        input.correctedLabel,
        input.comment,
        input.idempotencyKey,
        infrastructure::sha256Hex(canonical),
        now};
    return store_.saveFeedback(feedback);
}

std::vector<domain::ClinicalFeedback> FeedbackService::list(
    const std::string& sessionId,
    const std::string& predictionId) const
{
    return store_.listFeedback(sessionId, predictionId);
}

} // namespace application
} // namespace treesem
