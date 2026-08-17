#include <cassert>
#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "ModelFixtures.h"
#include "application/BusinessException.h"
#include "application/PredictionService.h"
#include "application/RecordServices.h"
#include "application/SessionService.h"
#include "infrastructure/persistence/InMemoryTreeSemStore.h"
#include "infrastructure/support/ValueSupport.h"
#include "model/IModelService.h"

namespace
{

class FakeModel : public treesem::model::IModelService
{
public:
    treesem::model::ModelResult predict(
        const treesem::model::ModelInput& input) const override
    {
        auto result = test::validModelResult();
        if (const auto* sample = std::get_if<treesem::model::SampleIndexInput>(&input);
            sample != nullptr && sample->sampleIndex == 1)
        {
            result.sampleIndex = 1;
            result.prediction.label = 1;
            result.prediction.positiveProbability = 0.8;
            result.prediction.confidence = 0.8;
            result.prediction.clusterId = 2;
            result.prediction.treeLeafId = 9;
            result.importantFeatures[0].standardizedValue = 0.3;
            result.importantFeatures[0].originalValue = 250.0;
            result.decisionPath.back().leafId = 9;
        }
        return result;
    }
};

} // namespace

int main()
{
    using namespace treesem;
    infrastructure::InMemoryTreeSemStore store;
    application::SessionService sessions(store, std::chrono::seconds(3600));
    FakeModel model;
    application::PredictionService predictions(model, store, sessions);
    application::ExplanationService explanations(store);
    application::HistoryService history(store);
    application::ComparisonService comparisons(store);
    application::FeedbackService feedback(store);

    auto first = predictions.createPrediction(
        model::SampleIndexInput{0}, std::nullopt, application::SessionAccess::Public);
    assert(first.second.created);
    assert(infrastructure::isValidOpaqueId(first.first.predictionId, "pred_"));
    assert(infrastructure::isValidOpaqueId(first.second.session.sessionId, "ses_"));

    auto second = predictions.createPrediction(
        model::SampleIndexInput{1}, first.second.session.sessionId,
        application::SessionAccess::Internal);
    assert(!second.second.created);
    assert(second.first.sessionId == first.first.sessionId);

    const auto loaded = explanations.get(
        first.first.sessionId, first.first.predictionId);
    assert(loaded.result.prediction.label == 0);

    auto page = history.list(first.first.sessionId, 1, std::nullopt);
    assert(page.items.size() == 1);
    assert(page.nextCursor.has_value());
    auto next = history.list(first.first.sessionId, 10, page.nextCursor);
    assert(next.items.size() == 1);
    assert(next.items.front().predictionId != page.items.front().predictionId);

    const auto comparison = comparisons.compare(
        first.first.sessionId, first.first.predictionId, second.first.predictionId);
    assert(comparison.labelChanged);
    assert(comparison.clusterChanged);
    assert(comparison.treeLeafChanged);
    assert(comparison.pathChanged);
    assert(comparison.positiveProbabilityDelta > 0.69);
    assert(!comparison.changedFeatures.empty());

    const auto identical = comparisons.compare(
        first.first.sessionId, first.first.predictionId, first.first.predictionId);
    assert(!identical.labelChanged);
    assert(!identical.modelVersionChanged);
    assert(!identical.clusterChanged);
    assert(!identical.treeLeafChanged);
    assert(!identical.pathChanged);
    assert(identical.positiveProbabilityDelta == 0.0);
    assert(identical.confidenceDelta == 0.0);
    assert(identical.changedFeatures.empty());

    application::FeedbackInput input{
        "demo-doctor-001",
        domain::FeedbackAssessment::Disagree,
        1,
        std::string("reviewed against the chart"),
        "feedback-key-0001"};
    auto saved = feedback.submit(
        first.first.sessionId, first.first.predictionId, input);
    assert(saved.created);
    assert(!saved.feedback.reviewerVerified);
    auto replay = feedback.submit(
        first.first.sessionId, first.first.predictionId, input);
    assert(!replay.created);
    assert(replay.feedback.feedbackId == saved.feedback.feedbackId);
    assert(feedback.list(first.first.sessionId, first.first.predictionId).size() == 1);

    bool conflict = false;
    input.comment = "a different payload";
    try
    {
        (void)feedback.submit(first.first.sessionId, first.first.predictionId, input);
    }
    catch (const application::BusinessException& error)
    {
        conflict = error.kind() ==
            application::BusinessException::Kind::IdempotencyConflict;
    }
    assert(conflict);

    input.idempotencyKey = "feedback-key-0002";
    input.comment = "   ";
    bool blankCommentRejected = false;
    try
    {
        (void)feedback.submit(first.first.sessionId, first.first.predictionId, input);
    }
    catch (const application::BusinessException& error)
    {
        blankCommentRejected = error.kind() ==
            application::BusinessException::Kind::InvalidInput;
    }
    assert(blankCommentRejected);

    application::FeedbackInput agreement{
        "demo-doctor-001",
        domain::FeedbackAssessment::Agree,
        std::nullopt,
        std::nullopt,
        "feedback-key-0003"};
    assert(feedback.submit(
        first.first.sessionId, first.first.predictionId, agreement).created);

    const auto publicReplacement = sessions.resolve(
        std::string("ses_cccccccccccccccccccccccccccccccc"),
        application::SessionAccess::Public);
    assert(publicReplacement.created);
    assert(publicReplacement.session.sessionId !=
        "ses_cccccccccccccccccccccccccccccccc");
    bool unknownInternalRejected = false;
    try
    {
        (void)sessions.resolve(
            std::string("ses_dddddddddddddddddddddddddddddddd"),
            application::SessionAccess::Internal);
    }
    catch (const application::BusinessException& error)
    {
        unknownInternalRejected = error.kind() ==
            application::BusinessException::Kind::NotFound;
    }
    assert(unknownInternalRejected);

    std::vector<std::thread> readers;
    for (int index = 0; index < 8; ++index)
    {
        readers.emplace_back([&]() {
            const auto touched = sessions.resolve(
                first.first.sessionId, application::SessionAccess::Internal);
            assert(!touched.created);
        });
    }
    for (auto& reader : readers) reader.join();

    const domain::HistoryCursor cursor{std::chrono::system_clock::now(),
                                       first.first.predictionId};
    assert(infrastructure::decodeHistoryCursor(
        infrastructure::encodeHistoryCursor(cursor)).predictionId ==
        first.first.predictionId);
}
