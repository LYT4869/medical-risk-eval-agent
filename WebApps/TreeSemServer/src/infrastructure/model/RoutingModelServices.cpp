#include "infrastructure/model/RoutingModelServices.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <muduo/base/Logging.h>

#include "model/ModelException.h"

namespace treesem
{
namespace infrastructure
{

FallbackModelService::FallbackModelService(
    const model::IModelService& primary,
    const model::IModelService& fallback)
    : primary_(primary)
    , fallback_(fallback)
{}

model::ModelResult FallbackModelService::predict(
    const model::ModelInput& input) const
{
    try
    {
        return primary_.predict(input);
    }
    catch (const model::ModelException& error)
    {
        if (error.kind() != model::ModelException::Kind::InferenceFailure)
        {
            throw;
        }
        LOG_WARN << "treeSem ONNX inference failed; using remote fallback";
        model::ModelResult result = fallback_.predict(input);
        result.servingBackend = "remote_fallback";
        return result;
    }
}

ShadowModelService::ShadowModelService(
    const model::IModelService& primary,
    const model::IModelService& shadow)
    : primary_(primary)
    , shadow_(shadow)
{}

model::ModelResult ShadowModelService::predict(
    const model::ModelInput& input) const
{
    model::ModelResult primaryResult = primary_.predict(input);
    try
    {
        const model::ModelResult shadowResult = shadow_.predict(input);
        const double probabilityDelta = std::abs(
            primaryResult.prediction.positiveProbability -
            shadowResult.prediction.positiveProbability);
        const double confidenceDelta = std::abs(
            primaryResult.prediction.confidence -
            shadowResult.prediction.confidence);
        const double maximumDelta = std::max(probabilityDelta, confidenceDelta);
        const bool discreteMismatch =
            primaryResult.prediction.label != shadowResult.prediction.label ||
            primaryResult.prediction.clusterId != shadowResult.prediction.clusterId ||
            primaryResult.prediction.treeLeafId != shadowResult.prediction.treeLeafId;
        if (discreteMismatch || maximumDelta > 1e-5)
        {
            LOG_WARN << "treeSem shadow mismatch model_version="
                     << primaryResult.modelVersion.value_or("unknown")
                     << " discrete=" << (discreteMismatch ? "true" : "false")
                     << " max_probability_delta=" << maximumDelta;
        }
    }
    catch (const std::exception&)
    {
        LOG_WARN << "treeSem shadow backend failed; primary result retained";
    }
    return primaryResult;
}

} // namespace infrastructure
} // namespace treesem
