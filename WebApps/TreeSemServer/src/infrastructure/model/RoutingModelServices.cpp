#include "infrastructure/model/RoutingModelServices.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include <muduo/base/Logging.h>

#include "model/ModelException.h"

namespace treesem
{
namespace infrastructure
{
namespace
{
std::string stableBackendLabel(const std::optional<std::string>& backend)
{
    if (!backend.has_value()) return "unknown";
    for (const char* allowed : {"onnx", "python_reference", "remote_fallback"})
        if (*backend == allowed) return *backend;
    return "other";
}
}

FallbackModelService::FallbackModelService(
    const model::IModelService& primary,
    const model::IModelService& fallback,
    std::shared_ptr<http::observability::MetricsRegistry> metrics)
    : primary_(primary)
    , fallback_(fallback), metrics_(std::move(metrics))
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
        if (metrics_)
            metrics_->increment("treesem_model_fallback_total",
                                {{"reason", "inference_failure"}});
        model::ModelResult result = fallback_.predict(input);
        result.servingBackend = "remote_fallback";
        return result;
    }
}

ShadowModelService::ShadowModelService(
    const model::IModelService& primary,
    const model::IModelService& shadow,
    std::shared_ptr<http::observability::MetricsRegistry> metrics)
    : primary_(primary)
    , shadow_(shadow), metrics_(std::move(metrics))
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
            if (metrics_)
                metrics_->increment("treesem_model_shadow_mismatch_total",
                    {{"type", discreteMismatch ? "discrete" : "numeric"}});
            LOG_WARN << "treeSem shadow mismatch model_version="
                     << primaryResult.modelVersion.value_or("unknown")
                     << " discrete=" << (discreteMismatch ? "true" : "false")
                     << " max_probability_delta=" << maximumDelta;
        }
    }
    catch (const std::exception&)
    {
        if (metrics_) metrics_->increment("treesem_model_shadow_errors_total");
        LOG_WARN << "treeSem shadow backend failed; primary result retained";
    }
    return primaryResult;
}

InstrumentedModelService::InstrumentedModelService(
    const model::IModelService& delegate,
    std::shared_ptr<http::observability::MetricsRegistry> metrics)
    : delegate_(delegate), metrics_(std::move(metrics))
{
    if (!metrics_) throw std::invalid_argument("model metrics registry is required");
}

model::ModelResult InstrumentedModelService::predict(
    const model::ModelInput& input) const
{
    const auto started = std::chrono::steady_clock::now();
    try
    {
        model::ModelResult result = delegate_.predict(input);
        const std::string backend = stableBackendLabel(result.servingBackend);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        metrics_->increment("treesem_model_inferences_total",
            {{"backend", backend}, {"result", "success"}});
        metrics_->observe("treesem_model_inference_duration_seconds",
            {{"backend", backend}}, elapsed);
        return result;
    }
    catch (...)
    {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        metrics_->increment("treesem_model_inferences_total",
            {{"backend", "unknown"}, {"result", "error"}});
        metrics_->observe("treesem_model_inference_duration_seconds",
            {{"backend", "unknown"}}, elapsed);
        throw;
    }
}

} // namespace infrastructure
} // namespace treesem
