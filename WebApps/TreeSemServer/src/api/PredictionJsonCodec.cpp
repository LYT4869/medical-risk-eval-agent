#include "api/PredictionJsonCodec.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "api/ApiException.h"

namespace treesem
{
namespace api
{
namespace
{

using Json = nlohmann::json;

[[noreturn]] void invalidRequest()
{
    throw ApiException(
        ApiException::Kind::InvalidRequest,
        "prediction request does not match the API contract");
}

Json decisionPathStepToJson(const model::DecisionPathStep& step)
{
    if (step.leafId.has_value())
    {
        if (step.nodeId.has_value() || step.featureIndex.has_value() ||
            step.featureName.has_value() || step.comparisonOperator.has_value() ||
            step.thresholdStandardized.has_value() || step.valueStandardized.has_value())
        {
            throw std::logic_error("leaf decision path step contains branch fields");
        }
        return Json{{"leaf_id", *step.leafId}};
    }

    if (!step.nodeId.has_value() || !step.featureIndex.has_value() ||
        !step.featureName.has_value() || !step.comparisonOperator.has_value() ||
        !step.thresholdStandardized.has_value() || !step.valueStandardized.has_value())
    {
        throw std::logic_error("branch decision path step is incomplete");
    }
    return Json{
        {"node_id", *step.nodeId},
        {"feature_index", *step.featureIndex},
        {"feature_name", *step.featureName},
        {"operator", *step.comparisonOperator},
        {"threshold_standardized", *step.thresholdStandardized},
        {"value_standardized", *step.valueStandardized},
    };
}

} // namespace

model::ModelInput PredictionJsonCodec::parseInput(const std::string& requestBody)
{
    Json request;
    try
    {
        request = Json::parse(requestBody);
    }
    catch (const Json::exception&)
    {
        throw ApiException(ApiException::Kind::InvalidJson, "invalid JSON");
    }

    try
    {
        if (!request.is_object())
        {
            invalidRequest();
        }
        const bool hasIndex = request.contains("sample_index");
        const bool hasFeatures = request.contains("preprocessed_features");
        if (hasIndex == hasFeatures)
        {
            invalidRequest();
        }

        if (hasIndex)
        {
            const Json& value = request.at("sample_index");
            if (!value.is_number_integer())
            {
                invalidRequest();
            }
            if (value.is_number_unsigned())
            {
                const std::uint64_t index = value.get<std::uint64_t>();
                if (index > static_cast<std::uint64_t>(
                                std::numeric_limits<std::int64_t>::max()))
                {
                    invalidRequest();
                }
                return model::SampleIndexInput{static_cast<std::int64_t>(index)};
            }
            const std::int64_t index = value.get<std::int64_t>();
            if (index < 0)
            {
                invalidRequest();
            }
            return model::SampleIndexInput{index};
        }

        const Json& values = request.at("preprocessed_features");
        if (!values.is_array() || values.size() != model::kPphInputDimension)
        {
            invalidRequest();
        }
        std::vector<double> features;
        features.reserve(values.size());
        for (const auto& value : values)
        {
            if (!value.is_number())
            {
                invalidRequest();
            }
            const double feature = value.get<double>();
            if (!std::isfinite(feature))
            {
                invalidRequest();
            }
            features.push_back(feature);
        }
        return model::PreprocessedFeaturesInput{std::move(features)};
    }
    catch (const ApiException&)
    {
        throw;
    }
    catch (const Json::exception&)
    {
        invalidRequest();
    }
}

std::string PredictionJsonCodec::serializeResult(const model::ModelResult& result)
{
    Json importantFeatures = Json::array();
    for (const model::ImportantFeature& feature : result.importantFeatures)
    {
        importantFeatures.push_back(Json{
            {"index", feature.index},
            {"name", feature.name},
            {"standardized_value", feature.standardizedValue},
            {"tree_importance", feature.treeImportance},
        });
    }

    Json decisionPath = Json::array();
    for (const model::DecisionPathStep& step : result.decisionPath)
    {
        decisionPath.push_back(decisionPathStepToJson(step));
    }

    Json response{
        {"model", result.modelName},
        {"dataset", result.dataset},
        {"input_source", result.inputSource},
        {"sample_index", result.sampleIndex.has_value() ? Json(*result.sampleIndex) : Json(nullptr)},
        {"prediction",
         {
             {"label", result.prediction.label},
             {"positive_probability", result.prediction.positiveProbability},
             {"confidence", result.prediction.confidence},
             {"cluster_id", result.prediction.clusterId},
             {"tree_probability", result.prediction.treeProbability},
             {"tree_leaf_id", result.prediction.treeLeafId},
         }},
        {"important_features", std::move(importantFeatures)},
        {"decision_path", std::move(decisionPath)},
    };
    return response.dump();
}

} // namespace api
} // namespace treesem
