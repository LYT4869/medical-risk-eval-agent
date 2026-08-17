#include "api/PredictionJsonCodec.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "api/ApiException.h"
#include "model/PphFeatureSchema.h"

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
            step.thresholdStandardized.has_value() || step.valueStandardized.has_value() ||
            step.featureDisplayName.has_value() || step.thresholdOriginal.has_value() ||
            step.valueOriginal.has_value() || step.unit.has_value())
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
    Json result{
        {"node_id", *step.nodeId},
        {"feature_index", *step.featureIndex},
        {"feature_name", *step.featureName},
        {"operator", *step.comparisonOperator},
        {"threshold_standardized", *step.thresholdStandardized},
        {"value_standardized", *step.valueStandardized},
    };
    if (step.featureDisplayName.has_value())
    {
        result["feature_display_name"] = *step.featureDisplayName;
        result["threshold_original"] = step.thresholdOriginal.has_value()
            ? Json(*step.thresholdOriginal)
            : Json(nullptr);
        result["value_original"] = step.valueOriginal.has_value()
            ? Json(*step.valueOriginal)
            : Json(nullptr);
        result["unit"] = step.unit.has_value() ? Json(*step.unit) : Json(nullptr);
    }
    return result;
}

} // namespace

model::ModelInput PredictionJsonCodec::parseInput(const std::string& requestBody)
{
    Json request;
    bool duplicateKey = false;
    std::map<int, std::set<std::string>> objectKeys;
    try
    {
        request = Json::parse(
            requestBody,
            [&duplicateKey, &objectKeys](
                int depth,
                Json::parse_event_t event,
                Json& parsed) {
                if (event == Json::parse_event_t::object_start)
                {
                    objectKeys[depth].clear();
                }
                else if (event == Json::parse_event_t::key)
                {
                    const std::string key = parsed.get<std::string>();
                    if (!objectKeys[depth].insert(key).second)
                    {
                        duplicateKey = true;
                    }
                }
                else if (event == Json::parse_event_t::object_end)
                {
                    objectKeys.erase(depth);
                }
                return true;
            });
        if (duplicateKey)
        {
            throw ApiException(ApiException::Kind::InvalidRequest, "duplicate JSON key");
        }
    }
    catch (const ApiException&)
    {
        throw;
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
        const bool hasRawFeatures = request.contains("raw_features");
        if (static_cast<int>(hasIndex) + static_cast<int>(hasFeatures) +
                static_cast<int>(hasRawFeatures) !=
            1)
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

        if (hasRawFeatures)
        {
            const Json& values = request.at("raw_features");
            if (!values.is_object() || values.size() != model::kPphFeatureNames.size())
            {
                invalidRequest();
            }
            std::vector<double> rawFeatures;
            rawFeatures.reserve(model::kPphFeatureNames.size());
            for (std::string_view name : model::kPphFeatureNames)
            {
                const auto found = values.find(std::string(name));
                if (found == values.end() || !found->is_number() || found->is_boolean())
                {
                    invalidRequest();
                }
                const double feature = found->get<double>();
                if (!std::isfinite(feature))
                {
                    invalidRequest();
                }
                rawFeatures.push_back(feature);
            }
            return model::RawClinicalFeaturesInput{std::move(rawFeatures)};
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
        Json serialized{
            {"index", feature.index},
            {"name", feature.name},
            {"standardized_value", feature.standardizedValue},
            {"tree_importance", feature.treeImportance},
        };
        if (feature.displayName.has_value())
        {
            serialized["display_name"] = *feature.displayName;
            serialized["original_value"] = feature.originalValue.has_value()
                ? Json(*feature.originalValue)
                : Json(nullptr);
            serialized["unit"] = feature.unit.has_value()
                ? Json(*feature.unit)
                : Json(nullptr);
        }
        importantFeatures.push_back(std::move(serialized));
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
    if (result.modelVersion.has_value())
    {
        response["model_version"] = *result.modelVersion;
    }
    if (result.servingBackend.has_value())
    {
        response["serving_backend"] = *result.servingBackend;
    }
    return response.dump();
}

} // namespace api
} // namespace treesem
