#include "infrastructure/model/RemoteTreeSemModelService.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include <nlohmann/json.hpp>

#include "client/PythonModelClient.h"
#include "model/ModelException.h"

namespace treesem
{
namespace infrastructure
{
namespace
{

using Json = nlohmann::json;

[[noreturn]] void invalidResponse(const std::string& reason)
{
    throw model::ModelException(model::ModelException::Kind::InvalidResponse, reason);
}

double finiteNumber(const Json& value, const char* field)
{
    if (!value.is_number())
    {
        invalidResponse(std::string(field) + " must be numeric");
    }
    const double result = value.get<double>();
    if (!std::isfinite(result))
    {
        invalidResponse(std::string(field) + " must be finite");
    }
    return result;
}

int nonNegativeInteger(const Json& value, const char* field)
{
    if (!value.is_number_integer())
    {
        invalidResponse(std::string(field) + " must be an integer");
    }
    if (value.is_number_unsigned())
    {
        const std::uint64_t result = value.get<std::uint64_t>();
        if (result > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        {
            invalidResponse(std::string(field) + " is out of range");
        }
        return static_cast<int>(result);
    }
    const std::int64_t result = value.get<std::int64_t>();
    if (result < 0 || result > std::numeric_limits<int>::max())
    {
        invalidResponse(std::string(field) + " is out of range");
    }
    return static_cast<int>(result);
}

std::string nonEmptyString(const Json& value, const char* field)
{
    if (!value.is_string())
    {
        invalidResponse(std::string(field) + " must be a string");
    }
    std::string result = value.get<std::string>();
    if (result.empty())
    {
        invalidResponse(std::string(field) + " must not be empty");
    }
    return result;
}

std::string serializeInput(const model::ModelInput& input)
{
    return std::visit(
        [](const auto& value) -> std::string {
            using Input = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Input, model::SampleIndexInput>)
            {
                if (value.sampleIndex < 0)
                {
                    throw model::ModelException(
                        model::ModelException::Kind::InvalidInput,
                        "sample_index must be non-negative");
                }
                return Json{{"sample_index", value.sampleIndex}}.dump();
            }
            else
            {
                if (value.values.size() != model::kPphInputDimension)
                {
                    throw model::ModelException(
                        model::ModelException::Kind::InvalidInput,
                        "preprocessed_features has an invalid dimension");
                }
                for (double feature : value.values)
                {
                    if (!std::isfinite(feature))
                    {
                        throw model::ModelException(
                            model::ModelException::Kind::InvalidInput,
                            "preprocessed_features must contain finite values");
                    }
                }
                return Json{{"preprocessed_features", value.values}}.dump();
            }
        },
        input);
}

model::ModelResult parseResult(const std::string& body)
{
    Json root;
    try
    {
        root = Json::parse(body);
        if (!root.is_object())
        {
            invalidResponse("model response must be an object");
        }

        model::ModelResult result;
        result.modelName = nonEmptyString(root.at("model"), "model");
        result.dataset = nonEmptyString(root.at("dataset"), "dataset");
        result.inputSource = nonEmptyString(root.at("input_source"), "input_source");

        const Json& sampleIndex = root.at("sample_index");
        if (!sampleIndex.is_null())
        {
            if (!sampleIndex.is_number_integer())
            {
                invalidResponse("sample_index must be an integer or null");
            }
            std::int64_t index = 0;
            if (sampleIndex.is_number_unsigned())
            {
                const std::uint64_t unsignedIndex =
                    sampleIndex.get<std::uint64_t>();
                if (unsignedIndex > static_cast<std::uint64_t>(
                                        std::numeric_limits<std::int64_t>::max()))
                {
                    invalidResponse("sample_index is out of range");
                }
                index = static_cast<std::int64_t>(unsignedIndex);
            }
            else
            {
                index = sampleIndex.get<std::int64_t>();
            }
            if (index < 0)
            {
                invalidResponse("sample_index must be non-negative");
            }
            result.sampleIndex = index;
        }

        const Json& prediction = root.at("prediction");
        if (!prediction.is_object())
        {
            invalidResponse("prediction must be an object");
        }
        result.prediction = model::PredictionValues{
            nonNegativeInteger(prediction.at("label"), "prediction.label"),
            finiteNumber(
                prediction.at("positive_probability"),
                "prediction.positive_probability"),
            finiteNumber(prediction.at("confidence"), "prediction.confidence"),
            nonNegativeInteger(prediction.at("cluster_id"), "prediction.cluster_id"),
            finiteNumber(
                prediction.at("tree_probability"),
                "prediction.tree_probability"),
            nonNegativeInteger(
                prediction.at("tree_leaf_id"),
                "prediction.tree_leaf_id"),
        };
        if ((result.prediction.label != 0 && result.prediction.label != 1) ||
            result.prediction.positiveProbability < 0.0 ||
            result.prediction.positiveProbability > 1.0 ||
            result.prediction.confidence < 0.0 ||
            result.prediction.confidence > 1.0 ||
            result.prediction.treeProbability < 0.0 ||
            result.prediction.treeProbability > 1.0)
        {
            invalidResponse("prediction values are out of range");
        }

        const Json& importantFeatures = root.at("important_features");
        if (!importantFeatures.is_array())
        {
            invalidResponse("important_features must be an array");
        }
        for (const Json& feature : importantFeatures)
        {
            if (!feature.is_object())
            {
                invalidResponse("important feature must be an object");
            }
            const int featureIndex = nonNegativeInteger(
                feature.at("index"), "important_features.index");
            if (featureIndex >= static_cast<int>(model::kPphInputDimension))
            {
                invalidResponse("important feature index is out of range");
            }
            result.importantFeatures.push_back(model::ImportantFeature{
                featureIndex,
                nonEmptyString(feature.at("name"), "important_features.name"),
                finiteNumber(
                    feature.at("standardized_value"),
                    "important_features.standardized_value"),
                finiteNumber(
                    feature.at("tree_importance"),
                    "important_features.tree_importance"),
            });
        }

        const Json& decisionPath = root.at("decision_path");
        if (!decisionPath.is_array() || decisionPath.empty())
        {
            invalidResponse("decision_path must be a non-empty array");
        }
        bool sawLeaf = false;
        for (std::size_t index = 0; index < decisionPath.size(); ++index)
        {
            const Json& step = decisionPath[index];
            if (!step.is_object())
            {
                invalidResponse("decision path step must be an object");
            }
            model::DecisionPathStep parsed;
            if (step.contains("leaf_id"))
            {
                if (step.size() != 1 || index + 1 != decisionPath.size() || sawLeaf)
                {
                    invalidResponse("decision path leaf must be the final step");
                }
                parsed.leafId = nonNegativeInteger(step.at("leaf_id"), "leaf_id");
                sawLeaf = true;
            }
            else
            {
                parsed.nodeId = nonNegativeInteger(step.at("node_id"), "node_id");
                parsed.featureIndex = nonNegativeInteger(
                    step.at("feature_index"), "feature_index");
                if (*parsed.featureIndex >=
                    static_cast<int>(model::kPphInputDimension))
                {
                    invalidResponse("decision path feature index is out of range");
                }
                parsed.featureName = nonEmptyString(
                    step.at("feature_name"), "feature_name");
                parsed.comparisonOperator = nonEmptyString(
                    step.at("operator"), "operator");
                if (*parsed.comparisonOperator != "<=" &&
                    *parsed.comparisonOperator != ">")
                {
                    invalidResponse("operator must be <= or >");
                }
                parsed.thresholdStandardized = finiteNumber(
                    step.at("threshold_standardized"), "threshold_standardized");
                parsed.valueStandardized = finiteNumber(
                    step.at("value_standardized"), "value_standardized");
            }
            result.decisionPath.push_back(std::move(parsed));
        }
        if (!sawLeaf)
        {
            invalidResponse("decision_path must end with a leaf");
        }
        if (*result.decisionPath.back().leafId != result.prediction.treeLeafId)
        {
            invalidResponse("decision path leaf does not match prediction");
        }
        return result;
    }
    catch (const model::ModelException&)
    {
        throw;
    }
    catch (const Json::exception& exception)
    {
        invalidResponse(std::string("invalid model response: ") + exception.what());
    }
}

model::ModelException translateClientException(
    const client::ModelAdapterException& exception)
{
    switch (exception.kind())
    {
    case client::ModelAdapterException::Kind::Timeout:
        return model::ModelException(model::ModelException::Kind::Timeout, exception.what());
    case client::ModelAdapterException::Kind::InvalidResponse:
        return model::ModelException(
            model::ModelException::Kind::InvalidResponse, exception.what());
    case client::ModelAdapterException::Kind::Unavailable:
        return model::ModelException(
            model::ModelException::Kind::Unavailable, exception.what());
    }
    return model::ModelException(
        model::ModelException::Kind::Unavailable, "unknown model client failure");
}

} // namespace

RemoteTreeSemModelService::RemoteTreeSemModelService(
    const client::IModelAdapterClient& client)
    : client_(client)
{}

model::ModelResult RemoteTreeSemModelService::predict(
    const model::ModelInput& input) const
{
    try
    {
        client::ModelAdapterResponse response = client_.predict(serializeInput(input));
        if (response.statusCode == 400)
        {
            throw model::ModelException(
                model::ModelException::Kind::InvalidInput,
                "model adapter rejected the request");
        }
        if (response.statusCode >= 500)
        {
            throw model::ModelException(
                model::ModelException::Kind::DownstreamFailure,
                "model adapter returned a server error");
        }
        if (response.statusCode != 200)
        {
            throw model::ModelException(
                model::ModelException::Kind::InvalidResponse,
                "model adapter returned an unsupported HTTP status");
        }
        return parseResult(response.body);
    }
    catch (const client::ModelAdapterException& exception)
    {
        throw translateClientException(exception);
    }
}

} // namespace infrastructure
} // namespace treesem
