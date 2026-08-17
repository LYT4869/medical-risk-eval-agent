#include "serialization/ModelResultJsonCodec.h"

#include <cmath>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace treesem
{
namespace serialization
{
namespace
{

using Json = nlohmann::json;

template<typename T>
std::optional<T> optionalValue(const Json& object, const char* name)
{
    if (!object.contains(name) || object.at(name).is_null()) return std::nullopt;
    return object.at(name).get<T>();
}

bool finiteProbability(double value)
{
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

void validateDecoded(const model::ModelResult& result)
{
    if (result.modelName.empty() || result.dataset.empty() ||
        result.inputSource.empty() ||
        (result.sampleIndex.has_value() && *result.sampleIndex < 0) ||
        (result.prediction.label != 0 && result.prediction.label != 1) ||
        !finiteProbability(result.prediction.positiveProbability) ||
        !finiteProbability(result.prediction.confidence) ||
        !finiteProbability(result.prediction.treeProbability) ||
        result.prediction.clusterId < 0 || result.prediction.treeLeafId < 0)
    {
        throw std::invalid_argument("stored model result values are invalid");
    }
    std::set<int> featureIndices;
    for (const auto& feature : result.importantFeatures)
    {
        if (feature.index < 0 || feature.index >= 49 || feature.name.empty() ||
            !std::isfinite(feature.standardizedValue) ||
            !std::isfinite(feature.treeImportance) ||
            (feature.originalValue.has_value() &&
             !std::isfinite(*feature.originalValue)) ||
            !featureIndices.insert(feature.index).second)
        {
            throw std::invalid_argument("stored important feature is invalid");
        }
    }
    for (const auto& step : result.decisionPath)
    {
        const bool leaf = step.leafId.has_value();
        if (leaf)
        {
            if (*step.leafId < 0 || step.nodeId.has_value() ||
                step.featureIndex.has_value() || step.featureName.has_value() ||
                step.comparisonOperator.has_value() ||
                step.thresholdStandardized.has_value() ||
                step.valueStandardized.has_value())
            {
                throw std::invalid_argument("stored decision leaf is invalid");
            }
            continue;
        }
        if (!step.nodeId.has_value() || *step.nodeId < 0 ||
            !step.featureIndex.has_value() || *step.featureIndex < 0 ||
            *step.featureIndex >= 49 || !step.featureName.has_value() ||
            !step.comparisonOperator.has_value() ||
            (*step.comparisonOperator != "<=" && *step.comparisonOperator != ">") ||
            !step.thresholdStandardized.has_value() ||
            !std::isfinite(*step.thresholdStandardized) ||
            !step.valueStandardized.has_value() ||
            !std::isfinite(*step.valueStandardized) ||
            (step.thresholdOriginal.has_value() &&
             !std::isfinite(*step.thresholdOriginal)) ||
            (step.valueOriginal.has_value() &&
             !std::isfinite(*step.valueOriginal)))
        {
            throw std::invalid_argument("stored decision branch is invalid");
        }
    }
}

} // namespace

std::string ModelResultJsonCodec::encode(const model::ModelResult& result)
{
    Json features = Json::array();
    for (const auto& feature : result.importantFeatures)
    {
        features.push_back({
            {"index", feature.index},
            {"name", feature.name},
            {"standardized_value", feature.standardizedValue},
            {"tree_importance", feature.treeImportance},
            {"display_name", feature.displayName.has_value()
                ? Json(*feature.displayName) : Json(nullptr)},
            {"original_value", feature.originalValue.has_value()
                ? Json(*feature.originalValue) : Json(nullptr)},
            {"unit", feature.unit.has_value() ? Json(*feature.unit) : Json(nullptr)}});
    }
    Json path = Json::array();
    for (const auto& step : result.decisionPath)
    {
        if (step.leafId.has_value())
        {
            path.push_back({{"leaf_id", *step.leafId}});
        }
        else
        {
            path.push_back({
                {"node_id", step.nodeId.has_value() ? Json(*step.nodeId) : Json(nullptr)},
                {"feature_index", step.featureIndex.has_value()
                    ? Json(*step.featureIndex) : Json(nullptr)},
                {"feature_name", step.featureName.has_value()
                    ? Json(*step.featureName) : Json(nullptr)},
                {"operator", step.comparisonOperator.has_value()
                    ? Json(*step.comparisonOperator) : Json(nullptr)},
                {"threshold_standardized", step.thresholdStandardized.has_value()
                    ? Json(*step.thresholdStandardized) : Json(nullptr)},
                {"value_standardized", step.valueStandardized.has_value()
                    ? Json(*step.valueStandardized) : Json(nullptr)},
                {"feature_display_name", step.featureDisplayName.has_value()
                    ? Json(*step.featureDisplayName) : Json(nullptr)},
                {"threshold_original", step.thresholdOriginal.has_value()
                    ? Json(*step.thresholdOriginal) : Json(nullptr)},
                {"value_original", step.valueOriginal.has_value()
                    ? Json(*step.valueOriginal) : Json(nullptr)},
                {"unit", step.unit.has_value() ? Json(*step.unit) : Json(nullptr)}});
        }
    }
    return Json{
        {"model", result.modelName},
        {"model_version", result.modelVersion.has_value()
            ? Json(*result.modelVersion) : Json(nullptr)},
        {"serving_backend", result.servingBackend.has_value()
            ? Json(*result.servingBackend) : Json(nullptr)},
        {"dataset", result.dataset},
        {"input_source", result.inputSource},
        {"sample_index", result.sampleIndex.has_value()
            ? Json(*result.sampleIndex) : Json(nullptr)},
        {"prediction", {
            {"label", result.prediction.label},
            {"positive_probability", result.prediction.positiveProbability},
            {"confidence", result.prediction.confidence},
            {"cluster_id", result.prediction.clusterId},
            {"tree_probability", result.prediction.treeProbability},
            {"tree_leaf_id", result.prediction.treeLeafId}}},
        {"important_features", std::move(features)},
        {"decision_path", std::move(path)}}.dump();
}

model::ModelResult ModelResultJsonCodec::decode(const std::string& value)
{
    const Json root = Json::parse(value);
    if (!root.is_object()) throw std::invalid_argument("stored model result is invalid");
    model::ModelResult result;
    result.modelName = root.at("model").get<std::string>();
    result.modelVersion = optionalValue<std::string>(root, "model_version");
    result.servingBackend = optionalValue<std::string>(root, "serving_backend");
    result.dataset = root.at("dataset").get<std::string>();
    result.inputSource = root.at("input_source").get<std::string>();
    result.sampleIndex = optionalValue<std::int64_t>(root, "sample_index");
    const Json& prediction = root.at("prediction");
    result.prediction = {
        prediction.at("label").get<int>(),
        prediction.at("positive_probability").get<double>(),
        prediction.at("confidence").get<double>(),
        prediction.at("cluster_id").get<int>(),
        prediction.at("tree_probability").get<double>(),
        prediction.at("tree_leaf_id").get<int>()};
    for (const auto& item : root.at("important_features"))
    {
        model::ImportantFeature feature;
        feature.index = item.at("index").get<int>();
        feature.name = item.at("name").get<std::string>();
        feature.standardizedValue = item.at("standardized_value").get<double>();
        feature.treeImportance = item.at("tree_importance").get<double>();
        feature.displayName = optionalValue<std::string>(item, "display_name");
        feature.originalValue = optionalValue<double>(item, "original_value");
        feature.unit = optionalValue<std::string>(item, "unit");
        result.importantFeatures.push_back(std::move(feature));
    }
    for (const auto& item : root.at("decision_path"))
    {
        model::DecisionPathStep step;
        if (item.contains("leaf_id"))
        {
            step.leafId = item.at("leaf_id").get<int>();
        }
        else
        {
            step.nodeId = optionalValue<int>(item, "node_id");
            step.featureIndex = optionalValue<int>(item, "feature_index");
            step.featureName = optionalValue<std::string>(item, "feature_name");
            step.comparisonOperator = optionalValue<std::string>(item, "operator");
            step.thresholdStandardized = optionalValue<double>(item, "threshold_standardized");
            step.valueStandardized = optionalValue<double>(item, "value_standardized");
            step.featureDisplayName = optionalValue<std::string>(item, "feature_display_name");
            step.thresholdOriginal = optionalValue<double>(item, "threshold_original");
            step.valueOriginal = optionalValue<double>(item, "value_original");
            step.unit = optionalValue<std::string>(item, "unit");
        }
        result.decisionPath.push_back(std::move(step));
    }
    validateDecoded(result);
    return result;
}

} // namespace serialization
} // namespace treesem
