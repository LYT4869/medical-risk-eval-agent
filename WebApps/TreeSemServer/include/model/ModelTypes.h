#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace treesem
{
namespace model
{

inline constexpr std::size_t kPphInputDimension = 49;

struct SampleIndexInput
{
    std::int64_t sampleIndex;
};

struct PreprocessedFeaturesInput
{
    std::vector<double> values;
};

using ModelInput = std::variant<SampleIndexInput, PreprocessedFeaturesInput>;

struct PredictionValues
{
    int label;
    double positiveProbability;
    double confidence;
    int clusterId;
    double treeProbability;
    int treeLeafId;
};

struct ImportantFeature
{
    int index;
    std::string name;
    double standardizedValue;
    double treeImportance;
};

struct DecisionPathStep
{
    std::optional<int> nodeId;
    std::optional<int> featureIndex;
    std::optional<std::string> featureName;
    std::optional<std::string> comparisonOperator;
    std::optional<double> thresholdStandardized;
    std::optional<double> valueStandardized;
    std::optional<int> leafId;
};

struct ModelResult
{
    std::string modelName;
    std::string dataset;
    std::string inputSource;
    std::optional<std::int64_t> sampleIndex;
    PredictionValues prediction;
    std::vector<ImportantFeature> importantFeatures;
    std::vector<DecisionPathStep> decisionPath;
};

} // namespace model
} // namespace treesem
