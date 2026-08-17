#include "infrastructure/model/NativeDecisionTree.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace treesem
{
namespace infrastructure
{

NativeDecisionTree::NativeDecisionTree(
    const ServingBundle& bundle,
    const FeaturePreprocessor& preprocessor)
    : bundle_(bundle)
    , preprocessor_(preprocessor)
{}

TreeEvaluation NativeDecisionTree::evaluate(
    const std::vector<float>& standardized) const
{
    if (standardized.size() != bundle_.features().size())
    {
        throw std::invalid_argument("tree input dimension is invalid");
    }
    TreeEvaluation result;
    int nodeId = bundle_.treeRoot();
    std::size_t visited = 0;
    while (true)
    {
        if (++visited > bundle_.treeNodes().size())
        {
            throw std::runtime_error("tree traversal exceeded node count");
        }
        const BundleTreeNode& node = bundle_.treeNodes().at(
            static_cast<std::size_t>(nodeId));
        if (node.leaf)
        {
            model::DecisionPathStep leaf;
            leaf.leafId = node.id;
            result.path.push_back(std::move(leaf));
            result.probability = node.value;
            result.leafId = node.id;
            return result;
        }
        const float value = standardized.at(
            static_cast<std::size_t>(node.featureIndex));
        const bool left = static_cast<double>(value) <= node.threshold;
        const BundleFeature& feature = bundle_.features().at(
            static_cast<std::size_t>(node.featureIndex));
        model::DecisionPathStep step;
        step.nodeId = node.id;
        step.featureIndex = node.featureIndex;
        step.featureName = feature.name;
        step.featureDisplayName = feature.displayName;
        step.comparisonOperator = left ? "<=" : ">";
        step.thresholdStandardized = node.threshold;
        step.valueStandardized = static_cast<double>(value);
        step.thresholdOriginal = preprocessor_.originalValue(
            node.featureIndex, node.threshold);
        step.valueOriginal = preprocessor_.originalValue(
            node.featureIndex, static_cast<double>(value));
        step.unit = feature.unit;
        result.path.push_back(std::move(step));
        nodeId = left ? node.left : node.right;
    }
}

std::vector<model::ImportantFeature> NativeDecisionTree::importantFeatures(
    const std::vector<float>& standardized,
    std::size_t limit) const
{
    if (standardized.size() != bundle_.features().size())
    {
        throw std::invalid_argument("tree input dimension is invalid");
    }
    std::vector<int> indices(bundle_.features().size());
    std::iota(indices.begin(), indices.end(), 0);
    const std::vector<double>& importances = bundle_.featureImportances();
    std::sort(indices.begin(), indices.end(), [&](int left, int right) {
        const double leftValue = importances.at(static_cast<std::size_t>(left));
        const double rightValue = importances.at(static_cast<std::size_t>(right));
        return leftValue == rightValue ? left < right : leftValue > rightValue;
    });
    if (indices.size() > limit)
    {
        indices.resize(limit);
    }
    std::vector<model::ImportantFeature> result;
    result.reserve(indices.size());
    for (int index : indices)
    {
        const BundleFeature& feature = bundle_.features().at(
            static_cast<std::size_t>(index));
        model::ImportantFeature item;
        item.index = index;
        item.name = feature.name;
        item.displayName = feature.displayName;
        item.standardizedValue = standardized.at(static_cast<std::size_t>(index));
        item.originalValue = preprocessor_.originalValue(
            index, item.standardizedValue);
        item.unit = feature.unit;
        item.treeImportance = importances.at(static_cast<std::size_t>(index));
        result.push_back(std::move(item));
    }
    return result;
}

} // namespace infrastructure
} // namespace treesem
