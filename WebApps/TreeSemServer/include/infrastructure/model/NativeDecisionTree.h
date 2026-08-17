#pragma once

#include <utility>
#include <vector>

#include "infrastructure/model/FeaturePreprocessor.h"
#include "model/ModelTypes.h"

namespace treesem
{
namespace infrastructure
{

struct TreeEvaluation
{
    double probability;
    int leafId;
    std::vector<model::DecisionPathStep> path;
};

class NativeDecisionTree
{
public:
    NativeDecisionTree(
        const ServingBundle& bundle,
        const FeaturePreprocessor& preprocessor);

    TreeEvaluation evaluate(const std::vector<float>& standardized) const;
    std::vector<model::ImportantFeature> importantFeatures(
        const std::vector<float>& standardized,
        std::size_t limit = 5) const;

private:
    const ServingBundle& bundle_;
    const FeaturePreprocessor& preprocessor_;
};

} // namespace infrastructure
} // namespace treesem
