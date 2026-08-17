#pragma once

#include <vector>

#include "infrastructure/model/ServingBundle.h"

namespace treesem
{
namespace infrastructure
{

class FeaturePreprocessor
{
public:
    explicit FeaturePreprocessor(const ServingBundle& bundle);

    std::vector<float> standardizeRaw(const std::vector<double>& raw) const;
    std::vector<float> validateStandardized(
        const std::vector<double>& standardized) const;
    double originalValue(int featureIndex, double standardized) const;

private:
    const ServingBundle& bundle_;
};

} // namespace infrastructure
} // namespace treesem
