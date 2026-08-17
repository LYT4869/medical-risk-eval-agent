#include "infrastructure/model/FeaturePreprocessor.h"

#include <cmath>

#include "model/ModelException.h"

namespace treesem
{
namespace infrastructure
{

FeaturePreprocessor::FeaturePreprocessor(const ServingBundle& bundle)
    : bundle_(bundle)
{}

std::vector<float> FeaturePreprocessor::standardizeRaw(
    const std::vector<double>& raw) const
{
    if (raw.size() != bundle_.features().size())
    {
        throw model::ModelException(
            model::ModelException::Kind::InvalidInput,
            "raw feature dimension is invalid");
    }
    std::vector<float> result;
    result.reserve(raw.size());
    for (std::size_t index = 0; index < raw.size(); ++index)
    {
        const BundleFeature& feature = bundle_.features().at(index);
        const double standardized = (raw[index] - feature.mean) / feature.scale;
        if (!std::isfinite(raw[index]) || !std::isfinite(standardized))
        {
            throw model::ModelException(
                model::ModelException::Kind::InvalidInput,
                "raw feature preprocessing produced a non-finite value");
        }
        result.push_back(static_cast<float>(standardized));
    }
    return result;
}

std::vector<float> FeaturePreprocessor::validateStandardized(
    const std::vector<double>& standardized) const
{
    if (standardized.size() != bundle_.features().size())
    {
        throw model::ModelException(
            model::ModelException::Kind::InvalidInput,
            "preprocessed feature dimension is invalid");
    }
    std::vector<float> result;
    result.reserve(standardized.size());
    for (double value : standardized)
    {
        const float converted = static_cast<float>(value);
        if (!std::isfinite(value) || !std::isfinite(converted))
        {
            throw model::ModelException(
                model::ModelException::Kind::InvalidInput,
                "preprocessed feature value is non-finite");
        }
        result.push_back(converted);
    }
    return result;
}

double FeaturePreprocessor::originalValue(
    int featureIndex,
    double standardized) const
{
    if (featureIndex < 0 ||
        featureIndex >= static_cast<int>(bundle_.features().size()))
    {
        throw std::out_of_range("feature index is out of range");
    }
    const BundleFeature& feature = bundle_.features().at(
        static_cast<std::size_t>(featureIndex));
    const double original = standardized * feature.scale + feature.mean;
    if (!std::isfinite(original))
    {
        throw std::runtime_error("inverse preprocessing produced a non-finite value");
    }
    return original;
}

} // namespace infrastructure
} // namespace treesem
