#include "application/PredictionService.h"

namespace treesem
{
namespace application
{

PredictionService::PredictionService(const model::IModelService& modelService)
    : modelService_(modelService)
{}

model::ModelResult PredictionService::predict(const model::ModelInput& input) const
{
    return modelService_.predict(input);
}

} // namespace application
} // namespace treesem
