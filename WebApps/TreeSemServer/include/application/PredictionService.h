#pragma once

#include "model/IModelService.h"

namespace treesem
{
namespace application
{

class PredictionService
{
public:
    explicit PredictionService(const model::IModelService& modelService);

    model::ModelResult predict(const model::ModelInput& input) const;

private:
    const model::IModelService& modelService_;
};

} // namespace application
} // namespace treesem
