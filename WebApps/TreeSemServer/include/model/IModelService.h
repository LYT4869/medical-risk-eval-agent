#pragma once

#include "model/ModelTypes.h"

namespace treesem
{
namespace model
{

class IModelService
{
public:
    virtual ~IModelService() = default;
    virtual ModelResult predict(const ModelInput& input) const = 0;
};

} // namespace model
} // namespace treesem
