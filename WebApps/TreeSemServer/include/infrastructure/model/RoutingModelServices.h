#pragma once

#include "model/IModelService.h"

namespace treesem
{
namespace infrastructure
{

class FallbackModelService final : public model::IModelService
{
public:
    FallbackModelService(
        const model::IModelService& primary,
        const model::IModelService& fallback);
    model::ModelResult predict(const model::ModelInput& input) const override;

private:
    const model::IModelService& primary_;
    const model::IModelService& fallback_;
};

class ShadowModelService final : public model::IModelService
{
public:
    ShadowModelService(
        const model::IModelService& primary,
        const model::IModelService& shadow);
    model::ModelResult predict(const model::ModelInput& input) const override;

private:
    const model::IModelService& primary_;
    const model::IModelService& shadow_;
};

} // namespace infrastructure
} // namespace treesem
