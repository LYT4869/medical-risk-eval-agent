#pragma once

#include "model/IModelService.h"
#include "observability/MetricsRegistry.h"
#include <memory>

namespace treesem
{
namespace infrastructure
{

class FallbackModelService final : public model::IModelService
{
public:
    FallbackModelService(
        const model::IModelService& primary,
        const model::IModelService& fallback,
        std::shared_ptr<http::observability::MetricsRegistry> metrics = {});
    model::ModelResult predict(const model::ModelInput& input) const override;

private:
    const model::IModelService& primary_;
    const model::IModelService& fallback_;
    std::shared_ptr<http::observability::MetricsRegistry> metrics_;
};

class ShadowModelService final : public model::IModelService
{
public:
    ShadowModelService(
        const model::IModelService& primary,
        const model::IModelService& shadow,
        std::shared_ptr<http::observability::MetricsRegistry> metrics = {});
    model::ModelResult predict(const model::ModelInput& input) const override;

private:
    const model::IModelService& primary_;
    const model::IModelService& shadow_;
    std::shared_ptr<http::observability::MetricsRegistry> metrics_;
};

class InstrumentedModelService final : public model::IModelService
{
public:
    InstrumentedModelService(
        const model::IModelService& delegate,
        std::shared_ptr<http::observability::MetricsRegistry> metrics);
    model::ModelResult predict(const model::ModelInput& input) const override;
private:
    const model::IModelService& delegate_;
    std::shared_ptr<http::observability::MetricsRegistry> metrics_;
};

} // namespace infrastructure
} // namespace treesem
