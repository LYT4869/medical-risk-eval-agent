#pragma once

#include <filesystem>
#include <memory>

#include "infrastructure/model/FeaturePreprocessor.h"
#include "infrastructure/model/NativeDecisionTree.h"
#include "model/IModelService.h"

namespace treesem
{
namespace infrastructure
{

class OnnxTreeSemModelService final : public model::IModelService
{
public:
    explicit OnnxTreeSemModelService(const std::filesystem::path& bundleDirectory);
    ~OnnxTreeSemModelService() override;

    model::ModelResult predict(const model::ModelInput& input) const override;
    const ServingBundle& bundle() const noexcept;

private:
    class Impl;

    ServingBundle bundle_;
    FeaturePreprocessor preprocessor_;
    NativeDecisionTree tree_;
    std::unique_ptr<Impl> impl_;
};

} // namespace infrastructure
} // namespace treesem
