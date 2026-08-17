#include "infrastructure/model/OnnxTreeSemModelService.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "model/ModelException.h"

namespace treesem
{
namespace infrastructure
{
namespace
{

struct ResolvedInput
{
    std::vector<float> features;
    std::string source;
    std::optional<std::int64_t> sampleIndex;
};

} // namespace

class OnnxTreeSemModelService::Impl
{
public:
    explicit Impl(const ServingBundle& bundle)
        : environment_(ORT_LOGGING_LEVEL_WARNING, "treeSem")
    {
        options_.SetIntraOpNumThreads(1);
        options_.SetInterOpNumThreads(1);
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(
            environment_, bundle.onnxPath().c_str(), options_);
        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 2)
        {
            throw std::runtime_error("ONNX input/output count is invalid");
        }
        const auto inputName = session_->GetInputNameAllocated(0, allocator);
        if (std::string(inputName.get()) != bundle.onnxInputName())
        {
            throw std::runtime_error("ONNX input name is invalid");
        }
        for (std::size_t index = 0; index < bundle.onnxOutputNames().size(); ++index)
        {
            const auto name = session_->GetOutputNameAllocated(index, allocator);
            if (std::string(name.get()) != bundle.onnxOutputNames().at(index))
            {
                throw std::runtime_error("ONNX output name is invalid");
            }
        }
        const Ort::TypeInfo inputType = session_->GetInputTypeInfo(0);
        const auto inputTensor = inputType.GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> inputShape = inputTensor.GetShape();
        if (inputTensor.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            inputShape.size() != 2 || inputShape[0] != -1 ||
            inputShape[1] != bundle.inputDimension())
        {
            throw std::runtime_error("ONNX input shape is invalid");
        }
        for (std::size_t index = 0; index < 2; ++index)
        {
            const Ort::TypeInfo outputType = session_->GetOutputTypeInfo(index);
            const auto outputTensor = outputType.GetTensorTypeAndShapeInfo();
            const std::vector<std::int64_t> shape = outputTensor.GetShape();
            const std::int64_t expected = index == 0
                ? bundle.classCount()
                : bundle.clusterCount();
            if (outputTensor.GetElementType() !=
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                shape.size() != 2 || shape[0] != -1 || shape[1] != expected)
            {
                throw std::runtime_error("ONNX output shape is invalid");
            }
        }
    }

    std::pair<std::vector<float>, std::vector<float>> run(
        const ServingBundle& bundle,
        const std::vector<float>& features) const
    {
        std::array<std::int64_t, 2> shape{
            1, static_cast<std::int64_t>(bundle.inputDimension())};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory,
            const_cast<float*>(features.data()),
            features.size(),
            shape.data(),
            shape.size());
        const std::array<const char*, 1> inputNames{
            bundle.onnxInputName().c_str()};
        const std::array<const char*, 2> outputNames{
            bundle.onnxOutputNames().at(0).c_str(),
            bundle.onnxOutputNames().at(1).c_str()};
        std::vector<Ort::Value> outputs = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &input,
            inputNames.size(),
            outputNames.data(),
            outputNames.size());
        if (outputs.size() != 2 || !outputs[0].IsTensor() || !outputs[1].IsTensor())
        {
            throw std::runtime_error("ONNX outputs are missing");
        }
        const auto probabilityShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto clusterShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (probabilityShape != std::vector<std::int64_t>{1, bundle.classCount()} ||
            clusterShape != std::vector<std::int64_t>{1, bundle.clusterCount()})
        {
            throw std::runtime_error("ONNX output shape is invalid");
        }
        const float* probabilities = outputs[0].GetTensorData<float>();
        const float* clusters = outputs[1].GetTensorData<float>();
        return {
            std::vector<float>(probabilities, probabilities + bundle.classCount()),
            std::vector<float>(clusters, clusters + bundle.clusterCount())};
    }

private:
    Ort::Env environment_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
};

OnnxTreeSemModelService::OnnxTreeSemModelService(
    const std::filesystem::path& bundleDirectory)
    : bundle_(bundleDirectory, true)
    , preprocessor_(bundle_)
    , tree_(bundle_, preprocessor_)
    , impl_(std::make_unique<Impl>(bundle_))
{}

OnnxTreeSemModelService::~OnnxTreeSemModelService() = default;

model::ModelResult OnnxTreeSemModelService::predict(
    const model::ModelInput& input) const
{
    ResolvedInput resolved;
    try
    {
        std::visit(
            [&](const auto& value) {
                using Input = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Input, model::SampleIndexInput>)
                {
                    if (value.sampleIndex < 0 ||
                        static_cast<std::uint64_t>(value.sampleIndex) >=
                            bundle_.referenceRows())
                    {
                        throw model::ModelException(
                            model::ModelException::Kind::InvalidInput,
                            "sample_index is unavailable in this serving bundle");
                    }
                    resolved.features = bundle_.referenceInput(
                        static_cast<std::size_t>(value.sampleIndex));
                    resolved.source = "pph_reference_dataset";
                    resolved.sampleIndex = value.sampleIndex;
                }
                else if constexpr (
                    std::is_same_v<Input, model::PreprocessedFeaturesInput>)
                {
                    resolved.features = preprocessor_.validateStandardized(value.values);
                    resolved.source = "request_preprocessed";
                }
                else
                {
                    resolved.features = preprocessor_.standardizeRaw(value.values);
                    resolved.source = "request_raw";
                }
            },
            input);
    }
    catch (const model::ModelException&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw model::ModelException(
            model::ModelException::Kind::InvalidInput, error.what());
    }

    std::vector<float> probabilities;
    std::vector<float> clusterLogits;
    try
    {
        std::tie(probabilities, clusterLogits) = impl_->run(bundle_, resolved.features);
    }
    catch (const std::exception& error)
    {
        throw model::ModelException(
            model::ModelException::Kind::InferenceFailure, error.what());
    }
    if (!std::all_of(
            probabilities.begin(), probabilities.end(),
            [](float value) { return std::isfinite(value); }) ||
        !std::all_of(
            clusterLogits.begin(), clusterLogits.end(),
            [](float value) { return std::isfinite(value); }))
    {
        throw model::ModelException(
            model::ModelException::Kind::InferenceFailure,
            "ONNX returned a non-finite value");
    }
    const int label = static_cast<int>(std::distance(
        probabilities.begin(),
        std::max_element(probabilities.begin(), probabilities.end())));
    const int clusterId = static_cast<int>(std::distance(
        clusterLogits.begin(),
        std::max_element(clusterLogits.begin(), clusterLogits.end())));
    const TreeEvaluation treeResult = tree_.evaluate(resolved.features);

    model::ModelResult result;
    result.modelName = "treeSem";
    result.modelVersion = bundle_.modelVersion();
    result.servingBackend = "onnx";
    result.dataset = "pph";
    result.inputSource = resolved.source;
    result.sampleIndex = resolved.sampleIndex;
    result.prediction = {
        label,
        probabilities.at(1),
        probabilities.at(static_cast<std::size_t>(label)),
        clusterId,
        treeResult.probability,
        treeResult.leafId};
    result.importantFeatures = tree_.importantFeatures(resolved.features);
    result.decisionPath = treeResult.path;
    return result;
}

const ServingBundle& OnnxTreeSemModelService::bundle() const noexcept
{
    return bundle_;
}

} // namespace infrastructure
} // namespace treesem
