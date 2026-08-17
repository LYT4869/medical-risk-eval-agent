#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace treesem
{
namespace infrastructure
{

struct BundleFeature
{
    int index;
    std::string name;
    std::string displayName;
    std::optional<std::string> unit;
    double mean;
    double scale;
};

struct BundleTreeNode
{
    int id;
    bool leaf;
    int left{-1};
    int right{-1};
    int featureIndex{-1};
    double threshold{0.0};
    double value{0.0};
};

class ServingBundle
{
public:
    explicit ServingBundle(const std::filesystem::path& directory,
                           bool requireOnnx);

    const std::filesystem::path& directory() const noexcept;
    const std::string& modelVersion() const noexcept;
    int inputDimension() const noexcept;
    int classCount() const noexcept;
    int clusterCount() const noexcept;
    const std::vector<BundleFeature>& features() const noexcept;
    const std::vector<BundleTreeNode>& treeNodes() const noexcept;
    const std::vector<double>& featureImportances() const noexcept;
    int treeRoot() const noexcept;
    const std::filesystem::path& onnxPath() const noexcept;
    const std::string& onnxInputName() const noexcept;
    const std::vector<std::string>& onnxOutputNames() const noexcept;
    bool hasReferenceInputs() const noexcept;
    std::size_t referenceRows() const noexcept;
    std::vector<float> referenceInput(std::size_t index) const;

private:
    std::filesystem::path directory_;
    std::string modelVersion_;
    int inputDimension_{0};
    int classCount_{0};
    int clusterCount_{0};
    std::vector<BundleFeature> features_;
    std::vector<BundleTreeNode> treeNodes_;
    std::vector<double> featureImportances_;
    int treeRoot_{0};
    std::filesystem::path onnxPath_;
    std::string onnxInputName_;
    std::vector<std::string> onnxOutputNames_;
    std::vector<float> referenceInputs_;
    std::size_t referenceRows_{0};
};

} // namespace infrastructure
} // namespace treesem
