#include "infrastructure/model/ServingBundle.h"

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include "model/ModelTypes.h"
#include "model/PphFeatureSchema.h"

namespace treesem
{
namespace infrastructure
{
namespace
{

using Json = nlohmann::json;

Json readJson(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("serving bundle JSON file is missing");
    }
    Json value;
    try
    {
        input >> value;
    }
    catch (const Json::exception&)
    {
        throw std::runtime_error("serving bundle JSON file is invalid");
    }
    if (!value.is_object())
    {
        throw std::runtime_error("serving bundle JSON root must be an object");
    }
    return value;
}

std::string sha256File(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("serving bundle file is missing");
    }
    EVP_MD_CTX* rawContext = EVP_MD_CTX_new();
    if (rawContext == nullptr)
    {
        throw std::runtime_error("failed to allocate checksum context");
    }
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
        rawContext, &EVP_MD_CTX_free);
    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    {
        throw std::runtime_error("failed to initialize checksum");
    }
    char buffer[64 * 1024];
    while (input)
    {
        input.read(buffer, sizeof(buffer));
        const std::streamsize count = input.gcount();
        if (count > 0 &&
            EVP_DigestUpdate(
                context.get(), buffer, static_cast<std::size_t>(count)) != 1)
        {
            throw std::runtime_error("failed to update checksum");
        }
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(context.get(), digest, &length) != 1)
    {
        throw std::runtime_error("failed to finalize checksum");
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < length; ++index)
    {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

double finiteNumber(const Json& value, const char* description)
{
    if (!value.is_number() || value.is_boolean())
    {
        throw std::runtime_error(std::string(description) + " must be numeric");
    }
    const double result = value.get<double>();
    if (!std::isfinite(result))
    {
        throw std::runtime_error(std::string(description) + " must be finite");
    }
    return result;
}

int exactInteger(const Json& value, const char* description)
{
    if (!value.is_number_integer())
    {
        throw std::runtime_error(std::string(description) + " must be an integer");
    }
    return value.get<int>();
}

std::filesystem::path checkedFile(
    const std::filesystem::path& directory,
    const Json& metadata)
{
    if (!metadata.is_object() || !metadata.contains("name") ||
        !metadata.at("name").is_string() || !metadata.contains("size") ||
        !metadata.at("size").is_number_unsigned() ||
        !metadata.contains("sha256") || !metadata.at("sha256").is_string())
    {
        throw std::runtime_error("serving bundle file metadata is invalid");
    }
    const std::filesystem::path name = metadata.at("name").get<std::string>();
    if (name.empty() || name.has_parent_path())
    {
        throw std::runtime_error("serving bundle file name is unsafe");
    }
    const std::filesystem::path path = directory / name;
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) != metadata.at("size").get<std::uintmax_t>() ||
        sha256File(path) != metadata.at("sha256").get<std::string>())
    {
        throw std::runtime_error("serving bundle checksum validation failed");
    }
    return path;
}

} // namespace

ServingBundle::ServingBundle(
    const std::filesystem::path& directory,
    bool requireOnnx)
    : directory_(std::filesystem::absolute(directory).lexically_normal())
{
    const Json manifest = readJson(directory_ / "manifest.json");
    if (manifest.value("bundle_schema_version", 0) != 1 ||
        manifest.value("model", "") != "treeSem" ||
        manifest.value("dataset", "") != "pph")
    {
        throw std::runtime_error("serving bundle identity is unsupported");
    }
    modelVersion_ = manifest.value("model_version", "");
    inputDimension_ = manifest.value("input_dim", 0);
    classCount_ = manifest.value("class_count", 0);
    clusterCount_ = manifest.value("cluster_count", 0);
    if (modelVersion_.empty() || inputDimension_ != 49 || classCount_ != 2 ||
        clusterCount_ <= 0)
    {
        throw std::runtime_error("serving bundle dimensions are invalid");
    }

    const auto filesIterator = manifest.find("files");
    if (filesIterator == manifest.end() || !filesIterator->is_object())
    {
        throw std::runtime_error("serving bundle files manifest is invalid");
    }
    std::unordered_map<std::string, std::filesystem::path> paths;
    for (auto iterator = filesIterator->begin(); iterator != filesIterator->end(); ++iterator)
    {
        paths.emplace(iterator.key(), checkedFile(directory_, iterator.value()));
    }
    for (const char* required : {"feature_schema", "preprocessing", "tree"})
    {
        if (paths.count(required) == 0)
        {
            throw std::runtime_error("serving bundle is missing a required file");
        }
    }

    const Json schema = readJson(paths.at("feature_schema"));
    const Json preprocessing = readJson(paths.at("preprocessing"));
    const Json& featureRows = schema.at("features");
    const Json& means = preprocessing.at("mean");
    const Json& scales = preprocessing.at("scale");
    if (!featureRows.is_array() || !means.is_array() || !scales.is_array() ||
        featureRows.size() != model::kPphInputDimension ||
        means.size() != model::kPphInputDimension ||
        scales.size() != model::kPphInputDimension)
    {
        throw std::runtime_error("serving bundle feature dimensions are invalid");
    }
    for (std::size_t index = 0; index < model::kPphInputDimension; ++index)
    {
        const Json& row = featureRows.at(index);
        if (!row.is_object() || exactInteger(row.at("index"), "feature index") !=
                                    static_cast<int>(index) ||
            !row.at("name").is_string() || !row.at("display_name").is_string() ||
            row.value("value_kind", "") != "numeric")
        {
            throw std::runtime_error("serving bundle feature schema is invalid");
        }
        BundleFeature feature;
        feature.index = static_cast<int>(index);
        feature.name = row.at("name").get<std::string>();
        feature.displayName = row.at("display_name").get<std::string>();
        if (feature.name != model::kPphFeatureNames.at(index) ||
            feature.displayName.empty())
        {
            throw std::runtime_error("serving bundle feature order is incompatible");
        }
        if (row.contains("unit") && !row.at("unit").is_null())
        {
            if (!row.at("unit").is_string())
            {
                throw std::runtime_error("serving bundle unit is invalid");
            }
            feature.unit = row.at("unit").get<std::string>();
        }
        feature.mean = finiteNumber(means.at(index), "feature mean");
        feature.scale = finiteNumber(scales.at(index), "feature scale");
        if (feature.scale <= 0.0)
        {
            throw std::runtime_error("serving bundle feature scale must be positive");
        }
        features_.push_back(std::move(feature));
    }

    const Json tree = readJson(paths.at("tree"));
    treeRoot_ = tree.value("root_node", -1);
    const Json& nodes = tree.at("nodes");
    const Json& importances = tree.at("global_feature_importance");
    if (!nodes.is_array() || nodes.empty() ||
        tree.value("node_count", 0) != static_cast<int>(nodes.size()) ||
        !importances.is_array() || importances.size() != model::kPphInputDimension ||
        treeRoot_ < 0 || treeRoot_ >= static_cast<int>(nodes.size()))
    {
        throw std::runtime_error("serving bundle tree dimensions are invalid");
    }
    for (const Json& value : importances)
    {
        featureImportances_.push_back(finiteNumber(value, "tree importance"));
    }
    for (std::size_t index = 0; index < nodes.size(); ++index)
    {
        const Json& row = nodes.at(index);
        BundleTreeNode node;
        node.id = exactInteger(row.at("id"), "tree node id");
        if (node.id != static_cast<int>(index) || !row.at("is_leaf").is_boolean())
        {
            throw std::runtime_error("serving bundle tree node is invalid");
        }
        node.leaf = row.at("is_leaf").get<bool>();
        node.value = finiteNumber(row.at("value"), "tree node value");
        if (!node.leaf)
        {
            node.left = exactInteger(row.at("left"), "tree left child");
            node.right = exactInteger(row.at("right"), "tree right child");
            node.featureIndex = exactInteger(row.at("feature_index"), "tree feature");
            node.threshold = finiteNumber(
                row.at("threshold_standardized"), "tree threshold");
            if (node.left < 0 || node.right < 0 ||
                node.left >= static_cast<int>(nodes.size()) ||
                node.right >= static_cast<int>(nodes.size()) ||
                node.featureIndex < 0 || node.featureIndex >= inputDimension_)
            {
                throw std::runtime_error("serving bundle tree edge is invalid");
            }
        }
        treeNodes_.push_back(node);
    }
    std::vector<int> visitState(treeNodes_.size(), 0);
    std::function<void(int)> visit = [&](int nodeId) {
        if (visitState.at(static_cast<std::size_t>(nodeId)) == 1)
        {
            throw std::runtime_error("serving bundle tree contains a cycle");
        }
        if (visitState.at(static_cast<std::size_t>(nodeId)) == 2)
        {
            return;
        }
        visitState.at(static_cast<std::size_t>(nodeId)) = 1;
        const BundleTreeNode& node = treeNodes_.at(static_cast<std::size_t>(nodeId));
        if (!node.leaf)
        {
            visit(node.left);
            visit(node.right);
        }
        visitState.at(static_cast<std::size_t>(nodeId)) = 2;
    };
    visit(treeRoot_);
    for (int state : visitState)
    {
        if (state != 2)
        {
            throw std::runtime_error("serving bundle tree contains unreachable nodes");
        }
    }

    const bool onnxPresent = manifest.contains("onnx") &&
                             manifest.at("onnx").value("present", false);
    if (requireOnnx && !onnxPresent)
    {
        throw std::runtime_error("serving bundle does not contain ONNX assets");
    }
    if (onnxPresent)
    {
        if (paths.count("model_onnx") == 0)
        {
            throw std::runtime_error("serving bundle ONNX file is missing");
        }
        const Json& onnx = manifest.at("onnx");
        onnxInputName_ = onnx.value("input_name", "");
        onnxOutputNames_ = onnx.value(
            "output_names", std::vector<std::string>{});
        const Json expectedInputShape = Json::array({"batch", inputDimension_});
        const Json expectedOutputShapes = Json::array({
            Json::array({"batch", classCount_}),
            Json::array({"batch", clusterCount_}),
        });
        if (onnxInputName_ != "preprocessed_features" ||
            onnxOutputNames_ != std::vector<std::string>{
                                    "class_probabilities", "cluster_logits"} ||
            onnx.value("opset", 0) != 17 ||
            onnx.value("input_shape", Json{}) != expectedInputShape ||
            onnx.value("output_shapes", Json{}) != expectedOutputShapes)
        {
            throw std::runtime_error("serving bundle ONNX contract is invalid");
        }
        onnxPath_ = paths.at("model_onnx");
    }

    const bool containsReference = manifest.value("contains_reference_dataset", false);
    const Json& referenceRows = manifest.at("reference_rows");
    if (!referenceRows.is_number_integer() || referenceRows.get<std::int64_t>() < 0)
    {
        throw std::runtime_error("serving bundle reference row count is invalid");
    }
    referenceRows_ = static_cast<std::size_t>(referenceRows.get<std::uint64_t>());
    if (containsReference)
    {
        if (paths.count("reference_inputs") == 0 || referenceRows_ == 0)
        {
            throw std::runtime_error("serving bundle reference dataset is invalid");
        }
        if (referenceRows_ >
            std::numeric_limits<std::size_t>::max() /
                (model::kPphInputDimension * sizeof(float)) ||
            std::filesystem::file_size(paths.at("reference_inputs")) !=
                referenceRows_ * model::kPphInputDimension * sizeof(float))
        {
            throw std::runtime_error("serving bundle reference dataset size is invalid");
        }
        std::ifstream input(paths.at("reference_inputs"), std::ios::binary);
        referenceInputs_.resize(referenceRows_ * model::kPphInputDimension);
        input.read(
            reinterpret_cast<char*>(referenceInputs_.data()),
            static_cast<std::streamsize>(referenceInputs_.size() * sizeof(float)));
        if (!input || input.peek() != std::ifstream::traits_type::eof())
        {
            throw std::runtime_error("serving bundle reference dataset size is invalid");
        }
        for (float value : referenceInputs_)
        {
            if (!std::isfinite(value))
            {
                throw std::runtime_error("serving bundle reference dataset is non-finite");
            }
        }
    }
    else if (referenceRows_ != 0)
    {
        throw std::runtime_error("serving bundle reference metadata is inconsistent");
    }
}

const std::filesystem::path& ServingBundle::directory() const noexcept { return directory_; }
const std::string& ServingBundle::modelVersion() const noexcept { return modelVersion_; }
int ServingBundle::inputDimension() const noexcept { return inputDimension_; }
int ServingBundle::classCount() const noexcept { return classCount_; }
int ServingBundle::clusterCount() const noexcept { return clusterCount_; }
const std::vector<BundleFeature>& ServingBundle::features() const noexcept { return features_; }
const std::vector<BundleTreeNode>& ServingBundle::treeNodes() const noexcept { return treeNodes_; }
const std::vector<double>& ServingBundle::featureImportances() const noexcept { return featureImportances_; }
int ServingBundle::treeRoot() const noexcept { return treeRoot_; }
const std::filesystem::path& ServingBundle::onnxPath() const noexcept { return onnxPath_; }
const std::string& ServingBundle::onnxInputName() const noexcept { return onnxInputName_; }
const std::vector<std::string>& ServingBundle::onnxOutputNames() const noexcept { return onnxOutputNames_; }
bool ServingBundle::hasReferenceInputs() const noexcept { return !referenceInputs_.empty(); }
std::size_t ServingBundle::referenceRows() const noexcept { return referenceRows_; }

std::vector<float> ServingBundle::referenceInput(std::size_t index) const
{
    if (!hasReferenceInputs() || index >= referenceRows_)
    {
        throw std::out_of_range("reference sample index is unavailable");
    }
    const auto begin = referenceInputs_.begin() +
                       static_cast<std::ptrdiff_t>(index * model::kPphInputDimension);
    return std::vector<float>(
        begin, begin + static_cast<std::ptrdiff_t>(model::kPphInputDimension));
}

} // namespace infrastructure
} // namespace treesem
