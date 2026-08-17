#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "infrastructure/model/FeaturePreprocessor.h"
#include "infrastructure/model/OnnxTreeSemModelService.h"
#include "infrastructure/model/ServingBundle.h"

namespace
{

void compareResult(
    const treesem::model::ModelResult& actual,
    const nlohmann::json& expected)
{
    const auto& prediction = expected.at("prediction");
    assert(actual.prediction.label == prediction.at("label").get<int>());
    assert(actual.prediction.clusterId == prediction.at("cluster_id").get<int>());
    assert(actual.prediction.treeLeafId == prediction.at("tree_leaf_id").get<int>());
    assert(std::abs(actual.prediction.positiveProbability -
                    prediction.at("positive_probability").get<double>()) <= 1e-5);
    assert(std::abs(actual.prediction.confidence -
                    prediction.at("confidence").get<double>()) <= 1e-5);
    assert(std::abs(actual.prediction.treeProbability -
                    prediction.at("tree_probability").get<double>()) <= 1e-12);
    const auto& expectedPath = expected.at("decision_path");
    assert(actual.decisionPath.size() == expectedPath.size());
    for (std::size_t index = 0; index < actual.decisionPath.size(); ++index)
    {
        if (expectedPath.at(index).contains("leaf_id"))
        {
            assert(actual.decisionPath.at(index).leafId ==
                   expectedPath.at(index).at("leaf_id").get<int>());
        }
        else
        {
            assert(actual.decisionPath.at(index).nodeId ==
                   expectedPath.at(index).at("node_id").get<int>());
            assert(actual.decisionPath.at(index).featureIndex ==
                   expectedPath.at(index).at("feature_index").get<int>());
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    assert(argc == 2);
    const std::filesystem::path bundleDirectory = argv[1];
    treesem::infrastructure::OnnxTreeSemModelService service(bundleDirectory);
    const auto& bundle = service.bundle();
    assert(bundle.modelVersion() == "pph-seed42-1a299a474ce5");
    assert(bundle.referenceRows() == 1489);

    const auto byIndex = service.predict(treesem::model::SampleIndexInput{0});
    const std::vector<float> reference = bundle.referenceInput(0);
    std::vector<double> standardized(reference.begin(), reference.end());
    const auto byPreprocessed = service.predict(
        treesem::model::PreprocessedFeaturesInput{standardized});
    assert(byIndex.prediction.label == byPreprocessed.prediction.label);
    assert(byIndex.prediction.clusterId == byPreprocessed.prediction.clusterId);
    assert(byIndex.prediction.treeLeafId == byPreprocessed.prediction.treeLeafId);
    assert(std::abs(byIndex.prediction.positiveProbability -
                    byPreprocessed.prediction.positiveProbability) <= 1e-7);

    std::vector<double> raw;
    for (std::size_t index = 0; index < standardized.size(); ++index)
    {
        const auto& feature = bundle.features().at(index);
        raw.push_back(standardized[index] * feature.scale + feature.mean);
    }
    const auto byRaw = service.predict(
        treesem::model::RawClinicalFeaturesInput{raw});
    assert(byIndex.prediction.label == byRaw.prediction.label);
    assert(byIndex.prediction.clusterId == byRaw.prediction.clusterId);
    assert(byIndex.prediction.treeLeafId == byRaw.prediction.treeLeafId);
    assert(std::abs(byIndex.prediction.positiveProbability -
                    byRaw.prediction.positiveProbability) <= 1e-6);

    std::ifstream goldenInput(bundleDirectory / "reference_outputs.json");
    nlohmann::json golden;
    goldenInput >> golden;
    for (const auto& item : golden.at("cases"))
    {
        const int index = item.at("reference_index").get<int>();
        compareResult(
            service.predict(treesem::model::SampleIndexInput{index}),
            item.at("result"));
    }
}
