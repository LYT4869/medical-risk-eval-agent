#include <cassert>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "ModelFixtures.h"
#include "api/ApiException.h"
#include "api/PredictionJsonCodec.h"
#include "model/PphFeatureSchema.h"

namespace
{

bool rejects(const std::string& body, treesem::api::ApiException::Kind expectedKind)
{
    try
    {
        (void)treesem::api::PredictionJsonCodec::parseInput(body);
    }
    catch (const treesem::api::ApiException& exception)
    {
        return exception.kind() == expectedKind;
    }
    return false;
}

} // namespace

int main()
{
    using treesem::api::ApiException;
    using treesem::api::PredictionJsonCodec;

    const treesem::model::ModelInput sample =
        PredictionJsonCodec::parseInput(R"({"sample_index":7})");
    assert(std::holds_alternative<treesem::model::SampleIndexInput>(sample));
    assert(std::get<treesem::model::SampleIndexInput>(sample).sampleIndex == 7);

    std::vector<double> values(treesem::model::kPphInputDimension, 0.25);
    const std::string featureBody =
        nlohmann::json{{"preprocessed_features", values}}.dump();
    const treesem::model::ModelInput features =
        PredictionJsonCodec::parseInput(featureBody);
    assert(std::holds_alternative<treesem::model::PreprocessedFeaturesInput>(features));
    assert(std::get<treesem::model::PreprocessedFeaturesInput>(features).values == values);

    nlohmann::json raw = nlohmann::json::object();
    for (std::string_view name : treesem::model::kPphFeatureNames)
    {
        raw[std::string(name)] = 1.0;
    }
    const treesem::model::ModelInput rawInput = PredictionJsonCodec::parseInput(
        nlohmann::json{{"raw_features", raw}}.dump());
    assert(std::holds_alternative<treesem::model::RawClinicalFeaturesInput>(rawInput));
    assert(std::get<treesem::model::RawClinicalFeaturesInput>(rawInput).values.size() == 49);

    assert(rejects("not-json", ApiException::Kind::InvalidJson));
    assert(rejects("[]", ApiException::Kind::InvalidRequest));
    assert(rejects("{}", ApiException::Kind::InvalidRequest));
    assert(rejects(R"({"sample_index":-1})", ApiException::Kind::InvalidRequest));
    assert(rejects(R"({"sample_index":true})", ApiException::Kind::InvalidRequest));
    assert(rejects(
        R"({"sample_index":0,"preprocessed_features":[]})",
        ApiException::Kind::InvalidRequest));
    assert(rejects(
        R"({"preprocessed_features":[1]})",
        ApiException::Kind::InvalidRequest));
    assert(rejects(
        R"({"preprocessed_features":[]})",
        ApiException::Kind::InvalidRequest));
    assert(rejects(
        R"({"preprocessed_features":[NaN]})",
        ApiException::Kind::InvalidJson));
    assert(rejects(
        R"({"sample_index":18446744073709551615})",
        ApiException::Kind::InvalidRequest));
    assert(rejects(
        R"({"sample_index":1e999})",
        ApiException::Kind::InvalidJson));
    assert(rejects(
        R"({"sample_index":0,"sample_index":1})",
        ApiException::Kind::InvalidRequest));
    nlohmann::json missingRaw = raw;
    missingRaw.erase("Age");
    assert(rejects(
        nlohmann::json{{"raw_features", missingRaw}}.dump(),
        ApiException::Kind::InvalidRequest));
    nlohmann::json extraRaw = raw;
    extraRaw["unknown"] = 1;
    assert(rejects(
        nlohmann::json{{"raw_features", extraRaw}}.dump(),
        ApiException::Kind::InvalidRequest));
    nlohmann::json booleanRaw = raw;
    booleanRaw["Age"] = true;
    assert(rejects(
        nlohmann::json{{"raw_features", booleanRaw}}.dump(),
        ApiException::Kind::InvalidRequest));

    const nlohmann::json result = nlohmann::json::parse(
        PredictionJsonCodec::serializeResult(test::validModelResult()));
    assert(result.at("model") == "treeSem");
    assert(result.at("model_version") == "pph-seed42-1a299a474ce5");
    assert(result.at("serving_backend") == "python_reference");
    assert(result.at("prediction").at("label") == 0);
    assert(result.at("important_features").size() == 1);
    assert(result.at("important_features").at(0).at("original_value") == 200.0);
    assert(result.at("important_features").at(0).at("unit").is_null());
    assert(result.at("decision_path").size() == 2);
    assert(result.at("decision_path").back().at("leaf_id") == 8);
    assert(result.at("decision_path").at(0).at("threshold_original") == 315.0);
}
