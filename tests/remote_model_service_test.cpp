#include <cassert>
#include <limits>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "ModelFixtures.h"
#include "client/PythonModelClient.h"
#include "infrastructure/model/RemoteTreeSemModelService.h"
#include "model/ModelException.h"

namespace
{

class FakeClient : public treesem::client::IModelAdapterClient
{
public:
    mutable std::string lastRequest;
    treesem::client::ModelAdapterResponse response{200, test::validAdapterResponse()};
    std::optional<treesem::client::ModelAdapterException::Kind> failure;

    treesem::client::ModelAdapterResponse predict(
        const std::string& requestBody) const override
    {
        lastRequest = requestBody;
        if (failure.has_value())
        {
            throw treesem::client::ModelAdapterException(*failure, "simulated failure");
        }
        return response;
    }
};

bool throwsKind(
    const treesem::infrastructure::RemoteTreeSemModelService& service,
    const treesem::model::ModelInput& input,
    treesem::model::ModelException::Kind expected)
{
    try
    {
        (void)service.predict(input);
    }
    catch (const treesem::model::ModelException& exception)
    {
        return exception.kind() == expected;
    }
    return false;
}

} // namespace

int main()
{
    using ClientFailure = treesem::client::ModelAdapterException::Kind;
    using ModelFailure = treesem::model::ModelException::Kind;

    FakeClient client;
    treesem::infrastructure::RemoteTreeSemModelService service(client);
    const treesem::model::ModelInput input = treesem::model::SampleIndexInput{3};

    const treesem::model::ModelResult result = service.predict(input);
    assert(result.modelName == "treeSem");
    assert(result.prediction.label == 0);
    assert(result.decisionPath.back().leafId == 8);
    assert(nlohmann::json::parse(client.lastRequest).at("sample_index") == 3);

    client.response = {200, "not-json"};
    assert(throwsKind(service, input, ModelFailure::InvalidResponse));

    client.response = {200, R"({"model":"treeSem"})"};
    assert(throwsKind(service, input, ModelFailure::InvalidResponse));

    client.response = {200, ""};
    assert(throwsKind(service, input, ModelFailure::InvalidResponse));

    nlohmann::json invalidProbability =
        nlohmann::json::parse(test::validAdapterResponse());
    invalidProbability["prediction"]["confidence"] = 1.5;
    client.response = {200, invalidProbability.dump()};
    assert(throwsKind(service, input, ModelFailure::InvalidResponse));

    nlohmann::json invalidLeaf = nlohmann::json::parse(test::validAdapterResponse());
    invalidLeaf["decision_path"].back()["leaf_id"] = 9;
    client.response = {200, invalidLeaf.dump()};
    assert(throwsKind(service, input, ModelFailure::InvalidResponse));

    client.response = {400, R"({"error":"invalid_request"})"};
    assert(throwsKind(service, input, ModelFailure::InvalidInput));

    client.response = {500, R"({"error":"failed"})"};
    assert(throwsKind(service, input, ModelFailure::DownstreamFailure));

    client.response = {404, R"({"error":"not_found"})"};
    assert(throwsKind(service, input, ModelFailure::InvalidResponse));

    client.failure = ClientFailure::Timeout;
    assert(throwsKind(service, input, ModelFailure::Timeout));
    client.failure = ClientFailure::Unavailable;
    assert(throwsKind(service, input, ModelFailure::Unavailable));
    client.failure = ClientFailure::InvalidResponse;
    assert(throwsKind(service, input, ModelFailure::InvalidResponse));

    client.failure.reset();
    client.response = {200, test::validAdapterResponse()};
    assert(throwsKind(
        service,
        treesem::model::SampleIndexInput{-1},
        ModelFailure::InvalidInput));
    assert(throwsKind(
        service,
        treesem::model::PreprocessedFeaturesInput{{1.0}},
        ModelFailure::InvalidInput));
    std::vector<double> nonFinite(treesem::model::kPphInputDimension, 0.0);
    nonFinite[3] = std::numeric_limits<double>::infinity();
    assert(throwsKind(
        service,
        treesem::model::PreprocessedFeaturesInput{nonFinite},
        ModelFailure::InvalidInput));
}
