#include <cassert>

#include "ModelFixtures.h"
#include "infrastructure/model/RoutingModelServices.h"
#include "model/ModelException.h"

namespace
{

class FakeService final : public treesem::model::IModelService
{
public:
    enum class Behavior { Success, InvalidInput, InferenceFailure, Unavailable };

    treesem::model::ModelResult predict(
        const treesem::model::ModelInput&) const override
    {
        ++calls;
        if (behavior == Behavior::InvalidInput)
        {
            throw treesem::model::ModelException(
                treesem::model::ModelException::Kind::InvalidInput, "invalid");
        }
        if (behavior == Behavior::InferenceFailure)
        {
            throw treesem::model::ModelException(
                treesem::model::ModelException::Kind::InferenceFailure, "failed");
        }
        if (behavior == Behavior::Unavailable)
        {
            throw treesem::model::ModelException(
                treesem::model::ModelException::Kind::Unavailable, "unavailable");
        }
        treesem::model::ModelResult result = test::validModelResult();
        result.prediction.positiveProbability = probability;
        return result;
    }

    Behavior behavior{Behavior::Success};
    double probability{0.1};
    mutable int calls{0};
};

template <typename Function>
void assertKind(Function function, treesem::model::ModelException::Kind expected)
{
    try
    {
        function();
        assert(false);
    }
    catch (const treesem::model::ModelException& error)
    {
        assert(error.kind() == expected);
    }
}

} // namespace

int main()
{
    const treesem::model::ModelInput input = treesem::model::SampleIndexInput{0};
    FakeService primary;
    FakeService remote;
    treesem::infrastructure::FallbackModelService fallback(primary, remote);

    auto result = fallback.predict(input);
    assert(primary.calls == 1 && remote.calls == 0);
    assert(result.servingBackend == "python_reference");

    primary.behavior = FakeService::Behavior::InferenceFailure;
    result = fallback.predict(input);
    assert(primary.calls == 2 && remote.calls == 1);
    assert(result.servingBackend == "remote_fallback");

    primary.behavior = FakeService::Behavior::InvalidInput;
    assertKind(
        [&]() { fallback.predict(input); },
        treesem::model::ModelException::Kind::InvalidInput);
    assert(remote.calls == 1);

    primary.behavior = FakeService::Behavior::InferenceFailure;
    remote.behavior = FakeService::Behavior::Unavailable;
    assertKind(
        [&]() { fallback.predict(input); },
        treesem::model::ModelException::Kind::Unavailable);
    assert(primary.calls == 4 && remote.calls == 2);

    primary.behavior = FakeService::Behavior::Success;
    remote.behavior = FakeService::Behavior::Success;
    primary.probability = 0.2;
    remote.probability = 0.8;
    treesem::infrastructure::ShadowModelService shadow(primary, remote);
    result = shadow.predict(input);
    assert(result.prediction.positiveProbability == 0.2);

    remote.behavior = FakeService::Behavior::Unavailable;
    result = shadow.predict(input);
    assert(result.prediction.positiveProbability == 0.2);
}
