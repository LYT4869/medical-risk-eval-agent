#include <cassert>
#include <future>
#include <stdexcept>
#include <string>

#include "ModelFixtures.h"
#include "api/PredictionController.h"
#include "application/PredictionService.h"
#include "model/IModelService.h"
#include "model/ModelException.h"

namespace
{

struct CapturedResponse
{
    int status;
    std::string body;
};

class FakeModelService : public treesem::model::IModelService
{
public:
    enum class Behavior
    {
        Success,
        Timeout,
        InferenceFailure,
        Failure,
    };

    treesem::model::ModelResult predict(
        const treesem::model::ModelInput&) const override
    {
        switch (behavior)
        {
        case Behavior::Success:
            return test::validModelResult();
        case Behavior::Timeout:
            throw treesem::model::ModelException(
                treesem::model::ModelException::Kind::Timeout,
                "simulated timeout");
        case Behavior::InferenceFailure:
            throw treesem::model::ModelException(
                treesem::model::ModelException::Kind::InferenceFailure,
                "simulated ONNX failure");
        case Behavior::Failure:
            throw std::runtime_error("simulated unexpected failure");
        }
        throw std::runtime_error("unreachable fake behavior");
    }

    Behavior behavior{Behavior::Success};
};

CapturedResponse invoke(
    const treesem::api::PredictionController& controller,
    const std::string& body)
{
    http::HttpRequest request;
    request.setBody(body);

    std::promise<CapturedResponse> promise;
    std::future<CapturedResponse> future = promise.get_future();
    controller.handle(
        std::move(request),
        [&promise](http::ResponseWriter writer) {
            http::HttpResponse response(false);
            writer(&response);
            promise.set_value(
                {static_cast<int>(response.getStatusCode()), response.body()});
        });
    return future.get();
}

} // namespace

int main()
{
    FakeModelService modelService;
    treesem::application::PredictionService predictionService(modelService);
    treesem::service::InferenceScheduler scheduler(1, 2);
    treesem::api::PredictionController controller(predictionService, scheduler);

    CapturedResponse response = invoke(controller, R"({"sample_index":0})");
    assert(response.status == 200);
    assert(response.body.find("\"model\":\"treeSem\"") != std::string::npos);
    assert(response.body.find("\"decision_path\"") != std::string::npos);

    response = invoke(controller, "not-json");
    assert(response.status == 400);
    assert(response.body.find("\"error\":\"invalid_json\"") !=
           std::string::npos);
    assert(response.body.find("\"message\"") != std::string::npos);

    response = invoke(controller, R"({"preprocessed_features":[1.0]})");
    assert(response.status == 400);
    assert(response.body.find("\"error\":\"invalid_request\"") !=
           std::string::npos);

    modelService.behavior = FakeModelService::Behavior::Timeout;
    response = invoke(controller, R"({"sample_index":0})");
    assert(response.status == 504);
    assert(response.body.find("\"error\":\"model_adapter_timeout\"") !=
           std::string::npos);
    assert(response.body.find("simulated timeout") == std::string::npos);

    modelService.behavior = FakeModelService::Behavior::Failure;
    response = invoke(controller, R"({"sample_index":0})");
    assert(response.status == 500);
    assert(response.body.find("\"error\":\"internal_error\"") !=
           std::string::npos);
    assert(response.body.find("simulated unexpected failure") ==
           std::string::npos);

    modelService.behavior = FakeModelService::Behavior::InferenceFailure;
    response = invoke(controller, R"({"sample_index":0})");
    assert(response.status == 500);
    assert(response.body.find("\"error\":\"model_inference_failed\"") !=
           std::string::npos);
    assert(response.body.find("simulated ONNX failure") == std::string::npos);

    scheduler.waitForIdle();
}
