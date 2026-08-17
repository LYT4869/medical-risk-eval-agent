#include "api/PredictionController.h"

#include <utility>

#include "api/ApiException.h"
#include "api/HttpErrorMapper.h"
#include "api/PredictionJsonCodec.h"
#include "model/ModelException.h"

namespace treesem
{
namespace api
{

PredictionController::PredictionController(
    const application::PredictionService& predictionService,
    service::InferenceScheduler& inferenceScheduler)
    : predictionService_(predictionService)
    , inferenceScheduler_(inferenceScheduler)
{}

void PredictionController::handle(
    http::HttpRequest request,
    http::AsyncResponder responder) const
{
    model::ModelInput input;
    try
    {
        input = PredictionJsonCodec::parseInput(request.getBody());
    }
    catch (const ApiException& exception)
    {
        responder(HttpErrorMapper::from(exception));
        return;
    }
    catch (...)
    {
        responder(HttpErrorMapper::internalError());
        return;
    }

    inferenceScheduler_.schedule(
        [input = std::move(input), this]() -> http::ResponseWriter {
            try
            {
                model::ModelResult result = predictionService_.predict(input);
                return HttpErrorMapper::success(
                    PredictionJsonCodec::serializeResult(result));
            }
            catch (const model::ModelException& exception)
            {
                return HttpErrorMapper::from(exception);
            }
            catch (...)
            {
                return HttpErrorMapper::internalError();
            }
        },
        std::move(responder));
}

} // namespace api
} // namespace treesem
