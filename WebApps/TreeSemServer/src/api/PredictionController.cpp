#include "api/PredictionController.h"

#include <utility>

#include "api/ApiException.h"
#include "api/BusinessJsonCodec.h"
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
    , predictionScheduler_(inferenceScheduler)
{}

PredictionController::PredictionController(
    const application::PredictionService& predictionService,
    service::BlockingTaskScheduler& predictionScheduler,
    bool cookieSecure,
    long sessionTtlSeconds)
    : predictionService_(predictionService)
    , predictionScheduler_(predictionScheduler)
    , persistent_(true)
    , cookieSecure_(cookieSecure)
    , sessionTtlSeconds_(sessionTtlSeconds)
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

    const bool internal = request.path().rfind("/internal/", 0) == 0;
    std::optional<std::string> suppliedSession;
    if (persistent_)
    {
        suppliedSession = BusinessJsonCodec::sessionId(
            request, internal ? application::SessionAccess::Internal
                              : application::SessionAccess::Public);
    }
    predictionScheduler_.schedule(
        [input = std::move(input), suppliedSession, internal, this]()
            -> http::ResponseWriter {
            try
            {
                if (persistent_)
                {
                    auto created = predictionService_.createPrediction(
                        input,
                        suppliedSession,
                        internal ? application::SessionAccess::Internal
                                 : application::SessionAccess::Public);
                    http::ResponseWriter success = HttpErrorMapper::success(
                        BusinessJsonCodec::serializePrediction(created.first));
                    const bool setCookie = !internal && created.second.created;
                    const std::string cookie = setCookie
                        ? BusinessJsonCodec::sessionCookie(
                              created.second.session.sessionId,
                              sessionTtlSeconds_, cookieSecure_)
                        : std::string();
                    return [success = std::move(success), setCookie, cookie](
                               http::HttpResponse* response) {
                        success(response);
                        if (setCookie) response->addHeader("Set-Cookie", cookie);
                    };
                }
                model::ModelResult result = predictionService_.predict(input);
                return HttpErrorMapper::success(
                    PredictionJsonCodec::serializeResult(result));
            }
            catch (const model::ModelException& exception)
            {
                return HttpErrorMapper::from(exception);
            }
            catch (const application::BusinessException& exception)
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
