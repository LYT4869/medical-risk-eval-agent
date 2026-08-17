#pragma once

#include "application/PredictionService.h"
#include "http/AsyncHttp.h"
#include "http/HttpRequest.h"
#include "service/InferenceScheduler.h"

namespace treesem
{
namespace api
{

class PredictionController
{
public:
    PredictionController(
        const application::PredictionService& predictionService,
        service::InferenceScheduler& inferenceScheduler);

    void handle(http::HttpRequest request, http::AsyncResponder responder) const;

private:
    const application::PredictionService& predictionService_;
    service::InferenceScheduler& inferenceScheduler_;
};

} // namespace api
} // namespace treesem
