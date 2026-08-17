#include "service/InferenceScheduler.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace treesem
{
namespace service
{
namespace
{

void respondWithJsonError(const http::AsyncResponder& responder,
                          http::HttpResponse::HttpStatusCode status,
                          const std::string& statusMessage,
                          const std::string& body)
{
    responder([status, statusMessage, body](http::HttpResponse* response) {
        response->setStatusCode(status);
        response->setStatusMessage(statusMessage);
        response->setContentType("application/json; charset=utf-8");
        response->setBody(body);
    });
}

} // namespace

InferenceScheduler::InferenceScheduler(std::size_t workerCount,
                                       std::size_t queueCapacity)
    : workers_(workerCount, queueCapacity)
{}

InferenceScheduler::SubmitResult InferenceScheduler::schedule(
    Job job,
    http::AsyncResponder responder)
{
    if (!job)
    {
        throw std::invalid_argument("job must not be empty");
    }
    if (!responder)
    {
        throw std::invalid_argument("responder must not be empty");
    }

    const SubmitResult result = workers_.trySubmit(
        [job = std::move(job), responder]() {
            try
            {
                http::ResponseWriter writer = job();
                if (!writer)
                {
                    throw std::runtime_error("job returned an empty response writer");
                }
                responder(std::move(writer));
            }
            catch (...)
            {
                respondWithJsonError(
                    responder,
                    http::HttpResponse::k500InternalServerError,
                    "Internal Server Error",
                    R"({"error":"inference_failed","message":"The inference task failed."})");
            }
        });

    if (result == SubmitResult::QueueFull)
    {
        respondWithJsonError(
            responder,
            http::HttpResponse::k503ServiceUnavailable,
            "Service Unavailable",
            R"({"error":"server_overloaded","message":"The inference queue is full."})");
    }
    else if (result == SubmitResult::Stopped)
    {
        respondWithJsonError(
            responder,
            http::HttpResponse::k503ServiceUnavailable,
            "Service Unavailable",
            R"({"error":"service_stopping","message":"The service is stopping."})");
    }

    return result;
}

void InferenceScheduler::waitForIdle()
{
    workers_.waitForIdle();
}

void InferenceScheduler::shutdown()
{
    workers_.shutdown();
}

} // namespace service
} // namespace treesem
