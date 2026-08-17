#include "service/BlockingTaskScheduler.h"

#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace treesem
{
namespace service
{
namespace
{

void respondWithError(const http::AsyncResponder& responder,
                      http::HttpResponse::HttpStatusCode status,
                      const std::string& statusMessage,
                      const std::string& code,
                      const std::string& message)
{
    const std::string body =
        nlohmann::json{{"error", code}, {"message", message}}.dump();
    responder([status, statusMessage, body](http::HttpResponse* response) {
        response->setStatusCode(status);
        response->setStatusMessage(statusMessage);
        response->setContentType("application/json; charset=utf-8");
        response->setBody(body);
    });
}

} // namespace

BlockingTaskScheduler::BlockingTaskScheduler(
    std::size_t workerCount,
    std::size_t queueCapacity,
    SchedulerErrorPolicy errorPolicy)
    : errorPolicy_(std::move(errorPolicy))
    , workers_(workerCount, queueCapacity)
{
    if (errorPolicy_.queueFullCode.empty() || errorPolicy_.stoppedCode.empty() ||
        errorPolicy_.taskFailedCode.empty())
    {
        throw std::invalid_argument("scheduler error codes must not be empty");
    }
}

BlockingTaskScheduler::SubmitResult BlockingTaskScheduler::schedule(
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
        [job = std::move(job), responder, policy = errorPolicy_]() {
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
                respondWithError(
                    responder,
                    http::HttpResponse::k500InternalServerError,
                    "Internal Server Error",
                    policy.taskFailedCode,
                    policy.taskFailedMessage);
            }
        });

    if (result == SubmitResult::QueueFull)
    {
        respondWithError(
            responder,
            http::HttpResponse::k503ServiceUnavailable,
            "Service Unavailable",
            errorPolicy_.queueFullCode,
            errorPolicy_.queueFullMessage);
    }
    else if (result == SubmitResult::Stopped)
    {
        respondWithError(
            responder,
            http::HttpResponse::k503ServiceUnavailable,
            "Service Unavailable",
            errorPolicy_.stoppedCode,
            errorPolicy_.stoppedMessage);
    }
    return result;
}

void BlockingTaskScheduler::waitForIdle()
{
    workers_.waitForIdle();
}

void BlockingTaskScheduler::shutdown()
{
    workers_.shutdown();
}

SchedulerErrorPolicy predictionSchedulerErrors()
{
    return {
        "prediction_overloaded",
        "The prediction queue is full.",
        "service_stopping",
        "The service is stopping.",
        "prediction_task_failed",
        "The prediction task failed."};
}

SchedulerErrorPolicy databaseSchedulerErrors()
{
    return {
        "database_overloaded",
        "The database task queue is full.",
        "service_stopping",
        "The service is stopping.",
        "database_task_failed",
        "The database task failed."};
}

} // namespace service
} // namespace treesem
