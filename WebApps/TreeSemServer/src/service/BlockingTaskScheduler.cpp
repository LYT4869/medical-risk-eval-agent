#include "service/BlockingTaskScheduler.h"

#include <chrono>
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
    SchedulerErrorPolicy errorPolicy,
    std::shared_ptr<http::observability::MetricsRegistry> metrics,
    std::string schedulerName)
    : errorPolicy_(std::move(errorPolicy))
    , metrics_(std::move(metrics))
    , schedulerName_(std::move(schedulerName))
    , workers_(workerCount, queueCapacity)
{
    if (errorPolicy_.queueFullCode.empty() || errorPolicy_.stoppedCode.empty() ||
        errorPolicy_.taskFailedCode.empty())
    {
        throw std::invalid_argument("scheduler error codes must not be empty");
    }
    if (metrics_ && schedulerName_.empty())
        throw std::invalid_argument("instrumented scheduler requires a stable name");
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

    const auto taskMetrics = metrics_;
    const std::string taskSchedulerName = schedulerName_;
    auto* const workerPool = &workers_;
    const SubmitResult result = workers_.trySubmit(
        [taskMetrics, taskSchedulerName, workerPool,
         job = std::move(job), responder, policy = errorPolicy_]() {
            const auto started = std::chrono::steady_clock::now();
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
            if (taskMetrics)
            {
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                taskMetrics->observe("treesem_scheduler_task_duration_seconds",
                                     {{"scheduler", taskSchedulerName}}, seconds);
                const std::size_t active = workerPool->activeTasks();
                taskMetrics->setGauge("treesem_scheduler_active_tasks",
                    {{"scheduler", taskSchedulerName}},
                    static_cast<double>(active > 0 ? active - 1 : 0));
                taskMetrics->setGauge("treesem_scheduler_queue_depth",
                    {{"scheduler", taskSchedulerName}},
                    static_cast<double>(workerPool->queuedTasks()));
            }
        });

    if (metrics_)
    {
        std::string outcome = "accepted";
        if (result == SubmitResult::QueueFull) outcome = "queue_full";
        else if (result == SubmitResult::Stopped) outcome = "stopped";
        metrics_->increment("treesem_scheduler_submissions_total",
            {{"scheduler", schedulerName_}, {"result", outcome}});
        metrics_->setGauge("treesem_scheduler_queue_depth",
            {{"scheduler", schedulerName_}},
            static_cast<double>(workers_.queuedTasks()));
    }

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

std::size_t BlockingTaskScheduler::queuedTasks() const { return workers_.queuedTasks(); }
std::size_t BlockingTaskScheduler::activeTasks() const { return workers_.activeTasks(); }
std::size_t BlockingTaskScheduler::workerCount() const noexcept { return workers_.workerCount(); }
std::size_t BlockingTaskScheduler::queueCapacity() const noexcept { return workers_.queueCapacity(); }

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

SchedulerErrorPolicy agentSchedulerErrors()
{
    return {"agent_overloaded", "The agent queue is full.",
            "service_stopping", "The service is stopping.",
            "agent_task_failed", "The agent task failed."};
}

SchedulerErrorPolicy authenticationSchedulerErrors()
{
    return {"authentication_overloaded", "The authentication queue is full.",
            "service_stopping", "The service is stopping.",
            "authentication_task_failed", "Authentication failed."};
}

} // namespace service
} // namespace treesem
