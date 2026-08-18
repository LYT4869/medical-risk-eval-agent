#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "concurrency/BoundedWorkerPool.h"
#include "http/AsyncHttp.h"
#include "observability/MetricsRegistry.h"

namespace treesem
{
namespace service
{

struct SchedulerErrorPolicy
{
    std::string queueFullCode;
    std::string queueFullMessage;
    std::string stoppedCode;
    std::string stoppedMessage;
    std::string taskFailedCode;
    std::string taskFailedMessage;
};

class BlockingTaskScheduler
{
public:
    using SubmitResult = concurrency::BoundedWorkerPool::SubmitResult;
    using Job = std::function<http::ResponseWriter()>;

    BlockingTaskScheduler(std::size_t workerCount,
                          std::size_t queueCapacity,
                          SchedulerErrorPolicy errorPolicy,
                          std::shared_ptr<http::observability::MetricsRegistry> metrics = {},
                          std::string schedulerName = {});

    SubmitResult schedule(Job job, http::AsyncResponder responder);
    void waitForIdle();
    void shutdown();
    std::size_t queuedTasks() const;
    std::size_t activeTasks() const;
    std::size_t workerCount() const noexcept;
    std::size_t queueCapacity() const noexcept;

private:
    SchedulerErrorPolicy errorPolicy_;
    std::shared_ptr<http::observability::MetricsRegistry> metrics_;
    std::string schedulerName_;
    concurrency::BoundedWorkerPool workers_;
};

SchedulerErrorPolicy predictionSchedulerErrors();
SchedulerErrorPolicy databaseSchedulerErrors();
SchedulerErrorPolicy agentSchedulerErrors();
SchedulerErrorPolicy authenticationSchedulerErrors();

} // namespace service
} // namespace treesem
