#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "concurrency/BoundedWorkerPool.h"
#include "http/AsyncHttp.h"

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
                          SchedulerErrorPolicy errorPolicy);

    SubmitResult schedule(Job job, http::AsyncResponder responder);
    void waitForIdle();
    void shutdown();

private:
    SchedulerErrorPolicy errorPolicy_;
    concurrency::BoundedWorkerPool workers_;
};

SchedulerErrorPolicy predictionSchedulerErrors();
SchedulerErrorPolicy databaseSchedulerErrors();
SchedulerErrorPolicy agentSchedulerErrors();
SchedulerErrorPolicy authenticationSchedulerErrors();

} // namespace service
} // namespace treesem
