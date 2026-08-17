#pragma once

#include <cstddef>
#include <functional>

#include "concurrency/BoundedWorkerPool.h"
#include "http/AsyncHttp.h"

namespace treesem
{
namespace service
{

class InferenceScheduler
{
public:
    using SubmitResult = concurrency::BoundedWorkerPool::SubmitResult;
    // The expensive job runs on a worker and returns only a lightweight
    // response writer. The scheduler guarantees one response attempt.
    using Job = std::function<http::ResponseWriter()>;

    InferenceScheduler(std::size_t workerCount, std::size_t queueCapacity);

    // Accepted jobs execute on a worker. Rejected jobs receive HTTP 503.
    SubmitResult schedule(Job job, http::AsyncResponder responder);

    void waitForIdle();
    void shutdown();

private:
    concurrency::BoundedWorkerPool workers_;
};

} // namespace service
} // namespace treesem
