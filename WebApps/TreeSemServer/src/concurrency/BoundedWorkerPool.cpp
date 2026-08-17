#include "concurrency/BoundedWorkerPool.h"

#include <stdexcept>
#include <utility>

namespace treesem
{
namespace concurrency
{

BoundedWorkerPool::BoundedWorkerPool(std::size_t workerCount,
                                     std::size_t queueCapacity,
                                     ErrorHandler errorHandler)
    : queueCapacity_(queueCapacity)
    , errorHandler_(std::move(errorHandler))
{
    if (workerCount == 0)
    {
        throw std::invalid_argument("workerCount must be greater than zero");
    }
    if (queueCapacity == 0)
    {
        throw std::invalid_argument("queueCapacity must be greater than zero");
    }

    workers_.reserve(workerCount);
    try
    {
        for (std::size_t index = 0; index < workerCount; ++index)
        {
            workers_.emplace_back(&BoundedWorkerPool::workerLoop, this);
        }
    }
    catch (...)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = State::Draining;
        }
        taskAvailable_.notify_all();
        for (std::thread& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        state_ = State::Stopped;
        throw;
    }
}

BoundedWorkerPool::~BoundedWorkerPool()
{
    shutdown();
}

BoundedWorkerPool::SubmitResult BoundedWorkerPool::trySubmit(Task task)
{
    if (!task)
    {
        throw std::invalid_argument("task must not be empty");
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::Running)
        {
            return SubmitResult::Stopped;
        }
        if (tasks_.size() >= queueCapacity_)
        {
            return SubmitResult::QueueFull;
        }
        tasks_.push_back(std::move(task));
    }
    taskAvailable_.notify_one();
    return SubmitResult::Accepted;
}

void BoundedWorkerPool::shutdown()
{
    std::call_once(shutdownOnce_, [this]() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = State::Draining;
        }
        taskAvailable_.notify_all();

        for (std::thread& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = State::Stopped;
        }
        idle_.notify_all();
    });
}

void BoundedWorkerPool::waitForIdle()
{
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this]() {
        return (tasks_.empty() && activeTasks_ == 0) || state_ == State::Stopped;
    });
}

std::size_t BoundedWorkerPool::workerCount() const noexcept
{
    return workers_.size();
}

std::size_t BoundedWorkerPool::queueCapacity() const noexcept
{
    return queueCapacity_;
}

std::size_t BoundedWorkerPool::queuedTasks() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size();
}

std::size_t BoundedWorkerPool::activeTasks() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return activeTasks_;
}

bool BoundedWorkerPool::isRunning() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ == State::Running;
}

void BoundedWorkerPool::workerLoop()
{
    while (true)
    {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            taskAvailable_.wait(lock, [this]() {
                return !tasks_.empty() || state_ != State::Running;
            });

            if (tasks_.empty())
            {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop_front();
            ++activeTasks_;
        }

        try
        {
            task();
        }
        catch (...)
        {
            reportTaskError(std::current_exception());
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --activeTasks_;
            if (tasks_.empty() && activeTasks_ == 0)
            {
                idle_.notify_all();
            }
        }
    }
}

void BoundedWorkerPool::reportTaskError(std::exception_ptr error) noexcept
{
    if (!errorHandler_)
    {
        return;
    }
    try
    {
        errorHandler_(std::move(error));
    }
    catch (...)
    {
        // Error reporting must never terminate a worker thread.
    }
}

} // namespace concurrency
} // namespace treesem
