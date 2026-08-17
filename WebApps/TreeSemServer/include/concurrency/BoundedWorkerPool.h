#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace treesem
{
namespace concurrency
{

class BoundedWorkerPool
{
public:
    using Task = std::function<void()>;
    using ErrorHandler = std::function<void(std::exception_ptr)>;

    enum class SubmitResult
    {
        Accepted,
        QueueFull,
        Stopped,
    };

    BoundedWorkerPool(std::size_t workerCount,
                      std::size_t queueCapacity,
                      ErrorHandler errorHandler = {});
    ~BoundedWorkerPool();

    BoundedWorkerPool(const BoundedWorkerPool&) = delete;
    BoundedWorkerPool& operator=(const BoundedWorkerPool&) = delete;

    // Never blocks waiting for queue space.
    SubmitResult trySubmit(Task task);

    // Stops accepting new tasks, drains accepted tasks, then joins workers.
    void shutdown();
    void waitForIdle();

    std::size_t workerCount() const noexcept;
    std::size_t queueCapacity() const noexcept;
    std::size_t queuedTasks() const;
    std::size_t activeTasks() const;
    bool isRunning() const;

private:
    enum class State
    {
        Running,
        Draining,
        Stopped,
    };

    void workerLoop();
    void reportTaskError(std::exception_ptr error) noexcept;

    const std::size_t queueCapacity_;
    ErrorHandler errorHandler_;

    mutable std::mutex mutex_;
    std::condition_variable taskAvailable_;
    std::condition_variable idle_;
    std::deque<Task> tasks_;
    std::vector<std::thread> workers_;
    std::size_t activeTasks_{0};
    State state_{State::Running};
    std::once_flag shutdownOnce_;
};

} // namespace concurrency
} // namespace treesem
