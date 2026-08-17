#include <atomic>
#include <cassert>
#include <future>
#include <stdexcept>
#include <thread>

#include "concurrency/BoundedWorkerPool.h"

int main()
{
    using Pool = treesem::concurrency::BoundedWorkerPool;

    std::atomic<int> completed{0};
    std::promise<void> firstStarted;
    std::promise<void> releaseFirst;
    std::shared_future<void> releaseSignal = releaseFirst.get_future().share();

    Pool pool(1, 1);
    assert(pool.workerCount() == 1);
    assert(pool.queueCapacity() == 1);

    assert(pool.trySubmit([&]() {
               firstStarted.set_value();
               releaseSignal.wait();
               ++completed;
           }) == Pool::SubmitResult::Accepted);
    firstStarted.get_future().wait();

    assert(pool.trySubmit([&]() { ++completed; }) == Pool::SubmitResult::Accepted);
    assert(pool.trySubmit([]() {}) == Pool::SubmitResult::QueueFull);

    releaseFirst.set_value();
    pool.waitForIdle();
    assert(completed.load() == 2);
    assert(pool.queuedTasks() == 0);
    assert(pool.activeTasks() == 0);

    pool.shutdown();
    assert(!pool.isRunning());
    assert(pool.trySubmit([]() {}) == Pool::SubmitResult::Stopped);

    std::atomic<int> errors{0};
    Pool resilientPool(1, 2, [&](std::exception_ptr) { ++errors; });
    resilientPool.trySubmit([]() { throw std::runtime_error("expected"); });
    resilientPool.trySubmit([&]() { ++completed; });
    resilientPool.waitForIdle();
    assert(errors.load() == 1);
    assert(completed.load() == 3);

    std::atomic<int> drained{0};
    std::promise<void> drainStarted;
    std::promise<void> releaseDrain;
    std::shared_future<void> drainSignal = releaseDrain.get_future().share();
    Pool drainingPool(1, 2);
    drainingPool.trySubmit([&]() {
        drainStarted.set_value();
        drainSignal.wait();
        ++drained;
    });
    drainStarted.get_future().wait();
    drainingPool.trySubmit([&]() { ++drained; });

    std::future<void> shutdown = std::async(std::launch::async, [&]() {
        drainingPool.shutdown();
    });
    while (drainingPool.isRunning())
    {
        std::this_thread::yield();
    }
    assert(drainingPool.trySubmit([]() {}) == Pool::SubmitResult::Stopped);
    releaseDrain.set_value();
    shutdown.get();
    assert(drained.load() == 2);
}
