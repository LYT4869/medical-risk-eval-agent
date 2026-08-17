#pragma once

#include <functional>
#include <memory>

namespace muduo
{
namespace net
{
class EventLoop;
}
} // namespace muduo

namespace treesem
{
namespace infrastructure
{

// Converts SIGINT/SIGTERM into an EventLoop callback through a non-blocking
// self-pipe. The signal handler itself performs only the async-signal-safe
// write(2) operation.
class ProcessSignalHandler
{
public:
    ProcessSignalHandler(
        muduo::net::EventLoop* eventLoop,
        std::function<void()> shutdownCallback);
    ~ProcessSignalHandler();

    ProcessSignalHandler(const ProcessSignalHandler&) = delete;
    ProcessSignalHandler& operator=(const ProcessSignalHandler&) = delete;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infrastructure
} // namespace treesem
