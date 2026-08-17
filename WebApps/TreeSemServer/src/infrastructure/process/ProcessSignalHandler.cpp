#include "infrastructure/process/ProcessSignalHandler.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <fcntl.h>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <unistd.h>

#include <muduo/net/Channel.h>
#include <muduo/net/EventLoop.h>

namespace treesem
{
namespace infrastructure
{
namespace
{

volatile std::sig_atomic_t signalWriteFileDescriptor = -1;

extern "C" void writeShutdownSignal(int signalNumber)
{
    const int savedErrno = errno;
    const int fileDescriptor = signalWriteFileDescriptor;
    if (fileDescriptor >= 0)
    {
        const std::uint8_t signalByte = static_cast<std::uint8_t>(signalNumber);
        const ssize_t ignored = ::write(fileDescriptor, &signalByte, sizeof signalByte);
        (void)ignored;
    }
    errno = savedErrno;
}

void throwSystemError(const char* operation)
{
    throw std::system_error(errno, std::generic_category(), operation);
}

} // namespace

class ProcessSignalHandler::Impl
{
public:
    Impl(muduo::net::EventLoop* eventLoop, std::function<void()> shutdownCallback)
        : callback_(std::move(shutdownCallback))
    {
        if (eventLoop == nullptr || !callback_)
        {
            throw std::invalid_argument(
                "ProcessSignalHandler requires an EventLoop and callback");
        }
        if (signalWriteFileDescriptor >= 0)
        {
            throw std::logic_error("only one ProcessSignalHandler may be active");
        }

        if (::pipe2(pipeFileDescriptors_, O_NONBLOCK | O_CLOEXEC) != 0)
        {
            throwSystemError("pipe2");
        }

        try
        {
            installSignalHandlers();
            signalWriteFileDescriptor = pipeFileDescriptors_[1];
            channel_ = std::make_unique<muduo::net::Channel>(
                eventLoop, pipeFileDescriptors_[0]);
            channel_->setReadCallback(
                [this](muduo::Timestamp) { handleSignalNotification(); });
            channel_->enableReading();
        }
        catch (...)
        {
            signalWriteFileDescriptor = -1;
            restoreSignalHandlers();
            ::close(pipeFileDescriptors_[0]);
            ::close(pipeFileDescriptors_[1]);
            throw;
        }
    }

    ~Impl()
    {
        signalWriteFileDescriptor = -1;
        restoreSignalHandlers();
        if (channel_)
        {
            channel_->disableAll();
            channel_->remove();
        }
        ::close(pipeFileDescriptors_[0]);
        ::close(pipeFileDescriptors_[1]);
    }

private:
    void installSignalHandlers()
    {
        struct sigaction action {};
        action.sa_handler = writeShutdownSignal;
        ::sigemptyset(&action.sa_mask);
        action.sa_flags = SA_RESTART;

        if (::sigaction(SIGINT, &action, &oldInterruptAction) != 0)
        {
            throwSystemError("sigaction(SIGINT)");
        }
        interruptInstalled_ = true;
        if (::sigaction(SIGTERM, &action, &oldTerminateAction) != 0)
        {
            throwSystemError("sigaction(SIGTERM)");
        }
        terminateInstalled_ = true;
    }

    void restoreSignalHandlers() noexcept
    {
        if (terminateInstalled_)
        {
            ::sigaction(SIGTERM, &oldTerminateAction, nullptr);
            terminateInstalled_ = false;
        }
        if (interruptInstalled_)
        {
            ::sigaction(SIGINT, &oldInterruptAction, nullptr);
            interruptInstalled_ = false;
        }
    }

    void handleSignalNotification()
    {
        std::uint8_t buffer[32];
        while (::read(pipeFileDescriptors_[0], buffer, sizeof buffer) > 0)
        {
        }
        callback_();
    }

    int pipeFileDescriptors_[2]{-1, -1};
    std::function<void()> callback_;
    std::unique_ptr<muduo::net::Channel> channel_;
    struct sigaction oldInterruptAction {};
    struct sigaction oldTerminateAction {};
    bool interruptInstalled_{false};
    bool terminateInstalled_{false};
};

ProcessSignalHandler::ProcessSignalHandler(
    muduo::net::EventLoop* eventLoop,
    std::function<void()> shutdownCallback)
    : impl_(std::make_unique<Impl>(eventLoop, std::move(shutdownCallback)))
{}

ProcessSignalHandler::~ProcessSignalHandler() = default;

} // namespace infrastructure
} // namespace treesem
