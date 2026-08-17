#include <cassert>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "config/TreeSemServerConfig.h"

namespace
{

class ScopedEnvironment
{
public:
    explicit ScopedEnvironment(std::vector<std::string> names)
        : names_(std::move(names))
    {
        for (const std::string& name : names_)
        {
            const char* value = std::getenv(name.c_str());
            original_.push_back(
                value == nullptr ? std::optional<std::string>{}
                                 : std::optional<std::string>{value});
            ::unsetenv(name.c_str());
        }
    }

    ~ScopedEnvironment()
    {
        for (std::size_t i = 0; i < names_.size(); ++i)
        {
            if (original_[i].has_value())
            {
                ::setenv(names_[i].c_str(), original_[i]->c_str(), 1);
            }
            else
            {
                ::unsetenv(names_[i].c_str());
            }
        }
    }

private:
    std::vector<std::string> names_;
    std::vector<std::optional<std::string>> original_;
};

template <typename Function>
void assertInvalid(Function function)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main()
{
    ScopedEnvironment environment({
        "TREESEM_MODEL_ADAPTER_URL",
        "TREESEM_MODEL_CONNECT_TIMEOUT_MS",
        "TREESEM_MODEL_TIMEOUT_MS",
        "TREESEM_INFERENCE_WORKERS",
        "TREESEM_INFERENCE_QUEUE_CAPACITY",
    });

    char program[] = "treesem_server";
    char* defaultArguments[] = {program, nullptr};
    treesem::config::TreeSemServerConfig config =
        treesem::config::TreeSemServerConfig::load(1, defaultArguments);
    assert(config.listenPort == 8080);
    assert(config.modelAdapterUrl == "http://127.0.0.1:18081/v1/predict");
    assert(config.modelConnectTimeoutMs == 500);
    assert(config.modelRequestTimeoutMs == 5000);
    assert(config.inferenceWorkerCount == 2);
    assert(config.inferenceQueueCapacity == 32);

    ::setenv("TREESEM_MODEL_ADAPTER_URL", "http://adapter/v1/predict", 1);
    ::setenv("TREESEM_MODEL_CONNECT_TIMEOUT_MS", "250", 1);
    ::setenv("TREESEM_MODEL_TIMEOUT_MS", "3000", 1);
    ::setenv("TREESEM_INFERENCE_WORKERS", "4", 1);
    ::setenv("TREESEM_INFERENCE_QUEUE_CAPACITY", "16", 1);
    char port[] = "19090";
    char* configuredArguments[] = {program, port, nullptr};
    config = treesem::config::TreeSemServerConfig::load(
        2, configuredArguments);
    assert(config.listenPort == 19090);
    assert(config.modelAdapterUrl == "http://adapter/v1/predict");
    assert(config.modelConnectTimeoutMs == 250);
    assert(config.modelRequestTimeoutMs == 3000);
    assert(config.inferenceWorkerCount == 4);
    assert(config.inferenceQueueCapacity == 16);

    char badPort[] = "8080junk";
    char* badPortArguments[] = {program, badPort, nullptr};
    assertInvalid([&]() {
        treesem::config::TreeSemServerConfig::load(2, badPortArguments);
    });

    char highPort[] = "70000";
    char* highPortArguments[] = {program, highPort, nullptr};
    assertInvalid([&]() {
        treesem::config::TreeSemServerConfig::load(2, highPortArguments);
    });

    char extra[] = "extra";
    char* extraArguments[] = {program, port, extra, nullptr};
    assertInvalid([&]() {
        treesem::config::TreeSemServerConfig::load(3, extraArguments);
    });

    ::setenv("TREESEM_INFERENCE_WORKERS", "0", 1);
    assertInvalid([&]() {
        treesem::config::TreeSemServerConfig::load(1, defaultArguments);
    });
    ::setenv("TREESEM_INFERENCE_WORKERS", "abc", 1);
    assertInvalid([&]() {
        treesem::config::TreeSemServerConfig::load(1, defaultArguments);
    });
}
