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
        "TREESEM_MODEL_BACKEND",
        "TREESEM_SERVING_BUNDLE_DIR",
        "TREESEM_STORAGE_BACKEND",
        "TREESEM_DB_HOST",
        "TREESEM_DB_PORT",
        "TREESEM_DB_NAME",
        "TREESEM_DB_USER",
        "TREESEM_DB_PASSWORD",
        "TREESEM_DB_POOL_SIZE",
        "TREESEM_DB_ACQUIRE_TIMEOUT_MS",
        "TREESEM_DB_CONNECT_TIMEOUT_MS",
        "TREESEM_DB_READ_TIMEOUT_MS",
        "TREESEM_DB_WRITE_TIMEOUT_MS",
        "TREESEM_DATABASE_WORKERS",
        "TREESEM_DATABASE_QUEUE_CAPACITY",
        "TREESEM_SESSION_TTL_SECONDS",
        "TREESEM_COOKIE_SECURE",
        "TREESEM_REFRESH_COOKIE_SECURE",
        "TREESEM_AUTH_MODE",
        "TREESEM_ACCESS_JWT_SECRET",
        "TREESEM_CAPABILITY_JWT_SECRET",
        "TREESEM_AGENT_SERVICE_SECRET",
        "TREESEM_DEPLOYMENT_ENV",
    });

    char program[] = "treesem_server";
    char* defaultArguments[] = {program, nullptr};
    ::setenv("TREESEM_STORAGE_BACKEND", "memory", 1);
    ::setenv("TREESEM_AUTH_MODE", "development", 1);
    treesem::config::TreeSemServerConfig config =
        treesem::config::TreeSemServerConfig::load(1, defaultArguments);
    assert(config.listenPort == 8080);
    assert(config.modelAdapterUrl == "http://127.0.0.1:18081/v1/predict");
    assert(config.modelConnectTimeoutMs == 500);
    assert(config.modelRequestTimeoutMs == 5000);
    assert(config.inferenceWorkerCount == 2);
    assert(config.inferenceQueueCapacity == 32);
    assert(config.modelBackend == treesem::config::ModelBackend::OnnxFallback);
    assert(config.servingBundleDirectory.empty());
    assert(config.storageBackend == treesem::config::StorageBackend::Memory);
    assert(config.databaseWorkerCount == 4);
    assert(config.databaseQueueCapacity == 64);
    assert(config.databaseHost == "127.0.0.1");
    assert(config.databasePort == 3307);
    assert(config.databaseName == "treesem");
    assert(config.databaseUser == "treesem_app");
    assert(config.databasePoolSize == 8);
    assert(config.databaseAcquireTimeoutMs == 500);
    assert(config.sessionTtlSeconds == 3600);
    assert(!config.cookieSecure);
    assert(!config.refreshCookieSecure);

    ::setenv("TREESEM_MODEL_ADAPTER_URL", "http://adapter/v1/predict", 1);
    ::setenv("TREESEM_MODEL_CONNECT_TIMEOUT_MS", "250", 1);
    ::setenv("TREESEM_MODEL_TIMEOUT_MS", "3000", 1);
    ::setenv("TREESEM_INFERENCE_WORKERS", "4", 1);
    ::setenv("TREESEM_INFERENCE_QUEUE_CAPACITY", "16", 1);
    ::setenv("TREESEM_MODEL_BACKEND", "shadow", 1);
    ::setenv("TREESEM_SERVING_BUNDLE_DIR", "/safe/example/bundle", 1);
    ::setenv("TREESEM_DATABASE_WORKERS", "3", 1);
    ::setenv("TREESEM_DATABASE_QUEUE_CAPACITY", "12", 1);
    ::setenv("TREESEM_DB_HOST", "db.internal", 1);
    ::setenv("TREESEM_DB_PORT", "3308", 1);
    ::setenv("TREESEM_DB_NAME", "treesem_test", 1);
    ::setenv("TREESEM_DB_USER", "test_user", 1);
    ::setenv("TREESEM_DB_POOL_SIZE", "6", 1);
    ::setenv("TREESEM_DB_ACQUIRE_TIMEOUT_MS", "250", 1);
    ::setenv("TREESEM_SESSION_TTL_SECONDS", "7200", 1);
    ::setenv("TREESEM_COOKIE_SECURE", "true", 1);
    ::setenv("TREESEM_REFRESH_COOKIE_SECURE", "true", 1);
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
    assert(config.modelBackend == treesem::config::ModelBackend::Shadow);
    assert(config.servingBundleDirectory == "/safe/example/bundle");
    assert(config.databaseWorkerCount == 3);
    assert(config.databaseQueueCapacity == 12);
    assert(config.databaseHost == "db.internal");
    assert(config.databasePort == 3308);
    assert(config.databaseName == "treesem_test");
    assert(config.databaseUser == "test_user");
    assert(config.databasePoolSize == 6);
    assert(config.databaseAcquireTimeoutMs == 250);
    assert(config.sessionTtlSeconds == 7200);
    assert(config.cookieSecure);
    assert(config.refreshCookieSecure);

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
    ::setenv("TREESEM_INFERENCE_WORKERS", "2", 1);
    ::setenv("TREESEM_MODEL_BACKEND", "unknown", 1);
    assertInvalid([&]() {
        treesem::config::TreeSemServerConfig::load(1, defaultArguments);
    });
    ::setenv("TREESEM_MODEL_BACKEND", "remote", 1);
    ::setenv("TREESEM_AUTH_MODE", "required", 1);
    ::setenv("TREESEM_ACCESS_JWT_SECRET",
             "access-secret-for-config-test-at-least-32-bytes", 1);
    ::setenv("TREESEM_CAPABILITY_JWT_SECRET",
             "capability-secret-for-config-test-at-least-32-bytes", 1);
    ::setenv("TREESEM_AGENT_SERVICE_SECRET",
             "agent-secret-for-config-test-at-least-32-bytes", 1);
    (void)treesem::config::TreeSemServerConfig::load(1, defaultArguments);
    ::setenv("TREESEM_AGENT_SERVICE_SECRET",
             "access-secret-for-config-test-at-least-32-bytes", 1);
    assertInvalid([&]() {
        treesem::config::TreeSemServerConfig::load(1, defaultArguments);
    });
    ::setenv("TREESEM_AGENT_SERVICE_SECRET",
             "agent-secret-for-config-test-at-least-32-bytes", 1);
    ::setenv("TREESEM_DEPLOYMENT_ENV", "production", 1);
    ::setenv("TREESEM_COOKIE_SECURE", "true", 1);
    ::setenv("TREESEM_REFRESH_COOKIE_SECURE", "false", 1);
    assertInvalid([&]() {
        treesem::config::TreeSemServerConfig::load(1, defaultArguments);
    });
    ::setenv("TREESEM_DEPLOYMENT_ENV", "local", 1);
    ::setenv("TREESEM_AUTH_MODE", "development", 1);
    ::setenv("TREESEM_STORAGE_BACKEND", "mysql", 1);
    ::unsetenv("TREESEM_DB_PASSWORD");
    assertInvalid([&]() {
        treesem::config::TreeSemServerConfig::load(1, defaultArguments);
    });
}
