#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cppconn/connection.h>

namespace treesem
{
namespace infrastructure
{

struct MySqlConnectionConfig
{
    std::string host;
    int port{3306};
    std::string database;
    std::string user;
    std::string password;
    std::size_t poolSize{8};
    std::chrono::milliseconds acquireTimeout{500};
    int connectTimeoutSeconds{1};
    int readTimeoutSeconds{2};
    int writeTimeoutSeconds{2};
};

class MySqlConnectionPool
{
public:
    class Lease
    {
    public:
        Lease() = default;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease();

        sql::Connection& connection() const;
        void invalidate() noexcept;

    private:
        friend class MySqlConnectionPool;
        Lease(MySqlConnectionPool* pool,
              std::unique_ptr<sql::Connection> connection);
        void release() noexcept;

        MySqlConnectionPool* pool_{nullptr};
        std::unique_ptr<sql::Connection> connection_;
        bool valid_{true};
    };

    explicit MySqlConnectionPool(MySqlConnectionConfig config);
    ~MySqlConnectionPool();
    MySqlConnectionPool(const MySqlConnectionPool&) = delete;
    MySqlConnectionPool& operator=(const MySqlConnectionPool&) = delete;

    Lease acquire();
    void shutdown();
    std::size_t size() const noexcept;

private:
    std::unique_ptr<sql::Connection> createConnection() const;
    void release(std::unique_ptr<sql::Connection> connection, bool valid) noexcept;

    MySqlConnectionConfig config_;
    mutable std::mutex mutex_;
    std::condition_variable availableCondition_;
    std::condition_variable drainedCondition_;
    std::vector<std::unique_ptr<sql::Connection>> available_;
    std::size_t liveConnections_{0};
    std::size_t borrowedConnections_{0};
    bool stopping_{false};
};

} // namespace infrastructure
} // namespace treesem
