#include "infrastructure/persistence/MySqlConnectionPool.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <mysql_driver.h>

#include "application/BusinessException.h"

namespace treesem
{
namespace infrastructure
{

MySqlConnectionPool::Lease::Lease(
    MySqlConnectionPool* pool,
    std::unique_ptr<sql::Connection> connection)
    : pool_(pool)
    , connection_(std::move(connection))
{}

MySqlConnectionPool::Lease::Lease(Lease&& other) noexcept
    : pool_(other.pool_)
    , connection_(std::move(other.connection_))
    , valid_(other.valid_)
{
    other.pool_ = nullptr;
}

MySqlConnectionPool::Lease& MySqlConnectionPool::Lease::operator=(Lease&& other) noexcept
{
    if (this != &other)
    {
        release();
        pool_ = other.pool_;
        connection_ = std::move(other.connection_);
        valid_ = other.valid_;
        other.pool_ = nullptr;
    }
    return *this;
}

MySqlConnectionPool::Lease::~Lease()
{
    release();
}

sql::Connection& MySqlConnectionPool::Lease::connection() const
{
    if (!connection_) throw std::logic_error("empty MySQL connection lease");
    return *connection_;
}

void MySqlConnectionPool::Lease::invalidate() noexcept
{
    valid_ = false;
}

void MySqlConnectionPool::Lease::release() noexcept
{
    if (pool_ != nullptr && connection_)
        pool_->release(std::move(connection_), valid_);
    pool_ = nullptr;
}

MySqlConnectionPool::MySqlConnectionPool(MySqlConnectionConfig config)
    : config_(std::move(config))
{
    if (config_.host.empty() || config_.database.empty() || config_.user.empty() ||
        config_.poolSize == 0 || config_.acquireTimeout.count() <= 0)
    {
        throw std::invalid_argument("invalid MySQL connection pool configuration");
    }
    available_.reserve(config_.poolSize);
    try
    {
        for (std::size_t index = 0; index < config_.poolSize; ++index)
        {
            available_.push_back(createConnection());
            ++liveConnections_;
        }
    }
    catch (...)
    {
        available_.clear();
        liveConnections_ = 0;
        throw;
    }
}

MySqlConnectionPool::~MySqlConnectionPool()
{
    shutdown();
}

std::unique_ptr<sql::Connection> MySqlConnectionPool::createConnection() const
{
    try
    {
        sql::ConnectOptionsMap options;
        options["hostName"] = config_.host;
        options["port"] = config_.port;
        options["userName"] = config_.user;
        options["password"] = config_.password;
        options["schema"] = config_.database;
        options["OPT_CONNECT_TIMEOUT"] = config_.connectTimeoutSeconds;
        options["OPT_READ_TIMEOUT"] = config_.readTimeoutSeconds;
        options["OPT_WRITE_TIMEOUT"] = config_.writeTimeoutSeconds;
        sql::mysql::MySQL_Driver* driver = sql::mysql::get_mysql_driver_instance();
        std::unique_ptr<sql::Connection> connection(driver->connect(options));
        connection->setSchema(config_.database);
        connection->setAutoCommit(true);
        std::unique_ptr<sql::Statement> statement(connection->createStatement());
        statement->execute("SET NAMES utf8mb4");
        statement->execute("SET time_zone = '+00:00'");
        statement->execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED");
        std::unique_ptr<sql::ResultSet> validation(
            statement->executeQuery("SELECT 1"));
        if (!validation->next())
        {
            throw application::BusinessException(
                application::BusinessException::Kind::DatabaseUnavailable,
                "MySQL validation query returned no row");
        }
        return connection;
    }
    catch (const sql::SQLException&)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::DatabaseUnavailable,
            "failed to establish a MySQL connection");
    }
}

MySqlConnectionPool::Lease MySqlConnectionPool::acquire()
{
    const auto deadline = std::chrono::steady_clock::now() + config_.acquireTimeout;
    std::unique_lock<std::mutex> lock(mutex_);
    while (true)
    {
        if (stopping_)
        {
            throw application::BusinessException(
                application::BusinessException::Kind::DatabaseUnavailable,
                "MySQL connection pool is stopping");
        }
        if (!available_.empty())
        {
            auto connection = std::move(available_.back());
            available_.pop_back();
            ++borrowedConnections_;
            lock.unlock();
            try
            {
                bool healthy = false;
                try
                {
                    if (connection && !connection->isClosed())
                    {
                        std::unique_ptr<sql::Statement> validation(
                            connection->createStatement());
                        std::unique_ptr<sql::ResultSet> result(
                            validation->executeQuery("SELECT 1"));
                        healthy = result->next();
                    }
                }
                catch (...) {}
                if (!healthy)
                {
                    connection.reset();
                    connection = createConnection(); // one bounded reconnect attempt
                }
                return Lease(this, std::move(connection));
            }
            catch (...)
            {
                lock.lock();
                --borrowedConnections_;
                --liveConnections_;
                drainedCondition_.notify_all();
                throw;
            }
        }
        if (liveConnections_ < config_.poolSize)
        {
            ++liveConnections_;
            ++borrowedConnections_;
            lock.unlock();
            try
            {
                return Lease(this, createConnection());
            }
            catch (...)
            {
                lock.lock();
                --liveConnections_;
                --borrowedConnections_;
                drainedCondition_.notify_all();
                throw;
            }
        }
        if (availableCondition_.wait_until(lock, deadline) == std::cv_status::timeout)
        {
            throw application::BusinessException(
                application::BusinessException::Kind::DatabaseBusy,
                "timed out waiting for a MySQL connection");
        }
    }
}

void MySqlConnectionPool::release(
    std::unique_ptr<sql::Connection> connection,
    bool valid) noexcept
{
    try
    {
        if (valid && connection)
        {
            if (!connection->getAutoCommit())
            {
                connection->rollback();
                connection->setAutoCommit(true);
            }
            if (connection->isClosed()) valid = false;
        }
    }
    catch (...) { valid = false; }

    if (!valid) connection.reset();
    std::lock_guard<std::mutex> lock(mutex_);
    if (borrowedConnections_ > 0) --borrowedConnections_;
    if (!connection)
    {
        if (liveConnections_ > 0) --liveConnections_;
    }
    else if (!stopping_)
    {
        available_.push_back(std::move(connection));
    }
    else
    {
        if (liveConnections_ > 0) --liveConnections_;
    }
    availableCondition_.notify_one();
    drainedCondition_.notify_all();
}

void MySqlConnectionPool::shutdown()
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopping_)
    {
        drainedCondition_.wait(lock, [&]() { return borrowedConnections_ == 0; });
        return;
    }
    stopping_ = true;
    availableCondition_.notify_all();
    drainedCondition_.wait(lock, [&]() { return borrowedConnections_ == 0; });
    liveConnections_ -= available_.size();
    available_.clear();
}

std::size_t MySqlConnectionPool::size() const noexcept
{
    return config_.poolSize;
}

} // namespace infrastructure
} // namespace treesem
