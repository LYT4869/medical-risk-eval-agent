#include <algorithm>
#include <cassert>
#include <future>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "service/InferenceScheduler.h"
#include "service/BlockingTaskScheduler.h"

int main()
{
    using Scheduler = treesem::service::InferenceScheduler;
    using SubmitResult = Scheduler::SubmitResult;

    std::mutex responsesMutex;
    std::vector<int> statuses;
    const http::AsyncResponder responder = [&](http::ResponseWriter writer) {
        http::HttpResponse response(false);
        writer(&response);
        std::lock_guard<std::mutex> lock(responsesMutex);
        statuses.push_back(static_cast<int>(response.getStatusCode()));
    };

    std::promise<void> firstStarted;
    std::promise<void> releaseFirst;
    std::shared_future<void> releaseSignal = releaseFirst.get_future().share();

    Scheduler scheduler(1, 1);
    assert(scheduler.schedule(
               [&]() -> http::ResponseWriter {
                   firstStarted.set_value();
                   releaseSignal.wait();
                   return [](http::HttpResponse* response) {
                       response->setStatusCode(http::HttpResponse::k200Ok);
                   };
               },
               responder) == SubmitResult::Accepted);
    firstStarted.get_future().wait();

    assert(scheduler.schedule(
               []() -> http::ResponseWriter {
                   return [](http::HttpResponse* response) {
                       response->setStatusCode(http::HttpResponse::k200Ok);
                   };
               },
               responder) == SubmitResult::Accepted);

    assert(scheduler.schedule([]() { return http::ResponseWriter{}; }, responder) ==
           SubmitResult::QueueFull);

    releaseFirst.set_value();
    scheduler.waitForIdle();

    {
        std::lock_guard<std::mutex> lock(responsesMutex);
        assert(statuses.size() == 3);
        assert(std::count(statuses.begin(), statuses.end(), 200) == 2);
        assert(std::count(statuses.begin(), statuses.end(), 503) == 1);
    }

    scheduler.shutdown();
    assert(scheduler.schedule([]() { return http::ResponseWriter{}; }, responder) ==
           SubmitResult::Stopped);
    {
        std::lock_guard<std::mutex> lock(responsesMutex);
        assert(statuses.back() == 503);
    }

    Scheduler failingScheduler(1, 1);
    assert(failingScheduler.schedule(
               []() -> http::ResponseWriter {
                   throw std::runtime_error("simulated inference failure");
               },
               responder) == SubmitResult::Accepted);
    failingScheduler.waitForIdle();
    {
        std::lock_guard<std::mutex> lock(responsesMutex);
        assert(statuses.back() == 500);
    }

    std::promise<void> databaseStarted;
    std::promise<void> releaseDatabase;
    auto databaseRelease = releaseDatabase.get_future().share();
    treesem::service::BlockingTaskScheduler databaseScheduler(
        1, 1, treesem::service::databaseSchedulerErrors());
    std::vector<std::string> databaseBodies;
    const http::AsyncResponder databaseResponder = [&](http::ResponseWriter writer) {
        http::HttpResponse response(false);
        writer(&response);
        std::lock_guard<std::mutex> lock(responsesMutex);
        databaseBodies.push_back(response.body());
    };
    assert(databaseScheduler.schedule(
        [&]() -> http::ResponseWriter {
            databaseStarted.set_value();
            databaseRelease.wait();
            return [](http::HttpResponse* response) {
                response->setStatusCode(http::HttpResponse::k200Ok);
            };
        }, databaseResponder) == SubmitResult::Accepted);
    databaseStarted.get_future().wait();
    assert(databaseScheduler.schedule(
        []() -> http::ResponseWriter {
            return [](http::HttpResponse* response) {
                response->setStatusCode(http::HttpResponse::k200Ok);
            };
        }, databaseResponder) == SubmitResult::Accepted);
    assert(databaseScheduler.schedule(
        []() { return http::ResponseWriter{}; }, databaseResponder) ==
        SubmitResult::QueueFull);
    {
        std::lock_guard<std::mutex> lock(responsesMutex);
        assert(databaseBodies.back().find("database_overloaded") != std::string::npos);
    }
    releaseDatabase.set_value();
    databaseScheduler.shutdown();
}
