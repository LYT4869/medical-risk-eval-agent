#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

#include "http/AsyncHttp.h"

int main()
{
    std::atomic<int> responseCount{0};
    http::AsyncResponder responder = http::makeOneShotResponder(
        [&responseCount](http::ResponseWriter writer) {
            http::HttpResponse response(false);
            writer(&response);
            assert(response.getStatusCode() == http::HttpResponse::k200Ok);
            ++responseCount;
        });

    std::vector<std::thread> callers;
    for (int index = 0; index < 32; ++index)
    {
        callers.emplace_back([responder]() {
            responder([](http::HttpResponse* response) {
                response->setStatusCode(http::HttpResponse::k200Ok);
            });
        });
    }
    for (std::thread& caller : callers)
    {
        caller.join();
    }
    assert(responseCount.load() == 1);
}
