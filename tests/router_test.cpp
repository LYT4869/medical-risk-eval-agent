#include <cassert>
#include <string>

#include "http/HttpRequest.h"
#include "http/HttpResponse.h"
#include "router/Router.h"

int main()
{
    http::router::Router router;
    bool called = false;
    std::string pathParameter;

    router.addRegexCallback(
        http::HttpRequest::kGet,
        "/predictions/:id",
        [&](const http::HttpRequest& request, http::HttpResponse*) {
            called = true;
            pathParameter = request.getPathParameters("id");
        });

    http::HttpRequest request;
    const std::string method = "GET";
    request.setMethod(method.data(), method.data() + method.size());
    const std::string path = "/predictions/p1001";
    request.setPath(path.data(), path.data() + path.size());

    http::HttpResponse response;
    assert(router.route(request, &response));
    assert(called);
    assert(pathParameter == "p1001");

    bool asyncCalled = false;
    router.registerAsyncCallback(
        http::HttpRequest::kPost,
        "/predict",
        [&](http::HttpRequest asyncRequest, http::AsyncResponder responder) {
            asyncCalled = asyncRequest.path() == "/predict";
            responder([](http::HttpResponse* asyncResponse) {
                asyncResponse->setStatusCode(http::HttpResponse::k202Accepted);
            });
        });

    http::HttpRequest asyncRequest;
    const std::string postMethod = "POST";
    asyncRequest.setMethod(postMethod.data(), postMethod.data() + postMethod.size());
    const std::string predictPath = "/predict";
    asyncRequest.setPath(predictPath.data(), predictPath.data() + predictPath.size());
    assert(router.hasAsyncCallback(asyncRequest.method(), asyncRequest.path()));

    int asyncStatus = 0;
    const http::AsyncResponder responder = [&](http::ResponseWriter writer) {
        http::HttpResponse asyncResponse(false);
        writer(&asyncResponse);
        asyncStatus = static_cast<int>(asyncResponse.getStatusCode());
    };
    assert(router.routeAsync(asyncRequest, responder));
    assert(asyncCalled);
    assert(asyncStatus == 202);

    bool dynamicAsyncCalled = false;
    router.addAsyncRoute(
        http::HttpRequest::kGet,
        "/predictions/:prediction_id/explanation",
        [&](http::HttpRequest routed, http::AsyncResponder asyncResponder) {
            dynamicAsyncCalled =
                routed.getPathParameters("prediction_id") == "pred_123";
            asyncResponder([](http::HttpResponse* asyncResponse) {
                asyncResponse->setStatusCode(http::HttpResponse::k200Ok);
            });
        });
    http::HttpRequest dynamicRequest;
    dynamicRequest.setMethod(method.data(), method.data() + method.size());
    const std::string dynamicPath = "/predictions/pred_123/explanation";
    dynamicRequest.setPath(dynamicPath.data(), dynamicPath.data() + dynamicPath.size());
    assert(router.hasAsyncCallback(dynamicRequest.method(), dynamicRequest.path()));
    assert(router.routeAsync(dynamicRequest, responder));
    assert(dynamicAsyncCalled);

    bool rejectedDuplicate = false;
    try
    {
        router.addAsyncRoute(
            http::HttpRequest::kGet,
            "/predictions/:prediction_id/explanation",
            [](http::HttpRequest, http::AsyncResponder) {});
    }
    catch (const std::invalid_argument&)
    {
        rejectedDuplicate = true;
    }
    assert(rejectedDuplicate);

    bool rejectedEquivalent = false;
    try
    {
        router.addAsyncRoute(
            http::HttpRequest::kGet,
            "/predictions/:another_name/explanation",
            [](http::HttpRequest, http::AsyncResponder) {});
    }
    catch (const std::invalid_argument&)
    {
        rejectedEquivalent = true;
    }
    assert(rejectedEquivalent);

    bool rejectedInvalidTemplate = false;
    try
    {
        router.addAsyncRoute(
            http::HttpRequest::kGet,
            "/predictions/prefix:prediction_id",
            [](http::HttpRequest, http::AsyncResponder) {});
    }
    catch (const std::invalid_argument&)
    {
        rejectedInvalidTemplate = true;
    }
    assert(rejectedInvalidTemplate);
}
