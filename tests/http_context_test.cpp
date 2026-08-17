#include <cassert>
#include <string>

#include <muduo/net/Buffer.h>

#include "http/HttpContext.h"

int main()
{
    http::HttpContext context;
    muduo::net::Buffer buffer;

    const std::string postRequest =
        "POST /api/v1/echo HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "content-length: 13\r\n"
        "\r\n"
        "{\"value\":123}";
    buffer.append(postRequest);

    assert(context.parseRequest(&buffer, muduo::Timestamp::now()));
    assert(context.gotAll());
    assert(context.request().method() == http::HttpRequest::kPost);
    assert(context.request().path() == "/api/v1/echo");
    assert(context.request().contentLength() == 13);
    assert(context.request().getBody() == "{\"value\":123}");

    context.reset();
    assert(!context.gotAll());
    assert(context.request().contentLength() == 0);
    assert(context.request().getBody().empty());

    const std::string getRequest =
        "GET /health HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "\r\n";
    buffer.append(getRequest);
    assert(context.parseRequest(&buffer, muduo::Timestamp::now()));
    assert(context.gotAll());
    assert(context.request().method() == http::HttpRequest::kGet);
    assert(context.request().path() == "/health");
    assert(context.request().getBody().empty());
}
