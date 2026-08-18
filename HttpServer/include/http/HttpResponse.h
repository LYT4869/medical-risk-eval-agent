#pragma once

#include <muduo/net/TcpServer.h>

namespace http
{

class HttpResponse 
{
public:
    enum HttpStatusCode
    {
        kUnknown,
        k200Ok = 200,
        k201Created = 201,
        k202Accepted = 202,
        k204NoContent = 204,
        k301MovedPermanently = 301,
        k400BadRequest = 400,
        k401Unauthorized = 401,
        k403Forbidden = 403,
        k404NotFound = 404,
        k409Conflict = 409,
        k429TooManyRequests = 429,
        k500InternalServerError = 500,
        k502BadGateway = 502,
        k503ServiceUnavailable = 503,
        k504GatewayTimeout = 504,
    };

    HttpResponse(bool close = true)
        : statusCode_(kUnknown)
        , closeConnection_(close)
    {}

    void setVersion(std::string version)
    { httpVersion_ = version; }
    void setStatusCode(HttpStatusCode code)
    { statusCode_ = code; }

    HttpStatusCode getStatusCode() const
    { return statusCode_; }

    void setStatusMessage(const std::string message)
    { statusMessage_ = message; }

    void setCloseConnection(bool on)
    { closeConnection_ = on; }

    bool closeConnection() const
    { return closeConnection_; }

    size_t bodySize() const
    { return body_.size(); }

    const std::string& body() const
    { return body_; }
    
    void setContentType(const std::string& contentType)
    { addHeader("Content-Type", contentType); }

    void setContentLength(uint64_t length)
    { addHeader("Content-Length", std::to_string(length)); }

    void addHeader(const std::string& key, const std::string& value)
    {
        if (key != "Set-Cookie") headers_.erase(key);
        headers_.emplace(key, value);
    }

    std::string getHeader(const std::string& key) const
    {
        const auto found = headers_.find(key);
        return found == headers_.end() ? std::string() : found->second;
    }
    
    void setBody(const std::string& body)
    { 
        body_ = body;
        // body_ += "\0";
    }

    void setStatusLine(const std::string& version,
                         HttpStatusCode statusCode,
                         const std::string& statusMessage);

    void setErrorHeader(){}

    void appendToBuffer(muduo::net::Buffer* outputBuf) const;
private:
    // Middleware and error mappers may construct a response before it reaches
    // the normal router responder.  Keep those responses valid on the wire;
    // the responder will still replace this with the request version.
    std::string                        httpVersion_{"HTTP/1.1"};
    HttpStatusCode                     statusCode_;
    std::string                        statusMessage_;
    bool                               closeConnection_;
    std::multimap<std::string, std::string> headers_;
    std::string                        body_;
    bool                               isFile_;
};

} // namespace http
