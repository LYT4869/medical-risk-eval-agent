#include "service/PythonModelProxy.h"

#include <cmath>
#include <utility>

#include <nlohmann/json.hpp>

#include "client/PythonModelClient.h"

namespace treesem
{
namespace service
{
namespace
{

using Json = nlohmann::json;

http::ResponseWriter jsonResponse(http::HttpResponse::HttpStatusCode status,
                                  std::string statusMessage,
                                  std::string body)
{
    return [status,
            statusMessage = std::move(statusMessage),
            body = std::move(body)](http::HttpResponse* response) {
        response->setStatusCode(status);
        response->setStatusMessage(statusMessage);
        response->setContentType("application/json; charset=utf-8");
        response->setBody(body);
    };
}

http::ResponseWriter errorResponse(http::HttpResponse::HttpStatusCode status,
                                   const std::string& statusMessage,
                                   const std::string& error)
{
    return jsonResponse(
        status,
        statusMessage,
        Json{{"error", error}}.dump());
}

std::string statusMessage(long status)
{
    switch (status)
    {
    case 200:
        return "OK";
    case 400:
        return "Bad Request";
    case 404:
        return "Not Found";
    default:
        return "Model Adapter Response";
    }
}

bool isValidRequest(const Json& request)
{
    if (!request.is_object())
    {
        return false;
    }
    const bool hasIndex = request.contains("sample_index");
    const bool hasFeatures = request.contains("preprocessed_features");
    if (hasIndex == hasFeatures)
    {
        return false;
    }
    if (hasIndex)
    {
        return request["sample_index"].is_number_integer() &&
               request["sample_index"].get<long long>() >= 0;
    }
    const Json& features = request["preprocessed_features"];
    if (!features.is_array() || features.empty())
    {
        return false;
    }
    for (const auto& value : features)
    {
        if (!value.is_number() || !std::isfinite(value.get<double>()))
        {
            return false;
        }
    }
    return true;
}

} // namespace

PythonModelProxy::PythonModelProxy(const client::IModelAdapterClient& client)
    : client_(client)
{}

http::ResponseWriter PythonModelProxy::predict(const std::string& requestBody) const
{
    Json request;
    try
    {
        request = Json::parse(requestBody);
    }
    catch (const Json::exception&)
    {
        return errorResponse(
            http::HttpResponse::k400BadRequest, "Bad Request", "invalid_json");
    }
    if (!isValidRequest(request))
    {
        return errorResponse(
            http::HttpResponse::k400BadRequest, "Bad Request", "invalid_request");
    }

    try
    {
        client::ModelAdapterResponse downstream = client_.predict(request.dump());
        if (!Json::accept(downstream.body))
        {
            return errorResponse(
                http::HttpResponse::k502BadGateway,
                "Bad Gateway",
                "invalid_model_adapter_response");
        }

        if (downstream.statusCode >= 500)
        {
            return errorResponse(
                http::HttpResponse::k502BadGateway,
                "Bad Gateway",
                "model_adapter_failed");
        }
        return jsonResponse(
            static_cast<http::HttpResponse::HttpStatusCode>(downstream.statusCode),
            statusMessage(downstream.statusCode),
            std::move(downstream.body));
    }
    catch (const client::ModelAdapterException& error)
    {
        if (error.kind() == client::ModelAdapterException::Kind::Timeout)
        {
            return errorResponse(
                http::HttpResponse::k504GatewayTimeout,
                "Gateway Timeout",
                "model_adapter_timeout");
        }
        return errorResponse(
            http::HttpResponse::k502BadGateway,
            "Bad Gateway",
            "model_adapter_unavailable");
    }
}

} // namespace service
} // namespace treesem
