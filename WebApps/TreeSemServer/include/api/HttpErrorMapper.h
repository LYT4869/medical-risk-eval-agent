#pragma once

#include <string>

#include "api/ApiException.h"
#include "http/AsyncHttp.h"
#include "http/HttpResponse.h"
#include "model/ModelException.h"
#include "application/BusinessException.h"
#include "client/IAgentClient.h"
#include "application/AuthService.h"

namespace treesem
{
namespace api
{

class HttpErrorMapper
{
public:
    static http::ResponseWriter json(
        http::HttpResponse::HttpStatusCode status,
        std::string statusMessage,
        std::string body);

    static http::ResponseWriter success(std::string body);
    static http::ResponseWriter from(const ApiException& error);
    static http::ResponseWriter from(const model::ModelException& error);
    static http::ResponseWriter from(const application::BusinessException& error);
    static http::ResponseWriter from(const client::AgentClientException& error);
    static http::ResponseWriter from(const application::AuthException& error);
    static http::ResponseWriter internalError();

private:
    static http::ResponseWriter error(
        http::HttpResponse::HttpStatusCode status,
        const std::string& statusMessage,
        const std::string& code,
        const std::string& safeMessage);
};

} // namespace api
} // namespace treesem
