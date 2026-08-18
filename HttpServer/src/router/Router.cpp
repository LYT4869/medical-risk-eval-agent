#include "../../include/router/Router.h"
#include <algorithm>
#include <cctype>
#include <muduo/base/Logging.h>
#include <set>

namespace http
{
namespace router
{

namespace
{

std::string routeShape(const std::string& path)
{
    std::string shape;
    for (std::size_t index = 0; index < path.size();)
    {
        if (path[index] != ':')
        {
            shape.push_back(path[index++]);
            continue;
        }
        shape.push_back(':');
        ++index;
        while (index < path.size() && path[index] != '/') ++index;
    }
    return shape;
}

template<typename Route>
bool sameDynamicRoute(const Route& route,
                      HttpRequest::Method method,
                      const std::string& path)
{
    return route.method_ == method &&
        routeShape(route.pathPattern_) == routeShape(path);
}

} // namespace

void Router::registerHandler(HttpRequest::Method method, const std::string &path, HandlerPtr handler)
{
    RouteKey key{method, path};
    if (!handler || handlers_.count(key) != 0 || callbacks_.count(key) != 0 ||
        asyncCallbacks_.count(key) != 0)
    {
        throw std::invalid_argument("invalid or duplicate route");
    }
    handlers_.emplace(std::move(key), std::move(handler));
}

void Router::registerCallback(HttpRequest::Method method, const std::string &path, const HandlerCallback &callback)
{
    RouteKey key{method, path};
    if (!callback || handlers_.count(key) != 0 || callbacks_.count(key) != 0 ||
        asyncCallbacks_.count(key) != 0)
    {
        throw std::invalid_argument("invalid or duplicate route");
    }
    callbacks_.emplace(std::move(key), callback);
}

void Router::registerAsyncCallback(HttpRequest::Method method,
                                   const std::string& path,
                                   const AsyncHttpCallback& callback)
{
    RouteKey key{method, path};
    if (!callback || handlers_.count(key) != 0 || callbacks_.count(key) != 0 ||
        asyncCallbacks_.count(key) != 0)
    {
        throw std::invalid_argument("invalid or duplicate route");
    }
    asyncCallbacks_.emplace(std::move(key), callback);
}

void Router::addAsyncRoute(HttpRequest::Method method,
                           const std::string& path,
                           const AsyncHttpCallback& callback)
{
    if (!callback)
    {
        throw std::invalid_argument("async route callback must not be empty");
    }
    const auto duplicate = std::find_if(
        asyncRegexCallbacks_.begin(), asyncRegexCallbacks_.end(),
        [&](const AsyncRouteCallbackObj& route) {
            return sameDynamicRoute(route, method, path);
        });
    if (duplicate != asyncRegexCallbacks_.end())
    {
        throw std::invalid_argument("duplicate async route");
    }
    asyncRegexCallbacks_.emplace_back(method, path, compilePath(path), callback);
}

void Router::addRegexHandler(HttpRequest::Method method,
                             const std::string& path,
                             HandlerPtr handler)
{
    if (!handler || std::any_of(
            regexHandlers_.begin(), regexHandlers_.end(),
            [&](const RouteHandlerObj& route) {
                return sameDynamicRoute(route, method, path);
            }))
    {
        throw std::invalid_argument("invalid or duplicate dynamic route");
    }
    regexHandlers_.emplace_back(method, path, compilePath(path), std::move(handler));
}

void Router::addRegexCallback(HttpRequest::Method method,
                              const std::string& path,
                              const HandlerCallback& callback)
{
    if (!callback || std::any_of(
            regexCallbacks_.begin(), regexCallbacks_.end(),
            [&](const RouteCallbackObj& route) {
                return sameDynamicRoute(route, method, path);
            }))
    {
        throw std::invalid_argument("invalid or duplicate dynamic route");
    }
    regexCallbacks_.emplace_back(method, path, compilePath(path), callback);
}

bool Router::hasAsyncCallback(HttpRequest::Method method, const std::string& path) const
{
    if (asyncCallbacks_.find(RouteKey{method, path}) != asyncCallbacks_.end())
    {
        return true;
    }
    return std::any_of(
        asyncRegexCallbacks_.begin(), asyncRegexCallbacks_.end(),
        [&](const AsyncRouteCallbackObj& route) {
            return route.method_ == method && std::regex_match(path, route.pathRegex_);
        });
}

std::optional<std::string> Router::matchingRoutePattern(
    HttpRequest::Method method, const std::string& path) const
{
    const RouteKey key{method, path};
    if (handlers_.count(key) != 0 || callbacks_.count(key) != 0 ||
        asyncCallbacks_.count(key) != 0)
    {
        return path;
    }
    const auto dynamicMatch = [&](const auto& routes) -> std::optional<std::string> {
        for (const auto& route : routes)
        {
            if (route.method_ == method && std::regex_match(path, route.pathRegex_))
            {
                return route.pathPattern_;
            }
        }
        return std::nullopt;
    };
    if (auto result = dynamicMatch(asyncRegexCallbacks_)) return result;
    if (auto result = dynamicMatch(regexHandlers_)) return result;
    return dynamicMatch(regexCallbacks_);
}

bool Router::routeAsync(const HttpRequest& req, const AsyncResponder& responder) const
{
    const auto callback = asyncCallbacks_.find(RouteKey{req.method(), req.path()});
    if (callback != asyncCallbacks_.end())
    {
        callback->second(req, responder);
        return true;
    }
    for (const auto& route : asyncRegexCallbacks_)
    {
        std::smatch match;
        const std::string path = req.path();
        if (route.method_ == req.method() &&
            std::regex_match(path, match, route.pathRegex_))
        {
            HttpRequest routedRequest(req);
            extractPathParameters(match, route.parameterNames_, routedRequest);
            route.callback_(std::move(routedRequest), responder);
            return true;
        }
    }
    return false;
}

Router::CompiledPath Router::compilePath(const std::string& pathPattern)
{
    if (pathPattern.empty() || pathPattern.front() != '/')
    {
        throw std::invalid_argument("route path must start with /");
    }
    std::string expression = "^";
    std::vector<std::string> names;
    std::set<std::string> uniqueNames;
    for (std::size_t index = 0; index < pathPattern.size();)
    {
        if (pathPattern[index] == ':')
        {
            if (index == 0 || pathPattern[index - 1] != '/')
            {
                throw std::invalid_argument(
                    "route parameter must occupy a complete path segment");
            }
            const std::size_t begin = ++index;
            while (index < pathPattern.size() && pathPattern[index] != '/')
            {
                ++index;
            }
            const std::string name = pathPattern.substr(begin, index - begin);
            const bool validName = !name.empty() &&
                (std::isalpha(static_cast<unsigned char>(name.front())) ||
                 name.front() == '_') &&
                std::all_of(name.begin() + 1, name.end(), [](unsigned char value) {
                    return std::isalnum(value) || value == '_';
                });
            if (!validName || !uniqueNames.insert(name).second)
            {
                throw std::invalid_argument(
                    "route parameter names must be valid and unique");
            }
            names.push_back(name);
            expression += "([^/]+)";
            continue;
        }
        const char value = pathPattern[index++];
        if (std::string(".^$|()[]*+?{}\\").find(value) != std::string::npos)
        {
            expression.push_back('\\');
        }
        expression.push_back(value);
    }
    expression += "$";
    return {std::regex(expression), std::move(names)};
}

void Router::extractPathParameters(const std::smatch& match,
                                   const std::vector<std::string>& names,
                                   HttpRequest& request)
{
    if (match.size() != names.size() + 1)
    {
        throw std::logic_error("route capture count does not match parameters");
    }
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        request.setPathParameters(names[index], match[index + 1].str());
    }
}

bool Router::route(const HttpRequest &req, HttpResponse *resp)
{
    RouteKey key{req.method(), req.path()};

    // 查找处理器
    auto handlerIt = handlers_.find(key);
    if (handlerIt != handlers_.end())
    {
        handlerIt->second->handle(req, resp);
        return true;
    }

    // 查找回调函数
    auto callbackIt = callbacks_.find(key);
    if (callbackIt != callbacks_.end())
    {
        callbackIt->second(req, resp);
        return true;
    }

    // 查找动态路由处理器
    for (const auto& route : regexHandlers_)
    {
        std::smatch match;
        std::string pathStr(req.path());
        // 如果方法匹配并且动态路由匹配，则执行处理器
        if (route.method_ == req.method() &&
            std::regex_match(pathStr, match, route.pathRegex_))
        {
            // Extract path parameters and add them to the request
            HttpRequest newReq(req); // 因为这里需要用这一次所以是可以改的
            extractPathParameters(match, route.parameterNames_, newReq);
            
            route.handler_->handle(newReq, resp);
            return true;
        }
    }

    // 查找动态路由回调函数
    for (const auto& route : regexCallbacks_)
    {
        std::smatch match;
        std::string pathStr(req.path());
        // 如果方法匹配并且动态路由匹配，则执行回调函数
        if (route.method_ == req.method() &&
            std::regex_match(pathStr, match, route.pathRegex_))
        {
             // Extract path parameters and add them to the request
            HttpRequest newReq(req); // 因为这里需要用这一次所以是可以改的
            extractPathParameters(match, route.parameterNames_, newReq);

            route.callback_(newReq, resp);
            return true;
        }
    }

    return false;
}

} // namespace router
} // namespace http
