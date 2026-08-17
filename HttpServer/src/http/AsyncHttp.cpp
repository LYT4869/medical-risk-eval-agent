#include "../../include/http/AsyncHttp.h"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <utility>

namespace http
{

AsyncResponder makeOneShotResponder(AsyncResponder responder)
{
    if (!responder)
    {
        throw std::invalid_argument("responder must not be empty");
    }
    auto responded = std::make_shared<std::atomic_bool>(false);
    return [responder = std::move(responder), responded](ResponseWriter writer) {
        if (!writer)
        {
            return;
        }
        bool expected = false;
        if (!responded->compare_exchange_strong(expected, true))
        {
            return;
        }
        responder(std::move(writer));
    };
}

} // namespace http
