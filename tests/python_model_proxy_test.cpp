#include <cassert>
#include <stdexcept>
#include <string>

#include "client/PythonModelClient.h"
#include "service/PythonModelProxy.h"

namespace
{

class FakeClient : public treesem::client::IModelAdapterClient
{
public:
    treesem::client::ModelAdapterResponse response{200, R"({"model":"treeSem"})"};
    bool timeout{false};

    treesem::client::ModelAdapterResponse predict(const std::string&) const override
    {
        if (timeout)
        {
            throw treesem::client::ModelAdapterException(
                treesem::client::ModelAdapterException::Kind::Timeout,
                "timeout");
        }
        return response;
    }
};

int execute(const http::ResponseWriter& writer)
{
    http::HttpResponse response(false);
    writer(&response);
    return static_cast<int>(response.getStatusCode());
}

} // namespace

int main()
{
    FakeClient client;
    treesem::service::PythonModelProxy proxy(client);

    assert(execute(proxy.predict(R"({"sample_index":0})")) == 200);
    assert(execute(proxy.predict("not-json")) == 400);
    assert(execute(proxy.predict(R"({"sample_index":-1})")) == 400);
    assert(execute(proxy.predict(R"({"sample_index":0,"preprocessed_features":[1]})")) ==
           400);

    client.response = {200, "not-json"};
    assert(execute(proxy.predict(R"({"sample_index":0})")) == 502);

    client.response = {500, R"({"error":"failed"})"};
    assert(execute(proxy.predict(R"({"sample_index":0})")) == 502);

    client.timeout = true;
    assert(execute(proxy.predict(R"({"sample_index":0})")) == 504);
}
