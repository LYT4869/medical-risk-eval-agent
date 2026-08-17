#include <cassert>
#include <chrono>
#include <optional>
#include <string>

#include "ModelFixtures.h"
#include "api/ApiException.h"
#include "api/BusinessJsonCodec.h"
#include "application/BusinessException.h"
#include "application/SessionService.h"
#include "infrastructure/persistence/InMemoryTreeSemStore.h"
#include "serialization/ModelResultJsonCodec.h"

namespace
{

void addHeader(http::HttpRequest& request,
               const std::string& name,
               const std::string& value)
{
    const std::string line = name + ": " + value;
    const char* begin = line.data();
    request.addHeader(begin, begin + name.size(), begin + line.size());
}

bool invalidFeedback(const std::string& body)
{
    try
    {
        (void)treesem::api::BusinessJsonCodec::parseFeedback(body, "request-key-01");
        return false;
    }
    catch (const treesem::api::ApiException&)
    {
        return true;
    }
}

} // namespace

int main()
{
    using namespace treesem;

    http::HttpRequest publicRequest;
    addHeader(publicRequest, "Cookie", "theme=dark; treeSemSession=ses_0123456789abcdef0123456789abcdef; x=1");
    assert(api::BusinessJsonCodec::sessionId(
        publicRequest, application::SessionAccess::Public) ==
        "ses_0123456789abcdef0123456789abcdef");

    http::HttpRequest internalRequest;
    addHeader(internalRequest, "Cookie", "treeSemSession=ses_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    addHeader(internalRequest, "X-TreeSem-Session-Id", "ses_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    assert(api::BusinessJsonCodec::sessionId(
        internalRequest, application::SessionAccess::Internal) ==
        "ses_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");

    const std::string cookie = api::BusinessJsonCodec::sessionCookie(
        "ses_0123456789abcdef0123456789abcdef", 3600, true);
    assert(cookie.find("HttpOnly") != std::string::npos);
    assert(cookie.find("SameSite=Lax") != std::string::npos);
    assert(cookie.find("Max-Age=3600") != std::string::npos);
    assert(cookie.find("Secure") != std::string::npos);

    http::HttpRequest historyRequest;
    const std::string query = "limit=100";
    historyRequest.setQueryParameters(query.data(), query.data() + query.size());
    assert(api::BusinessJsonCodec::parseHistoryLimit(historyRequest) == 100);

    http::HttpRequest invalidLimit;
    const std::string badQuery = "limit=101";
    invalidLimit.setQueryParameters(badQuery.data(), badQuery.data() + badQuery.size());
    bool rejectedLimit = false;
    try { (void)api::BusinessJsonCodec::parseHistoryLimit(invalidLimit); }
    catch (const api::ApiException&) { rejectedLimit = true; }
    assert(rejectedLimit);

    assert(invalidFeedback(
        R"({"reviewer_reference":"doctor","assessment":"agree","unexpected":1})"));
    const auto feedback = api::BusinessJsonCodec::parseFeedback(
        R"({"reviewer_reference":"doctor","assessment":"uncertain","comment":"review needed"})",
        "request-key-01");
    assert(feedback.assessment == domain::FeedbackAssessment::Uncertain);
    assert(feedback.comment == "review needed");

    const model::ModelResult original = test::validModelResult();
    const model::ModelResult decoded = serialization::ModelResultJsonCodec::decode(
        serialization::ModelResultJsonCodec::encode(original));
    assert(decoded.modelName == original.modelName);
    assert(decoded.modelVersion == original.modelVersion);
    assert(decoded.prediction.label == original.prediction.label);
    assert(decoded.importantFeatures.size() == original.importantFeatures.size());
    assert(decoded.decisionPath.size() == original.decisionPath.size());

    infrastructure::InMemoryTreeSemStore store;
    application::SessionService sessions(store, std::chrono::seconds(3600));
    bool missingInternalRejected = false;
    try
    {
        (void)sessions.resolve(std::nullopt, application::SessionAccess::Internal);
    }
    catch (const application::BusinessException& error)
    {
        missingInternalRejected =
            error.kind() == application::BusinessException::Kind::InvalidSession;
    }
    assert(missingInternalRejected);
}
