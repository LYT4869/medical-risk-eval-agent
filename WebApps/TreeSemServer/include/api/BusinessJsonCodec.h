#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "application/RecordServices.h"
#include "application/SessionService.h"
#include "domain/BusinessTypes.h"
#include "http/HttpRequest.h"

namespace treesem
{
namespace api
{

class BusinessJsonCodec
{
public:
    static std::optional<std::string> sessionId(
        const http::HttpRequest& request,
        application::SessionAccess access);
    static std::string sessionCookie(const std::string& sessionId,
                                     long ttlSeconds,
                                     bool secure);

    static std::string serializePrediction(const domain::PredictionRecord& record);
    static std::string serializeExplanation(const domain::PredictionRecord& record);
    static std::string serializeHistory(const domain::HistoryPage& page);
    static std::string serializeComparison(
        const domain::PredictionComparison& comparison);
    static std::string serializeFeedback(
        const domain::ClinicalFeedback& feedback);
    static std::string serializeFeedbackList(
        const std::vector<domain::ClinicalFeedback>& feedback);

    static std::pair<std::string, std::string> parseComparison(
        const std::string& body);
    static application::FeedbackInput parseFeedback(
        const std::string& body,
        const std::string& idempotencyKey);
    static std::size_t parseHistoryLimit(const http::HttpRequest& request);
    static std::optional<domain::HistoryCursor> parseHistoryCursor(
        const http::HttpRequest& request);
};

} // namespace api
} // namespace treesem
