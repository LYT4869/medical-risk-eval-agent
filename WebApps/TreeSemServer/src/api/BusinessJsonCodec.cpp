#include "api/BusinessJsonCodec.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "api/ApiException.h"
#include "api/PredictionJsonCodec.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem
{
namespace api
{
namespace
{

using Json = nlohmann::json;

[[noreturn]] void invalidRequest()
{
    throw ApiException(ApiException::Kind::InvalidRequest, "invalid business request");
}

std::string trim(std::string value)
{
    const auto first = std::find_if_not(value.begin(), value.end(), [](char item) {
        return std::isspace(static_cast<unsigned char>(item));
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](char item) {
        return std::isspace(static_cast<unsigned char>(item));
    }).base();
    return first < last ? std::string(first, last) : std::string();
}

Json predictionSummary(const domain::PredictionSummary& summary)
{
    return {
        {"prediction_id", summary.predictionId},
        {"created_at", infrastructure::formatUtc(summary.createdAt)},
        {"label", summary.label},
        {"positive_probability", summary.positiveProbability},
        {"confidence", summary.confidence},
        {"model_version", summary.modelVersion.has_value()
            ? Json(*summary.modelVersion) : Json(nullptr)},
        {"serving_backend", summary.servingBackend.has_value()
            ? Json(*summary.servingBackend) : Json(nullptr)}};
}

} // namespace

std::optional<std::string> BusinessJsonCodec::sessionId(
    const http::HttpRequest& request,
    application::SessionAccess access)
{
    if (access == application::SessionAccess::Internal)
    {
        const std::string value = trim(request.getHeader("X-TreeSem-Session-Id"));
        return value.empty() ? std::nullopt : std::optional<std::string>(value);
    }
    const std::string cookie = request.getHeader("Cookie");
    std::size_t begin = 0;
    while (begin < cookie.size())
    {
        const std::size_t end = cookie.find(';', begin);
        const std::string token = trim(cookie.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin));
        const std::size_t equals = token.find('=');
        if (equals != std::string::npos && trim(token.substr(0, equals)) == "treeSemSession")
        {
            const std::string value = trim(token.substr(equals + 1));
            return value.empty() ? std::nullopt : std::optional<std::string>(value);
        }
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return std::nullopt;
}

std::string BusinessJsonCodec::sessionCookie(
    const std::string& sessionId,
    long ttlSeconds,
    bool secure)
{
    std::string cookie = "treeSemSession=" + sessionId +
        "; Path=/; HttpOnly; SameSite=Lax; Max-Age=" +
        std::to_string(ttlSeconds);
    if (secure) cookie += "; Secure";
    return cookie;
}

std::string BusinessJsonCodec::serializePrediction(
    const domain::PredictionRecord& record)
{
    Json result = Json::parse(PredictionJsonCodec::serializeResult(record.result));
    result["prediction_id"] = record.predictionId;
    result["session_id"] = record.sessionId;
    result["created_at"] = infrastructure::formatUtc(record.createdAt);
    return result.dump();
}

std::string BusinessJsonCodec::serializeExplanation(
    const domain::PredictionRecord& record)
{
    Json full = Json::parse(PredictionJsonCodec::serializeResult(record.result));
    Json result{
        {"prediction_id", record.predictionId},
        {"session_id", record.sessionId},
        {"created_at", infrastructure::formatUtc(record.createdAt)},
        {"model", full.at("model")},
        {"model_version", full.contains("model_version")
            ? full.at("model_version") : Json(nullptr)},
        {"prediction", full.at("prediction")},
        {"important_features", full.at("important_features")},
        {"decision_path", full.at("decision_path")}};
    return result.dump();
}

std::string BusinessJsonCodec::serializeHistory(const domain::HistoryPage& page)
{
    Json items = Json::array();
    for (const auto& item : page.items) items.push_back(predictionSummary(item));
    return Json{
        {"session_id", page.sessionId},
        {"items", std::move(items)},
        {"next_cursor", page.nextCursor.has_value()
            ? Json(infrastructure::encodeHistoryCursor(*page.nextCursor))
            : Json(nullptr)}}.dump();
}

std::string BusinessJsonCodec::serializeComparison(
    const domain::PredictionComparison& comparison)
{
    Json differences = Json::array();
    for (const auto& feature : comparison.changedFeatures)
    {
        differences.push_back({
            {"index", feature.index},
            {"name", feature.name},
            {"standardized_value_a", feature.standardizedValueA.has_value()
                ? Json(*feature.standardizedValueA) : Json(nullptr)},
            {"standardized_value_b", feature.standardizedValueB.has_value()
                ? Json(*feature.standardizedValueB) : Json(nullptr)},
            {"standardized_delta", feature.standardizedDelta.has_value()
                ? Json(*feature.standardizedDelta) : Json(nullptr)},
            {"original_value_a", feature.originalValueA.has_value()
                ? Json(*feature.originalValueA) : Json(nullptr)},
            {"original_value_b", feature.originalValueB.has_value()
                ? Json(*feature.originalValueB) : Json(nullptr)},
            {"original_delta", feature.originalDelta.has_value()
                ? Json(*feature.originalDelta) : Json(nullptr)},
            {"unit", feature.unit.has_value() ? Json(*feature.unit) : Json(nullptr)}});
    }
    return Json{
        {"prediction_a", predictionSummary(comparison.predictionA)},
        {"prediction_b", predictionSummary(comparison.predictionB)},
        {"label_changed", comparison.labelChanged},
        {"model_version_changed", comparison.modelVersionChanged},
        {"positive_probability_delta", comparison.positiveProbabilityDelta},
        {"confidence_delta", comparison.confidenceDelta},
        {"cluster_changed", comparison.clusterChanged},
        {"tree_leaf_changed", comparison.treeLeafChanged},
        {"path_changed", comparison.pathChanged},
        {"changed_features", std::move(differences)}}.dump();
}

std::string BusinessJsonCodec::serializeFeedback(
    const domain::ClinicalFeedback& feedback)
{
    return Json{
        {"feedback_id", feedback.feedbackId},
        {"prediction_id", feedback.predictionId},
        {"session_id", feedback.sessionId},
        {"reviewer_reference", feedback.reviewerReference},
        {"reviewer_verified", feedback.reviewerVerified},
        {"assessment", domain::toString(feedback.assessment)},
        {"corrected_label", feedback.correctedLabel.has_value()
            ? Json(*feedback.correctedLabel) : Json(nullptr)},
        {"comment", feedback.comment.has_value()
            ? Json(*feedback.comment) : Json(nullptr)},
        {"created_at", infrastructure::formatUtc(feedback.createdAt)}}.dump();
}

std::string BusinessJsonCodec::serializeFeedbackList(
    const std::vector<domain::ClinicalFeedback>& feedback)
{
    Json result = Json::array();
    for (const auto& item : feedback)
        result.push_back(Json::parse(serializeFeedback(item)));
    return Json{{"items", std::move(result)}}.dump();
}

std::pair<std::string, std::string> BusinessJsonCodec::parseComparison(
    const std::string& body)
{
    try
    {
        const Json root = Json::parse(body);
        if (!root.is_object() || root.size() != 2 ||
            !root.contains("prediction_id_a") || !root.contains("prediction_id_b") ||
            !root.at("prediction_id_a").is_string() ||
            !root.at("prediction_id_b").is_string()) invalidRequest();
        return {root.at("prediction_id_a").get<std::string>(),
                root.at("prediction_id_b").get<std::string>()};
    }
    catch (const ApiException&) { throw; }
    catch (const Json::parse_error&)
    {
        throw ApiException(ApiException::Kind::InvalidJson, "invalid JSON");
    }
    catch (const Json::exception&) { invalidRequest(); }
}

application::FeedbackInput BusinessJsonCodec::parseFeedback(
    const std::string& body,
    const std::string& idempotencyKey)
{
    try
    {
        const Json root = Json::parse(body);
        if (!root.is_object() || root.size() < 2 || root.size() > 4 ||
            !root.contains("reviewer_reference") || !root.contains("assessment") ||
            !root.at("reviewer_reference").is_string() ||
            !root.at("assessment").is_string()) invalidRequest();
        for (const auto& item : root.items())
        {
            if (item.key() != "reviewer_reference" &&
                item.key() != "assessment" &&
                item.key() != "corrected_label" &&
                item.key() != "comment")
            {
                invalidRequest();
            }
        }
        application::FeedbackInput result;
        result.reviewerReference = root.at("reviewer_reference").get<std::string>();
        result.assessment = domain::parseFeedbackAssessment(
            root.at("assessment").get<std::string>());
        result.idempotencyKey = idempotencyKey;
        if (root.contains("corrected_label"))
        {
            if (!root.at("corrected_label").is_number_integer()) invalidRequest();
            result.correctedLabel = root.at("corrected_label").get<int>();
        }
        if (root.contains("comment"))
        {
            if (!root.at("comment").is_string()) invalidRequest();
            result.comment = root.at("comment").get<std::string>();
        }
        return result;
    }
    catch (const ApiException&) { throw; }
    catch (const std::invalid_argument&) { invalidRequest(); }
    catch (const Json::parse_error&)
    {
        throw ApiException(ApiException::Kind::InvalidJson, "invalid JSON");
    }
    catch (const Json::exception&) { invalidRequest(); }
}

std::size_t BusinessJsonCodec::parseHistoryLimit(const http::HttpRequest& request)
{
    const std::string raw = request.getQueryParameters("limit");
    if (raw.empty()) return 20;
    try
    {
        std::size_t consumed = 0;
        const unsigned long value = std::stoul(raw, &consumed);
        if (consumed != raw.size() || value == 0 || value > 100) invalidRequest();
        return static_cast<std::size_t>(value);
    }
    catch (const ApiException&) { throw; }
    catch (...) { invalidRequest(); }
}

std::optional<domain::HistoryCursor> BusinessJsonCodec::parseHistoryCursor(
    const http::HttpRequest& request)
{
    const std::string raw = request.getQueryParameters("cursor");
    if (raw.empty()) return std::nullopt;
    try { return infrastructure::decodeHistoryCursor(raw); }
    catch (...) { invalidRequest(); }
}

} // namespace api
} // namespace treesem
