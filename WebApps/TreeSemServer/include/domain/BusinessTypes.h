#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "model/ModelTypes.h"

namespace treesem
{
namespace domain
{

using TimePoint = std::chrono::system_clock::time_point;

struct SessionRecord
{
    std::string sessionId;
    std::optional<std::string> currentPredictionId;
    std::string status{"active"};
    std::uint64_t version{0};
    TimePoint createdAt;
    TimePoint lastAccessedAt;
    TimePoint expiresAt;
};

struct PredictionRecord
{
    std::string predictionId;
    std::string sessionId;
    std::optional<std::string> subjectUserId;
    std::optional<std::string> createdByUserId;
    model::ModelResult result;
    TimePoint createdAt;
};

struct PredictionSummary
{
    std::string predictionId;
    TimePoint createdAt;
    int label;
    double positiveProbability;
    double confidence;
    std::optional<std::string> modelVersion;
    std::optional<std::string> servingBackend;
};

struct HistoryCursor
{
    TimePoint createdAt;
    std::string predictionId;
};

struct HistoryPage
{
    std::string sessionId;
    std::vector<PredictionSummary> items;
    std::optional<HistoryCursor> nextCursor;
};

struct FeatureDifference
{
    int index;
    std::string name;
    std::optional<double> standardizedValueA;
    std::optional<double> standardizedValueB;
    std::optional<double> standardizedDelta;
    std::optional<double> originalValueA;
    std::optional<double> originalValueB;
    std::optional<double> originalDelta;
    std::optional<std::string> unit;
};

struct PredictionComparison
{
    PredictionSummary predictionA;
    PredictionSummary predictionB;
    bool labelChanged{false};
    bool modelVersionChanged{false};
    double positiveProbabilityDelta{0.0};
    double confidenceDelta{0.0};
    bool clusterChanged{false};
    bool treeLeafChanged{false};
    bool pathChanged{false};
    std::vector<FeatureDifference> changedFeatures;
};

enum class FeedbackAssessment
{
    Agree,
    Disagree,
    Uncertain,
};

std::string toString(FeedbackAssessment assessment);
FeedbackAssessment parseFeedbackAssessment(const std::string& value);

struct ClinicalFeedback
{
    std::string feedbackId;
    std::string predictionId;
    std::string sessionId;
    std::string reviewerReference;
    bool reviewerVerified{false};
    FeedbackAssessment assessment;
    std::optional<int> correctedLabel;
    std::optional<std::string> comment;
    std::string idempotencyKey;
    std::string payloadSha256;
    TimePoint createdAt;
};

struct FeedbackSaveResult
{
    ClinicalFeedback feedback;
    bool created{false};
};

enum class AgentRunStatus { Running, Completed, Failed };
std::string toString(AgentRunStatus status);

struct AgentToolSummary
{
    std::string name;
    std::string status;
    std::uint64_t durationMs{0};
};

struct AgentRunRecord
{
    std::string runId;
    std::string sessionId;
    std::string idempotencyKey;
    std::string payloadSha256;
    AgentRunStatus status{AgentRunStatus::Running};
    int stepCount{0};
    std::vector<AgentToolSummary> tools;
    std::vector<std::string> groundingPredictionIds;
    std::optional<std::string> finalMessageId;
    std::optional<std::string> errorCode;
    TimePoint startedAt;
    std::optional<TimePoint> completedAt;
    std::optional<std::string> actorUserId;
    std::optional<std::string> subjectUserId;
};

struct ChatMessage
{
    std::string messageId;
    std::string sessionId;
    std::string runId;
    std::string role;
    std::string content;
    TimePoint createdAt;
    std::optional<std::string> actorUserId;
    std::optional<std::string> subjectUserId;
};

struct ChatCursor
{
    TimePoint createdAt;
    std::string messageId;
};

struct ChatPage
{
    std::string sessionId;
    std::vector<ChatMessage> items;
    std::optional<ChatCursor> nextCursor;
};

struct AgentRunStartResult
{
    AgentRunRecord run;
    std::optional<ChatMessage> finalMessage;
    bool created{false};
};

} // namespace domain
} // namespace treesem
