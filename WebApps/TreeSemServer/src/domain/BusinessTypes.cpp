#include "domain/BusinessTypes.h"

#include <stdexcept>

namespace treesem
{
namespace domain
{

std::string toString(AgentRunStatus status)
{
    switch (status)
    {
    case AgentRunStatus::Running: return "running";
    case AgentRunStatus::Completed: return "completed";
    case AgentRunStatus::Failed: return "failed";
    }
    throw std::invalid_argument("unknown agent run status");
}

std::string toString(FeedbackAssessment assessment)
{
    switch (assessment)
    {
    case FeedbackAssessment::Agree:
        return "agree";
    case FeedbackAssessment::Disagree:
        return "disagree";
    case FeedbackAssessment::Uncertain:
        return "uncertain";
    }
    throw std::invalid_argument("unknown feedback assessment");
}

FeedbackAssessment parseFeedbackAssessment(const std::string& value)
{
    if (value == "agree")
    {
        return FeedbackAssessment::Agree;
    }
    if (value == "disagree")
    {
        return FeedbackAssessment::Disagree;
    }
    if (value == "uncertain")
    {
        return FeedbackAssessment::Uncertain;
    }
    throw std::invalid_argument("unknown feedback assessment");
}

} // namespace domain
} // namespace treesem
