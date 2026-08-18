#include "infrastructure/persistence/InMemoryTreeSemStore.h"

#include <algorithm>

#include "application/BusinessException.h"

namespace treesem
{
namespace infrastructure
{

void InMemoryTreeSemStore::createSession(const domain::SessionRecord& session)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!sessions_.emplace(session.sessionId, session).second)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict,
            "session id collision");
    }
}

std::optional<domain::SessionRecord> InMemoryTreeSemStore::touchActiveSession(
    const std::string& sessionId,
    domain::TimePoint accessedAt,
    domain::TimePoint expiresAt)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = sessions_.find(sessionId);
    if (found == sessions_.end() || found->second.status != "active" ||
        found->second.expiresAt <= accessedAt)
    {
        return std::nullopt;
    }
    found->second.lastAccessedAt = accessedAt;
    found->second.expiresAt = expiresAt;
    ++found->second.version;
    return found->second;
}

void InMemoryTreeSemStore::savePredictionAndSetCurrent(
    const domain::PredictionRecord& prediction,
    domain::TimePoint sessionExpiresAt)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto session = sessions_.find(prediction.sessionId);
    if (session == sessions_.end() || session->second.status != "active" ||
        session->second.expiresAt <= prediction.createdAt)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict,
            "session expired before prediction commit");
    }
    if (!predictions_.emplace(prediction.predictionId, prediction).second)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict,
            "prediction id collision");
    }
    session->second.currentPredictionId = prediction.predictionId;
    session->second.lastAccessedAt = prediction.createdAt;
    session->second.expiresAt = sessionExpiresAt;
    ++session->second.version;
}

std::optional<domain::PredictionRecord> InMemoryTreeSemStore::findPrediction(
    const std::string& sessionId,
    const std::string& predictionId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = predictions_.find(predictionId);
    if (found == predictions_.end() || found->second.sessionId != sessionId)
    {
        return std::nullopt;
    }
    return found->second;
}

domain::HistoryPage InMemoryTreeSemStore::listPredictions(
    const std::string& sessionId,
    std::size_t limit,
    const std::optional<domain::HistoryCursor>& cursor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<domain::PredictionRecord> records;
    for (const auto& entry : predictions_)
    {
        const auto& record = entry.second;
        const bool beforeCursor = !cursor.has_value() ||
            record.createdAt < cursor->createdAt ||
            (record.createdAt == cursor->createdAt &&
             record.predictionId < cursor->predictionId);
        if (record.sessionId == sessionId && beforeCursor)
        {
            records.push_back(record);
        }
    }
    std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
        if (left.createdAt != right.createdAt)
        {
            return left.createdAt > right.createdAt;
        }
        return left.predictionId > right.predictionId;
    });
    domain::HistoryPage page;
    page.sessionId = sessionId;
    const std::size_t count = std::min(limit, records.size());
    for (std::size_t index = 0; index < count; ++index)
    {
        const auto& record = records[index];
        page.items.push_back({
            record.predictionId,
            record.createdAt,
            record.result.prediction.label,
            record.result.prediction.positiveProbability,
            record.result.prediction.confidence,
            record.result.modelVersion,
            record.result.servingBackend});
    }
    if (records.size() > limit && !page.items.empty())
    {
        page.nextCursor = domain::HistoryCursor{
            page.items.back().createdAt, page.items.back().predictionId};
    }
    return page;
}

std::optional<domain::PredictionRecord> InMemoryTreeSemStore::findPredictionBySubject(
    const std::string& subject, const std::string& predictionId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = predictions_.find(predictionId);
    if (found == predictions_.end() || found->second.subjectUserId != subject)
        return std::nullopt;
    return found->second;
}

std::optional<domain::PredictionRecord> InMemoryTreeSemStore::findPredictionAny(
    const std::string& predictionId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = predictions_.find(predictionId);
    return found == predictions_.end() ? std::nullopt
                                      : std::optional<domain::PredictionRecord>(found->second);
}

domain::HistoryPage InMemoryTreeSemStore::listPredictionsBySubject(
    const std::string& subject, std::size_t limit,
    const std::optional<domain::HistoryCursor>& cursor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<domain::PredictionRecord> records;
    for (const auto& item : predictions_)
        if (item.second.subjectUserId == subject &&
            (!cursor.has_value() || item.second.createdAt < cursor->createdAt ||
             (item.second.createdAt == cursor->createdAt &&
              item.second.predictionId < cursor->predictionId)))
            records.push_back(item.second);
    std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
        return a.createdAt != b.createdAt ? a.createdAt > b.createdAt
                                         : a.predictionId > b.predictionId;
    });
    domain::HistoryPage page; page.sessionId = subject;
    for (std::size_t i = 0; i < std::min(limit, records.size()); ++i)
    {
        const auto& record = records[i];
        page.items.push_back({record.predictionId, record.createdAt,
            record.result.prediction.label, record.result.prediction.positiveProbability,
            record.result.prediction.confidence, record.result.modelVersion,
            record.result.servingBackend});
    }
    if (records.size() > limit && !page.items.empty())
        page.nextCursor = domain::HistoryCursor{
            page.items.back().createdAt, page.items.back().predictionId};
    return page;
}

domain::FeedbackSaveResult InMemoryTreeSemStore::saveFeedback(
    const domain::ClinicalFeedback& feedback)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto prediction = predictions_.find(feedback.predictionId);
    if (prediction == predictions_.end() ||
        prediction->second.sessionId != feedback.sessionId)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::NotFound,
            "prediction not found");
    }
    const auto existing = std::find_if(
        feedback_.begin(), feedback_.end(), [&](const auto& item) {
            return item.sessionId == feedback.sessionId &&
                item.idempotencyKey == feedback.idempotencyKey;
        });
    if (existing != feedback_.end())
    {
        if (existing->payloadSha256 != feedback.payloadSha256)
        {
            throw application::BusinessException(
                application::BusinessException::Kind::IdempotencyConflict,
                "idempotency key was reused with another payload");
        }
        return {*existing, false};
    }
    feedback_.push_back(feedback);
    return {feedback, true};
}

std::vector<domain::ClinicalFeedback> InMemoryTreeSemStore::listFeedback(
    const std::string& sessionId,
    const std::string& predictionId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto prediction = predictions_.find(predictionId);
    if (prediction == predictions_.end() || prediction->second.sessionId != sessionId)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::NotFound,
            "prediction not found");
    }
    std::vector<domain::ClinicalFeedback> result;
    for (const auto& item : feedback_)
    {
        if (item.sessionId == sessionId && item.predictionId == predictionId)
        {
            result.push_back(item);
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.createdAt < right.createdAt;
    });
    return result;
}

bool InMemoryTreeSemStore::ping()
{
    return true;
}

domain::AgentRunStartResult InMemoryTreeSemStore::startAgentRun(
    const domain::AgentRunRecord& run,
    const domain::ChatMessage& userMessage)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& entry : agentRuns_)
    {
        const auto& existing = entry.second;
        if (existing.sessionId == run.sessionId &&
            existing.idempotencyKey == run.idempotencyKey)
        {
            if (existing.payloadSha256 != run.payloadSha256)
                throw application::BusinessException(
                    application::BusinessException::Kind::IdempotencyConflict,
                    "chat idempotency conflict");
            std::optional<domain::ChatMessage> final;
            if (existing.finalMessageId.has_value())
            {
                const auto found = std::find_if(
                    chatMessages_.begin(), chatMessages_.end(), [&](const auto& item) {
                        return item.messageId == *existing.finalMessageId;
                    });
                if (found != chatMessages_.end()) final = *found;
            }
            return {existing, final, false};
        }
    }
    if (sessions_.find(run.sessionId) == sessions_.end() ||
        agentRuns_.find(run.runId) != agentRuns_.end())
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict, "agent run collision");
    agentRuns_.emplace(run.runId, run);
    chatMessages_.push_back(userMessage);
    return {run, std::nullopt, true};
}

void InMemoryTreeSemStore::completeAgentRun(
    const domain::AgentRunRecord& run,
    const domain::ChatMessage& assistantMessage)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = agentRuns_.find(run.runId);
    if (found == agentRuns_.end() || found->second.status != domain::AgentRunStatus::Running)
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict, "agent run is not active");
    found->second = run;
    chatMessages_.push_back(assistantMessage);
}

void InMemoryTreeSemStore::failAgentRun(
    const std::string& runId,
    const std::string& errorCode,
    domain::TimePoint completedAt)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = agentRuns_.find(runId);
    if (found != agentRuns_.end() && found->second.status == domain::AgentRunStatus::Running)
    {
        found->second.status = domain::AgentRunStatus::Failed;
        found->second.errorCode = errorCode;
        found->second.completedAt = completedAt;
    }
}

domain::ChatPage InMemoryTreeSemStore::listChatMessages(
    const std::string& sessionId,
    std::size_t limit,
    const std::optional<domain::ChatCursor>& cursor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<domain::ChatMessage> records;
    for (const auto& item : chatMessages_)
    {
        if (item.sessionId == sessionId &&
            (!cursor.has_value() || item.createdAt < cursor->createdAt ||
             (item.createdAt == cursor->createdAt && item.messageId < cursor->messageId)))
            records.push_back(item);
    }
    std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
        return a.createdAt != b.createdAt ? a.createdAt > b.createdAt
                                         : a.messageId > b.messageId;
    });
    domain::ChatPage page;
    page.sessionId = sessionId;
    if (records.size() > limit)
    {
        records.resize(limit);
        page.nextCursor = domain::ChatCursor{
            records.back().createdAt, records.back().messageId};
    }
    page.items = std::move(records);
    return page;
}

std::vector<domain::ChatMessage> InMemoryTreeSemStore::recentChatMessages(
    const std::string& sessionId, std::size_t limit)
{
    auto page = listChatMessages(sessionId, limit, std::nullopt);
    std::reverse(page.items.begin(), page.items.end());
    return page.items;
}

void InMemoryTreeSemStore::createUser(const domain::UserRecord& user)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : users_)
        if (item.second.emailNormalized == user.emailNormalized)
            throw application::BusinessException(
                application::BusinessException::Kind::Conflict, "account exists");
    if (!users_.emplace(user.userId, user).second)
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict, "user collision");
}

std::optional<domain::UserRecord> InMemoryTreeSemStore::findUserByEmail(
    const std::string& email)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : users_)
        if (item.second.emailNormalized == email) return item.second;
    return std::nullopt;
}

std::optional<domain::UserRecord> InMemoryTreeSemStore::findUserById(
    const std::string& id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = users_.find(id);
    return found == users_.end() ? std::nullopt
                                : std::optional<domain::UserRecord>(found->second);
}

bool InMemoryTreeSemStore::hasAdminUser()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return std::any_of(users_.begin(), users_.end(), [](const auto& item) {
        return item.second.role == domain::UserRole::Admin;
    });
}

void InMemoryTreeSemStore::recordLoginFailure(
    const std::string& id, int count,
    const std::optional<domain::TimePoint>& lockedUntil, domain::TimePoint now)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = users_.find(id);
    if (found == users_.end()) return;
    found->second.failedLoginCount = count;
    found->second.lockedUntil = lockedUntil;
    found->second.updatedAt = now;
}

void InMemoryTreeSemStore::recordLoginSuccess(
    const std::string& id, domain::TimePoint now)
{
    recordLoginFailure(id, 0, std::nullopt, now);
}

void InMemoryTreeSemStore::createRefreshSession(
    const domain::RefreshSession& session)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!refreshSessions_.emplace(session.refreshSessionId, session).second)
        throw application::BusinessException(
            application::BusinessException::Kind::Conflict, "refresh collision");
}

persistence::RefreshRotateResult InMemoryTreeSemStore::rotateRefreshSession(
    const std::string& oldHash, const domain::RefreshSession& replacement,
    domain::TimePoint now, std::optional<domain::RefreshSession>& oldSession)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = std::find_if(refreshSessions_.begin(), refreshSessions_.end(),
        [&](const auto& item) { return item.second.tokenSha256 == oldHash; });
    if (found == refreshSessions_.end()) return persistence::RefreshRotateResult::NotFound;
    oldSession = found->second;
    if (found->second.consumedAt.has_value() || found->second.revokedAt.has_value())
    {
        for (auto& item : refreshSessions_)
            if (item.second.tokenFamilyId == found->second.tokenFamilyId &&
                !item.second.revokedAt.has_value()) item.second.revokedAt = now;
        return persistence::RefreshRotateResult::Reused;
    }
    if (found->second.expiresAt <= now) return persistence::RefreshRotateResult::Expired;
    found->second.consumedAt = now;
    found->second.replacedById = replacement.refreshSessionId;
    refreshSessions_.emplace(replacement.refreshSessionId, replacement);
    return persistence::RefreshRotateResult::Rotated;
}

void InMemoryTreeSemStore::revokeRefreshFamily(
    const std::string& family, domain::TimePoint now)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : refreshSessions_)
        if (item.second.tokenFamilyId == family && !item.second.revokedAt.has_value())
            item.second.revokedAt = now;
}

std::optional<domain::RefreshSession> InMemoryTreeSemStore::findRefreshSession(
    const std::string& hash)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : refreshSessions_)
        if (item.second.tokenSha256 == hash) return item.second;
    return std::nullopt;
}

domain::DoctorPatientAssignment InMemoryTreeSemStore::saveAssignment(
    const domain::DoctorPatientAssignment& assignment)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : assignments_)
    {
        if (item.second.doctorUserId == assignment.doctorUserId &&
            item.second.patientUserId == assignment.patientUserId)
        {
            item.second = assignment;
            return item.second;
        }
    }
    assignments_.emplace(assignment.assignmentId, assignment);
    return assignment;
}

bool InMemoryTreeSemStore::revokeAssignment(
    const std::string& id, domain::TimePoint now)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = assignments_.find(id);
    if (found == assignments_.end() || found->second.status != "active") return false;
    found->second.status = "revoked"; found->second.revokedAt = now; return true;
}

bool InMemoryTreeSemStore::hasActiveAssignment(
    const std::string& doctor, const std::string& patient)
{
    std::lock_guard<std::mutex> lock(mutex_);
    return std::any_of(assignments_.begin(), assignments_.end(), [&](const auto& item) {
        return item.second.doctorUserId == doctor &&
            item.second.patientUserId == patient && item.second.status == "active";
    });
}

std::vector<domain::DoctorPatientAssignment> InMemoryTreeSemStore::listAssignments()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<domain::DoctorPatientAssignment> result;
    for (const auto& item : assignments_) result.push_back(item.second);
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.createdAt > b.createdAt;
    });
    return result;
}

void InMemoryTreeSemStore::appendAudit(const domain::AuditEvent& event)
{
    std::lock_guard<std::mutex> lock(mutex_); audit_.push_back(event);
}

domain::AuditPage InMemoryTreeSemStore::listAudit(
    std::size_t limit, const std::optional<domain::AuditCursor>& cursor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    domain::AuditPage page;
    for (auto it = audit_.rbegin(); it != audit_.rend(); ++it)
        if (!cursor.has_value() || it->createdAt < cursor->createdAt ||
            (it->createdAt == cursor->createdAt && it->eventId < cursor->eventId))
            page.items.push_back(*it);
    if (page.items.size() > limit)
    {
        page.items.resize(limit);
        page.nextCursor = domain::AuditCursor{
            page.items.back().createdAt, page.items.back().eventId};
    }
    return page;
}

bool InMemoryTreeSemStore::bindSession(
    const std::string& session, const std::string& owner, const std::string& subject)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (sessions_.find(session) == sessions_.end()) return false;
    auto found = sessionOwners_.find(session);
    if (found != sessionOwners_.end() && found->second.first != owner) return false;
    sessionOwners_[session] = {owner, subject}; return true;
}

std::optional<std::string> InMemoryTreeSemStore::sessionSubjectForActor(
    const std::string& session, const std::string& actor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = sessionOwners_.find(session);
    if (found == sessionOwners_.end() || found->second.first != actor) return std::nullopt;
    return found->second.second;
}

bool InMemoryTreeSemStore::isAgentRunRunning(
    const std::string& runId, const std::string& sessionId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = agentRuns_.find(runId);
    return found != agentRuns_.end() && found->second.sessionId == sessionId &&
        found->second.status == domain::AgentRunStatus::Running;
}

} // namespace infrastructure
} // namespace treesem
