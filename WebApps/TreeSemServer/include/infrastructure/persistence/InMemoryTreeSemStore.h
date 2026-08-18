#pragma once

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "persistence/ITreeSemStore.h"
#include "persistence/ISecurityStore.h"

namespace treesem
{
namespace infrastructure
{

class InMemoryTreeSemStore : public persistence::ITreeSemStore,
                             public persistence::ISecurityStore
{
public:
    void createSession(const domain::SessionRecord& session) override;
    std::optional<domain::SessionRecord> touchActiveSession(
        const std::string& sessionId,
        domain::TimePoint accessedAt,
        domain::TimePoint expiresAt) override;
    void savePredictionAndSetCurrent(
        const domain::PredictionRecord& prediction,
        domain::TimePoint sessionExpiresAt) override;
    std::optional<domain::PredictionRecord> findPrediction(
        const std::string& sessionId,
        const std::string& predictionId) override;
    domain::HistoryPage listPredictions(
        const std::string& sessionId,
        std::size_t limit,
        const std::optional<domain::HistoryCursor>& cursor) override;
    std::optional<domain::PredictionRecord> findPredictionBySubject(
        const std::string& subjectUserId,
        const std::string& predictionId) override;
    std::optional<domain::PredictionRecord> findPredictionAny(
        const std::string& predictionId) override;
    domain::HistoryPage listPredictionsBySubject(
        const std::string& subjectUserId,
        std::size_t limit,
        const std::optional<domain::HistoryCursor>& cursor) override;
    domain::FeedbackSaveResult saveFeedback(
        const domain::ClinicalFeedback& feedback) override;
    std::vector<domain::ClinicalFeedback> listFeedback(
        const std::string& sessionId,
        const std::string& predictionId) override;
    domain::AgentRunStartResult startAgentRun(
        const domain::AgentRunRecord& run,
        const domain::ChatMessage& userMessage) override;
    void completeAgentRun(
        const domain::AgentRunRecord& run,
        const domain::ChatMessage& assistantMessage) override;
    void failAgentRun(const std::string& runId,
                      const std::string& errorCode,
                      domain::TimePoint completedAt) override;
    domain::ChatPage listChatMessages(
        const std::string& sessionId,
        std::size_t limit,
        const std::optional<domain::ChatCursor>& cursor) override;
    std::vector<domain::ChatMessage> recentChatMessages(
        const std::string& sessionId, std::size_t limit) override;
    bool ping() override;
    void createUser(const domain::UserRecord& user) override;
    std::optional<domain::UserRecord> findUserByEmail(
        const std::string& normalizedEmail) override;
    std::optional<domain::UserRecord> findUserById(
        const std::string& userId) override;
    bool hasAdminUser() override;
    void recordLoginFailure(const std::string& userId, int failedCount,
                            const std::optional<domain::TimePoint>& lockedUntil,
                            domain::TimePoint now) override;
    void recordLoginSuccess(const std::string& userId,
                            domain::TimePoint now) override;
    void createRefreshSession(const domain::RefreshSession& session) override;
    persistence::RefreshRotateResult rotateRefreshSession(
        const std::string& oldTokenHash,
        const domain::RefreshSession& replacement,
        domain::TimePoint now,
        std::optional<domain::RefreshSession>& oldSession) override;
    void revokeRefreshFamily(const std::string& familyId,
                             domain::TimePoint now) override;
    std::optional<domain::RefreshSession> findRefreshSession(
        const std::string& tokenHash) override;
    domain::DoctorPatientAssignment saveAssignment(
        const domain::DoctorPatientAssignment& assignment) override;
    bool revokeAssignment(const std::string& assignmentId,
                          domain::TimePoint now) override;
    bool hasActiveAssignment(const std::string& doctorUserId,
                             const std::string& patientUserId) override;
    std::vector<domain::DoctorPatientAssignment> listAssignments() override;
    void appendAudit(const domain::AuditEvent& event) override;
    domain::AuditPage listAudit(
        std::size_t limit,
        const std::optional<domain::AuditCursor>& cursor) override;
    bool bindSession(const std::string& sessionId,
                     const std::string& ownerUserId,
                     const std::string& subjectUserId) override;
    std::optional<std::string> sessionSubjectForActor(
        const std::string& sessionId,
        const std::string& actorUserId) override;
    bool isAgentRunRunning(const std::string& runId,
                           const std::string& sessionId) override;

private:
    std::mutex mutex_;
    std::unordered_map<std::string, domain::SessionRecord> sessions_;
    std::unordered_map<std::string, domain::PredictionRecord> predictions_;
    std::vector<domain::ClinicalFeedback> feedback_;
    std::unordered_map<std::string, domain::AgentRunRecord> agentRuns_;
    std::vector<domain::ChatMessage> chatMessages_;
    std::unordered_map<std::string, domain::UserRecord> users_;
    std::unordered_map<std::string, domain::RefreshSession> refreshSessions_;
    std::unordered_map<std::string, domain::DoctorPatientAssignment> assignments_;
    std::vector<domain::AuditEvent> audit_;
    std::unordered_map<std::string, std::pair<std::string, std::string>> sessionOwners_;
};

} // namespace infrastructure
} // namespace treesem
