#include "infrastructure/persistence/MySqlTreeSemStore.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <cppconn/datatype.h>
#include <cppconn/exception.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <muduo/base/Logging.h>
#include <nlohmann/json.hpp>

#include "application/BusinessException.h"
#include "infrastructure/support/ValueSupport.h"
#include "serialization/ModelResultJsonCodec.h"

namespace treesem
{
namespace infrastructure
{
namespace
{

std::string mysqlTime(domain::TimePoint value)
{
    std::string result = formatUtc(value);
    result[10] = ' ';
    result.pop_back();
    return result;
}

domain::TimePoint fromMysqlTime(std::string value)
{
    if (value.size() == 19) value += ".000000";
    if (value.size() != 26) throw std::invalid_argument("invalid MySQL timestamp");
    value[10] = 'T';
    value.push_back('Z');
    return parseUtc(value);
}

std::string stringValue(sql::ResultSet& result, const char* name)
{
    return result.getString(name).asStdString();
}

std::optional<std::string> optionalString(sql::ResultSet& result, const char* name)
{
    const sql::SQLString value = result.getString(name);
    if (result.isNull(name)) return std::nullopt;
    return value.asStdString();
}

domain::SessionRecord readSession(sql::ResultSet& result)
{
    domain::SessionRecord session;
    session.sessionId = stringValue(result, "session_id");
    session.currentPredictionId = optionalString(result, "current_prediction_id");
    session.status = stringValue(result, "status");
    session.version = result.getUInt64("version");
    session.createdAt = fromMysqlTime(stringValue(result, "created_at"));
    session.lastAccessedAt = fromMysqlTime(stringValue(result, "last_accessed_at"));
    session.expiresAt = fromMysqlTime(stringValue(result, "expires_at"));
    return session;
}

domain::PredictionRecord readPrediction(sql::ResultSet& result)
{
    domain::PredictionRecord record;
    record.predictionId = stringValue(result, "prediction_id");
    record.sessionId = stringValue(result, "session_id");
    if (!result.isNull("subject_user_id"))
        record.subjectUserId = stringValue(result, "subject_user_id");
    if (!result.isNull("created_by_user_id"))
        record.createdByUserId = stringValue(result, "created_by_user_id");
    record.result = serialization::ModelResultJsonCodec::decode(
        stringValue(result, "result_json"));
    record.createdAt = fromMysqlTime(stringValue(result, "created_at"));
    return record;
}

domain::ClinicalFeedback readFeedback(sql::ResultSet& result)
{
    domain::ClinicalFeedback feedback;
    feedback.feedbackId = stringValue(result, "feedback_id");
    feedback.predictionId = stringValue(result, "prediction_id");
    feedback.sessionId = stringValue(result, "session_id");
    feedback.reviewerReference = stringValue(result, "reviewer_reference");
    feedback.reviewerVerified = result.getBoolean("reviewer_verified");
    feedback.assessment = domain::parseFeedbackAssessment(
        stringValue(result, "assessment"));
    if (!result.isNull("corrected_label"))
        feedback.correctedLabel = result.getInt("corrected_label");
    feedback.comment = optionalString(result, "comment");
    feedback.idempotencyKey = stringValue(result, "idempotency_key");
    feedback.payloadSha256 = stringValue(result, "payload_sha256");
    feedback.createdAt = fromMysqlTime(stringValue(result, "created_at"));
    return feedback;
}

domain::AgentRunStatus parseAgentRunStatus(const std::string& value)
{
    if (value == "running") return domain::AgentRunStatus::Running;
    if (value == "completed") return domain::AgentRunStatus::Completed;
    if (value == "failed") return domain::AgentRunStatus::Failed;
    throw std::invalid_argument("invalid stored agent run status");
}

domain::ChatMessage readChatMessage(sql::ResultSet& result)
{
    domain::ChatMessage message;
    message.messageId = stringValue(result, "message_id");
    message.sessionId = stringValue(result, "session_id");
    message.runId = stringValue(result, "run_id");
    message.role = stringValue(result, "role");
    message.content = stringValue(result, "content");
    message.createdAt = fromMysqlTime(stringValue(result, "created_at"));
    message.actorUserId = optionalString(result, "actor_user_id");
    message.subjectUserId = optionalString(result, "subject_user_id");
    return message;
}

domain::AgentRunRecord readAgentRun(sql::ResultSet& result)
{
    domain::AgentRunRecord run;
    run.runId = stringValue(result, "run_id");
    run.sessionId = stringValue(result, "session_id");
    run.idempotencyKey = stringValue(result, "idempotency_key");
    run.payloadSha256 = stringValue(result, "payload_sha256");
    run.status = parseAgentRunStatus(stringValue(result, "status"));
    run.stepCount = result.getInt("step_count");
    const auto tools = nlohmann::json::parse(stringValue(result, "tool_summary_json"));
    for (const auto& item : tools)
        run.tools.push_back({item.at("name").get<std::string>(),
                             item.at("status").get<std::string>(),
                             item.at("duration_ms").get<std::uint64_t>()});
    run.groundingPredictionIds = nlohmann::json::parse(
        stringValue(result, "grounding_ids_json")).get<std::vector<std::string>>();
    run.finalMessageId = optionalString(result, "final_message_id");
    run.errorCode = optionalString(result, "error_code");
    run.startedAt = fromMysqlTime(stringValue(result, "started_at"));
    if (!result.isNull("completed_at"))
        run.completedAt = fromMysqlTime(stringValue(result, "completed_at"));
    run.actorUserId = optionalString(result, "actor_user_id");
    run.subjectUserId = optionalString(result, "subject_user_id");
    return run;
}

std::string toolsJson(const std::vector<domain::AgentToolSummary>& tools)
{
    nlohmann::json result = nlohmann::json::array();
    for (const auto& item : tools)
        result.push_back({{"name", item.name}, {"status", item.status},
                          {"duration_ms", item.durationMs}});
    return result.dump();
}

domain::UserRecord readUser(sql::ResultSet& result)
{
    domain::UserRecord user;
    user.userId = stringValue(result, "user_id");
    user.emailNormalized = stringValue(result, "email_normalized");
    user.passwordPhc = stringValue(result, "password_phc");
    user.displayName = stringValue(result, "display_name");
    user.role = domain::parseUserRole(stringValue(result, "role"));
    user.status = stringValue(result, "status");
    user.failedLoginCount = result.getInt("failed_login_count");
    if (!result.isNull("locked_until"))
        user.lockedUntil = fromMysqlTime(stringValue(result, "locked_until"));
    user.tokenVersion = result.getUInt64("token_version");
    user.createdAt = fromMysqlTime(stringValue(result, "created_at"));
    user.updatedAt = fromMysqlTime(stringValue(result, "updated_at"));
    return user;
}

domain::RefreshSession readRefresh(sql::ResultSet& result)
{
    domain::RefreshSession refresh;
    refresh.refreshSessionId = stringValue(result, "refresh_session_id");
    refresh.tokenFamilyId = stringValue(result, "token_family_id");
    refresh.userId = stringValue(result, "user_id");
    refresh.tokenSha256 = stringValue(result, "token_sha256");
    refresh.expiresAt = fromMysqlTime(stringValue(result, "expires_at"));
    if (!result.isNull("consumed_at"))
        refresh.consumedAt = fromMysqlTime(stringValue(result, "consumed_at"));
    if (!result.isNull("revoked_at"))
        refresh.revokedAt = fromMysqlTime(stringValue(result, "revoked_at"));
    refresh.replacedById = optionalString(result, "replaced_by_id");
    refresh.createdAt = fromMysqlTime(stringValue(result, "created_at"));
    refresh.lastUsedAt = fromMysqlTime(stringValue(result, "last_used_at"));
    return refresh;
}

domain::AuditEvent readAudit(sql::ResultSet& result)
{
    domain::AuditEvent event;
    event.eventId = stringValue(result, "event_id");
    event.requestId = stringValue(result, "request_id");
    event.actorUserId = optionalString(result, "actor_user_id");
    const auto role = optionalString(result, "actor_role");
    if (role.has_value()) event.actorRole = domain::parseUserRole(*role);
    event.action = stringValue(result, "action");
    event.resourceType = stringValue(result, "resource_type");
    event.resourceId = optionalString(result, "resource_id");
    event.outcome = stringValue(result, "outcome");
    event.reasonCode = stringValue(result, "reason_code");
    event.createdAt = fromMysqlTime(stringValue(result, "created_at"));
    return event;
}

domain::DoctorPatientAssignment readAssignment(sql::ResultSet& result)
{
    domain::DoctorPatientAssignment item;
    item.assignmentId = stringValue(result, "assignment_id");
    item.doctorUserId = stringValue(result, "doctor_user_id");
    item.patientUserId = stringValue(result, "patient_user_id");
    item.status = stringValue(result, "status");
    item.createdByAdminId = stringValue(result, "created_by_admin_id");
    item.createdAt = fromMysqlTime(stringValue(result, "created_at"));
    if (!result.isNull("revoked_at"))
        item.revokedAt = fromMysqlTime(stringValue(result, "revoked_at"));
    return item;
}

[[noreturn]] void translateSql(const sql::SQLException& error)
{
    const std::string state = error.getSQLState();
    LOG_ERROR << "treeSem MySQL operation failed error_code="
              << error.getErrorCode() << " sql_state=" << state;
    const auto kind = error.getErrorCode() == 1062
        ? application::BusinessException::Kind::Conflict
        : state.rfind("08", 0) == 0
            ? application::BusinessException::Kind::DatabaseUnavailable
            : application::BusinessException::Kind::PersistenceFailure;
    throw application::BusinessException(kind, "MySQL operation failed");
}

template<typename Function>
auto withSqlTranslation(MySqlConnectionPool::Lease& lease, Function&& function)
    -> decltype(function())
{
    try { return function(); }
    catch (const application::BusinessException&) { throw; }
    catch (const sql::SQLException& error)
    {
        if (error.getSQLState().rfind("08", 0) == 0)
            lease.invalidate();
        translateSql(error);
    }
    catch (const nlohmann::json::exception&)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::PersistenceFailure,
            "stored model JSON is invalid");
    }
    catch (const std::exception&)
    {
        throw application::BusinessException(
            application::BusinessException::Kind::PersistenceFailure,
            "stored database value is invalid");
    }
}

void rollbackNoThrow(sql::Connection& connection) noexcept
{
    try { connection.rollback(); connection.setAutoCommit(true); }
    catch (...) {}
}

} // namespace

MySqlStoreConfig MySqlStoreConfig::from(const config::TreeSemServerConfig& config)
{
    MySqlStoreConfig result;
    result.connection.host = config.databaseHost;
    result.connection.port = config.databasePort;
    result.connection.database = config.databaseName;
    result.connection.user = config.databaseUser;
    result.connection.password = config.databasePassword;
    result.connection.poolSize = config.databasePoolSize;
    result.connection.acquireTimeout =
        std::chrono::milliseconds(config.databaseAcquireTimeoutMs);
    result.connection.connectTimeoutSeconds = static_cast<int>(
        std::max(1L, (config.databaseConnectTimeoutMs + 999) / 1000));
    result.connection.readTimeoutSeconds = static_cast<int>(
        std::max(1L, (config.databaseReadTimeoutMs + 999) / 1000));
    result.connection.writeTimeoutSeconds = static_cast<int>(
        std::max(1L, (config.databaseWriteTimeoutMs + 999) / 1000));
    return result;
}

MySqlTreeSemStore::MySqlTreeSemStore(MySqlStoreConfig config)
    : pool_(std::move(config.connection))
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(
            lease.connection().prepareStatement(
                "SELECT COUNT(*) AS count FROM schema_migrations WHERE version = ?"));
        statement->setString(1, "003_m6_auth_security");
        std::unique_ptr<sql::ResultSet> result(statement->executeQuery());
        if (!result->next() || result->getInt("count") != 1)
        {
            throw application::BusinessException(
                application::BusinessException::Kind::PersistenceFailure,
                "required database migration is missing");
        }
        return 0;
    });
}

void MySqlTreeSemStore::createSession(const domain::SessionRecord& session)
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(
            lease.connection().prepareStatement(
                "INSERT INTO treesem_sessions "
                "(session_id,current_prediction_id,status,version,created_at,last_accessed_at,expires_at) "
                "VALUES (?,NULL,?,?,?,?,?)"));
        statement->setString(1, session.sessionId);
        statement->setString(2, session.status);
        statement->setUInt64(3, session.version);
        statement->setString(4, mysqlTime(session.createdAt));
        statement->setString(5, mysqlTime(session.lastAccessedAt));
        statement->setString(6, mysqlTime(session.expiresAt));
        statement->executeUpdate();
        return 0;
    });
}

std::optional<domain::SessionRecord> MySqlTreeSemStore::touchActiveSession(
    const std::string& sessionId,
    domain::TimePoint accessedAt,
    domain::TimePoint expiresAt)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() -> std::optional<domain::SessionRecord> {
        std::unique_ptr<sql::PreparedStatement> update(
            lease.connection().prepareStatement(
                "UPDATE treesem_sessions SET last_accessed_at=?,expires_at=?,version=version+1 "
                "WHERE session_id=? AND status='active' AND expires_at>?"));
        update->setString(1, mysqlTime(accessedAt));
        update->setString(2, mysqlTime(expiresAt));
        update->setString(3, sessionId);
        update->setString(4, mysqlTime(accessedAt));
        if (update->executeUpdate() != 1) return std::nullopt;
        std::unique_ptr<sql::PreparedStatement> select(
            lease.connection().prepareStatement(
                "SELECT * FROM treesem_sessions WHERE session_id=?"));
        select->setString(1, sessionId);
        std::unique_ptr<sql::ResultSet> result(select->executeQuery());
        if (!result->next()) return std::nullopt;
        return readSession(*result);
    });
}

void MySqlTreeSemStore::savePredictionAndSetCurrent(
    const domain::PredictionRecord& prediction,
    domain::TimePoint sessionExpiresAt)
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        sql::Connection& connection = lease.connection();
        connection.setAutoCommit(false);
        try
        {
            std::unique_ptr<sql::PreparedStatement> lock(
                connection.prepareStatement(
                    "SELECT session_id FROM treesem_sessions "
                    "WHERE session_id=? AND status='active' AND expires_at>? FOR UPDATE"));
            lock->setString(1, prediction.sessionId);
            lock->setString(2, mysqlTime(prediction.createdAt));
            std::unique_ptr<sql::ResultSet> locked(lock->executeQuery());
            if (!locked->next())
            {
                rollbackNoThrow(connection);
                throw application::BusinessException(
                    application::BusinessException::Kind::Conflict,
                    "session expired before prediction commit");
            }
            std::unique_ptr<sql::PreparedStatement> insert(
                connection.prepareStatement(
                    "INSERT INTO treesem_predictions "
                    "(prediction_id,session_id,model_name,model_version,serving_backend,input_source,"
                    "sample_index,label,positive_probability,confidence,cluster_id,tree_probability,"
                    "tree_leaf_id,result_json,created_at,subject_user_id,created_by_user_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"));
            insert->setString(1, prediction.predictionId);
            insert->setString(2, prediction.sessionId);
            insert->setString(3, prediction.result.modelName);
            if (prediction.result.modelVersion.has_value()) insert->setString(4, *prediction.result.modelVersion);
            else insert->setNull(4, sql::DataType::VARCHAR);
            if (prediction.result.servingBackend.has_value()) insert->setString(5, *prediction.result.servingBackend);
            else insert->setNull(5, sql::DataType::VARCHAR);
            insert->setString(6, prediction.result.inputSource);
            if (prediction.result.sampleIndex.has_value()) insert->setInt64(7, *prediction.result.sampleIndex);
            else insert->setNull(7, sql::DataType::BIGINT);
            insert->setInt(8, prediction.result.prediction.label);
            insert->setDouble(9, prediction.result.prediction.positiveProbability);
            insert->setDouble(10, prediction.result.prediction.confidence);
            insert->setInt(11, prediction.result.prediction.clusterId);
            insert->setDouble(12, prediction.result.prediction.treeProbability);
            insert->setInt(13, prediction.result.prediction.treeLeafId);
            insert->setString(14, serialization::ModelResultJsonCodec::encode(prediction.result));
            insert->setString(15, mysqlTime(prediction.createdAt));
            if (prediction.subjectUserId.has_value()) insert->setString(16, *prediction.subjectUserId);
            else insert->setNull(16, sql::DataType::VARCHAR);
            if (prediction.createdByUserId.has_value()) insert->setString(17, *prediction.createdByUserId);
            else insert->setNull(17, sql::DataType::VARCHAR);
            insert->executeUpdate();

            std::unique_ptr<sql::PreparedStatement> update(
                connection.prepareStatement(
                    "UPDATE treesem_sessions SET current_prediction_id=?,last_accessed_at=?,"
                    "expires_at=?,version=version+1 WHERE session_id=?"));
            update->setString(1, prediction.predictionId);
            update->setString(2, mysqlTime(prediction.createdAt));
            update->setString(3, mysqlTime(sessionExpiresAt));
            update->setString(4, prediction.sessionId);
            if (update->executeUpdate() != 1)
                throw std::runtime_error("session update did not affect one row");
            connection.commit();
            connection.setAutoCommit(true);
        }
        catch (...)
        {
            rollbackNoThrow(connection);
            throw;
        }
        return 0;
    });
}

std::optional<domain::PredictionRecord> MySqlTreeSemStore::findPrediction(
    const std::string& sessionId,
    const std::string& predictionId)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() -> std::optional<domain::PredictionRecord> {
        std::unique_ptr<sql::PreparedStatement> statement(
            lease.connection().prepareStatement(
                "SELECT prediction_id,session_id,subject_user_id,created_by_user_id,result_json,created_at "
                "FROM treesem_predictions WHERE session_id=? AND prediction_id=?"));
        statement->setString(1, sessionId);
        statement->setString(2, predictionId);
        std::unique_ptr<sql::ResultSet> result(statement->executeQuery());
        if (!result->next()) return std::nullopt;
        return readPrediction(*result);
    });
}

domain::HistoryPage MySqlTreeSemStore::listPredictions(
    const std::string& sessionId,
    std::size_t limit,
    const std::optional<domain::HistoryCursor>& cursor)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        const std::string sqlText = cursor.has_value()
            ? "SELECT prediction_id,created_at,label,positive_probability,confidence,model_version,serving_backend "
              "FROM treesem_predictions WHERE session_id=? AND "
              "(created_at<? OR (created_at=? AND prediction_id<?)) "
              "ORDER BY created_at DESC,prediction_id DESC LIMIT ?"
            : "SELECT prediction_id,created_at,label,positive_probability,confidence,model_version,serving_backend "
              "FROM treesem_predictions WHERE session_id=? "
              "ORDER BY created_at DESC,prediction_id DESC LIMIT ?";
        std::unique_ptr<sql::PreparedStatement> statement(
            lease.connection().prepareStatement(sqlText));
        statement->setString(1, sessionId);
        int parameter = 2;
        if (cursor.has_value())
        {
            statement->setString(parameter++, mysqlTime(cursor->createdAt));
            statement->setString(parameter++, mysqlTime(cursor->createdAt));
            statement->setString(parameter++, cursor->predictionId);
        }
        statement->setUInt(parameter, static_cast<unsigned int>(limit + 1));
        std::unique_ptr<sql::ResultSet> rows(statement->executeQuery());
        domain::HistoryPage page;
        page.sessionId = sessionId;
        while (rows->next())
        {
            page.items.push_back({
                stringValue(*rows, "prediction_id"),
                fromMysqlTime(stringValue(*rows, "created_at")),
                rows->getInt("label"),
                static_cast<double>(rows->getDouble("positive_probability")),
                static_cast<double>(rows->getDouble("confidence")),
                optionalString(*rows, "model_version"),
                optionalString(*rows, "serving_backend")});
        }
        if (page.items.size() > limit)
        {
            page.items.resize(limit);
            page.nextCursor = domain::HistoryCursor{
                page.items.back().createdAt, page.items.back().predictionId};
        }
        return page;
    });
}

std::optional<domain::PredictionRecord> MySqlTreeSemStore::findPredictionBySubject(
    const std::string& subject, const std::string& predictionId)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() -> std::optional<domain::PredictionRecord> {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT prediction_id,session_id,subject_user_id,created_by_user_id,result_json,created_at "
            "FROM treesem_predictions WHERE subject_user_id=? AND prediction_id=?"));
        statement->setString(1, subject); statement->setString(2, predictionId);
        std::unique_ptr<sql::ResultSet> row(statement->executeQuery());
        return row->next() ? std::optional<domain::PredictionRecord>(readPrediction(*row))
                           : std::nullopt;
    });
}

std::optional<domain::PredictionRecord> MySqlTreeSemStore::findPredictionAny(
    const std::string& predictionId)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() -> std::optional<domain::PredictionRecord> {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT prediction_id,session_id,subject_user_id,created_by_user_id,result_json,created_at "
            "FROM treesem_predictions WHERE prediction_id=?"));
        statement->setString(1, predictionId);
        std::unique_ptr<sql::ResultSet> row(statement->executeQuery());
        return row->next() ? std::optional<domain::PredictionRecord>(readPrediction(*row))
                           : std::nullopt;
    });
}

domain::HistoryPage MySqlTreeSemStore::listPredictionsBySubject(
    const std::string& subject, std::size_t limit,
    const std::optional<domain::HistoryCursor>& cursor)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        const std::string query = cursor.has_value()
            ? "SELECT prediction_id,created_at,label,positive_probability,confidence,model_version,serving_backend "
              "FROM treesem_predictions WHERE subject_user_id=? AND "
              "(created_at<? OR (created_at=? AND prediction_id<?)) "
              "ORDER BY created_at DESC,prediction_id DESC LIMIT ?"
            : "SELECT prediction_id,created_at,label,positive_probability,confidence,model_version,serving_backend "
              "FROM treesem_predictions WHERE subject_user_id=? "
              "ORDER BY created_at DESC,prediction_id DESC LIMIT ?";
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(query));
        statement->setString(1, subject); int parameter = 2;
        if (cursor.has_value())
        {
            statement->setString(parameter++, mysqlTime(cursor->createdAt));
            statement->setString(parameter++, mysqlTime(cursor->createdAt));
            statement->setString(parameter++, cursor->predictionId);
        }
        statement->setUInt(parameter, static_cast<unsigned>(limit + 1));
        std::unique_ptr<sql::ResultSet> rows(statement->executeQuery());
        domain::HistoryPage page; page.sessionId = subject;
        while (rows->next())
            page.items.push_back({stringValue(*rows, "prediction_id"),
                fromMysqlTime(stringValue(*rows, "created_at")), rows->getInt("label"),
                static_cast<double>(rows->getDouble("positive_probability")),
                static_cast<double>(rows->getDouble("confidence")),
                optionalString(*rows, "model_version"), optionalString(*rows, "serving_backend")});
        if (page.items.size() > limit)
        {
            page.items.resize(limit);
            page.nextCursor = domain::HistoryCursor{
                page.items.back().createdAt, page.items.back().predictionId};
        }
        return page;
    });
}

domain::FeedbackSaveResult MySqlTreeSemStore::saveFeedback(
    const domain::ClinicalFeedback& feedback)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        sql::Connection& connection = lease.connection();
        connection.setAutoCommit(false);
        try
        {
            std::unique_ptr<sql::PreparedStatement> lock(
                connection.prepareStatement(
                    "SELECT session_id FROM treesem_sessions WHERE session_id=? FOR UPDATE"));
            lock->setString(1, feedback.sessionId);
            std::unique_ptr<sql::ResultSet> locked(lock->executeQuery());
            if (!locked->next())
                throw application::BusinessException(
                    application::BusinessException::Kind::NotFound, "session not found");
            std::unique_ptr<sql::PreparedStatement> existingStatement(
                connection.prepareStatement(
                    "SELECT * FROM treesem_prediction_feedback "
                    "WHERE session_id=? AND idempotency_key=?"));
            existingStatement->setString(1, feedback.sessionId);
            existingStatement->setString(2, feedback.idempotencyKey);
            std::unique_ptr<sql::ResultSet> existing(existingStatement->executeQuery());
            if (existing->next())
            {
                domain::ClinicalFeedback stored = readFeedback(*existing);
                if (stored.payloadSha256 != feedback.payloadSha256)
                    throw application::BusinessException(
                        application::BusinessException::Kind::IdempotencyConflict,
                        "idempotency key conflict");
                connection.commit();
                connection.setAutoCommit(true);
                return domain::FeedbackSaveResult{std::move(stored), false};
            }
            std::unique_ptr<sql::PreparedStatement> prediction(
                connection.prepareStatement(
                    "SELECT prediction_id FROM treesem_predictions "
                    "WHERE prediction_id=? AND session_id=?"));
            prediction->setString(1, feedback.predictionId);
            prediction->setString(2, feedback.sessionId);
            std::unique_ptr<sql::ResultSet> predictionRow(prediction->executeQuery());
            if (!predictionRow->next())
                throw application::BusinessException(
                    application::BusinessException::Kind::NotFound,
                    "prediction not found");
            std::unique_ptr<sql::PreparedStatement> insert(
                connection.prepareStatement(
                    "INSERT INTO treesem_prediction_feedback "
                    "(feedback_id,prediction_id,session_id,reviewer_reference,reviewer_verified,"
                    "assessment,corrected_label,comment,idempotency_key,payload_sha256,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)"));
            insert->setString(1, feedback.feedbackId);
            insert->setString(2, feedback.predictionId);
            insert->setString(3, feedback.sessionId);
            insert->setString(4, feedback.reviewerReference);
            insert->setBoolean(5, feedback.reviewerVerified);
            insert->setString(6, domain::toString(feedback.assessment));
            if (feedback.correctedLabel.has_value()) insert->setInt(7, *feedback.correctedLabel);
            else insert->setNull(7, sql::DataType::TINYINT);
            if (feedback.comment.has_value()) insert->setString(8, *feedback.comment);
            else insert->setNull(8, sql::DataType::VARCHAR);
            insert->setString(9, feedback.idempotencyKey);
            insert->setString(10, feedback.payloadSha256);
            insert->setString(11, mysqlTime(feedback.createdAt));
            insert->executeUpdate();
            connection.commit();
            connection.setAutoCommit(true);
            return domain::FeedbackSaveResult{feedback, true};
        }
        catch (...)
        {
            rollbackNoThrow(connection);
            throw;
        }
    });
}

std::vector<domain::ClinicalFeedback> MySqlTreeSemStore::listFeedback(
    const std::string& sessionId,
    const std::string& predictionId)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> prediction(
            lease.connection().prepareStatement(
                "SELECT prediction_id FROM treesem_predictions "
                "WHERE prediction_id=? AND session_id=?"));
        prediction->setString(1, predictionId);
        prediction->setString(2, sessionId);
        std::unique_ptr<sql::ResultSet> found(prediction->executeQuery());
        if (!found->next())
            throw application::BusinessException(
                application::BusinessException::Kind::NotFound,
                "prediction not found");
        std::unique_ptr<sql::PreparedStatement> statement(
            lease.connection().prepareStatement(
                "SELECT * FROM treesem_prediction_feedback "
                "WHERE session_id=? AND prediction_id=? ORDER BY created_at,feedback_id"));
        statement->setString(1, sessionId);
        statement->setString(2, predictionId);
        std::unique_ptr<sql::ResultSet> rows(statement->executeQuery());
        std::vector<domain::ClinicalFeedback> result;
        while (rows->next()) result.push_back(readFeedback(*rows));
        return result;
    });
}

bool MySqlTreeSemStore::ping()
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::Statement> statement(
            lease.connection().createStatement());
        std::unique_ptr<sql::ResultSet> result(statement->executeQuery("SELECT 1"));
        return result->next();
    });
}

domain::AgentRunStartResult MySqlTreeSemStore::startAgentRun(
    const domain::AgentRunRecord& run,
    const domain::ChatMessage& userMessage)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        auto& connection = lease.connection();
        connection.setAutoCommit(false);
        try
        {
            std::unique_ptr<sql::PreparedStatement> lock(connection.prepareStatement(
                "SELECT session_id FROM treesem_sessions WHERE session_id=? FOR UPDATE"));
            lock->setString(1, run.sessionId);
            std::unique_ptr<sql::ResultSet> locked(lock->executeQuery());
            if (!locked->next()) throw application::BusinessException(
                application::BusinessException::Kind::NotFound, "session not found");
            std::unique_ptr<sql::PreparedStatement> existing(connection.prepareStatement(
                "SELECT * FROM treesem_agent_runs WHERE session_id=? AND idempotency_key=?"));
            existing->setString(1, run.sessionId);
            existing->setString(2, run.idempotencyKey);
            std::unique_ptr<sql::ResultSet> row(existing->executeQuery());
            if (row->next())
            {
                domain::AgentRunRecord stored = readAgentRun(*row);
                if (stored.payloadSha256 != run.payloadSha256)
                    throw application::BusinessException(
                        application::BusinessException::Kind::IdempotencyConflict,
                        "chat idempotency conflict");
                std::optional<domain::ChatMessage> final;
                if (stored.finalMessageId.has_value())
                {
                    std::unique_ptr<sql::PreparedStatement> message(connection.prepareStatement(
                        "SELECT * FROM treesem_chat_messages WHERE message_id=?"));
                    message->setString(1, *stored.finalMessageId);
                    std::unique_ptr<sql::ResultSet> messageRow(message->executeQuery());
                    if (messageRow->next()) final = readChatMessage(*messageRow);
                }
                connection.commit(); connection.setAutoCommit(true);
                return domain::AgentRunStartResult{std::move(stored), std::move(final), false};
            }
            std::unique_ptr<sql::PreparedStatement> insertRun(connection.prepareStatement(
                "INSERT INTO treesem_agent_runs "
                "(run_id,session_id,idempotency_key,payload_sha256,status,step_count,"
                "tool_summary_json,grounding_ids_json,started_at,actor_user_id,subject_user_id) "
                "VALUES (?,?,?,?,?,0,'[]','[]',?,?,?)"));
            insertRun->setString(1, run.runId); insertRun->setString(2, run.sessionId);
            insertRun->setString(3, run.idempotencyKey); insertRun->setString(4, run.payloadSha256);
            insertRun->setString(5, domain::toString(run.status));
            insertRun->setString(6, mysqlTime(run.startedAt));
            if (run.actorUserId.has_value()) insertRun->setString(7, *run.actorUserId);
            else insertRun->setNull(7, sql::DataType::VARCHAR);
            if (run.subjectUserId.has_value()) insertRun->setString(8, *run.subjectUserId);
            else insertRun->setNull(8, sql::DataType::VARCHAR);
            insertRun->executeUpdate();
            std::unique_ptr<sql::PreparedStatement> insertMessage(connection.prepareStatement(
                "INSERT INTO treesem_chat_messages "
                "(message_id,session_id,run_id,role,content,created_at,actor_user_id,subject_user_id) "
                "VALUES (?,?,?,?,?,?,?,?)"));
            insertMessage->setString(1, userMessage.messageId);
            insertMessage->setString(2, userMessage.sessionId);
            insertMessage->setString(3, userMessage.runId);
            insertMessage->setString(4, userMessage.role);
            insertMessage->setString(5, userMessage.content);
            insertMessage->setString(6, mysqlTime(userMessage.createdAt));
            if (userMessage.actorUserId.has_value())
                insertMessage->setString(7, *userMessage.actorUserId);
            else insertMessage->setNull(7, sql::DataType::VARCHAR);
            if (userMessage.subjectUserId.has_value())
                insertMessage->setString(8, *userMessage.subjectUserId);
            else insertMessage->setNull(8, sql::DataType::VARCHAR);
            insertMessage->executeUpdate();
            connection.commit(); connection.setAutoCommit(true);
            return domain::AgentRunStartResult{run, std::nullopt, true};
        }
        catch (...) { rollbackNoThrow(connection); throw; }
    });
}

void MySqlTreeSemStore::completeAgentRun(
    const domain::AgentRunRecord& run,
    const domain::ChatMessage& assistantMessage)
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        auto& connection = lease.connection(); connection.setAutoCommit(false);
        try
        {
            std::unique_ptr<sql::PreparedStatement> insert(connection.prepareStatement(
                "INSERT INTO treesem_chat_messages "
                "(message_id,session_id,run_id,role,content,created_at,actor_user_id,subject_user_id) "
                "VALUES (?,?,?,?,?,?,?,?)"));
            insert->setString(1, assistantMessage.messageId);
            insert->setString(2, assistantMessage.sessionId);
            insert->setString(3, assistantMessage.runId);
            insert->setString(4, assistantMessage.role);
            insert->setString(5, assistantMessage.content);
            insert->setString(6, mysqlTime(assistantMessage.createdAt));
            if (assistantMessage.actorUserId.has_value())
                insert->setString(7, *assistantMessage.actorUserId);
            else insert->setNull(7, sql::DataType::VARCHAR);
            if (assistantMessage.subjectUserId.has_value())
                insert->setString(8, *assistantMessage.subjectUserId);
            else insert->setNull(8, sql::DataType::VARCHAR);
            insert->executeUpdate();
            std::unique_ptr<sql::PreparedStatement> update(connection.prepareStatement(
                "UPDATE treesem_agent_runs SET status='completed',step_count=?,"
                "tool_summary_json=?,grounding_ids_json=?,final_message_id=?,completed_at=? "
                "WHERE run_id=? AND status='running'"));
            update->setInt(1, run.stepCount); update->setString(2, toolsJson(run.tools));
            update->setString(3, nlohmann::json(run.groundingPredictionIds).dump());
            update->setString(4, assistantMessage.messageId);
            update->setString(5, mysqlTime(*run.completedAt)); update->setString(6, run.runId);
            if (update->executeUpdate() != 1) throw application::BusinessException(
                application::BusinessException::Kind::Conflict, "agent run is not active");
            connection.commit(); connection.setAutoCommit(true);
        }
        catch (...) { rollbackNoThrow(connection); throw; }
        return 0;
    });
}

void MySqlTreeSemStore::failAgentRun(
    const std::string& runId, const std::string& errorCode,
    domain::TimePoint completedAt)
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "UPDATE treesem_agent_runs SET status='failed',error_code=?,completed_at=? "
            "WHERE run_id=? AND status='running'"));
        statement->setString(1, errorCode); statement->setString(2, mysqlTime(completedAt));
        statement->setString(3, runId); statement->executeUpdate(); return 0;
    });
}

domain::ChatPage MySqlTreeSemStore::listChatMessages(
    const std::string& sessionId, std::size_t limit,
    const std::optional<domain::ChatCursor>& cursor)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        const std::string query = cursor.has_value()
            ? "SELECT * FROM treesem_chat_messages WHERE session_id=? AND "
              "(created_at<? OR (created_at=? AND message_id<?)) "
              "ORDER BY created_at DESC,message_id DESC LIMIT ?"
            : "SELECT * FROM treesem_chat_messages WHERE session_id=? "
              "ORDER BY created_at DESC,message_id DESC LIMIT ?";
        std::unique_ptr<sql::PreparedStatement> statement(
            lease.connection().prepareStatement(query));
        statement->setString(1, sessionId); int parameter = 2;
        if (cursor.has_value())
        {
            statement->setString(parameter++, mysqlTime(cursor->createdAt));
            statement->setString(parameter++, mysqlTime(cursor->createdAt));
            statement->setString(parameter++, cursor->messageId);
        }
        statement->setUInt(parameter, static_cast<unsigned>(limit + 1));
        std::unique_ptr<sql::ResultSet> rows(statement->executeQuery());
        domain::ChatPage page; page.sessionId = sessionId;
        while (rows->next()) page.items.push_back(readChatMessage(*rows));
        if (page.items.size() > limit)
        {
            page.items.resize(limit);
            page.nextCursor = domain::ChatCursor{
                page.items.back().createdAt, page.items.back().messageId};
        }
        return page;
    });
}

std::vector<domain::ChatMessage> MySqlTreeSemStore::recentChatMessages(
    const std::string& sessionId, std::size_t limit)
{
    auto page = listChatMessages(sessionId, limit, std::nullopt);
    std::reverse(page.items.begin(), page.items.end());
    return page.items;
}

void MySqlTreeSemStore::createUser(const domain::UserRecord& user)
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "INSERT INTO treesem_users (user_id,email_normalized,password_phc,display_name,"
            "role,status,failed_login_count,locked_until,token_version,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,NULL,?,?,?)"));
        statement->setString(1, user.userId); statement->setString(2, user.emailNormalized);
        statement->setString(3, user.passwordPhc); statement->setString(4, user.displayName);
        statement->setString(5, domain::toString(user.role)); statement->setString(6, user.status);
        statement->setInt(7, user.failedLoginCount); statement->setUInt64(8, user.tokenVersion);
        statement->setString(9, mysqlTime(user.createdAt)); statement->setString(10, mysqlTime(user.updatedAt));
        statement->executeUpdate(); return 0;
    });
}

std::optional<domain::UserRecord> MySqlTreeSemStore::findUserByEmail(
    const std::string& email)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() -> std::optional<domain::UserRecord> {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT * FROM treesem_users WHERE email_normalized=?"));
        statement->setString(1, email); std::unique_ptr<sql::ResultSet> row(statement->executeQuery());
        return row->next() ? std::optional<domain::UserRecord>(readUser(*row)) : std::nullopt;
    });
}

std::optional<domain::UserRecord> MySqlTreeSemStore::findUserById(const std::string& id)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() -> std::optional<domain::UserRecord> {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT * FROM treesem_users WHERE user_id=?"));
        statement->setString(1, id); std::unique_ptr<sql::ResultSet> row(statement->executeQuery());
        return row->next() ? std::optional<domain::UserRecord>(readUser(*row)) : std::nullopt;
    });
}

bool MySqlTreeSemStore::hasAdminUser()
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT COUNT(*) AS count FROM treesem_users WHERE role='admin'"));
        std::unique_ptr<sql::ResultSet> row(statement->executeQuery());
        return row->next() && row->getInt("count") > 0;
    });
}

void MySqlTreeSemStore::recordLoginFailure(
    const std::string& id, int count, const std::optional<domain::TimePoint>& locked,
    domain::TimePoint now)
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "UPDATE treesem_users SET failed_login_count=?,locked_until=?,updated_at=? WHERE user_id=?"));
        statement->setInt(1, count);
        if (locked.has_value()) statement->setString(2, mysqlTime(*locked));
        else statement->setNull(2, sql::DataType::TIMESTAMP);
        statement->setString(3, mysqlTime(now)); statement->setString(4, id);
        statement->executeUpdate(); return 0;
    });
}

void MySqlTreeSemStore::recordLoginSuccess(const std::string& id, domain::TimePoint now)
{
    recordLoginFailure(id, 0, std::nullopt, now);
}

void MySqlTreeSemStore::createRefreshSession(const domain::RefreshSession& refresh)
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "INSERT INTO treesem_refresh_sessions "
            "(refresh_session_id,token_family_id,user_id,token_sha256,expires_at,"
            "created_at,last_used_at) VALUES (?,?,?,?,?,?,?)"));
        statement->setString(1, refresh.refreshSessionId);
        statement->setString(2, refresh.tokenFamilyId); statement->setString(3, refresh.userId);
        statement->setString(4, refresh.tokenSha256); statement->setString(5, mysqlTime(refresh.expiresAt));
        statement->setString(6, mysqlTime(refresh.createdAt)); statement->setString(7, mysqlTime(refresh.lastUsedAt));
        statement->executeUpdate(); return 0;
    });
}

std::optional<domain::RefreshSession> MySqlTreeSemStore::findRefreshSession(
    const std::string& hash)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() -> std::optional<domain::RefreshSession> {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT * FROM treesem_refresh_sessions WHERE token_sha256=?"));
        statement->setString(1, hash); std::unique_ptr<sql::ResultSet> row(statement->executeQuery());
        return row->next() ? std::optional<domain::RefreshSession>(readRefresh(*row)) : std::nullopt;
    });
}

persistence::RefreshRotateResult MySqlTreeSemStore::rotateRefreshSession(
    const std::string& hash, const domain::RefreshSession& replacement,
    domain::TimePoint now, std::optional<domain::RefreshSession>& oldSession)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        auto& connection = lease.connection(); connection.setAutoCommit(false);
        try
        {
            std::unique_ptr<sql::PreparedStatement> select(connection.prepareStatement(
                "SELECT * FROM treesem_refresh_sessions WHERE token_sha256=? FOR UPDATE"));
            select->setString(1, hash); std::unique_ptr<sql::ResultSet> row(select->executeQuery());
            if (!row->next()) { connection.commit(); connection.setAutoCommit(true);
                return persistence::RefreshRotateResult::NotFound; }
            oldSession = readRefresh(*row);
            if (oldSession->consumedAt.has_value() || oldSession->revokedAt.has_value())
            {
                std::unique_ptr<sql::PreparedStatement> revoke(connection.prepareStatement(
                    "UPDATE treesem_refresh_sessions SET revoked_at=? "
                    "WHERE token_family_id=? AND revoked_at IS NULL"));
                revoke->setString(1, mysqlTime(now)); revoke->setString(2, oldSession->tokenFamilyId);
                revoke->executeUpdate(); connection.commit(); connection.setAutoCommit(true);
                return persistence::RefreshRotateResult::Reused;
            }
            if (oldSession->expiresAt <= now) { connection.commit(); connection.setAutoCommit(true);
                return persistence::RefreshRotateResult::Expired; }
            std::unique_ptr<sql::PreparedStatement> update(connection.prepareStatement(
                "UPDATE treesem_refresh_sessions SET consumed_at=?,replaced_by_id=? "
                "WHERE refresh_session_id=?"));
            update->setString(1, mysqlTime(now)); update->setString(2, replacement.refreshSessionId);
            update->setString(3, oldSession->refreshSessionId); update->executeUpdate();
            std::unique_ptr<sql::PreparedStatement> insert(connection.prepareStatement(
                "INSERT INTO treesem_refresh_sessions "
                "(refresh_session_id,token_family_id,user_id,token_sha256,expires_at,created_at,last_used_at) "
                "VALUES (?,?,?,?,?,?,?)"));
            insert->setString(1, replacement.refreshSessionId); insert->setString(2, replacement.tokenFamilyId);
            insert->setString(3, replacement.userId); insert->setString(4, replacement.tokenSha256);
            insert->setString(5, mysqlTime(replacement.expiresAt)); insert->setString(6, mysqlTime(replacement.createdAt));
            insert->setString(7, mysqlTime(replacement.lastUsedAt)); insert->executeUpdate();
            connection.commit(); connection.setAutoCommit(true);
            return persistence::RefreshRotateResult::Rotated;
        }
        catch (...) { rollbackNoThrow(connection); throw; }
    });
}

void MySqlTreeSemStore::revokeRefreshFamily(
    const std::string& family, domain::TimePoint now)
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "UPDATE treesem_refresh_sessions SET revoked_at=? "
            "WHERE token_family_id=? AND revoked_at IS NULL"));
        statement->setString(1, mysqlTime(now)); statement->setString(2, family);
        statement->executeUpdate(); return 0;
    });
}

domain::DoctorPatientAssignment MySqlTreeSemStore::saveAssignment(
    const domain::DoctorPatientAssignment& assignment)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "INSERT INTO treesem_doctor_patient_assignments "
            "(assignment_id,doctor_user_id,patient_user_id,status,created_by_admin_id,created_at) "
            "VALUES (?,?,?,?,?,?) ON DUPLICATE KEY UPDATE status='active',revoked_at=NULL,"
            "created_by_admin_id=VALUES(created_by_admin_id),created_at=VALUES(created_at)"));
        statement->setString(1, assignment.assignmentId); statement->setString(2, assignment.doctorUserId);
        statement->setString(3, assignment.patientUserId); statement->setString(4, assignment.status);
        statement->setString(5, assignment.createdByAdminId); statement->setString(6, mysqlTime(assignment.createdAt));
        statement->executeUpdate();
        std::unique_ptr<sql::PreparedStatement> select(lease.connection().prepareStatement(
            "SELECT assignment_id FROM treesem_doctor_patient_assignments "
            "WHERE doctor_user_id=? AND patient_user_id=?"));
        select->setString(1, assignment.doctorUserId); select->setString(2, assignment.patientUserId);
        std::unique_ptr<sql::ResultSet> row(select->executeQuery()); row->next();
        auto result = assignment; result.assignmentId = stringValue(*row, "assignment_id"); return result;
    });
}

bool MySqlTreeSemStore::revokeAssignment(const std::string& id, domain::TimePoint now)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "UPDATE treesem_doctor_patient_assignments SET status='revoked',revoked_at=? "
            "WHERE assignment_id=? AND status='active'"));
        statement->setString(1, mysqlTime(now)); statement->setString(2, id);
        return statement->executeUpdate() == 1;
    });
}

bool MySqlTreeSemStore::hasActiveAssignment(
    const std::string& doctor, const std::string& patient)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT COUNT(*) AS count FROM treesem_doctor_patient_assignments "
            "WHERE doctor_user_id=? AND patient_user_id=? AND status='active'"));
        statement->setString(1, doctor); statement->setString(2, patient);
        std::unique_ptr<sql::ResultSet> row(statement->executeQuery());
        return row->next() && row->getInt("count") == 1;
    });
}

std::vector<domain::DoctorPatientAssignment> MySqlTreeSemStore::listAssignments()
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT * FROM treesem_doctor_patient_assignments "
            "ORDER BY created_at DESC,assignment_id DESC"));
        std::unique_ptr<sql::ResultSet> rows(statement->executeQuery());
        std::vector<domain::DoctorPatientAssignment> result;
        while (rows->next()) result.push_back(readAssignment(*rows));
        return result;
    });
}

void MySqlTreeSemStore::appendAudit(const domain::AuditEvent& event)
{
    auto lease = pool_.acquire();
    withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "INSERT INTO treesem_audit_events "
            "(event_id,request_id,actor_user_id,actor_role,action,resource_type,resource_id,"
            "outcome,reason_code,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)"));
        statement->setString(1, event.eventId); statement->setString(2, event.requestId);
        if (event.actorUserId.has_value()) statement->setString(3, *event.actorUserId);
        else statement->setNull(3, sql::DataType::VARCHAR);
        if (event.actorRole.has_value()) statement->setString(4, domain::toString(*event.actorRole));
        else statement->setNull(4, sql::DataType::VARCHAR);
        statement->setString(5, event.action); statement->setString(6, event.resourceType);
        if (event.resourceId.has_value()) statement->setString(7, *event.resourceId);
        else statement->setNull(7, sql::DataType::VARCHAR);
        statement->setString(8, event.outcome); statement->setString(9, event.reasonCode);
        statement->setString(10, mysqlTime(event.createdAt)); statement->executeUpdate(); return 0;
    });
}

domain::AuditPage MySqlTreeSemStore::listAudit(
    std::size_t limit, const std::optional<domain::AuditCursor>& cursor)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        const std::string query = cursor.has_value()
            ? "SELECT * FROM treesem_audit_events WHERE "
              "(created_at<? OR (created_at=? AND event_id<?)) "
              "ORDER BY created_at DESC,event_id DESC LIMIT ?"
            : "SELECT * FROM treesem_audit_events ORDER BY created_at DESC,event_id DESC LIMIT ?";
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(query));
        int parameter = 1;
        if (cursor.has_value())
        {
            statement->setString(parameter++, mysqlTime(cursor->createdAt));
            statement->setString(parameter++, mysqlTime(cursor->createdAt));
            statement->setString(parameter++, cursor->eventId);
        }
        statement->setUInt(parameter, static_cast<unsigned>(limit + 1));
        std::unique_ptr<sql::ResultSet> rows(statement->executeQuery()); domain::AuditPage page;
        while (rows->next()) page.items.push_back(readAudit(*rows));
        if (page.items.size() > limit)
        {
            page.items.resize(limit);
            page.nextCursor = domain::AuditCursor{page.items.back().createdAt,
                                                  page.items.back().eventId};
        }
        return page;
    });
}

bool MySqlTreeSemStore::bindSession(
    const std::string& session, const std::string& owner, const std::string& subject)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "UPDATE treesem_sessions SET owner_user_id=?,subject_user_id=? "
            "WHERE session_id=? AND (owner_user_id IS NULL OR owner_user_id=?)"));
        statement->setString(1, owner); statement->setString(2, subject);
        statement->setString(3, session); statement->setString(4, owner);
        return statement->executeUpdate() == 1;
    });
}

std::optional<std::string> MySqlTreeSemStore::sessionSubjectForActor(
    const std::string& session, const std::string& actor)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() -> std::optional<std::string> {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT subject_user_id FROM treesem_sessions "
            "WHERE session_id=? AND owner_user_id=?"));
        statement->setString(1, session); statement->setString(2, actor);
        std::unique_ptr<sql::ResultSet> row(statement->executeQuery());
        if (!row->next() || row->isNull("subject_user_id")) return std::nullopt;
        return stringValue(*row, "subject_user_id");
    });
}

bool MySqlTreeSemStore::isAgentRunRunning(
    const std::string& runId, const std::string& sessionId)
{
    auto lease = pool_.acquire();
    return withSqlTranslation(lease, [&]() {
        std::unique_ptr<sql::PreparedStatement> statement(lease.connection().prepareStatement(
            "SELECT COUNT(*) AS count FROM treesem_agent_runs "
            "WHERE run_id=? AND session_id=? AND status='running'"));
        statement->setString(1, runId); statement->setString(2, sessionId);
        std::unique_ptr<sql::ResultSet> row(statement->executeQuery());
        return row->next() && row->getInt("count") == 1;
    });
}

} // namespace infrastructure
} // namespace treesem
