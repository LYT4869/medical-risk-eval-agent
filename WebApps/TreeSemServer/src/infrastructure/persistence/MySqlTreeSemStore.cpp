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
        statement->setString(1, "001_m4_core");
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
                    "tree_leaf_id,result_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"));
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
                "SELECT prediction_id,session_id,result_json,created_at "
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
            insert->setBoolean(5, false);
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

} // namespace infrastructure
} // namespace treesem
