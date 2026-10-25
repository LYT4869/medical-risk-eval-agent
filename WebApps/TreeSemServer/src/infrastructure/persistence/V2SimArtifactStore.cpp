#include "infrastructure/persistence/V2SimArtifactStore.h"

#include <memory>
#include <string>
#include <utility>

#include <cppconn/exception.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include <openssl/sha.h>

#include "application/BusinessException.h"

namespace treesem::infrastructure
{
namespace
{
using Error = application::BusinessException;
using Kind = application::BusinessException::Kind;
using Json = nlohmann::json;

[[noreturn]] void fail(Kind kind, const char* message)
{
    throw Error(kind, message);
}

std::string hash(const std::string& text)
{
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(text.data()), text.size(), digest);
    static const char alphabet[] = "0123456789abcdef";
    std::string result;
    result.reserve(SHA256_DIGEST_LENGTH * 2);
    for (unsigned char byte : digest)
    {
        result += alphabet[byte >> 4];
        result += alphabet[byte & 15];
    }
    return result;
}

template<typename Work>
auto guarded(MySqlConnectionPool::Lease& lease, Work&& work)
    -> decltype(work(lease.connection()))
{
    try { return work(lease.connection()); }
    catch (const Error&) { throw; }
    catch (const sql::SQLException& e)
    {
        if (e.getSQLState().rfind("08", 0) == 0)
            lease.invalidate();
        if (e.getErrorCode() == 1062)
            fail(Kind::Conflict, "conflicting V2 synthetic record");
        fail(Kind::PersistenceFailure, "V2 synthetic database operation failed");
    }
    catch (...)
    {
        fail(Kind::PersistenceFailure, "unexpected V2 database failure");
    }
}

template<typename Work>
Json transaction(sql::Connection& db, Work&& work)
{
    db.setAutoCommit(false);
    try
    {
        Json result = work();
        db.commit();
        db.setAutoCommit(true);
        return result;
    }
    catch (...)
    {
        try { db.rollback(); db.setAutoCommit(true); } catch (...) {}
        throw;
    }
}

std::unique_ptr<sql::PreparedStatement> query(
    sql::Connection& db, const std::string& statement)
{
    return std::unique_ptr<sql::PreparedStatement>(db.prepareStatement(statement));
}

std::string required(sql::ResultSet& row, const std::string& key)
{
    return row.getString(key).asStdString();
}

void checkOwnership(sql::ResultSet& row, const std::string& doctorId,
                    const std::string& patientId)
{
    if (required(row, "doctor_user_id") != doctorId ||
        required(row, "patient_user_id") != patientId ||
        required(row, "classification") != "SIM")
    {
        fail(Kind::NotFound, "synthetic run not accessible");
    }
}
} // namespace

V2SimArtifactStore::V2SimArtifactStore(MySqlConnectionConfig config)
    : pool_(std::move(config))
{
    auto lease = pool_.acquire();
    guarded(lease, [](sql::Connection& db) {
        auto st = query(db, "SELECT COUNT(*) AS n FROM schema_migrations WHERE version=?");
        st->setString(1, "008_v2_contract_versions");
        auto result = std::unique_ptr<sql::ResultSet>(st->executeQuery());
        if (!result->next() || result->getInt("n") != 1)
            fail(Kind::PersistenceFailure, "V2 SIM migrations 006/007/008 missing");
        return 0;
    });
}

Json V2SimArtifactStore::create(const std::string& runId,
                                 const std::string& doctorId,
                                 const std::string& patientId,
                                 const std::string& idemKey,
                                 const std::string& summary,
                                 const std::string& overview,
                                 const std::string& question)
{
    if (summary.empty() || overview.empty() || summary.size() > 1200 ||
        overview.size() > 1200 || question.size() > 1200)
        fail(Kind::InvalidInput, "invalid SIM Task Contract fields");
    const auto fingerprint = hash(Json::array({summary, overview, question}).dump());
    const auto legacy = hash(Json::array({summary, overview}).dump());
    auto lease = pool_.acquire();
    return guarded(lease, [&](sql::Connection& db) {
        return transaction(db, [&]() -> Json {
            auto insert = query(db,
                "INSERT IGNORE INTO treesem_v2_runs "
                "(run_id,doctor_user_id,patient_user_id,idempotency_key,"
                "request_sha256,contract_revision,business_revision,"
                "contract_summary,investigation_overview,status,classification) "
                "VALUES (?,?,?,?,?,1,0,?,?,'awaiting_confirmation','SIM')");
            insert->setString(1, runId);
            insert->setString(2, doctorId);
            insert->setString(3, patientId);
            insert->setString(4, idemKey);
            insert->setString(5, fingerprint);
            insert->setString(6, summary);
            insert->setString(7, overview);
            const bool created = insert->executeUpdate() == 1;
            auto read = query(db,
                "SELECT run_id,request_sha256,status,contract_revision "
                "FROM treesem_v2_runs WHERE doctor_user_id=? AND patient_user_id=? "
                "AND idempotency_key=?");
            read->setString(1, doctorId);
            read->setString(2, patientId);
            read->setString(3, idemKey);
            auto row = std::unique_ptr<sql::ResultSet>(read->executeQuery());
            if (!row->next()) fail(Kind::PersistenceFailure, "failed to create V2 Run");
            const std::string foundHash = required(*row, "request_sha256");
            if (foundHash != fingerprint && !(question.empty() && foundHash == legacy))
                fail(Kind::IdempotencyConflict, "run key reused for a different Contract");
            const std::string persistentId = required(*row, "run_id");
            const int contractRev = row->getInt("contract_revision");
            const std::string status = required(*row, "status");
            if (created)
            {
                auto draft = query(db,
                    "INSERT INTO treesem_v2_contract_versions "
                    "(run_id,contract_revision,summary,investigation_overview,"
                    "clarification_question,clarification_answer) "
                    "VALUES (?,1,?,?,NULLIF(?,''),NULL)");
                draft->setString(1, persistentId);
                draft->setString(2, summary);
                draft->setString(3, overview);
                draft->setString(4, question);
                draft->executeUpdate();
            }
            return Json{{"run_id", persistentId},
                        {"status", status}, {"contract_revision", contractRev},
                        {"classification", "SIM"}, {"fixture_id", "P001"}};
        });
    });
}

Json V2SimArtifactStore::status(const std::string& runId,
                                 const std::string& doctorId,
                                 const std::string& patientId)
{
    auto lease = pool_.acquire();
    return guarded(lease, [&](sql::Connection& db) -> Json {
        auto read = query(db,
            "SELECT r.run_id,r.doctor_user_id,r.patient_user_id,r.classification,"
            "r.fixture_id,r.status,r.contract_revision,r.business_revision,"
            "v.summary,v.investigation_overview,v.clarification_question,"
            "v.clarification_answer FROM treesem_v2_runs r "
            "JOIN treesem_v2_contract_versions v ON "
            "v.run_id=r.run_id AND v.contract_revision=r.contract_revision "
            "WHERE r.run_id=?");
        read->setString(1, runId);
        auto row = std::unique_ptr<sql::ResultSet>(read->executeQuery());
        if (!row->next()) fail(Kind::NotFound, "unknown V2 Run/Contract");
        checkOwnership(*row, doctorId, patientId);
        const auto question = row->isNull("clarification_question")
            ? "" : required(*row, "clarification_question");
        const auto answer = row->isNull("clarification_answer")
            ? "" : required(*row, "clarification_answer");
        return Json{{"run_id", runId},
                    {"status", required(*row, "status")},
                    {"contract_revision", row->getInt("contract_revision")},
                    {"revision", row->getInt("business_revision")},
                    {"summary", required(*row, "summary")},
                    {"overview", required(*row, "investigation_overview")},
                    {"clarification_question", question},
                    {"clarification_answer", answer},
                    {"classification", "SIM"},
                    {"fixture_id", required(*row, "fixture_id")}};
    });
}

Json V2SimArtifactStore::confirm(const std::string& runId,
                                  const std::string& doctorId,
                                  const std::string& patientId,
                                  int expectedRevision,
                                  int contractRevision)
{
    if (expectedRevision < 0 || contractRevision < 1)
        fail(Kind::InvalidInput, "invalid Contract confirmation version");
    const std::string opId = "op_" +
        hash(runId + "|contract|" + std::to_string(expectedRevision)).substr(0, 32);
    const std::string fingerprint = hash(Json::array({
        runId, doctorId, patientId, "confirm", expectedRevision, contractRevision,
    }).dump());
    auto lease = pool_.acquire();
    return guarded(lease, [&](sql::Connection& db) {
        return transaction(db, [&]() -> Json {
            auto lock = query(db, "SELECT doctor_user_id,patient_user_id,"
                "classification,status,business_revision,contract_revision "
                "FROM treesem_v2_runs WHERE run_id=? FOR UPDATE");
            lock->setString(1, runId);
            auto row = std::unique_ptr<sql::ResultSet>(lock->executeQuery());
            if (!row->next()) fail(Kind::NotFound, "unknown V2 Run");
            checkOwnership(*row, doctorId, patientId);
            auto old = query(db, "SELECT request_sha256,result_json "
                "FROM treesem_v2_operations WHERE run_id=? AND operation_id=?");
            old->setString(1, runId);
            old->setString(2, opId);
            auto previous = std::unique_ptr<sql::ResultSet>(old->executeQuery());
            if (previous->next())
            {
                if (required(*previous, "request_sha256") != fingerprint)
                    fail(Kind::IdempotencyConflict, "Contract operation replay changed");
                return Json::parse(required(*previous, "result_json"));
            }
            const int version = row->getInt("contract_revision");
            const int revision = row->getInt("business_revision");
            const std::string status = required(*row, "status");
            if (status != "awaiting_confirmation" ||
                revision != expectedRevision || version != contractRevision)
                fail(Kind::Conflict, "stale Contract confirmation");
            auto draft = query(db, "SELECT clarification_question "
                "FROM treesem_v2_contract_versions "
                "WHERE run_id=? AND contract_revision=?");
            draft->setString(1, runId);
            draft->setInt(2, contractRevision);
            auto current = std::unique_ptr<sql::ResultSet>(draft->executeQuery());
            if (!current->next()) fail(Kind::Conflict, "missing versioned Contract");
            if (!current->isNull("clarification_question"))
                fail(Kind::Conflict, "unanswered clarification must be resolved");
            auto update = query(db, "UPDATE treesem_v2_runs "
                "SET status='contract_confirmed',business_revision=? "
                "WHERE run_id=? AND business_revision=? AND "
                "contract_revision=? AND status='awaiting_confirmation'");
            update->setInt(1, expectedRevision + 1);
            update->setString(2, runId);
            update->setInt(3, expectedRevision);
            update->setInt(4, contractRevision);
            if (update->executeUpdate() != 1) fail(Kind::Conflict, "Run changed");
            const Json result{{"run_id", runId}, {"status", "contract_confirmed"},
                              {"revision", expectedRevision + 1},
                              {"contract_revision", contractRevision}};
            auto journal = query(db, "INSERT INTO treesem_v2_operations "
                "(run_id,operation_id,phase,request_sha256,result_json) "
                "VALUES (?,?,'contract',?,?)");
            journal->setString(1, runId);
            journal->setString(2, opId);
            journal->setString(3, fingerprint);
            journal->setString(4, result.dump());
            journal->executeUpdate();
            return result;
        });
    });
}

Json V2SimArtifactStore::editContract(
    const std::string& runId, const std::string& doctorId,
    const std::string& patientId, const std::string& decision,
    int expectedRevision, int contractRevision, const std::string& summary,
    const std::string& overview, const std::string& answer)
{
    if (expectedRevision < 0 || contractRevision < 1 || contractRevision >= 10 ||
        (decision != "revise" && decision != "clarify" && decision != "cancel"))
        fail(Kind::InvalidInput, "invalid Contract edit/clarification");
    if (decision == "revise" &&
        (summary.empty() || overview.empty() ||
         summary.size() > 1200 || overview.size() > 1200 || !answer.empty()))
        fail(Kind::InvalidInput, "invalid revised Contract goal or outline");
    if (decision == "clarify" &&
        (answer.empty() || answer.size() > 1200 ||
         !summary.empty() || !overview.empty()))
        fail(Kind::InvalidInput, "invalid clarification answer");
    if (decision == "cancel" &&
        (!summary.empty() || !overview.empty() || !answer.empty()))
        fail(Kind::InvalidInput, "cancel must not carry an edit payload");

    const std::string opId = "op_" +
        hash(runId + "|contract|" + std::to_string(expectedRevision)).substr(0, 32);
    const std::string fingerprint = hash(Json::array({
        runId, doctorId, patientId, decision, expectedRevision, contractRevision,
        summary, overview, answer,
    }).dump());
    auto lease = pool_.acquire();
    return guarded(lease, [&](sql::Connection& db) {
        return transaction(db, [&]() -> Json {
            auto lock = query(db,
                "SELECT doctor_user_id,patient_user_id,classification,"
                "status,business_revision,contract_revision "
                "FROM treesem_v2_runs WHERE run_id=? FOR UPDATE");
            lock->setString(1, runId);
            auto row = std::unique_ptr<sql::ResultSet>(lock->executeQuery());
            if (!row->next()) fail(Kind::NotFound, "unknown V2 Run");
            checkOwnership(*row, doctorId, patientId);
            auto previous = query(db, "SELECT request_sha256,result_json "
                "FROM treesem_v2_operations WHERE run_id=? AND operation_id=?");
            previous->setString(1, runId);
            previous->setString(2, opId);
            auto recorded = std::unique_ptr<sql::ResultSet>(previous->executeQuery());
            if (recorded->next())
            {
                if (required(*recorded, "request_sha256") != fingerprint)
                    fail(Kind::IdempotencyConflict, "Contract operation changed");
                return Json::parse(required(*recorded, "result_json"));
            }
            if (required(*row, "status") != "awaiting_confirmation" ||
                row->getInt("business_revision") != expectedRevision ||
                row->getInt("contract_revision") != contractRevision)
                fail(Kind::Conflict, "stale Contract edit or finished Run");
            if (decision == "cancel")
            {
                auto update = query(db,
                    "UPDATE treesem_v2_runs SET status='cancelled',"
                    "business_revision=? WHERE run_id=? "
                    "AND status='awaiting_confirmation' AND "
                    "contract_revision=? AND business_revision=?");
                update->setInt(1, expectedRevision + 1);
                update->setString(2, runId);
                update->setInt(3, contractRevision);
                update->setInt(4, expectedRevision);
                if (update->executeUpdate() != 1)
                    fail(Kind::Conflict, "Run changed during cancellation");
                const Json result{{"run_id", runId},
                    {"status", "cancelled"}, {"revision", expectedRevision + 1},
                    {"contract_revision", contractRevision}};
                auto journal = query(db, "INSERT INTO treesem_v2_operations "
                    "(run_id,operation_id,phase,request_sha256,result_json) "
                    "VALUES (?,?,'contract',?,?)");
                journal->setString(1, runId);
                journal->setString(2, opId);
                journal->setString(3, fingerprint);
                journal->setString(4, result.dump());
                journal->executeUpdate();
                return result;
            }
            auto read = query(db, "SELECT summary,investigation_overview,"
                "clarification_question,clarification_answer FROM "
                "treesem_v2_contract_versions WHERE run_id=? "
                "AND contract_revision=?");
            read->setString(1, runId);
            read->setInt(2, contractRevision);
            auto current = std::unique_ptr<sql::ResultSet>(read->executeQuery());
            if (!current->next()) fail(Kind::Conflict, "Contract version missing");
            const std::string question = current->isNull("clarification_question")
                ? "" : required(*current, "clarification_question");
            const std::string priorAnswer = current->isNull("clarification_answer")
                ? "" : required(*current, "clarification_answer");
            if (decision == "clarify" && question.empty())
                fail(Kind::Conflict, "no clarification is pending");
            const std::string nextSummary = decision == "revise" ? summary :
                required(*current, "summary");
            const std::string nextOverview = decision == "revise" ? overview :
                required(*current, "investigation_overview");
            const std::string nextQuestion = decision == "revise" ? question : "";
            const std::string nextAnswer = decision == "revise" ? priorAnswer : answer;

            auto save = query(db, "INSERT INTO treesem_v2_contract_versions "
                "(run_id,contract_revision,summary,investigation_overview,"
                "clarification_question,clarification_answer) "
                "VALUES (?,?,?,?,NULLIF(?,''),NULLIF(?,''))");
            save->setString(1, runId);
            save->setInt(2, contractRevision + 1);
            save->setString(3, nextSummary);
            save->setString(4, nextOverview);
            save->setString(5, nextQuestion);
            save->setString(6, nextAnswer);
            save->executeUpdate();
            auto update = query(db, "UPDATE treesem_v2_runs SET "
                "contract_revision=?,business_revision=?,contract_summary=?,"
                "investigation_overview=? WHERE run_id=? AND "
                "status='awaiting_confirmation' AND "
                "contract_revision=? AND business_revision=?");
            update->setInt(1, contractRevision + 1);
            update->setInt(2, expectedRevision + 1);
            update->setString(3, nextSummary);
            update->setString(4, nextOverview);
            update->setString(5, runId);
            update->setInt(6, contractRevision);
            update->setInt(7, expectedRevision);
            if (update->executeUpdate() != 1) fail(Kind::Conflict, "Run changed");
            const Json result{{"run_id", runId},
                              {"status", "awaiting_confirmation"},
                              {"revision", expectedRevision + 1},
                              {"contract_revision", contractRevision + 1}};
            auto journal = query(db, "INSERT INTO treesem_v2_operations "
                "(run_id,operation_id,phase,request_sha256,result_json) "
                "VALUES (?,?,'contract',?,?)");
            journal->setString(1, runId);
            journal->setString(2, opId);
            journal->setString(3, fingerprint);
            journal->setString(4, result.dump());
            journal->executeUpdate();
            return result;
        });
    });
}

Json V2SimArtifactStore::put(const std::string& runId,
                              const std::string& doctorId,
                              const std::string& patientId,
                              const std::string& operationId,
                              const std::string& kind,
                              const std::string& artifactKey,
                              const Json& payload)
{
    if (kind != "evidence" && kind != "issue" && kind != "draft")
        fail(Kind::InvalidInput, "invalid artifact kind");
    if (!payload.is_object() || payload.value("classification", "") != "SIM")
        fail(Kind::InvalidInput, "only synthetic artifacts are allowed");
    if (payload.contains("run_id") &&
        (!payload.at("run_id").is_string() ||
         payload.at("run_id").get<std::string>() != runId))
        fail(Kind::Conflict, "artifact references a different Run");
    if (kind == "issue" && payload.value("no_diagnosis", false) != true)
        fail(Kind::InvalidInput, "M2 SIM issue cannot assert a diagnosis");
    if (kind == "draft" &&
        (payload.value("report_id", "") != "report_" + runId ||
         payload.value("draft_revision", 0) != 1 ||
         payload.value("status", "") != "restricted_draft" ||
         payload.value("not_signed", false) != true ||
         payload.value("template_only", false) != true))
        fail(Kind::InvalidInput, "only unsigned P001 template Draft v1 allowed");
    const std::string serialized = payload.dump();
    const std::string contentHash = hash(serialized);
    const std::string requestHash = hash(Json::array(
        {runId,kind,artifactKey,contentHash}).dump());
    auto lease = pool_.acquire();
    return guarded(lease, [&](sql::Connection& db) {
        return transaction(db, [&]() -> Json {
            auto lock = query(db,
                "SELECT doctor_user_id,patient_user_id,classification,status "
                "FROM treesem_v2_runs WHERE run_id=? FOR UPDATE");
            lock->setString(1, runId);
            auto run = std::unique_ptr<sql::ResultSet>(lock->executeQuery());
            if (!run->next()) fail(Kind::NotFound, "unknown V2 Run");
            checkOwnership(*run, doctorId, patientId);
            auto old = query(db,
                "SELECT request_sha256,result_json FROM treesem_v2_operations "
                "WHERE run_id=? AND operation_id=?");
            old->setString(1, runId); old->setString(2, operationId);
            auto prior = std::unique_ptr<sql::ResultSet>(old->executeQuery());
            if (prior->next())
            {
                if (required(*prior, "request_sha256") != requestHash)
                    fail(Kind::IdempotencyConflict, "operation changed on replay");
                return Json::parse(required(*prior, "result_json"));
            }
            if (required(*run, "status") != "contract_confirmed")
                fail(Kind::Conflict, "Run is not open for artifacts");
            if (kind != "evidence")
            {
                const std::string requiredKind = kind == "issue" ? "evidence" : "issue";
                auto count = query(db,
                    "SELECT COUNT(*) AS n FROM treesem_v2_artifacts "
                    "WHERE run_id=? AND kind=?");
                count->setString(1, runId); count->setString(2, requiredKind);
                auto found = std::unique_ptr<sql::ResultSet>(count->executeQuery());
                if (!found->next() || found->getInt("n") < 1)
                    fail(Kind::Conflict, "required previous artifact is absent");
            }
            auto existing = query(db,
                "SELECT payload_sha256 FROM treesem_v2_artifacts "
                "WHERE run_id=? AND kind=? AND artifact_key=?");
            existing->setString(1, runId);
            existing->setString(2, kind);
            existing->setString(3, artifactKey);
            auto found = std::unique_ptr<sql::ResultSet>(existing->executeQuery());
            if (found->next())
            {
                if (required(*found, "payload_sha256") != contentHash)
                    fail(Kind::Conflict, "immutable artifact was changed");
            }
            else
            {
                auto insert = query(db,
                    "INSERT INTO treesem_v2_artifacts "
                    "(run_id,kind,artifact_key,payload_json,payload_sha256) "
                    "VALUES (?,?,?,?,?)");
                insert->setString(1, runId);
                insert->setString(2, kind);
                insert->setString(3, artifactKey);
                insert->setString(4, serialized);
                insert->setString(5, contentHash);
                insert->executeUpdate();
            }
            const Json result{{"run_id",runId}, {"kind",kind},
                              {"artifact_key",artifactKey}, {"classification","SIM"}};
            auto journal = query(db,
                "INSERT INTO treesem_v2_operations "
                "(run_id,operation_id,phase,request_sha256,result_json) VALUES (?,?,?,?,?)");
            journal->setString(1, runId);
            journal->setString(2, operationId);
            journal->setString(3, kind);
            journal->setString(4, requestHash);
            journal->setString(5, result.dump());
            journal->executeUpdate();
            if (kind == "draft")
            {
                auto update = query(db,
                    "UPDATE treesem_v2_runs SET status='draft_ready' WHERE run_id=?");
                update->setString(1, runId);
                update->executeUpdate();
            }
            return result;
        });
    });
}

Json V2SimArtifactStore::get(const std::string& runId,
                              const std::string& doctorId,
                              const std::string& patientId,
                              const std::string& kind,
                              const std::string& artifactKey)
{
    auto lease = pool_.acquire();
    return guarded(lease, [&](sql::Connection& db) -> Json {
        auto owner = query(db,
            "SELECT doctor_user_id,patient_user_id,classification,status "
            "FROM treesem_v2_runs WHERE run_id=?");
        owner->setString(1, runId);
        auto run = std::unique_ptr<sql::ResultSet>(owner->executeQuery());
        if (!run->next()) fail(Kind::NotFound, "unknown V2 Run");
        checkOwnership(*run, doctorId, patientId);
        auto item = query(db,
            "SELECT payload_json FROM treesem_v2_artifacts WHERE "
            "run_id=? AND kind=? AND artifact_key=?");
        item->setString(1, runId); item->setString(2, kind);
        item->setString(3, artifactKey);
        auto record = std::unique_ptr<sql::ResultSet>(item->executeQuery());
        if (!record->next()) fail(Kind::NotFound, "unknown V2 artifact");
        return Json::parse(required(*record, "payload_json"));
    });
}
} // namespace treesem::infrastructure
