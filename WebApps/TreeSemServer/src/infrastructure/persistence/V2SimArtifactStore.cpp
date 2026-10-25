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
        st->setString(1, "006_v2_sim_artifacts");
        auto result = std::unique_ptr<sql::ResultSet>(st->executeQuery());
        if (!result->next() || result->getInt("n") != 1)
            fail(Kind::PersistenceFailure, "V2 SIM migration 006 missing");
        return 0;
    });
}

Json V2SimArtifactStore::create(const std::string& runId,
                                 const std::string& doctorId,
                                 const std::string& patientId,
                                 const std::string& idemKey,
                                 const std::string& summary,
                                 const std::string& overview)
{
    const auto fingerprint = hash(Json::array({summary, overview}).dump());
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
            insert->executeUpdate();
            auto read = query(db, "SELECT run_id,request_sha256,status,contract_revision "
                "FROM treesem_v2_runs WHERE doctor_user_id=? AND patient_user_id=? "
                "AND idempotency_key=?");
            read->setString(1, doctorId);
            read->setString(2, patientId);
            read->setString(3, idemKey);
            auto row = std::unique_ptr<sql::ResultSet>(read->executeQuery());
            if (!row->next()) fail(Kind::PersistenceFailure, "failed to create V2 Run");
            if (required(*row, "request_sha256") != fingerprint)
                fail(Kind::IdempotencyConflict, "run key reused for a different request");
            return Json{{"run_id", required(*row, "run_id")},
                        {"status", required(*row, "status")},
                        {"contract_revision", row->getInt("contract_revision")},
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
            "SELECT run_id,doctor_user_id,patient_user_id,classification,fixture_id,status,"
            "contract_revision,business_revision,contract_summary,investigation_overview "
            "FROM treesem_v2_runs WHERE run_id=?");
        read->setString(1, runId);
        auto row = std::unique_ptr<sql::ResultSet>(read->executeQuery());
        if (!row->next()) fail(Kind::NotFound, "unknown V2 Run");
        checkOwnership(*row, doctorId, patientId);
        return Json{{"run_id", runId},
                    {"status", required(*row, "status")},
                    {"contract_revision", row->getInt("contract_revision")},
                    {"revision", row->getInt("business_revision")},
                    {"summary", required(*row, "contract_summary")},
                    {"overview", required(*row, "investigation_overview")},
                    {"classification", "SIM"},
                    {"fixture_id", required(*row, "fixture_id")}};
    });
}

Json V2SimArtifactStore::confirm(const std::string& runId,
                                  const std::string& doctorId,
                                  const std::string& patientId,
                                  int expectedRevision)
{
    auto lease = pool_.acquire();
    return guarded(lease, [&](sql::Connection& db) {
        return transaction(db, [&]() -> Json {
            auto read = query(db,
                "SELECT doctor_user_id,patient_user_id,classification,status,"
                "business_revision,contract_revision FROM treesem_v2_runs "
                "WHERE run_id=? FOR UPDATE");
            read->setString(1, runId);
            auto row = std::unique_ptr<sql::ResultSet>(read->executeQuery());
            if (!row->next()) fail(Kind::NotFound, "unknown V2 Run");
            checkOwnership(*row, doctorId, patientId);
            const std::string status = required(*row, "status");
            const int revision = row->getInt("business_revision");
            if (status == "contract_confirmed" && expectedRevision == 0 &&
                revision == 1)
                return Json{{"run_id",runId}, {"status",status}, {"revision",revision}};
            if (status != "awaiting_confirmation" || revision != expectedRevision ||
                expectedRevision != 0)
                fail(Kind::Conflict, "Contract requires an explicit current confirmation");
            auto update = query(db,
                "UPDATE treesem_v2_runs SET status='contract_confirmed',"
                "business_revision=1 WHERE run_id=? AND business_revision=0 "
                "AND status='awaiting_confirmation'");
            update->setString(1, runId);
            if (update->executeUpdate() != 1) fail(Kind::Conflict, "Run changed");
            return Json{{"run_id",runId}, {"status","contract_confirmed"},
                        {"revision",1}};
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
