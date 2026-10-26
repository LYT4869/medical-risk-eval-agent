#include "infrastructure/persistence/V2SimArtifactStore.h"

#include <memory>
#include <set>
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
using Json = nlohmann::json;
using Error = application::BusinessException;
using Kind = Error::Kind;

constexpr const char* kLogical = "LIS-P001-HB-20261010";

[[noreturn]] void fail(Kind kind, const char* description)
{
    throw Error(kind, description);
}

std::string digest(const std::string& data)
{
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash);
    static const char hex[] = "0123456789abcdef";
    std::string value;
    for (const unsigned char b : hash)
    {
        value += hex[b >> 4];
        value += hex[b & 0xf];
    }
    return value;
}

auto stmt(sql::Connection& db, const std::string& text)
{
    return std::unique_ptr<sql::PreparedStatement>(db.prepareStatement(text));
}

auto records(sql::PreparedStatement& prepared)
{
    return std::unique_ptr<sql::ResultSet>(prepared.executeQuery());
}

std::string field(sql::ResultSet& result, const char* name)
{
    return result.getString(name).asStdString();
}

template <typename Work>
Json withConnection(MySqlConnectionPool::Lease& lease, Work work)
{
    try { return work(lease.connection()); }
    catch (const Error&) { throw; }
    catch (const sql::SQLException& error)
    {
        if (error.getSQLState().rfind("08", 0) == 0)
            lease.invalidate();
        if (error.getErrorCode() == 1062)
            fail(Kind::Conflict, "synthetic correction unique key conflict");
        fail(Kind::PersistenceFailure, "synthetic LIS correction SQL error");
    }
    catch (...) { fail(Kind::PersistenceFailure, "synthetic LIS correction failed"); }
}

template <typename Work>
Json transaction(sql::Connection& db, Work work)
{
    db.setAutoCommit(false);
    try
    {
        Json response = work();
        db.commit();
        db.setAutoCommit(true);
        return response;
    }
    catch (...)
    {
        try { db.rollback(); db.setAutoCommit(true); } catch (...) {}
        throw;
    }
}

void owns(sql::Connection& db, const std::string& run,
          const std::string& doctor, const std::string& patient,
          bool lock)
{
    auto query = stmt(db,
        std::string("SELECT doctor_user_id,patient_user_id,classification,"
                    "fixture_id,status FROM treesem_v2_runs WHERE run_id=?") +
        (lock ? " FOR UPDATE" : ""));
    query->setString(1, run);
    auto result = records(*query);
    if (!result->next())
        fail(Kind::NotFound, "unknown synthetic Run");
    if (field(*result, "doctor_user_id") != doctor ||
        field(*result, "patient_user_id") != patient ||
        field(*result, "classification") != "SIM" ||
        field(*result, "fixture_id") != "P001")
        fail(Kind::NotFound, "synthetic Run not accessible");
    const std::string status = field(*result, "status");
    if (status != "contract_confirmed" && status != "draft_ready")
        fail(Kind::Conflict, "confirmed Contract required for M4");
}

Json eventReplay(sql::Connection& db, const std::string& run,
                 const std::string& id, const std::string& fingerprint)
{
    auto check = stmt(db, "SELECT request_sha256,result_json FROM "
        "treesem_v2_m4_correction_events WHERE run_id=? AND event_id=?");
    check->setString(1, run);
    check->setString(2, id);
    auto old = records(*check);
    if (!old->next()) return nullptr;
    if (field(*old, "request_sha256") != fingerprint)
        fail(Kind::IdempotencyConflict, "correction event replay payload changed");
    return Json::parse(field(*old, "result_json"));
}

void saveEvent(sql::Connection& db, const std::string& run,
               const std::string& eventId, const std::string& kind,
               const std::string& fingerprint, const Json& output)
{
    auto command = stmt(db, "INSERT INTO treesem_v2_m4_correction_events "
        "(run_id,event_id,event_kind,request_sha256,result_json) "
        "VALUES (?,?,?,?,?)");
    command->setString(1, run);
    command->setString(2, eventId);
    command->setString(3, kind);
    command->setString(4, fingerprint);
    command->setString(5, output.dump());
    command->executeUpdate();
}

void saveSource(sql::Connection& db, const std::string& run,
                const std::string& source, int version, int hb)
{
    auto query = stmt(db, "INSERT INTO treesem_v2_m4_source_versions "
        "(run_id,evidence_id,logical_id,source_version,hb_g_l,source_kind) "
        "VALUES (?,?,?,?,?,'LIS_SIM')");
    query->setString(1, run);
    query->setString(2, source);
    query->setString(3, kLogical);
    query->setInt(4, version);
    query->setInt(5, hb);
    query->executeUpdate();
}

void saveValidity(sql::Connection& db, const std::string& run,
                  const std::string& kind, const std::string& ref,
                  const std::string& status, const std::string& event)
{
    auto query = stmt(db, "INSERT INTO treesem_v2_m4_validity "
        "(run_id,object_kind,object_id,status,cause_event_id) "
        "VALUES (?,?,?,?,NULLIF(?,'')) "
        "ON DUPLICATE KEY UPDATE status=VALUES(status),"
        "cause_event_id=VALUES(cause_event_id)");
    query->setString(1, run);
    query->setString(2, kind);
    query->setString(3, ref);
    query->setString(4, status);
    query->setString(5, event);
    query->executeUpdate();
}

std::string sourceValidity(sql::Connection& db, const std::string& run,
                           const std::string& reference)
{
    auto query = stmt(db, "SELECT status FROM treesem_v2_m4_validity "
        "WHERE run_id=? AND object_kind='evidence' AND object_id=?");
    query->setString(1, run);
    query->setString(2, reference);
    auto row = records(*query);
    return row->next() ? field(*row, "status") : "missing";
}
} // namespace

Json V2SimArtifactStore::stageHistoricalLab(
    const std::string& runId, const std::string& doctorId,
    const std::string& patientId)
{
    const std::string fingerprint = digest(Json::array({
        runId, doctorId, patientId, "M4-HISTORY", kLogical, "E-004", 1, 110
    }).dump());
    auto lease = pool_.acquire();
    return withConnection(lease, [&](sql::Connection& db) {
        return transaction(db, [&]() -> Json {
            owns(db, runId, doctorId, patientId, true);
            const Json replay = eventReplay(db, runId, "M4-HISTORY", fingerprint);
            if (!replay.is_null()) return replay;
            // M4's S-04 is a historical *synthetic dependent statement*, not a
            // claim that the prior M3 Draft contained an Hb statement.
            saveSource(db, runId, "E-004", 1, 110);
            saveValidity(db, runId, "evidence", "E-004", "current", "");
            auto statement = stmt(db,
                "INSERT INTO treesem_v2_m4_statement_snapshots "
                "(run_id,statement_id,evidence_id,value_g_l) VALUES (?,'S-04','E-004',110)");
            statement->setString(1, runId);
            statement->executeUpdate();
            saveValidity(db, runId, "statement", "S-04", "current", "");
            const Json result{
                {"run_id",runId}, {"event_id","M4-HISTORY"},
                {"source_id","E-004"}, {"source_version",1},
                {"statement_id","S-04"}, {"status","historical_seeded"},
                {"classification","SIM"}
            };
            saveEvent(db, runId, "M4-HISTORY", "historical_seed",
                      fingerprint, result);
            return result;
        });
    });
}

Json V2SimArtifactStore::applyOfficialSimCorrection(
    const std::string& runId, const std::string& doctorId,
    const std::string& patientId)
{
    // These values are compiled into the *server-side fixture policy*.
    // No caller may supply "trusted source", a free-form event or Hb value.
    const std::string fingerprint = digest(Json::array({
        runId, doctorId, patientId, "EV-001", "LIS", kLogical,
        "E-004", 1, 110, "E-005", 2, 130, "2026-10-13"
    }).dump());
    auto lease = pool_.acquire();
    return withConnection(lease, [&](sql::Connection& db) {
        return transaction(db, [&]() -> Json {
            owns(db, runId, doctorId, patientId, true);
            const Json replay = eventReplay(db, runId, "EV-001", fingerprint);
            if (!replay.is_null()) return replay;
            auto history = stmt(db, "SELECT logical_id,source_version,hb_g_l "
                "FROM treesem_v2_m4_source_versions "
                "WHERE run_id=? AND evidence_id='E-004'");
            history->setString(1, runId);
            auto old = records(*history);
            if (!old->next() || field(*old, "logical_id") != kLogical ||
                old->getInt("source_version") != 1 ||
                old->getInt("hb_g_l") != 110 ||
                sourceValidity(db, runId, "E-004") != "current")
                fail(Kind::Conflict, "unverified old LIS version");
            auto previous = stmt(db, "SELECT evidence_id,value_g_l "
                "FROM treesem_v2_m4_statement_snapshots "
                "WHERE run_id=? AND statement_id='S-04'");
            previous->setString(1, runId);
            auto statement = records(*previous);
            if (!statement->next() || field(*statement, "evidence_id") != "E-004" ||
                statement->getInt("value_g_l") != 110)
                fail(Kind::Conflict, "missing exact historical statement");
            saveSource(db, runId, "E-005", 2, 130);
            // The new statement is an immutable *new* source-backed
            // observation, not a rewrite of the historical S-04.
            auto newStatement = stmt(db,
                "INSERT INTO treesem_v2_m4_statement_snapshots "
                "(run_id,statement_id,evidence_id,value_g_l) "
                "VALUES (?,'S-05','E-005',130)");
            newStatement->setString(1, runId);
            newStatement->executeUpdate();
            saveValidity(db, runId, "statement", "S-05", "current", "");
            saveValidity(db, runId, "evidence", "E-004", "stale", "EV-001");
            saveValidity(db, runId, "statement", "S-04", "stale", "EV-001");
            saveValidity(db, runId, "evidence", "E-005", "current", "");
            const Json result{
                {"run_id",runId}, {"event_id","EV-001"},
                {"event_kind","official_correction_sim"},
                {"trusted_source","LIS_SIM_FIXTURE"},
                {"logical_id",kLogical}, {"previous","E-004"},
                {"current","E-005"}, {"previous_hb_g_l",110},
                {"current_hb_g_l",130}, {"affected_statements",Json::array({"S-04"})},
                {"new_statement_id","S-05"},
                {"new_issue_id","I-05"}, {"issue_status","needs_review"},
                {"classification","SIM"}, {"effective_on","2026-10-13"}
            };
            // Event and its dependent I-05 are in one InnoDB transaction.
            saveEvent(db, runId, "EV-001", "official_correction_sim",
                      fingerprint, result);
            auto followup = stmt(db, "INSERT INTO treesem_v2_m4_followups "
                "(run_id,issue_id,cause_event_id,old_evidence_id,new_evidence_id,status) "
                "VALUES (?,'I-05','EV-001','E-004','E-005','needs_review')");
            followup->setString(1, runId);
            followup->executeUpdate();
            return result;
        });
    });
}

Json V2SimArtifactStore::applyM6LateCorrection(
    const std::string& runId, const std::string& doctorId,
    const std::string& patientId)
{
    // Fixed doctor-triggered SIM-only late event; not a fake free-form LIS
    // webhook. E-005 becomes historical and E-006 is a new source snapshot.
    const std::string fingerprint = digest(Json::array({
        runId,doctorId,patientId,"EV-002","LIS",kLogical,
        "E-005",2,130,"E-006",3,125,"2026-10-18"
    }).dump());
    auto lease=pool_.acquire();
    return withConnection(lease,[&](sql::Connection& db) {
        return transaction(db,[&]() -> Json {
            owns(db,runId,doctorId,patientId,true);
            const Json replay=eventReplay(db,runId,"EV-002",fingerprint);
            if(!replay.is_null()) return replay;
            auto previous=stmt(db,"SELECT logical_id,source_version,hb_g_l "
                "FROM treesem_v2_m4_source_versions WHERE run_id=? "
                "AND evidence_id='E-005'");
            previous->setString(1,runId);
            auto source=records(*previous);
            if(!source->next() || field(*source,"logical_id")!=kLogical ||
               source->getInt("source_version")!=2 ||
               source->getInt("hb_g_l")!=130 ||
               sourceValidity(db,runId,"E-005")!="current")
                fail(Kind::Conflict,"cannot apply unverified late LIS amendment");
            auto statement=stmt(db,"SELECT evidence_id,value_g_l "
                "FROM treesem_v2_m4_statement_snapshots WHERE run_id=? "
                "AND statement_id='S-05'");
            statement->setString(1,runId);
            auto old=records(*statement);
            if(!old->next() || field(*old,"evidence_id")!="E-005" ||
               old->getInt("value_g_l")!=130)
                fail(Kind::Conflict,"current LIS statement snapshot missing");
            saveSource(db,runId,"E-006",3,125);
            auto add=stmt(db,"INSERT INTO treesem_v2_m4_statement_snapshots "
                "(run_id,statement_id,evidence_id,value_g_l) "
                "VALUES (?,'S-06','E-006',125)");
            add->setString(1,runId);
            add->executeUpdate();
            saveValidity(db,runId,"evidence","E-005","stale","EV-002");
            saveValidity(db,runId,"statement","S-05","stale","EV-002");
            saveValidity(db,runId,"evidence","E-006","current","");
            saveValidity(db,runId,"statement","S-06","current","");
            const Json result{
                {"run_id",runId},{"event_id","EV-002"},
                {"event_kind","official_correction_sim"},
                {"trusted_source","LIS_SIM_FIXTURE"},
                {"previous","E-005"},{"current","E-006"},
                {"previous_hb_g_l",130},{"current_hb_g_l",125},
                {"stale_statement_ids",Json::array({"S-05"})},
                {"new_statement_id","S-06"},{"new_issue_id","I-06"},
                {"issue_status","needs_review"},{"classification","SIM"},
                {"effective_on","2026-10-18"}
            };
            saveEvent(db,runId,"EV-002","official_correction_sim",fingerprint,result);
            auto issue=stmt(db,"INSERT INTO treesem_v2_m4_followups "
                "(run_id,issue_id,cause_event_id,old_evidence_id,new_evidence_id,status) "
                "VALUES (?,'I-06','EV-002','E-005','E-006','needs_review')");
            issue->setString(1,runId);
            issue->executeUpdate();
            return result;
        });
    });
}

Json V2SimArtifactStore::correctionState(
    const std::string& runId, const std::string& doctorId,
    const std::string& patientId)
{
    auto lease = pool_.acquire();
    return withConnection(lease, [&](sql::Connection& db) -> Json {
        owns(db, runId, doctorId, patientId, false);
        Json sources = Json::array();
        auto source = stmt(db, "SELECT s.evidence_id,s.logical_id,s.source_version,"
            "s.hb_g_l,v.status FROM treesem_v2_m4_source_versions s "
            "JOIN treesem_v2_m4_validity v ON v.run_id=s.run_id AND "
            "v.object_kind='evidence' AND v.object_id=s.evidence_id "
            "WHERE s.run_id=? ORDER BY s.source_version");
        source->setString(1, runId);
        auto rows = records(*source);
        while (rows->next())
            sources.push_back({
                {"evidence_id",field(*rows,"evidence_id")},
                {"logical_id",field(*rows,"logical_id")},
                {"version",rows->getInt("source_version")},
                {"hb_g_l",rows->getInt("hb_g_l")},
                {"validity",field(*rows,"status")}
            });
        Json statements = Json::array();
        auto statement = stmt(db, "SELECT s.statement_id,s.evidence_id,"
            "s.value_g_l,v.status FROM treesem_v2_m4_statement_snapshots s "
            "JOIN treesem_v2_m4_validity v ON v.run_id=s.run_id AND "
            "v.object_kind='statement' AND v.object_id=s.statement_id "
            "WHERE s.run_id=? ORDER BY s.statement_id");
        statement->setString(1, runId);
        auto stmtRows = records(*statement);
        while (stmtRows->next())
            statements.push_back({
                {"statement_id",field(*stmtRows,"statement_id")},
                {"evidence_id",field(*stmtRows,"evidence_id")},
                {"hb_g_l",stmtRows->getInt("value_g_l")},
                {"validity",field(*stmtRows,"status")}
            });
        Json events = Json::array();
        auto ev = stmt(db, "SELECT event_id FROM treesem_v2_m4_correction_events "
            "WHERE run_id=? ORDER BY created_at,event_id");
        ev->setString(1, runId);
        auto evRows = records(*ev);
        while (evRows->next()) events.push_back(field(*evRows, "event_id"));
        Json issues = Json::array();
        auto follow = stmt(db, "SELECT issue_id,status,old_evidence_id,"
            "new_evidence_id FROM treesem_v2_m4_followups WHERE run_id=?");
        follow->setString(1, runId);
        auto issueRows = records(*follow);
        while (issueRows->next())
            issues.push_back({
                {"issue_id",field(*issueRows,"issue_id")},
                {"status",field(*issueRows,"status")},
                {"old_evidence_id",field(*issueRows,"old_evidence_id")},
                {"new_evidence_id",field(*issueRows,"new_evidence_id")}
            });
        return {
            {"run_id",runId}, {"classification","SIM"},
            {"source_versions",sources}, {"statements",statements},
            {"events",events}, {"followup_issues",issues},
            {"model_capability","not_verified_for_prenatal"}
        };
    });
}

Json V2SimArtifactStore::checkCurrentSources(
    const std::string& runId, const std::string& doctorId,
    const std::string& patientId, const Json& evidenceIds)
{
    if (!evidenceIds.is_array() || evidenceIds.empty() ||
        evidenceIds.size() > 8)
        fail(Kind::InvalidInput, "current report requires bounded source IDs");
    std::set<std::string> unique;
    for (const auto& ref : evidenceIds)
    {
        if (!ref.is_string() || !unique.insert(ref.get<std::string>()).second)
            fail(Kind::InvalidInput, "duplicate or invalid source ID");
    }
    auto lease = pool_.acquire();
    return withConnection(lease, [&](sql::Connection& db) -> Json {
        owns(db, runId, doctorId, patientId, false);
        Json stale = Json::array();
        Json missing = Json::array();
        for (const auto& id : unique)
        {
            if (id == "E-004" || id == "E-005" || id == "E-006")
            {
                const auto state = sourceValidity(db, runId, id);
                if (state == "stale") stale.push_back(id);
                if (state == "missing") missing.push_back(id);
            }
            else if (id == "E-001" || id == "E-002" || id == "E-003")
            {
                // Neither LIS correction nor cross-source voting may silently
                // invalidate two independently documented delivery estimates.
                auto query = stmt(db, "SELECT COUNT(*) AS n FROM "
                    "treesem_v2_artifacts WHERE run_id=? AND kind='evidence' "
                    "AND JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.evidence_id'))=?");
                query->setString(1, runId);
                query->setString(2, id);
                auto row = records(*query);
                if (!row->next() || row->getInt("n") < 1)
                    missing.push_back(id);
            }
            else
                fail(Kind::InvalidInput, "source not in authorized SIM fixture");
        }
        const bool admissible = stale.empty() && missing.empty();
        return {
            {"run_id",runId}, {"classification","SIM"},
            {"evidence_ids",evidenceIds},
            {"admissible",admissible},
            {"status",admissible?"candidate_source_ids_current":
                                   "blocked_stale_or_missing_sources"},
            {"stale_refs",stale}, {"missing_refs",missing},
            {"affected_statement_ids",stale.empty()?Json::array():
                                                    Json::array({"S-04"})},
            {"model_capability","blocked_unverified_prenatal"},
            {"scope","source_version_only_not_semantic_grounding"}
        };
    });
}
} // namespace treesem::infrastructure
