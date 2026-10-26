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
using Json = nlohmann::json;
using Error = application::BusinessException;
using Kind = Error::Kind;
[[noreturn]] void fail(Kind kind, const char* msg) { throw Error(kind, msg); }

auto prepared(sql::Connection& db, const std::string& query)
{
    return std::unique_ptr<sql::PreparedStatement>(db.prepareStatement(query));
}
auto rows(sql::PreparedStatement& query)
{
    return std::unique_ptr<sql::ResultSet>(query.executeQuery());
}
std::string field(sql::ResultSet& row, const char* key)
{
    return row.getString(key).asStdString();
}
std::string sha256(const std::string& content)
{
    unsigned char bytes[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(content.data()),
           content.size(), bytes);
    static const char hex[]="0123456789abcdef";
    std::string out;
    for(unsigned char v : bytes) { out += hex[v>>4]; out += hex[v&15]; }
    return out;
}
template <typename Work>
Json connected(MySqlConnectionPool::Lease& lease, Work work)
{
    try { return work(lease.connection()); }
    catch(const Error&) { throw; }
    catch(const sql::SQLException& e)
    {
        if(e.getSQLState().rfind("08",0)==0) lease.invalidate();
        if(e.getErrorCode()==1062) fail(Kind::Conflict,"Writer proposal exists");
        fail(Kind::PersistenceFailure,"M6 MySQL proposal operation failed");
    }
    catch(...) { fail(Kind::PersistenceFailure,"M6 proposal read/write failed"); }
}
template <typename Work>
Json transacted(sql::Connection& db, Work work)
{
    db.setAutoCommit(false);
    try {
        Json value=work();
        db.commit();
        db.setAutoCommit(true);
        return value;
    }
    catch(...) {
        try { db.rollback(); db.setAutoCommit(true); } catch(...) {}
        throw;
    }
}
int authorized(sql::Connection& db,const std::string& runId,
               const std::string& doctorId,const std::string& patientId,
               bool lock)
{
    auto command=prepared(db,
        std::string("SELECT doctor_user_id,patient_user_id,classification,"
                    "fixture_id,status,contract_revision "
                    "FROM treesem_v2_runs WHERE run_id=?") +
        (lock ? " FOR UPDATE" : ""));
    command->setString(1,runId);
    auto row=rows(*command);
    if(!row->next()) fail(Kind::NotFound,"unknown M6 Run");
    if(field(*row,"doctor_user_id")!=doctorId ||
       field(*row,"patient_user_id")!=patientId ||
       field(*row,"classification")!="SIM" ||
       field(*row,"fixture_id")!="P001")
       fail(Kind::NotFound,"M6 proposal not in authorized synthetic Run");
    if(field(*row,"status")!="draft_ready")
       fail(Kind::Conflict,"M6 proposal requires the completed M3 Run");
    return row->getInt("contract_revision");
}
Json saved(sql::Connection& db,const std::string& runId,
           const std::string& proposalKey)
{
    auto query=prepared(db,"SELECT contract_revision,document_json,payload_sha256 "
        "FROM treesem_v2_m6_writer_proposals WHERE run_id=? AND proposal_key=?");
    query->setString(1,runId);
    query->setString(2,proposalKey);
    auto row=rows(*query);
    if(!row->next()) fail(Kind::NotFound,"M6 proposal not yet staged");
    const Json document=Json::parse(field(*row,"document_json"));
    if(sha256(document.dump())!=field(*row,"payload_sha256"))
        fail(Kind::PersistenceFailure,"M6 staged proposal integrity failed");
    return Json{{"run_id",runId},{"proposal_key",proposalKey},
        {"contract_revision",row->getInt("contract_revision")},
        {"sha256",field(*row,"payload_sha256")},{"document",document},
        {"classification","SIM"},{"status","staged_untrusted"}};
}
} // namespace

Json V2SimArtifactStore::stageM6Proposal(
    const std::string& runId,const std::string& doctorId,
    const std::string& patientId,const std::string& proposalKey,
    const Json& document)
{
    if(!document.is_object() || document.value("run_id",std::string())!=runId ||
       document.value("report_id",std::string())!="m5_"+runId ||
       document.value("classification",std::string())!="SIM" ||
       document.value("status",std::string())!="restricted_draft" ||
       document.value("draft_revision",0)!=1 ||
       document.value("not_signed",false)!=true)
        fail(Kind::InvalidInput,"invalid untrusted P001 Writer proposal identity");
    const std::string fingerprint=sha256(document.dump());
    auto lease=pool_.acquire();
    return connected(lease,[&](sql::Connection& db) {
        return transacted(db,[&]() -> Json {
            const int contract=authorized(db,runId,doctorId,patientId,true);
            if(!document.contains("contract_revision") ||
               !document["contract_revision"].is_number_integer() ||
               document["contract_revision"].get<int>()!=contract)
                fail(Kind::Conflict,"proposal Contract changed since review");
            auto existing=prepared(db,"SELECT proposal_key,payload_sha256 "
                "FROM treesem_v2_m6_writer_proposals WHERE run_id=?");
            existing->setString(1,runId);
            auto prior=rows(*existing);
            if(prior->next()) {
                if(field(*prior,"proposal_key")!=proposalKey ||
                   field(*prior,"payload_sha256")!=fingerprint)
                    fail(Kind::IdempotencyConflict,
                         "same Run proposal cannot be overwritten on model retry");
                return saved(db,runId,proposalKey);
            }
            auto insert=prepared(db,"INSERT INTO treesem_v2_m6_writer_proposals "
                "(run_id,proposal_key,contract_revision,document_json,payload_sha256) "
                "VALUES (?,?,?,?,?)");
            insert->setString(1,runId);
            insert->setString(2,proposalKey);
            insert->setInt(3,contract);
            insert->setString(4,document.dump());
            insert->setString(5,fingerprint);
            insert->executeUpdate();
            return saved(db,runId,proposalKey);
        });
    });
}

Json V2SimArtifactStore::getM6Proposal(
    const std::string& runId,const std::string& doctorId,
    const std::string& patientId,const std::string& proposalKey)
{
    auto lease=pool_.acquire();
    return connected(lease,[&](sql::Connection& db) -> Json {
        const int version=authorized(db,runId,doctorId,patientId,false);
        const Json record=saved(db,runId,proposalKey);
        if(record["contract_revision"].get<int>()!=version)
            fail(Kind::Conflict,"proposal no longer matches confirmed Contract");
        return record;
    });
}

Json V2SimArtifactStore::publishM6Proposal(
    const std::string& runId,const std::string& doctorId,
    const std::string& patientId,const std::string& proposalKey,
    const std::string& operationId)
{
    const Json record=getM6Proposal(runId,doctorId,patientId,proposalKey);
    // Publication has its OWN fresh MySQL transaction: verifyDocument(),
    // authoritative Run/Contract and currentEvidence() all run before COMMIT.
    // A newer LIS revision after staging will prevent publication.
    return publishM5Report(runId,doctorId,patientId,operationId,
                           record.at("document"));
}
} // namespace treesem::infrastructure
