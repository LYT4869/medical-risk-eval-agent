#include "infrastructure/persistence/V2SimArtifactStore.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

[[noreturn]] void fail(Kind code, const char* why) { throw Error(code, why); }
void check(bool good, const char* why)
{
    if (!good) fail(Kind::InvalidInput, why);
}

auto prepared(sql::Connection& db, const std::string& sql)
{
    return std::unique_ptr<sql::PreparedStatement>(db.prepareStatement(sql));
}
auto selected(sql::PreparedStatement& statement)
{
    return std::unique_ptr<sql::ResultSet>(statement.executeQuery());
}
std::string field(sql::ResultSet& result, const char* name)
{
    return result.getString(name).asStdString();
}
std::string digest(const std::string& text)
{
    unsigned char bytes[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(text.data()), text.size(), bytes);
    static const char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned char byte : bytes) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    return result;
}
template<typename Work>
Json inConnection(MySqlConnectionPool::Lease& lease, Work work)
{
    try { return work(lease.connection()); }
    catch (const Error&) { throw; }
    catch (const sql::SQLException& error)
    {
        if (error.getSQLState().rfind("08", 0) == 0) lease.invalidate();
        if (error.getErrorCode() == 1062) fail(Kind::Conflict, "duplicate synthetic report version");
        fail(Kind::PersistenceFailure, "synthetic M5 report SQL failed");
    }
    catch (...) { fail(Kind::PersistenceFailure, "unexpected report database error"); }
}
template<typename Work>
Json atomic(sql::Connection& db, Work work)
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

std::string str(const Json& object, const char* key)
{
    check(object.is_object() && object.contains(key) && object.at(key).is_string(),
          "missing or invalid string in M5 report");
    return object.at(key).get<std::string>();
}
int integer(const Json& object, const char* key)
{
    check(object.is_object() && object.contains(key) &&
          object.at(key).is_number_integer(), "missing or invalid report revision");
    return object.at(key).get<int>();
}
bool truth(const Json& object, const char* key)
{
    check(object.is_object() && object.contains(key) &&
          object.at(key).is_boolean(), "missing or invalid boolean in report");
    return object.at(key).get<bool>();
}
const Json& part(const Json& object, const char* key)
{
    check(object.is_object() && object.contains(key),
          "missing M5 report structure field");
    return object.at(key);
}
void exactFields(const Json& object, std::initializer_list<const char*> names)
{
    check(object.is_object() && object.size()==names.size(),
          "unexpected/unreviewed fields in synthetic medical report");
    for (const auto* name : names)
        check(object.contains(name), "missing reviewed report field");
}
std::size_t utf8Length(const std::string& str)
{
    std::size_t length = 0;
    for (unsigned char ch : str)
        if ((ch & 0xc0) != 0x80) ++length;
    return length;
}
void ownedRun(sql::Connection& db, const std::string& run, const std::string& doctor,
              const std::string& patient, int* contractRevision, bool lock)
{
    auto query = prepared(db, std::string(
        "SELECT doctor_user_id,patient_user_id,classification,fixture_id,"
        "status,contract_revision FROM treesem_v2_runs WHERE run_id=?") +
        (lock ? " FOR UPDATE" : ""));
    query->setString(1, run);
    auto result = selected(*query);
    if (!result->next()) fail(Kind::NotFound, "unknown M5 Run");
    if (field(*result, "doctor_user_id") != doctor ||
        field(*result, "patient_user_id") != patient ||
        field(*result, "classification") != "SIM" ||
        field(*result, "fixture_id") != "P001")
        fail(Kind::NotFound, "Report does not belong to selected patient and doctor");
    if (field(*result, "status") != "draft_ready")
        fail(Kind::Conflict, "M5 requires completed M3 restricted Draft");
    if (contractRevision) *contractRevision = result->getInt("contract_revision");
}

// All current claims must bind to the *same authorized Run*. Unlike a new
// paragraph's candidate citations, authoritative M4 validity is queried from
// MySQL inside the transaction which stores a Report revision.
void currentEvidence(sql::Connection& db, const std::string& run)
{
    auto checkStatus = [&](const std::string& kind, const std::string& key,
                           const std::string& expected)
    {
        auto query = prepared(db, "SELECT status FROM treesem_v2_m4_validity "
            "WHERE run_id=? AND object_kind=? AND object_id=?");
        query->setString(1, run);
        query->setString(2, kind);
        query->setString(3, key);
        auto result = selected(*query);
        if (!result->next() || field(*result, "status") != expected)
            fail(Kind::Conflict, "report source version became stale or missing");
    };
    checkStatus("evidence", "E-004", "stale");
    checkStatus("evidence", "E-005", "current");
    checkStatus("statement", "S-04", "stale");
    checkStatus("statement", "S-05", "current");
    auto query = prepared(db, "SELECT evidence_id,source_version,hb_g_l,"
        "logical_id FROM treesem_v2_m4_source_versions WHERE run_id=?");
    query->setString(1, run);
    auto result = selected(*query);
    int seen = 0;
    while (result->next())
    {
        const std::string ref = field(*result, "evidence_id");
        if (field(*result, "logical_id") != "LIS-P001-HB-20261010")
            fail(Kind::Conflict, "unrecognized LIS logical source");
        if (ref == "E-004" && result->getInt("source_version") == 1 &&
            result->getInt("hb_g_l") == 110) ++seen;
        if (ref == "E-005" && result->getInt("source_version") == 2 &&
            result->getInt("hb_g_l") == 130) ++seen;
    }
    if (seen != 2) fail(Kind::Conflict, "report Hb source provenance incomplete");
    auto statements = prepared(db, "SELECT statement_id,evidence_id,value_g_l "
        "FROM treesem_v2_m4_statement_snapshots WHERE run_id=?");
    statements->setString(1, run);
    auto rows = selected(*statements);
    int verified = 0;
    while (rows->next())
    {
        const auto id = field(*rows, "statement_id");
        if (id == "S-04" && field(*rows,"evidence_id") == "E-004" &&
            rows->getInt("value_g_l") == 110) ++verified;
        if (id == "S-05" && field(*rows,"evidence_id") == "E-005" &&
            rows->getInt("value_g_l") == 130) ++verified;
    }
    if (verified != 2) fail(Kind::Conflict, "report Hb statement lineage incomplete");

    auto follow = prepared(db, "SELECT status FROM treesem_v2_m4_followups "
        "WHERE run_id=? AND issue_id='I-05'");
    follow->setString(1, run);
    auto issue = selected(*follow);
    if (!issue->next() || field(*issue, "status") != "needs_review")
        fail(Kind::Conflict, "unresolved official source correction not disclosed");
    // Independent SQL re-check of each synthetic M3 Evidence snapshot:
    // a forged citation and three arbitrary Evidence rows must not bypass
    // the programmed 250/approximately-800 source contract.
    auto evidence = prepared(db, "SELECT "
        "JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.evidence_id')) AS eid,"
        "JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.record.logical_id')) AS logicalid,"
        "JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.record.kind')) AS sourcekind,"
        "JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.record.version')) AS v,"
        "JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.record.blood_loss_ml')) AS loss,"
        "JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.record.approximate')) AS approx "
        "FROM treesem_v2_artifacts WHERE run_id=? AND kind='evidence'");
    evidence->setString(1, run);
    auto evidenceRows = selected(*evidence);
    int evidenceCount = 0;
    int evidenceRowsSeen = 0;
    while (evidenceRows->next())
    {
        ++evidenceRowsSeen;
        const auto ref = field(*evidenceRows, "eid");
        const auto logical = field(*evidenceRows, "logicalid");
        const auto kind = field(*evidenceRows, "sourcekind");
        if (field(*evidenceRows, "v") != "1")
            fail(Kind::Conflict, "unapproved original Evidence source version");
        if (ref == "E-001" && logical == "EMR-P001-ANC-20261010" &&
            kind == "patient_report") evidenceCount += 1;
        else if (ref == "E-002" && logical == "EMR-P001-DEL-20220508" &&
                 kind == "original_delivery" &&
                 field(*evidenceRows, "loss") == "250") evidenceCount += 2;
        else if (ref == "E-003" && logical == "REF-P001-20261012" &&
                 kind == "referral_summary" &&
                 field(*evidenceRows, "loss") == "800" &&
                 field(*evidenceRows, "approx") == "true") evidenceCount += 4;
        else
            fail(Kind::Conflict, "M3 Evidence payload contradicts P001 source");
    }
    if (evidenceCount != 7 || evidenceRowsSeen != 3)
        fail(Kind::Conflict, "M3 verified Evidence set incomplete");
    auto issueQ = prepared(db, "SELECT "
        "JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.issue_id')) AS iid,"
        "JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.status')) AS issue_state,"
        "JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.no_diagnosis')) AS no_diagnosis "
        "FROM treesem_v2_artifacts WHERE run_id=? AND kind='issue'");
    issueQ->setString(1, run);
    auto issueRows = selected(*issueQ);
    int issuesSeen = 0;
    int issueRowsSeen = 0;
    while (issueRows->next())
    {
        ++issueRowsSeen;
        const auto id = field(*issueRows,"iid");
        const auto state = field(*issueRows,"issue_state");
        if (field(*issueRows,"no_diagnosis") != "true")
            fail(Kind::Conflict,"IssueResult improperly asserts diagnosis");
        if (id == "I-01" && state == "closed_with_limitations") issuesSeen += 1;
        else if (id == "I-02" && state == "closed_unresolved") issuesSeen += 2;
        else if (id == "I-03" && state == "blocked") issuesSeen += 4;
        else if (id == "I-04" && state == "closed_unresolved") issuesSeen += 8;
        else fail(Kind::Conflict, "unapproved issue conclusion");
    }
    if (issuesSeen != 15 || issueRowsSeen != 4)
        fail(Kind::Conflict, "Report lacks expected IssueResult limits");
}

struct Fact
{
    const char* id;
    const char* full;
    const char* brief;
    std::vector<std::string> refs;
};
const std::vector<Fact>& facts()
{
    static const std::vector<Fact> all = {
        {"F-01", "患者档案包含一条患者自述类病史资料", "存在患者自述病史资料", {"E-001"}},
        {"F-02", "原始分娩记录中的估计失血量为250 mL", "原始记录估计失血250 mL", {"E-002"}},
        {"F-03", "另一份转诊摘要记载约800 mL失血量", "转诊摘要约800 mL", {"E-003"}},
        {"F-04", "上述两份来源记录的估计值不同，原因尚未核实", "两份来源的差异尚未核实", {"E-002","E-003"}},
        {"F-05", "当前专业模型未验证适用于产前任务，故本报告不提供风险分数", "产前模型未获验证，本报告无风险分数", {}},
        {"F-06", "同一LIS检验记录经官方更正后的血红蛋白数值为130 g/L", "LIS更正后Hb为130 g/L", {"E-005"}},
        {"F-07", "旧版110 g/L的血红蛋白记录已被同源新版本取代，不能作为当前数值", "旧版110 g/L已失效，不能作为当前值", {"E-004","E-005"}},
        {"F-08", "已有资料不能核实既往产后出血的正式诊断，不据此确认疾病", "已有资料不足以确认既往产后出血诊断", {"E-001","E-002","E-003"}},
        {"F-09", "现有调查保留来源差异及待复核项，这只是未签发的受限辅助草稿", "来源差异待复核，当前仅为受限草稿", {"E-002","E-003"}},
        {"F-10", "本合成调查没有检索经审核的指南或外部文献，不声称完成研究依据核对", "本次未核对医学指南及外部文献", {}},
    };
    return all;
}
struct Section
{
    const char* name;
    const char* paragraph;
    std::vector<std::string> factIds;
};
const std::vector<Section>& sectionPlan()
{
    static const std::vector<Section> p = {
        {"patient_summary","P-01",{"F-01","F-06"}},
        {"investigation","P-02",{"F-02","F-03","F-04"}},
        {"model_reference","P-03",{"F-05"}},
        {"research_basis","P-04",{"F-10"}},
        {"uncertainties","P-05",{"F-07","F-08"}},
        {"overall_summary","P-06",{"F-09"}},
    };
    return p;
}
const Fact& findFact(const std::string& id)
{
    for (const auto& fact : facts())
        if (id == fact.id) return fact;
    fail(Kind::InvalidInput, "unapproved factual assertion ID");
}

void verifyDocument(const Json& document, const std::string& run,
                    int expectedContract, int expectedRevision, bool isEdit)
{
    exactFields(document, {"classification","run_id","report_id",
        "draft_revision","contract_revision","status","not_signed",
        "clinical_unverified","synthetic_catalog_only","model_score",
        "writer_kind","source_snapshot","coverage","sections","grounding"});
    check(str(document,"run_id") == run &&
          str(document,"report_id") == "m5_" + run &&
          str(document,"classification") == "SIM" &&
          str(document,"status") == "restricted_draft" &&
          integer(document,"draft_revision") == expectedRevision &&
          integer(document,"contract_revision") == expectedContract &&
          truth(document,"not_signed") &&
          truth(document,"clinical_unverified") &&
          truth(document,"synthetic_catalog_only"),
          "M5 report identity, revision or restricted flags invalid");
    check(document.contains("model_score") && document.at("model_score").is_null(),
          "unverified prenatal model may not claim a score");
    const std::string writerKind = str(document,"writer_kind");
    check(isEdit ? writerKind == "sim_deterministic_patch" :
          (writerKind == "fake_catalog" ||
           writerKind == "local_model_checked_catalog"),
          "Writer was not independently verified for this prototype");

    const auto& coverage = part(document,"coverage");
    exactFields(coverage, {"status","method","contract_revision",
        "issue_states","objectives","reason_codes","source_snapshot",
        "no_clinical_diagnosis","not_signed"});
    check(str(coverage,"method") ==
          "fixed_p001_objectives_not_semantic_contract_interpretation",
          "unknown Report Coverage method");
    check(str(coverage,"status") == "restricted" &&
          integer(coverage,"contract_revision") == expectedContract &&
          truth(coverage,"not_signed") &&
          truth(coverage,"no_clinical_diagnosis"),
          "Coverage cannot silently certify clinical completeness");
    const auto& states = part(coverage,"issue_states");
    check(str(states,"I-01") == "closed_with_limitations" &&
          str(states,"I-02") == "closed_unresolved" &&
          str(states,"I-03") == "blocked" &&
          str(states,"I-04") == "closed_unresolved" &&
          str(states,"I-05") == "needs_review",
          "IssueResult/Correction limitations must appear in Coverage");
    const Json expectedReasons = Json::array({
        "DIFFERENT_SOURCE_ESTIMATES_UNRESOLVED",
        "PRENATAL_MODEL_NOT_VERIFIED",
        "LAB_REVISION_NEEDS_REVIEW",
        "NO_APPROVED_LITERATURE_RETRIEVAL"
    });
    check(part(coverage,"reason_codes") == expectedReasons,
          "unverified report limitation was omitted");
    check(part(coverage,"source_snapshot") == part(document,"source_snapshot"),
          "Coverage and report reference different source versions");
    const Json expectedObjectives = Json::array({
        Json{{"key","prior_history"},{"disposition","limited"},
             {"issues",Json::array({"I-01"})}},
        Json{{"key","source_comparison"},{"disposition","unresolved"},
             {"issues",Json::array({"I-02","I-04"})}},
        Json{{"key","model_reference"},{"disposition","blocked"},
             {"issues",Json::array({"I-03"})}},
        Json{{"key","lab_correction"},{"disposition","needs_review"},
             {"issues",Json::array({"I-05"})}},
        Json{{"key","literature"},{"disposition","not_searched"},
             {"issues",Json::array()}}
    });
    check(part(coverage,"objectives") == expectedObjectives,
          "Report Coverage silently changed confirmed P001 objective dispositions");
    const Json snapshots = {"E-001:v1","E-002:v1","E-003:v1",
                            "E-004:v1:historical_stale","E-005:v2"};
    check(part(document,"source_snapshot") == snapshots,
          "source snapshot is missing or stale");
    const auto& grounding = part(document,"grounding");
    exactFields(grounding, {"reviewer","total_sentences","total_assertions",
        "supported_assertions","unsupported_assertions","admission"});
    check(str(grounding,"reviewer") ==
          "strict_p001_catalog_not_general_clinical_semantics" &&
          integer(grounding,"total_sentences") == 6 &&
          integer(grounding,"total_assertions") == 10 &&
          integer(grounding,"supported_assertions") == 10 &&
          integer(grounding,"unsupported_assertions") == 0 &&
          str(grounding,"admission") == "restricted_only",
          "unreviewed factual assertions");
    const auto& sections = part(document,"sections");
    check(sections.is_array() && sections.size() == 6,
          "missing fixed report sections");
    std::size_t assertionCount = 0;
    for (std::size_t i = 0; i < sectionPlan().size(); ++i)
    {
        const auto& expected = sectionPlan()[i];
        const auto& section = sections[i];
        exactFields(section, {"section_id","paragraphs"});
        check(str(section,"section_id") == expected.name,
              "unexpected section title or ordering");
        const auto& paragraphs = part(section,"paragraphs");
        check(paragraphs.is_array() && paragraphs.size() == 1,
              "expected one provenance-bounded paragraph");
        const auto& p = paragraphs[0];
        exactFields(p, {"section_id","paragraph_id","text","render_variant",
            "fact_ids","candidate_evidence_refs","grounding_status","sentences"});
        check(str(p,"section_id") == expected.name &&
              str(p,"paragraph_id") == expected.paragraph &&
              str(p,"grounding_status") == "all_assertions_supported_sim_catalog",
              "unsupported paragraph or unknown provenance");
        const auto& variant = str(p,"render_variant");
        check(variant == "full" || variant == "short",
              "unsupported paragraph rewrite mode");
        check(part(p,"fact_ids") == Json(expected.factIds),
              "paragraph inserted, removed or reordered assertions");
        std::vector<std::string> refs;
        std::string canonical;
        for (const auto& factId : expected.factIds)
        {
            const auto& fact = findFact(factId);
            if (!canonical.empty()) canonical += "；";
            canonical += variant == "full" ? fact.full : fact.brief;
            for (const auto& ref : fact.refs)
                if (std::find(refs.begin(),refs.end(),ref) == refs.end())
                    refs.push_back(ref);
        }
        canonical += "。";
        check(str(p,"text") == canonical &&
              part(p,"candidate_evidence_refs") == Json(refs),
              "candidate citation or free-form assertion changed");
        const auto& sentences = part(p,"sentences");
        check(sentences.is_array() && sentences.size() == 1,
              "missing sentence or unsplit medical text");
        const auto& sentence = sentences[0];
        exactFields(sentence, {"sentence_id","start","end","text",
            "status","assertions"});
        const std::string suffix = std::string(expected.paragraph).substr(2);
        check(str(sentence,"sentence_id") == "S-" + suffix + "-01" &&
              str(sentence,"text") == canonical &&
              str(sentence,"status") == "all_assertions_reviewed_sim_catalog" &&
              integer(sentence,"start") == 0 &&
              integer(sentence,"end") == static_cast<int>(utf8Length(canonical)),
              "sentence IDs or character spans did not match rendered text");
        const auto& assertions = part(sentence,"assertions");
        check(assertions.is_array() && assertions.size() == expected.factIds.size(),
              "sentence omitted some independent factual assertions");
        for (std::size_t k = 0; k < assertions.size(); ++k)
        {
            const auto& claim = assertions[k];
            exactFields(claim, {"assertion_id","fact_id","text",
                "start","end","status","evidence_refs"});
            const auto& f = findFact(expected.factIds[k]);
            const std::string phrase = variant == "full" ? f.full : f.brief;
            const std::string aid = "A-" + suffix + "-01-0" + std::to_string(k+1);
            const auto offset = canonical.find(phrase);
            check(offset != std::string::npos &&
                  str(claim,"fact_id") == f.id &&
                  str(claim,"assertion_id") == aid &&
                  str(claim,"text") == phrase &&
                  str(claim,"status") == "supported_sim_catalog" &&
                  integer(claim,"start") ==
                      static_cast<int>(utf8Length(canonical.substr(0,offset))) &&
                  integer(claim,"end") ==
                      static_cast<int>(utf8Length(canonical.substr(0,offset+phrase.size()))),
                  "assertion phrase/span is not supported by trusted catalog");
            const auto& bindings = part(claim,"evidence_refs");
            check(bindings.is_array() && bindings.size() == f.refs.size(),
                  "statement source binding count changed");
            for (std::size_t n = 0; n < f.refs.size(); ++n)
            {
                const auto& link = bindings[n];
                exactFields(link, {"evidence_id","source_version","logical_id",
                    "temporal_scope","reference_kind"});
                const auto& ref = f.refs[n];
                check(str(link,"evidence_id") == ref &&
                      integer(link,"source_version") == (ref == "E-005" ? 2 : 1) &&
                      str(link,"temporal_scope") ==
                      (ref == "E-004" ? "historical_stale" :
                       ref == "E-005" ? "current" : "historical_T1"),
                      "source version or historical/current scope was forged");
                check(str(link,"logical_id") ==
                      (ref=="E-004" || ref=="E-005" ?
                          "LIS-P001-HB-20261010" :
                       ref=="E-001" ? "EMR-P001-ANC-20261010" :
                       ref=="E-002" ? "EMR-P001-DEL-20220508" :
                           "REF-P001-20261012"),
                      "source logical record was forged");
                check(str(link,"reference_kind") ==
                      (ref=="E-004" || ref=="E-005" ? "LIS_SIM" : "P001_SIM_EMR"),
                      "source namespace was forged");
            }
            ++assertionCount;
        }
    }
    check(assertionCount == 10, "report did not review each required assertion");
}

void ensureReportCurrent(sql::Connection& db, const std::string& run,
                         const Json& doc, int contractRevision)
{
    check(integer(doc,"contract_revision") == contractRevision,
          "report Contract version diverged");
    currentEvidence(db, run);
}

Json savedOp(sql::Connection& db, const std::string& run,
             const std::string& op, const std::string& fingerprint)
{
    auto query = prepared(db, "SELECT request_sha256,result_json "
        "FROM treesem_v2_m5_operations WHERE run_id=? AND operation_id=?");
    query->setString(1, run);
    query->setString(2, op);
    auto result = selected(*query);
    if (!result->next()) return nullptr;
    if (field(*result,"request_sha256") != fingerprint)
        fail(Kind::IdempotencyConflict, "report operation ID reused with different patch");
    return Json::parse(field(*result,"result_json"));
}
void insertOp(sql::Connection& db, const std::string& run,
              const std::string& op, const std::string& fingerprint,
              const Json& result)
{
    auto query = prepared(db, "INSERT INTO treesem_v2_m5_operations "
        "(run_id,operation_id,request_sha256,result_json) VALUES (?,?,?,?)");
    query->setString(1,run);
    query->setString(2,op);
    query->setString(3,fingerprint);
    query->setString(4,result.dump());
    query->executeUpdate();
}
void saveRevision(sql::Connection& db, const std::string& run,
                  int revision, const Json& document)
{
    auto query = prepared(db, "INSERT INTO treesem_v2_m5_report_revisions "
        "(run_id,revision,document_json,payload_sha256) VALUES (?,?,?,?)");
    const std::string text = document.dump();
    query->setString(1,run);
    query->setInt(2,revision);
    query->setString(3,text);
    query->setString(4,digest(text));
    query->executeUpdate();
}
Json readRevision(sql::Connection& db, const std::string& run, int revision)
{
    auto query = prepared(db, "SELECT document_json,payload_sha256 "
        "FROM treesem_v2_m5_report_revisions WHERE run_id=? AND revision=?");
    query->setString(1,run);
    query->setInt(2,revision);
    auto row = selected(*query);
    if (!row->next()) fail(Kind::NotFound, "unknown report revision");
    const std::string payload = field(*row,"document_json");
    Json document = Json::parse(payload);
    // MySQL JSON may normalize whitespace/key ordering. Its logical value
    // must survive serialization. Never rely on string-byte equality here.
    if (digest(document.dump()) != field(*row,"payload_sha256"))
        fail(Kind::PersistenceFailure,"report snapshot integrity mismatch");
    return document;
}
} // namespace

Json V2SimArtifactStore::publishM5Report(
    const std::string& runId, const std::string& doctorId,
    const std::string& patientId, const std::string& operationId,
    const Json& document)
{
    const std::string fingerprint = digest(Json::array({
        runId,doctorId,patientId,"publish",operationId,document
    }).dump());
    auto lease=pool_.acquire();
    return inConnection(lease,[&](sql::Connection& db) {
        return atomic(db,[&]() -> Json {
            int contractRev=0;
            ownedRun(db,runId,doctorId,patientId,&contractRev,true);
            const Json previous=savedOp(db,runId,operationId,fingerprint);
            if (!previous.is_null()) return previous;
            verifyDocument(document,runId,contractRev,1,false);
            ensureReportCurrent(db,runId,document,contractRev);
            auto exists=prepared(db,"SELECT current_revision FROM "
                "treesem_v2_m5_reports WHERE run_id=?");
            exists->setString(1,runId);
            auto existsRow=selected(*exists);
            if (existsRow->next())
                fail(Kind::Conflict,"different initial report already exists for Run");
            auto insert=prepared(db,"INSERT INTO treesem_v2_m5_reports "
                "(run_id,report_id,contract_revision,current_revision,status,classification) "
                "VALUES (?,?,?,1,'restricted_draft','SIM')");
            insert->setString(1,runId);
            insert->setString(2,"m5_"+runId);
            insert->setInt(3,contractRev);
            insert->executeUpdate();
            saveRevision(db,runId,1,document);
            const Json response{{"run_id",runId},{"report_id","m5_"+runId},
                {"revision",1},{"status","restricted_draft"},
                {"classification","SIM"},{"not_signed",true}};
            insertOp(db,runId,operationId,fingerprint,response);
            return response;
        });
    });
}

Json V2SimArtifactStore::getM5Revision(
    const std::string& runId, const std::string& doctorId,
    const std::string& patientId, const std::string& reportId, int revision)
{
    check(revision>=1 && revision<=20,"invalid requested report revision");
    auto lease=pool_.acquire();
    return inConnection(lease,[&](sql::Connection& db) -> Json {
        int contractRev=0;
        ownedRun(db,runId,doctorId,patientId,&contractRev,false);
        auto find=prepared(db,"SELECT report_id,contract_revision,current_revision "
            "FROM treesem_v2_m5_reports WHERE run_id=?");
        find->setString(1,runId);
        auto row=selected(*find);
        if (!row->next() || field(*row,"report_id")!=reportId)
            fail(Kind::NotFound,"selected report not in this Run/patient");
        if (revision>row->getInt("current_revision"))
            fail(Kind::NotFound,"report revision not yet published");
        const Json document=readRevision(db,runId,revision);
        // Reading an immutable old Draft is allowed for audit, but the UI
        // must not silently present its previously current Hb as current
        // after a later trusted correction. Never rewrite archived text.
        auto validity=prepared(db,"SELECT object_id,status FROM "
            "treesem_v2_m4_validity WHERE run_id=? "
            "AND ((object_kind='evidence' AND object_id='E-005') "
            "OR (object_kind='statement' AND object_id='S-05'))");
        validity->setString(1,runId);
        auto active=selected(*validity);
        int validCount=0;
        while(active->next()) {
            if(field(*active,"status")=="current") ++validCount;
        }
        const bool currentSources = validCount == 2;
        return Json{{"run_id",runId},{"report_id",reportId},
            {"revision",revision},{"current_revision",row->getInt("current_revision")},
            {"document",document},{"classification","SIM"},{"signed",false},
            {"source_freshness",currentSources?"current":"stale_requires_investigation"},
            {"signing_eligible",false},
            {"freshness_as_of","current_authorized_run_overlay"}};
    });
}

Json V2SimArtifactStore::m5DoctorMessage(
    const std::string& runId, const std::string& doctorId,
    const std::string& patientId, const std::string& reportId,
    const std::string& conversationId, const std::string& operationId,
    int baseRevision, const std::string& paragraphId,
    const std::string& intent, const Json& proposedDocument)
{
    check(!conversationId.empty() && conversationId.size()<=64 &&
          baseRevision>=1 && baseRevision<=19 &&
          (intent=="explain" || intent=="shorten") &&
          !paragraphId.empty() && paragraphId.size()<16,
          "invalid bounded doctor report message");
    if (intent=="shorten")
        check(!operationId.empty() && operationId.size()<=64 &&
              proposedDocument.is_object(),
              "shortening requires explicit operation ID and full proposed patch");
    else
        check(proposedDocument.is_null(),"explanation does not modify the report");
    const std::string fingerprint = digest(Json::array({
        runId,doctorId,patientId,reportId,conversationId,operationId,
        baseRevision,paragraphId,intent,proposedDocument
    }).dump());
    auto lease=pool_.acquire();
    return inConnection(lease,[&](sql::Connection& db) {
        return atomic(db,[&]() -> Json {
            int contractRev=0;
            ownedRun(db,runId,doctorId,patientId,&contractRev,true);
            auto owner=prepared(db,"SELECT report_id,contract_revision,"
                "current_revision FROM treesem_v2_m5_reports WHERE run_id=? FOR UPDATE");
            owner->setString(1,runId);
            auto row=selected(*owner);
            if (!row->next() || field(*row,"report_id")!=reportId)
                fail(Kind::NotFound,"report identity not selected or unauthorized");
            const int current=row->getInt("current_revision");
            if (intent=="shorten")
            {
                // Before replaying a successful old patch receipt, recheck
                // sources: a later LIS correction must not masquerade as a
                // currently valid edit simply because op_id is familiar.
                currentEvidence(db,runId);
                const Json old=savedOp(db,runId,operationId,fingerprint);
                if (!old.is_null()) return old;
            }
            if (current!=baseRevision)
                fail(Kind::Conflict,"report was modified; stale base_revision");
            const Json oldDoc=readRevision(db,runId,baseRevision);
            const auto& sections=part(oldDoc,"sections");
            int index=-1;
            for (std::size_t k=0;k<sections.size();++k)
                if (str(sections[k]["paragraphs"][0],"paragraph_id")==paragraphId)
                    index=static_cast<int>(k);
            if (index<0) fail(Kind::InvalidInput,"paragraph not in selected report");
            const auto& oldPara=sections.at(index)["paragraphs"][0];
            if (intent=="explain")
            {
                // Explanations are also subject to current source eligibility;
                // citing a stale Hb as *current* must never pass silently.
                ensureReportCurrent(db,runId,oldDoc,contractRev);
                return Json{{"run_id",runId},{"report_id",reportId},
                    {"base_revision",baseRevision},{"paragraph_id",paragraphId},
                    {"conversation_id",conversationId},{"intent","explain"},
                    {"text",str(oldPara,"text")},
                    {"assertions",oldPara["sentences"][0]["assertions"]},
                    {"source_note","P001_SIM_catalog_not_clinical_truth"},
                    {"created_revision",nullptr}};
            }
            if (current>=19)
                fail(Kind::Conflict,"M5 prototype revision budget exhausted");
            // No unsourced paragraphs, changed fact IDs, or sibling edits.
            Json allowed=oldDoc;
            allowed["draft_revision"]=current+1;
            allowed["writer_kind"]="sim_deterministic_patch";
            allowed["sections"][index]["paragraphs"][0]=
                part(proposedDocument,"sections").at(index)["paragraphs"][0];
            if (proposedDocument!=allowed)
                fail(Kind::InvalidInput,"patch changed data outside selected paragraph");
            if (str(oldPara,"render_variant")!="full" ||
                str(allowed["sections"][index]["paragraphs"][0],"render_variant")!="short")
                fail(Kind::Conflict,"paragraph already shortened or no progress");
            verifyDocument(proposedDocument,runId,contractRev,current+1,true);
            ensureReportCurrent(db,runId,proposedDocument,contractRev);
            auto update=prepared(db,"UPDATE treesem_v2_m5_reports "
                "SET current_revision=? WHERE run_id=? AND current_revision=? "
                "AND contract_revision=?");
            update->setInt(1,current+1);
            update->setString(2,runId);
            update->setInt(3,current);
            update->setInt(4,contractRev);
            if (update->executeUpdate()!=1)
                fail(Kind::Conflict,"Report CAS base_revision no longer current");
            saveRevision(db,runId,current+1,proposedDocument);
            const Json response{{"run_id",runId},{"report_id",reportId},
                {"base_revision",baseRevision},{"revision",current+1},
                {"paragraph_id",paragraphId},{"conversation_id",conversationId},
                {"intent","shorten"},{"status","restricted_draft"},
                {"classification","SIM"},{"not_signed",true}};
            insertOp(db,runId,operationId,fingerprint,response);
            return response;
        });
    });
}
} // namespace treesem::infrastructure
