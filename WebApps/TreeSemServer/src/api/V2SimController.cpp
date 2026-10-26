#include "api/V2SimController.h"

#include <cctype>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>
#include <openssl/rand.h>

#include "api/HttpErrorMapper.h"
#include "application/BusinessException.h"
#include "infrastructure/support/ValueSupport.h"

namespace treesem::api
{
namespace
{
using Json = nlohmann::json;
using Error = application::BusinessException;
using Kind = application::BusinessException::Kind;

[[noreturn]] void reject(const char* msg)
{
    throw Error(Kind::InvalidInput, msg);
}

Json object(const http::HttpRequest& request)
{
    if (request.getBody().size() > 12000)
        reject("synthetic API request exceeds limit");
    const Json doc = Json::parse(request.getBody(), nullptr, false);
    if (!doc.is_object()) reject("expected a JSON object");
    return doc;
}

std::string str(const Json& doc, const char* key, std::size_t limit)
{
    if (!doc.contains(key) || !doc.at(key).is_string())
        reject("missing synthetic API string field");
    const std::string result = doc.at(key).get<std::string>();
    if (result.empty() || result.size() > limit)
        reject("invalid synthetic API string length");
    return result;
}

std::string id(const std::string& candidate)
{
    if (candidate.empty() || candidate.size() > 96)
        reject("invalid artifact identifier");
    for (unsigned char ch : candidate)
        if (!(std::isalnum(ch) || ch == '-' || ch == '_' || ch == ':' || ch == '.'))
            reject("invalid artifact identifier character");
    return candidate;
}

std::string newRunId()
{
    unsigned char random[16];
    if (RAND_bytes(random, sizeof(random)) != 1)
        throw Error(Kind::PersistenceFailure, "unable to generate Run ID");
    static const char hex[] = "0123456789abcdef";
    std::string result = "run_";
    for (unsigned char byte : random)
    {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}

template<typename Function>
http::ResponseWriter safe(Function&& function)
{
    try { return function(); }
    catch (const Error& e) { return HttpErrorMapper::from(e); }
    catch (const application::AuthException& e) { return HttpErrorMapper::from(e); }
    catch (...) { return HttpErrorMapper::internalError(); }
}

http::ResponseWriter ok(const Json& result)
{
    return HttpErrorMapper::success(result.dump());
}
} // namespace

V2SimController::V2SimController(
    const application::AuthService& auth, persistence::ISecurityStore& security,
    const application::AuditService& audit,
    const security::JwtService& jwt,
    infrastructure::V2SimArtifactStore& artifacts,
    service::BlockingTaskScheduler& scheduler)
    : auth_(auth), security_(security), audit_(audit), jwt_(jwt),
      artifacts_(artifacts), scheduler_(scheduler)
{}

std::string V2SimController::authorizedDoctor(const http::HttpRequest& request) const
{
    const std::string doctorId = request.getHeader("X-TreeSem-Actor-Id");
    const std::string role = request.getHeader("X-TreeSem-Actor-Role");
    const std::string patientId = request.getPathParameters("patient_id");
    if (doctorId.empty() || role != "doctor" || patientId.empty())
        throw application::AuthException(application::AuthException::Kind::Forbidden,
                                          "doctor token and patient selection required");
    const auto doctor = security_.findUserById(doctorId);
    if (!doctor || doctor->status != "active" ||
        doctor->role != domain::UserRole::Doctor)
        throw application::AuthException(application::AuthException::Kind::InvalidToken,
                                          "doctor account inactive");
    domain::ActorContext context;
    context.userId = doctorId;
    context.role = domain::UserRole::Doctor;
    context.subjectUserId = patientId;
    context.requestId = request.requestContext().requestId;
    auth_.authorizeSubject(context, patientId); // re-check active assignment
    return doctorId;
}

std::string V2SimController::authorizedInternal(const http::HttpRequest& request) const
{
    // Middleware MUST have verified a scoped V2 capability and overwritten the
    // actor, role, subject and run headers. Never trust caller JSON for identity.
    const std::string runId = request.getPathParameters("run_id");
    const std::string claimedRun = request.getHeader("X-TreeSem-Agent-Run-Id");
    const std::string actor = request.getHeader("X-TreeSem-Actor-Id");
    const std::string role = request.getHeader("X-TreeSem-Actor-Role");
    const std::string patient = request.getHeader("X-TreeSem-Subject-Id");
    if (request.path().rfind("/internal/v2/", 0) != 0 ||
        runId.empty() || runId != claimedRun ||
        request.getHeader("X-TreeSem-Session-Id") != "v2:" + runId ||
        role != "doctor" || actor.empty() || patient.empty())
        throw application::AuthException(application::AuthException::Kind::Forbidden,
                                          "scoped V2 capability required");
    const auto account = security_.findUserById(actor);
    if (!account || account->status != "active" ||
        account->role != domain::UserRole::Doctor)
        throw application::AuthException(application::AuthException::Kind::Forbidden,
                                          "doctor account not active");
    domain::ActorContext context;
    context.userId = actor;
    context.role = domain::UserRole::Doctor;
    context.subjectUserId = patient;
    context.requestId = request.requestContext().requestId;
    auth_.authorizeSubject(context, patient);
    return actor;
}

void V2SimController::status(http::HttpRequest request,
                             http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const bool internal = request.path().rfind("/internal/v2/", 0) == 0;
            const std::string actor = internal
                ? authorizedInternal(request) : authorizedDoctor(request);
            const std::string patient = internal
                ? request.getHeader("X-TreeSem-Subject-Id")
                : request.getPathParameters("patient_id");
            const std::string runId = id(request.getPathParameters("run_id"));
            const auto result = artifacts_.status(runId, actor, patient);
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::capability(http::HttpRequest request,
                                 http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            const std::string patient = request.getPathParameters("patient_id");
            const std::string runId = id(request.getPathParameters("run_id"));
            const auto state = artifacts_.status(runId, actor, patient);
            if (state.at("status") != "contract_confirmed" &&
                state.at("status") != "draft_ready")
                throw Error(Kind::Conflict, "doctor must confirm Contract first");
            security::CapabilityContext scope;
            scope.actorId = actor;
            scope.actorRole = domain::UserRole::Doctor;
            scope.subjectUserId = patient;
            scope.runId = runId;
            scope.sessionId = "v2:" + runId;
            scope.allowedTools = {"v2_sim_status", "v2_sim_artifact_read",
                                  "v2_sim_artifact_write"};
            const auto now = domain::TimePoint(std::chrono::microseconds(
                infrastructure::epochMicroseconds(std::chrono::system_clock::now())));
            const std::string token = jwt_.issueCapability(scope, now);
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.capability.issue", "v2_run", runId,
                          "allowed", "confirmed_run_short_lived");
            return ok(Json{{"run_id", runId}, {"capability_token", token},
                           {"token_type", "Bearer"}, {"scope", "v2_sim_artifacts"},
                           {"classification", "SIM"}});
        });
    }, std::move(responder));
}

void V2SimController::internalStatus(http::HttpRequest request,
                                     http::AsyncResponder responder) const
{
    status(std::move(request), std::move(responder));
}

void V2SimController::internalPut(http::HttpRequest request,
                                  http::AsyncResponder responder) const
{
    put(std::move(request), std::move(responder));
}

void V2SimController::internalGet(http::HttpRequest request,
                                  http::AsyncResponder responder) const
{
    get(std::move(request), std::move(responder));
}

void V2SimController::create(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            const Json body = object(request);
            if (str(body, "fixture_id", 16) != "P001")
                reject("only explicit synthetic P001 fixture permitted");
            const std::string key = id(str(body, "idempotency_key", 64));
            const std::string summary = str(body, "summary", 1200);
            const std::string overview = str(body, "overview", 1200);
            const std::string question = body.contains("clarification_question")
                ? str(body, "clarification_question", 1200) : "";
            const auto result = artifacts_.create(
                newRunId(), actor, request.getPathParameters("patient_id"),
                key, summary, overview, question);
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.run.create", "v2_run",
                          result.at("run_id").get<std::string>(),
                          "allowed", "authorized_synthetic_run");
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::confirm(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            const Json body = object(request);
            if (!body.contains("expected_revision") ||
                !body.at("expected_revision").is_number_integer())
                reject("expected_revision is required");
            const int revision = body.at("expected_revision").get<int>();
            if (body.contains("contract_revision") &&
                !body.at("contract_revision").is_number_integer())
                reject("contract_revision must be an integer");
            const int contractRevision = body.contains("contract_revision")
                ? body.at("contract_revision").get<int>() : 1;
            const std::string runId = id(request.getPathParameters("run_id"));
            const auto result = artifacts_.confirm(
                runId, actor, request.getPathParameters("patient_id"),
                revision, contractRevision);
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.contract.confirm", "v2_run", runId,
                          "allowed", "explicit_doctor_confirm");
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::editContract(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            const Json body = object(request);
            if (!body.contains("expected_revision") ||
                !body.at("expected_revision").is_number_integer() ||
                !body.contains("contract_revision") ||
                !body.at("contract_revision").is_number_integer())
                reject("both Contract and Run revisions must be integers");
            const std::string decision = str(body, "decision", 16);
            const std::string summary = body.contains("summary")
                ? str(body, "summary", 1200) : "";
            const std::string overview = body.contains("overview")
                ? str(body, "overview", 1200) : "";
            const std::string answer = body.contains("answer")
                ? str(body, "answer", 1200) : "";
            const std::string runId = id(request.getPathParameters("run_id"));
            const auto result = artifacts_.editContract(
                runId, actor, request.getPathParameters("patient_id"), decision,
                body.at("expected_revision").get<int>(),
                body.at("contract_revision").get<int>(),
                summary, overview, answer);
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.contract." + decision, "v2_run", runId,
                          "allowed", "versioned_doctor_contract_action");
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::put(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedInternal(request);
            const Json body = object(request);
            const std::string runId = id(request.getPathParameters("run_id"));
            const std::string operation = id(str(body, "operation_id", 64));
            const std::string kind = id(str(body, "kind", 24));
            const std::string artifact = id(str(body, "artifact_key", 96));
            if (!body.contains("payload") || !body.at("payload").is_object())
                reject("missing synthetic artifact payload");
            const auto result = artifacts_.put(
                runId, actor, request.getHeader("X-TreeSem-Subject-Id"), operation,
                kind, artifact, body.at("payload"));
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.artifact.write", "v2_run", runId,
                          "allowed", "synthetic_artifact_write");
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::stageHistoricalLab(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            if (!object(request).empty())
                reject("M4 historical fixture accepts no caller-supplied facts");
            const std::string run = id(request.getPathParameters("run_id"));
            const Json value = artifacts_.stageHistoricalLab(
                run, actor, request.getPathParameters("patient_id"));
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.m4.history", "v2_run", run,
                          "allowed", "fixed_synthetic_historical_LIS");
            return ok(value);
        });
    }, std::move(responder));
}

void V2SimController::applyOfficialSimCorrection(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            // Deliberate doctor-triggered replay of the one server-bundled
            // LIS SIM fixture, NOT a production LIS event ingestion endpoint.
            if (!object(request).empty())
                reject("M4 correction cannot accept arbitrary medical facts");
            const std::string run = id(request.getPathParameters("run_id"));
            const Json value = artifacts_.applyOfficialSimCorrection(
                run, actor, request.getPathParameters("patient_id"));
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.m4.official_fixture_event", "v2_run", run,
                          "allowed", "fixed_EV-001_event");
            return ok(value);
        });
    }, std::move(responder));
}

void V2SimController::correctionState(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            const std::string run = id(request.getPathParameters("run_id"));
            const Json value = artifacts_.correctionState(
                run, actor, request.getPathParameters("patient_id"));
            return ok(value);
        });
    }, std::move(responder));
}

void V2SimController::checkCurrentSources(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            const Json body = object(request);
            if (body.size() != 1 || !body.contains("evidence_ids"))
                reject("M4 source gate accepts evidence_ids only");
            const std::string run = id(request.getPathParameters("run_id"));
            const Json result = artifacts_.checkCurrentSources(
                run, actor, request.getPathParameters("patient_id"),
                body.at("evidence_ids"));
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::m6Stage(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request=std::move(request)] {
        return safe([&]() {
            const std::string doctor=authorizedInternal(request);
            const Json body=object(request);
            if(body.size()!=2 || !body.contains("document") ||
               !body.at("document").is_object())
                reject("M6 stage accepts a proposal key and restricted document");
            const std::string run=id(request.getPathParameters("run_id"));
            const std::string key=id(str(body,"proposal_key",64));
            const Json result=artifacts_.stageM6Proposal(
                run,doctor,request.getHeader("X-TreeSem-Subject-Id"),
                key,body.at("document"));
            audit_.record(doctor,"doctor",request.requestContext().requestId,
                          "v2.sim.m6.stage","v2_run",run,
                          "allowed","untrusted_writer_precommit_proposal");
            return ok(result);
        });
    },std::move(responder));
}

void V2SimController::m6Proposal(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request=std::move(request)] {
        return safe([&]() {
            const std::string doctor=authorizedInternal(request);
            const std::string run=id(request.getPathParameters("run_id"));
            const std::string key=id(request.getPathParameters("proposal_key"));
            return ok(artifacts_.getM6Proposal(
                run,doctor,request.getHeader("X-TreeSem-Subject-Id"),key));
        });
    },std::move(responder));
}

void V2SimController::m6Publish(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request=std::move(request)] {
        return safe([&]() {
            const std::string doctor=authorizedInternal(request);
            const Json body=object(request);
            if(body.size()!=2)
                reject("publish must refer only to a saved proposal and operation");
            const std::string run=id(request.getPathParameters("run_id"));
            const Json result=artifacts_.publishM6Proposal(
                run,doctor,request.getHeader("X-TreeSem-Subject-Id"),
                id(str(body,"proposal_key",64)),
                id(str(body,"operation_id",64)));
            audit_.record(doctor,"doctor",request.requestContext().requestId,
                          "v2.sim.m6.publish","v2_run",run,
                          "allowed","published_previously_staged_document");
            return ok(result);
        });
    },std::move(responder));
}

void V2SimController::m6LateCorrection(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request=std::move(request)] {
        return safe([&]() {
            const std::string doctor=authorizedDoctor(request);
            if(!object(request).empty())
                reject("late LIS SIM event has no caller-controlled medical facts");
            const std::string run=id(request.getPathParameters("run_id"));
            const Json result=artifacts_.applyM6LateCorrection(
                run,doctor,request.getPathParameters("patient_id"));
            audit_.record(doctor,"doctor",request.requestContext().requestId,
                          "v2.sim.m6.late_LIS","v2_run",run,
                          "allowed","fixed_EV-002_late_correction");
            return ok(result);
        });
    },std::move(responder));
}

void V2SimController::m5Publish(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            // Only M2's short-lived Run + Tool scoped capability may publish
            // the initially reviewed synthetic draft. Access JWT cannot.
            const std::string doctor = authorizedInternal(request);
            const Json body = object(request);
            if (!body.contains("document") || !body.at("document").is_object())
                reject("M5 publish requires a structured reviewed document");
            const std::string run = id(request.getPathParameters("run_id"));
            const std::string operation = id(str(body, "operation_id", 64));
            const Json result = artifacts_.publishM5Report(
                run, doctor, request.getHeader("X-TreeSem-Subject-Id"),
                operation, body.at("document"));
            audit_.record(doctor, "doctor", request.requestContext().requestId,
                          "v2.sim.m5.publish", "v2_run", run,
                          "allowed", "restricted_catalog_verified_report");
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::m5Revision(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string doctor = authorizedDoctor(request);
            const std::string run = id(request.getPathParameters("run_id"));
            const std::string report = id(request.getPathParameters("report_id"));
            const std::string version = id(request.getPathParameters("revision"));
            if (version.size() > 2)
                reject("invalid report revision");
            int revision = 0;
            for (unsigned char ch : version)
            {
                if (!std::isdigit(ch)) reject("revision must be decimal");
                revision = 10 * revision + (ch - '0');
            }
            const Json result = artifacts_.getM5Revision(
                run, doctor, request.getPathParameters("patient_id"),
                report, revision);
            audit_.record(doctor, "doctor", request.requestContext().requestId,
                          "v2.sim.m5.revision.read", "m5_report", report,
                          "allowed", "explicit_report_version");
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::m5Message(
    http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string doctor = authorizedDoctor(request);
            const std::string run = id(request.getPathParameters("run_id"));
            const std::string report = id(request.getPathParameters("report_id"));
            const Json body = object(request);
            if (!body.contains("base_revision") ||
                !body.at("base_revision").is_number_integer())
                reject("explicit base_revision required");
            const std::string intent = str(body, "intent", 16);
            if (intent != "explain" && intent != "shorten")
                reject("only bounded explanation or same-Evidence shortening available");
            const std::string conversation = id(str(body,"conversation_id",64));
            const std::string paragraph = id(str(body,"paragraph_id",16));
            const std::string operation = intent == "shorten"
                ? id(str(body,"operation_id",64)) : "";
            if (intent == "explain" &&
                (body.contains("operation_id") || body.contains("document")))
                reject("read-only explanation cannot carry a patch");
            if (intent == "shorten" &&
                (!body.contains("document") || !body.at("document").is_object()))
                reject("shortening requires a complete versioned draft patch");
            const Json result = artifacts_.m5DoctorMessage(
                run, doctor, request.getPathParameters("patient_id"), report,
                conversation, operation, body.at("base_revision").get<int>(),
                paragraph, intent,
                intent == "shorten" ? body.at("document") : Json(nullptr));
            audit_.record(doctor, "doctor", request.requestContext().requestId,
                          "v2.sim.m5.message." + intent, "m5_report", report,
                          "allowed", "explicit_report_base_revision");
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::get(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const bool internal = request.path().rfind("/internal/v2/", 0) == 0;
            const std::string actor = internal
                ? authorizedInternal(request) : authorizedDoctor(request);
            const std::string patient = internal
                ? request.getHeader("X-TreeSem-Subject-Id")
                : request.getPathParameters("patient_id");
            const std::string runId = id(request.getPathParameters("run_id"));
            const std::string kind = id(request.getPathParameters("kind"));
            const std::string artifact = id(request.getPathParameters("artifact_key"));
            const auto result = artifacts_.get(runId, actor, patient, kind, artifact);
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.artifact.read", "v2_run", runId,
                          "allowed", "synthetic_artifact_read");
            return ok(result);
        });
    }, std::move(responder));
}
} // namespace treesem::api
