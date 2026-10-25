#include "api/V2SimController.h"

#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>
#include <openssl/rand.h>

#include "api/HttpErrorMapper.h"
#include "application/BusinessException.h"

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
    infrastructure::V2SimArtifactStore& artifacts,
    service::BlockingTaskScheduler& scheduler)
    : auth_(auth), security_(security), audit_(audit),
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

void V2SimController::create(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            const Json body = object(request);
            const std::string key = id(str(body, "idempotency_key", 64));
            const std::string summary = str(body, "summary", 1200);
            const std::string overview = str(body, "overview", 1200);
            const auto result = artifacts_.create(
                newRunId(), actor, request.getPathParameters("patient_id"),
                key, summary, overview);
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
            const std::string runId = id(request.getPathParameters("run_id"));
            const auto result = artifacts_.confirm(
                runId, actor, request.getPathParameters("patient_id"), revision);
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.contract.confirm", "v2_run", runId,
                          "allowed", "explicit_doctor_confirm");
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::put(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            const Json body = object(request);
            const std::string runId = id(request.getPathParameters("run_id"));
            const std::string operation = id(str(body, "operation_id", 64));
            const std::string kind = id(str(body, "kind", 24));
            const std::string artifact = id(str(body, "artifact_key", 96));
            if (!body.contains("payload") || !body.at("payload").is_object())
                reject("missing synthetic artifact payload");
            const auto result = artifacts_.put(
                runId, actor, request.getPathParameters("patient_id"), operation,
                kind, artifact, body.at("payload"));
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.artifact.write", "v2_run", runId,
                          "allowed", "synthetic_artifact_write");
            return ok(result);
        });
    }, std::move(responder));
}

void V2SimController::get(http::HttpRequest request, http::AsyncResponder responder) const
{
    scheduler_.schedule([this, request = std::move(request)] {
        return safe([&]() {
            const std::string actor = authorizedDoctor(request);
            const std::string runId = id(request.getPathParameters("run_id"));
            const std::string kind = id(request.getPathParameters("kind"));
            const std::string artifact = id(request.getPathParameters("artifact_key"));
            const auto result = artifacts_.get(
                runId, actor, request.getPathParameters("patient_id"),
                kind, artifact);
            audit_.record(actor, "doctor", request.requestContext().requestId,
                          "v2.sim.artifact.read", "v2_run", runId,
                          "allowed", "synthetic_artifact_read");
            return ok(result);
        });
    }, std::move(responder));
}
} // namespace treesem::api
