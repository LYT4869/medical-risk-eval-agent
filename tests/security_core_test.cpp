#include <cassert>
#include <chrono>
#include <string>

#include "application/AuthService.h"
#include "infrastructure/persistence/InMemoryTreeSemStore.h"
#include "security/JwtService.h"
#include "security/PasswordHasher.h"
#include "security/SecurityMiddleware.h"

int main()
{
    using namespace treesem;
    const std::string accessSecret = "access-secret-for-tests-at-least-32-bytes";
    const std::string capabilitySecret = "capability-secret-for-tests-at-least-32-bytes";
    security::PasswordHasher passwords;
    const std::string encoded = passwords.hash("correct horse battery staple");
    assert(passwords.verify(encoded, "correct horse battery staple"));
    assert(!passwords.verify(encoded, "incorrect password value"));

    infrastructure::InMemoryTreeSemStore store;
    security::JwtService jwt({accessSecret, capabilitySecret,
        std::chrono::seconds(900), std::chrono::seconds(120)});
    application::AuthService auth(store, passwords, jwt,
        std::chrono::seconds(900), std::chrono::seconds(604800));
    const auto patient = auth.registerPatient(
        "PATIENT@example.com", "patient password long enough", "Patient",
        std::nullopt, "req_11111111111111111111111111111111");
    auto actor = auth.authenticate(patient.accessToken,
        "req_22222222222222222222222222222222");
    assert(actor.userId == patient.user.userId);
    assert(actor.role == domain::UserRole::Patient);

    std::string tampered = patient.accessToken;
    tampered.back() = tampered.back() == 'a' ? 'b' : 'a';
    bool rejected = false;
    try { (void)auth.authenticate(tampered, "req_33333333333333333333333333333333"); }
    catch (const application::AuthException&) { rejected = true; }
    assert(rejected);

    const auto refreshed = auth.refresh(patient.refreshToken,
        "req_44444444444444444444444444444444");
    assert(!refreshed.accessToken.empty());
    rejected = false;
    try { (void)auth.refresh(patient.refreshToken,
        "req_55555555555555555555555555555555"); }
    catch (const application::AuthException&) { rejected = true; }
    assert(rejected);

    security::CapabilityContext capability{actor.userId, actor.role,
        "ses_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", actor.userId,
        "run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", {"get_prediction"}};
    const auto now = domain::TimePoint(std::chrono::system_clock::now());
    const auto token = jwt.issueCapability(capability, now);
    const auto decoded = jwt.verifyCapability(token, now);
    assert(decoded.sessionId == capability.sessionId);
    assert(decoded.allowedTools.size() == 1);

    security::SecurityMiddleware middleware(jwt, true, "http://127.0.0.1:3000");
    http::HttpRequest allowed;
    const std::string get = "GET";
    const std::string predictionPath = "/internal/v1/predictions/pred_test";
    assert(allowed.setMethod(get.data(), get.data() + get.size()));
    allowed.setPath(predictionPath.data(), predictionPath.data() + predictionPath.size());
    allowed.setHeader("Authorization", "Bearer " + token);
    allowed.setHeader("X-TreeSem-Session-Id", capability.sessionId);
    middleware.before(allowed);
    assert(allowed.getHeader("X-TreeSem-Agent-Run-Id") == capability.runId);

    http::HttpRequest outOfScope;
    const std::string explanationPath = "/internal/v1/explanations/pred_test";
    assert(outOfScope.setMethod(get.data(), get.data() + get.size()));
    outOfScope.setPath(explanationPath.data(),
                       explanationPath.data() + explanationPath.size());
    outOfScope.setHeader("Authorization", "Bearer " + token);
    outOfScope.setHeader("X-TreeSem-Session-Id", capability.sessionId);
    rejected = false;
    try { middleware.before(outOfScope); }
    catch (const http::HttpResponse& response)
    {
        rejected = response.getStatusCode() == http::HttpResponse::k403Forbidden;
    }
    assert(rejected);

    http::HttpRequest wrongAudience;
    assert(wrongAudience.setMethod(get.data(), get.data() + get.size()));
    wrongAudience.setPath(predictionPath.data(),
                          predictionPath.data() + predictionPath.size());
    wrongAudience.setHeader("Authorization", "Bearer " + patient.accessToken);
    wrongAudience.setHeader("X-TreeSem-Session-Id", capability.sessionId);
    rejected = false;
    try { middleware.before(wrongAudience); }
    catch (const http::HttpResponse& response)
    {
        rejected = response.getStatusCode() == http::HttpResponse::k403Forbidden;
    }
    assert(rejected);

    http::HttpRequest corsRequest;
    const std::string publicPath = "/api/v1/auth/me";
    assert(corsRequest.setMethod(get.data(), get.data() + get.size()));
    corsRequest.setPath(publicPath.data(), publicPath.data() + publicPath.size());
    corsRequest.setHeader("Origin", "http://127.0.0.1:3000");
    corsRequest.setHeader("Authorization", "Bearer " + patient.accessToken);
    middleware.before(corsRequest);
    http::HttpResponse corsResponse;
    middleware.after(corsRequest, corsResponse);
    assert(corsResponse.getHeader("Access-Control-Allow-Origin") ==
           "http://127.0.0.1:3000");
    assert(corsResponse.getHeader("Access-Control-Allow-Credentials") == "true");

    http::HttpRequest preflight;
    const std::string options = "OPTIONS";
    assert(preflight.setMethod(options.data(), options.data() + options.size()));
    preflight.setPath(publicPath.data(), publicPath.data() + publicPath.size());
    preflight.setHeader("Origin", "http://127.0.0.1:3000");
    bool preflightAccepted = false;
    try { middleware.before(preflight); }
    catch (const http::HttpResponse& response)
    {
        preflightAccepted = response.getStatusCode() ==
            http::HttpResponse::k204NoContent &&
            response.getHeader("Access-Control-Allow-Origin") ==
                "http://127.0.0.1:3000";
    }
    assert(preflightAccepted);

    domain::UserRecord admin;
    admin.userId = "usr_" + std::string(32, 'c');
    admin.emailNormalized = "admin@example.com";
    admin.passwordPhc = encoded;
    admin.displayName = "Admin";
    admin.role = domain::UserRole::Admin;
    admin.createdAt = now;
    admin.updatedAt = now;
    store.createUser(admin);
    domain::ActorContext adminActor{admin.userId, admin.role, 0, std::nullopt,
        admin.userId, "req_" + std::string(32, '6')};
    const auto doctor = auth.createUser(adminActor, "doctor@example.com",
        "doctor password long enough", "Doctor", domain::UserRole::Doctor);
    const auto assignment = auth.assign(
        adminActor, doctor.userId, patient.user.userId);
    domain::ActorContext doctorActor{doctor.userId, doctor.role, 0,
        std::nullopt, patient.user.userId, "req_" + std::string(32, '7')};
    auth.authorizeSubject(doctorActor, patient.user.userId);
    auth.revokeAssignment(adminActor, assignment.assignmentId);
    rejected = false;
    try { auth.authorizeSubject(doctorActor, patient.user.userId); }
    catch (const application::AuthException&) { rejected = true; }
    assert(rejected);

    for (int attempt = 0; attempt < 5; ++attempt)
    {
        try
        {
            (void)auth.login("patient@example.com", "wrong password value",
                std::nullopt, "req_" + std::string(32, '8'));
        }
        catch (const application::AuthException&) {}
    }
    bool locked = false;
    try
    {
        (void)auth.login("patient@example.com", "patient password long enough",
            std::nullopt, "req_" + std::string(32, '9'));
    }
    catch (const application::AuthException& error)
    {
        locked = error.kind() == application::AuthException::Kind::RateLimited;
    }
    assert(locked);

    bool unknownLimited = false;
    for (int attempt = 0; attempt < 5; ++attempt)
    {
        try
        {
            (void)auth.login("unknown@example.com", "wrong password value",
                std::nullopt, "req_" + std::string(32, 'a'));
        }
        catch (const application::AuthException& error)
        {
            unknownLimited = error.kind() ==
                application::AuthException::Kind::RateLimited;
        }
    }
    assert(unknownLimited);
}
