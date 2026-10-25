#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "infrastructure/persistence/MySqlConnectionPool.h"

namespace treesem::infrastructure
{
// Prototype only: a doctor-scoped, SIM-labelled repository. Nothing written
// here may be treated as sourced HIS/EMR/EHR clinical evidence.
class V2SimArtifactStore
{
public:
    explicit V2SimArtifactStore(MySqlConnectionConfig config);

    nlohmann::json create(const std::string& runId,
                          const std::string& doctorId,
                          const std::string& patientId,
                          const std::string& idemKey,
                          const std::string& summary,
                          const std::string& overview);

    nlohmann::json status(const std::string& runId,
                          const std::string& doctorId,
                          const std::string& patientId);

    nlohmann::json confirm(const std::string& runId,
                           const std::string& doctorId,
                           const std::string& patientId,
                           int expectedRevision);

    nlohmann::json put(const std::string& runId,
                       const std::string& doctorId,
                       const std::string& patientId,
                       const std::string& operationId,
                       const std::string& kind,
                       const std::string& artifactKey,
                       const nlohmann::json& payload);

    nlohmann::json get(const std::string& runId,
                       const std::string& doctorId,
                       const std::string& patientId,
                       const std::string& kind,
                       const std::string& artifactKey);

private:
    MySqlConnectionPool pool_;
};
} // namespace treesem::infrastructure
