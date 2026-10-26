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
                          const std::string& overview,
                          const std::string& clarificationQuestion = "");

    nlohmann::json status(const std::string& runId,
                          const std::string& doctorId,
                          const std::string& patientId);

    nlohmann::json confirm(const std::string& runId,
                           const std::string& doctorId,
                           const std::string& patientId,
                           int expectedRevision,
                           int expectedContractRevision = 1);

    nlohmann::json editContract(const std::string& runId,
                                const std::string& doctorId,
                                const std::string& patientId,
                                const std::string& decision,
                                int expectedRevision,
                                int expectedContractRevision,
                                const std::string& summary,
                                const std::string& overview,
                                const std::string& answer);

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

    // M4: SIM-only, server-bundled synthetic LIS correction event EV-001.
    // No free-form evidence updates or fake "official" caller payload.
    nlohmann::json stageHistoricalLab(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId);
    nlohmann::json applyOfficialSimCorrection(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId);
    nlohmann::json correctionState(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId);
    nlohmann::json checkCurrentSources(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId,
        const nlohmann::json& evidenceIds);

    // M5: server-side independently checked, SIM-only report snapshots.
    // Initial Writer publishing requires run-scoped Agent capability;
    // doctor report messages use authenticated explicit report identity.
    nlohmann::json publishM5Report(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId, const std::string& operationId,
        const nlohmann::json& document);

    nlohmann::json getM5Revision(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId, const std::string& reportId,
        int revision);

    nlohmann::json m5DoctorMessage(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId, const std::string& reportId,
        const std::string& conversationId, const std::string& operationId,
        int baseRevision, const std::string& paragraphId,
        const std::string& intent, const nlohmann::json& proposedDocument);

    // M6: staged model proposal is untrusted until existing M5 publish validator
    // rechecks every Assertion and current source in a fresh SQL transaction.
    nlohmann::json stageM6Proposal(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId, const std::string& proposalKey,
        const nlohmann::json& document);
    nlohmann::json getM6Proposal(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId, const std::string& proposalKey);
    nlohmann::json publishM6Proposal(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId, const std::string& proposalKey,
        const std::string& operationId);

    // Late arrival of a second *fixed SIM* authoritative LIS amendment.
    // This is NOT patient-wide hospital LIS ingestion.
    nlohmann::json applyM6LateCorrection(
        const std::string& runId, const std::string& doctorId,
        const std::string& patientId);

private:
    MySqlConnectionPool pool_;
};
} // namespace treesem::infrastructure
