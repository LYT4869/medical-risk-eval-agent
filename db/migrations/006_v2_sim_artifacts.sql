-- TreeSem V2 M2: isolated, explicitly SIM-only Artifact API prototype.
-- Never interpret these synthetic artifacts as audited clinical records.
CREATE TABLE IF NOT EXISTS treesem_v2_runs (
    run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    doctor_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    patient_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    idempotency_key VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    request_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    contract_revision INT NOT NULL DEFAULT 1,
    business_revision INT NOT NULL DEFAULT 0,
    contract_summary TEXT NOT NULL,
    investigation_overview TEXT NOT NULL,
    status VARCHAR(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    classification VARCHAR(8) CHARACTER SET ascii COLLATE ascii_bin NOT NULL DEFAULT 'SIM',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_v2_run_doctor FOREIGN KEY (doctor_user_id)
        REFERENCES treesem_users(user_id) ON DELETE RESTRICT,
    CONSTRAINT fk_v2_run_patient FOREIGN KEY (patient_user_id)
        REFERENCES treesem_users(user_id) ON DELETE RESTRICT,
    CONSTRAINT uq_v2_run_creation UNIQUE (doctor_user_id, patient_user_id, idempotency_key),
    CONSTRAINT chk_v2_run_status CHECK
        (status IN ('awaiting_confirmation', 'contract_confirmed', 'draft_ready')),
    CONSTRAINT chk_v2_run_sim CHECK (classification = 'SIM'),
    INDEX idx_v2_run_patient (patient_user_id, created_at DESC, run_id DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_v2_artifacts (
    run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    kind VARCHAR(24) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    artifact_key VARCHAR(96) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    payload_json JSON NOT NULL,
    payload_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (run_id, kind, artifact_key),
    CONSTRAINT fk_v2_artifact_run FOREIGN KEY (run_id)
        REFERENCES treesem_v2_runs(run_id) ON DELETE RESTRICT,
    CONSTRAINT chk_v2_artifact_kind CHECK (kind IN ('evidence','issue','draft')),
    INDEX idx_v2_artifact_run_kind (run_id, kind)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_v2_operations (
    run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    operation_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    phase VARCHAR(24) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    request_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    result_json JSON NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (run_id, operation_id),
    CONSTRAINT fk_v2_operation_run FOREIGN KEY (run_id)
        REFERENCES treesem_v2_runs(run_id) ON DELETE RESTRICT,
    INDEX idx_v2_operation_run (run_id, phase)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO schema_migrations(version) VALUES ('006_v2_sim_artifacts');
