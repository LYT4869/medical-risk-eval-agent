CREATE TABLE treesem_users (
    user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    email_normalized VARCHAR(254) NOT NULL UNIQUE,
    password_phc VARCHAR(512) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    role VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    status VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    failed_login_count INT NOT NULL DEFAULT 0,
    locked_until DATETIME(6) NULL,
    token_version BIGINT UNSIGNED NOT NULL DEFAULT 0,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    CONSTRAINT chk_treesem_user_role CHECK (role IN ('patient','doctor','admin')),
    CONSTRAINT chk_treesem_user_status CHECK (status IN ('active','disabled'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE treesem_refresh_sessions (
    refresh_session_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    token_family_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    token_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
    expires_at DATETIME(6) NOT NULL,
    consumed_at DATETIME(6) NULL,
    revoked_at DATETIME(6) NULL,
    replaced_by_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    created_at DATETIME(6) NOT NULL,
    last_used_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_treesem_refresh_user FOREIGN KEY (user_id)
        REFERENCES treesem_users(user_id) ON DELETE RESTRICT,
    INDEX idx_treesem_refresh_family (token_family_id),
    INDEX idx_treesem_refresh_expiry (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE treesem_doctor_patient_assignments (
    assignment_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    doctor_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    patient_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    status VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    created_by_admin_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    created_at DATETIME(6) NOT NULL,
    revoked_at DATETIME(6) NULL,
    CONSTRAINT fk_treesem_assignment_doctor FOREIGN KEY (doctor_user_id)
        REFERENCES treesem_users(user_id) ON DELETE RESTRICT,
    CONSTRAINT fk_treesem_assignment_patient FOREIGN KEY (patient_user_id)
        REFERENCES treesem_users(user_id) ON DELETE RESTRICT,
    CONSTRAINT fk_treesem_assignment_admin FOREIGN KEY (created_by_admin_id)
        REFERENCES treesem_users(user_id) ON DELETE RESTRICT,
    CONSTRAINT uq_treesem_assignment_pair UNIQUE (doctor_user_id, patient_user_id),
    CONSTRAINT chk_treesem_assignment_status CHECK (status IN ('active','revoked'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE treesem_audit_events (
    event_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    request_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    actor_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    actor_role VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NULL,
    action VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    resource_type VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    resource_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    outcome VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    reason_code VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    created_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_treesem_audit_actor FOREIGN KEY (actor_user_id)
        REFERENCES treesem_users(user_id) ON DELETE RESTRICT,
    CONSTRAINT chk_treesem_audit_outcome CHECK (outcome IN ('allowed','denied','failed')),
    INDEX idx_treesem_audit_actor (actor_user_id, created_at DESC, event_id DESC),
    INDEX idx_treesem_audit_action (action, created_at DESC, event_id DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

ALTER TABLE treesem_sessions
    ADD COLUMN owner_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    ADD COLUMN subject_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    ADD INDEX idx_treesem_session_owner (owner_user_id),
    ADD CONSTRAINT fk_treesem_session_owner FOREIGN KEY (owner_user_id)
        REFERENCES treesem_users(user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT fk_treesem_session_subject FOREIGN KEY (subject_user_id)
        REFERENCES treesem_users(user_id) ON DELETE RESTRICT;

ALTER TABLE treesem_predictions
    ADD COLUMN subject_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    ADD COLUMN created_by_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    ADD INDEX idx_treesem_prediction_subject
        (subject_user_id, created_at DESC, prediction_id DESC);

ALTER TABLE treesem_agent_runs
    ADD COLUMN actor_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    ADD COLUMN subject_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL;

ALTER TABLE treesem_chat_messages
    ADD COLUMN actor_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    ADD COLUMN subject_user_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL;

INSERT INTO schema_migrations(version) VALUES ('003_m6_auth_security');
