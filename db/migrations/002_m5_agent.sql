CREATE TABLE IF NOT EXISTS treesem_agent_runs (
    run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    session_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    idempotency_key VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    payload_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    status VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    step_count INT NOT NULL DEFAULT 0,
    tool_summary_json JSON NOT NULL,
    grounding_ids_json JSON NOT NULL,
    final_message_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    error_code VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    started_at DATETIME(6) NOT NULL,
    completed_at DATETIME(6) NULL,
    CONSTRAINT fk_treesem_agent_run_session FOREIGN KEY (session_id)
        REFERENCES treesem_sessions(session_id) ON DELETE RESTRICT,
    CONSTRAINT uq_treesem_agent_run_idempotency UNIQUE (session_id, idempotency_key),
    CONSTRAINT chk_treesem_agent_run_status CHECK (status IN ('running','completed','failed')),
    INDEX idx_treesem_agent_run_session (session_id, started_at DESC, run_id DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_chat_messages (
    message_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    session_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    role VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_treesem_chat_session FOREIGN KEY (session_id)
        REFERENCES treesem_sessions(session_id) ON DELETE RESTRICT,
    CONSTRAINT fk_treesem_chat_run FOREIGN KEY (run_id)
        REFERENCES treesem_agent_runs(run_id) ON DELETE RESTRICT,
    CONSTRAINT chk_treesem_chat_role CHECK (role IN ('user','assistant')),
    INDEX idx_treesem_chat_history (session_id, created_at DESC, message_id DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO schema_migrations(version) VALUES ('002_m5_agent');
