-- M1 Gateway follow-up: immutable Task Contract versions and clarification history.
-- Do not mutate historical migrations 006/007; existing SIM Runs bootstrap as v1.
CREATE TABLE IF NOT EXISTS treesem_v2_contract_versions (
    run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    contract_revision INT NOT NULL,
    summary TEXT NOT NULL,
    investigation_overview TEXT NOT NULL,
    clarification_question TEXT NULL,
    clarification_answer TEXT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (run_id, contract_revision),
    CONSTRAINT fk_v2_contract_run FOREIGN KEY (run_id)
        REFERENCES treesem_v2_runs(run_id) ON DELETE RESTRICT,
    CONSTRAINT chk_v2_contract_rev CHECK (contract_revision >= 1),
    INDEX idx_v2_contract_run_time (run_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Existing M2-B databases contain only initial v1 Contracts.
-- Insert is repeatable if migration was interrupted after table creation.
INSERT IGNORE INTO treesem_v2_contract_versions
    (run_id, contract_revision, summary, investigation_overview,
     clarification_question, clarification_answer)
SELECT run_id, contract_revision, contract_summary, investigation_overview,
       NULL, NULL
FROM treesem_v2_runs;

INSERT IGNORE INTO schema_migrations(version) VALUES ('008_v2_contract_versions');
