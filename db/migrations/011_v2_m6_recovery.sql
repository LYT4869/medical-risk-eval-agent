-- M6 P001 SIM: durable pre-publish Writer proposal/outbox.
-- The report and the proposed document remain patient-specific business data
-- in MySQL, never in a LangGraph Checkpoint. Only one initial proposal per Run.
CREATE TABLE IF NOT EXISTS treesem_v2_m6_writer_proposals (
  run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  proposal_key VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  contract_revision INT NOT NULL,
  document_json JSON NOT NULL,
  payload_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY(run_id,proposal_key),
  CONSTRAINT fk_v2_m6_proposal_run FOREIGN KEY(run_id)
    REFERENCES treesem_v2_runs(run_id) ON DELETE RESTRICT,
  CONSTRAINT chk_v2_m6_contract_revision CHECK(contract_revision>=1),
  CONSTRAINT uq_v2_m6_initial_proposal UNIQUE KEY(run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO schema_migrations(version) VALUES ('011_v2_m6_recovery');
