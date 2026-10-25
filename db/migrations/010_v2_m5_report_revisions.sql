-- M5: versioned, immutable P001 SIM restricted report snapshots.
-- A report is explicitly tied to exactly one authorized generation Run.
-- Doctor report revisions require base_revision CAS and operation idempotency.
CREATE TABLE IF NOT EXISTS treesem_v2_m5_reports (
  run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  report_id VARCHAR(80) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  contract_revision INT NOT NULL,
  current_revision INT NOT NULL,
  status VARCHAR(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  classification VARCHAR(8) CHARACTER SET ascii COLLATE ascii_bin NOT NULL DEFAULT 'SIM',
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
    ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY(run_id),
  UNIQUE KEY uq_v2_m5_report_id (report_id),
  CONSTRAINT fk_v2_m5_report_run FOREIGN KEY(run_id)
    REFERENCES treesem_v2_runs(run_id) ON DELETE RESTRICT,
  CONSTRAINT chk_v2_m5_report_scope CHECK(
    status = 'restricted_draft' AND classification = 'SIM'
    AND current_revision >= 1 AND contract_revision >= 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_v2_m5_report_revisions (
  run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  revision INT NOT NULL,
  document_json JSON NOT NULL,
  payload_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY(run_id,revision),
  CONSTRAINT fk_v2_m5_report_revision FOREIGN KEY(run_id)
    REFERENCES treesem_v2_m5_reports(run_id) ON DELETE RESTRICT,
  CONSTRAINT chk_v2_m5_revision CHECK (revision >= 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_v2_m5_operations (
  run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  operation_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  request_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  result_json JSON NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY(run_id,operation_id),
  CONSTRAINT fk_v2_m5_ops_run FOREIGN KEY(run_id)
    REFERENCES treesem_v2_runs(run_id) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO schema_migrations(version) VALUES ('010_v2_m5_report_revisions');
