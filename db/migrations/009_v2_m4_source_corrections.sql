-- M4 P001 SIM event replay: immutable LIS snapshots / statements plus separate
-- validity overlays. No real LIS feed is attached to this testing interface.
CREATE TABLE IF NOT EXISTS treesem_v2_m4_source_versions (
  run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  evidence_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  logical_id VARCHAR(96) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  source_version INT NOT NULL,
  hb_g_l INT NOT NULL,
  source_kind VARCHAR(16) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (run_id,evidence_id),
  UNIQUE KEY uq_v2_m4_logical_version (run_id,logical_id,source_version),
  CONSTRAINT fk_v2_m4_source_run FOREIGN KEY (run_id)
    REFERENCES treesem_v2_runs(run_id) ON DELETE RESTRICT,
  CONSTRAINT chk_v2_m4_source_kind CHECK (source_kind = 'LIS_SIM')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_v2_m4_statement_snapshots (
  run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  statement_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  evidence_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  value_g_l INT NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY(run_id,statement_id),
  CONSTRAINT fk_v2_m4_statement_source FOREIGN KEY (run_id,evidence_id)
    REFERENCES treesem_v2_m4_source_versions(run_id,evidence_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_v2_m4_validity (
  run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  object_kind VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  object_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  status VARCHAR(16) NOT NULL,
  cause_event_id VARCHAR(24) CHARACTER SET ascii COLLATE ascii_bin NULL,
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
    ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY(run_id,object_kind,object_id),
  CONSTRAINT fk_v2_m4_validity_run FOREIGN KEY(run_id)
    REFERENCES treesem_v2_runs(run_id) ON DELETE RESTRICT,
  CONSTRAINT chk_v2_m4_validity_kind CHECK(object_kind IN ('evidence','statement')),
  CONSTRAINT chk_v2_m4_validity_status CHECK(status IN ('current','stale'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_v2_m4_correction_events (
  run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  event_id VARCHAR(24) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  event_kind VARCHAR(24) NOT NULL,
  request_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  result_json JSON NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY(run_id,event_id),
  CONSTRAINT fk_v2_m4_event_run FOREIGN KEY(run_id)
    REFERENCES treesem_v2_runs(run_id) ON DELETE RESTRICT,
  CONSTRAINT chk_v2_m4_event_kind CHECK(event_kind IN ('historical_seed','official_correction_sim'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_v2_m4_followups (
  run_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  issue_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  cause_event_id VARCHAR(24) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  old_evidence_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  new_evidence_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  status VARCHAR(24) NOT NULL,
  PRIMARY KEY(run_id,issue_id),
  CONSTRAINT fk_v2_m4_followup_event FOREIGN KEY(run_id,cause_event_id)
    REFERENCES treesem_v2_m4_correction_events(run_id,event_id),
  CONSTRAINT chk_v2_m4_followup_status CHECK(status IN ('needs_review'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO schema_migrations(version) VALUES ('009_v2_m4_source_corrections');
