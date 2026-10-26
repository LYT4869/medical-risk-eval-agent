-- M9: isolate dynamic, unverified M8 artifacts from legacy verified P001 kinds.
-- Additive schema migration only; old evidence/issue/draft and M5/M6 admission unchanged.
ALTER TABLE treesem_v2_artifacts
    DROP CHECK chk_v2_artifact_kind,
    ADD CONSTRAINT chk_v2_artifact_kind CHECK
      (kind IN ('evidence','issue','draft','m8_seed','m8_evidence','m8_issue'));

INSERT IGNORE INTO schema_migrations(version) VALUES ('012_v2_m9_dynamic_artifacts');
