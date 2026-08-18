ALTER TABLE treesem_agent_runs
    ADD COLUMN skill_id VARCHAR(64) NULL AFTER knowledge_index_version,
    ADD COLUMN skill_version VARCHAR(32) NULL AFTER skill_id,
    ADD COLUMN skill_catalog_version VARCHAR(128) NULL AFTER skill_version;

INSERT INTO schema_migrations(version) VALUES ('005_m8_agent_skills');
