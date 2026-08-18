ALTER TABLE treesem_agent_runs
    ADD COLUMN grounding_sources_json JSON NULL AFTER grounding_ids_json,
    ADD COLUMN knowledge_index_version VARCHAR(128) NULL AFTER grounding_sources_json;

UPDATE treesem_agent_runs
SET grounding_sources_json = JSON_OBJECT('ids', JSON_ARRAY(), 'citations', JSON_ARRAY())
WHERE grounding_sources_json IS NULL;

ALTER TABLE treesem_agent_runs
    MODIFY grounding_sources_json JSON NOT NULL;

INSERT INTO schema_migrations(version) VALUES ('004_m7_knowledge_grounding');
