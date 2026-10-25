-- Incremental M2-B upgrade: do not rewrite already-applied migration 006.
-- Every V2 SIM Run explicitly references the sole bounded P001 test fixture.
ALTER TABLE treesem_v2_runs
    ADD COLUMN fixture_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin
        NOT NULL DEFAULT 'P001' AFTER classification;

ALTER TABLE treesem_v2_runs
    DROP CHECK chk_v2_run_sim;

ALTER TABLE treesem_v2_runs
    ADD CONSTRAINT chk_v2_run_sim
        CHECK (classification = 'SIM' AND fixture_id = 'P001');

INSERT IGNORE INTO schema_migrations(version) VALUES ('007_v2_sim_fixture_scope');
