CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(64) PRIMARY KEY,
    applied_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_sessions (
    session_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    current_prediction_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NULL,
    status VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    version BIGINT UNSIGNED NOT NULL DEFAULT 0,
    created_at DATETIME(6) NOT NULL,
    last_accessed_at DATETIME(6) NOT NULL,
    expires_at DATETIME(6) NOT NULL,
    INDEX idx_treesem_sessions_expires_at (expires_at),
    CONSTRAINT chk_treesem_session_status CHECK (status IN ('active', 'expired', 'revoked'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_predictions (
    prediction_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    session_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    model_version VARCHAR(128) NULL,
    serving_backend VARCHAR(32) NULL,
    input_source VARCHAR(64) NOT NULL,
    sample_index BIGINT NULL,
    label TINYINT NOT NULL,
    positive_probability DOUBLE NOT NULL,
    confidence DOUBLE NOT NULL,
    cluster_id INT NOT NULL,
    tree_probability DOUBLE NOT NULL,
    tree_leaf_id INT NOT NULL,
    result_json JSON NOT NULL,
    created_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_treesem_prediction_session
        FOREIGN KEY (session_id) REFERENCES treesem_sessions(session_id)
        ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT chk_treesem_prediction_label CHECK (label IN (0, 1)),
    CONSTRAINT chk_treesem_positive_probability
        CHECK (positive_probability >= 0 AND positive_probability <= 1),
    CONSTRAINT chk_treesem_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT chk_treesem_tree_probability
        CHECK (tree_probability >= 0 AND tree_probability <= 1),
    INDEX idx_treesem_prediction_history
        (session_id, created_at DESC, prediction_id DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS treesem_prediction_feedback (
    feedback_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    prediction_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    session_id VARCHAR(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    reviewer_reference VARCHAR(128) NOT NULL,
    reviewer_verified BOOLEAN NOT NULL DEFAULT FALSE,
    assessment VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    corrected_label TINYINT NULL,
    comment VARCHAR(1000) NULL,
    idempotency_key VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    payload_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    created_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_treesem_feedback_prediction
        FOREIGN KEY (prediction_id) REFERENCES treesem_predictions(prediction_id)
        ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT fk_treesem_feedback_session
        FOREIGN KEY (session_id) REFERENCES treesem_sessions(session_id)
        ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT uq_treesem_feedback_idempotency UNIQUE (session_id, idempotency_key),
    CONSTRAINT chk_treesem_feedback_assessment
        CHECK (assessment IN ('agree', 'disagree', 'uncertain')),
    CONSTRAINT chk_treesem_feedback_corrected_label
        CHECK (corrected_label IS NULL OR corrected_label IN (0, 1)),
    CONSTRAINT chk_treesem_feedback_semantics CHECK (
        (assessment = 'agree' AND corrected_label IS NULL) OR
        (assessment = 'uncertain' AND corrected_label IS NULL AND
            comment IS NOT NULL AND CHAR_LENGTH(TRIM(comment)) > 0) OR
        (assessment = 'disagree' AND corrected_label IS NOT NULL AND
            comment IS NOT NULL AND CHAR_LENGTH(TRIM(comment)) > 0)
    ),
    INDEX idx_treesem_feedback_prediction (prediction_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO schema_migrations(version) VALUES ('001_m4_core');
