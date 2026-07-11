-- Registry for immutable materialized plans selected by method and memory ratio.
-- Apply this manually only after the target benchmark database is ready.

CREATE TABLE IF NOT EXISTS benchmark_plan_registry (
    registry_id BIGSERIAL PRIMARY KEY,
    method TEXT NOT NULL CHECK (method IN ('ours', 'honeybee', 'veda', 'effveda')),
    memory_ratio NUMERIC(12, 4) NOT NULL CHECK (memory_ratio >= 1.0),
    plan_id BIGINT NOT NULL,
    table_prefix TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('building', 'ready', 'failed', 'retired')),
    measured_space_bytes BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (method, memory_ratio)
);

CREATE TABLE IF NOT EXISTS benchmark_plan_relations (
    registry_id BIGINT NOT NULL REFERENCES benchmark_plan_registry(registry_id) ON DELETE CASCADE,
    relation_name TEXT NOT NULL,
    relation_kind TEXT NOT NULL CHECK (
        relation_kind IN (
            'partition',
            'index',
            'plan_metadata',
            'partition_metadata',
            'route_metadata',
            'pattern_metadata',
            'mapping_metadata'
        )
    ),
    PRIMARY KEY (registry_id, relation_name)
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'benchmark_plan_relations_relation_kind_check'
          AND conrelid = 'benchmark_plan_relations'::regclass
    ) THEN
        ALTER TABLE benchmark_plan_relations
            DROP CONSTRAINT benchmark_plan_relations_relation_kind_check;
    END IF;
END $$;

ALTER TABLE benchmark_plan_relations
    ADD CONSTRAINT benchmark_plan_relations_relation_kind_check
    CHECK (
        relation_kind IN (
            'partition',
            'index',
            'plan_metadata',
            'partition_metadata',
            'route_metadata',
            'pattern_metadata',
            'mapping_metadata'
        )
    );

CREATE INDEX IF NOT EXISTS benchmark_plan_registry_lookup_idx
    ON benchmark_plan_registry (method, memory_ratio)
    WHERE state = 'ready';
