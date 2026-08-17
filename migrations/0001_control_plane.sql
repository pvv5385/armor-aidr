-- Control-plane schema: named profiles + their detector checks, the
-- application_id -> profile mapping, and the per-request decision log.
-- See crates/storage/src/policy_store.rs.
--
-- `id`/`application_id` are the human-readable slugs already used by the
-- file-based config (armor_core::policy::schema::PolicyConfig.id,
-- config/profiles/*.yaml's `id:`), not surrogate UUIDs — this is a pure
-- storage swap for the existing resolution logic (crates/api/src/profiles.rs),
-- not a new identity scheme.

CREATE TABLE profiles (
    id              TEXT PRIMARY KEY,
    description     TEXT,
    -- Stored as JSONB (not TEXT) so these round-trip through
    -- serde_json::to_value/from_value exactly like every other
    -- armor_core::policy::schema value already does (see sync.rs's
    -- SyncPayload) — no hand-written enum<->string mapping to keep in sync.
    execution_mode  JSONB NOT NULL DEFAULT '"parallel"'::jsonb,
    fail_mode       JSONB NOT NULL DEFAULT '"fail_open"'::jsonb,
    normalize       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE checks (
    id          UUID PRIMARY KEY,
    profile_id  TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    category    TEXT NOT NULL,
    enabled     BOOLEAN NOT NULL DEFAULT true,
    on_fail     JSONB NOT NULL DEFAULT '"deny"'::jsonb,
    mode        JSONB NOT NULL DEFAULT '"block"'::jsonb,
    fail_mode   JSONB NOT NULL DEFAULT '"fail_open"'::jsonb,
    priority    INTEGER NOT NULL DEFAULT 0,
    options     JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX checks_profile_id_idx ON checks (profile_id);

-- No ON DELETE CASCADE here, deliberately: an application still pointing at
-- a profile blocks that profile's deletion via a foreign-key violation,
-- which crates/api/src/control_plane.rs maps to a 409 rather than silently
-- orphaning (or cascading away) an application's assignment.
CREATE TABLE applications (
    application_id  TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL REFERENCES profiles(id),
    name            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX applications_profile_id_idx ON applications (profile_id);

-- Per-request decision log — mirrors crates/api/src/audit.rs's
-- EvaluationEvent/CheckSummary (metadata only: category names and
-- pass/fail, never the request body or a hit's matched span).
CREATE TABLE evaluation_logs (
    id              UUID PRIMARY KEY,
    event_id        TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    application_id  TEXT,
    profile_id      TEXT,
    occurred_at     TIMESTAMPTZ NOT NULL,
    stage           TEXT NOT NULL,
    verdict         TEXT NOT NULL,
    checks          JSONB NOT NULL,
    latency_ms      DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX evaluation_logs_occurred_at_idx ON evaluation_logs (occurred_at DESC);
CREATE INDEX evaluation_logs_application_id_idx ON evaluation_logs (application_id);
