-- The DB-backed profile store (crates/storage/src/policy_store.rs) only
-- ever persisted a check's deterministic-layer config (category/enabled/
-- on_fail/mode/fail_mode/options) — `strategy`/`backends`/`scorecard`
-- (armor_core::policy::schema::CheckConfig's ML-escalation fields) were
-- silently dropped on every save, both through the control-plane API and
-- through seed_default. This adds the missing columns so a profile edited
-- or seeded through Postgres can actually configure ML escalation, same as
-- a hand-written config/policies.yaml already could via the file loader.

ALTER TABLE checks
    ADD COLUMN strategy  JSONB,
    ADD COLUMN backends  JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN scorecard JSONB;
