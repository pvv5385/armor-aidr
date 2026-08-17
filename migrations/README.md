# migrations/

`sqlx` migrations for `armor-storage`, embedded at
compile time via `sqlx::migrate!("../../migrations")` in
`crates/storage/src/policy_store.rs` (path is relative to
`crates/storage/Cargo.toml`) and run automatically on `PgPolicyStore::connect`.

- `0001_control_plane.sql` — profiles (+ their detector checks),
  applications (`application_id -> profile_id`), and the per-request
  `evaluation_logs` decision log. See `crates/storage/src/policy_store.rs`
  and `crates/storage/src/audit_events.rs`.
- `0002_scan_id_and_client_request_id.sql` — renames
  `evaluation_logs.event_id` to `scan_id` and adds `client_request_id`
  (nullable). A separate migration rather than an edit to `0001` because
  `sqlx::migrate!` fingerprints each already-applied file and refuses to
  run if one changed underneath it — once a migration has shipped, alter
  the schema in a new file, never edit the old one in place.

- `0003_sessions_and_vault.sql` — `sessions` (durable per-session
  counters: lifetime request/token totals and fixed-window rate state,
  replacing the process-global maps in `armor_core::detectors::{abuse,
  unbounded_consumption}`), `vault_entries` (reversible anonymization —
  encrypted values plus an HMAC blind index), and
  `vault_placeholder_sequences` (atomic placeholder ordinals). See
  `crates/storage/src/sessions.rs` and `crates/storage/src/vault.rs`; each
  table's comment in the migration explains the choice it encodes.
- `0004_remove_check_priority.sql` — drops `checks.priority`: execution
  order is now a fixed backend-owned cheapest-first ranking
  (`armor_core::detectors::default_order`), not a per-check config value.

**Everything here keys on `session_id`, with no `tenant_id`.** That is safe
only for a single-tenant deployment — see `crates/storage/src/sessions.rs`'s
module doc for the reasoning and the migration path (widen both primary keys
to `(tenant_id, session_id)` once auth can resolve a tenant).

## Testing against a real database

The `sessions`/`vault` tests need Postgres, because what they assert *is*
database behavior — atomic upserts, unique-index races, `ON DELETE CASCADE`:

```bash
export ARMOR_TEST_DATABASE_URL=postgres://user:pass@localhost:5432/armor_test
cargo test -p armor-storage
```

Unset, they skip with a notice. CI always sets it (`.github/workflows/ci.yml`
runs a `postgres` service and fails the build if the tests report as
skipped).
