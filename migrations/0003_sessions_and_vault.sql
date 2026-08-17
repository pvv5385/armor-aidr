-- Session/conversation state and the reversible-anonymization vault — the
-- two pieces `crates/storage/src/lib.rs` previously described as not yet
-- implemented.
--
-- Both tables key on `session_id` alone. See crates/storage/src/sessions.rs's
-- module doc for the tenancy caveat this carries and the migration that
-- closes it; the short version is that these tables are safe for a
-- single-tenant deployment and MUST gain `tenant_id` before more than one
-- tenant shares a database.

-- Durable per-session counters. Replaces the process-global
-- Mutex<HashMap> in armor_core::detectors::{abuse, unbounded_consumption},
-- which silently under-counts once more than one replica sits behind a load
-- balancer (each process counts only the requests it personally saw).
--
-- One row per session, updated in place — deliberately not an append-only
-- event log. The detectors need "how many requests/tokens so far", which is
-- a counter, and a row-per-request table would make every check a COUNT(*)
-- over a growing partition. `evaluation_logs` already holds the
-- per-request history for anything that needs it, keyed by the same
-- `session_id` — so counters and history live in separate tables, each
-- shaped for its own access pattern, rather than one table trying to do
-- both jobs.
CREATE TABLE sessions (
    session_id              TEXT PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Lifetime totals, for unbounded_consumption's session budgets.
    -- BIGINT rather than INTEGER: a long-lived agent session can plausibly
    -- exceed 2^31 tokens, and an overflow here would wrap a budget check
    -- from "over" to "under", i.e. fail open on the exact input it exists
    -- to catch.
    request_count           BIGINT NOT NULL DEFAULT 0,
    total_tokens            BIGINT NOT NULL DEFAULT 0,

    -- Fixed-window rate state, for abuse. Mirrors detectors/abuse.rs's
    -- WindowState exactly (window start + count, reset on rollover) so the
    -- durable path and the in-process fallback share one mechanism rather
    -- than being two different rate limiters with two different behaviors.
    window_started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_request_count    BIGINT NOT NULL DEFAULT 0,

    -- Retention. NULL means "no expiry configured"; sessions::purge_expired
    -- deletes rows whose expires_at has passed. Set from
    -- ARMOR_SESSION_TTL_SECONDS at touch time so a TTL change applies to
    -- live sessions on their next request rather than only to new ones.
    expires_at              TIMESTAMPTZ
);

-- purge_expired's sweep predicate. Partial, since rows with no expiry are
-- never swept and would otherwise bloat the index.
CREATE INDEX sessions_expires_at_idx ON sessions (expires_at)
    WHERE expires_at IS NOT NULL;

-- Reversible anonymization (the llm-guard `Vault` shape): detected PII is
-- replaced with a stable placeholder, the same value always maps to the
-- same placeholder within a session, and a paired deanonymize step restores
-- the original for a trusted downstream consumer.
--
-- This table holds recoverable PII, which makes it the one part of this
-- schema that can fail an enterprise security review on its own. Two
-- properties defend it:
--
--   1. Values are stored as AES-256-GCM ciphertext, encrypted in
--      armor-storage before they reach Postgres. A dump of this table, or
--      of the underlying volume, yields ciphertext — the key never enters
--      the database, the query string, or the query log.
--   2. Lookup by value uses `value_index`, an HMAC-SHA256 blind index
--      keyed by the same secret, NOT a bare hash of the plaintext. PII is
--      low-entropy (there are far fewer plausible email addresses than
--      2^256), so a bare SHA-256 column would be trivially reversible by
--      brute force and would hand an attacker exactly what the encryption
--      was protecting.
CREATE TABLE vault_entries (
    id                UUID PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,

    -- The token substituted into the text, e.g. `<PII:EMAIL_ADDRESS:1>` —
    -- the same shape armor_core::engine::redact already emits, so a caller
    -- that switches from redact-and-discard to vault-backed anonymization
    -- sees no change in the text format.
    placeholder       TEXT NOT NULL,
    category          TEXT NOT NULL,
    rule_id           TEXT NOT NULL,

    -- AES-256-GCM. The nonce is per-entry and random; it is not secret and
    -- is stored alongside, as the construction intends.
    value_ciphertext  BYTEA NOT NULL,
    value_nonce       BYTEA NOT NULL,

    -- HMAC-SHA256(key, plaintext) — the blind index described above. This
    -- is what makes "have I already minted a placeholder for this value in
    -- this session?" a single indexed lookup instead of decrypting every
    -- row in the session.
    value_index       BYTEA NOT NULL,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at        TIMESTAMPTZ
);

-- Stable placeholder reuse: the same value seen twice in one
-- session resolves to the existing placeholder rather than minting a second
-- one. Enforced as a constraint, not just a SELECT-then-INSERT in
-- application code, so two concurrent requests in the same session cannot
-- race into two placeholders for one value.
CREATE UNIQUE INDEX vault_entries_session_value_idx
    ON vault_entries (session_id, value_index);

-- deanonymize's lookup path, and the guarantee that one placeholder means
-- one value within a session.
CREATE UNIQUE INDEX vault_entries_session_placeholder_idx
    ON vault_entries (session_id, placeholder);

CREATE INDEX vault_entries_expires_at_idx ON vault_entries (expires_at)
    WHERE expires_at IS NOT NULL;

-- Placeholder ordinals (`<PII:EMAIL_ADDRESS:1>`, `...:2>`), allocated
-- atomically per (session, category, rule).
--
-- The obvious implementation — `SELECT COUNT(*) + 1` then INSERT — is
-- wrong under concurrency, and not in a subtle way: at READ COMMITTED,
-- every request vaulting a *different* value in the same session reads the
-- same count and proposes the same ordinal, so all but one hit the unique
-- index above. Retrying converges one winner per round, which for N
-- concurrent values is N rounds of contention.
--
-- A counter row instead makes allocation a single atomic increment under a
-- row lock, so N concurrent callers get N distinct ordinals in one pass
-- with no retry path to get wrong. Ordinals are consumed on mint, not on
-- reuse, and a lost race can leave a gap — placeholders have to be
-- distinct, not gapless.
CREATE TABLE vault_placeholder_sequences (
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    category    TEXT NOT NULL,
    rule_id     TEXT NOT NULL,
    next_index  BIGINT NOT NULL DEFAULT 1,
    PRIMARY KEY (session_id, category, rule_id)
);
