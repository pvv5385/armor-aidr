//! Reversible anonymization (the shape `llm-guard`'s `Vault` uses):
//! detected PII is replaced with a stable, session-scoped
//! placeholder; a paired deanonymize step restores the original for a
//! trusted downstream consumer — the tool call that actually needs the real
//! email address, or the authenticated end user viewing their own data
//! back.
//!
//! This is a materially different capability from
//! `armor_core::engine::redact`, which is redact-and-discard: it masks
//! spans in a copy of the text and the original is gone. Here the mapping
//! is kept, so the transformation is invertible.
//!
//! # Threat model, and what "encrypted" means here
//!
//! Storing recoverable PII is what makes this the one part of the storage
//! layer that can fail an enterprise security review on its own. Three
//! decisions define what this module actually defends against:
//!
//! 1. **Values are encrypted in this process, not by the database.**
//!    AES-256-GCM, key from a [`KeyProvider`], ciphertext and a per-entry
//!    random nonce written to Postgres. The key never appears in a query
//!    string, so it cannot leak through `pg_stat_statements`, a slow-query
//!    log, or a query audit trail — which is the specific failure mode that
//!    ruled out doing this with `pgcrypto`. An attacker with a database
//!    dump, a stolen backup, or read access to the volume gets ciphertext.
//! 2. **Value lookup uses a keyed blind index, not a hash.** The same
//!    value must map to the same placeholder within a session, which means
//!    asking "have I seen this value before?" without
//!    decrypting the session's rows. A plain `SHA-256(value)` column would
//!    answer that *and* hand an attacker a trivially brute-forcible index:
//!    PII is low-entropy, and enumerating plausible email addresses or
//!    phone numbers against a bare hash is cheap. So the index is
//!    `HMAC-SHA256(key, value)` — unforgeable without the key, and
//!    therefore useless in a dump that doesn't include it.
//! 3. **Nothing here is reachable over HTTP.** There is no deanonymize
//!    endpoint, deliberately. No RBAC model exists yet to gate who is
//!    allowed to call deanonymize; shipping an endpoint first and
//!    authorization later would mean a window where the vault is a PII
//!    disclosure API. Callers that need it link this crate.
//!
//! What this does **not** defend against, stated plainly: an attacker who
//! compromises the running process, or who obtains both the database and
//! `ARMOR_VAULT_KEY`, recovers the plaintext. Splitting those two is the
//! entire point of the [`KeyProvider`] indirection — a KMS-backed
//! implementation keeps the key material outside the deployment, and drops
//! in without a schema change.

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use base64::Engine as _;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use rand::RngCore;
use sha2::Sha256;
use sqlx::postgres::PgPool;
use uuid::Uuid;
use zeroize::Zeroizing;

type HmacSha256 = Hmac<Sha256>;

/// AES-256-GCM's nonce width. Random per entry, stored alongside the
/// ciphertext — a nonce is not secret, it only has to be unique per key.
const NONCE_LEN: usize = 12;

#[derive(Debug, thiserror::Error)]
pub enum VaultError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("vault key unavailable: {0}")]
    Key(String),
    /// Deliberately opaque. A decrypt failure means the ciphertext was
    /// tampered with, truncated, or encrypted under a different key — and
    /// the caller can act on none of those differently, while a detailed
    /// message is a padding-oracle-shaped gift to anyone probing.
    #[error("decryption failed")]
    Decrypt,
    #[error("encryption failed")]
    Encrypt,
    #[error("stored value is not valid UTF-8")]
    Corrupt,
}

/// Where the vault's 256-bit symmetric key comes from.
///
/// The indirection exists so key custody can change without touching the
/// schema or the query layer: [`EnvKeyProvider`] is the self-hosted
/// default, and a KMS/HSM-backed implementation (fetching a data key,
/// caching it, rotating on a schedule) implements the same trait.
///
/// Returns [`Zeroizing`] so the key is wiped from memory on drop rather
/// than lingering in a freed allocation.
pub trait KeyProvider: Send + Sync {
    fn key(&self) -> Result<Zeroizing<[u8; 32]>, VaultError>;
}

/// Reads the key from an environment variable (default `ARMOR_VAULT_KEY`)
/// as standard base64 of exactly 32 bytes.
///
/// The key is read once at construction and held, rather than re-read per
/// call: a mid-flight key change would silently make every existing row
/// undecryptable, so it should be a restart (and, eventually, a real
/// rotation path that re-encrypts), not a value that can drift underneath a
/// running process.
pub struct EnvKeyProvider {
    key: Zeroizing<[u8; 32]>,
}

impl EnvKeyProvider {
    pub const DEFAULT_VAR: &'static str = "ARMOR_VAULT_KEY";

    pub fn from_env() -> Result<Self, VaultError> {
        Self::from_env_var(Self::DEFAULT_VAR)
    }

    pub fn from_env_var(var: &str) -> Result<Self, VaultError> {
        let raw = std::env::var(var).map_err(|_| VaultError::Key(format!("{var} is not set")))?;
        Self::from_base64(&raw)
    }

    pub fn from_base64(encoded: &str) -> Result<Self, VaultError> {
        let decoded = Zeroizing::new(
            base64::engine::general_purpose::STANDARD
                .decode(encoded.trim())
                .map_err(|_| VaultError::Key("not valid base64".into()))?,
        );
        let key: [u8; 32] = decoded.as_slice().try_into().map_err(|_| {
            VaultError::Key(format!(
                "expected 32 bytes of key material, got {}",
                decoded.len()
            ))
        })?;
        Ok(Self {
            key: Zeroizing::new(key),
        })
    }

    /// Generates a fresh key and returns it base64-encoded, ready to be put
    /// in `ARMOR_VAULT_KEY`. For operator tooling and tests — never called
    /// implicitly, because a vault that silently generates its own key on
    /// boot would produce a different key per replica and per restart, and
    /// every stored value would become undecryptable the moment it was
    /// written by a process other than the one reading it.
    pub fn generate_base64() -> String {
        let mut key = Zeroizing::new([0u8; 32]);
        OsRng.fill_bytes(key.as_mut());
        base64::engine::general_purpose::STANDARD.encode(key.as_ref())
    }
}

impl KeyProvider for EnvKeyProvider {
    fn key(&self) -> Result<Zeroizing<[u8; 32]>, VaultError> {
        Ok(self.key.clone())
    }
}

/// One stored mapping, as returned by [`Vault::anonymize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaultedValue {
    /// The token to substitute into the text, e.g.
    /// `<PII:EMAIL_ADDRESS:1>` — same shape
    /// `armor_core::engine::redact` already emits.
    pub placeholder: String,
    /// False when this value already had a placeholder in this session and
    /// the existing one was reused — the observable form of this module's
    /// placeholder-stability requirement.
    pub minted: bool,
}

/// The value being vaulted, plus the labels its placeholder is built from.
#[derive(Debug, Clone, Copy)]
pub struct NewSecret<'a> {
    pub value: &'a str,
    /// Detector category, e.g. `pii`.
    pub category: &'a str,
    /// Rule that matched, e.g. `EMAIL_ADDRESS`.
    pub rule_id: &'a str,
}

pub struct Vault {
    pool: PgPool,
    keys: Box<dyn KeyProvider>,
    /// Retention for individual entries. `None` means entries live until
    /// their session is purged or erased (`ON DELETE CASCADE`).
    ttl_seconds: Option<i64>,
}

impl Vault {
    pub fn new(pool: PgPool, keys: Box<dyn KeyProvider>) -> Self {
        Self {
            pool,
            keys,
            ttl_seconds: None,
        }
    }

    /// Sets a per-entry retention window. Independent of the session TTL:
    /// an entry can expire while its session is still active, which is what
    /// a short PII-retention policy on a long-running conversation needs.
    pub fn with_ttl_seconds(mut self, ttl_seconds: Option<i64>) -> Self {
        self.ttl_seconds = ttl_seconds;
        self
    }

    /// Store a value and return its placeholder, reusing the existing
    /// placeholder if this session has already vaulted this exact value.
    ///
    /// The session row must already exist (`sessions::touch` creates it);
    /// the foreign key means an unknown session is a database error rather
    /// than an orphaned vault entry.
    ///
    /// Concurrency, in two parts, because they fail differently:
    ///
    /// - **Same value, twice.** Resolved by the unique index on
    ///   `(session_id, value_index)` inside the `INSERT ... ON CONFLICT DO
    ///   UPDATE ... RETURNING` below, not by a `SELECT`-then-`INSERT` here.
    ///   Two concurrent requests vaulting the same value agree on one
    ///   placeholder; check-then-act would mint two. `DO UPDATE SET
    ///   placeholder = vault_entries.placeholder` is a deliberate no-op
    ///   write — `DO NOTHING` returns no row on conflict, which would force
    ///   a second round trip to read the existing placeholder back.
    /// - **Different values, at once.** Resolved by allocating the
    ///   placeholder ordinal from `vault_placeholder_sequences`, an atomic
    ///   per-`(session, category, rule)` increment, rather than from
    ///   `COUNT(*)`. See that table's comment in
    ///   `migrations/0003_sessions_and_vault.sql` for why the counting
    ///   version is wrong at READ COMMITTED.
    ///
    /// The fast path for an already-vaulted value is a plain indexed
    /// lookup, so repeated mentions of the same value in a long
    /// conversation neither burn ordinals nor write.
    pub async fn anonymize(
        &self,
        session_id: &str,
        secret: NewSecret<'_>,
    ) -> Result<VaultedValue, VaultError> {
        let key = self.keys.key()?;
        let value_index = blind_index(&key, session_id, secret.value);

        if let Some(placeholder) = sqlx::query_scalar::<_, String>(
            "SELECT placeholder FROM vault_entries \
             WHERE session_id = $1 AND value_index = $2",
        )
        .bind(session_id)
        .bind(&value_index)
        .fetch_optional(&self.pool)
        .await?
        {
            return Ok(VaultedValue {
                placeholder,
                minted: false,
            });
        }

        let next_index: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO vault_placeholder_sequences (session_id, category, rule_id, next_index)
            VALUES ($1, $2, $3, 1)
            ON CONFLICT (session_id, category, rule_id)
                DO UPDATE SET next_index = vault_placeholder_sequences.next_index + 1
            RETURNING next_index
            "#,
        )
        .bind(session_id)
        .bind(secret.category)
        .bind(secret.rule_id)
        .fetch_one(&self.pool)
        .await?;

        // Built through `armor_core`'s own labeller, not a local `format!`,
        // because these placeholders have to be byte-identical to the ones
        // `engine::redact` splices into the text — a vaulted placeholder
        // that doesn't match what the caller sees is one nobody can look up.
        // The stored `rule_id` column below stays the raw detector value;
        // only the rendered placeholder is normalized.
        let placeholder = format!(
            "<{}:{}>",
            armor_core::engine::redact::placeholder_label(secret.category, secret.rule_id),
            next_index
        );
        let (ciphertext, nonce) = encrypt(&key, secret.value)?;

        // `xmax = 0` is how Postgres distinguishes a row this statement
        // actually inserted from one `ON CONFLICT DO UPDATE` touched: a
        // freshly inserted tuple has no updating transaction. Comparing the
        // returned placeholder against the one proposed above would be
        // wrong — a caller that lost the same-value race gets the winner's
        // placeholder back and must report `minted: false`.
        let (placeholder, minted): (String, bool) = sqlx::query_as(
            r#"
            INSERT INTO vault_entries
                (id, session_id, placeholder, category, rule_id,
                 value_ciphertext, value_nonce, value_index, expires_at)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8,
                 CASE WHEN $9::bigint IS NULL THEN NULL
                      ELSE now() + make_interval(secs => $9::bigint)
                 END)
            ON CONFLICT (session_id, value_index)
                DO UPDATE SET placeholder = vault_entries.placeholder
            RETURNING placeholder, (xmax = 0) AS minted
            "#,
        )
        .bind(Uuid::now_v7())
        .bind(session_id)
        .bind(&placeholder)
        .bind(secret.category)
        .bind(secret.rule_id)
        .bind(&ciphertext)
        .bind(&nonce)
        .bind(&value_index)
        .bind(self.ttl_seconds)
        .fetch_one(&self.pool)
        .await?;

        Ok(VaultedValue {
            placeholder,
            minted,
        })
    }

    /// Recover the original value behind a placeholder, or `None` if this
    /// session never minted it (or it has since been purged or erased).
    ///
    /// Scoped to `session_id` by the query, not by the caller: a
    /// placeholder from one session is not resolvable from another even
    /// though placeholder strings repeat across sessions by design.
    pub async fn deanonymize(
        &self,
        session_id: &str,
        placeholder: &str,
    ) -> Result<Option<String>, VaultError> {
        let row: Option<(Vec<u8>, Vec<u8>)> = sqlx::query_as(
            "SELECT value_ciphertext, value_nonce FROM vault_entries \
             WHERE session_id = $1 AND placeholder = $2",
        )
        .bind(session_id)
        .bind(placeholder)
        .fetch_optional(&self.pool)
        .await?;

        let Some((ciphertext, nonce)) = row else {
            return Ok(None);
        };

        let key = self.keys.key()?;
        decrypt(&key, &ciphertext, &nonce).map(Some)
    }

    /// Every placeholder held for a session, newest last. The read side of
    /// a "what do you still hold about me?" request, and what an operator
    /// inspects before erasing. Returns placeholders and labels only —
    /// never plaintext, so this is safe to log or display where
    /// [`Self::deanonymize`] would not be.
    pub async fn list_placeholders(
        &self,
        session_id: &str,
    ) -> Result<Vec<StoredPlaceholder>, VaultError> {
        let rows = sqlx::query_as::<_, StoredPlaceholder>(
            "SELECT placeholder, category, rule_id, created_at, expires_at \
             FROM vault_entries WHERE session_id = $1 ORDER BY created_at, placeholder",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Delete entries whose own `expires_at` has passed, returning how many
    /// went. Narrower than [`crate::sessions::purge_expired`], which takes
    /// whole sessions (and their entries) once the *session* expires.
    pub async fn purge_expired(&self, now: Option<DateTime<Utc>>) -> Result<u64, VaultError> {
        let result = sqlx::query(
            "DELETE FROM vault_entries \
             WHERE expires_at IS NOT NULL AND expires_at <= COALESCE($1::timestamptz, now())",
        )
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Right-to-erasure for one session's stored PII, leaving the session's
    /// counters intact. Use [`crate::sessions::erase`] to drop both.
    pub async fn erase_session(&self, session_id: &str) -> Result<u64, VaultError> {
        let result = sqlx::query("DELETE FROM vault_entries WHERE session_id = $1")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }
}

/// A placeholder's metadata, without the value behind it.
#[derive(Debug, Clone, serde::Serialize, sqlx::FromRow)]
pub struct StoredPlaceholder {
    pub placeholder: String,
    pub category: String,
    pub rule_id: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// `HMAC-SHA256(key, session_id || 0x00 || value)` — see this module's doc
/// comment on why this is keyed rather than a bare digest.
///
/// `session_id` is folded in as a per-session salt so the *same* PII value
/// vaulted in two different sessions produces two unrelated indices. Without
/// it, `value_index` depended only on `value`, so identical bytes landed in
/// `vault_entries` for every session that ever vaulted that value — a
/// passive attacker with read access to the table (a DB dump, a replica,
/// `pg_stat_statements`) could cluster sessions by shared `value_index`
/// without ever having the vault key or decrypting anything, learning "these
/// N otherwise-unrelated sessions mention the same secret" for free. Scoping
/// the index to `session_id` closes that: two sessions never share an index
/// for the same value, even though within *one* session the index (and so
/// the placeholder) still stays stable across repeated mentions — the
/// `0x00` separator prevents `(session_id="ab", value="c")` and
/// `(session_id="a", value="bc")` from hashing identically.
fn blind_index(key: &[u8; 32], session_id: &str, value: &str) -> Vec<u8> {
    // Fully qualified: `KeyInit` (pulled in for AES-GCM) and `Mac` both
    // offer a `new_from_slice`, so the bare call is ambiguous here.
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(session_id.as_bytes());
    mac.update(&[0]);
    mac.update(value.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

fn encrypt(key: &[u8; 32], plaintext: &str) -> Result<(Vec<u8>, Vec<u8>), VaultError> {
    let cipher = Aes256Gcm::new(key.into());
    let mut nonce_bytes = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_bytes())
        .map_err(|_| VaultError::Encrypt)?;
    Ok((ciphertext, nonce_bytes.to_vec()))
}

fn decrypt(key: &[u8; 32], ciphertext: &[u8], nonce: &[u8]) -> Result<String, VaultError> {
    if nonce.len() != NONCE_LEN {
        return Err(VaultError::Decrypt);
    }
    let cipher = Aes256Gcm::new(key.into());
    let plaintext = cipher
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|_| VaultError::Decrypt)?;
    String::from_utf8(plaintext).map_err(|_| VaultError::Corrupt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sessions::{self, Touch};
    use crate::test_support::{test_pool, unique_id};

    fn test_keys() -> Box<dyn KeyProvider> {
        Box::new(EnvKeyProvider::from_base64(&EnvKeyProvider::generate_base64()).unwrap())
    }

    async fn session_in(pool: &PgPool) -> String {
        let id = unique_id("vault-sess");
        sessions::touch(
            pool,
            Touch {
                session_id: &id,
                estimated_tokens: 0,
                window_seconds: 60.0,
                ttl_seconds: None,
                now: None,
            },
        )
        .await
        .expect("create session");
        id
    }

    // --- Pure crypto, no database ---

    #[test]
    fn encrypt_decrypt_round_trips() {
        let key = [7u8; 32];
        let (ciphertext, nonce) = encrypt(&key, "ada@example.com").unwrap();
        assert_ne!(ciphertext, b"ada@example.com");
        assert_eq!(
            decrypt(&key, &ciphertext, &nonce).unwrap(),
            "ada@example.com"
        );
    }

    #[test]
    fn the_same_plaintext_encrypts_differently_every_time() {
        // Random per-entry nonce: two rows holding the same value must not
        // be recognizable as equal from the ciphertext alone.
        let key = [7u8; 32];
        let (a, _) = encrypt(&key, "ada@example.com").unwrap();
        let (b, _) = encrypt(&key, "ada@example.com").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn decrypting_under_the_wrong_key_fails_rather_than_returning_garbage() {
        let (ciphertext, nonce) = encrypt(&[7u8; 32], "ada@example.com").unwrap();
        assert!(matches!(
            decrypt(&[8u8; 32], &ciphertext, &nonce),
            Err(VaultError::Decrypt)
        ));
    }

    #[test]
    fn tampered_ciphertext_is_rejected() {
        // The authentication half of AEAD: a flipped bit must fail, not
        // decrypt to a different plaintext.
        let key = [7u8; 32];
        let (mut ciphertext, nonce) = encrypt(&key, "ada@example.com").unwrap();
        ciphertext[0] ^= 0x01;
        assert!(matches!(
            decrypt(&key, &ciphertext, &nonce),
            Err(VaultError::Decrypt)
        ));
    }

    #[test]
    fn blind_index_is_stable_per_key_and_differs_across_keys() {
        let a = blind_index(&[1u8; 32], "session-1", "ada@example.com");
        let b = blind_index(&[1u8; 32], "session-1", "ada@example.com");
        let c = blind_index(&[2u8; 32], "session-1", "ada@example.com");
        let d = blind_index(&[1u8; 32], "session-1", "grace@example.com");
        assert_eq!(
            a, b,
            "same key + same session + same value must collide, that is the point"
        );
        assert_ne!(a, c, "a different key must produce a different index");
        assert_ne!(a, d);
    }

    #[test]
    fn blind_index_differs_across_sessions_for_the_same_value() {
        // The whole point of salting by session_id: a DB-dump attacker with
        // no vault key must not be able to tell that two different sessions
        // vaulted the same underlying value just by comparing `value_index`
        // bytes.
        let a = blind_index(&[1u8; 32], "session-a", "ada@example.com");
        let b = blind_index(&[1u8; 32], "session-b", "ada@example.com");
        assert_ne!(a, b);
    }

    #[test]
    fn blind_index_separator_prevents_session_value_boundary_collision() {
        // Without a separator between session_id and value, ("ab", "c") and
        // ("a", "bc") would hash identically.
        let a = blind_index(&[1u8; 32], "ab", "c");
        let b = blind_index(&[1u8; 32], "a", "bc");
        assert_ne!(a, b);
    }

    #[test]
    fn key_provider_rejects_wrong_length_and_bad_base64() {
        assert!(matches!(
            EnvKeyProvider::from_base64("c2hvcnQ="), // "short", 5 bytes
            Err(VaultError::Key(_))
        ));
        assert!(matches!(
            EnvKeyProvider::from_base64("not base64!!"),
            Err(VaultError::Key(_))
        ));
        assert!(EnvKeyProvider::from_base64(&EnvKeyProvider::generate_base64()).is_ok());
    }

    #[test]
    fn generated_keys_are_distinct() {
        assert_ne!(
            EnvKeyProvider::generate_base64(),
            EnvKeyProvider::generate_base64()
        );
    }

    // --- Against a real database ---

    #[tokio::test]
    async fn anonymize_then_deanonymize_round_trips() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;
        let vault = Vault::new(pool.clone(), test_keys());

        let stored = vault
            .anonymize(
                &session,
                NewSecret {
                    value: "ada@example.com",
                    category: "pii",
                    rule_id: "email_address",
                },
            )
            .await
            .unwrap();

        assert!(stored.minted);
        assert_eq!(stored.placeholder, "<PII:EMAIL_ADDRESS:1>");
        assert_eq!(
            vault
                .deanonymize(&session, &stored.placeholder)
                .await
                .unwrap(),
            Some("ada@example.com".to_string())
        );
    }

    #[tokio::test]
    async fn the_same_value_reuses_its_placeholder_within_a_session() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;
        let vault = Vault::new(pool.clone(), test_keys());
        let secret = NewSecret {
            value: "ada@example.com",
            category: "pii",
            rule_id: "email_address",
        };

        let first = vault.anonymize(&session, secret).await.unwrap();
        let second = vault.anonymize(&session, secret).await.unwrap();

        assert!(first.minted);
        assert!(!second.minted, "the second call must reuse, not mint");
        assert_eq!(first.placeholder, second.placeholder);
        assert_eq!(vault.list_placeholders(&session).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn distinct_values_get_distinct_numbered_placeholders() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;
        let vault = Vault::new(pool.clone(), test_keys());
        let secret = |value| NewSecret {
            value,
            category: "pii",
            rule_id: "email_address",
        };

        let first = vault
            .anonymize(&session, secret("ada@example.com"))
            .await
            .unwrap();
        let second = vault
            .anonymize(&session, secret("grace@example.com"))
            .await
            .unwrap();

        assert_eq!(first.placeholder, "<PII:EMAIL_ADDRESS:1>");
        assert_eq!(second.placeholder, "<PII:EMAIL_ADDRESS:2>");
        assert_eq!(
            vault
                .deanonymize(&session, &second.placeholder)
                .await
                .unwrap(),
            Some("grace@example.com".to_string())
        );
    }

    #[tokio::test]
    async fn placeholders_are_scoped_to_their_session() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let (a, b) = (session_in(&pool).await, session_in(&pool).await);
        let vault = Vault::new(pool.clone(), test_keys());
        let secret = NewSecret {
            value: "ada@example.com",
            category: "pii",
            rule_id: "email_address",
        };

        let in_a = vault.anonymize(&a, secret).await.unwrap();
        let in_b = vault.anonymize(&b, secret).await.unwrap();

        // Same placeholder *string* in both sessions — they're numbered
        // per session — but each only resolves inside its own.
        assert_eq!(in_a.placeholder, in_b.placeholder);
        assert!(in_b.minted, "a second session must mint its own entry");
        assert!(vault
            .deanonymize(&a, &in_a.placeholder)
            .await
            .unwrap()
            .is_some());
        assert!(vault
            .deanonymize(&a, "<PII:EMAIL_ADDRESS:99>")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn concurrent_anonymize_of_one_value_yields_one_placeholder() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;

        // The unique index on (session_id, value_index) is what makes this
        // safe; a SELECT-then-INSERT would mint duplicates here.
        let mut handles = Vec::new();
        for _ in 0..16 {
            let pool = pool.clone();
            let session = session.clone();
            handles.push(tokio::spawn(async move {
                Vault::new(pool, test_keys_fixed())
                    .anonymize(
                        &session,
                        NewSecret {
                            value: "ada@example.com",
                            category: "pii",
                            rule_id: "email_address",
                        },
                    )
                    .await
                    .expect("concurrent anonymize")
            }));
        }

        // A `HashSet`, not `Vec::dedup` — dedup only collapses *consecutive*
        // duplicates, so it would pass on interleaved distinct placeholders,
        // which is exactly the failure this test exists to catch.
        let mut placeholders = std::collections::HashSet::new();
        let mut minted_count = 0;
        for handle in handles {
            let stored = handle.await.unwrap();
            minted_count += usize::from(stored.minted);
            placeholders.insert(stored.placeholder);
        }
        assert_eq!(placeholders.len(), 1, "one value must mean one placeholder");
        assert_eq!(minted_count, 1, "exactly one caller mints; the rest reuse");

        let vault = Vault::new(pool.clone(), test_keys_fixed());
        assert_eq!(vault.list_placeholders(&session).await.unwrap().len(), 1);
    }

    /// Concurrency test needs every task to share one key, or each would
    /// compute a different blind index and the unique constraint would
    /// never fire.
    fn test_keys_fixed() -> Box<dyn KeyProvider> {
        Box::new(
            EnvKeyProvider::from_base64(
                &base64::engine::general_purpose::STANDARD.encode([9u8; 32]),
            )
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn concurrent_anonymize_of_distinct_values_does_not_collide_on_an_ordinal() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;

        // Distinct values race the *placeholder* index rather than the
        // value index: each proposes `<PII:EMAIL_ADDRESS:1>` from the same
        // COUNT(*), and only one can have it. Without the retry in
        // `anonymize`, the losers surface a raw unique violation.
        let mut handles = Vec::new();
        for i in 0..12 {
            let pool = pool.clone();
            let session = session.clone();
            handles.push(tokio::spawn(async move {
                Vault::new(pool, test_keys_fixed())
                    .anonymize(
                        &session,
                        NewSecret {
                            value: &format!("user{i}@example.com"),
                            category: "pii",
                            rule_id: "email_address",
                        },
                    )
                    .await
                    .expect("distinct-value anonymize must not fail on ordinal contention")
            }));
        }

        let mut placeholders = std::collections::HashSet::new();
        for handle in handles {
            let stored = handle.await.unwrap();
            assert!(stored.minted, "every distinct value mints its own entry");
            placeholders.insert(stored.placeholder);
        }
        assert_eq!(
            placeholders.len(),
            12,
            "12 distinct values must get 12 distinct placeholders"
        );

        // And every one of them still resolves to the right plaintext.
        let vault = Vault::new(pool.clone(), test_keys_fixed());
        let mut recovered = std::collections::HashSet::new();
        for stored in vault.list_placeholders(&session).await.unwrap() {
            recovered.insert(
                vault
                    .deanonymize(&session, &stored.placeholder)
                    .await
                    .unwrap()
                    .expect("every listed placeholder resolves"),
            );
        }
        assert_eq!(recovered.len(), 12);
    }

    #[tokio::test]
    async fn a_wrong_key_cannot_read_an_existing_entry() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;
        let stored = Vault::new(pool.clone(), test_keys_fixed())
            .anonymize(
                &session,
                NewSecret {
                    value: "ada@example.com",
                    category: "pii",
                    rule_id: "email_address",
                },
            )
            .await
            .unwrap();

        // Same row, different key — this is the database-dump scenario.
        let other = Vault::new(pool.clone(), test_keys());
        assert!(matches!(
            other.deanonymize(&session, &stored.placeholder).await,
            Err(VaultError::Decrypt)
        ));
    }

    #[tokio::test]
    async fn ciphertext_in_the_database_does_not_contain_the_plaintext() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;
        Vault::new(pool.clone(), test_keys())
            .anonymize(
                &session,
                NewSecret {
                    value: "ada@example.com",
                    category: "pii",
                    rule_id: "email_address",
                },
            )
            .await
            .unwrap();

        let (ciphertext, index): (Vec<u8>, Vec<u8>) = sqlx::query_as(
            "SELECT value_ciphertext, value_index FROM vault_entries WHERE session_id = $1",
        )
        .bind(&session)
        .fetch_one(&pool)
        .await
        .unwrap();

        let needle = b"ada@example.com";
        assert!(!ciphertext.windows(needle.len()).any(|w| w == needle));
        assert!(!index.windows(needle.len()).any(|w| w == needle));
    }

    #[tokio::test]
    async fn erasing_a_session_drops_its_vault_entries() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;
        let vault = Vault::new(pool.clone(), test_keys());
        let stored = vault
            .anonymize(
                &session,
                NewSecret {
                    value: "ada@example.com",
                    category: "pii",
                    rule_id: "email_address",
                },
            )
            .await
            .unwrap();

        // Right-to-erasure through the session: ON DELETE CASCADE has to
        // take the PII with it, or erasure leaves recoverable data behind.
        assert!(sessions::erase(&pool, &session).await.unwrap());
        assert!(vault
            .deanonymize(&session, &stored.placeholder)
            .await
            .unwrap()
            .is_none());
        assert!(vault.list_placeholders(&session).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn entry_ttl_expires_independently_of_the_session() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;
        let vault = Vault::new(pool.clone(), test_keys()).with_ttl_seconds(Some(30));
        let stored = vault
            .anonymize(
                &session,
                NewSecret {
                    value: "ada@example.com",
                    category: "pii",
                    rule_id: "email_address",
                },
            )
            .await
            .unwrap();

        // Purging is table-wide, so its return count also covers rows other
        // tests (and earlier runs against the same scratch database) left
        // behind. Assert on *this* entry's fate instead — a count assertion
        // here is a flake waiting to happen.
        let purged = vault
            .purge_expired(Some(Utc::now() + chrono::Duration::seconds(60)))
            .await
            .unwrap();
        assert!(purged >= 1);
        assert!(vault
            .deanonymize(&session, &stored.placeholder)
            .await
            .unwrap()
            .is_none());
        // The session itself is untouched.
        assert!(sessions::get(&pool, &session).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn entries_without_a_ttl_are_never_purged() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = session_in(&pool).await;
        let vault = Vault::new(pool.clone(), test_keys());
        vault
            .anonymize(
                &session,
                NewSecret {
                    value: "ada@example.com",
                    category: "pii",
                    rule_id: "email_address",
                },
            )
            .await
            .unwrap();

        vault
            .purge_expired(Some(Utc::now() + chrono::Duration::days(3650)))
            .await
            .unwrap();
        assert_eq!(vault.list_placeholders(&session).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn anonymize_against_an_unknown_session_is_rejected() {
        let Some(pool) = test_pool().await else {
            return;
        };
        // The foreign key, not application code, is what stops an orphaned
        // vault entry from being created.
        let result = Vault::new(pool, test_keys())
            .anonymize(
                "session-that-was-never-created",
                NewSecret {
                    value: "ada@example.com",
                    category: "pii",
                    rule_id: "email_address",
                },
            )
            .await;
        assert!(matches!(result, Err(VaultError::Db(_))));
    }
}
