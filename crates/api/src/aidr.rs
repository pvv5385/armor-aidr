//! AIDR scan request schema, shared by `POST /api/v1/aidr/scan` and every
//! gateway adapter under `/integrations/*/v1/aidr/scan`. One request shape
//! covers every stage (simple text, full `messages` including
//! `tool_calls`, agent state); this module builds a single text blob from
//! whichever fields a caller populated, then runs it through the
//! orchestrator and the shared audit/telemetry pipeline — no per-vendor
//! branching past this point.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use armor_core::{
    detectors,
    engine::decision::Decision,
    engine::orchestrator,
    models::{CheckAction, EnforcementMode, Severity, Verdict},
};

use crate::{
    audit::{CheckSummary, EvaluationEvent, LayerSummary},
    redaction, session_state,
    state::AppState,
};

/// One message in an OpenAI-shaped `messages` array — `content` may be a
/// plain string or a content-part array (multimodal), and an assistant
/// message may carry `tool_calls` alongside or instead of `content`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Message {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Vec<Value>,
}

/// Root-level `metadata` object: routing, policy-matching, and telemetry
/// data, kept separate from the content being scanned (`text`/`messages`).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Metadata {
    /// Which point in the request lifecycle this is, e.g.
    /// `input`/`output`/`tool`/`retrieval`/`agent-plan`/`agent-step`/
    /// `memory-write`. Accepted permissively and forwarded into the audit
    /// event/traces rather than validated against a fixed enum. Defaults
    /// to `"input"` when absent.
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub application_id: Option<String>,
    /// Caller-supplied correlation id — echoed back as
    /// `ScanResponse.request_id` and stored on the audit trail as
    /// `client_request_id`. Validated like the `X-Armor-Session-Id` header
    /// (`routes::resolve_session_id`): 1-128 visible-ASCII bytes. Distinct
    /// from `scan_id`, which is Armor's own id and always present.
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    /// Agent plan/state context, e.g. `current_plan`/`proposed_action`/
    /// `authorization_level`. Untyped since its shape varies by
    /// deployment; folded into the scanned text so pattern-based detectors
    /// (e.g. `excessive_agency`) can see it, same treatment as
    /// `tool_calls`.
    #[serde(default)]
    pub agent_state: Option<Value>,
    /// Anything else the caller sends — never scanned, but preserved
    /// through `#[serde(flatten)]` rather than silently rejected.
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, Value>,
}

/// Request envelope for `POST /api/v1/aidr/scan` and every
/// `/integrations/*/v1/aidr/scan` adapter (after vendor-specific
/// normalization). `text` alone covers the simple case; `messages` covers
/// full OpenAI-shaped conversations, including `tool_calls`, for the
/// advanced agent/memory checks. Same endpoint, same schema, richer
/// payload — no separate "advanced" route.
///
/// OpenAI chat-completions compatibility: a caller that forwards a raw
/// OpenAI-shaped request puts `request_id`, `application`, and `user_id` at
/// the *root* (alongside `messages`) instead of inside `metadata`. Those
/// three root fields are accepted and folded into their canonical
/// `metadata` counterparts by `normalize` (which `run_scan` calls). Root
/// fields are aliases for the metadata schema — `metadata` stays
/// authoritative when the same value is present in both places.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AidrScanRequest {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub metadata: Metadata,
    /// OpenAI-compat alias for `metadata.request_id`. Echoed back as
    /// `ScanResponse.request_id` once folded in by `normalize`.
    #[serde(default)]
    pub request_id: Option<String>,
    /// OpenAI-compat alias for `metadata.application_id` — which profile's
    /// checks run (`profiles.rs`). An unmapped value falls back to `default`.
    #[serde(default)]
    pub application: Option<String>,
    /// OpenAI-compat alias for `metadata.user_id`.
    #[serde(default)]
    pub user_id: Option<String>,
}

impl AidrScanRequest {
    /// Folds the OpenAI-compat root fields (`request_id`, `application`,
    /// `user_id`) into their canonical `metadata` counterparts, so the rest
    /// of the pipeline only ever reads `metadata`. `metadata` is
    /// authoritative: a root field is only consulted when the matching
    /// `metadata` field is absent, never overriding it.
    fn normalize(mut self) -> Self {
        if self.metadata.request_id.is_none() {
            self.metadata.request_id = self.request_id.take();
        }
        if self.metadata.application_id.is_none() {
            self.metadata.application_id = self.application.take();
        }
        if self.metadata.user_id.is_none() {
            self.metadata.user_id = self.user_id.take();
        }
        self
    }
}

fn content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" "),
        other => other.to_string(),
    }
}

/// Flattens `text`, every message's `content`/`tool_calls`, and
/// `metadata.agent_state` into one string for the engine to scan.
/// `armor-core` stays a flat "text in, checks out" engine, so this is
/// where the richer request shape collapses back down. Tool calls and
/// agent state are JSON-serialized rather than field-plucked so
/// pattern-based detectors (regex/keyword) can match against them without
/// `armor-core` needing to know their shape.
pub fn build_scan_text(request: &AidrScanRequest) -> String {
    let mut parts = Vec::new();
    if !request.text.is_empty() {
        parts.push(request.text.clone());
    }
    for message in &request.messages {
        if let Some(content) = &message.content {
            let text = content_to_text(content);
            if !text.is_empty() {
                parts.push(text);
            }
        }
        for tool_call in &message.tool_calls {
            parts.push(tool_call.to_string());
        }
    }
    if let Some(state) = &request.metadata.agent_state {
        parts.push(state.to_string());
    }
    parts.join("\n")
}

/// Result of a shared scan run.
pub struct ScanOutcome {
    pub decision: Decision,
    /// Wall-clock cost of this request's engine run — the deterministic
    /// sweep plus redaction (`redaction.rs`) — and the same value recorded
    /// on the audit trail.
    pub latency_ms: f64,
    /// Armor's own per-request id, minted once in `run_scan`. Always
    /// present.
    pub scan_id: String,
    /// The caller's own `metadata.request_id` (already validated), carried
    /// through so the response can echo it back. `None` when not supplied.
    pub client_request_id: Option<String>,
}

/// Public response body for `POST /api/v1/aidr/scan` and every
/// `/integrations/*/v1/aidr/scan` adapter that mirrors it (Portkey builds
/// its own vendor-specific `{"verdict": bool, ...}` shape instead, per its
/// own webhook contract). A trimmed view of the engine's internal
/// `Decision`/`CheckOutcome` (`armor_core::engine::decision`); `latency_ms`
/// is this request's own wall-clock engine cost, not the sum of per-check
/// latencies, since checks may run concurrently.
///
/// `scan_id`/`request_id` are two levels of identity: `request_id` is the
/// caller-supplied per-call id (absent if they didn't send one), `scan_id`
/// is Armor's own id for this evaluation and is always present. `scan_id`
/// is authoritative for correlating with the audit trail even when a
/// caller never sends a `request_id`.
#[derive(Debug, Clone, Serialize)]
pub struct ScanResponse {
    pub scan_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub verdict: Verdict,
    pub checks: Vec<ScanCheckResult>,
    pub redacted_text: String,
    pub latency_ms: f64,
}

/// One resolved profile check's result. Every enabled check is listed
/// (`checks.len()` == the profile's enabled check count) so a caller can
/// see the full sweep, not just what fired — `flagged`/`action_taken`
/// make clear which entries actually did anything.
#[derive(Debug, Clone, Serialize)]
pub struct ScanCheckResult {
    pub category: String,
    /// `true` when this check's `CheckOutcome::passed` was `false` — i.e.
    /// it found something, not merely that it's configured to.
    pub flagged: bool,
    pub action_taken: ActionTaken,
    /// Only present when `flagged`; a clean check has no severity to report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<Severity>,
    /// Hit count, not the hits themselves — `redacted_text` already carries
    /// the placeholder detail; this contract has never exposed raw spans.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hits: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// This check's own wall-clock cost (`CheckOutcome::latency_ms`), always
    /// present regardless of `flagged` — lets a caller (e.g. the control
    /// plane's chat tester) show a per-detector timing breakdown, not just
    /// the request-level `ScanResponse.latency_ms` total.
    pub latency_ms: f64,
}

/// What actually happened for a flagged check — the same `mode` + `action`
/// table `engine::decision::compose` composes the verdict from, read one
/// check at a time:
///
/// - `Blocked` — `Deny` on a `Block`-mode check.
/// - `Redacted` — `Redact` on a `Block`-mode check. Its spans are masked in
///   `redacted_text`, which the caller is expected to send instead of the
///   original.
/// - `Warned` — either action on a `Warn`-mode check: advisory, nothing was
///   enforced.
/// - `Logged` — everything non-enforcing: `Monitor`-mode at any action, and
///   the `Flag`/`Log` actions nothing produces yet.
///
/// An unflagged check is always `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ActionTaken {
    None,
    Blocked,
    Redacted,
    Warned,
    Logged,
}

fn build_checks(decision: &Decision) -> Vec<ScanCheckResult> {
    decision
        .outcomes
        .iter()
        .map(|o| {
            let flagged = !o.passed;
            let enforcing = matches!(o.action, CheckAction::Deny | CheckAction::Redact);
            let action_taken = match (flagged, enforcing, o.mode) {
                (false, _, _) => ActionTaken::None,
                (_, true, EnforcementMode::Block) if o.action == CheckAction::Deny => {
                    ActionTaken::Blocked
                }
                (_, true, EnforcementMode::Block) => ActionTaken::Redacted,
                (_, true, EnforcementMode::Warn) => ActionTaken::Warned,
                _ => ActionTaken::Logged,
            };
            ScanCheckResult {
                category: o.category.clone(),
                flagged,
                action_taken,
                severity: flagged.then_some(o.severity),
                hits: flagged.then_some(o.hits.len()),
                error: o.error.clone(),
                latency_ms: o.latency_ms,
            }
        })
        .collect()
}

impl From<&ScanOutcome> for ScanResponse {
    fn from(outcome: &ScanOutcome) -> Self {
        let decision = &outcome.decision;
        Self {
            scan_id: outcome.scan_id.clone(),
            request_id: outcome.client_request_id.clone(),
            verdict: decision.verdict,
            latency_ms: outcome.latency_ms,
            checks: build_checks(decision),
            redacted_text: decision.redacted_text.clone(),
        }
    }
}

/// Cap on `metadata.request_id`, mirroring `routes::MAX_SESSION_ID_LEN` —
/// it's client-supplied and otherwise an unbounded side channel.
const MAX_CLIENT_REQUEST_ID_LEN: usize = 128;

/// `metadata.request_id` is client-supplied, so it gets the same treatment
/// as the `X-Armor-Session-Id` header (`routes::resolve_session_id`):
/// non-empty, capped, visible ASCII only (`0x21..=0x7e` — no control
/// characters, no raw whitespace, nothing that isn't printable). A body
/// field rather than a header, so it can't lean on `HeaderValue`'s own
/// validation the way the session id does; this reimplements the same
/// bar directly.
fn validate_client_request_id(id: &str) -> Result<(), StatusCode> {
    if id.is_empty() || id.len() > MAX_CLIENT_REQUEST_ID_LEN {
        return Err(StatusCode::BAD_REQUEST);
    }
    if !id.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
        return Err(StatusCode::BAD_REQUEST);
    }
    Ok(())
}

/// Runs `request` through the orchestrator and records an audit/telemetry
/// event — shared by `/api/v1/aidr/scan` and every gateway adapter, so
/// every caller gets the same audit trail.
///
/// Errs with `500` only if the blocking worker thread itself panicked —
/// per-check panics are already caught by the orchestrator's `catch_unwind`
/// and turned into a failed check outcome, so this only fires for a panic
/// in the orchestrator's own plumbing (e.g. `build_views`/`compose`)
/// outside that per-check guard. Near-unreachable, but a 500 beats taking
/// the handler down with it. Also errs with `400` if `metadata.request_id`
/// fails `validate_client_request_id`.
pub async fn run_scan(
    state: &AppState,
    session_id: &str,
    request: AidrScanRequest,
) -> Result<ScanOutcome, StatusCode> {
    let request = request.normalize();
    let mode = request
        .metadata
        .mode
        .clone()
        .unwrap_or_else(|| "input".to_string());
    let application_id = request.metadata.application_id.clone();
    if let Some(id) = &request.metadata.request_id {
        validate_client_request_id(id)?;
    }
    let client_request_id = request.metadata.request_id.clone();
    // Minted once, up front, so the audit event and the response
    // (`ScanResponse.scan_id`) carry the exact same value.
    let scan_id = uuid::Uuid::new_v4().to_string();
    let text = build_scan_text(&request);

    // `application_id` resolves which profile's checks run — see
    // `profiles.rs`. An absent or unrecognized id falls back to the
    // default profile, so this is never a hard failure.
    // `load()` takes a reference-counted snapshot of the current resolver;
    // the snapshot stays alive for this request even if the sync task swaps
    // in a new resolver between now and when we use `policy`.
    let resolver = state.profiles.load();
    let policy = resolver.resolve(application_id.as_deref());
    let profile_id = policy.id.clone();

    // Durable session counters for `abuse`/`unbounded_consumption`, when
    // either is enabled and a database is configured — otherwise this
    // returns `policy` unchanged and costs nothing. See `session_state`'s
    // module doc for why this one database call is allowed on the scan
    // path.
    let policy = session_state::apply(state, policy, session_id, &text).await;

    let started = Instant::now();
    // The engine runs in two phases, with `armor-api` in between. Only
    // redaction sits in the seam today; the async escalation pass lands here,
    // which is why the sweep hands back the views it scanned rather than
    // dropping them.
    let escalation_policy = policy.clone();
    let scan_text = text.clone();
    let swept = tokio::task::spawn_blocking(move || {
        orchestrator::run_deterministic(
            &policy,
            &text,
            detectors::get_check,
            orchestrator::DEFAULT_CHECK_TIMEOUT,
            orchestrator::DEFAULT_GUARDRAIL_TIMEOUT,
        )
    })
    .await
    .map_err(|join_err| {
        tracing::error!(error = %join_err, "engine worker thread panicked");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let decision = match swept {
        Ok((views, mut outcomes)) => {
            // The async escalation pass: ask the inference sidecar about
            // checks whose deterministic risk band is gray-zone. No-op
            // when the tier is off (state.inference is None) or when the
            // policy has no strategies.
            crate::ml::escalate(state, &escalation_policy, &views, &mut outcomes).await;
            redaction::compose(state, session_id, &scan_text, outcomes).await
        }
        // The whole-run budget blew before any check finished; `fail_mode`
        // decides, and `redacted_text` is the input unchanged because
        // nothing is known to be worth masking.
        Err(exceeded) => exceeded.into_decision(&scan_text),
    };
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;

    // Metadata only — category names and pass/fail, never the request body
    // or a hit's matched span, so this is safe to ship to a log backend
    // outside the customer's own boundary. Shared by the tracing log below
    // and the telemetry/audit sinks (`state.telemetry`/`state.audit_sink`)
    // so this mapping happens exactly once per request.
    let fired_checks: Vec<&str> = decision
        .outcomes
        .iter()
        .filter(|o| !o.passed)
        .map(|o| o.category.as_str())
        .collect();
    tracing::info!(
        scan_id = %scan_id,
        client_request_id = ?client_request_id,
        mode = %mode,
        application_id = ?application_id,
        profile_id = %profile_id,
        verdict = ?decision.verdict,
        fired_checks = ?fired_checks,
        "evaluation completed"
    );

    let event = EvaluationEvent {
        scan_id: scan_id.clone(),
        session_id: session_id.to_string(),
        client_request_id: client_request_id.clone(),
        occurred_at_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        stage: mode,
        verdict: format!("{:?}", decision.verdict).to_uppercase(),
        checks: decision
            .outcomes
            .iter()
            .map(|o| CheckSummary {
                category: o.category.clone(),
                passed: o.passed,
                action: format!("{:?}", o.action).to_lowercase(),
                severity: format!("{:?}", o.severity).to_lowercase(),
            })
            .collect(),
        latency_ms,
        application_id: application_id.clone(),
        profile_id: Some(profile_id.clone()),
        layers: build_layer_summaries(&decision.outcomes),
        model_version: decision
            .outcomes
            .iter()
            .find_map(|o| o.model_version.clone()),
    };

    state.telemetry.emit(event.clone());
    state.heartbeat.record_evaluation();

    // Fire-and-forget: `AuditSink::record` is sync I/O (durable local
    // spool, and — when `DATABASE_URL` is set — a Postgres write, see
    // `audit.rs`'s `PgAuditSink`), so it runs on the blocking-pool thread
    // it documents needing rather than the async request path. Not
    // awaited, matching `audit.rs`'s documented failure semantics: the
    // verdict is already decided, there's nothing left to deny on a write
    // failure.
    let audit_sink = state.audit_sink.clone();
    let audit_event = event.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = audit_sink.record(&audit_event) {
            tracing::warn!(error = %e, "failed to record audit event");
        }
    });

    Ok(ScanOutcome {
        decision,
        latency_ms,
        scan_id,
        client_request_id,
    })
}

/// Build per-check layer summaries from outcomes that ran through escalation.
/// Returns `None` when no outcome has any layers (deterministic-only path).
fn build_layer_summaries(
    outcomes: &[armor_core::engine::decision::CheckOutcome],
) -> Option<Vec<LayerSummary>> {
    let mut all_layers = Vec::new();
    for outcome in outcomes {
        for layer in &outcome.layers {
            all_layers.push(LayerSummary {
                layer: format!("{:?}", layer.layer).to_lowercase(),
                passed: layer.passed,
                selected: layer.selected,
                model_version: layer.model_version.clone(),
                latency_ms: Some(layer.latency_ms),
                error: layer.error.clone(),
            });
        }
    }
    if all_layers.is_empty() {
        None
    } else {
        Some(all_layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_scanned_as_is() {
        let req = AidrScanRequest {
            text: "hello world".to_string(),
            ..Default::default()
        };
        assert_eq!(build_scan_text(&req), "hello world");
    }

    #[test]
    fn message_string_content_is_included() {
        let req = AidrScanRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: Some(Value::String("find me flights".to_string())),
                tool_calls: Vec::new(),
            }],
            ..Default::default()
        };
        assert_eq!(build_scan_text(&req), "find me flights");
    }

    #[test]
    fn message_content_parts_join_their_text() {
        let req = AidrScanRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: Some(serde_json::json!([
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"}
                ])),
                tool_calls: Vec::new(),
            }],
            ..Default::default()
        };
        assert_eq!(build_scan_text(&req), "part one part two");
    }

    #[test]
    fn tool_call_arguments_are_scanned() {
        let req = AidrScanRequest {
            messages: vec![Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: vec![serde_json::json!({
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "update_agent_memory",
                        "arguments": "{\"key\": \"user_identity\", \"value\": \"system_admin\"}"
                    }
                })],
            }],
            ..Default::default()
        };
        let text = build_scan_text(&req);
        assert!(text.contains("update_agent_memory"));
        assert!(text.contains("system_admin"));
    }

    #[test]
    fn agent_state_is_scanned() {
        let mut req = AidrScanRequest::default();
        req.metadata.agent_state = Some(serde_json::json!({
            "proposed_action": "DROP TABLE prod_users",
            "authorization_level": "read_only"
        }));
        let text = build_scan_text(&req);
        assert!(text.contains("DROP TABLE prod_users"));
    }

    #[test]
    fn empty_request_scans_as_empty_string() {
        assert_eq!(build_scan_text(&AidrScanRequest::default()), "");
    }

    #[test]
    fn openai_root_fields_fold_into_metadata() {
        let req: AidrScanRequest = serde_json::from_value(serde_json::json!({
            "request_id": "abc123",
            "application": "customer-support",
            "user_id": "user-123",
            "messages": [{"role": "user", "content": "My SSN is 123-45-6789."}],
        }))
        .unwrap();
        let req = req.normalize();
        assert_eq!(req.metadata.request_id.as_deref(), Some("abc123"));
        assert_eq!(
            req.metadata.application_id.as_deref(),
            Some("customer-support")
        );
        assert_eq!(req.metadata.user_id.as_deref(), Some("user-123"));
        assert!(build_scan_text(&req).contains("My SSN is 123-45-6789."));
    }

    #[test]
    fn metadata_wins_over_openai_root_fields() {
        let req: AidrScanRequest = serde_json::from_value(serde_json::json!({
            "request_id": "root-id",
            "application": "root-app",
            "user_id": "root-user",
            "metadata": {
                "request_id": "meta-id",
                "application_id": "meta-app",
                "user_id": "meta-user"
            }
        }))
        .unwrap();
        let req = req.normalize();
        assert_eq!(req.metadata.request_id.as_deref(), Some("meta-id"));
        assert_eq!(req.metadata.application_id.as_deref(), Some("meta-app"));
        assert_eq!(req.metadata.user_id.as_deref(), Some("meta-user"));
    }

    #[test]
    fn openai_example_payload_resolves_and_scans() {
        // The exact shape from the OpenAI-compat discussion: root-level
        // request/application/user ids plus a `messages` array.
        let req: AidrScanRequest = serde_json::from_value(serde_json::json!({
            "request_id": "abc123",
            "application": "customer-support",
            "user_id": "user-123",
            "metadata": { "tenant": "enterprise-a", "region": "us-east-1" },
            "messages": [
                { "role": "system", "content": "You are a banking assistant." },
                { "role": "user", "content": "My SSN is 123-45-6789." },
                { "role": "assistant", "content": "How can I help?" },
                { "role": "user", "content": "Transfer my money." }
            ]
        }))
        .unwrap();
        let req = req.normalize();
        assert_eq!(req.metadata.request_id.as_deref(), Some("abc123"));
        assert_eq!(
            req.metadata.application_id.as_deref(),
            Some("customer-support")
        );
        assert_eq!(req.metadata.user_id.as_deref(), Some("user-123"));
        let text = build_scan_text(&req);
        assert!(text.contains("My SSN is 123-45-6789."));
        assert!(text.contains("Transfer my money."));
    }
}
