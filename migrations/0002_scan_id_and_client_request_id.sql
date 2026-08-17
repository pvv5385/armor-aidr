-- Renames `evaluation_logs.event_id` to `scan_id` and adds
-- `client_request_id` — see `crates/api/src/aidr.rs`'s `ScanResponse`
-- (`scan_id`/`request_id`) and `EvaluationEvent` (`scan_id`/
-- `client_request_id`). `0001_control_plane.sql` stays as originally
-- applied rather than being edited in place, since `sqlx::migrate!`
-- fingerprints each migration file and refuses to run if an
-- already-applied one changed underneath it.

ALTER TABLE evaluation_logs RENAME COLUMN event_id TO scan_id;

-- Caller-supplied correlation id (`metadata.request_id`, or an adapter's
-- vendor-native equivalent — e.g. LiteLLM's `litellm_call_id`) echoed back
-- as `ScanResponse.request_id`. NULL when the caller didn't supply one —
-- distinct from `scan_id` above, which is Armor's own id and always
-- present.
ALTER TABLE evaluation_logs ADD COLUMN client_request_id TEXT;
