-- Adds per-check layer trace and model version to the audit trail, so the
-- logs viewer can show which layer (deterministic, ML) produced each check's
-- verdict and which model was used. `layers` is JSONB (nullable) containing
-- the same `LayerSummary` array that `audit::EvaluationEvent.layers` carries;
-- `model_version` is TEXT (nullable) — the selected layer's `model_version`
-- string, or NULL on the deterministic-only path.

ALTER TABLE evaluation_logs
    ADD COLUMN layers JSONB,
    ADD COLUMN model_version TEXT;
