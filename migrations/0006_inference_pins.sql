-- Per-task model pins: which model (and revision) each task should use,
-- persisted by the control plane and pushed to edge instances through the
-- existing sync bundle (sync.rs). The sync payload gains a `pins` array
-- so ARMOR_MODE=edge instances receive them without a second channel.
--
-- `threshold` is the per-task confidence threshold below which the model's
-- verdict is treated as advisory-only (a coarse alternative to the
-- per-check scorecard gate for operators who want a single knob).

CREATE TABLE inference_pins (
    task            TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    revision        TEXT NOT NULL,
    sha256          TEXT,
    threshold       DOUBLE PRECISION,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
