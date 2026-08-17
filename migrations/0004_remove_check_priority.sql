-- Check execution order is no longer configurable: the orchestrator sorts
-- sequentially by a fixed backend-owned cheapest-first ranking
-- (armor_core::detectors::default_order), so the per-check `priority`
-- column is dead weight. Dropped in a new migration rather than edited
-- into 0001 (sqlx fingerprints already-applied files).
ALTER TABLE checks DROP COLUMN priority;
