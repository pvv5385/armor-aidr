//! Measures Armor's own p50/p95/p99 latency for a full policy run against
//! the shipped default policy (`config/policies.yaml`) — a verified,
//! measured number rather than a target. Deliberately Armor's own engine
//! time only, no HTTP/gateway hop on top: measure and publish our own
//! latency, not end-to-end.
//!
//! View count is a multiplier on every latency number here, which is why
//! this is measured against the *shipped* `normalize` config (NFKC +
//! strip-invisible + HTML-entity views enabled, ROT13/base64 off) rather
//! than an arbitrary hypothetical view count — that's the real number a
//! default deployment actually gets.
//!
//! No `criterion` dependency: this repo has no benchmarking harness
//! elsewhere, and a plain warmup + N-sample percentile computation over
//! `std::time::Instant` is enough to defend or correct the stated budget
//! without adding a new dependency for it. Run with
//! `cargo test --release --test latency_benchmark -- --nocapture` to see
//! the full report — the default (non-release) profile is deliberately
//! *not* asserted against, since debug-build timings aren't representative
//! of what a deployment actually runs.

use std::time::Instant;

use armor_core::engine::orchestrator;
use armor_core::policy::loader;

/// ~500 bytes of realistic chat-message prose — a representative payload
/// size. Latency scales with both payload size and view count, so this
/// benchmark holds both fixed (this payload, the shipped default policy's
/// view set) to measure one repeatable point on that curve.
const PAYLOAD: &str = "Hi, I'm working on a quarterly report for our engineering \
team and I wanted to get some help structuring the executive summary. \
We shipped four major features this quarter, reduced our p95 API latency \
by about eighteen percent, and onboarded two new customers in the \
healthcare vertical. Could you help me turn these bullet points into a \
polished three-paragraph summary that a non-technical VP could skim in \
under a minute? I'd like it to sound confident but not overhyped.";

const WARMUP_ITERATIONS: usize = 100;
const SAMPLE_ITERATIONS: usize = 1000;

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx]
}

#[test]
fn p95_latency_against_the_shipped_default_policy() {
    let policy = loader::load(include_str!("../../../config/policies.yaml"))
        .expect("config/policies.yaml must load");

    for _ in 0..WARMUP_ITERATIONS {
        orchestrator::run_checks(&policy, PAYLOAD, armor_core::detectors::get_check);
    }

    let mut samples_ms = Vec::with_capacity(SAMPLE_ITERATIONS);
    for _ in 0..SAMPLE_ITERATIONS {
        let start = Instant::now();
        orchestrator::run_checks(&policy, PAYLOAD, armor_core::detectors::get_check);
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = percentile(&samples_ms, 0.50);
    let p95 = percentile(&samples_ms, 0.95);
    let p99 = percentile(&samples_ms, 0.99);
    let mean: f64 = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;

    eprintln!(
        "latency over {} iterations, {}-byte payload, shipped default policy ({} checks): \
         mean={mean:.3}ms p50={p50:.3}ms p95={p95:.3}ms p99={p99:.3}ms",
        SAMPLE_ITERATIONS,
        PAYLOAD.len(),
        policy.checks.len(),
    );

    // Debug builds run this policy's regex sweep several times slower than
    // release — a tight assertion here would fail under plain `cargo test`
    // for reasons that have nothing to do with the engine being slow, so
    // this is a coarse regression guard (catches an accidentally quadratic
    // detector, not a few-percent latency regression) rather than the
    // budget check itself. The real number (`cargo test --release`) is
    // checked below, with headroom over the marketed <2ms p95 target rather
    // than an assertion pinned right at it — CI hardware varies, and
    // this is a regression guard against something going structurally
    // wrong (e.g. an O(n^2) detector), not a promise that every machine
    // hits the exact marketed number. As measured on this machine at the
    // time this test was written: p95 ~1.6ms, p99 ~3.2ms — i.e. the <2ms
    // *p95* claim holds, but p99 does not.
    #[cfg(debug_assertions)]
    assert!(
        p95 < 50.0,
        "p95 latency {p95:.3}ms exceeds the coarse debug-build regression guard (50ms) — \
         re-run with --release for the number that actually matters"
    );
    #[cfg(not(debug_assertions))]
    assert!(
        p95 < 3.0,
        "p95 latency {p95:.3}ms exceeds the release-build regression guard (3ms, with \
         headroom over the marketed <2ms p95 target)"
    );
}
