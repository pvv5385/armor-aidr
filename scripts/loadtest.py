#!/usr/bin/env python3
"""Constant-rate load test against armor-api's ``POST /api/v1/aidr/scan``.

Paces requests at a fixed, configurable rate (requests per minute) while
cycling a variable set of payload sizes, and prints a latency / throughput /
verdict summary. Stdlib-only (urllib + threading) — no external dependencies.

Examples::

    # ~1 request/sec for 2 minutes, payload sizes cycling 256B -> 64KB
    python3 scripts/loadtest.py --rpm 60 --duration 120

    # 90 req/min, 500 requests total, larger payloads
    python3 scripts/loadtest.py --rpm 90 --sizes 1024,16384,262144 --total 500

    # against a deployed instance that requires an API key
    python3 scripts/loadtest.py --base-url https://armor.example.com \\
        --api-key <key> --rpm 100 --duration 300

The armor-api instance must already be running and reachable at BASE_URL.
See ``docs/AIDR_IMPLEMENTATION.md`` for the request schema this hits and
``crates/api/src/config.rs`` for ``ARMOR_MAX_BODY_BYTES`` (default 2 MiB) —
payload sizes above that will be rejected with a 413.
"""
import argparse
import concurrent.futures
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_SIZES = "256,1024,8192,65536"

# Deterministic filler so a run is reproducible and the text is long enough
# that multi-pass detectors have real work to do at the larger sizes.
_FILLER = (
    "The travel assistant helps customers plan itineraries, compare flights "
    "and hotels, and book reservations. It can summarize a quarterly report, "
    "flag budget overruns, and answer questions about the deployment. "
    "Requests to the inference gateway are scanned before reaching the model; "
    "the guardrail checks prompt text for injection attempts, leaked secrets, "
    "credit card numbers, and other sensitive content before deciding "
    "whether to allow, warn, or block the call. "
    "customer@example.com api_token=sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ "
)


def make_text(size: int) -> str:
    """Deterministic text of roughly ``size`` ASCII bytes (filler repeated/truncated)."""
    reps = size // len(_FILLER) + 1
    return (_FILLER * reps)[:size]


def build_body(size: int, mode: str) -> dict:
    return {"text": make_text(size), "metadata": {"mode": mode}}


def send_request(
    url: str,
    body: dict,
    headers: dict,
    timeout: float,
    session_id: str,
) -> dict:
    """One POST; returns a record with latency/status/verdict or an error."""
    req = urllib.request.Request(  # nosemgrep: dynamic-urllib-use-detected -- base URL is operator-controlled CLI/env input, defaults to localhost
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    if session_id:
        req.add_header("x-armor-session-id", session_id)
    started = time.monotonic()
    status, verdict, raw = None, None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosemgrep: dynamic-urllib-use-detected -- see above
            status = resp.status
            raw = json.loads(resp.read().decode("utf-8"))
        verdict = str(raw.get("verdict", "")).upper()
        error = None
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            error = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            error = str(e)
    except Exception as e:  # noqa: BLE001 -- record any transport failure as an error record
        status = None
        error = str(e)
    latency_ms = (time.monotonic() - started) * 1000.0
    return {
        "status": status,
        "verdict": verdict,
        "latency_ms": latency_ms,
        "error": error,
    }


def fmt(ms: float) -> str:
    return f"{ms:,.1f} ms"


def percentile(values: list, pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * pct) - 1)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:8100", help="armor-api base URL (default: http://localhost:8100)")
    parser.add_argument("--endpoint", default="/api/v1/aidr/scan", help="scan endpoint (default: /api/v1/aidr/scan)")
    parser.add_argument("--rpm", type=int, default=60, help="constant request rate in requests per minute (default: 60)")
    parser.add_argument("--duration", type=float, default=60.0, help="run duration in seconds (default: 60; ignored when --total is set)")
    parser.add_argument("--total", type=int, default=None, help="exact number of requests (default: derived from --duration)")
    parser.add_argument("--sizes", default=DEFAULT_SIZES, help=f"comma-separated payload sizes in bytes, cycled round-robin (default: {DEFAULT_SIZES})")
    parser.add_argument("--mode", default="input", help="metadata.mode stamped on each request (default: input)")
    parser.add_argument("--concurrency", type=int, default=1, help="in-flight request cap (default: 1)")
    parser.add_argument("--timeout", type=float, default=10.0, help="per-request timeout in seconds (default: 10)")
    parser.add_argument("--api-key", default=None, help="value for the X-API-Key header when ARMOR_AUTH_MODE=apikey")
    parser.add_argument("--session-id", default=None, help="fixed x-armor-session-id so all requests share one session")
    args = parser.parse_args()

    if args.rpm <= 0:
        parser.error("--rpm must be > 0")
    if args.concurrency <= 0:
        parser.error("--concurrency must be > 0")
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if not sizes:
        parser.error("--sizes must be a non-empty comma-separated list of byte counts")
    if any(s <= 0 for s in sizes):
        parser.error("payload sizes must be positive")

    # armor-api's default body cap is 2 MiB (ARMOR_MAX_BODY_BYTES) — flag
    # sizes that will come back 413 so the run isn't silently wasted.
    for s in sizes:
        if s > 1_800_000:
            print(f"warning: size {s} bytes is close to/above the default ARMOR_MAX_BODY_BYTES (2 MiB) and may be rejected", file=sys.stderr)

    url = args.base_url.rstrip("/") + args.endpoint
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    interval = 60.0 / args.rpm
    total = args.total if args.total is not None else max(1, int(args.duration / interval))
    print(
        f"targeting {args.rpm} req/min ({interval:.3f}s cadence) for {total} requests "
        f"across {len(sizes)} payload sizes {sizes}, concurrency={args.concurrency}",
        file=sys.stderr,
    )

    lock = threading.Lock()
    sem = threading.BoundedSemaphore(args.concurrency)
    results = []

    def worker(index: int, size: int, body: dict):
        try:
            record = send_request(url, body, headers, args.timeout, args.session_id)
        finally:
            sem.release()
        record.update({"index": index, "size": size})
        with lock:
            results.append(record)

    started = time.monotonic()
    next_at = started
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for index in range(total):
                now = time.monotonic()
                if now < next_at:
                    time.sleep(next_at - now)
                sem.acquire()
                # Schedule the next send from whichever is later: the plan or
                # actual dispatch time. Anchoring only to the plan lets a
                # semaphore stall (or a slow response under concurrency=1)
                # pile up a backlog that then fires back-to-back once
                # capacity frees — anchoring to "now" here drops the backlog
                # instead of bursting it out.
                next_at = max(next_at, time.monotonic()) + interval
                size = sizes[index % len(sizes)]
                pool.submit(worker, index, size, build_body(size, args.mode))
        pool.shutdown(wait=True)
    except KeyboardInterrupt:
        print("\ninterrupted — printing partial results", file=sys.stderr)

    elapsed = time.monotonic() - started
    if not results:
        print("no requests completed", file=sys.stderr)
        return 1

    ok = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    latencies = [r["latency_ms"] for r in ok]

    print()
    print("── summary ──────────────────────────────────────────────────")
    print(f"requests sent      : {len(results)} (target {total})")
    print(f"completed ok       : {len(ok)}   errors: {len(errors)}")
    print(f"wall-clock         : {elapsed:.1f}s")
    print(f"achieved rate      : {len(results) / elapsed * 60.0:.1f} req/min ({len(results) / elapsed:.2f} req/s)")
    if latencies:
        print(f"latency            : avg {statistics.mean(latencies):,.1f} ms | "
              f"p50 {percentile(latencies, 0.50):,.1f} ms | "
              f"p95 {percentile(latencies, 0.95):,.1f} ms | "
              f"p99 {percentile(latencies, 0.99):,.1f} ms")
    if errors:
        statuses = {}
        for e in errors:
            statuses[e["status"]] = statuses.get(e["status"], 0) + 1
        print(f"error statuses     : {statuses}")
        print(f"first error        : {errors[0]['error']}")

    verdicts = {}
    for r in ok:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    if verdicts:
        print(f"verdicts           : {verdicts}")

    print()
    print("── per payload size ─────────────────────────────────────────")
    print(f"{'size':>9}  {'count':>6}  {'avg ms':>8}  {'p95 ms':>8}  {'errors':>6}")
    for size in sizes:
        group = [r for r in results if r["size"] == size]
        if not group:
            continue
        group_ok = [r for r in group if r["error"] is None]
        avg = statistics.mean(r["latency_ms"] for r in group_ok) if group_ok else float("nan")
        p95 = percentile([r["latency_ms"] for r in group_ok], 0.95)
        errs = len(group) - len(group_ok)
        print(f"{size:>9}  {len(group):>6}  {avg:>8.1f}  {p95:>8.1f}  {errs:>6}")
    print()

    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
