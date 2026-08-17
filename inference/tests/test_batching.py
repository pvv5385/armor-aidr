"""Batching, saturation, and the invariants that keep one caller's score from
reaching another caller."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from armor_inference.batching import BatchProcessor, Saturated, params_key
from armor_inference.runners.base import InferOutput, Runner


class RecordingRunner(Runner):
    """Records the batches it was handed, so a test can assert coalescing
    actually happened rather than inferring it from timing."""

    def __init__(self, delay: float = 0.0):
        self.batches: List[List[str]] = []
        self.params_seen: List[Optional[Dict[str, Any]]] = []
        self.delay = delay

    def infer_batch(self, texts, params=None):
        if self.delay:
            import time

            time.sleep(self.delay)
        self.batches.append(list(texts))
        self.params_seen.append(params)
        return [InferOutput(decision="ALLOW", risk_score=0) for _ in texts]


async def _drain(proc: BatchProcessor) -> None:
    await proc.stop()


async def test_concurrent_submits_coalesce_into_one_forward_pass():
    runner = RecordingRunner()
    proc = BatchProcessor(runner, max_batch_size=8, max_wait_ms=25, budget_ms=2000)
    await proc.start()
    try:
        await asyncio.gather(*(proc.submit(f"text-{i}") for i in range(5)))
    finally:
        await _drain(proc)

    assert runner.batches == [[f"text-{i}" for i in range(5)]]
    stats = proc.stats()
    assert stats["batches_processed"] == 1
    assert stats["items_processed"] == 5
    assert stats["avg_batch_size"] == 5.0


async def test_batch_size_is_capped():
    runner = RecordingRunner()
    proc = BatchProcessor(runner, max_batch_size=3, max_wait_ms=25, budget_ms=2000)
    await proc.start()
    try:
        await asyncio.gather(*(proc.submit(f"t{i}") for i in range(7)))
    finally:
        await _drain(proc)

    assert all(len(b) <= 3 for b in runner.batches)
    assert sum(len(b) for b in runner.batches) == 7


async def test_differing_params_are_batched_separately_not_bypassed():
    runner = RecordingRunner()
    proc = BatchProcessor(runner, max_batch_size=8, max_wait_ms=25, budget_ms=2000)
    await proc.start()
    try:
        await asyncio.gather(
            proc.submit("a", {"lang": "en"}),
            proc.submit("b", {"lang": "en"}),
            proc.submit("c", {"lang": "fr"}),
        )
    finally:
        await _drain(proc)

    grouped = sorted(runner.batches, key=len, reverse=True)
    assert grouped == [["a", "b"], ["c"]]
    assert {params_key(p) for p in runner.params_seen} == {
        params_key({"lang": "en"}),
        params_key({"lang": "fr"}),
    }


def test_params_key_is_order_independent():
    assert params_key({"a": 1, "b": 2}) == params_key({"b": 2, "a": 1})
    assert params_key(None) == params_key({}) == ""
    assert params_key({"a": 1}) != params_key({"a": 2})


async def test_a_full_queue_rejects_rather_than_blocking():
    """Over capacity the answer is 429. A guardrail that blocks indefinitely
    is one that gets taken out of the request path.

    The worker is deliberately never started, so the queue depth is exactly
    what the test put there — the same assertion against a running worker
    depends on how far it happened to drain, which is a flake.
    """
    proc = BatchProcessor(RecordingRunner(), max_queue=2, budget_ms=5000)
    queued = [asyncio.create_task(proc.submit(f"t{i}")) for i in range(2)]
    await asyncio.sleep(0)  # let both reach the queue

    with pytest.raises(Saturated, match="saturated"):
        await proc.submit("one-too-many")

    assert proc.stats()["rejected"] == 1
    assert proc.stats()["queue_depth"] == 2
    for task in queued:
        task.cancel()
    await asyncio.gather(*queued, return_exceptions=True)


async def test_exceeding_the_budget_rejects_and_abandons_the_work():
    """A caller past its budget is gone; the batch must not spend a forward
    pass producing a result for it."""
    runner = RecordingRunner(delay=0.2)
    proc = BatchProcessor(runner, max_batch_size=1, max_wait_ms=1, max_queue=64, budget_ms=30)
    await proc.start()
    try:
        blocker = asyncio.create_task(proc.submit("slow"))
        await asyncio.sleep(0.01)
        with pytest.raises(Saturated, match="budget"):
            await proc.submit("waits-too-long")
        await asyncio.gather(blocker, return_exceptions=True)
        await asyncio.sleep(0.25)
    finally:
        await _drain(proc)

    assert proc.stats()["rejected"] >= 1
    assert proc.stats()["abandoned"] >= 1
    assert "waits-too-long" not in [t for batch in runner.batches for t in batch]


async def test_a_runner_exception_reaches_every_waiter_and_the_worker_survives():
    class Boom(Runner):
        def __init__(self):
            self.calls = 0

        def infer_batch(self, texts, params=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model exploded")
            return [InferOutput(decision="ALLOW", risk_score=0) for _ in texts]

    runner = Boom()
    proc = BatchProcessor(runner, max_batch_size=4, max_wait_ms=20, budget_ms=2000)
    await proc.start()
    try:
        results = await asyncio.gather(
            proc.submit("a"), proc.submit("b"), return_exceptions=True
        )
        assert all(isinstance(r, RuntimeError) for r in results)
        # The worker is still alive and serving.
        assert (await proc.submit("c")).decision == "ALLOW"
    finally:
        await _drain(proc)


async def test_a_misaligned_runner_fails_the_batch_rather_than_misattributing():
    """Positional alignment is the batcher's contract with a runner. If it
    breaks, callers must get an error — not each other's scores."""

    class Misaligned(Runner):
        def infer_batch(self, texts, params=None):
            return [InferOutput(decision="BLOCK", risk_score=99)]  # one output, N inputs

    proc = BatchProcessor(Misaligned(), max_batch_size=4, max_wait_ms=20, budget_ms=2000)
    await proc.start()
    try:
        results = await asyncio.gather(
            proc.submit("a"), proc.submit("b"), return_exceptions=True
        )
    finally:
        await _drain(proc)

    assert all(isinstance(r, RuntimeError) for r in results)
    assert all("outputs for" in str(r) for r in results)
