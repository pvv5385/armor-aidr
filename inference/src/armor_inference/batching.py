"""Dynamic batching: coalesce concurrent `/v1/infer` calls for one task into a
single forward pass, without letting any caller wait past a budget.

One worker per task drains the queue, so inference is serialized (a runner
never re-enters) and batches form up to `max_batch_size` within a
`max_wait_ms` window. Over queue depth or over budget the answer is
`Saturated` → 429, never a call that blocks indefinitely: a guardrail that
stops answering is a guardrail that gets removed from the request path.

Dependency-free — asyncio only.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from armor_inference.runners.base import InferOutput, Runner

# One queued item: the text, the key its params group under, the params
# themselves, and the future waiting on its result.
_Item = Tuple[str, str, Optional[Dict[str, Any]], "asyncio.Future"]


class Saturated(Exception):
    """The queue is full, or an item waited past the budget. The caller turns
    this into a 429."""


def params_key(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


class BatchProcessor:
    def __init__(
        self,
        runner: Runner,
        *,
        max_batch_size: int = 16,
        max_wait_ms: int = 10,
        max_queue: int = 256,
        budget_ms: int = 2000,
    ):
        self.runner = runner
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_ms = max(0, max_wait_ms)
        self.max_queue = max(1, max_queue)
        self.budget_ms = max(1, budget_ms)
        self._queue: "asyncio.Queue[_Item]" = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._running = False
        self.batches_processed = 0
        self.items_processed = 0
        self.rejected = 0
        self.abandoned = 0

    async def start(self) -> None:
        if self._worker is None:
            self._running = True
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, text: str, params: Optional[Dict[str, Any]] = None) -> InferOutput:
        """Enqueue one item and await its result, or raise `Saturated`."""
        if self._queue.qsize() >= self.max_queue:
            self.rejected += 1
            raise Saturated("inference queue saturated")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((text, params_key(params), params, fut))
        try:
            return await asyncio.wait_for(fut, timeout=self.budget_ms / 1000.0)
        except asyncio.TimeoutError:
            # `wait_for` cancels the future on timeout, which is what lets the
            # worker skip it below instead of spending a forward pass on a
            # result nobody is waiting for. That only matters under load —
            # which is the only time it happens.
            self.rejected += 1
            raise Saturated("inference exceeded latency budget") from None

    async def _collect_batch(self) -> List[_Item]:
        """Drain up to `max_batch_size` items, waiting at most `max_wait_ms`
        after the first one."""
        batch = [await self._queue.get()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_wait_ms / 1000.0
        while len(batch) < self.max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    def _group(self, batch: List[_Item]) -> List[List[_Item]]:
        """Split a drained batch into runs sharing the same params. Callers
        with identical params (the overwhelming majority: none) still coalesce
        into one forward pass."""
        groups: Dict[str, List[_Item]] = {}
        for item in batch:
            groups.setdefault(item[1], []).append(item)
        return list(groups.values())

    async def _run(self) -> None:
        while self._running:
            try:
                batch = await asyncio.wait_for(self._collect_batch(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Items whose caller already gave up (budget elapsed, client
            # disconnected) cost nothing further.
            live = [item for item in batch if not item[3].done()]
            self.abandoned += len(batch) - len(live)

            for group in self._group(live):
                texts = [text for text, _, _, _ in group]
                params = group[0][2]
                try:
                    # Runners are sync and CPU-bound; off-loop so the service
                    # keeps accepting requests while a batch runs.
                    outputs = await asyncio.to_thread(self.runner.infer_batch, texts, params)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — every waiter needs to hear it
                    for _, _, _, fut in group:
                        if not fut.done():
                            fut.set_exception(exc)
                    continue
                if len(outputs) != len(texts):
                    # Positional alignment is the batcher's whole contract with
                    # a runner. A length mismatch means results would be handed
                    # to the wrong callers — fail the batch rather than answer
                    # one caller with another's score.
                    err = RuntimeError(
                        f"runner returned {len(outputs)} outputs for {len(texts)} inputs"
                    )
                    for _, _, _, fut in group:
                        if not fut.done():
                            fut.set_exception(err)
                    continue
                self.batches_processed += 1
                self.items_processed += len(group)
                # strict=False preserves the existing behavior deliberately: a
                # runner returning fewer outputs than inputs leaves the extra
                # futures unresolved, and the caller's own deadline handles
                # that. Raising here instead would abort the whole batch loop
                # and strand every other in-flight request with it.
                for (_, _, _, fut), out in zip(group, outputs, strict=False):
                    if not fut.done():
                        fut.set_result(out)

    def stats(self) -> dict:
        return {
            "batches_processed": self.batches_processed,
            "items_processed": self.items_processed,
            "rejected": self.rejected,
            "abandoned": self.abandoned,
            "queue_depth": self._queue.qsize(),
            "max_batch_size": self.max_batch_size,
            "avg_batch_size": (
                round(self.items_processed / self.batches_processed, 2)
                if self.batches_processed
                else 0.0
            ),
        }
