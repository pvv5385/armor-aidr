from __future__ import annotations

import threading
from collections import OrderedDict
from hashlib import sha256
from typing import Any, Optional


def content_key(model_version: str, text: str) -> str:
    """`model_version:sha256(text)` — no normalization. The model version is
    part of the key so a hot-swap can never serve a verdict from the model it
    replaced."""
    digest = sha256((text or "").encode("utf-8")).hexdigest()
    return f"{model_version}:{digest}"


class ContentHashCache:
    """Thread-safe LRU with hit/miss counters, surfaced on `/v1/stats`.

    Thread-safe rather than loop-confined because the batcher runs inference
    in a worker thread and the counters are read from the event loop.
    """

    def __init__(self, maxsize: int = 4096):
        self.maxsize = max(1, maxsize)
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> dict:
        with self._lock:
            hits, misses, size = self.hits, self.misses, len(self._store)
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "size": size,
            "maxsize": self.maxsize,
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }
