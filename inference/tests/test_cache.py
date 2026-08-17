"""The cache key. Its one job is to never be the layer that loses fidelity."""

from __future__ import annotations

from armor_inference.cache import ContentHashCache, content_key


def test_key_is_case_and_whitespace_sensitive():
    """`the alternative implementation` lowercases and collapses whitespace here, which trades
    detection fidelity for hit rate. For a secret scanner or an evasion test,
    these pairs are different inputs and must not share a verdict."""
    assert content_key("m@1", "AKIAIOSFODNN7EXAMPLE") != content_key("m@1", "akiaiosfodnn7example")
    assert content_key("m@1", "ignore  previous") != content_key("m@1", "ignore previous")
    assert content_key("m@1", " padded ") != content_key("m@1", "padded")


def test_key_is_scoped_to_the_model_version():
    """A hot-swap must not be able to serve a verdict computed by the model it
    replaced."""
    assert content_key("m@1", "text") != content_key("m@2", "text")


def test_identical_text_hits():
    cache = ContentHashCache(maxsize=4)
    key = content_key("m@1", "same")
    assert cache.get(key) is None
    cache.put(key, "verdict")
    assert cache.get(key) == "verdict"
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_lru_evicts_the_least_recently_used():
    cache = ContentHashCache(maxsize=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now the most recent, so 'b' is next out
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.size == 2


def test_stats_report_a_hit_rate():
    cache = ContentHashCache(maxsize=8)
    cache.put("k", "v")
    for _ in range(3):
        cache.get("k")
    cache.get("missing")
    stats = cache.stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.75
