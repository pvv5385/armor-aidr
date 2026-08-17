"""Shared fixtures.

Every test here runs with **no ML dependencies installed** — that is the
property the suite exists to protect. If a change makes onnxruntime or torch
necessary to boot, these tests stop collecting, which is the signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from armor_inference import catalog

# The repo's own catalog, located from this file rather than from the working
# directory or from where the package happens to be installed. Several tests
# assert against the shipped pins, and they must mean the same thing whether
# the suite runs from `inference/`, from the repo root, or against a wheel.
REPO_CATALOG = Path(__file__).resolve().parents[2] / "config" / "ml_catalog.yaml"

# Env vars the service reads. Cleared around every test so one test's
# configuration cannot leak into the next.
_ENV_KEYS = [
    "ARMOR_INFERENCE_TASKS",
    "ARMOR_INFERENCE_PROFILE",
    "ARMOR_INFERENCE_MAX_BATCH",
    "ARMOR_INFERENCE_MAX_WAIT_MS",
    "ARMOR_INFERENCE_MAX_QUEUE",
    "ARMOR_INFERENCE_BUDGET_MS",
    "ARMOR_INFERENCE_CACHE_SIZE",
    "ARMOR_INFERENCE_ARTIFACTS_DIR",
    "ARMOR_INFERENCE_ALLOW_INSTALL",
    "ARMOR_INFERENCE_AUTH_TOKEN",
    "ARMOR_ML_CATALOG",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert REPO_CATALOG.is_file(), f"expected the shipped catalog at {REPO_CATALOG}"
    monkeypatch.setenv("ARMOR_ML_CATALOG", str(REPO_CATALOG))
    catalog.load_catalog.cache_clear()
    yield
    catalog.load_catalog.cache_clear()


@pytest.fixture
def client_factory(monkeypatch):
    """A `TestClient` over the real app, with env applied before startup.

    The client must be used as a context manager so FastAPI's lifespan
    actually runs — without it the registry is never built and every route
    500s on a missing `app.state`.
    """

    def make(**env: str) -> TestClient:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        from armor_inference.main import app

        return TestClient(app)

    return make


@pytest.fixture
def client(client_factory):
    with client_factory() as c:
        yield c
