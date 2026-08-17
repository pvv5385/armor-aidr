"""Getting weights onto a running container.

Two supported routes: mount them at `/models`, or install them after start.
This file covers the second, including the parts that keep it from being a
way to swap a deployment's detection layer over HTTP.
"""

from __future__ import annotations

import time

import pytest
import yaml

from armor_inference import catalog, fetch
from armor_inference.install import InstallDisabled, InstallManager
from armor_inference.runners._artifacts import (
    artifact_dirname,
    artifact_sha256,
    resolve_artifact_dir,
    verify_pinned,
)
from armor_inference.runners.base import RunnerUnavailable

_TERMINAL = ("complete", "installed", "failed")


def _wait_for(manager: InstallManager, job_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get_job(job_id)
        if job.status in _TERMINAL:
            return job
        time.sleep(0.01)
    raise AssertionError(f"install job {job_id} did not settle: {manager.get_job(job_id)}")


# ── Artifact resolution and pinning ────────────────────────────────────────


def test_artifact_dirname_flattens_the_model_id():
    assert artifact_dirname("protectai/deberta-v3-base") == "protectai__deberta-v3-base"
    # A separator in a model id must not become a path component.
    assert "/" not in artifact_dirname("../../etc/passwd")


def test_resolve_artifact_dir_is_always_local(monkeypatch, tmp_path):
    monkeypatch.setenv("ARMOR_INFERENCE_ARTIFACTS_DIR", str(tmp_path))
    assert resolve_artifact_dir({"model_id": "org/model"}) == str(tmp_path / "org__model")
    assert resolve_artifact_dir({"artifacts_dir": "/explicit"}) == "/explicit"


def test_the_digest_covers_names_as_well_as_contents(tmp_path):
    """Renaming `model.onnx` to `model_quantized.onnx` changes which graph is
    served, so it has to change the digest."""
    (tmp_path / "model.onnx").write_bytes(b"weights")
    first = artifact_sha256(str(tmp_path))
    (tmp_path / "model.onnx").rename(tmp_path / "model_quantized.onnx")
    assert artifact_sha256(str(tmp_path)) != first


def test_the_digest_is_stable_across_walks(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.bin").write_bytes(b"\x00\x01")
    assert artifact_sha256(str(tmp_path)) == artifact_sha256(str(tmp_path))


def test_verify_pinned_fails_closed(tmp_path):
    missing = tmp_path / "absent"
    with pytest.raises(RunnerUnavailable, match="no implicit download"):
        verify_pinned(str(missing), None)

    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.onnx").write_bytes(b"weights")
    verify_pinned(str(artifact), None)  # unpinned is allowed
    verify_pinned(str(artifact), artifact_sha256(str(artifact)))  # matching pin

    with pytest.raises(RunnerUnavailable, match="sha256 mismatch"):
        verify_pinned(str(artifact), "0" * 64)


def test_a_tampered_artifact_fails_verification(tmp_path):
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.onnx").write_bytes(b"reviewed weights")
    pinned = artifact_sha256(str(artifact))

    (artifact / "model.onnx").write_bytes(b"swapped weights")
    with pytest.raises(RunnerUnavailable, match="sha256 mismatch"):
        verify_pinned(str(artifact), pinned)


# ── Fetch targets ──────────────────────────────────────────────────────────


def test_fetch_defaults_to_the_catalog_pin():
    target = fetch.resolve_fetch_target("prompt_injection")
    assert target["model_id"] == "protectai/deberta-v3-base-prompt-injection-v2"
    assert target["revision"] == "main"
    assert target["runner"] == "classifier"
    # The shipped catalog never pins a digest — see test_config_and_catalog's
    # test_the_shipped_catalog_parses_and_is_complete.
    assert target["expected_sha256"] is None


def _catalog_with_pinned_digest(tmp_path, monkeypatch, digest):
    path = tmp_path / "ml_catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tasks": {
                    "t": {
                        "runner": "classifier",
                        "model_id": "a/one",
                        "revision": "main",
                        "sha256": digest,
                    }
                },
                "candidates": {"t": [{"model_id": "a/one"}, {"model_id": "b/two"}]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARMOR_ML_CATALOG", str(path))
    catalog.load_catalog.cache_clear()


def test_resolve_fetch_target_forwards_an_operator_pinned_digest(tmp_path, monkeypatch):
    _catalog_with_pinned_digest(tmp_path, monkeypatch, "deadbeef" * 8)
    target = fetch.resolve_fetch_target("t")
    assert target["expected_sha256"] == "deadbeef" * 8


def test_a_pinned_digest_does_not_apply_to_a_shortlisted_override(tmp_path, monkeypatch):
    # The pin is a claim about the *default* pin's bytes; a different vetted
    # model on the shortlist is a different artifact and has no digest here.
    _catalog_with_pinned_digest(tmp_path, monkeypatch, "deadbeef" * 8)
    target = fetch.resolve_fetch_target("t", model_id="b/two")
    assert target["expected_sha256"] is None


def test_fetch_model_raises_when_the_download_does_not_match_the_catalog_pin(
    tmp_path, monkeypatch
):
    """End-to-end: an operator pinned a digest, and what actually landed on
    disk does not match it — the exact tamper/substitution case a
    self-certifying digest cannot catch."""
    _catalog_with_pinned_digest(tmp_path, monkeypatch, "0" * 64)
    target = fetch.resolve_fetch_target("t")

    def fake_downloader(model_id, revision, dest_dir):
        from pathlib import Path

        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        Path(dest_dir, "model.onnx").write_bytes(b"not what was reviewed")

    with pytest.raises(RunnerUnavailable, match="integrity check failed"):
        fetch.fetch_model(
            target["model_id"],
            target["revision"],
            str(tmp_path / "dest"),
            downloader=fake_downloader,
            expected_sha256=target["expected_sha256"],
        )


def test_an_off_shortlist_model_needs_an_explicit_override():
    """The easy path is a reviewed model; anything else requires saying so."""
    with pytest.raises(ValueError, match="not on the vetted shortlist"):
        fetch.resolve_fetch_target("prompt_injection", model_id="rando/backdoored-bert")

    target = fetch.resolve_fetch_target(
        "prompt_injection", model_id="rando/backdoored-bert", allow_unvetted=True
    )
    assert target["model_id"] == "rando/backdoored-bert"


def test_a_shortlisted_alternative_is_accepted():
    target = fetch.resolve_fetch_target("toxicity", model_id="unitary/multilingual-toxic-xlm-roberta")
    assert target["model_id"] == "unitary/multilingual-toxic-xlm-roberta"


def test_an_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown task"):
        fetch.resolve_fetch_target("telepathy")


def test_fetch_model_returns_the_digest_of_what_landed(tmp_path):
    """The operator pins a hash of the tree on disk, not one the hub
    advertised."""

    def fake_download(model_id, revision, dest_dir):
        from pathlib import Path

        Path(dest_dir, "model.onnx").write_bytes(b"exported graph")
        Path(dest_dir, "tokenizer.json").write_text("{}", encoding="utf-8")

    dest = tmp_path / "org__model"
    digest = fetch.fetch_model("org/model", "main", str(dest), downloader=fake_download)
    assert digest == artifact_sha256(str(dest))
    verify_pinned(str(dest), digest)


def test_the_export_stack_is_not_required_to_import(tmp_path):
    """The serving image has no torch. Asking it to export must produce an
    error that names the extra, not an ImportError at module load."""
    with pytest.raises(RunnerUnavailable, match=r"\[export\] extra"):
        fetch._require(lambda: __import__("definitely_not_installed_xyz"), "optimum")


# ── Pre-built ONNX detection/download ──────────────────────────────────────


def _fake_list_repo_files(files):
    def _list(repo_id, revision=None):
        return list(files)

    return _list


def test_find_prebuilt_onnx_prefers_the_quantized_subfolder_variant(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        _fake_list_repo_files(
            {"onnx/model_quantized.onnx", "onnx/model.onnx", "tokenizer.json", "config.json"}
        ),
    )
    assert fetch._find_prebuilt_onnx("org/model", "main") == "onnx/model_quantized.onnx"


def test_find_prebuilt_onnx_falls_back_to_fp32_without_a_quantized_variant(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        _fake_list_repo_files({"onnx/model.onnx", "tokenizer.json"}),
    )
    assert fetch._find_prebuilt_onnx("org/model", "main") == "onnx/model.onnx"


def test_find_prebuilt_onnx_returns_none_without_a_fast_tokenizer(monkeypatch):
    # A graph with no tokenizer.json isn't actually servable — _heavy.py's
    # _find_tokenizer has no fallback — so it's not worth preferring over a
    # from-source export that builds a fast tokenizer itself.
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        _fake_list_repo_files({"onnx/model_quantized.onnx", "vocab.txt"}),
    )
    assert fetch._find_prebuilt_onnx("org/model", "main") is None


def test_find_prebuilt_onnx_returns_none_when_the_repo_has_no_onnx(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        _fake_list_repo_files({"pytorch_model.bin", "tokenizer.json", "config.json"}),
    )
    assert fetch._find_prebuilt_onnx("org/model", "main") is None


def test_download_prebuilt_onnx_flattens_the_subfolder_and_pulls_data_shards(
    tmp_path, monkeypatch
):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    repo_files = {
        "onnx/model_q4.onnx",
        "onnx/model_q4.onnx_data",
        "onnx/model_q4.onnx_data_1",
        # A sibling variant that must NOT be pulled — only files sharing the
        # *chosen* graph's exact basename are wanted.
        "onnx/model_fp16.onnx",
        "onnx/model_fp16.onnx_data",
        "config.json",
        "tokenizer.json",
        "viterbi_calibration.json",
    }
    source = tmp_path / "hub_cache"
    source.mkdir()
    downloaded: list[str] = []

    def fake_hf_hub_download(repo_id, revision, filename):
        downloaded.append(filename)
        local = source / filename.replace("/", "__")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(f"content of {filename}".encode())
        return str(local)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", _fake_list_repo_files(repo_files))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    dest = tmp_path / "dest"
    fetch._download_prebuilt_onnx("org/model", "main", str(dest), "onnx/model_q4.onnx")

    landed = {p.name for p in dest.iterdir()}
    assert landed == {
        "model_q4.onnx",
        "model_q4.onnx_data",
        "model_q4.onnx_data_1",
        "config.json",
        "tokenizer.json",
        "viterbi_calibration.json",
    }
    assert "onnx/model_fp16.onnx" not in downloaded
    assert "onnx/model_fp16.onnx_data" not in downloaded


def test_default_downloader_prefers_prebuilt_onnx_over_export(monkeypatch):
    monkeypatch.setattr(fetch, "_find_prebuilt_onnx", lambda m, r: "onnx/model.onnx")
    calls = []
    monkeypatch.setattr(
        fetch, "_download_prebuilt_onnx", lambda m, r, d, p: calls.append((m, r, d, p))
    )
    monkeypatch.setattr(
        fetch,
        "_export_onnx",
        lambda *a, **k: pytest.fail("export must not run when a pre-built graph exists"),
    )
    fetch._default_downloader("org/model", "main", "/dest", "ner")
    assert calls == [("org/model", "main", "/dest", "onnx/model.onnx")]


def test_default_downloader_falls_back_to_export_without_a_prebuilt_graph(monkeypatch):
    monkeypatch.setattr(fetch, "_find_prebuilt_onnx", lambda m, r: None)
    calls = []
    monkeypatch.setattr(
        fetch, "_export_onnx", lambda m, r, d, runner: calls.append((m, r, d, runner))
    )
    fetch._default_downloader("org/model", "main", "/dest", "ner")
    assert calls == [("org/model", "main", "/dest", "ner")]


# ── Install jobs ───────────────────────────────────────────────────────────


def test_installs_are_disabled_by_default():
    """A service that can be told over HTTP to fetch and load new weights is a
    service whose detection layer can be replaced over HTTP."""
    manager = InstallManager()
    assert not manager.enabled
    with pytest.raises(InstallDisabled, match="ARMOR_INFERENCE_ALLOW_INSTALL"):
        manager.start_install("prompt_injection")


def test_an_install_job_runs_and_reports_its_digest(monkeypatch, tmp_path):
    monkeypatch.setenv("ARMOR_INFERENCE_ARTIFACTS_DIR", str(tmp_path))

    def fake_fetch(model_id, revision, dest_dir, runner=None, expected_sha256=None):
        from pathlib import Path

        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        Path(dest_dir, "model.onnx").write_bytes(b"graph")
        return "abc123"

    monkeypatch.setattr(fetch, "fetch_model", fake_fetch)

    manager = InstallManager(enabled=True)
    try:
        job = manager.start_install("prompt_injection")
        settled = _wait_for(manager, job.job_id)
        assert settled.status == "complete"
        assert settled.sha256 == "abc123"
        assert settled.error is None
        assert settled.completed_at is not None
    finally:
        manager.shutdown()


def test_a_fetch_that_lands_but_will_not_load_is_installed_not_failed(monkeypatch, tmp_path):
    """The weights are on disk and verified; only the load failed. Reporting
    that as `failed` sends an operator back to re-download gigabytes that are
    already correct."""
    monkeypatch.setenv("ARMOR_INFERENCE_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(fetch, "fetch_model", lambda *a, **k: "abc123")

    def wont_load(job, spec):
        raise RuntimeError("no classifier runner in this build")

    manager = InstallManager(enabled=True)
    try:
        job = manager.start_install("prompt_injection", on_complete=wont_load)
        settled = _wait_for(manager, job.job_id)
        assert settled.status == "installed"
        assert settled.sha256 == "abc123"
        assert settled.error is None
        assert "no classifier runner" in settled.load_error
    finally:
        manager.shutdown()


def test_a_failed_install_is_recorded_not_raised(monkeypatch, tmp_path):
    monkeypatch.setenv("ARMOR_INFERENCE_ARTIFACTS_DIR", str(tmp_path))

    def boom(*args, **kwargs):
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(fetch, "fetch_model", boom)

    manager = InstallManager(enabled=True)
    try:
        job = manager.start_install("toxicity")
        settled = _wait_for(manager, job.job_id)
        assert settled.status == "failed"
        assert "hub unreachable" in settled.error
        # The worker survives to run the next one.
        assert manager._active_job() is None
    finally:
        manager.shutdown()


def test_an_unknown_task_does_not_occupy_the_install_slot():
    manager = InstallManager(enabled=True)
    try:
        with pytest.raises(ValueError, match="unknown task"):
            manager.start_install("telepathy")
        assert manager.list_jobs() == []
    finally:
        manager.shutdown()


def test_installs_are_serialized(monkeypatch, tmp_path):
    monkeypatch.setenv("ARMOR_INFERENCE_ARTIFACTS_DIR", str(tmp_path))
    release = __import__("threading").Event()

    def slow_fetch(model_id, revision, dest_dir, runner=None, expected_sha256=None):
        release.wait(timeout=5)
        return "sha"

    monkeypatch.setattr(fetch, "fetch_model", slow_fetch)

    manager = InstallManager(enabled=True)
    try:
        first = manager.start_install("prompt_injection")
        # Wait until the worker has actually picked it up, so "in progress"
        # is a fact rather than a race.
        deadline = time.time() + 2
        while manager.get_job(first.job_id).status == "pending" and time.time() < deadline:
            time.sleep(0.01)

        with pytest.raises(RuntimeError, match="already in progress"):
            manager.start_install("toxicity")
    finally:
        release.set()
        manager.shutdown()


# ── The HTTP surface ───────────────────────────────────────────────────────

_TOKEN = "test-token"
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


def test_install_and_reload_need_a_bearer_token_even_unconfigured(client):
    """No `ARMOR_INFERENCE_AUTH_TOKEN` set is still a zero-config boot — but
    unlike scoring, install/reload must never be reachable with no credential
    at all. A token is generated for the boot instead."""
    resp = client.post("/v1/models/install", json={"task": "prompt_injection"})
    assert resp.status_code == 401

    resp = client.post(
        "/v1/models/reload", json={"task": "toxicity", "spec": {"runner": "stub"}}
    )
    assert resp.status_code == 401

    generated = client.app.state.mutation_token
    assert generated
    resp = client.post(
        "/v1/models/reload",
        json={"task": "toxicity", "spec": {"runner": "stub"}},
        headers={"Authorization": f"Bearer {generated}"},
    )
    assert resp.status_code == 200


def test_install_endpoint_is_403_when_disabled(client_factory):
    with client_factory(ARMOR_INFERENCE_AUTH_TOKEN=_TOKEN) as client:
        resp = client.post(
            "/v1/models/install", json={"task": "prompt_injection"}, headers=_AUTH
        )
        assert resp.status_code == 403
        assert "ARMOR_INFERENCE_ALLOW_INSTALL" in resp.json()["detail"]


def test_install_endpoint_is_403_when_disabled(client_factory):
    with client_factory(ARMOR_INFERENCE_AUTH_TOKEN=_TOKEN) as client:
        resp = client.post(
            "/v1/models/install", json={"task": "prompt_injection"}, headers=_AUTH
        )
        assert resp.status_code == 403
        assert "ARMOR_INFERENCE_ALLOW_INSTALL" in resp.json()["detail"]


def _poll(client, job_id, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/models/install/{job_id}").json()
        if job["status"] in _TERMINAL:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not settle")


def test_install_fetches_then_hot_swaps_the_task(client_factory, monkeypatch, tmp_path):
    """The whole point of installing after start: a running container ends up
    serving a model it did not have when it booted, with no restart.

    The catalog's `prompt_injection` is a `classifier` — this substitutes a
    constructible stub runner for that kind at the registry's factory seam,
    so the test doesn't need real ONNX weights. What is under test is the
    install → fetch → reload → serve wiring, not the ONNX runner.
    """
    from armor_inference import registry as registry_mod
    from armor_inference.runners.base import StubRunner

    monkeypatch.setitem(
        registry_mod.RUNNER_FACTORIES,
        "classifier",
        lambda task, spec: StubRunner(task=task, threshold=float(spec.get("threshold", 0.5))),
    )

    def fake_fetch(model_id, revision, dest_dir, runner=None, expected_sha256=None):
        from pathlib import Path

        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        Path(dest_dir, "model.onnx").write_bytes(b"graph")
        return "deadbeef"

    monkeypatch.setattr(fetch, "fetch_model", fake_fetch)

    with client_factory(
        ARMOR_INFERENCE_ALLOW_INSTALL="true",
        ARMOR_INFERENCE_AUTH_TOKEN=_TOKEN,
        ARMOR_INFERENCE_ARTIFACTS_DIR=str(tmp_path),
        ARMOR_INFERENCE_TASKS='{"prompt_injection":{"runner":"stub"}}',
    ) as client:
        client.headers.update(_AUTH)
        assert client.get("/v1/models").json()["models"][0]["model_version"] == "stub@v1"

        started = client.post("/v1/models/install", json={"task": "prompt_injection"})
        assert started.status_code == 202
        job = _poll(client, started.json()["job_id"])

        assert job["status"] == "complete", job
        assert job["sha256"] == "deadbeef"

        models = {m["task"]: m for m in client.get("/v1/models").json()["models"]}
        assert models["prompt_injection"]["available"]
        assert models["prompt_injection"]["sha256"] == "deadbeef"
        assert (
            models["prompt_injection"]["model_version"]
            == "protectai/deberta-v3-base-prompt-injection-v2@main"
        )
        # Serving under the new identity, without a restart.
        scored = client.post("/v1/infer/prompt_injection", json={"text": "hello"})
        assert scored.status_code == 200
        assert scored.json()["model_version"] == "protectai/deberta-v3-base-prompt-injection-v2@main"


def test_install_without_the_runner_reports_installed_and_keeps_serving(
    client_factory, monkeypatch, tmp_path
):
    """The honest outcome when the artifact lands but the heavy deps
    (onnxruntime) its runner needs aren't installed in this image: the
    service says exactly that instead of pretending either success or a
    failed download."""
    monkeypatch.setattr(fetch, "fetch_model", lambda *a, **k: "deadbeef")

    with client_factory(
        ARMOR_INFERENCE_ALLOW_INSTALL="true",
        ARMOR_INFERENCE_AUTH_TOKEN=_TOKEN,
        ARMOR_INFERENCE_ARTIFACTS_DIR=str(tmp_path),
        ARMOR_INFERENCE_TASKS='{"prompt_injection":{"runner":"stub"}}',
    ) as client:
        client.headers.update(_AUTH)
        started = client.post("/v1/models/install", json={"task": "prompt_injection"})
        job = _poll(client, started.json()["job_id"])

        assert job["status"] == "installed", job
        assert job["sha256"] == "deadbeef"
        assert job["error"] is None
        # The runner exists but the heavy deps (onnxruntime) are not
        # installed in the test image — exactly the state the test protects.
        assert "onnxruntime" in job["load_error"]
        assert client.get("/healthz").status_code == 200


def test_polling_an_unknown_job_is_404(client_factory):
    with client_factory(ARMOR_INFERENCE_ALLOW_INSTALL="true") as client:
        assert client.get("/v1/models/install/nope").status_code == 404


def test_install_rejects_an_off_shortlist_model(client_factory):
    with client_factory(
        ARMOR_INFERENCE_ALLOW_INSTALL="true", ARMOR_INFERENCE_AUTH_TOKEN=_TOKEN
    ) as client:
        resp = client.post(
            "/v1/models/install",
            json={"task": "prompt_injection", "model_id": "rando/backdoored-bert"},
            headers=_AUTH,
        )
        assert resp.status_code == 400
        assert "vetted shortlist" in resp.json()["detail"]


def test_reload_swaps_a_task_over_http(client_factory):
    with client_factory(ARMOR_INFERENCE_AUTH_TOKEN=_TOKEN) as client:
        resp = client.post(
            "/v1/models/reload",
            json={
                "task": "toxicity",
                "spec": {"runner": "stub", "model_id": "acme/v2", "revision": "r1"},
            },
            headers=_AUTH,
        )
        assert resp.status_code == 200
        assert resp.json()["model_version"] == "acme/v2@r1"

        scored = client.post(
            "/v1/infer/toxicity", json={"text": "x", "model_id": "acme/v2"}, headers=_AUTH
        )
        assert scored.status_code == 200


def test_reload_rejects_an_off_shortlist_model_for_a_heavy_runner(client_factory):
    """Unlike `stub`, a heavy runner's `load()` actually touches the
    filesystem for whatever `model_id` it is given — this is the same
    vetted-shortlist guarantee `install` gets via `resolve_fetch_target`,
    applied to the manual reload path too."""
    with client_factory(ARMOR_INFERENCE_AUTH_TOKEN=_TOKEN) as client:
        resp = client.post(
            "/v1/models/reload",
            json={
                "task": "prompt_injection",
                "spec": {"runner": "classifier", "model_id": "rando/backdoored-bert"},
            },
            headers=_AUTH,
        )
        assert resp.status_code == 400
        assert "vetted shortlist" in resp.json()["detail"]


def test_reload_rejects_an_explicit_artifacts_dir_for_a_heavy_runner(client_factory, tmp_path):
    """`artifacts_dir` is a raw filesystem path with no containment check
    (`resolve_artifact_dir` trusts it verbatim) — accepting it straight from
    the request body would let an authenticated caller point a heavy runner
    at any directory on the container, not just a vetted model's."""
    with client_factory(ARMOR_INFERENCE_AUTH_TOKEN=_TOKEN) as client:
        resp = client.post(
            "/v1/models/reload",
            json={
                "task": "prompt_injection",
                "spec": {
                    "runner": "classifier",
                    "model_id": "protectai/deberta-v3-base-prompt-injection-v2",
                    "artifacts_dir": str(tmp_path),
                },
            },
            headers=_AUTH,
        )
        assert resp.status_code == 400
        assert "artifacts_dir" in resp.json()["detail"]
