"""The `armor-inference` HTTP service.

    POST /v1/infer/{task}          score against the task's model
    GET  /v1/models                what this pool can serve right now
    GET  /v1/stats                 cache + batcher metrics
    GET  /v1/hardware              host CPU/RAM/GPU inventory
    GET  /healthz                  liveness
    POST /v1/models/install        operator-triggered fetch (202 + job_id)
    GET  /v1/models/install/{id}   poll it
    POST /v1/models/reload         hot-swap a task's runner

Boots on the dependency-free `StubRunner`: no ML stack, no weights, no
network. Everything heavier is something an operator turned on.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from armor_inference import __version__
from armor_inference.batching import Saturated, params_key
from armor_inference.cache import ContentHashCache, content_key
from armor_inference.catalog import task_overview, vetted_model_ids
from armor_inference.config import InferenceConfig
from armor_inference.contract import (
    BatchInferResponse,
    InferRequest,
    InferResponse,
    InferResult,
    ModelInfo,
)
from armor_inference.hardware import get_hardware_info
from armor_inference.install import InstallDisabled, InstallManager
from armor_inference.registry import RunnerRegistry

# Nothing configures the root logger otherwise: uvicorn's dictConfig only
# touches its own loggers, so without this every INFO line this package emits
# (a successful load, an install job's progress) is dropped — and the failure
# mode is a silent one, where a task that never loaded looks the same as a
# task nobody asked about.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

logger = logging.getLogger(__name__)

# How long an install job waits for its hot-reload to finish.
_RELOAD_TIMEOUT_S = 300


def _publish_mutation_token(token: str, token_file: str) -> None:
    """Best-effort: drop the freshly generated mutation token at
    `token_file` so `armor-core` (`crates/api/src/control_plane.rs`'s
    `resolve_inference_token`) can pick it up without an operator wiring
    `ARMOR_INFERENCE_AUTH_TOKEN` on both sides by hand.

    `docker-compose.yml` mounts a small shared volume at this path in both
    containers — armor-core's mount is read-only, so the two containers that
    can already reach the sidecar's install endpoint (armor-core, by design
    the sidecar's only network caller — see inference/README.md) are the
    only ones that can ever read this file; nothing widens who could
    authenticate as a result of this existing.

    Silently gives up on any `OSError` (permission denied, parent directory
    missing) rather than failing boot — a bare, non-compose run has nothing
    mounted at this path and simply keeps today's behavior: a token that's
    only ever in the boot log.
    """
    try:
        path = Path(token_file)
        path.write_text(token)
        os.chmod(path, 0o644)
    except OSError:
        logger.debug("could not publish mutation token to %s — continuing without it", token_file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = InferenceConfig.from_env()
    registry = RunnerRegistry(config)
    await registry.start_all()
    app.state.config = config
    app.state.registry = registry
    app.state.cache = ContentHashCache(maxsize=config.cache_maxsize)
    app.state.install_manager = InstallManager(enabled=config.allow_install)
    # install/reload can change what a task serves, so — unlike scoring,
    # which stays open with no configuration — they always need a bearer
    # token. An operator who set ARMOR_INFERENCE_AUTH_TOKEN gets one shared
    # secret for everything; one who didn't still gets these two endpoints
    # protected, via a token generated fresh each boot and logged once. A
    # quickstart never needs to set anything to try /v1/infer, and never gets
    # a wide-open model-swap endpoint either.
    app.state.mutation_token = config.auth_token or secrets.token_urlsafe(32)
    if not config.auth_token:
        logger.warning(
            "ARMOR_INFERENCE_AUTH_TOKEN not set — generated a token for this boot to "
            "protect POST /v1/models/install and /v1/models/reload "
            "(Authorization: Bearer %s). Set ARMOR_INFERENCE_AUTH_TOKEN for a stable "
            "token across restarts.",
            app.state.mutation_token,
        )
        _publish_mutation_token(app.state.mutation_token, config.token_file)
    # Install jobs run on a worker thread and finish by hot-reloading a task,
    # which is async. Capture the loop now so the callback has something to
    # schedule onto.
    app.state.loop = asyncio.get_running_loop()

    models = registry.list_models()
    available = [m.task for m in models if m.available]
    unavailable = [m.task for m in models if not m.available]
    logger.info("armor-inference %s ready — serving: %s", __version__, available or "(none)")
    if unavailable:
        logger.warning(
            "tasks configured but unavailable: %s — they return 503; the service is up",
            unavailable,
        )
    try:
        yield
    finally:
        app.state.install_manager.shutdown()
        await registry.stop_all()


app = FastAPI(title="Armor Inference Service", version=__version__, lifespan=lifespan)


# ── Auth ───────────────────────────────────────────────────────────────────


def _check_bearer(authorization: Optional[str], expected: str) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    # Constant-time: the comparison is over a shared secret, and `==` on
    # strings leaks its length and its matching prefix through timing.
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


async def require_token(
    request: Request, authorization: Optional[str] = Header(default=None)
) -> None:
    """No-op unless `ARMOR_INFERENCE_AUTH_TOKEN` is set. `/healthz` is exempt
    so a liveness probe does not need a credential."""
    expected = getattr(request.app.state, "config", None)
    expected = expected.auth_token if expected else ""
    if not expected:
        return
    _check_bearer(authorization, expected)


async def require_mutation_token(
    request: Request, authorization: Optional[str] = Header(default=None)
) -> None:
    """Unlike `require_token`, this never no-ops: `app.state.mutation_token`
    (set in `lifespan`) is always populated, either from
    `ARMOR_INFERENCE_AUTH_TOKEN` or a token generated for this boot — so
    install/reload always require a real bearer token, on a zero-config
    startup too."""
    _check_bearer(authorization, request.app.state.mutation_token)


# ── Scoring ────────────────────────────────────────────────────────────────


def _result(out, model_version: str) -> InferResult:
    return InferResult(
        decision=out.decision,
        risk_score=out.risk_score,
        confidence=out.confidence,
        label_scores=out.label_scores,
        calibrated_score=out.calibrated_score,
        threshold=out.threshold,
        model_version=model_version,
    )


def _resolve_batcher(registry: RunnerRegistry, task: str, req: InferRequest):
    """Route to the batcher serving the model this request asked for.

    Unpinned → the task's active slot. Pinned → the active slot when it
    matches, else a loaded variant of exactly that model. A pin nothing
    satisfies is a **409, never a silent score against whatever is loaded** —
    a caller that asked for a specific model and got a different one has no
    way to know its results are not what it validated.
    """
    if not registry.known_task(task):
        raise HTTPException(status_code=404, detail=f"unknown task '{task}'")

    if not req.model_id:
        batcher = registry.get(task)
        if batcher is None:
            info = next((m for m in registry.list_models() if m.task == task), None)
            detail = f"task '{task}' unavailable"
            if info is not None and info.detail:
                detail = f"{detail}: {info.detail}"
            raise HTTPException(status_code=503, detail=detail)
        return batcher, registry.model_version(task) or "unknown"

    resolved = registry.get_for(task, req.model_id, req.revision)
    if resolved is not None:
        return resolved
    if registry.get(task) is None:
        raise HTTPException(status_code=503, detail=f"task '{task}' unavailable")
    want = req.model_id + (f"@{req.revision}" if req.revision else "")
    raise HTTPException(
        status_code=409,
        detail=(
            f"model pin mismatch: requested {want}, "
            f"loaded {registry.model_version(task) or 'nothing'}"
        ),
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(require_token)])
async def list_models(request: Request):
    return {"models": [m.model_dump() for m in request.app.state.registry.list_models()]}


@app.get("/v1/models/catalog", dependencies=[Depends(require_token)])
async def models_catalog():
    """Static catalog metadata (display name, rationale, vetted shortlist)
    per task — distinct from `GET /v1/models`'s live registry state."""
    return {"tasks": task_overview()}


@app.get("/v1/stats", dependencies=[Depends(require_token)])
async def stats(request: Request):
    return {
        "cache": request.app.state.cache.stats(),
        "batchers": request.app.state.registry.batcher_stats(),
    }


@app.get("/v1/hardware", dependencies=[Depends(require_token)])
async def hardware():
    """Best-effort host inventory (CPU, RAM, GPU) for the control-plane UI's
    inference page — not per-task, unlike `GET /v1/models`'s `device` field."""
    return get_hardware_info().model_dump()


@app.post("/v1/infer/{task}", dependencies=[Depends(require_token)])
async def infer(task: str, req: InferRequest, request: Request):
    registry: RunnerRegistry = request.app.state.registry
    cache: ContentHashCache = request.app.state.cache
    batcher, model_version = _resolve_batcher(registry, task, req)

    texts = req.items()
    t0 = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    # Explicit batch: no per-item cache. The caller asked for N scores in one
    # call and gets N back, in order.
    if req.texts is not None:
        # Submit concurrently so the batcher can coalesce them into one
        # forward pass; `return_exceptions` so a rejection on one item does
        # not leave the others as unretrieved exceptions.
        outs = await asyncio.gather(
            *(batcher.submit(t, req.params) for t in texts), return_exceptions=True
        )
        for out in outs:
            if isinstance(out, Saturated):
                raise HTTPException(status_code=429, detail=str(out))
            if isinstance(out, BaseException):
                raise out
        return BatchInferResponse(
            results=[_result(o, model_version) for o in outs],
            latency_ms=elapsed_ms(),
            model_version=model_version,
        )

    # Single item: content-hash cache, then the batcher. The key includes the
    # params, so two callers with different runner params never share a
    # verdict.
    text = texts[0]
    key = content_key(f"{model_version}|{params_key(req.params)}", text)
    cached = cache.get(key)
    if cached is not None:
        return InferResponse(
            **_result(cached, model_version).model_dump(),
            latency_ms=elapsed_ms(),
            cached=True,
        )
    try:
        out = await batcher.submit(text, req.params)
    except Saturated as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    cache.put(key, out)
    return InferResponse(
        **_result(out, model_version).model_dump(), latency_ms=elapsed_ms(), cached=False
    )


# ── Model lifecycle ────────────────────────────────────────────────────────


class InstallRequest(BaseModel):
    task: str
    model_id: Optional[str] = None
    revision: Optional[str] = None


class ReloadRequest(BaseModel):
    task: str
    spec: dict


@app.post("/v1/models/install", dependencies=[Depends(require_mutation_token)])
async def install_model(req: InstallRequest, request: Request):
    """Start a background install; returns 202 and a `job_id` to poll.

    This is how a running container gets weights when they are not mounted.
    Disabled unless `ARMOR_INFERENCE_ALLOW_INSTALL` is set.
    """
    manager: InstallManager = request.app.state.install_manager
    registry: RunnerRegistry = request.app.state.registry
    loop: asyncio.AbstractEventLoop = request.app.state.loop

    def hot_reload(job, spec) -> None:
        # Called on the install worker thread. Hop back onto the loop the
        # registry lives on, and block this thread until the swap is done so
        # the job is not reported complete before the model is actually
        # serving. The timeout is generous — loading a multi-gigabyte graph is
        # slow — but bounded, so a wedged load leaves the job in `installed`
        # with a reason rather than occupying the install slot forever.
        future = asyncio.run_coroutine_threadsafe(registry.reload_task(job.task, spec), loop)
        info = future.result(timeout=_RELOAD_TIMEOUT_S)
        if not info.available:
            raise RuntimeError(f"installed {job.model_id} but it failed to load: {info.detail}")

    try:
        job = manager.start_install(
            req.task, model_id=req.model_id, revision=req.revision, on_complete=hot_reload
        )
    except InstallDisabled as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(status_code=202, content=job.to_dict())


@app.get("/v1/models/install/{job_id}", dependencies=[Depends(require_token)])
async def get_install_status(job_id: str, request: Request):
    job = request.app.state.install_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job '{job_id}'")
    return job.to_dict()


def _validate_reload_spec(task: str, spec: dict) -> None:
    """`install` only ever reaches a heavy runner through
    `resolve_fetch_target`, which pins `model_id` to the task's vetted
    shortlist and computes `artifacts_dir` itself. This endpoint takes a raw,
    caller-supplied `spec` instead — the manual escape hatch for hot-swapping
    onto weights an operator already placed on disk — so it has to apply the
    same two guarantees by hand, or a caller with the mutation token could use
    it to point a heavy runner at an arbitrary local path or an unreviewed
    model. `stub` carries neither risk (no file or network access), so it is
    exempt.
    """
    kind = spec.get("runner", "stub")
    if kind == "stub":
        return
    if spec.get("artifacts_dir"):
        raise HTTPException(
            status_code=400,
            detail=(
                "'artifacts_dir' cannot be set directly on a reload spec for a "
                f"'{kind}' runner; the artifact must resolve from "
                "$ARMOR_INFERENCE_ARTIFACTS_DIR/<model_id> like an install does"
            ),
        )
    model_id = spec.get("model_id")
    if model_id:
        vetted = vetted_model_ids(task)
        if vetted and model_id not in vetted:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{model_id}' is not on the vetted shortlist for task '{task}'. "
                    f"Choose one of {vetted}."
                ),
            )


@app.post(
    "/v1/models/reload", dependencies=[Depends(require_mutation_token)], response_model=ModelInfo
)
async def reload_model(req: ReloadRequest, request: Request):
    """Hot-swap a task's active runner onto `spec` — the manual counterpart of
    what an install job does for itself."""
    _validate_reload_spec(req.task, req.spec)
    registry: RunnerRegistry = request.app.state.registry
    return await registry.reload_task(req.task, req.spec)
