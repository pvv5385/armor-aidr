"""Install jobs: getting weights onto a running container.

This is the answer to "the image has no weights, so how does a deployment get
them?" — the second of two supported answers, the first being a `/models`
mount. An operator posts to `/v1/models/install`, the download and export run
in a background thread, and the task is hot-reloaded onto the new artifact
when it finishes. Nothing downloads on boot and nothing downloads because
traffic arrived.

Two properties that keep this from being a supply-chain hole:

* It is **off unless enabled** (`ARMOR_INFERENCE_ALLOW_INSTALL`). A service
  that can be told over HTTP to fetch and load new weights is a service whose
  detection layer can be replaced over HTTP.
* It fetches **from the vetted catalog only**. `resolve_fetch_target`'s
  `allow_unvetted` escape hatch is reachable from the CLI, where a human is
  typing, and deliberately not from this path.

Jobs are in-memory: a restart loses the history, not the artifact. The
artifact is on disk and the digest is in the job's result, which is what
actually needed to survive.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ACTIVE_STATES = ("pending", "downloading", "loading")


class InstallDisabled(RuntimeError):
    """Installs are not enabled on this service."""


@dataclass
class InstallJob:
    """One install, and specifically the difference between its two halves.

    `pending → downloading → loading → complete`, or off to `installed` or
    `failed`. The two terminal-but-not-`complete` states are worth keeping
    apart:

    * `failed` — the fetch did not produce an artifact. Nothing changed.
    * `installed` — the artifact is on disk and its digest is recorded, but
      the task did not come up on it (`load_error`). Collapsing this into
      `failed` would tell an operator to re-download several gigabytes that
      are already there and fine.
    """

    job_id: str
    task: str
    model_id: str
    revision: str
    runner: str
    status: str = "pending"
    sha256: Optional[str] = None
    error: Optional[str] = None
    load_error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InstallManager:
    """Serialized background installs.

    One worker thread, one install at a time. Not for throughput — a fetch is
    minutes long and CPU-bound, and two concurrent exports on a container
    sized for inference will make the service miss its latency budget while
    both crawl. Serializing also means the "is an install running?" question
    has one answer.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._jobs: Dict[str, InstallJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="armor-install")

    def shutdown(self) -> None:
        # Don't wait: a fetch can be minutes long and holding shutdown open
        # for it turns a rolling restart into an outage. The artifact is
        # written atomically enough that a killed fetch leaves a directory
        # that fails verification rather than one that loads wrong.
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _active_job(self) -> Optional[InstallJob]:
        for job in self._jobs.values():
            if job.status in _ACTIVE_STATES:
                return job
        return None

    def start_install(
        self,
        task: str,
        model_id: Optional[str] = None,
        revision: Optional[str] = None,
        *,
        on_complete=None,
    ) -> InstallJob:
        """Queue an install. Raises `InstallDisabled` when installs are off,
        `RuntimeError` when one is already running, `ValueError` on an unknown
        task or an off-shortlist model.

        `on_complete(job, spec)` runs on the worker thread after a successful
        fetch — the service passes a callback that hot-reloads the task.
        """
        if not self.enabled:
            raise InstallDisabled(
                "model installs are disabled; set ARMOR_INFERENCE_ALLOW_INSTALL=true "
                "to enable them, or mount pre-fetched weights at "
                "$ARMOR_INFERENCE_ARTIFACTS_DIR"
            )

        from armor_inference.fetch import resolve_fetch_target

        # Resolve before taking the slot: an unknown task should be a 400, not
        # a job that occupies the single install slot and then fails.
        target = resolve_fetch_target(task, model_id=model_id, revision=revision)

        with self._lock:
            active = self._active_job()
            if active is not None:
                raise RuntimeError(
                    f"install already in progress: {active.model_id} (job {active.job_id})"
                )
            job = InstallJob(
                job_id=str(uuid.uuid4()),
                task=target["task"],
                model_id=target["model_id"],
                revision=target["revision"],
                runner=target["runner"],
            )
            self._jobs[job.job_id] = job

        self._executor.submit(self._run_install, job, target, on_complete)
        logger.info("install job %s started: %s for task '%s'", job.job_id, job.model_id, job.task)
        return job

    def get_job(self, job_id: str) -> Optional[InstallJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[InstallJob]:
        return list(self._jobs.values())

    def _run_install(self, job: InstallJob, target: Dict[str, Any], on_complete) -> None:
        from armor_inference.fetch import fetch_model

        try:
            job.status = "downloading"
            logger.info(
                "job %s: fetching %s@%s → %s",
                job.job_id,
                target["model_id"],
                target["revision"],
                target["dest_dir"],
            )
            job.sha256 = fetch_model(
                target["model_id"],
                target["revision"],
                target["dest_dir"],
                runner=target["runner"],
                expected_sha256=target.get("expected_sha256"),
            )
        except Exception as exc:  # noqa: BLE001 — a failed install must not kill the worker
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = time.time()
            logger.exception("job %s: fetch failed: %s", job.job_id, exc)
            return

        if on_complete is None:
            job.status = "complete"
            job.completed_at = time.time()
            logger.info("job %s: complete (sha256=%s)", job.job_id, job.sha256)
            return

        try:
            job.status = "loading"
            on_complete(
                job,
                {
                    "runner": target["runner"],
                    "model_id": target["model_id"],
                    "revision": target["revision"],
                    "sha256": job.sha256,
                    "artifacts_dir": target["dest_dir"],
                },
            )
            job.status = "complete"
            logger.info("job %s: complete and serving (sha256=%s)", job.job_id, job.sha256)
        except Exception as exc:  # noqa: BLE001
            # The weights are on disk and verified; only the load failed. Say
            # so precisely rather than sending the operator back to re-fetch
            # several gigabytes that are already correct.
            job.status = "installed"
            job.load_error = str(exc)
            logger.exception(
                "job %s: artifact installed (sha256=%s) but the task did not load: %s",
                job.job_id,
                job.sha256,
                exc,
            )
        finally:
            job.completed_at = time.time()
