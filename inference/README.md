# armor-inference

The optional model-backed inference sidecar.

**Armor does not need this service.** Every check runs its deterministic path
without it, and the product enforces policy exactly as it does today. This
tier is strictly additive: it exists to score the inputs the rule banks cannot
see — an injection phrased in a way nobody wrote a pattern for, a name or an
address the regex PII tier has no shape for. `armor-api` reaches it only over
HTTP; there is no in-process model loading anywhere in the Rust side.

Status: the service, the contract, batching, caching, the registry, and the
install path are here and tested, and the runners execute real ONNX graphs
(classifier, embedding, NER, NLI — see `armor_inference/runners/`).
`armor-api` calls the service via `ml::escalate` (`crates/api/src/ml.rs`)
whenever `ARMOR_INFERENCE_URL` is set and a check's policy strategy
escalates to it; unset, the tier is off and this service is never reached.

---

## Running it

Rules only, which is the default and involves no ML and no downloads:

```bash
docker compose up
```

Add the tier:

```bash
ARMOR_INFERENCE_URL=http://inference:9000 docker compose --profile ml up   # or: make ml-up
```

(The env var is what has `armor-api` actually call the sidecar — the `ml`
profile alone just starts the container. Leaving it unset boots the sidecar
inert, same as not adding the profile at all.)

Under compose, `ARMOR_INFERENCE_PROFILE=catalog` is on by default
(`docker-compose.yml`): the sidecar tries to load every task in
`config/ml_catalog.yaml` at its pinned model, and since no weights are
fetched yet, each one reports `available: false` (503 on that path) rather
than failing the whole service — see "Getting real models in" below. Run
the bare service with no env vars set and it defaults to
`ARMOR_INFERENCE_PROFILE=stub` instead: three tasks on the dependency-free
`StubRunner`, no ML dependencies, no weights, no network access — enough to
verify the infrastructure (contract, batching, cache, registry) before any
real model is involved.

Outside compose:

```bash
pip install "./inference[dev]"          # serving stack: add [onnx]
python -m uvicorn armor_inference.main:app --port 9000
pytest inference/tests -q
```

---

## Getting real models in

Weights are never baked into the image. A model is a supply-chain artifact: if
it ships inside the image, the image tag stops telling you what is scoring
your traffic. Three supported routes, all of them an explicit operator act.

### 1. Fetch via Docker (no local heavy install)

Every catalog default (`MODELS.md`) installs by downloading an already-
published ONNX graph from the model's own repo — no torch needed at all.
Some vetted *alternatives* publish no ONNX and need a local export instead,
which needs the `armor-inference-fetch` CLI's `[export]` extra — torch +
optimum. Rather than installing that stack on your own machine, run it in a
throwaway container built from `Dockerfile.inference`'s `export` stage:

```bash
make ml-list                          # the vetted shortlist per task
make ml-fetch TASK=prompt_injection   # downloads, exports to ONNX, quantizes,
                                       # prints the sha256 to pin
```

(equivalently: `docker compose --profile ml-fetch run --rm ml-fetch --task
prompt_injection`). It writes into the same `armor-models` volume the
`inference` service mounts — or `${ARMOR_MODELS_DIR}` if you've pointed that
at a bind mount — and prints the ready-to-use task spec below. Nothing here
is stored *and* trusted: the printed sha256 still needs a human to read it
and pin it, same as route 2.

### 2. Mount a directory

Fetch and export offline instead, on a machine that has the heavy stack:

```bash
pip install "./inference[export]"                    # torch + optimum, ~2GB
armor-inference-fetch --list                         # the vetted shortlist
armor-inference-fetch --task prompt_injection        # downloads, exports to
                                                     # ONNX, quantizes, prints
                                                     # the sha256 to pin
```

Either way, put the weights where the container can see them and start with
the pin in place:

```bash
ARMOR_MODELS_DIR=./models docker compose --profile ml up      # or: make ml-up
```

with, on the `inference` service:

```yaml
ARMOR_INFERENCE_TASKS: >
  {"prompt_injection":{"runner":"classifier",
    "model_id":"protectai/deberta-v3-base-prompt-injection-v2",
    "revision":"main","sha256":"<printed by the fetch>","threshold":0.5}}
```

The runner verifies that digest before it loads. A mismatch marks the task
`available: false` and returns 503 — it never serves a model it cannot
identify.

### 3. Install onto a running container

For deployments where preparing a mount is the awkward part:

```bash
curl -XPOST localhost:9000/v1/models/install \
     -H 'content-type: application/json' \
     -d '{"task":"prompt_injection"}'
# → 202 {"job_id": "...", "status": "pending"}

curl localhost:9000/v1/models/install/<job_id>
```

Under compose the sidecar's port is **not published** (it answers only on the
internal network, to `armor-core`), so the host-side `curl` above works
only when you run the service standalone on loopback. Inside the compose
network, drive it from a container that can reach `inference:9000`:

```bash
docker compose exec inference curl -XPOST localhost:9000/v1/models/install \
     -H 'content-type: application/json' \
     -d '{"task":"prompt_injection"}'
```

The fetch runs on a background thread (one at a time), writes into
`ARMOR_INFERENCE_ARTIFACTS_DIR`, and hot-swaps the task onto the new artifact
when it finishes — no restart. Job states:

| status | meaning |
|---|---|
| `pending` / `downloading` / `loading` | in flight |
| `complete` | fetched, verified, and now serving |
| `installed` | on disk and verified, but the task did not load (`load_error`) — do **not** re-download |
| `failed` | the fetch itself failed (`error`); nothing changed |

Two constraints on this path, both deliberate:

* **Off unless enabled** (`ARMOR_INFERENCE_ALLOW_INSTALL`). A service that can
  be told over HTTP to fetch and load new weights is a service whose detection
  layer can be replaced over HTTP. `docker compose --profile ml` turns it on,
  because reaching for that profile is the operator action that makes it
  reasonable; the image's own default is off.
* **Vetted models only.** The endpoint installs what
  `config/ml_catalog.yaml` lists for the task. The CLI's `--allow-unvetted`
  escape hatch is not exposed over HTTP.

Every catalog default installs over route 3 out of the box — no export
needed, since each one downloads an already-published ONNX graph. Some
vetted *alternatives* (`config/ml_catalog.yaml`'s `candidates`, e.g.
`unitary/toxic-bert`) publish no ONNX and need the `[export]` extra instead,
which the serving image only carries when built with `--build-arg
WITH_EXPORT=true` (see `Dockerfile.inference`); pinning one of those without
that build arg makes route 3 fail the same "extra not installed" way it
would outside Docker. Building with `WITH_EXPORT=true` trades image size for
convenience — ~1.9GB instead of ~500MB, torch idle once every task has
loaded — so that install works for those alternatives too, in-process,
the same way `make ml-fetch` or `armor-inference-fetch` would.

That trade also moves *where* an export's resource cost lands, if you take
it. Export + quantization is a CPU- and memory-heavy step — torch briefly
resident at several GB even for these sub-1B-param models — and with
`WITH_EXPORT=true` it runs inside the same long-lived container serving
every other already-loaded task, not a disposable one-off container that
only cost time if it OOM'd. On a memory-constrained host, prefer route 1 (a
separate container) or route 2 for an export-needing model instead of
triggering it through route 3.

---

## GPU acceleration

Every ONNX-backed runner (`classifier`, `ner`, `embedding`, `nli`) auto-
detects hardware at load time — no code change, no per-task config, just
what's installed and present:

1. `ARMOR_INFERENCE_DEVICE` (default `auto`) picks a device family: `cpu`,
   `cuda`, or `rocm`. `auto` takes whichever accelerator the installed
   `onnxruntime` build reports via `get_available_providers()` — CPU if
   none.
2. Session creation is the real test, not just the provider list: a GPU
   that fails to initialize (bad driver, no such device, OOM) makes the
   runner log a warning and fall back to CPU rather than refuse to serve.
   Explicitly requesting `cuda`/`rocm` on a build with no matching
   provider *compiled in* fails loud at load instead — that's a
   misconfigured image, not a transient device problem.
3. `GET /v1/models` reports the execution provider that ended up serving
   each task as `"device": "cpu" | "cuda" | "rocm"` — check this, not just
   whether the box has a GPU, to confirm a task is actually accelerated.
4. An individual task can override the service-wide default via its
   catalog/`ARMOR_INFERENCE_TASKS` spec (`device: "cuda"`), e.g. to pin
   one heavy model onto the GPU while everything else stays on CPU.

**Installing the accelerated stack:**

- **Nvidia**: `pip install "./inference[cuda]"` instead of `[onnx]` — pulls
  `onnxruntime-gpu` (CUDA + TensorRT execution providers). Mutually
  exclusive with `[onnx]`; both packages install the same `onnxruntime`
  module name. Requires CUDA/cuDNN runtime libraries on the host or base
  image — the pip package alone does not vendor them.
- **AMD**: onnxruntime has no ROCm wheel on PyPI. Build onnxruntime from
  source with `--use_rocm`, or install AMD's own prebuilt ROCm-enabled
  onnxruntime distribution, on top of a ROCm-capable base image/host.
  Nothing else in this codebase needs to change — once that build reports
  `ROCMExecutionProvider` in `get_available_providers()`, `auto`-detection
  picks it up the same way it does CUDA.

**Whether it's worth doing** depends heavily on the model: see the sizes
in `config/ml_catalog.yaml` (`servable_runners`'s tasks are all small,
short-sequence classifiers/NER/embedding models — sub-300M params — where
CPU is often competitive or faster once host↔device transfer is counted;
a larger model like `topic_intent`'s `facebook/bart-large-mnli` candidate,
or a future `guard_llm` task, is the shape that actually benefits).
Benchmark on the target hardware rather than assuming; `ARMOR_INFERENCE_DEVICE=cpu`
next to `=cuda` with the same task spec makes that an A/B, not a guess.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ARMOR_INFERENCE_PROFILE` | `stub` | `catalog` loads every task in `ml_catalog.yaml` at its pinned model |
| `ARMOR_INFERENCE_TASKS` | *(unset)* | JSON task→spec map; overrides the profile. Malformed ⇒ refuses to boot |
| `ARMOR_INFERENCE_ARTIFACTS_DIR` | `<repo root>/models` | `/models` in the image — same directory `ARMOR_MODELS_DIR=./models` bind-mounts for a container run |
| `ARMOR_ML_CATALOG` | *(searched)* | Path to the catalog |
| `ARMOR_INFERENCE_ALLOW_INSTALL` | `false` | Enables `POST /v1/models/install` |
| `ARMOR_INFERENCE_AUTH_TOKEN` | *(unset)* | When set, `/v1` requires `Authorization: Bearer`. `POST /v1/models/install` and `POST /v1/models/reload` always require a bearer token regardless — unset, one is generated at boot and printed once to the log |
| `ARMOR_INFERENCE_MAX_BATCH` | `16` | Items coalesced into one forward pass |
| `ARMOR_INFERENCE_MAX_WAIT_MS` | `10` | How long a batch waits to fill |
| `ARMOR_INFERENCE_MAX_QUEUE` | `256` | Over this ⇒ 429 |
| `ARMOR_INFERENCE_BUDGET_MS` | `2000` | Queue-side ceiling; the caller's own deadline is the authoritative one |
| `ARMOR_INFERENCE_CACHE_SIZE` | `4096` | Result-cache entries |
| `ARMOR_INFERENCE_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `rocm` — which execution provider `onnxruntime` sessions use. See [GPU acceleration](#gpu-acceleration) |

---

## The API

```
POST /v1/infer/{task}          score; 404 unknown task, 503 unavailable,
                               409 pin mismatch, 429 saturated
GET  /v1/models                what can serve right now, and why not
GET  /v1/stats                 cache + batcher counters
GET  /healthz                  liveness (never requires a token)
POST /v1/models/install        202 + job_id
GET  /v1/models/install/{id}   poll
POST /v1/models/reload         hot-swap a task onto a spec
```

The request and response shapes are one contract in three places, and they
have to stay in step: `contract.py` here, `crates/inference-client/src/contract.rs`
in Rust, and `proto/inference.proto` for the gRPC transport that may replace
HTTP later.

---

## Testing

The sidecar answers only to `armor-api`: under `docker compose --profile ml`
its port is not published at all, so a host-side request never reaches it.
`armor-api`'s scan handler calls it via `crate::ml::escalate`
(`crates/api/src/ml.rs`) whenever `ARMOR_INFERENCE_URL` is set and a check's
policy strategy includes `local_ml`; with no URL set, `AppState::inference`
is `None` and the pass is skipped entirely.

It can also be exercised standalone, without `armor-api`, through its own
test suite:

```
pytest inference/tests
```

---

## Design notes

Things that look like details and are not.

**A pin that nothing satisfies is a 409, never a score.** If a caller asks for
`model_id=X` and the service has `Y` loaded, it refuses. Silently answering
with `Y` gives the caller results it has no way to know are not the ones it
validated.

**The cache key hashes the exact text.** No lowercasing, no whitespace
collapsing. `AKIAIOSFODNN7EXAMPLE` and its lowercase twin are different inputs
to a secret scanner, and homoglyph-spaced variants are what an evasion test is
made of. The lost hit rate is the price of the cache never being the layer
that loses fidelity.

**A broken model costs one task.** Missing dependency, absent artifact, digest
mismatch, corrupt tokenizer — each marks its own task `available: false` and
leaves the rest serving. Losing a detection layer should not be losing the
tier.

**Saturation answers 429 rather than blocking.** A guardrail that stops
answering under load is a guardrail that gets removed from the request path.

**Per-request `params` are batched, not bypassed.** Items are grouped by their
params so callers with the same ones still coalesce into one forward pass.

**The stub reports no `calibrated_score`.** It has never been benchmarked, and
that field is what the scorecard gate reads. `confidence` is whatever the head
emitted; `calibrated_score` is what a benchmark measured. Collapsing the two
loses the only signal that says whether a threshold means anything.

See `MODELS.md` for the model catalog and licenses.
