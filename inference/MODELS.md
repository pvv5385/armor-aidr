# Models

No weights ship with Armor. This file records what
[`config/ml_catalog.yaml`](../config/ml_catalog.yaml) points at, under what
licence, so that pinning a model is a decision made with the terms in view
rather than one made by a default.

Armor's own code, including this sidecar, is Apache-2.0 (see
[`LICENSE`](../LICENSE)). **A model's licence is not Armor's licence.**
Downloading a checkpoint means accepting that model's terms, and some of
them restrict what you may do with the output.

---

## Pinned defaults

| Task | Model | Licence | Commercial use | Size | Hardware |
|---|---|---|---|---|---|
| `prompt_injection` | `protectai/deberta-v3-base-prompt-injection-v2` | Apache-2.0 | yes | 184M | CPU, ~50–100 ms |
| `toxicity` | `Xenova/toxic-bert`* | Apache-2.0 | yes | 110M | CPU, ~40 ms |
| `over_refusal` | `protectai/distilroberta-base-rejection-v1` | Apache-2.0 | yes | 82M | CPU, ~20 ms |
| `pii_ner` | `Davlan/bert-base-multilingual-cased-ner-hrl` | Apache-2.0 | yes | 178M | CPU |
| `topic_intent` | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | yes | 22M | CPU, ~5 ms |

Every default is OSI-permissive. That is a selection criterion, not a
coincidence: a guardrail you cannot deploy commercially without reading a
behavioural appendix is a guardrail with a footnote attached.

\* `Xenova/toxic-bert` is a pre-exported ONNX mirror of `unitary/toxic-bert` —
same weights, same 6-label head, same scores. The Hub repo itself carries no
explicit license field (it's an unofficial community mirror, not a licensing
decision by the original author); the substance is `unitary/toxic-bert`'s
Apache-2.0, since it's a technical re-export with nothing added. Pinned as
the default anyway because it's the only one of the two that installs by
direct download — see "Install path" below. Pin `unitary/toxic-bert` instead
(listed under alternatives) if you'd rather the author's own repo be the
trust boundary.

## Install path: downloaded vs. exported

Every default above installs (`POST /v1/models/install`, `make ml-fetch`,
`armor-inference-fetch`) by downloading an already-published ONNX graph from
the model's own repo — no torch, no local export, and the bytes are
identical for every operator who fetches the same `model_id@revision`. Some
vetted *alternatives* publish no ONNX graph and can only be obtained by
exporting the original checkpoint locally (torch + optimum, the `[export]`
extra — see `Dockerfile.inference`'s `WITH_EXPORT` build arg, off by
default). Each alternative below that needs this is noted as such; picking
one means either rebuilding the serving image with `WITH_EXPORT=true` or
fetching it out-of-band with `make ml-fetch` / `armor-inference-fetch`.

## Vetted alternatives

`armor-inference-fetch --list` prints these with the current pin marked.

| Task | Model | Licence | Notes |
|---|---|---|---|
| `prompt_injection` | `vijil/mbert-prompt-injection` | Apache-2.0 | Multilingual (104 languages); the default is English-only |
| `toxicity` | `unitary/toxic-bert` | Apache-2.0 | The original checkpoint `Xenova/toxic-bert` re-exports — same weights/scores, author's own repo as the trust boundary. Publishes no ONNX itself: **needs `[export]`** (`WITH_EXPORT=true`) |
| `toxicity` | `unitary/multilingual-toxic-xlm-roberta` | Apache-2.0 | Jigsaw-trained, 7 languages. Publishes neither ONNX nor a fast tokenizer: **needs `[export]`** |
| `toxicity` | `gravitee-io/distilbert-multilingual-toxicity-classifier` | **OpenRAIL++** | Fastest of the three — see the caveat below. Publishes ONNX; no export needed |
| `pii_ner` | `iiiorg/piiranha-v1-detect-personal-information` | MIT | PII-specific labels rather than generic PER/ORG/LOC |
| `topic_intent` | `BAAI/bge-small-en-v1.5` | MIT | Slightly stronger, slightly slower |

**OpenRAIL++** permits commercial self-hosting but attaches a behavioural
use-restriction appendix. It is not OSI-open, and the appendix is not
optional. Read it before pinning that row.

## Rejected candidates

- **`Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2`** (`pii_ner`) — the
  best-known ai4privacy-trained PII NER checkpoint, and an earlier candidate
  for this task. Not pinned, default or otherwise: it carries a
  non-commercial (NC) licence, which fails the OSI-permissive bar every row
  above has to clear. `Davlan/bert-base-multilingual-cased-ner-hrl`
  (Apache-2.0, default) and `iiiorg/piiranha-v1-detect-personal-information`
  (MIT, alternative) cover the same unstructured-PII gap it targeted, without
  the restriction.

---

## Digests

The catalog pins `model_id@revision` and deliberately records **no `sha256`**.

A digest in this file would be a digest of a download somebody else did.
Yours is computed over the tree that actually landed on your disk. For the
defaults above (downloaded ONNX, see "Install path") that's the same tree
for everyone who fetches the same `model_id@revision`. For an alternative
that needs a local export, the tree includes a quantization step whose
output depends on your toolchain versions — so a digest pinned from one
export is not guaranteed to match another operator's re-export of the same
model:

```console
$ armor-inference-fetch --task prompt_injection
...
sha256: 3f2a…
```

Put that in the task spec and the runner verifies it on every load. A
mismatch marks the task `available: false` rather than serving weights it
cannot identify. Skipping the pin is allowed — `GET /v1/models` then reports
`sha256: null`, which is the honest description of that state.

---

## What a model does not get you

A model that loads is not a model that has earned the right to block. The
scorecard gate requires a measured benchmark — F1, AUROC, ECE,
false-positive rate, p95 latency, per-language sample counts, staleness —
before a model-backed check may enforce, and a missing metric fails the
gate. Until then a model-backed check runs pinned to warn. The
vendor-reported accuracy figures in the catalog are provenance, not
evidence: none of them were measured on Armor's suites.
