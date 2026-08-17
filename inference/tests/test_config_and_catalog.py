"""Configuration resolution and the shared catalog."""

from __future__ import annotations

import pytest
import yaml

from armor_inference import catalog
from armor_inference.config import InferenceConfig


def test_the_boot_default_needs_no_ml_stack():
    """`docker run` on the image, with nothing mounted and nothing installed,
    has to produce a working service."""
    cfg = InferenceConfig.from_env()
    assert cfg.task_specs
    assert all(spec["runner"] == "stub" for spec in cfg.task_specs.values())
    assert not cfg.allow_install
    assert cfg.auth_token == ""


def test_explicit_tasks_json_wins(monkeypatch):
    monkeypatch.setenv(
        "ARMOR_INFERENCE_TASKS",
        '{"toxicity":{"runner":"classifier","model_id":"unitary/toxic-bert","threshold":0.7}}',
    )
    cfg = InferenceConfig.from_env()
    assert list(cfg.task_specs) == ["toxicity"]
    assert cfg.task_specs["toxicity"]["threshold"] == 0.7


@pytest.mark.parametrize("bad", ["{not json}", "[]", "{}", '{"toxicity": "classifier"}'])
def test_malformed_tasks_json_raises_rather_than_silently_stubbing(monkeypatch, bad):
    """Booting on keyword heuristics because an env var had a trailing comma —
    while /v1/models reports the tasks as available — is the kind of silent
    downgrade nobody notices until it matters."""
    monkeypatch.setenv("ARMOR_INFERENCE_TASKS", bad)
    with pytest.raises(ValueError, match="ARMOR_INFERENCE_TASKS"):
        InferenceConfig.from_env()


def test_catalog_profile_loads_every_catalogued_task(monkeypatch):
    monkeypatch.setenv("ARMOR_INFERENCE_PROFILE", "catalog")
    catalog.load_catalog.cache_clear()
    cfg = InferenceConfig.from_env()
    assert set(cfg.task_specs) == set(catalog.task_names())
    assert cfg.task_specs["prompt_injection"]["runner"] == "classifier"
    assert cfg.task_specs["prompt_injection"]["model_id"]


def test_an_unknown_profile_falls_back_to_stubs(monkeypatch):
    monkeypatch.setenv("ARMOR_INFERENCE_PROFILE", "turbo")
    cfg = InferenceConfig.from_env()
    assert all(spec["runner"] == "stub" for spec in cfg.task_specs.values())


def test_artifact_path_defaults_under_repo_root(monkeypatch):
    cfg = InferenceConfig()
    assert cfg.artifact_path("org__model").endswith("/models/org__model")
    assert "/.armor/" not in cfg.artifact_path("org__model")
    cfg.artifacts_dir = "/models"
    assert cfg.artifact_path("org__model") == "/models/org__model"


# ── The shipped catalog ────────────────────────────────────────────────────


def test_the_shipped_catalog_parses_and_is_complete():
    cat = catalog.load_catalog()
    assert cat["tasks"], "config/ml_catalog.yaml should be found from a checkout"
    servable = set(cat["servable_runners"])
    for task, entry in cat["tasks"].items():
        assert entry["runner"] in servable, f"{task} names a runner nothing can build"
        assert entry.get("model_id"), f"{task} has no pinned model"
        assert entry.get("license"), f"{task} has no license recorded"
        # The digest is the operator's, computed over their own download. A
        # value here would be a supply-chain claim this file cannot back.
        assert "sha256" not in entry, f"{task} must not pin a digest in the catalog"


def test_every_candidate_is_vetted_for_its_task():
    for task in catalog.task_names():
        vetted = catalog.vetted_model_ids(task)
        assert vetted, f"{task} has no installable model"
        assert catalog.load_catalog()["tasks"][task]["model_id"] in vetted


def test_every_task_has_a_display_name_and_its_shortlist():
    # The control-plane UI's model picker needs both: a display name so
    # `over_refusal` reads as more than a raw task key, and the candidate
    # list (possibly just the one pinned model) to choose between.
    overview = catalog.task_overview()
    assert {row["task"] for row in overview} == set(catalog.task_names())
    for row in overview:
        assert row["display_name"] and row["display_name"] != row["task"]
        assert row["candidates"], f"{row['task']} has no installable model"
        assert sum(c["is_current_pin"] for c in row["candidates"]) == 1


def test_task_spec_forwards_an_operator_pinned_digest(tmp_path, monkeypatch):
    # The shipped catalog never carries one (see the test above) — but an
    # operator's own copy may, once they've reviewed a fetch, and it has to
    # reach `runners/_artifacts.py`'s `verify_pinned` at boot the same way
    # `model_id`/`revision` do or a swapped on-disk artifact goes unnoticed.
    path = tmp_path / "ml_catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tasks": {
                    "t": {
                        "runner": "classifier",
                        "model_id": "a/one",
                        "sha256": "deadbeef" * 8,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARMOR_ML_CATALOG", str(path))
    catalog.load_catalog.cache_clear()

    assert catalog.task_spec("t")["sha256"] == "deadbeef" * 8


def test_a_missing_catalog_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("ARMOR_ML_CATALOG", str(tmp_path / "absent.yaml"))
    monkeypatch.setattr(catalog, "_SEARCH_PATHS", (lambda: str(tmp_path / "absent.yaml"),))
    catalog.load_catalog.cache_clear()
    assert catalog.load_catalog()["tasks"] == {}
    assert catalog.task_names() == []


def test_a_malformed_catalog_raises(monkeypatch, tmp_path):
    path = tmp_path / "ml_catalog.yaml"
    path.write_text("tasks: [this is a list, not a mapping\n", encoding="utf-8")
    monkeypatch.setenv("ARMOR_ML_CATALOG", str(path))
    catalog.load_catalog.cache_clear()
    with pytest.raises(catalog.CatalogError):
        catalog.load_catalog()


def test_candidates_can_be_filtered_to_open_licenses(tmp_path, monkeypatch):
    path = tmp_path / "ml_catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tasks": {"t": {"runner": "classifier", "model_id": "a/one"}},
                "candidates": {
                    "t": [
                        {"model_id": "a/one", "open_license": True},
                        {"model_id": "b/two", "open_license": False},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARMOR_ML_CATALOG", str(path))
    catalog.load_catalog.cache_clear()

    assert [c["model_id"] for c in catalog.candidates("t")] == ["a/one", "b/two"]
    assert [c["model_id"] for c in catalog.candidates("t", open_only=True)] == ["a/one"]
    assert catalog.candidates("t")[0]["is_current_pin"] is True
