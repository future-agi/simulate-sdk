import asyncio
import copy
import json
from pathlib import Path

import pytest

from fi.simulate import (
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    detect_manifest_command,
    load_manifest,
    missing_manifest_env,
    optimize_manifest_file,
    run_manifest_file,
    validate_manifest_env,
)


def _local_run_manifest():
    return {
        "version": MANIFEST_SCHEMA_VERSION,
        "name": "public-manifest-api",
        "required_env": ["SIMULATE_PUBLIC_MANIFEST_KEY"],
        "scenario": {
            "name": "public-manifest-api",
            "dataset": [
                {
                    "persona": {"name": "Maya", "role": "sdk-owner"},
                    "situation": "Maya needs a Python API smoke run.",
                    "outcome": "The public manifest API returns a stable payload.",
                }
            ],
        },
        "agent": {
            "type": "scripted",
            "content": "The public manifest runtime executed successfully.",
        },
        "simulation": {"engine": "local_text", "max_turns": 1, "min_turns": 1},
        "evaluation": {"enabled": False},
    }


def test_public_manifest_api_detects_commands_and_validates_env(monkeypatch):
    manifest = _local_run_manifest()

    assert detect_manifest_command(manifest) == "run"
    assert detect_manifest_command({**manifest, "optimization": {"target": {}}}) == "optimize"
    assert detect_manifest_command({**manifest, "redteam": {"attacks": []}}) == "redteam"

    monkeypatch.delenv("SIMULATE_PUBLIC_MANIFEST_KEY", raising=False)
    assert missing_manifest_env(manifest) == ["SIMULATE_PUBLIC_MANIFEST_KEY"]
    with pytest.raises(ManifestError, match="SIMULATE_PUBLIC_MANIFEST_KEY"):
        validate_manifest_env(manifest)


def test_public_run_manifest_file_executes_real_local_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMULATE_PUBLIC_MANIFEST_KEY", "real-local-public-key")
    manifest = _local_run_manifest()
    original = copy.deepcopy(manifest)
    manifest_path = tmp_path / "run.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_manifest(manifest_path)
    result = asyncio.run(
        run_manifest_file(
            manifest_path,
            name="public-api-run",
            no_eval=True,
        )
    )

    assert loaded == manifest
    assert manifest == original
    assert result["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert result["name"] == "public-api-run"
    assert result["status"] == "passed"
    assert result["exit_code"] == 0
    assert result["evaluation"] is None
    assert result["summary"]["case_count"] == 1
    assert result["report"]["results"][0]["messages"][-1]["content"]
    assert "public manifest runtime executed" in result["report"]["results"][0][
        "transcript"
    ].lower()


def test_public_optimize_manifest_file_runs_when_agent_opt_is_available(monkeypatch):
    pytest.importorskip("fi.opt")
    monkeypatch.setenv("SIMULATE_CLI_OPT_EXAMPLE_KEY", "real-local-public-opt-key")
    manifest_path = Path(__file__).resolve().parents[1] / "examples" / "cli_optimizer_portfolio_optimization.json"

    result = optimize_manifest_file(manifest_path, max_candidates=3)

    assert result["status"] == "passed"
    assert result["exit_code"] == 0
    assert result["summary"]["optimization_score"] >= 0.9
    assert result["optimization"]["best_config"]["simulation"]["environments"][0]["data"][
        "selected_optimizer"
    ] == "bandit"
