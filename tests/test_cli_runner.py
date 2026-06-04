import copy
import json
import pytest

from fi.simulate.cli import main


def _portfolio_manifest():
    return {
        "version": "agent-simulate.cli.v1",
        "name": "optimizer-portfolio-cli",
        "required_env": ["SIMULATE_CLI_TEST_KEY"],
        "scenario": {
            "name": "optimizer-portfolio-cli",
            "dataset": [
                {
                    "persona": {"name": "Riya", "role": "ci-owner"},
                    "situation": "Riya needs CI evidence for optimizer backend allocation.",
                    "outcome": "The optimizer portfolio gate passes.",
                }
            ],
        },
        "agent": {
            "type": "scripted",
            "content": "Optimizer portfolio inspected from the CLI.",
            "tool_calls": [
                {"id": "status", "name": "optimizer_portfolio_status", "arguments": {}},
                {"id": "list", "name": "list_optimizer_backends", "arguments": {"status": "completed"}},
                {"id": "backend", "name": "inspect_optimizer_backend", "arguments": {"optimizer": "bandit"}},
                {"id": "ablation", "name": "inspect_optimizer_ablation", "arguments": {}},
            ],
        },
        "simulation": {
            "engine": "local_text",
            "max_turns": 1,
            "min_turns": 1,
            "environments": [
                {
                    "type": "optimizer_backend_portfolio",
                    "data": {
                        "name": "cli-portfolio",
                        "selected_optimizer": "bandit",
                        "final_score": 1.0,
                        "improved": True,
                        "rollback_decision": {"rollback_required": False},
                        "feedback_cases": [{"id": "case"}],
                        "diagnoses": [{"component": "multi_agent"}],
                        "search_paths": [
                            "optimizer.backend_portfolio.backends",
                            "optimizer.backend_selector.policy",
                        ],
                        "backend_plan": [
                            {"optimizer": "agent", "rank": 1},
                            {"optimizer": "tpe", "rank": 2},
                            {"optimizer": "bandit", "rank": 3},
                        ],
                        "backend_runs": [
                            {"optimizer": "agent", "status": "completed", "final_score": 0.84, "improved": True},
                            {"optimizer": "tpe", "status": "completed", "final_score": 0.91, "improved": True},
                            {"optimizer": "bandit", "status": "completed", "final_score": 1.0, "improved": True},
                        ],
                        "backend_lineage": [
                            {"optimizer": "agent", "selection_relation": "equivalent", "patch_paths": ["optimizer.backend_portfolio.backends"]},
                            {"optimizer": "tpe", "selection_relation": "supporting", "patch_paths": ["optimizer.backend_selector.policy"]},
                            {"optimizer": "bandit", "selection_relation": "selected", "patch_paths": ["optimizer.backend_portfolio.backends"]},
                        ],
                        "ablation_report": {
                            "selected_optimizer": "bandit",
                            "selected_candidate_id": "candidate_bandit",
                            "dependency": "backend_consensus",
                            "consensus_backends": ["agent", "tpe"],
                            "selected_backend_required": False,
                        },
                    },
                }
            ],
        },
        "evaluation": {
            "agent_report": {
                "threshold": 0.9,
                "config": {
                    "required_tools": [
                        "optimizer_portfolio_status",
                        "list_optimizer_backends",
                        "inspect_optimizer_backend",
                        "inspect_optimizer_ablation",
                    ],
                    "available_tools": [
                        "optimizer_portfolio_status",
                        "list_optimizer_backends",
                        "inspect_optimizer_backend",
                        "inspect_optimizer_ablation",
                    ],
                    "required_optimizer_portfolio": [
                        "optimizer_portfolio",
                        "backend_plan",
                        "backend_run",
                        "backend_lineage",
                        "selected_optimizer",
                        "ablation",
                        "consensus",
                        "selected_relation",
                        "diagnostic",
                        "feedback",
                        "search_path",
                        "improvement",
                        "rollback_decision",
                        "agent",
                        "tpe",
                        "bandit",
                    ],
                    "optimizer_portfolio_quality": {
                        "required_backends": ["agent", "tpe", "bandit"],
                        "required_completed_backends": ["agent", "tpe", "bandit"],
                        "required_consensus_backends": ["agent", "tpe"],
                        "required_selection_relations": ["selected", "equivalent", "supporting"],
                        "required_dependencies": ["backend_consensus"],
                        "required_search_paths": [
                            "optimizer.backend_portfolio.backends",
                            "optimizer.backend_selector.policy",
                        ],
                        "min_backend_plan_count": 3,
                        "min_backend_run_count": 3,
                        "min_completed_backends": 3,
                        "min_lineage_count": 3,
                        "min_consensus_backends": 2,
                        "min_feedback_cases": 1,
                        "min_diagnostics": 1,
                        "min_search_paths": 2,
                        "min_improved_backends": 3,
                        "min_final_score": 0.99,
                        "max_failed_backends": 0,
                        "require_selected_optimizer": True,
                        "require_backend_plan": True,
                        "require_backend_runs": True,
                        "require_backend_lineage": True,
                        "require_completed_backend": True,
                        "require_ablation": True,
                        "require_consensus": True,
                        "require_selected_relation": True,
                        "require_diagnostics": True,
                        "require_feedback": True,
                        "require_search_paths": True,
                        "require_improvement": True,
                        "require_rollback_decision": True,
                    },
                    "metric_weights": {
                        "optimizer_portfolio_coverage": 5.0,
                        "optimizer_portfolio_quality": 10.0,
                        "final_response_quality": 1.0,
                    },
                },
            }
        },
    }


def _bad_portfolio_data():
    return {
        "name": "cli-portfolio-bad",
        "selected_optimizer": "agent",
        "final_score": 0.2,
        "improved": False,
        "rollback_decision": {},
        "feedback_cases": [],
        "diagnoses": [],
        "search_paths": [],
        "backend_plan": [{"optimizer": "agent", "rank": 1}],
        "backend_runs": [{"optimizer": "agent", "status": "completed", "final_score": 0.2}],
        "backend_lineage": [],
        "ablation_report": {
            "selected_optimizer": "agent",
            "selected_candidate_id": "candidate_agent",
            "dependency": "single_backend",
            "consensus_backends": [],
            "selected_backend_required": True,
        },
    }


def _good_portfolio_data():
    data = copy.deepcopy(_portfolio_manifest()["simulation"]["environments"][0]["data"])
    data["name"] = "cli-portfolio-good"
    return data


def _optimization_manifest():
    manifest = _portfolio_manifest()
    manifest["name"] = "optimizer-portfolio-cli-optimization"
    manifest["required_env"] = ["SIMULATE_CLI_OPT_TEST_KEY"]
    manifest["simulation"]["environments"] = []
    manifest["optimization"] = {
        "threshold": 0.9,
        "target": {
            "name": "optimizer-portfolio-cli",
            "layers": ["harness", "multi_agent", "evaluator"],
            "base_config": {
                "simulation": {
                    "environments": [
                        {
                            "type": "optimizer_backend_portfolio",
                            "data": _bad_portfolio_data(),
                        }
                    ]
                }
            },
            "search_space": {
                "simulation.environments.0.data": [
                    _bad_portfolio_data(),
                    _good_portfolio_data(),
                ]
            },
        },
        "optimizer": {
            "max_candidates": 3,
            "diagnostic_score_threshold": 0.9,
        },
    }
    return manifest


def test_cli_runner_executes_manifest_and_writes_json_and_junit(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMULATE_CLI_TEST_KEY", "real-local-test-key")
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "result.json"
    junit_path = tmp_path / "result.junit.xml"
    manifest_path.write_text(json.dumps(_portfolio_manifest()), encoding="utf-8")

    exit_code = main(["run", str(manifest_path), "--output", str(output_path), "--junit", str(junit_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["summary"]["metric_averages"]["optimizer_portfolio_coverage"] == 1.0
    assert payload["summary"]["metric_averages"]["optimizer_portfolio_quality"] == 1.0
    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")


def test_cli_runner_fails_fast_on_missing_required_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SIMULATE_CLI_TEST_KEY", raising=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_portfolio_manifest()), encoding="utf-8")

    assert main(["run", str(manifest_path), "--quiet"]) == 2


def test_cli_runner_dry_run_validates_manifest_without_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMULATE_CLI_TEST_KEY", "real-local-test-key")
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "dry-run.json"
    manifest_path.write_text(json.dumps(_portfolio_manifest()), encoding="utf-8")

    exit_code = main(["run", str(manifest_path), "--dry-run", "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["summary"]["scenario_cases"] == 1
    assert payload["summary"]["environment_count"] == 1


def test_cli_runner_optimizes_manifest_search_paths_and_writes_outputs(tmp_path, monkeypatch):
    pytest.importorskip("fi.opt")
    monkeypatch.setenv("SIMULATE_CLI_OPT_TEST_KEY", "real-local-opt-key")
    manifest_path = tmp_path / "optimize.json"
    output_path = tmp_path / "optimize-result.json"
    junit_path = tmp_path / "optimize-result.junit.xml"
    manifest_path.write_text(json.dumps(_optimization_manifest()), encoding="utf-8")

    exit_code = main([
        "optimize",
        str(manifest_path),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.9
    assert payload["optimization"]["best_config"]["simulation"]["environments"][0]["data"]["selected_optimizer"] == "bandit"
    assert payload["optimization"]["history"]
    assert min(item["score"] for item in payload["optimization"]["history"]) < 0.9
    assert max(item["score"] for item in payload["optimization"]["history"]) >= 0.9
    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")


def test_cli_runner_optimize_dry_run_reports_search_space(tmp_path, monkeypatch):
    pytest.importorskip("fi.opt")
    monkeypatch.setenv("SIMULATE_CLI_OPT_TEST_KEY", "real-local-opt-key")
    manifest_path = tmp_path / "optimize.json"
    output_path = tmp_path / "optimize-dry-run.json"
    manifest_path.write_text(json.dumps(_optimization_manifest()), encoding="utf-8")

    exit_code = main(["optimize", str(manifest_path), "--dry-run", "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["summary"]["search_path_count"] == 1
    assert payload["summary"]["max_candidates"] == 3


def test_cli_runner_optimize_fails_fast_on_missing_required_env(tmp_path, monkeypatch):
    pytest.importorskip("fi.opt")
    monkeypatch.delenv("SIMULATE_CLI_OPT_TEST_KEY", raising=False)
    manifest_path = tmp_path / "optimize.json"
    manifest_path.write_text(json.dumps(_optimization_manifest()), encoding="utf-8")

    assert main(["optimize", str(manifest_path), "--quiet"]) == 2
