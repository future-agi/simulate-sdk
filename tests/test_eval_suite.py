import json
from pathlib import Path

from fi.simulate import run_eval_suite_file
from fi.simulate.cli import main


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _suite(*, assertion_value: str = "policy") -> dict:
    return {
        "version": "agent-simulate.eval.v1",
        "name": "local-eval-suite",
        "providers": [{"id": "echo", "type": "echo"}],
        "prompts": [{"id": "support", "template": "{{question}}"}],
        "tests": [
            {
                "id": "policy_lookup",
                "vars": {"question": "Where is the policy?"},
                "assert": [{"type": "contains", "value": assertion_value}],
            }
        ],
    }


def test_eval_suite_cli_runs_local_echo_provider_and_writes_json_junit_sarif(tmp_path):
    suite_path = _write_json(tmp_path / "suite.json", _suite())
    output_path = tmp_path / "result.json"
    junit_path = tmp_path / "result.junit.xml"
    sarif_path = tmp_path / "result.sarif.json"

    exit_code = main([
        "eval",
        str(suite_path),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-simulate.eval.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["case_count"] == 1
    assert payload["summary"]["assertion_count"] == 1
    assert payload["summary"]["failed_assertion_count"] == 0
    assert payload["eval_suite"]["cases"][0]["output"] == "Where is the policy?"
    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    assert json.loads(sarif_path.read_text(encoding="utf-8"))["runs"][0]["results"] == []


def test_eval_suite_cli_fails_on_assertion_and_reports_case_path(tmp_path):
    suite_path = _write_json(tmp_path / "suite.json", _suite(assertion_value="refund"))
    output_path = tmp_path / "result.json"
    junit_path = tmp_path / "result.junit.xml"
    sarif_path = tmp_path / "result.sarif.json"

    exit_code = main([
        "eval",
        str(suite_path),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
    ])

    assert exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["summary"]["failed_case_count"] == 1
    assert payload["summary"]["failed_assertion_count"] == 1
    assert payload["evaluation"]["cases"][0]["id"] == "echo::support::policy_lookup"
    assert payload["evaluation"]["findings"][0]["type"] == "eval_assertion_failed"
    assert "failures=\"1\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["runs"][0]["results"][0]["ruleId"] == "eval_assertion_failed"


def test_eval_suite_sdk_matches_cli_payload_shape(tmp_path):
    suite_path = _write_json(tmp_path / "suite.json", _suite())
    output_path = tmp_path / "result.json"

    exit_code = main(["eval", str(suite_path), "--output", str(output_path)])
    cli_payload = json.loads(output_path.read_text(encoding="utf-8"))
    sdk_payload = run_eval_suite_file(suite_path)

    assert exit_code == 0
    assert sdk_payload["kind"] == cli_payload["kind"]
    assert sdk_payload["status"] == cli_payload["status"]
    assert sdk_payload["summary"] == cli_payload["summary"]
    assert sdk_payload["eval_suite"]["cases"] == cli_payload["eval_suite"]["cases"]


def test_eval_suite_resolves_data_and_python_provider_relative_to_config(tmp_path, monkeypatch):
    suite_dir = tmp_path / "nested"
    suite_dir.mkdir()
    (suite_dir / "data.jsonl").write_text(
        json.dumps(
            {
                "id": "policy",
                "vars": {"question": "policy"},
                "assert": [{"type": "contains", "value": "handled policy"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (suite_dir / "provider.py").write_text(
        "def run(prompt, vars, test, provider):\n"
        "    return f\"handled {vars['question']} from {prompt}\"\n",
        encoding="utf-8",
    )
    suite_path = _write_json(
        suite_dir / "suite.json",
        {
            "version": "agent-simulate.eval.v1",
            "name": "relative-suite",
            "providers": [{"id": "py", "type": "python_callable", "target": "provider.py:run"}],
            "prompts": [{"id": "support", "template": "{{question}}"}],
            "tests_file": "data.jsonl",
        },
    )

    monkeypatch.chdir(tmp_path)
    payload = run_eval_suite_file(suite_path)

    assert payload["status"] == "passed"
    assert payload["summary"]["case_count"] == 1
    assert payload["eval_suite"]["cases"][0]["output"] == "handled policy from policy"
