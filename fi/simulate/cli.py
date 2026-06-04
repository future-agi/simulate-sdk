from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from xml.etree import ElementTree

from fi.simulate import (
    AgentMemoryLineageEnvironment,
    AgentResponse,
    FrameworkImportManifestEnvironment,
    ObservabilityReplayEnvironment,
    OptimizerPortfolioEnvironment,
    OptimizerTraceEnvironment,
    Persona,
    RedTeamReadinessEnvironment,
    Scenario,
    TestRunner,
    WorkspaceRunEnvironment,
)
from fi.simulate.evaluation import evaluate_agent_report


CLI_SCHEMA_VERSION = "agent-simulate.cli.v1"


class ManifestError(ValueError):
    """Raised when a CLI manifest cannot be executed safely."""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        try:
            result = asyncio.run(run_manifest_command(args))
        except ManifestError as exc:
            print(f"agent-simulate: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"agent-simulate: run failed: {exc}", file=sys.stderr)
            return 3
        if not result.get("outputs_written") and not getattr(args, "quiet", False):
            print(json.dumps(_public_result(result), indent=2, sort_keys=True))
        return int(result.get("exit_code", 1))
    parser.print_help()
    return 2


async def run_manifest_command(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    if args.name:
        manifest["name"] = args.name
    if args.threshold is not None:
        manifest.setdefault("evaluation", {}).setdefault("agent_report", {})["threshold"] = args.threshold
    if args.no_eval:
        manifest.setdefault("evaluation", {})["enabled"] = False

    started = time.time()
    required_env = _required_env(manifest)
    missing_env = [key for key in required_env if not os.environ.get(key)]
    if missing_env:
        raise ManifestError(f"missing required environment variable(s): {', '.join(sorted(missing_env))}")
    _apply_manifest_env(manifest)
    if args.dry_run:
        result = {
            "schema_version": CLI_SCHEMA_VERSION,
            "name": str(manifest.get("name") or manifest_path.stem),
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": {
                "required_env": sorted(required_env),
                "scenario_cases": len(_scenario_dataset(manifest)),
                "environment_count": len(_environment_specs(manifest)),
            },
            "duration_seconds": round(time.time() - started, 4),
        }
        return _write_outputs(result, manifest, args, manifest_path)

    report = await _run_local_text_manifest(manifest, manifest_path)
    evaluation = None
    evaluation_enabled = bool(manifest.get("evaluation")) and manifest.get("evaluation", {}).get("enabled", True) is not False
    if evaluation_enabled:
        agent_report = dict(manifest.get("evaluation", {}).get("agent_report") or manifest.get("agent_report") or {})
        evaluation = evaluate_agent_report(
            report,
            config=dict(agent_report.get("config") or {}),
            threshold=float(agent_report.get("threshold", 0.7)),
            attach=True,
        )

    result = _run_result(
        manifest=manifest,
        report=report,
        evaluation=evaluation,
        duration_seconds=round(time.time() - started, 4),
    )
    return _write_outputs(result, manifest, args, manifest_path)


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency clarity
            raise ManifestError("YAML manifests require PyYAML; use JSON or install PyYAML.") from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ManifestError("manifest root must be an object")
    return dict(data)


async def _run_local_text_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> Any:
    simulation = dict(manifest.get("simulation") or {})
    engine = str(simulation.get("engine") or "local_text").lower().replace("-", "_")
    if engine not in {"local_text", "local"}:
        raise ManifestError(f"unsupported simulation.engine for CLI slice: {engine}")

    scenario = _build_scenario(manifest)
    agent_callback = _build_agent_callback(dict(manifest.get("agent") or {}), manifest_path.parent)
    environments = _build_environments(_environment_specs(manifest), manifest_path.parent)
    return await TestRunner().run_test(
        scenario=scenario,
        agent_callback=agent_callback,
        environment=environments,
        max_turns=int(simulation.get("max_turns", 1)),
        min_turns=int(simulation.get("min_turns", 1)),
        modality=str(simulation.get("modality") or "text"),
        attacks=simulation.get("attacks"),
        auto_execute_tools=bool(simulation.get("auto_execute_tools", True)),
    )


def _build_scenario(manifest: Mapping[str, Any]) -> Scenario:
    raw = dict(manifest.get("scenario") or {})
    if not raw:
        raise ManifestError("manifest requires a scenario")
    dataset = raw.get("dataset")
    if not isinstance(dataset, list) or not dataset:
        raise ManifestError("scenario.dataset must contain at least one persona")
    personas = []
    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, Mapping):
            raise ManifestError(f"scenario.dataset[{index}] must be an object")
        personas.append(
            Persona(
                persona=dict(item.get("persona") or {"name": f"persona-{index}"}),
                situation=str(item.get("situation") or ""),
                outcome=str(item.get("outcome") or ""),
            )
        )
    return Scenario(
        name=str(raw.get("name") or manifest.get("name") or "agent-simulate-cli"),
        description=raw.get("description"),
        dataset=personas,
    )


def _build_agent_callback(agent: Mapping[str, Any], base_dir: Path) -> Callable[..., Any]:
    agent_type = str(agent.get("type") or "scripted").lower().replace("-", "_")
    if agent_type == "scripted":
        responses = list(agent.get("responses") or [])
        if not responses:
            responses = [
                {
                    "content": agent.get("content", "CLI scripted agent response."),
                    "tool_calls": agent.get("tool_calls", []),
                    "metadata": agent.get("metadata", {}),
                }
            ]

        def scripted(input: Any) -> AgentResponse:
            index = int(getattr(input, "turn_index", 0))
            spec = dict(responses[min(index, len(responses) - 1)])
            return AgentResponse(
                content=str(spec.get("content") or ""),
                tool_calls=list(spec.get("tool_calls") or []),
                metadata=dict(spec.get("metadata") or {}),
            )

        return scripted
    if agent_type == "echo":
        prefix = str(agent.get("prefix") or "")

        def echo(input: Any) -> AgentResponse:
            message = getattr(input, "new_message", {}) or {}
            return AgentResponse(content=f"{prefix}{message.get('content', '')}")

        return echo
    if agent_type in {"python", "python_callable"}:
        target = str(agent.get("callable") or "")
        if not target:
            raise ManifestError("agent.type=python requires agent.callable")
        return _load_callable(target, base_dir)
    raise ManifestError(f"unsupported agent.type: {agent_type}")


def _build_environments(specs: Iterable[Mapping[str, Any]], base_dir: Path) -> List[Any]:
    environments = []
    for index, spec in enumerate(specs, start=1):
        if not isinstance(spec, Mapping):
            raise ManifestError(f"environment[{index}] must be an object")
        env_type = str(spec.get("type") or spec.get("kind") or "").lower().replace("-", "_")
        payload = _environment_payload(dict(spec), base_dir)
        if env_type in {"optimizer_backend_portfolio", "optimizer_portfolio"}:
            environments.append(OptimizerPortfolioEnvironment(payload))
        elif env_type in {"optimizer_society_trace", "optimizer_trace"}:
            environments.append(OptimizerTraceEnvironment(payload))
        elif env_type == "agent_memory_lineage":
            environments.append(AgentMemoryLineageEnvironment(payload))
        elif env_type == "red_team_readiness":
            environments.append(RedTeamReadinessEnvironment(payload))
        elif env_type == "framework_import":
            environments.append(FrameworkImportManifestEnvironment(payload))
        elif env_type == "workspace_run_manifest":
            environments.append(WorkspaceRunEnvironment(payload))
        elif env_type == "observability_replay":
            environments.append(ObservabilityReplayEnvironment(payload))
        else:
            raise ManifestError(f"unsupported environment type: {env_type or '<missing>'}")
    return environments


def _environment_payload(spec: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    if "source" in spec:
        source = Path(str(spec["source"]))
        if not source.is_absolute():
            source = base_dir / source
        return load_manifest(source)
    if isinstance(spec.get("data"), Mapping):
        return dict(spec["data"])
    return {
        key: value
        for key, value in spec.items()
        if key not in {"type", "kind", "source"}
    }


def _run_result(
    *,
    manifest: Mapping[str, Any],
    report: Any,
    evaluation: Any,
    duration_seconds: float,
) -> Dict[str, Any]:
    report_payload = _to_plain(report)
    evaluation_payload = _to_plain(evaluation) if evaluation is not None else None
    passed = bool(evaluation_payload.get("passed")) if isinstance(evaluation_payload, Mapping) else True
    summary = {
        "case_count": len(getattr(report, "results", []) or []),
        "evaluation_score": evaluation_payload.get("score") if isinstance(evaluation_payload, Mapping) else None,
        "evaluation_passed": evaluation_payload.get("passed") if isinstance(evaluation_payload, Mapping) else None,
        "metric_averages": (
            evaluation_payload.get("summary", {}).get("metric_averages", {})
            if isinstance(evaluation_payload, Mapping)
            else {}
        ),
    }
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "name": str(manifest.get("name") or "agent-simulate-cli"),
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "summary": summary,
        "report": report_payload,
        "evaluation": evaluation_payload,
        "duration_seconds": duration_seconds,
    }


def _write_outputs(
    result: Dict[str, Any],
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
    manifest_path: Path,
) -> Dict[str, Any]:
    outputs = _output_paths(manifest, args, manifest_path.parent)
    written: List[str] = []
    for path in outputs.get("json", []):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_public_result(result), indent=2, sort_keys=True, default=str), encoding="utf-8")
        written.append(str(path))
    for path in outputs.get("junit", []):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_junit_xml(result), encoding="utf-8")
        written.append(str(path))
    result["outputs_written"] = written
    return result


def _output_paths(manifest: Mapping[str, Any], args: argparse.Namespace, base_dir: Path) -> Dict[str, List[Path]]:
    outputs = {"json": [], "junit": []}
    manifest_outputs = dict(manifest.get("outputs") or {})
    raw_json = [
        *_coerce_list(manifest_outputs.get("json")),
        *_coerce_list(getattr(args, "output", [])),
    ]
    raw_junit = [
        *_coerce_list(manifest_outputs.get("junit")),
        *_coerce_list(getattr(args, "junit", [])),
    ]
    for value in raw_json:
        path = _resolve_output_path(str(value), base_dir)
        if path.suffix.lower() in {".xml", ".junit"} or path.name.endswith(".junit.xml"):
            outputs["junit"].append(path)
        else:
            outputs["json"].append(path)
    outputs["junit"].extend(_resolve_output_path(str(value), base_dir) for value in raw_junit)
    return outputs


def _junit_xml(result: Mapping[str, Any]) -> str:
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), Mapping) else {}
    cases = list(evaluation.get("cases") or []) if isinstance(evaluation, Mapping) else []
    if not cases:
        cases = [{"index": index, "score": 1.0, "passed": result.get("status") == "passed"} for index in range(result.get("summary", {}).get("case_count", 1))]
    failures = sum(1 for case in cases if not case.get("passed"))
    root = ElementTree.Element(
        "testsuites",
        tests=str(len(cases)),
        failures=str(failures),
        errors="0",
        time=str(result.get("duration_seconds", 0.0)),
    )
    suite = ElementTree.SubElement(
        root,
        "testsuite",
        name=str(result.get("name") or "agent-simulate-cli"),
        tests=str(len(cases)),
        failures=str(failures),
        errors="0",
        time=str(result.get("duration_seconds", 0.0)),
    )
    for case in cases:
        case_name = f"case {case.get('index', len(suite))}"
        testcase = ElementTree.SubElement(
            suite,
            "testcase",
            name=case_name,
            classname=str(result.get("name") or "agent-simulate-cli"),
            time="0",
        )
        if not case.get("passed"):
            failure = ElementTree.SubElement(
                testcase,
                "failure",
                message=f"score={case.get('score')}",
            )
            metrics = case.get("metrics") or []
            failure.text = json.dumps({"score": case.get("score"), "metrics": metrics}, default=str)
    return ElementTree.tostring(root, encoding="unicode")


def _required_env(manifest: Mapping[str, Any]) -> List[str]:
    env = dict(manifest.get("env") or {})
    values = [
        *list(manifest.get("required_env") or []),
        *list(env.get("required") or []),
        *list(env.get("required_keys") or []),
    ]
    return sorted({str(value) for value in values if str(value)})


def _apply_manifest_env(manifest: Mapping[str, Any]) -> None:
    env = dict(manifest.get("env") or {})
    values = dict(env.get("set") or env.get("values") or {})
    for key, value in values.items():
        os.environ.setdefault(str(key), str(value))


def _environment_specs(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    simulation = dict(manifest.get("simulation") or {})
    environments = simulation.get("environments", simulation.get("environment", manifest.get("environments", [])))
    if environments is None:
        return []
    if isinstance(environments, Mapping):
        return [environments]
    return list(environments)


def _scenario_dataset(manifest: Mapping[str, Any]) -> List[Any]:
    return list(dict(manifest.get("scenario") or {}).get("dataset") or [])


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _load_callable(target: str, base_dir: Path) -> Callable[..., Any]:
    module_name, _, function_name = target.partition(":")
    if not module_name or not function_name:
        raise ManifestError("python callable must use 'module:function' or 'path.py:function'")
    if module_name.endswith(".py") or "/" in module_name:
        module_path = Path(module_name)
        if not module_path.is_absolute():
            module_path = base_dir / module_path
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ManifestError(f"cannot load python module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    callback = getattr(module, function_name, None)
    if not callable(callback):
        raise ManifestError(f"python callable not found: {target}")
    return callback


def _resolve_output_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value


def _public_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(result)
    payload.pop("outputs_written", None)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-simulate",
        description="Run Future AGI simulation/evaluation manifests locally or in CI.",
    )
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="Run a local simulation/evaluation manifest.")
    run.add_argument("manifest", help="Path to a JSON/YAML manifest.")
    run.add_argument("-o", "--output", action="append", default=[], help="Write JSON output to this path. .xml paths are treated as JUnit.")
    run.add_argument("--junit", action="append", default=[], help="Write compact JUnit XML output.")
    run.add_argument("--threshold", type=float, default=None, help="Override evaluation.agent_report.threshold.")
    run.add_argument("--name", default=None, help="Override the run name.")
    run.add_argument("--no-eval", action="store_true", help="Run simulation only.")
    run.add_argument("--dry-run", action="store_true", help="Validate manifest/env without executing.")
    run.add_argument("--quiet", action="store_true", help="Do not print JSON summary when no output path is configured.")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
