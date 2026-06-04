from __future__ import annotations

import copy
import importlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional


CLI_SCHEMA_VERSION = "agent-simulate.cli.v1"
MANIFEST_SCHEMA_VERSION = CLI_SCHEMA_VERSION


class ManifestError(ValueError):
    """Raised when a simulation manifest cannot be executed safely."""


@dataclass(frozen=True)
class ManifestRunOptions:
    name: Optional[str] = None
    threshold: Optional[float] = None
    no_eval: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class ManifestOptimizationOptions:
    name: Optional[str] = None
    threshold: Optional[float] = None
    max_candidates: Optional[int] = None
    dry_run: bool = False


def load_manifest_file(path: str | Path) -> Dict[str, Any]:
    """Load a JSON/YAML manifest using the same validation as the CLI."""

    return _cli().load_manifest(Path(path).expanduser().resolve())


load_manifest = load_manifest_file


def detect_manifest_command(
    manifest: Mapping[str, Any],
) -> Literal["run", "redteam", "optimize"]:
    """Return the default command for a manifest shape."""

    if manifest.get("redteam") is not None or manifest.get("red_team") is not None:
        return "redteam"
    if manifest.get("optimization") is not None:
        return "optimize"
    return "run"


def required_manifest_env(manifest: Mapping[str, Any]) -> list[str]:
    """Return all environment keys required before a manifest can execute."""

    return sorted(_cli()._required_env(manifest))


def missing_manifest_env(manifest: Mapping[str, Any]) -> list[str]:
    """Return required environment keys that are not present in os.environ."""

    return [key for key in required_manifest_env(manifest) if not os.environ.get(key)]


def validate_manifest_env(manifest: Mapping[str, Any]) -> None:
    """Raise ManifestError if required environment variables are missing."""

    missing_env = missing_manifest_env(manifest)
    if missing_env:
        raise ManifestError(
            "missing required environment variable(s): "
            f"{', '.join(sorted(missing_env))}"
        )


def apply_manifest_env(manifest: Mapping[str, Any]) -> None:
    """Apply manifest env.set values to process env, matching CLI behavior."""

    _cli()._apply_manifest_env(manifest)


async def run_local_text_manifest(
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
) -> Any:
    """Run a local_text/local manifest and return the raw TestReport."""

    return await _cli()._run_local_text_manifest(
        manifest,
        Path(manifest_path).expanduser().resolve(),
    )


def evaluate_manifest_report(manifest: Mapping[str, Any], report: Any) -> Any:
    """Score a report with the manifest's evaluation.agent_report block."""

    return _cli()._evaluate_manifest_report(manifest, report)


async def run_manifest_file(
    path: str | Path,
    *,
    options: Optional[ManifestRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    no_eval: Optional[bool] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load and execute a run manifest, returning the CLI-compatible payload."""

    manifest_path = Path(path).expanduser().resolve()
    return await run_manifest(
        load_manifest_file(manifest_path),
        manifest_path=manifest_path,
        options=_run_options(
            options,
            name=name,
            threshold=threshold,
            no_eval=no_eval,
            dry_run=dry_run,
        ),
    )


async def run_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    options: Optional[ManifestRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    no_eval: Optional[bool] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute a run manifest without writing CLI artifacts."""

    cli = _cli()
    opts = _run_options(
        options,
        name=name,
        threshold=threshold,
        no_eval=no_eval,
        dry_run=dry_run,
    )
    runtime_manifest = copy.deepcopy(dict(manifest))
    manifest_path = Path(manifest_path).expanduser().resolve()
    if opts.name:
        runtime_manifest["name"] = opts.name
    if opts.threshold is not None:
        runtime_manifest.setdefault("evaluation", {}).setdefault(
            "agent_report", {}
        )["threshold"] = opts.threshold
    if opts.no_eval:
        runtime_manifest.setdefault("evaluation", {})["enabled"] = False

    started = time.time()
    validate_manifest_env(runtime_manifest)
    apply_manifest_env(runtime_manifest)
    if opts.dry_run:
        return {
            "schema_version": CLI_SCHEMA_VERSION,
            "name": str(runtime_manifest.get("name") or manifest_path.stem),
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": {
                "required_env": required_manifest_env(runtime_manifest),
                "scenario_cases": len(cli._scenario_dataset(runtime_manifest)),
                "environment_count": len(cli._environment_specs(runtime_manifest)),
            },
            "duration_seconds": round(time.time() - started, 4),
        }

    report = await run_local_text_manifest(runtime_manifest, manifest_path)
    evaluation = evaluate_manifest_report(runtime_manifest, report)
    return cli._run_result(
        manifest=runtime_manifest,
        report=report,
        evaluation=evaluation,
        duration_seconds=round(time.time() - started, 4),
    )


async def redteam_manifest_file(
    path: str | Path,
    *,
    options: Optional[ManifestRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load and execute a red-team manifest, returning the CLI-compatible payload."""

    manifest_path = Path(path).expanduser().resolve()
    return await redteam_manifest(
        load_manifest_file(manifest_path),
        manifest_path=manifest_path,
        options=_run_options(
            options,
            name=name,
            threshold=threshold,
            dry_run=dry_run,
        ),
    )


run_redteam_manifest_file = redteam_manifest_file


async def redteam_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    options: Optional[ManifestRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute a red-team manifest without writing CLI artifacts."""

    cli = _cli()
    opts = _run_options(
        options,
        name=name,
        threshold=threshold,
        dry_run=dry_run,
    )
    runtime_manifest = copy.deepcopy(dict(manifest))
    manifest_path = Path(manifest_path).expanduser().resolve()
    if opts.name:
        runtime_manifest["name"] = opts.name
    if opts.threshold is not None:
        runtime_manifest.setdefault("evaluation", {}).setdefault(
            "agent_report", {}
        )["threshold"] = opts.threshold

    started = time.time()
    redteam_summary = cli._prepare_redteam_manifest(runtime_manifest)
    validate_manifest_env(runtime_manifest)
    apply_manifest_env(runtime_manifest)
    if opts.dry_run:
        result = {
            "schema_version": CLI_SCHEMA_VERSION,
            "name": str(runtime_manifest.get("name") or manifest_path.stem),
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": {
                "required_env": required_manifest_env(runtime_manifest),
                "scenario_cases": len(cli._scenario_dataset(runtime_manifest)),
                "environment_count": len(cli._environment_specs(runtime_manifest)),
                "redteam": redteam_summary,
            },
            "redteam": redteam_summary,
            "duration_seconds": round(time.time() - started, 4),
        }
        return result

    report = await run_local_text_manifest(runtime_manifest, manifest_path)
    evaluation = evaluate_manifest_report(runtime_manifest, report)
    result = cli._run_result(
        manifest=runtime_manifest,
        report=report,
        evaluation=evaluation,
        duration_seconds=round(time.time() - started, 4),
    )
    redteam_result = cli._redteam_result_summary(
        runtime_manifest,
        result.get("evaluation"),
    )
    result["redteam"] = redteam_result
    result["summary"]["redteam"] = redteam_result
    return result


run_redteam_manifest = redteam_manifest


def optimize_manifest_file(
    path: str | Path,
    *,
    options: Optional[ManifestOptimizationOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load and optimize a manifest, returning the CLI-compatible payload."""

    manifest_path = Path(path).expanduser().resolve()
    return optimize_manifest(
        load_manifest_file(manifest_path),
        manifest_path=manifest_path,
        options=_optimization_options(
            options,
            name=name,
            threshold=threshold,
            max_candidates=max_candidates,
            dry_run=dry_run,
        ),
    )


def optimize_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    options: Optional[ManifestOptimizationOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Optimize a manifest without writing CLI artifacts."""

    cli = _cli()
    opts = _optimization_options(
        options,
        name=name,
        threshold=threshold,
        max_candidates=max_candidates,
        dry_run=dry_run,
    )
    runtime_manifest = copy.deepcopy(dict(manifest))
    manifest_path = Path(manifest_path).expanduser().resolve()
    if opts.name:
        runtime_manifest["name"] = opts.name
    if opts.threshold is not None:
        runtime_manifest.setdefault("optimization", {})["threshold"] = opts.threshold
    if opts.max_candidates is not None:
        runtime_manifest.setdefault("optimization", {}).setdefault(
            "optimizer", {}
        )["max_candidates"] = opts.max_candidates

    started = time.time()
    validate_manifest_env(runtime_manifest)
    apply_manifest_env(runtime_manifest)
    optimization = cli._optimization_config(runtime_manifest)
    if opts.dry_run:
        return {
            "schema_version": CLI_SCHEMA_VERSION,
            "name": str(runtime_manifest.get("name") or manifest_path.stem),
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": {
                "required_env": required_manifest_env(runtime_manifest),
                "search_path_count": len(
                    cli._target_config(optimization).get("search_space", {})
                ),
                "max_candidates": cli._optimizer_config(optimization).get(
                    "max_candidates"
                ),
            },
            "duration_seconds": round(time.time() - started, 4),
        }

    problem = build_manifest_optimization_problem(
        runtime_manifest,
        manifest_path=manifest_path,
        name=str(runtime_manifest.get("name") or manifest_path.stem),
    )
    result = problem.optimize()
    return cli._optimization_result(
        manifest=runtime_manifest,
        optimization_result=result,
        threshold=float(optimization.get("threshold", 0.7)),
        duration_seconds=round(time.time() - started, 4),
    )


def build_manifest_optimization_problem(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    name: Optional[str] = None,
) -> Any:
    """Build an agent-opt ManifestOptimizationProblem for this manifest."""

    cli = _cli()
    manifest_path = Path(manifest_path).expanduser().resolve()
    optimization = cli._optimization_config(manifest)
    manifest_base = copy.deepcopy(dict(manifest))
    manifest_base.pop("optimization", None)

    try:
        from fi.opt import ManifestOptimizationProblem
    except Exception as exc:  # pragma: no cover - optional dependency clarity
        raise ManifestError("agent-opt is required for manifest optimization.") from exc

    def evaluate_manifest(candidate_manifest: Mapping[str, Any], candidate: Any) -> Any:
        return run_local_text_manifest(candidate_manifest, manifest_path)

    def score_manifest(
        candidate_manifest: Mapping[str, Any],
        report: Any,
        candidate: Any,
    ) -> Dict[str, Any]:
        evaluation = evaluate_manifest_report(candidate_manifest, report)
        score = float(getattr(evaluation, "score", 1.0 if evaluation is None else 0.0))
        return {
            "score": score,
            "metadata": {
                "agent_report_evaluation": (
                    cli._to_plain(evaluation) if evaluation is not None else None
                ),
                "report_summary": cli._report_summary(report),
            },
        }

    return ManifestOptimizationProblem.from_manifest(
        {**manifest_base, "optimization": optimization},
        evaluate_manifest=evaluate_manifest,
        score_manifest=score_manifest,
        name=name or str(manifest.get("name") or manifest_path.stem),
    )


def _cli() -> Any:
    return importlib.import_module("fi.simulate.cli")


def _run_options(
    options: Optional[ManifestRunOptions],
    *,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    no_eval: Optional[bool] = None,
    dry_run: Optional[bool] = None,
) -> ManifestRunOptions:
    opts = options or ManifestRunOptions()
    return ManifestRunOptions(
        name=opts.name if name is None else name,
        threshold=opts.threshold if threshold is None else threshold,
        no_eval=opts.no_eval if no_eval is None else no_eval,
        dry_run=opts.dry_run if dry_run is None else dry_run,
    )


def _optimization_options(
    options: Optional[ManifestOptimizationOptions],
    *,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> ManifestOptimizationOptions:
    opts = options or ManifestOptimizationOptions()
    return ManifestOptimizationOptions(
        name=opts.name if name is None else name,
        threshold=opts.threshold if threshold is None else threshold,
        max_candidates=opts.max_candidates if max_candidates is None else max_candidates,
        dry_run=opts.dry_run if dry_run is None else dry_run,
    )


__all__ = [
    "CLI_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "ManifestOptimizationOptions",
    "ManifestRunOptions",
    "apply_manifest_env",
    "build_manifest_optimization_problem",
    "detect_manifest_command",
    "evaluate_manifest_report",
    "load_manifest",
    "load_manifest_file",
    "missing_manifest_env",
    "optimize_manifest",
    "optimize_manifest_file",
    "redteam_manifest",
    "redteam_manifest_file",
    "required_manifest_env",
    "run_local_text_manifest",
    "run_manifest",
    "run_manifest_file",
    "run_redteam_manifest",
    "run_redteam_manifest_file",
    "validate_manifest_env",
]
