from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..simulation.models import TestReport


def evaluate_report(
    report: TestReport,
    *,
    eval_templates: Iterable[str] | None = ("task_completion", "tone", "is_helpful"),
    eval_specs: Sequence[dict] | None = None,
    model_name: str = "turing_flash",
    api_key: str | None = None,
    secret_key: str | None = None,
    extra_inputs: Mapping[str, str] | None = None,
) -> TestReport:
    """
    Evaluate each test case transcript using Future AGI ai-evaluation SDK.

    - Templates like "task_completion" will receive input and output fields
      mapped from persona and transcript.
    - "tone" will receive the whole transcript as input.

    Docs:
    - GitHub: https://github.com/future-agi/ai-evaluation
    - Getting started: https://docs.futureagi.com/future-agi/get-started/evaluation/running-your-first-eval#evaluate-using-sdk
    """

    try:
        from fi.evals import Evaluator
    except Exception as e:  # pragma: no cover - import error clarity
        raise RuntimeError(
            "ai-evaluation package is required. Install with `pip install ai-evaluation`."
        ) from e

    evaluator = Evaluator(fi_api_key=api_key, fi_secret_key=secret_key)

    for result in report.results:
        persona = result.persona
        transcript = result.transcript

        scores: dict[str, dict] = {}

        def resolve_source(key: str) -> str | None:
            if key == "transcript":
                return transcript
            if key == "persona.situation":
                return persona.situation
            if key == "persona.outcome":
                return persona.outcome
            if key == "audio_input_path":
                return getattr(result, "audio_input_path", None)
            if key == "audio_output_path":
                return getattr(result, "audio_output_path", None)
            return None

        # If eval_specs provided, use explicit mappings per template
        if eval_specs:
            for spec in eval_specs:
                template = spec.get("template")
                mapping: Mapping[str, str] = spec.get("map", {})  # desired_input_key -> source_key
                if not template:
                    continue
                inputs: dict[str, str] = {}
                for dest, source in mapping.items():
                    val = resolve_source(source)
                    if val is not None:
                        inputs[dest] = val
                if extra_inputs:
                    inputs.update(extra_inputs)
                try:
                    ev = evaluator.evaluate(eval_templates=template, inputs=inputs, model_name=model_name)
                    item = ev.eval_results[0] if ev and getattr(ev, "eval_results", None) else None
                    scores[template] = {
                        "output": getattr(item, "output", None),
                        "reason": getattr(item, "reason", None),
                        "score": getattr(item, "score", None),
                    }
                except Exception as e:
                    scores[template] = {"error": str(e), "inputs": inputs}
        else:
            # Fallback: simple built-ins by template name
            for template in (eval_templates or []):
                inputs: dict[str, str] = {}
                if template == "tone":
                    inputs = {"input": transcript}
                elif template == "task_completion":
                    inputs = {"input": persona.situation, "output": transcript}
                elif template == "is_helpful":
                    inputs = {"input": transcript}
                else:
                    inputs = {"input": transcript}

                if extra_inputs:
                    inputs.update(extra_inputs)

                try:
                    ev = evaluator.evaluate(eval_templates=template, inputs=inputs, model_name=model_name)
                    item = ev.eval_results[0] if ev and getattr(ev, "eval_results", None) else None
                    scores[template] = {
                        "output": getattr(item, "output", None),
                        "reason": getattr(item, "reason", None),
                        "score": getattr(item, "score", None),
                    }
                except Exception as e:
                    scores[template] = {"error": str(e)}

        result.evaluation = scores

    return report


