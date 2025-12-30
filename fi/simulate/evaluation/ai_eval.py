from __future__ import annotations

from typing import Iterable, Mapping, Sequence
import os
import base64

from fi.simulate.simulation.models import TestReport


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
                val = getattr(result, "audio_input_path", None)
                return os.path.abspath(val) if val and os.path.exists(val) else val
            if key == "audio_output_path":
                val = getattr(result, "audio_output_path", None)
                return os.path.abspath(val) if val and os.path.exists(val) else val
            if key == "audio_combined_path":
                val = getattr(result, "audio_combined_path", None)
                return os.path.abspath(val) if val and os.path.exists(val) else val
            return None

        def _encode_audio_inputs(inputs: dict[str, str]) -> dict[str, str]:
            """Strict encoding: if a value is a local audio file path, replace that value with base64.

            - Never rename keys or add aliases.
            - Do not add extra fields (no audio_mime or data URI).
            """
            audio_exts = {".wav", ".ogg", ".mp3", ".m4a", ".flac", ".aac"}
            for k, v in list(inputs.items()):
                if isinstance(v, str) and os.path.exists(v):
                    _, ext = os.path.splitext(v.lower())
                    if ext in audio_exts:
                        try:
                            with open(v, "rb") as f:
                                data = f.read()
                            inputs[k] = base64.b64encode(data).decode("ascii")
                        except Exception:
                            # Leave as-is on read failure
                            pass
            return inputs

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
                inputs = _encode_audio_inputs(inputs)
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
                inputs = _encode_audio_inputs(inputs)

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


