![Future AGI](banner.png)

<div align="center">

# Simulate — test AI agents before users meet them

**Python SDK for the `Simulate` pillar of [Future AGI](https://github.com/future-agi/future-agi).**
Run voice and text agents against persona-driven scenarios, capture transcripts and audio, and feed results straight into evals.

<p>
  <a href="https://pypi.org/project/agent-simulate/"><img src="https://img.shields.io/pypi/v/agent-simulate?style=flat-square&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/agent-simulate/"><img src="https://img.shields.io/pypi/pyversions/agent-simulate?style=flat-square" alt="Python versions"></a>
  <a href="https://github.com/future-agi/simulate-sdk/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="Apache 2.0 License"></a>
  <a href="https://pypi.org/project/agent-simulate/"><img src="https://img.shields.io/pypi/dm/agent-simulate?style=flat-square&label=downloads" alt="PyPI downloads"></a>
  <a href="https://discord.gg/UjZ2gRT5p"><img src="https://img.shields.io/badge/discord-join-5865F2?style=flat-square" alt="Discord"></a>
</p>

<p>
  <a href="https://pypi.org/project/agent-simulate/"><b>PyPI</b></a> ·
  <a href="https://docs.futureagi.com/docs/simulation"><b>Docs</b></a> ·
  <a href="https://app.futureagi.com"><b>Platform</b></a> ·
  <a href="https://github.com/future-agi/future-agi"><b>Main repo</b></a> ·
  <a href="https://discord.gg/UjZ2gRT5p"><b>Discord</b></a> ·
  <a href="https://github.com/future-agi/simulate-sdk/issues"><b>Issues</b></a>
</p>

</div>

---

## Why this SDK?

AI agents hallucinate. They fabricate facts and misquote policies, and a bad output in production is already in the world by the time anyone notices. You can't unit-test "don't make things up" — you run the agent against realistic conversations before real users do.

`agent-simulate` is the client SDK for the `Simulate` pillar of Future AGI. It drives voice and text agents through persona-driven scenarios and hands the transcripts + audio to the `Evaluate` pillar for scoring.

---

<div align="center">
  <img src="simulate-repo.gif" alt="agent-simulate Demo" width="70%" />
</div>

## What's in the box

<table>
<tr>
<td width="33%" valign="top">

### Voice agents (LiveKit)

Connect a simulated customer to an agent sitting in a **LiveKit** room over WebRTC. Full multi-turn conversation, **per-speaker and combined WAV recordings**, and a complete transcript.

</td>
<td width="33%" valign="top">

### Text agents (Cloud)

Orchestrate thousands of multi-turn text conversations against any agent framework (**OpenAI**, **Anthropic**, **LangChain**, **Gemini**, or your own), via Future AGI's hosted simulation backend.

</td>
<td width="33%" valign="top">

### Evaluation-ready

Results drop straight into [`ai-evaluation`](https://github.com/future-agi/ai-evaluation) via the `evaluate_report` helper. Score with **50+ built-in metrics** or your own rubrics — task completion, tone, audio quality, groundedness, and others.

</td>
</tr>
</table>

---

## Install

```bash
# Core SDK
pip install agent-simulate

# With voice (LiveKit) support
pip install "agent-simulate[livekit]"

# With evaluation helpers
pip install "agent-simulate[evaluation]"

# Everything
pip install "agent-simulate[all]"
```

Requires Python **3.10–3.13**.

<details>
<summary><b>Voice mode: download Silero VAD weights (one time)</b></summary>

The LiveKit engine uses Silero VAD for voice-activity detection. Run this once after installing the `[livekit]` extra:

```python
from livekit.plugins import silero

if __name__ == "__main__":
    silero.VAD.load()
```

</details>

---

## 🚀 Quickstart — Voice agent (LiveKit)

Connects a simulated customer (`Alice`) to a deployed voice agent waiting in a LiveKit room, records the call, and scores the transcript.

```python
import asyncio
import os
from dotenv import load_dotenv
from fi.simulate import AgentDefinition, Scenario, Persona, TestRunner
from fi.simulate.evaluation import evaluate_report

load_dotenv()

async def main():
    # 1. Point at your deployed voice agent
    agent = AgentDefinition(
        name="my-support-agent",
        url=os.environ["LIVEKIT_URL"],
        room_name="support-room",
        system_prompt="Helpful support agent",
    )

    # 2. Describe the test case
    scenario = Scenario(
        name="Password Reset",
        dataset=[
            Persona(
                persona={"name": "Alice", "mood": "frustrated"},
                situation="She cannot log into her account.",
                outcome="The agent should guide her through a password reset.",
            ),
        ],
    )

    # 3. Run the simulation
    runner = TestRunner()
    report = await runner.run_test(
        agent_definition=agent,
        scenario=scenario,
        record_audio=True,  # writes per-speaker + combined WAVs
    )

    # 4. Inspect results
    for r in report.results:
        print(r.transcript)
        print(r.audio_combined_path)

    # 5. Score with Future AGI evals
    evaluated = evaluate_report(
        report,
        eval_specs=[
            {"template": "task_completion",
             "map": {"input": "persona.situation", "output": "transcript"}},
            {"template": "audio_quality",
             "map": {"input_audio": "audio_combined_path"}},
        ],
    )

    for r in evaluated.results:
        for name, scores in (r.evaluation or {}).items():
            print(name, scores["score"], scores["reason"])

asyncio.run(main())
```

**Required environment variables** (voice mode):

```bash
LIVEKIT_URL="wss://your-livekit-server.com"
LIVEKIT_API_KEY="..."
LIVEKIT_API_SECRET="..."
OPENAI_API_KEY="..."           # for the simulated customer
FI_API_KEY="..."               # for evaluation
FI_SECRET_KEY="..."
```

---

## 🚀 Quickstart — Text agent (Cloud)

Cloud mode runs the scenario orchestration on Future AGI's backend and calls **your** agent over a local callback. Use it when you want thousands of parallel text conversations against an OpenAI, Anthropic, LangChain, or Gemini agent without running LiveKit.

1. Create a simulation run from the [Future AGI platform](https://app.futureagi.com) and copy its `run_id` (or name).
2. Wire your agent to the runner:

```python
import asyncio
import os
from openai import AsyncOpenAI
from fi.simulate import TestRunner, OpenAIAgentWrapper

async def main():
    # Your agent, wrapped in a zero-config adapter
    wrapper = OpenAIAgentWrapper(
        client=AsyncOpenAI(),
        model="gpt-4o-mini",
        system_prompt="You are a helpful support agent.",
    )

    runner = TestRunner(
        api_key=os.environ["FI_API_KEY"],
        secret_key=os.environ["FI_SECRET_KEY"],
    )

    report = await runner.run_test(
        run_test_name="support-agent-smoke-test",  # or run_id="..."
        agent_callback=wrapper,
        concurrency=5,
    )

asyncio.run(main())
```

Scores and transcripts land in the platform dashboard. The local `TestReport` is intentionally empty — metrics live in the backend so you can compare runs over time.

<sub>See [`examples/test_cloud_simulation.py`](examples/test_cloud_simulation.py) for a full tool-using walkthrough. That example shows a custom `AgentWrapper` subclass around the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) — useful when you need tool-call capture beyond the built-in `OpenAIAgentWrapper`.</sub>

---

## Agent wrappers

Built-in adapters for common SDKs plus a framework-neutral adapter layer. The generic adapter carries normalized messages, tools, memory, events, and multimodal artifacts, so the same report shape works for text, voice, image, browser/CUA, and framework-specific runtimes.

| Wrapper | Wraps | Import |
|---|---|---|
| `OpenAIAgentWrapper` | `openai.OpenAI` / `AsyncOpenAI` (chat.completions) | `from fi.simulate import OpenAIAgentWrapper` |
| `AnthropicAgentWrapper` | `anthropic.Anthropic` / `AsyncAnthropic` | `from fi.simulate import AnthropicAgentWrapper` |
| `GeminiAgentWrapper` | `google.generativeai.GenerativeModel` | `from fi.simulate import GeminiAgentWrapper` |
| `LangChainAgentWrapper` | Any LangChain `Runnable` / chain | `from fi.simulate import LangChainAgentWrapper` |
| `GenericAgentWrapper` / `wrap_agent` | Any callable/object with `call`, `ainvoke`, `invoke`, `run`, `send`, `respond`, or `chat` | `from fi.simulate import wrap_agent` |
| `wrap_framework` | Import-free presets for LangChain, LangGraph, CrewAI, AutoGen, LlamaIndex, OpenAI Agents, LiveKit, Pipecat, browser/CUA, vision agents, and more | `from fi.simulate import wrap_framework` |
| Mock wrappers | Scripted, echo, and rule-based agents for deterministic regression tests | `from fi.simulate import ScriptedAgentWrapper` |
| Custom | Anything — subclass `AgentWrapper` | `from fi.simulate import AgentWrapper` |

Rolling your own wrapper is a 20-line class — see [CONTRIBUTING.md → Adding a new agent wrapper](CONTRIBUTING.md#-adding-a-new-agent-wrapper).

---

## Local self-contained simulation

Use `TestRunner` with `agent_callback` and a `scenario` or `topic` to run locally without LiveKit, Future AGI cloud, or model keys. Reports include transcript, normalized messages, tool calls, artifacts, events, and metadata.

```python
from fi.simulate import SyntheticDataGenerator, TestRunner, wrap_framework

scenario = SyntheticDataGenerator().generate("browser checkout support", seed=11)
agent = wrap_framework("computer_use", browser_agent)

report = await TestRunner().run_test(
    scenario=scenario,
    agent_callback=agent,
    modality="cua",
    max_turns=2,
)

print(report.results[0].messages)
print(report.results[0].artifacts)
print(report.results[0].events)
```

See [`examples/local_multimodal_simulation.py`](examples/local_multimodal_simulation.py) for a full offline browser/CUA-style cookbook.

### Local environments

Use environment adapters when the agent needs a world to act on: mocked APIs,
transient tool/API faults, browser/CUA state, voice turns, image fixtures,
files, framework traces, or multi-agent handoffs.
Browser adapters can emit DOM snapshots, screenshots, coordinate-region
assertions, screenshot/action diff evidence, prompt-injection surfaces, action
replay, console/network logs, and trace artifacts. Voice adapters can replay VAD/STT/TTS
events, Pipecat-style frames, latency profiles, barge-in/overlap handling, call
routing, noise metadata, audio artifacts, timelines, and voice trace artifacts.
Framework trace adapters can replay native orchestration
spans from LangChain/LangGraph, OpenAI Agents, CrewAI, AutoGen, LiveKit,
Pipecat, or custom runtimes. Adversarial packs add hostile retrieved context,
file content, browser DOM, and memory-like context for indirect prompt-injection
tests. The local engine exposes environment tools through `AgentInput.tools`,
auto-executes matching tool calls, and records tool results, per-call
`state_updates`, final environment state, artifacts, and events in the report.
Multi-agent rooms can also carry expected handoffs, review requirements, and
reconciliation requirements so evaluators can score contract correctness instead
of only checking that a handoff trace exists.

```python
from fi.simulate import (
    AdversarialEnvironmentPack,
    AutonomyLoopEnvironment,
    BrowserEnvironment,
    FileEnvironment,
    FrameworkTraceEnvironment,
    ImageEnvironment,
    MultiAgentRoomEnvironment,
    RetrievalMemoryEnvironment,
    ToolFaultInjectionEnvironment,
    ToolMockEnvironment,
    TestRunner,
    VoiceEnvironment,
    normalize_framework_trace_events,
)

report = await TestRunner().run_test(
    scenario=scenario,
    agent_callback=agent,
    environment=[
        ToolFaultInjectionEnvironment({
            "search_order": {"count": 1, "error": "timeout"},
        }),
        ToolMockEnvironment({
            "search_order": lambda args, ctx: {
                "content": "Order is ready",
                "state_updates": {"case": {"resolved": True}},
            }
        }),
        AutonomyLoopEnvironment(
            goal="Resolve the case with observe/orient/plan/act/verify/reflect evidence.",
            expected_plan={"required_steps": ["lookup", "policy", "respond"]},
            expected_verification={"required_checks": ["order found"], "passed_required": True},
            expected_reflection={"required_terms": ["verify", "policy"]},
            expected_memory={"required_keys": ["order_id", "status"]},
            expected_skills=[{"name": "refund_policy_check", "required_steps": ["lookup", "verify"]}],
            expected_stop={"should_stop": True},
        ),
        FrameworkTraceEnvironment(
            framework="openai_agents",
            spans=[
                {"id": "agent_1", "name": "agent_span", "type": "agent"},
                {"id": "tool_1", "name": "function_span search_order", "type": "tool"},
            ],
        ),
        RetrievalMemoryEnvironment({
            "refund_policy_current": {
                "content": "Order 123 can be refunded under the current policy.",
                "version": "v2",
                "current": True,
            }
        }, memory={"order_id": "123"}),
        BrowserEnvironment(
            url="https://shop.example.com/checkout",
            dom="<button id='review'>Review</button>",
            screenshot_uri="file:///fixtures/checkout.png",
            allowed_domains=["shop.example.com"],
            console_logs=[{"level": "info", "message": "checkout loaded"}],
            network_log=[{"url": "https://shop.example.com/api/order", "status": 200}],
        ),
        VoiceEnvironment(
            [{"id": "caller_1", "transcript": "Please check order 123."}],
            routes={"billing": {"kind": "queue", "name": "billing_specialist"}},
            noise_profile={"noise_db": 62, "processed_noise_db": 24},
            frame_replay=[
                {"type": "InputAudioRawFrame", "timestamp_ms": 0},
                {"type": "TranscriptionFrame", "text": "Please check order 123."},
                {"type": "TTSStartedFrame", "timestamp_ms": 650},
                {"type": "TTSAudioRawFrame", "duration_ms": 900},
                {"type": "OverlappingSpeechFrame", "overlap_ms": 180},
            ],
        ),
        ImageEnvironment({
            "receipt": {"uri": "file:///fixtures/receipt.png", "description": "Order receipt"}
        }),
        FileEnvironment({"refund-policy.md": "Refunds require approval."}),
        AdversarialEnvironmentPack(),
        MultiAgentRoomEnvironment(
            ["support_agent", "policy_specialist"],
            handoff_contracts={
                "policy_specialist": {"required_output": "policy decision"}
            },
            expected_handoffs=[
                {"to": "policy_specialist", "context_keys": ["order_id"]}
            ],
            expected_reconciliation={"accepted_source": "policy_specialist"},
        ),
    ],
    max_turns=1,
)
```

Put `ToolFaultInjectionEnvironment` before the real or mocked tool adapter to
fail the first N matching calls, emit structured `tool_fault` and
`tool_execution` events, and then allow later retries to fall through to the
next environment adapter.

For production trace capture, use TraceAI/OpenTelemetry and store/export those
spans through Future AGI or another OTel-compatible backend. The local
`FrameworkTraceEnvironment` can ingest raw TraceAI/OpenTelemetry-style span
records directly, or you can normalize them first:

```python
trace_records = normalize_framework_trace_events(
    "traceai",
    [
        {
            "name": "langgraph node support_agent",
            "attributes": {
                "gen_ai.span.kind": "CHAIN",
                "langgraph.state.updates": {"step": "planned"},
            },
        },
        {
            "name": "search_order",
            "attributes": {
                "gen_ai.span.kind": "TOOL",
                "gen_ai.tool.name": "search_order",
            },
        },
    ],
)

environment = FrameworkTraceEnvironment(framework="traceai", events=trace_records)
```

See [`examples/local_environment_adapters.py`](examples/local_environment_adapters.py) for a full local world simulation with ai-evaluation scoring, and [`examples/local_voice_image_environments.py`](examples/local_voice_image_environments.py) for a voice + image artifact cookbook.
See [`examples/local_browser_trace_replay.py`](examples/local_browser_trace_replay.py) for a browser/CUA trace replay cookbook with screenshots, DOM, selector and coordinate-region action fixtures, screenshot/action diff evidence, prompt-injection surfaces, mutable browser state, console/network logs, DOM mutation events, and action replay.
See [`examples/local_voice_replay_routing.py`](examples/local_voice_replay_routing.py) for a voice replay cookbook with VAD/STT/TTS events, Pipecat-style frame replay, noise/overlap evidence, interruption handling, and call routing.
See [`examples/local_framework_trace_replay.py`](examples/local_framework_trace_replay.py) for a framework trace replay cookbook with native spans/events from orchestration frameworks.
See [`examples/local_retrieval_memory_attribution.py`](examples/local_retrieval_memory_attribution.py) for a retrieval/memory attribution cookbook with queries, ranked documents, retrieval scores, citations, memory reads/writes, and freshness evidence.
See [`examples/local_autonomy_loop_trace.py`](examples/local_autonomy_loop_trace.py) for an autonomy-loop cookbook with observe/orient/plan/act/verify/reflect, feedback, memory, skill-library evidence, and plan/verifier/reflection/memory/skill/stop quality checks.
See [`examples/local_multi_agent_handoff_trace.py`](examples/local_multi_agent_handoff_trace.py) for a multi-agent handoff cookbook with roles, contracts, delegated work, review, reconciliation, trace coverage, and coordination-quality scoring.
See [`examples/local_adversarial_environment_pack.py`](examples/local_adversarial_environment_pack.py) for an indirect prompt-injection pentest using hostile tool, file, browser, and memory surfaces.

### Local pentest scenarios

Use `generate_pentest` to create deterministic adversarial personas for common
agent failure modes: prompt injection, secret exfiltration, unsafe actions,
browser/CUA policy bypass, memory contamination, tool abuse, cross-user data
exfiltration, and voice turn-taking.

```python
from fi.simulate import SyntheticDataGenerator, TestRunner, evaluate_agent_report

scenario = SyntheticDataGenerator().generate_pentest(
    "checkout support",
    attack_vectors=["prompt_injection", "secret_exfiltration", "browser_cua"],
    seed=17,
)

report = await TestRunner().run_test(
    scenario=scenario,
    agent_callback=agent,
    max_turns=3,
    min_turns=3,
    modality="cua",
)

evaluation = evaluate_agent_report(
    report,
    config={"allowed_domains": ["shop.example.com"]},
)
```

See [`examples/local_pentest_scenarios.py`](examples/local_pentest_scenarios.py) for a full local adversarial simulation and scoring cookbook.

### Synthetic tool-world scenarios

Use `generate_tool_task` when you need a runnable synthetic API world, not just
synthetic personas. The returned bundle includes a `Scenario`, tool schemas, a
`ToolMockEnvironment`, expected final state, expected tool outcomes, and an
`ai-evaluation` config.

```python
from fi.simulate import SyntheticDataGenerator, TestRunner, evaluate_agent_report

bundle = SyntheticDataGenerator().generate_tool_task(
    "order fulfillment",
    target_status="shipped",
    seed=8,
)

report = await TestRunner().run_test(
    scenario=bundle.scenario,
    agent_callback=agent,
    environment=bundle.make_environment(),
    max_turns=1,
)

evaluation = evaluate_agent_report(report, config=bundle.agent_report_config)
```

See [`examples/local_synthetic_tool_task.py`](examples/local_synthetic_tool_task.py) for the complete local synthetic tool-world cookbook.

---

## Evaluation

The `evaluate_report` helper delegates to [`ai-evaluation`](https://github.com/future-agi/ai-evaluation), Future AGI's `Evaluate` SDK. It accepts either a named template list or field-mapped specs:

```python
from fi.simulate.evaluation import evaluate_report

# Named templates with sensible defaults
evaluate_report(report, eval_templates=("task_completion", "tone", "is_helpful"))

# Or explicit field mapping — including audio, trajectories, tools, and artifacts
evaluate_report(
    report,
    eval_specs=[
        {"template": "task_completion",
         "map": {"input": "persona.situation", "output": "transcript"}},
        {"template": "agent_trajectory",
         "map": {"input": "messages", "tools": "tool_calls", "events": "events"}},
        {"template": "audio_quality",
         "map": {"input_audio": "audio_combined_path"}},
    ],
)
```

For fully local agent scoring, use `evaluate_agent_report`. It scores the
normalized simulation trace directly: trajectory, tool use, prompt-injection
resistance, environment-injection resistance, memory integrity, browser/CUA
safety, browser action outcome/state success, browser trace coverage, voice
turn-taking, voice interaction quality, voice trace coverage, framework trace
coverage, tool argument schema validation, retrieval context quality, source grounding,
retrieval/memory attribution, autonomy-loop coverage, autonomy-loop quality,
multi-agent trace coverage, multi-agent coordination quality, artifact coverage,
and expected state.

```python
from fi.simulate import evaluate_agent_report

evaluation = evaluate_agent_report(
    report,
    config={
        "required_tools": ["search_order"],
        "available_tools": ["search_order"],
        "allowed_domains": ["shop.example.com"],
        "memory_allowed_keys": ["order_id", "status"],
        "required_artifact_types": ["image", "audio"],
        "required_browser_trace": ["dom", "screenshot", "action", "coordinate_region", "screenshot_diff", "prompt_injection_surface", "dom_mutation", "state", "console", "network"],
        "expected_browser_actions": [{"selector": "#confirm", "success": True}],
        "expected_browser_regions": [
            {"name": "confirm_button", "bounds": {"x": 160, "y": 380, "width": 180, "height": 54}}
        ],
        "expected_browser_screenshot_diffs": ["confirm_visual_delta"],
        "forbidden_browser_prompt_injection_targets": ["coupon_iframe"],
        "expected_browser_state": {"url": "https://shop.example.com/done"},
        "expected_browser_dom_contains": ["Done"],
        "required_voice_trace": ["audio", "vad", "stt", "tts", "interruption", "route", "frame", "noise", "overlap", "timeline"],
        "expected_voice_route": "billing",
        "expected_voice_transcript_contains": ["order 123"],
        "required_voice_frame_types": ["InputAudioRawFrame", "TranscriptionFrame", "TTSStartedFrame", "TTSAudioRawFrame"],
        "max_voice_overlap_ms": 250,
        "max_voice_noise_db": 35,
        "required_framework_trace": ["agent", "model", "tool", "handoff", "guardrail"],
        "required_retrieval_memory_trace": ["query", "document", "citation", "memory_read", "memory_write"],
        "expected_retrieval_doc_ids": ["refund_policy_current"],
        "forbidden_retrieval_doc_ids": ["refund_policy_old"],
        "require_current_retrieval": True,
        "require_source_grounding": True,
        "required_autonomy_loop": ["observe", "orient", "plan", "act", "verify", "reflect"],
        "expected_autonomy_plan": {"required_steps": ["lookup", "policy", "respond"], "min_steps": 3},
        "expected_autonomy_verification": {"required_checks": ["order found"], "passed_required": True},
        "expected_autonomy_reflection": {"required_terms": ["verify", "policy"]},
        "expected_autonomy_memory": {"required_keys": ["order_id", "status"]},
        "expected_autonomy_skills": [{"name": "refund_policy_check", "required_steps": ["lookup", "verify"]}],
        "expected_autonomy_stop": {"should_stop": True},
        "required_multi_agent_trace": ["role", "contract", "handoff", "review", "reconciliation"],
        "required_multi_agent_roles": ["support_agent", "policy_specialist", "qa_reviewer"],
        "expected_multi_agent_handoffs": [{"to": "policy_specialist", "context_keys": ["order_id"]}],
        "expected_multi_agent_reviews": [{"reviewer": "qa_reviewer", "criteria": ["policy"]}],
        "expected_multi_agent_reconciliation": {"accepted_source": "policy_specialist"},
        "expected_state": {"case": {"resolved": True}},
    },
)

print(evaluation.score)
print(report.results[0].evaluation["agent_report"]["case_score"])
```

See [`examples/local_agent_report_evaluation.py`](examples/local_agent_report_evaluation.py) for a full local simulate -> evaluate cookbook.

50+ metrics are available out of the box — groundedness, faithfulness, tool-use correctness, RAG context relevance, hallucination, PII, toxicity, bias, audio quality, and custom rubrics. See the [evaluation docs](https://docs.futureagi.com/docs/evaluation) for the full catalog.

---

## How this fits into Future AGI

`agent-simulate` is one of six pillars in the [Future AGI](https://github.com/future-agi/future-agi) platform:

> **Simulate → Evaluate → Control → Monitor → Optimize** · with **Agent Command Center** as the runtime gateway.

Traces from simulations flow into `Monitor`, scores flow into `Evaluate`, and failures feed `Optimize` — one loop, on your infrastructure.

| SDK | Pillar | Purpose |
|---|---|---|
| [**agent-simulate**](https://github.com/future-agi/simulate-sdk) *(you are here)* | Simulate | Voice + text agent simulation |
| [**ai-evaluation**](https://github.com/future-agi/ai-evaluation) | Evaluate | 50+ metrics, LLM-as-judge, guardrail scanners |
| [**traceAI**](https://github.com/future-agi/traceAI) | Monitor | OpenTelemetry tracing for 50+ AI frameworks |
| [**agent-opt**](https://github.com/future-agi/agent-opt) | Optimize | 6 prompt-optimization algorithms |

<sub> [Full platform README →](https://github.com/future-agi/future-agi)</sub>

---

## Roadmap

<table>
<tr>
<th width="33%">Shipped</th>
<th width="33%">In progress</th>
<th width="33%">Coming up</th>
</tr>
<tr valign="top">
<td>

- [x] LiveKit voice simulation engine
- [x] Cloud simulation engine
- [x] Self-contained local text/CUA simulation engine
- [x] OpenAI / Anthropic / Gemini / LangChain wrappers
- [x] Generic framework adapter presets
- [x] Multimodal artifacts + event trajectories
- [x] Local environment adapters for mocked tools/APIs, framework trace replay, retrieval/memory attribution, autonomy-loop traces, multi-agent handoff traces, browser/CUA trace replay with coordinate regions and screenshot diffs, voice frame replay/routing/noise/overlap, images, files, adversarial packs, and multi-agent rooms
- [x] Deterministic synthetic data generator
- [x] Self-contained synthetic tool-world generator with schemas, mocks, state expectations, and evaluator config
- [x] Deterministic pentest scenario generator
- [x] Per-speaker + combined audio capture
- [x] Scenario auto-generation from a topic
- [x] `evaluate_report` integration with `ai-evaluation`
- [x] Local `evaluate_agent_report` scoring for trajectory, tools, memory, framework trace coverage, retrieval/memory attribution, autonomy-loop coverage, autonomy-loop quality, multi-agent trace coverage, multi-agent coordination quality, browser/CUA action outcome and grounding quality, browser trace coverage, voice trace coverage, voice interaction quality, environment injection, and pentest signals
- [x] Tool-call capture in wrapper responses

</td>
<td>

- [ ] Conversation-graph scenarios (branching flows)
- [ ] Streaming audio waveform, diarization, and perceptual audio metrics
- [ ] Streaming transcript API
- [ ] First-class Pipecat/VAPI/Retell voice backends over the generic voice artifact/event contract

</td>
<td>

- [ ] Larger adversarial persona library (jailbreak, PII probing, CUA prompt injection)
- [ ] Multi-agent scenarios
- [ ] On-device VAD + STT for air-gapped runs
- [ ] Regression dashboards in-SDK

</td>
</tr>
</table>

---

## 🤝 Contributing

We love contributions — bug fixes, new wrappers, framework integrations, docs, examples.

1. Browse [`good first issue`](https://github.com/future-agi/simulate-sdk/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
2. Read the [Contributing Guide](CONTRIBUTING.md)
3. Say hi on [Discord](https://discord.gg/UjZ2gRT5p)
4. Sign the CLA on your first PR (automatic bot)

---

## 🌍 Community & support

| | |
|---|---|
| 💬 [**Discord**](https://discord.gg/UjZ2gRT5p) | Real-time help from the team and community |
| 🗨️ [**GitHub Discussions**](https://github.com/orgs/future-agi/discussions) | Ideas, questions, roadmap input |
| 🐦 [**Twitter / X**](https://twitter.com/futureagi) | Release announcements |
| 📝 [**Blog**](https://futureagi.com/blog) | Engineering & research posts |
| 📧 **support@futureagi.com** | Cloud account / billing |
| 🔐 **security@futureagi.com** | Private vulnerability disclosure (see [SECURITY.md](SECURITY.md)) |

---

## 📄 License

`agent-simulate` is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

<div align="center">

**Built by the [Future AGI](https://futureagi.com) team and [contributors worldwide](https://github.com/future-agi/simulate-sdk/graphs/contributors).**

If this SDK helps you ship better agents, a ⭐ helps more teams find us.

[🌐 futureagi.com](https://futureagi.com) · [📖 docs.futureagi.com](https://docs.futureagi.com) · [☁️ app.futureagi.com](https://app.futureagi.com)

</div>
