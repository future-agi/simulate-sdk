<div align="center">

# 🧪 Agent Simulate SDK

**A Python SDK for testing conversational voice AI agents through realistic customer simulations.**  
Built by [Future AGI](https://futureagi.com) | [Docs](https://docs.futureagi.com) | [Platform](https://app.futureagi.com)

</div>

---

## 🚀 Overview

**Agent Simulate** is a testing framework for deployed voice AI agents. It automates realistic customer conversations, records the full interaction, and integrates with evaluation pipelines to help you ship reliable agents.

- 📞 **Test Deployed Agents**: Connect directly to your agent in a LiveKit room.
- 🎭 **Persona-driven Scenarios**: Define customer personas, situations, and goals.
- 🎙️ **Full Audio & Transcripts**: Capture complete conversation audio and text.
- 📊 **Integrate Evaluations**: Use `ai-evaluation` to score agent performance.
---

## Key Features
| Feature                       | Description                                                                              |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| **Agent Definition**          | Configure the connection to your deployed agent, including room, prompts, and credentials. |
| **Scenario Creation**         | Programmatically define test cases with unique customer personas, situations, and desired outcomes. |
| **Automated Test Runner**     | Orchestrates the simulation, connects the persona to the agent, and manages the conversation flow. |
| **Audio/Transcript Capture**  | Automatically records individual and combined audio tracks, plus a full text transcript. |
| **Evaluation Integration**    | Pass test results (audio, transcripts) to the `ai-evaluation` library for scoring. |
| **Extensible & Customizable** | Customize STT, TTS, and LLM providers for the simulated customer via `SimulatorAgentDefinition`. |
---

## 🔧 Installation



```bash
# Install through pip 
pip install agent-simulate
```

If you want to fork the project and install it in editable mode, you can use the following command:
```bash
git clone https://github.com/future-agi/simulate-sdk.git
cd simulate-sdk
poetry install        # primary build system
# or: pip install -e .
```

### Download VAD Model Weights

The SDK uses Silero VAD for voice activity detection. Download the required model weights by running this script once:

```python
from livekit.plugins import silero

if __name__ == "__main__":
    print("Downloading Silero VAD model...")
    silero.VAD.load()
    print("Download complete.")
```
---

## 🧑‍💻 Quickstart

### 1. 🔐 Set Environment Variables

Create a `.env` file with your credentials:

```bash
# LiveKit Server Details
LIVEKIT_URL="wss://your-livekit-server.com"
LIVEKIT_API_KEY="your-api-key"
LIVEKIT_API_SECRET="your-api-secret"

# OpenAI API Key (for the default simulated customer)
OPENAI_API_KEY="your-openai-key"

# Future AGI Evaluation Keys (for running evaluations)
FI_API_KEY="your-fi-api-key"
FI_SECRET_KEY="your-fi-secret-key"
```

### 2. ✅ Run a Simulation

This example connects a simulated customer ("Alice") to your deployed agent.

```python
import asyncio
import os
from dotenv import load_dotenv
from fi.simulate import AgentDefinition, Scenario, Persona, TestRunner
from fi.simulate.evaluation import evaluate_report

load_dotenv()

async def main():
    # 1. Define your deployed agent
    agent_definition = AgentDefinition(
        name="my-support-agent",
        url=os.environ["LIVEKIT_URL"],
        room_name="support-room",  # The room where your agent is waiting
        system_prompt="Helpful support agent", # The system prompt for the agent that defines its behavior
    )

    # 2. Create a test scenario
    scenario = Scenario(
        name="Customer Support Test",
        dataset=[
            Persona(
                persona={"name": "Alice", "mood": "frustrated"},
                situation="She cannot log into her account.",
                outcome="The agent should guide her through password reset.",
            ),
        ]
    )

    # 3. Run the test
    runner = TestRunner()
    report = await runner.run_test(
        agent_definition,
        scenario,
        record_audio=True,  # Capture WAV files
    )

    # 4. View results
    for result in report.results:
        print(f"Transcript: {result.transcript}")
        print(f"Combined Audio Path: {result.audio_combined_path}")

    # 5. Evaluate the Report
    # Each spec maps a template name to source keys resolved from the test case.
    # Keys in "map" are the eval template's input names; values are source fields
    # (e.g., "transcript", "persona.situation", "audio_combined_path").
    evaluated_report = evaluate_report(
        report,
        eval_specs=[
            {
                "template": "task_completion",
                "map": {"input": "persona.situation", "output": "transcript"},
            },
            {
                "template": "audio_quality",
                "map": {"input_audio": "audio_combined_path"},
            },
        ],
    )

    # View evaluation results
    # result.evaluation is a dict keyed by template name.
    # Each value is a dict with "score", "output", and "reason" keys.
    for result in evaluated_report.results:
        if result.evaluation:
            for template_name, scores in result.evaluation.items():
                print(f"Evaluation for {template_name}:")
                print(f"  Score: {scores.get('score')}")
                print(f"  Output: {scores.get('output')}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## LLM Evaluation with Future AGI Platform

This SDK's `evaluate_report` helper uses the [ai-evaluation](https://github.com/future-agi/ai-evaluation) library under the hood. For dataset curation, custom eval templates, production monitoring, and more, see the [Future AGI platform docs](https://docs.futureagi.com).

---

## 🗺️ Roadmap 

* [x] **Core Simulation Engine**
* [x] **Persona-driven Scenarios**
* [x] **Audio & Transcript Recording**
* [x] **`ai-evaluation` Integration Helper**
* [ ] **Advanced Scenarios (Conversation Graphs)**
* [ ] **Deeper Performance Metrics (Latency, Interruption Rates)**

---

## 🤝 Contributing

We welcome contributions! To report issues, suggest templates, or contribute improvements, please open a GitHub issue or PR.

---
