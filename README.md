# Future AGI Simulate SDK

A Python SDK for testing conversational voice AI agents through realistic customer simulations.

## Overview

This SDK allows you to rigorously test your deployed voice AI agents by simulating realistic customer conversations. It connects a "simulated customer" to your agent, records the conversation, and provides detailed transcripts and evaluations.

## Key Features

- **Agent Definition**: Describe your deployed agent's configuration
- **Scenario Creation**: Define test cases with customer personas, situations, and desired outcomes
- **Automated Testing**: Run multiple test scenarios against your agent
- **Transcript Collection**: Automatically record full conversation transcripts
- **Evaluation Integration**: Integrate with Future AGI's evaluation service (ai-evaluation)

## Installation

```bash
pip install -r requirements.txt
```

### Download VAD Model Weights

The SDK uses Silero VAD for voice activity detection. Download the required model weights:

```python
from livekit.plugins import silero

if __name__ == "__main__":
    print("Downloading Silero VAD model...")
    silero.VAD.load()
    print("Download complete.")
```

Run this script once before using the SDK.

## Quick Start

### Prerequisites

1. **A deployed voice AI agent**: Your agent must be running and connected to a LiveKit room.
2. **LiveKit server access**: You need the URL, API key, and secret for the LiveKit server.
3. **OpenAI API key**: For the simulated customer agent.

### Environment Variables

Create a `.env` file:

```bash
# LiveKit Server Details
LIVEKIT_URL="wss://your-livekit-server.com"
LIVEKIT_API_KEY="your-api-key"
LIVEKIT_API_SECRET="your-api-secret"

# OpenAI API Key (for simulated customer)
OPENAI_API_KEY="your-openai-key"

# Future AGI Evaluation Keys (optional, for evaluations)
FI_API_KEY="your-fi-api-key"
FI_SECRET_KEY="your-fi-secret-key"
```

### Example Usage

```python
from fi.simulate import AgentDefinition, Scenario, Persona, TestRunner

# 1. Define your deployed agent
agent_definition = AgentDefinition(
    name="my-support-agent",
    url="wss://your-livekit-server.com",
    room_name="agent-room", # The room where your agent is waiting
    system_prompt="You are a helpful support agent.",
)

# 2. Create test scenarios
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
report = runner.run_test(agent_definition, scenario)

# 4. View results
for result in report.results:
    print(f"Transcript: {result.transcript}")
```

## Project Structure

```
fi/simulate/
├── agent/
│   └── definition.py    # AgentDefinition and configuration models
├── simulation/
│   ├── models.py        # Persona, Scenario, TestReport models
│   ├── runner.py        # TestRunner (the main testing engine)
│   └── generator.py     # ScenarioGenerator (AI-powered scenario creation)
└── __init__.py
```

## Coming Soon

- **Evaluation Integration**: Automatically evaluate agent performance using Future AGI's evaluation service
- **Advanced Scenarios**: Support for conversation graphs and complex flows
- **Metrics Collection**: Latency, interruption rates, and other conversation metrics

## License

Open source (license TBD)
