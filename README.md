# simulate-sdk

`simulate-sdk` has been consolidated into `agent-learning-kit`.

The simulation runtime, examples, tests, and new development now live in:

- `../agent-learning-kit/src/fi/simulate`
- `../agent-learning-kit/src/agent_learning/simulate.py`
- `../agent-learning-kit/examples`

Use the unified SDK:

```bash
pip install agent-learning-kit
agent-learn run examples/run_manifest.json
agent-learn redteam examples/redteam_manifest.json
agent-learn suite examples/agent_learning_suite.json
```

Python callers should import:

```python
from agent_learning import simulate
```

This repository is retained only for source history and migration context. Do
not publish it as a second simulation SDK.
