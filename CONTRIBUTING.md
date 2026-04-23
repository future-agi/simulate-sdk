# Contributing to simulate-sdk

Thanks for your interest in contributing! This project exists because of people like you.

`agent-simulate` is the Python SDK for testing and evaluating voice AI agents in the Future AGI ecosystem. We welcome contributions of all kinds: bug fixes, new agent wrappers, framework integrations, docs improvements, examples, and issue triage.

---

## Quick links

- 🐛 [Report a bug](https://github.com/future-agi/simulate-sdk/issues/new?template=bug_report.yml)
- ✨ [Request a feature](https://github.com/future-agi/simulate-sdk/issues/new?template=feature_request.yml)
- 🔖 [Good first issues](https://github.com/future-agi/simulate-sdk/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- 💬 [Join Discord](https://discord.gg/UjZ2gRT5p)
- 📖 [SDK docs](https://docs.futureagi.com/docs/simulation)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold it. Report unacceptable behavior to **conduct@futureagi.com**.

---

## Contributor License Agreement (CLA)

Before we can merge your first pull request, you'll need to sign our Contributor License Agreement. This is a one-click process that runs automatically on your first PR — you'll see a link to sign, we merge after.

The CLA grants Future AGI, Inc. the rights to use your contribution (including an Apache-style patent grant), while letting you retain copyright. It also lets us re-license portions of the project later if needed. The full text is at [CLA.md](https://github.com/future-agi/future-agi/blob/main/CLA.md).

---

## Development setup

### 1. Fork and clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/future-agi/simulate-sdk.git  # replace with your fork URL
cd simulate-sdk
```

### 2. Install dependencies

Using Poetry (recommended):

```bash
poetry install --extras all
```

The `all` extra pulls in both `livekit-agents` and `ai-evaluation`, which are needed to run the full test suite.

Alternatively, using pip in editable mode:

```bash
pip install -e ".[all]"
```

### 3. Run tests

```bash
poetry run pytest
```

### 4. Format and lint

```bash
poetry run black . && poetry run ruff check --fix .
```

Run both before pushing. The PR checks will fail if either reports issues.

---

## How to contribute

### 🐛 Reporting bugs

Before filing, search [existing issues](https://github.com/future-agi/simulate-sdk/issues) to see if it's already reported. A good bug report includes:

- `agent-simulate` version (`pip show agent-simulate`)
- Python version and OS
- Whether you are using LiveKit mode or Cloud mode
- Exact reproduction steps
- Expected vs. actual behavior
- Relevant stack trace or logs

### ✨ Proposing features

For anything larger than about two hours of work, **open an issue first** so we can discuss design before you write code. This saves everyone time.

Good feature requests:

- Describe the problem you're trying to solve
- Show the desired API with a short code snippet
- List alternatives you considered
- Call out anything you're unsure about

### 🔧 Your first PR

1. Pick a [`good first issue`](https://github.com/future-agi/simulate-sdk/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) (or open one)
2. Comment that you're working on it so we don't double up
3. Branch from `main`: `git checkout -b type/short-description`
4. Make your change — keep the diff small and focused
5. Add tests (every bug fix needs a regression test)
6. Make sure `poetry run pytest` and the format checks pass
7. Push and open a PR using the template
8. Sign the CLA when the bot asks

---

## 🧩 Adding a new agent wrapper

Agent wrappers let the simulation engine call any LLM or agent framework. All wrappers live in `fi/simulate/agent/wrappers/`.

### Steps

1. Create a new file in `fi/simulate/agent/wrappers/`, e.g. `mistral.py`
2. Subclass `AgentWrapper` from `fi/simulate/agent/wrapper.py`
3. Implement the `call` method with this exact signature:

```python
async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
    ...
```

Returning a plain `str` is fine for simple cases. Return an `AgentResponse` when you need to surface tool calls or metadata.

4. Export the new class from `fi/simulate/agent/wrappers/__init__.py`
5. Add the export to `fi/simulate/__init__.py` so it is part of the public API
6. Add tests in `tests/test_wrappers.py` following the existing mock patterns

Reference implementations to read before writing your own:

- `fi/simulate/agent/wrappers/openai.py` — async/sync client detection
- `fi/simulate/agent/wrappers/anthropic.py` — system prompt extraction
- `fi/simulate/agent/wrappers/gemini.py` — history format conversion
- `fi/simulate/agent/wrappers/langchain.py` — `ainvoke` / `invoke` fallback

---

## Code style

- **Formatting:** Black + Ruff (line length 88, the default for both tools)
- **Imports:** Ruff handles import ordering
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`
- **Branch names:** `type/short-description` (e.g. `fix/langchain-ainvoke-fallback`)

---

## Pull-request checklist

Before requesting review:

- [ ] PR description explains **what** and **why** (not just **how** — the diff shows that)
- [ ] Tests added or updated (regression test for every bug fix)
- [ ] `poetry run pytest` passes
- [ ] `poetry run black --check . && poetry run ruff check .` passes
- [ ] `CHANGELOG.md` updated if the change is user-facing
- [ ] No hardcoded secrets, API keys, or PII
- [ ] CLA signed (the bot will prompt on your first PR)

Your PR will be reviewed by a maintainer within **3 business days**. If it's been longer, feel free to `@mention` one of us.

---

## Project layout

```
simulate-sdk/
├── fi/
│   └── simulate/
│       ├── __init__.py           # Public API surface
│       ├── agent/
│       │   ├── definition.py     # AgentDefinition, SimulatorAgentDefinition, config models
│       │   ├── wrapper.py        # AgentWrapper base class, AgentInput, AgentResponse
│       │   └── wrappers/         # Built-in wrapper implementations
│       │       ├── openai.py
│       │       ├── anthropic.py
│       │       ├── gemini.py
│       │       └── langchain.py
│       ├── simulation/
│       │   ├── models.py         # Persona, Scenario, TestReport, TestCaseResult
│       │   ├── runner.py         # TestRunner
│       │   ├── generator.py      # ScenarioGenerator
│       │   └── engines/          # LiveKit and Cloud simulation backends
│       ├── evaluation/
│       │   └── ai_eval.py        # evaluate_report
│       └── recording/            # Room recording utilities
├── tests/
│   ├── test_runner.py
│   ├── test_wrappers.py
│   └── test_custom_agents.py
├── pyproject.toml
└── CHANGELOG.md
```

---

## Releases

This project follows [SemVer](https://semver.org/). Release notes live in [CHANGELOG.md](CHANGELOG.md). Releases are published to PyPI as [`agent-simulate`](https://pypi.org/project/agent-simulate/).

---

## Thanks

We read every issue and PR. If we don't respond within a few days, it's not intentional — please ping us. We're a small team trying to build something great with you.

Star the repo ⭐, join Discord 💬, and ship something you're proud of. ❤️
