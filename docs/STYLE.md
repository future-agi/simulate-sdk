# Future AGI — SDK README Style & Vocabulary Guide

A single-file reference for polishing any SDK in the Future AGI ecosystem — this repo (`simulate-sdk`) plus the 2 remaining SDKs going open-source alongside it. Copy this file into the target SDK repo, apply the rules below, and your README will ship on-brand without you re-deriving the voice from scratch.

Source of truth for voice: the main platform README at [`future-agi/future-agi`](https://github.com/future-agi/future-agi). If anything in this guide contradicts the main README, the main README wins — update this file.

---

## 1. Brand voice — the rules

**Short. Declarative. Concrete.** Every sentence earns its spot.

- **Write in 2nd person active voice.** "You run X" beats "Users can run X."
- **Lead with the fact, not the setup.** "`evaluate_report` scores transcripts" beats "This helper function, which we designed to be easy to use, scores transcripts."
- **Use concrete numbers.** "50+ metrics" beats "many metrics." "P99 ≤ 21 ms" beats "fast."
- **Keep sentences short.** If a sentence has more than one comma, rewrite it.
- **Name things once.** Define `Persona` on first use, then just say `Persona` — not "the Persona object" or "a Persona instance."
- **Show, don't sell.** A 15-line code snippet that runs is worth a 200-word pitch.

### Banned phrases (hard nos)

Grep for these in the final README — zero hits expected:

| Don't say | Say instead |
|---|---|
| "dive into", "deep dive" | Link to the section or the code. |
| "seamless", "seamlessly" | Cut the word. If integration is trivial, show it in 3 lines of code. |
| "effortless" | Same — cut or show. |
| "leverage" | "use" |
| "utilize" | "use" |
| "robust", "powerful" | Cut and replace with the specific capability. |
| "revolutionary", "cutting-edge", "next-gen" | Cut. Let the feature speak. |
| "unleash", "unlock", "empower" | Cut. |
| "one-stop shop", "swiss-army knife" | Cut. |
| "simply", "just" | Remove — they minimize the reader's difficulty and usually lie. |
| "AI-powered" (on its own) | Be specific: "LLM-as-judge scoring", "OpenTelemetry tracing". |
| "blazing fast" | Cite a number with a test. |
| "game-changing" | Cut. |

### Stylistic tells to avoid

- **Rule-of-three padding.** "Fast, reliable, and scalable." Pick one and make it specific, or drop the list.
- **Em-dash soup.** Use em-dashes sparingly (main README averages ~1 per section). Don't chain them.
- **Fake contrast.** "Not just X — Y." Usually Y on its own is stronger.
- **"It's never been easier to..."** — cut.
- **Adverb stacking.** "Dramatically significantly faster" → pick one, ideally with a number.
- **Colon title dumps.** `"## Monitoring: observability for modern AI: the new way"` → pick one subtitle.

---

## 2. Product vocabulary — canonical terms

### Product name

- **Future AGI** — two words, spaces, both capitalized. Never "FutureAGI", "Future-AGI", "future agi", or "FAGI".
- Legal entity: **Future AGI, Inc.** (use in LICENSE, NOTICE, CLA only).

### Platform pillars (fixed capitalization, always)

When referring to a platform capability as a *product pillar*, use **Title Case**:

- **Simulate** — agent simulation
- **Evaluate** — evaluation metrics and judges
- **Control** — guardrails and policies
- **Monitor** — OpenTelemetry tracing and dashboards
- **Optimize** — prompt optimization
- **Agent Command Center** — the LLM gateway (not "the gateway" in body copy; that's fine informally)

In code, API names, or shell examples, lowercase naturally: `simulate()`, `evaluate_report`, `tracer`, etc.

### Positioning & taglines

Main platform tagline (do not reuse verbatim on SDK READMEs — write an SDK-specific one):

> **AI Agents hallucinate. Fix it faster.**

SDK positioning template:

> **`<SDK name>` — `<one-sentence value prop>`**
> Python SDK for the `<Pillar>` pillar of [Future AGI](https://github.com/future-agi/future-agi). `<what it does in 1 sentence>`.

Example (this SDK):

> **Simulate — test AI agents before users meet them**
> Python SDK for the `Simulate` pillar of Future AGI. Run voice and text agents against persona-driven scenarios, capture transcripts and audio, and feed results straight into evals.

### Support matrix — what this SDK can and can't do

Be honest. The main platform README promotes the **aspirational** capability set across the whole product. Each SDK should promote only what it **actually supports today**.

> **⚠️ Template block — replace before shipping.** Each SDK needs its own version of this section in its own README, derived from the actual code surface. **Do not copy-paste this list into another SDK.** Grep the code, verify every bullet, and write your own.

**Example for `simulate-sdk`** (the SDK this guide originally shipped with):

- ✅ Voice agents via **LiveKit** (local WebRTC).
- ✅ Text agents via Cloud mode + wrappers for **OpenAI, Anthropic, LangChain, Gemini**.
- ❌ Do **not** claim VAPI / Retell / Pipecat support. That's a main-platform claim; this SDK doesn't implement those backends yet.
- ⚠️ **Audio capture is opt-in** (`record_audio=True`). Don't imply it's automatic.
- ⚠️ **Cloud mode returns an empty local `TestReport`** — scores live in the backend dashboard. Call this out explicitly.

**How to build this section for a new SDK:**

1. List every public function/class exported from the package root.
2. For each, write one line: "✅ what works" or "⚠️ caveat" or "❌ what users might expect but the SDK does NOT do."
3. Cross-check against the main platform README — anywhere the platform claims something but this SDK doesn't implement it, add an explicit ❌ bullet so users don't assume overlap.
4. Over-promising is worse than under-promising. Every claim must have a test or a demo that proves it.

### Nouns (consistent capitalization in prose; lowercase in code)

Prose `Persona`, `Scenario`, `Run`, `Transcript`, `Evaluation`, `Run Test` — title case when referring to the product concept. Lowercase when using the identifier in code.

Example: "A `Run Test` groups multiple Runs" (prose) vs `run_test_name="..."` (code).

---

## 3. README structure — the section order

Every SDK README in the ecosystem follows this order. Keep it, skip sections that don't apply, but don't reorder.

1. **Banner** — centered logo, link to `futureagi.com`. Use the shared logo asset in `future-agi/future-agi/.github/assets/logo-banner{,-dark}.png` with a `<picture>` tag for dark mode.
2. **H1** — `<SDK positioning>` per the template above.
3. **Sub** — one-sentence value prop (bolded) + one explanatory sentence.
4. **Badges row** — flat-square style, in this order: PyPI version · Python versions · License · Downloads · Discord.
5. **Nav row** — PyPI · Docs · Platform · Main repo · Discord · Issues.
6. **`---`**
7. **Why this SDK?** — 2–3 short paragraphs. Tie back to the "agents hallucinate" thesis without repeating the main-README verbatim.
8. **What's in the box** — 3-column `<table>`. Each column is a single capability headline + 2–3 lines.
9. **Install** — one code block with core + each extra. Python version range. Optional `<details>` for one-time setup (VAD weights, CLI auth, etc.).
10. **Quickstart** — one per supported mode. **Each runnable as-is.** Include required env vars.
11. **API highlights** — small tables for wrappers, models, adapters, whatever the SDK exposes.
12. **Companion SDK** — if this SDK pairs with another (eg. `simulate` → `ai-evaluation`), show the pairing.
13. **How this fits into Future AGI** — mini map: pillar loop + 1-row table of sibling SDKs.
14. **Roadmap** — 3-column `<table>`: Shipped · In progress · Coming up. Keep it honest; prune done items that are >6 months old.
15. **🤝 Contributing** — 4 bullet points: good first issues · Contributing Guide · Discord · CLA.
16. **🌍 Community & support** — 2-column table: channel · purpose.
17. **📄 License** — one line, link to LICENSE + NOTICE.
18. **Footer** — centered, "Built by …", star-ask, 3 canonical links.

---

## 4. Visual grammar — badges, emoji, tables

### Badges

- **Style:** `style=flat-square` (never `for-the-badge` on SDK READMEs).
- **Order:** package version → language version → license → downloads → community.
- **Source:** prefer `shields.io` over custom PNGs.
- **Alt text:** always set, specific (e.g., `alt="PyPI"`, not `alt="badge"`).

### Emoji policy

Sparing. Match main README's placement:

- 🚀 — Quickstart headings
- 🤝 — Contributing
- 🌍 — Community
- 📄 — License
- 🐛 / ✨ / 🔖 — issue-triage verbs
- ⭐ — star-ask only, in footer
- ❤️ — closing line of CONTRIBUTING and SECURITY

**Don't** sprinkle emoji on every heading. **Don't** use flag emoji. **Don't** use emoji in code comments, commit messages, or file paths.

### Tables

Use `<table>` (HTML) for side-by-side layouts (3-column feature cards, quickstart variants). Use markdown pipe tables for data (wrappers, classifiers, env vars).

Always include `width="33%"` on `<td>` cells when laying out a 3-column card — GitHub's default column widths are uneven otherwise.

### Code blocks

- Specify the language (```` ```python ````, ```` ```bash ````, ```` ```typescript ````).
- Keep Quickstart snippets **runnable top-to-bottom** with only required env vars. Cut optional parameters from the first example.
- Output lines, if shown, go in a separate block.

### Collapsibles

Use `<details>` for anything that applies to a minority of users but is load-bearing for them — VAD weight download, air-gapped install, alternative installers. Don't hide the main path behind one.

---

## 5. Canonical URLs

Copy-paste these. Don't make up variants.

| Purpose | URL |
|---|---|
| Website | https://futureagi.com |
| Docs | https://docs.futureagi.com |
| Platform | https://app.futureagi.com |
| API | https://api.futureagi.com |
| Main repo | https://github.com/future-agi/future-agi |
| Discord | https://discord.gg/UjZ2gRT5p |
| Twitter | https://twitter.com/futureagi |
| Status | https://status.futureagi.com |
| Blog | https://futureagi.com/blog |
| YouTube | https://youtube.com/@futureagi |
| Support email | support@futureagi.com |
| Security email | security@futureagi.com |
| Conduct email | conduct@futureagi.com |

### Sibling SDKs (update as each ships)

| SDK | GitHub | Package |
|---|---|---|
| `agent-simulate` | `future-agi/simulate-sdk` | PyPI: `agent-simulate` |
| `ai-evaluation` | `future-agi/ai-evaluation` | PyPI: `ai-evaluation`, npm: `@future-agi/ai-evaluation` |
| `traceAI` | `future-agi/traceAI` | PyPI: `fi-instrumentation-otel`, npm: `@traceai/fi-core` |
| `agent-opt` | `future-agi/agent-opt` | PyPI: `agent-opt` |
| `futureagi` (platform SDK) | `future-agi/futureagi-sdk` | PyPI: `futureagi` |
| `agentcc` (gateway) | `future-agi/agent-command-center-sdk` | PyPI: `agentcc`, npm: `@agentcc/client` |

---

## 6. PyPI / npm metadata — SEO checklist

For every Python SDK, make sure `pyproject.toml` has:

- **`description`** — 100–140 characters. Start with the SDK name, end with "— the `<Pillar>` SDK for Future AGI." Example: `"Simulate, record, and score conversations with voice and text AI agents — the Simulate SDK for Future AGI."`
- **`keywords`** — 8–12 entries. Always include: `ai`, `llm`, `agents`, `future-agi`. Then 4–8 SDK-specific terms (e.g., `voice-ai`, `livekit`, `simulation`, `evaluation`).
- **`classifiers`** — at minimum: License (`Apache Software License`), `Development Status`, all supported Python minor versions, `Topic :: Scientific/Engineering :: Artificial Intelligence`, and one SDK-specific topic (Testing, Monitoring, etc.).
- **`[tool.poetry.urls]`** — `Homepage`, `Documentation`, `Repository`, `Issues`, `Changelog`, plus `Main platform` pointing at `future-agi/future-agi`.

For every TypeScript SDK, the npm `package.json` equivalents: `description`, `keywords`, `homepage`, `repository`, `bugs`, `license: "Apache-2.0"`.

---

## 7. GitHub repo settings — apply after merge

Beyond the README, the SDK repo itself needs a few knobs set.

### Topics (repo Settings → Topics)

Baseline for every SDK: `python` (or `typescript`), `ai-agents`, `llm`, `future-agi`, `observability`.

Add 5–10 SDK-specific topics from the SDK's `keywords` list. For `simulate-sdk`:

```
python ai-agents voice-ai livekit llm-evaluation agent-testing
conversational-ai simulation openai anthropic langchain gemini
future-agi observability openai-agents-sdk
```

### About section

Match the `pyproject.toml` `description` field verbatim. Include the docs URL in the Website field.

### Branch protection

- `main` — protected. Require PR review, require CLA signed, require CI green.

### Labels

Copy main repo's label set via `gh label clone future-agi/future-agi` (or equivalent). Minimum: `bug`, `feature`, `triage`, `good first issue`, `help wanted`, `documentation`, `breaking change`.

### Releases

Tag per SemVer, publish a GitHub Release per PyPI version, paste the CHANGELOG entry as the release body.

---

## 8. Files every SDK repo must have

All mirrored from the main repo's style, scoped down for an SDK:

| File | Status | Source |
|---|---|---|
| `README.md` | Custom per SDK | this guide |
| `LICENSE` | Copy Apache 2.0 verbatim | `future-agi/future-agi/LICENSE` |
| `NOTICE` | Copy + swap product name | `future-agi/future-agi/NOTICE` |
| `CONTRIBUTING.md` | Scoped per SDK | `future-agi/future-agi/CONTRIBUTING.md` as template |
| `SECURITY.md` | Copy, swap in-scope / out-of-scope sections | `future-agi/future-agi/SECURITY.md` |
| `CHANGELOG.md` | Per-SDK history | — |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Scoped per SDK | main repo's |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Scoped per SDK | main repo's |
| `.github/ISSUE_TEMPLATE/config.yml` | Same contact links everywhere | main repo's |
| `.github/PULL_REQUEST_TEMPLATE.md` | Scoped per SDK | main repo's |
| `docs/STYLE.md` | This file | this file |

---

## 9. Pre-ship checklist

Run through this before merging the open-source polish PR:

- [ ] README renders cleanly on GitHub (badges load, `<picture>` dark mode works, all relative links resolve)
- [ ] No banned phrases (grep list in §1)
- [ ] Quickstart(s) run top-to-bottom against a real deployment
- [ ] Every claim in the README is true of shipped code (no aspirational features)
- [ ] `pyproject.toml` has `description`, `keywords`, `classifiers`, `urls` — and `poetry check` passes
- [ ] `python -m build` produces `dist/*`; `twine check dist/*` passes
- [ ] `PKG-INFO` inside the built tarball shows the description, keywords, URLs
- [ ] `LICENSE`, `NOTICE`, `CONTRIBUTING.md`, `SECURITY.md` present at repo root
- [ ] `.github/ISSUE_TEMPLATE/` and `.github/PULL_REQUEST_TEMPLATE.md` present
- [ ] GitHub repo topics applied (see §7)
- [ ] Repo About / Website fields set
- [ ] `CHANGELOG.md` has an entry for this docs pass

---

## 10. When in doubt

Compare to the main repo README side-by-side. If a section feels heavier or lighter than its counterpart there, you're probably out of calibration.

The main README is the source of voice — this file is its reusable cheat sheet. If the main README changes, refresh this file; if this file drifts from the main, the main wins.
