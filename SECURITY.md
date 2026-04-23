# Security Policy

## Reporting a Vulnerability

The Future AGI team takes security seriously. If you discover a security vulnerability in `agent-simulate` (the simulate-sdk), please report it privately — **do not open a public GitHub issue.**

**Email:** **security@futureagi.com**

Please include as much of the following as you can:

- Type of issue (e.g. remote code execution, credential leak, SSRF, dependency vulnerability)
- SDK version (`pip show agent-simulate`) and Python version
- Whether the issue is in local LiveKit mode, Cloud mode, or both
- Full paths of source file(s) related to the issue
- Step-by-step instructions to reproduce
- Proof-of-concept or exploit code (if possible)
- Impact — how an attacker might exploit it

## Response timeline

- **Acknowledgement:** within 24 hours of report (Mon–Fri, Pacific & IST)
- **Initial assessment:** within 3 business days
- **Fix target:** depends on severity (see below)
- **Public disclosure:** coordinated with the reporter, typically 7–90 days after a patch is available

### Severity and fix targets

| Severity | Examples | Target |
|---|---|---|
| 🔴 Critical | RCE via malicious input, credential exfiltration, auth bypass | Patch within 72 hours |
| 🟠 High | Privilege escalation, tenant isolation breach, secret leak to logs | Patch within 7 days |
| 🟡 Medium | Information disclosure, injection with limited scope | Patch within 30 days |
| 🟢 Low | Hardening gaps, minor info leak | Next scheduled release |

## Scope

**In scope:**

- The `agent-simulate` Python package on PyPI
- The `future-agi/simulate-sdk` GitHub repository
- Client-side HTTP code in the SDK that talks to `api.futureagi.com` (how the SDK sends requests — not the server itself)

**Out of scope (report to the appropriate repo):**

- Server-side vulnerabilities in `api.futureagi.com` or `app.futureagi.com` — report to [`future-agi/future-agi`](https://github.com/future-agi/future-agi)
- Third-party dependencies (`livekit-agents`, `ai-evaluation`, LLM provider SDKs) — report upstream
- Denial-of-service via traffic volume
- Social-engineering attacks on Future AGI employees

## Safe harbor

We will not pursue legal action against security researchers who:

- Make a good-faith effort to avoid privacy violations, destruction of data, and service interruption
- Only interact with accounts they own or with explicit permission of the account holder
- Do not exploit a vulnerability beyond what is necessary to confirm its existence
- Report the vulnerability promptly
- Do not publicly disclose the vulnerability before a patch is released

## Acknowledgement

We maintain a [Security Researcher Hall of Fame](https://futureagi.com/security/hall-of-fame) and are happy to credit reporters who wish to be named. For qualifying reports, we run a bug bounty via HackerOne — contact security@futureagi.com for details.

## PGP

If you prefer encrypted communication, our PGP key is available at:
<https://futureagi.com/.well-known/pgp-key.txt>

---

Thanks for helping keep Future AGI and our users safe. ❤️
