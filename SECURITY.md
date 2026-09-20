# Security policy

vibe-forge is a local-first policy router for coding assistants. It routes
coding subtasks to local LLMs via Ollama and does not phone home.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:
[Report a vulnerability](https://github.com/qtjg/vibe-forge/security/advisories/new).

Please do not open a public issue for anything that could put users at risk
before a fix ships.

## Scope

In scope:

- The Ollama client and prompt-routing logic
- Model selection and task-classification correctness
- Any path that could leak prompts or source code outside the local machine
- The dashboard and any network surface it exposes

Out of scope:

- Behavior of third-party Ollama models (report those upstream)
- Local misconfiguration of Ollama itself

## Supported versions

Fixes ship to `main` and the latest PyPI release. Older releases are not
patched; the fix is to update.
