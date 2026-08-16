---
name: Feature request
about: Suggest an idea for vibe-forge
title: "[feature] "
labels: enhancement
assignees: ''
---

**Problem / motivation**
What problem does this solve? (e.g. "benchmark results are hard to compare across releases")

**Proposed solution**
Concise description of the feature and how it should behave.

**Alternative approaches**
Anything else you considered.

**Relation to routing pipeline**
Which layer does this touch?

- [ ] scorer (`vibeforge/router/complexity.py`)
- [ ] registry / models.yaml
- [ ] executor (Ollama)
- [ ] benchmark suite
- [ ] dashboard / API
- [ ] CLI
- [ ] other

**Testing plan**
How should this be tested? (fakes preferred — no live Ollama in CI)