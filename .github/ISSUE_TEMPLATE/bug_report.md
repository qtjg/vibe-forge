---
name: Bug report
about: Report something that isn't working as expected
title: "[bug] "
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what is broken.

**Steps to reproduce**
1. Run `vibeforge route "..." --type ...` (or whichever command)
2. ...
3. See error

**Expected behavior**
What should have happened instead.

**Environment**
- vibe-forge version (`vibeforge --version`)
- Python version
- OS
- Ollama version + models pulled (`ollama list`)
- RAM/VRAM (relevant for hardware-aware routing)

**Diagnostics**
```bash
vibeforge route "the prompt" --type debug --execute --json
```

**Additional context**
- Config file contents (redact anything sensitive): `models.yaml`
- Dashboard API responses, if relevant: `curl http://localhost:8420/api/decisions`