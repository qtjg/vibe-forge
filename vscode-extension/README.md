# vibe-forge VS Code extension (experimental)

Routes the selected text through the local vibe-forge dashboard: picks a
model tier for the selection, shows the decision, and can run the prompt
through the chosen Ollama model. All decision logic lives in the Python
package — this extension is thin glue over the dashboard API
(`POST /api/route`, `POST /api/execute`), which is covered by
`tests/test_dashboard.py`.

## Prerequisites

- `vibeforge serve` running locally (default `http://127.0.0.1:8420`).
- Ollama running with at least the routing targets pulled
  (`ollama pull llama3.1:latest` etc., per your `models.yaml`).

## Try it locally

1. Install the extension:

   - VS Code GUI: `Extensions` → `...` → `Install from VSIX...`
     (see `code --install-extension` below for a CLI option), or
   - from the repo root, press `F5` inside VS Code with this folder open
     (Extension Development Host).

2. Open a file, select some text, and run **vibe-forge: Route Selection**
   (command palette or the editor context menu).

3. The extension shows the decision (complexity tier, chosen model,
   reason). Click **Run it** to execute the prompt against that model;
   the generated output appears in the `vibe-forge` output channel.

The dashboard URL is configurable via `vibeForge.dashboardUrl`
(default `http://127.0.0.1:8420`).

## Notes

- No build step; plain CommonJS. `npm run lint` just syntax-checks
  (`node --check`).
- This is glue, not a product: expect the extension to keep changing
  with the dashboard API.