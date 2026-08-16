// vibe-forge extension: route the selected text through the local
// vibe-forge dashboard (vibeforge serve) and optionally run it.
//
// All decision logic lives in the Python package; this file is thin
// glue: POST /api/route, show the decision, POST /api/execute on demand.
//
// Tested indirectly: the API contract it calls is covered by
// tests/test_dashboard.py (route + execute + validation + failures).

"use strict";

const vscode = require("vscode");

/** Task types accepted by the dashboard API (mirrors vibeforge.types.TaskType). */
const TASK_TYPES = [
  "autocomplete",
  "debug",
  "explain",
  "refactor",
  "review",
  "documentation",
  "test",
  "code",
];

function dashboardUrl() {
  const configured = vscode.workspace
    .getConfiguration("vibeForge")
    .get("dashboardUrl", "http://127.0.0.1:8420");
  return String(configured).replace(/\/+$/, "");
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`dashboard returned ${response.status}: ${body.detail || response.statusText}`);
  }
  return body;
}

/**
 * Route `prompt` through the dashboard and return the decision dict.
 * Exported for testability (no vscode dependency in the pure functions).
 */
async function routeTask(prompt, taskType, url) {
  return postJson(`${url}/api/route`, {
    prompt,
    task_type: taskType,
    context: "",
  });
}

/**
 * Run `prompt` against a model tag and return the execution result.
 * Exported for testability.
 */
async function runTask(prompt, modelTag, url) {
  return postJson(`${url}/api/execute`, { prompt, model_tag: modelTag });
}

async function pickTaskType() {
  const picked = await vscode.window.showQuickPick(TASK_TYPES, {
    placeHolder: "What kind of task is the selection?",
  });
  return picked || undefined;
}

async function routeSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    void vscode.window.showWarningMessage("vibe-forge: no active editor.");
    return;
  }
  const selection = editor.selection;
  const prompt = selection.isEmpty
    ? `${editor.document.fileName}\n${editor.document.getText()}`
    : editor.document.getText(selection);
  if (!prompt.trim()) {
    void vscode.window.showWarningMessage("vibe-forge: selection is empty.");
    return;
  }

  const taskType = await pickTaskType();
  if (!taskType) {
    return;
  }

  const url = dashboardUrl();
  let decision;
  try {
    decision = await routeTask(prompt, taskType, url);
  } catch (error) {
    void vscode.window.showErrorMessage(
      `vibe-forge: ${error.message} — is the dashboard running? Start it with "vibeforge serve".`
    );
    return;
  }

  const run = await vscode.window.showInformationMessage(
    `vibe-forge: ${decision.complexity} → ${decision.model} (${decision.ollama_tag}) — ${decision.reason}`,
    "Run it"
  );
  if (run !== "Run it") {
    return;
  }

  try {
    const result = await runTask(prompt, decision.ollama_tag, url);
    const channel = vscode.window.createOutputChannel("vibe-forge");
    if (result.status === "ok") {
      channel.appendLine(`vibe-forge run (${decision.ollama_tag}, ${result.latency_ms}ms):`);
      channel.appendLine("─".repeat(60));
      channel.appendLine(result.output || "(empty response)");
      channel.show();
    } else {
      void vscode.window.showErrorMessage(
        `vibe-forge: run failed (${result.error_kind}): ${result.error}`
      );
    }
  } catch (error) {
    void vscode.window.showErrorMessage(`vibe-forge: ${error.message}`);
  }
}

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("vibeForge.routeSelection", routeSelection)
  );
}

function deactivate() {}

module.exports = { activate, deactivate, routeTask, runTask };