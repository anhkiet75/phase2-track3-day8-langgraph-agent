"""Report generation helper.

TODO(student): implement report rendering using MetricsReport data
and the template in reports/lab_report_template.md.
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path
from typing import Any, cast

from .metrics import MetricsReport


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _graph_mermaid() -> str:
    try:
        from .graph import build_graph
        from .persistence import build_checkpointer

        graph = cast(Any, build_graph(checkpointer=build_checkpointer("memory")))
        raw_mermaid = graph.get_graph().draw_mermaid()
        if raw_mermaid.startswith("---"):
            parts = raw_mermaid.split("---\n", 2)
            if len(parts) == 3:
                raw_mermaid = parts[2]
        raw_mermaid = raw_mermaid.replace(
            "classDef default fill:#f2f0ff,line-height:1.2",
            "classDef default fill:#dbeafe,stroke:#0369a1,color:#0f172a,"
            "stroke-width:2px,line-height:1.2",
        )
        raw_mermaid = raw_mermaid.replace(
            "classDef first fill-opacity:0",
            "classDef first fill:#e0f2fe,stroke:#0369a1,color:#0f172a,stroke-width:2px",
        )
        raw_mermaid = raw_mermaid.replace(
            "classDef last fill:#bfb6fc",
            "classDef last fill:#bfdbfe,stroke:#0369a1,color:#0f172a,stroke-width:2px",
        )
        raw_mermaid += (
            "\nlinkStyle default stroke:#0ea5e9,stroke-width:2px;"
            "\nclassDef edgeLabel fill:#ffffff,color:#111827;"
        )
        theme_init = (
            '%%{init: {"theme":"dark","themeVariables":{'
            '"background":"#1f2430",'
            '"primaryColor":"#dbeafe",'
            '"primaryTextColor":"#0f172a",'
            '"primaryBorderColor":"#38bdf8",'
            '"lineColor":"#38bdf8",'
            '"secondaryColor":"#bfdbfe",'
            '"secondaryTextColor":"#0f172a",'
            '"tertiaryColor":"#bfdbfe",'
            '"tertiaryTextColor":"#0f172a",'
            '"nodeTextColor":"#0f172a",'
            '"textColor":"#e5e7eb"'
            '}}}%%'
        )
        return f"{theme_init}\n{raw_mermaid}"
    except Exception:
        return "graph TD\n  unavailable[Diagram unavailable]"


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    TODO(student): Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    summary_rows = "\n".join(
        [
            f"| Total scenarios | {metrics.total_scenarios} |",
            f"| Success rate | {metrics.success_rate:.2%} |",
            f"| Avg nodes visited | {metrics.avg_nodes_visited:.2f} |",
            f"| Total retries | {metrics.total_retries} |",
            f"| Total interrupts | {metrics.total_interrupts} |",
            f"| Resume success | {metrics.resume_success} |",
        ]
    )
    scenario_rows = "\n".join(
        (
            f"| {item.scenario_id} | {item.expected_route} | {item.actual_route} | "
            f"{item.success} | {item.retry_count} | {item.interrupt_count} | "
            f"{item.approval_result or '-'} |"
        )
        for item in metrics.scenario_metrics
    )
    approved_total = sum(
        1 for item in metrics.scenario_metrics if item.approval_result == "approved"
    )
    rejected_total = sum(
        1 for item in metrics.scenario_metrics if item.approval_result == "rejected"
    )
    edit_total = sum(1 for item in metrics.scenario_metrics if item.approval_result == "edit")
    approval_results = {
        "approved": approved_total,
        "rejected": rejected_total,
        "edit": edit_total,
    }
    architecture = (
        "The graph uses a typed LangGraph state with append-only audit fields and "
        "overwrite-only decision fields. The core flow is `intake -> classify`, "
        "then conditional routing to `answer`, `tool`, `clarify`, or "
        "`risky_action -> approval`. Tool-backed flows pass through `evaluate` and, "
        "on transient typed timeout errors, enter the bounded `retry` loop before "
        "either recovering or moving to `dead_letter`. All terminal branches pass "
        "through `finalize` for audit completeness."
    )
    retry_analysis = (
        "typed timeout-style tool errors trigger `evaluate -> retry`; the loop is "
        "bounded by `attempt < max_attempts`, which prevents infinite execution "
        "and escalates to `dead_letter` when recovery fails."
    )
    risky_analysis = (
        "risky requests are isolated behind `risky_action -> approval`; if approval "
        "is denied, the graph routes to `clarify` instead of executing the side "
        "effect."
    )
    persistence_text = (
        "The workflow compiles with a checkpointer and uses `thread_id` per scenario "
        "run. When a persistent saver is configured, state history can be inspected "
        "or replayed from the same thread, and the metrics payload records whether "
        "that persistence path was observed successfully."
    )
    improvement_text = (
        "If I had one more day, I would productionize three areas first: stronger "
        "LLM-as-judge evaluation in `evaluate_node`, richer persistence demo "
        "coverage with explicit resume/replay tests, and provider-specific prompt "
        "tuning for more robust hidden-scenario classification."
    )
    commit_hash = _git_commit()
    graph_mermaid = _graph_mermaid()
    return f"""# Day 08 Lab Report

## 1. Team / student

- Name:
- Repo/commit: {commit_hash}
- Date: {date.today().isoformat()}

## 2. Architecture

{architecture}

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append | lightweight execution breadcrumbs |
| tool_results | append | preserves tool outputs across retries |
| errors | append | keeps typed retry/dead-letter history |
| events | append | audit trail and metrics source |
| route | overwrite | latest routing decision only |
| evaluation_result | overwrite | retry gate for current tool result |
| pending_question | overwrite | active clarification request |
| proposed_action | overwrite | current risky action awaiting approval |
| approval | overwrite | latest approval decision |
| final_answer | overwrite | terminal user-facing output |

## 4. Metrics summary

| Metric | Value |
|---|---:|
{summary_rows}

## 5. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Approval result |
|---|---|---|---:|---:|---:|---|
{scenario_rows}

Approval totals:
approved={approval_results["approved"]},
rejected={approval_results["rejected"]},
edit={approval_results["edit"]}.

## 6. Failure analysis

1. Retry or tool failure: {retry_analysis}
2. Risky action without approval: {risky_analysis}

## 7. Persistence / recovery evidence

{persistence_text}

## 8. Extension work

- SQLite checkpointer support via `SqliteSaver` with WAL mode.
- Optional real human-in-the-loop approval using `LANGGRAPH_INTERRUPT=true`.
- Approval interrupt request/response schema with explicit approval decisions:
  `approved`, `rejected`, and `edit`.
- Structured audit events with typed timeout metadata for retries and dead-letter handling.

## 9. Graph diagram

```mermaid
{graph_mermaid}
```

## 10. Improvement plan

{improvement_text}
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
