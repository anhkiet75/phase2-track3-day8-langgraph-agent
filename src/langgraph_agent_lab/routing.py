"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState


def _approval_approved(state: AgentState) -> bool:
    approval = state.get("approval")
    if isinstance(approval, dict):
        return bool(approval.get("approved"))
    if approval is not None:
        return bool(approval.approved)
    return False


def _approval_decision(state: AgentState) -> str:
    approval = state.get("approval")
    if isinstance(approval, dict):
        return str(approval.get("decision", "rejected"))
    if approval is not None:
        return approval.decision
    return "rejected"


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node.

    Mapping:
    - "simple"       → "answer"
    - "tool"         → "tool"
    - "missing_info" → "clarify"
    - "risky"        → "risky_action"
    - "error"        → "retry"
    - unknown/default → "answer"

    Hint: use a dict mapping for clean implementation.
    """
    return {
        "simple": "answer",
        "tool": "tool",
        "missing_info": "clarify",
        "risky": "risky_action",
        "error": "retry",
    }.get(state.get("route", ""), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Decide if tool result is satisfactory or needs retry.

    This is the 'done?' check that creates the retry loop —
    a key LangGraph advantage over linear LCEL chains.

    - If evaluation_result == "needs_retry" → "retry"
    - Otherwise → "answer"
    """
    return "retry" if state.get("evaluation_result") == "needs_retry" else "answer"


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry the tool or give up.

    MUST be bounded — unbounded retry loops will fail grading.

    - If attempt < max_attempts → "tool" (try again)
    - If attempt >= max_attempts → "dead_letter" (give up, escalate)
    """
    return "tool" if state.get("attempt", 0) < state.get("max_attempts", 0) else "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Route based on human approval decision.

    - If approved → "tool" (proceed with risky action)
    - If rejected → "clarify" (ask user for alternative)
    """
    decision = _approval_decision(state)
    if _approval_approved(state):
        return "tool"
    if decision == "edit":
        return "edit"
    return "clarify"
