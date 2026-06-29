"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any, cast

from pydantic import BaseModel

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


class ClassificationOutput(BaseModel):
    route: str
    risk_level: str = "low"
    reason: str = ""


class ApprovalInterruptRequest(BaseModel):
    query: str
    proposed_action: str | None = None
    risk_level: str = "high"
    allowed_decisions: tuple[str, str, str] = ("approved", "rejected", "edit")


class ApprovalInterruptResponse(BaseModel):
    decision: str = "rejected"
    reviewer: str = "human-reviewer"
    comment: str = ""


def _latency_ms(start: float) -> int:
    return int((perf_counter() - start) * 1000)


def _fallback_route(query: str) -> ClassificationOutput:
    text = query.lower()
    risky_words = ("refund", "delete", "cancel", "send", "confirmation email")
    tool_words = ("lookup", "status", "search", "track", "find order")
    error_words = ("timeout", "failure", "failed", "crash", "cannot recover", "error")
    missing_prefixes = ("can you fix", "help", "it broke")
    if any(word in text for word in risky_words):
        return ClassificationOutput(
            route="risky",
            risk_level="high",
            reason="side-effecting action",
        )
    if any(word in text for word in tool_words):
        return ClassificationOutput(route="tool", risk_level="low", reason="requires lookup")
    if any(word in text for word in error_words):
        return ClassificationOutput(route="error", risk_level="low", reason="operational failure")
    if len(text.split()) <= 4 or any(text.startswith(prefix) for prefix in missing_prefixes):
        return ClassificationOutput(
            route="missing_info",
            risk_level="low",
            reason="missing context",
        )
    return ClassificationOutput(route="simple", risk_level="low", reason="general guidance")


def _normalize_route(query: str, route: str) -> str:
    text = query.lower().strip()
    how_to_prefixes = (
        "how do i ",
        "how can i ",
        "what is ",
        "where do i ",
        "where can i ",
    )
    if route == "missing_info" and any(text.startswith(prefix) for prefix in how_to_prefixes):
        return "simple"
    return route


def _approval_result_text(approval: dict[str, Any] | ApprovalDecision | None) -> str:
    if isinstance(approval, dict):
        decision = str(approval.get("decision", "rejected"))
        reviewer = str(approval.get("reviewer", "unknown"))
        comment = str(approval.get("comment", ""))
        return f"Decision: {decision}; reviewer: {reviewer}; comment: {comment}"
    if approval is not None:
        return (
            f"Decision: {approval.decision}; reviewer: {approval.reviewer}; "
            f"comment: {approval.comment}"
        )
    return "No approval step."


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    start = perf_counter()
    query = state.get("query", "")
    try:
        llm = cast(Any, get_llm(temperature=0.0)).with_structured_output(ClassificationOutput)
        result = llm.invoke(
            [
                (
                    "system",
                    "Classify support requests into one route: simple, tool, "
                    "missing_info, risky, error. Priority order is risky > tool > "
                    "missing_info > error > simple. Return risky for side effects "
                    "like refunds, deletes, cancellations, or sending emails. "
                    "Return tool for information lookup/search requests. Return "
                    "missing_info for vague requests. Return error for system "
                    "failures such as timeout/crash/unavailable. Set risk_level "
                    "to high only for risky.",
                ),
                ("human", query),
            ]
        )
    except Exception as exc:
        result = _fallback_route(query)
        result.reason = f"fallback:{type(exc).__name__}"
    valid_routes = {"simple", "tool", "missing_info", "risky", "error"}
    route = result.route if result.route in valid_routes else "simple"
    route = _normalize_route(query, route)
    risk_level = "high" if route == "risky" else result.risk_level or "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                risk_level=risk_level,
                reason=result.reason,
                latency_ms=_latency_ms(start),
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    query = state.get("query", "")
    if route == "error" and attempt < 2:
        result = f"ERROR[TIMEOUT]: transient tool timeout on attempt {attempt + 1}"
        event_type = "tool_error"
        error_type = "timeout"
    elif route == "risky":
        result = f"ACTION_OK: approved action prepared for query '{query}'"
        event_type = "completed"
        error_type = None
    elif route == "tool":
        result = f"LOOKUP_OK: retrieved support data for '{query}'"
        event_type = "completed"
        error_type = None
    else:
        result = f"RECOVERY_OK: recovered execution for '{query}'"
        event_type = "completed"
        error_type = None
    return {
        "tool_results": [result],
        "events": [
            make_event(
                "tool",
                event_type,
                "tool executed",
                attempt=attempt,
                route=route,
                error_type=error_type,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR[" in latest else "success"
    error_type = "timeout" if "ERROR[TIMEOUT]" in latest else None
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"evaluation={evaluation_result}",
                error_type=error_type,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    start = perf_counter()
    query = state.get("query", "")
    tool_context = "\n".join(state.get("tool_results", [])) or "No tool results."
    approval = state.get("approval")
    approval_text = _approval_result_text(approval)
    try:
        llm = cast(Any, get_llm(temperature=0.2))
        message = llm.invoke(
            [
                (
                    "system",
                    "Write a concise support-agent response grounded only in the "
                    "provided context. If a tool result exists, use it explicitly. "
                    "If approval exists, mention the approved action. Do not invent "
                    "facts outside the context.",
                ),
                (
                    "human",
                    f"Query: {query}\nTool context:\n{tool_context}\n"
                    f"Approval context:\n{approval_text}",
                ),
            ]
        )
        final_answer = getattr(message, "content", str(message)).strip()
    except Exception as exc:
        suffix = f" Approved context: {approval_text}" if approval else ""
        final_answer = f"Support response for '{query}'. Context: {tool_context}.{suffix}".strip()
        final_answer += f" [fallback:{type(exc).__name__}]"
    return {
        "final_answer": final_answer,
        "events": [
            make_event("answer", "completed", "response generated", latency_ms=_latency_ms(start))
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    question = (
        f"Can you clarify the exact issue for '{state.get('query', '')}' "
        "and include any order, account, "
        "or error details so I can help safely?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    action = f"Proposed risky action for review: {state.get('query', '')}"
    return {
        "proposed_action": action,
        "events": [make_event("risky_action", "completed", "approval required", risk_level="high")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str},
    "events": [make_event(...)]}
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        request = ApprovalInterruptRequest(
            query=state.get("query", ""),
            proposed_action=state.get("proposed_action"),
            risk_level=state.get("risk_level", "high"),
        )
        payload = interrupt(request.model_dump()) or {}
        response = ApprovalInterruptResponse.model_validate(payload)
        valid_decisions = {"approved", "rejected", "edit"}
        decision = response.decision if response.decision in valid_decisions else "rejected"
        approved = decision == "approved"
        reviewer = response.reviewer
        comment = response.comment or "interactive decision"
        event_type = "interrupted"
    else:
        decision = "approved"
        approved = True
        reviewer = "mock-reviewer"
        comment = "approved by default for offline execution"
        event_type = "completed"
    return {
        "approval": {
            "approved": approved,
            "decision": decision,
            "reviewer": reviewer,
            "comment": comment,
        },
        "events": [
            make_event(
                "approval",
                event_type,
                "approval decision recorded",
                approved=approved,
                decision=decision,
                reviewer=reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0) + 1
    message = f"RETRY_ERROR[TIMEOUT]: retry attempt {attempt} after transient failure"
    return {
        "attempt": attempt,
        "errors": [message],
        "events": [
            make_event(
                "retry",
                "completed",
                "retry scheduled",
                attempt=attempt,
                error_type="timeout",
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    final_answer = (
        "I could not complete this request after the maximum retry limit. "
        "Please escalate to a human operator with the incident details."
    )
    return {
        "final_answer": final_answer,
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "request moved to dead letter",
                attempt=state.get("attempt", 0),
                max_attempts=state.get("max_attempts", 0),
                error_type="timeout",
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    approval = state.get("approval")
    approved = None
    if isinstance(approval, dict):
        approved = bool(approval.get("approved"))
        decision = str(approval.get("decision", "rejected"))
    elif approval is not None:
        approved = approval.approved
        decision = approval.decision
    else:
        decision = None
    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route"),
                attempt=state.get("attempt", 0),
                approved=approved,
                decision=decision,
            )
        ]
    }
