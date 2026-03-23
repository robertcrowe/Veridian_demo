"""
Shared tool implementations and agentic loop for the Veridian IT support agents.

Public API
----------
search_knowledge_base(query, kb_type)  -> dict
create_ticket(summary, priority, category, assigned_team) -> dict
get_escalation_policy(category)        -> dict
get_tool_definitions(agent_type)       -> list   ("base" = 3 tools, "adapted" = 2 tools)
run_agent_loop(client, model, messages, tools) -> dict
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from mistralai.client import Mistral

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_json(filename: str) -> Any:
    with open(_DATA_DIR / filename) as f:
        return json.load(f)


_KB: dict = _load_json("kb.json")
_INTERNAL_KB: dict = _load_json("internal_kb.json")
_ESCALATION: dict = _load_json("escalation_policy.json")

# In-memory ticket store (demo only — resets each process)
_TICKET_STORE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_knowledge_base(query: str, kb_type: str = "generic") -> dict:
    """
    Full-text search over the generic or Veridian-specific knowledge base.

    kb_type: "generic"   -> kb.json       (company-agnostic IT procedures)
             "internal"  -> internal_kb.json  (Veridian-specific procedures)
    """
    kb = _KB if kb_type == "generic" else _INTERNAL_KB
    query_tokens = set(query.lower().split())
    scored = []
    for article in kb["articles"]:
        text = (
            article["title"] + " " + article["content"] + " " + " ".join(article["tags"])
        ).lower()
        score = sum(1 for t in query_tokens if t in text)
        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [a for _, a in scored[:2]]

    if not top:
        return {"result": "No relevant articles found.", "articles": []}

    return {
        "result": f"Found {len(top)} relevant article(s).",
        "articles": [
            {"id": a["id"], "title": a["title"], "content": a["content"]}
            for a in top
        ],
    }


def create_ticket(
    summary: str,
    priority: str,
    category: str,
    assigned_team: str,
) -> dict:
    """
    Create a new IT support ticket and return its ID and SLA.

    priority: P1 | P2 | P3
    category: one of the 8 intent labels
    assigned_team: team name from the escalation policy (e.g. "IT Ops", "Security Team")
    """
    sla_map = {
        "P1": "15 minutes",
        "P2": "1 hour",
        "P3": "4 hours",
    }
    ticket_id = f"TKT-{str(uuid.uuid4())[:6].upper()}"
    ticket = {
        "id": ticket_id,
        "summary": summary,
        "category": category,
        "priority": priority.upper(),
        "assigned_team": assigned_team,
        "status": "open",
        "sla": sla_map.get(priority.upper(), "4 hours"),
    }
    _TICKET_STORE[ticket_id] = ticket
    return {
        "ticket_id": ticket_id,
        "status": "created",
        "priority": ticket["priority"],
        "assigned_team": assigned_team,
        "expected_response": ticket["sla"],
        "message": (
            f"Ticket {ticket_id} created and assigned to {assigned_team}. "
            f"Expected first response within {ticket['sla']}."
        ),
    }


def get_escalation_policy(category: str) -> dict:
    """
    Look up the full escalation path, SLA definitions, and automatic actions
    for a given issue category.

    category: one of the 8 intent labels
    """
    paths = {p["intent"]: p for p in _ESCALATION["escalation_paths"]}
    if category not in paths:
        return {"error": f"No escalation policy found for category '{category}'."}

    policy = paths[category]
    return {
        "category": category,
        "description": policy["description"],
        "sla_overrides": policy.get("sla_overrides", {}),
        "tiers": policy["tiers"],
        "automatic_actions": policy["automatic_actions"],
    }


# ---------------------------------------------------------------------------
# Internal dispatch helper (used by run_agent_loop)
# ---------------------------------------------------------------------------

_TOOL_IMPLEMENTATIONS: dict[str, Any] = {
    "search_knowledge_base": search_knowledge_base,
    "create_ticket": create_ticket,
    "get_escalation_policy": get_escalation_policy,
}

_INTENT_LABELS = [
    "access_request",
    "security_incident",
    "hardware_issue",
    "software_issue",
    "onboarding",
    "payments_incident",
    "expense_request",
    "general_question",
]


def _dispatch(name: str, arguments: str | dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    fn = _TOOL_IMPLEMENTATIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    result = fn(**arguments)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool schemas (Mistral function-calling format)
# ---------------------------------------------------------------------------

_SEARCH_KB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the IT knowledge base for articles relevant to the user's issue. "
            "Use kb_type='generic' for standard IT procedures, "
            "or kb_type='internal' for Veridian-specific procedures, contacts, and tooling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Key terms or a short description of the issue.",
                },
                "kb_type": {
                    "type": "string",
                    "enum": ["generic", "internal"],
                    "description": (
                        "'generic' for company-agnostic IT procedures; "
                        "'internal' for Veridian-specific procedures (Nexus, Prism, Helix, Jamf, Okta)."
                    ),
                    "default": "generic",
                },
            },
            "required": ["query"],
        },
    },
}

_CREATE_TICKET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": (
            "Create a new IT support ticket. Use this when the issue cannot be resolved "
            "immediately or requires human action. Always tell the user their ticket ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-line summary of the issue.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3"],
                    "description": (
                        "P1 = critical / business-blocking (15-min SLA); "
                        "P2 = significant impact (1-hour SLA); "
                        "P3 = limited impact (4-hour SLA)."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": _INTENT_LABELS,
                    "description": "Issue category matching the intent taxonomy.",
                },
                "assigned_team": {
                    "type": "string",
                    "description": (
                        "Team responsible for this ticket, e.g. 'IT Ops', 'Security Team', "
                        "'SRE On-Call', 'DevOps', 'People Ops'."
                    ),
                },
            },
            "required": ["summary", "priority", "category", "assigned_team"],
        },
    },
}

_GET_ESCALATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_escalation_policy",
        "description": (
            "Look up the escalation path, SLA definitions, and automatic actions "
            "for a given issue category. Use this to tell the user who to contact "
            "and what the expected response time is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": _INTENT_LABELS,
                    "description": "Issue category to look up.",
                },
            },
            "required": ["category"],
        },
    },
}

# All three tool schemas, keyed for easy lookup
_ALL_SCHEMAS: dict[str, dict] = {
    "search_knowledge_base": _SEARCH_KB_SCHEMA,
    "create_ticket": _CREATE_TICKET_SCHEMA,
    "get_escalation_policy": _GET_ESCALATION_SCHEMA,
}


def get_tool_definitions(agent_type: str = "base") -> list[dict]:
    """
    Return the list of tool schemas for the given agent type.

    agent_type="base"    -> all three tools (search_knowledge_base, create_ticket, get_escalation_policy)
    agent_type="adapted" -> create_ticket + get_escalation_policy only
                           (KB search is redundant when the classifier already knows the intent)
    """
    if agent_type == "adapted":
        return [_CREATE_TICKET_SCHEMA, _GET_ESCALATION_SCHEMA]
    return [_SEARCH_KB_SCHEMA, _CREATE_TICKET_SCHEMA, _GET_ESCALATION_SCHEMA]


# ---------------------------------------------------------------------------
# Shared agentic loop
# ---------------------------------------------------------------------------

def run_agent_loop(
    client: Mistral,
    model: str,
    messages: list,
    tools: list[dict],
    max_iterations: int = 6,
) -> dict:
    """
    Run the Mistral agentic loop until the model stops requesting tool calls
    or max_iterations is reached.

    Returns
    -------
    dict with keys:
        response        str   — final text response from the model
        tools_called    list  — names of every tool invoked (in order, with duplicates)
        tool_results    list  — [{name, arguments, result}, ...] for each call
        steps           int   — number of loop iterations completed
        llm_calls       int   — number of calls made to client.chat.complete
        latency_ms      float — total wall-clock time in milliseconds
        ticket_id       str | None — ID from the first create_ticket call, if any
    """
    t0 = time.monotonic()
    tools_called: list[str] = []
    tool_results: list[dict] = []
    steps = 0
    llm_calls = 0
    response = ""
    ticket_id: str | None = None

    for _ in range(max_iterations):
        steps += 1
        resp = client.chat.complete(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        llm_calls += 1
        msg = resp.choices[0].message

        if not msg.tool_calls:
            response = msg.content or ""
            break

        # Collect results before appending to messages (order matters for v2 SDK)
        tool_results_map: dict[str, str] = {}
        for tc in msg.tool_calls:
            result_str = _dispatch(tc.function.name, tc.function.arguments)
            tool_results_map[tc.id] = result_str
            tools_called.append(tc.function.name)
            tool_results.append({
                "name": tc.function.name,
                "arguments": tc.function.arguments,
                "result": result_str,
            })
            # Extract ticket ID from first successful create_ticket call
            if tc.function.name == "create_ticket" and ticket_id is None:
                try:
                    parsed = json.loads(result_str)
                    ticket_id = parsed.get("ticket_id")
                except (json.JSONDecodeError, AttributeError):
                    pass

        # v2 SDK: append the AssistantMessage object directly so tool_calls
        # serialise correctly, then append each tool result message.
        messages.append(msg)
        for tc in msg.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_results_map[tc.id],
            })
    else:
        response = (
            "I was unable to fully resolve this issue within the allowed steps. "
            "Please contact it-help@veridian.io or open a ticket at the IT portal."
        )

    latency_ms = (time.monotonic() - t0) * 1000
    return {
        "response": response,
        "tools_called": tools_called,
        "tool_results": tool_results,
        "steps": steps,
        "llm_calls": llm_calls,
        "latency_ms": latency_ms,
        "ticket_id": ticket_id,
    }
