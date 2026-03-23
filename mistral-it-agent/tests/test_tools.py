"""
Unit tests for agents/tools.py.

Covers: search_knowledge_base, create_ticket, get_escalation_policy,
        get_tool_definitions.

NOT tested here:
  - run_agent_loop — requires a live or mocked Mistral client; covered in
    test_agents.py where the client is mocked via unittest.mock.
  - Disk I/O failures — data files are treated as stable test fixtures;
    absence is caught at import time, not per-test.
"""

import re

import pytest

from agents.tools import (
    create_ticket,
    get_escalation_policy,
    get_tool_definitions,
    search_knowledge_base,
)

# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------

def test_search_knowledge_base_generic():
    result = search_knowledge_base("VPN gateway authentication", kb_type="generic")

    assert isinstance(result, dict)
    assert "articles" in result
    assert isinstance(result["articles"], list)
    assert len(result["articles"]) > 0

    article = result["articles"][0]
    assert "id" in article
    assert "title" in article
    assert "content" in article
    # Generic KB article IDs start with "KB-"
    assert article["id"].startswith("KB-")


def test_search_knowledge_base_internal():
    result = search_knowledge_base("Nexus artifact token", kb_type="internal")

    assert isinstance(result, dict)
    assert len(result["articles"]) > 0
    # VKB-001 covers Nexus; internal KB IDs start with "VKB-"
    ids = [a["id"] for a in result["articles"]]
    assert any(i.startswith("VKB-") for i in ids)
    # Content should mention Nexus
    combined_content = " ".join(a["content"] for a in result["articles"])
    assert "Nexus" in combined_content


def test_search_knowledge_base_no_results():
    result = search_knowledge_base("zzzyyyxxx_gibberish_query_42")

    assert isinstance(result, dict)
    assert result["articles"] == []
    # Should return gracefully, not raise
    assert "result" in result


# ---------------------------------------------------------------------------
# create_ticket
# ---------------------------------------------------------------------------

def test_create_ticket():
    result = create_ticket(
        summary="Laptop screen cracked after drop",
        priority="P2",
        category="hardware_issue",
        assigned_team="IT Ops",
    )

    assert isinstance(result, dict)
    # Ticket ID format: TKT- followed by 6 uppercase hex characters
    assert re.match(r"^TKT-[0-9A-F]{6}$", result["ticket_id"]), (
        f"ticket_id '{result['ticket_id']}' does not match expected TKT-XXXXXX format"
    )
    assert result["priority"] == "P2"
    assert result["assigned_team"] == "IT Ops"
    assert result["status"] == "created"
    assert "expected_response" in result
    assert "message" in result
    assert result["ticket_id"] in result["message"]


def test_create_ticket_ids_are_unique():
    """Each create_ticket call must produce a distinct ticket ID."""
    ids = {
        create_ticket("issue A", "P3", "general_question", "IT Ops")["ticket_id"],
        create_ticket("issue B", "P3", "general_question", "IT Ops")["ticket_id"],
        create_ticket("issue C", "P3", "general_question", "IT Ops")["ticket_id"],
    }
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# get_escalation_policy
# ---------------------------------------------------------------------------

def test_get_escalation_policy_known():
    result = get_escalation_policy("security_incident")

    assert isinstance(result, dict)
    assert "error" not in result
    assert result["category"] == "security_incident"
    # Tiers are the escalation path
    assert "tiers" in result
    assert isinstance(result["tiers"], list)
    assert len(result["tiers"]) > 0
    # SLA information is in sla_overrides
    assert "sla_overrides" in result
    assert isinstance(result["sla_overrides"], dict)
    # Automatic actions document what happens without human intervention
    assert "automatic_actions" in result


def test_get_escalation_policy_payments_incident_is_p1():
    """payments_incident must document the always-P1 rule."""
    result = get_escalation_policy("payments_incident")

    assert "error" not in result
    sla_text = str(result.get("sla_overrides", {}))
    assert "P1" in sla_text


def test_get_escalation_policy_unknown():
    """Unknown category should return an error dict, not raise an exception."""
    result = get_escalation_policy("definitely_not_a_real_intent")

    assert isinstance(result, dict)
    assert "error" in result
    # Must not raise — graceful fallback is the contract


# ---------------------------------------------------------------------------
# get_tool_definitions
# ---------------------------------------------------------------------------

def test_get_tool_definitions_base():
    tools = get_tool_definitions("base")

    assert isinstance(tools, list)
    assert len(tools) == 3

    names = set()
    for tool in tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        names.add(fn["name"])

    assert "search_knowledge_base" in names
    assert "create_ticket" in names
    assert "get_escalation_policy" in names


def test_get_tool_definitions_adapted():
    tools = get_tool_definitions("adapted")

    assert isinstance(tools, list)
    assert len(tools) == 2

    names = {t["function"]["name"] for t in tools}
    assert "search_knowledge_base" not in names, (
        "adapted agent must not include search_knowledge_base — "
        "the classifier already handles routing"
    )
    assert "create_ticket" in names
    assert "get_escalation_policy" in names


def test_get_tool_definitions_schemas_are_valid():
    """Every tool schema must have the required fields and a non-empty enum for category."""
    for agent_type in ("base", "adapted"):
        for tool in get_tool_definitions(agent_type):
            fn = tool["function"]
            params = fn["parameters"]["properties"]
            if "category" in params:
                assert "enum" in params["category"]
                assert len(params["category"]["enum"]) == 8, (
                    f"category enum in {fn['name']} must list all 8 intent labels"
                )
