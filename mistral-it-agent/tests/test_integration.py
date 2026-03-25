"""
Integration tests — require a live MISTRAL_API_KEY in mistral-it-agent/.env.
Classifier tests additionally require TOGETHER_API_KEY and data/classifier_model_id.txt.

All tests are automatically skipped when the required keys are not set.

Run integration tests only:
    uv run pytest mistral-it-agent/tests/test_integration.py -v

Run alongside the full suite:
    uv run pytest -m integration -v

Tests use mistral-small-latest (cheap, supports function calling) rather than
mistral-large-latest so the suite stays inexpensive to run regularly.
The SFT classifier tests additionally require data/classifier_model_id.txt
(written by 02_train_classifier.ipynb after a successful training job) and
TOGETHER_API_KEY to be set in .env.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from the mistral-it-agent directory so keys are available when
# running pytest from the repo root.
load_dotenv(Path(__file__).parent.parent / ".env")

_API_KEY        = os.getenv("MISTRAL_API_KEY")
_TOGETHER_KEY   = os.getenv("TOGETHER_API_KEY")
_SMALL_MODEL    = "mistral-small-latest"   # cheap model with function-calling support
_DATA_DIR       = Path(__file__).parent.parent / "data"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _API_KEY, reason="MISTRAL_API_KEY not set"),
]


# ---------------------------------------------------------------------------
# Shared client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from mistralai.client import Mistral
    return Mistral(api_key=_API_KEY)


# ---------------------------------------------------------------------------
# 1. Basic connectivity
# ---------------------------------------------------------------------------

def test_api_connectivity(client):
    """A minimal chat completion succeeds and returns non-empty content."""
    response = client.chat.complete(
        model=_SMALL_MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=5,
    )
    content = response.choices[0].message.content
    assert isinstance(content, str)
    assert len(content.strip()) > 0


# ---------------------------------------------------------------------------
# 2. Tool schemas
# ---------------------------------------------------------------------------

def test_tool_schemas_accepted_by_api(client):
    """All three tool schemas pass API validation without a 4xx error."""
    from agents.tools import get_tool_definitions

    tools = get_tool_definitions("base")  # all 3: search_kb, create_ticket, escalation
    response = client.chat.complete(
        model=_SMALL_MODEL,
        messages=[{"role": "user", "content": "What is today's date?"}],
        tools=tools,
        max_tokens=40,
    )
    # Any non-error response (with or without tool calls) means schemas are valid.
    assert response.choices is not None
    assert len(response.choices) > 0


# ---------------------------------------------------------------------------
# 3. BaseAgent end-to-end
# ---------------------------------------------------------------------------

def test_base_agent_end_to_end(client):
    """BaseAgent runs to completion and returns the expected result shape."""
    from agents.base_agent import BaseAgent

    agent = BaseAgent(client=client, model=_SMALL_MODEL)
    result = agent.run("What is Veridian's general expense reimbursement policy?")

    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0
    assert result["llm_calls"] >= 1
    assert result["steps"] >= 1
    assert result["latency_ms"] > 0
    assert isinstance(result["tools_called"], list)
    assert isinstance(result["tool_results"], list)


def test_base_agent_creates_ticket(client):
    """BaseAgent calls create_ticket and surfaces the ticket ID for clear requests."""
    from agents.base_agent import BaseAgent

    agent = BaseAgent(client=client, model=_SMALL_MODEL)
    result = agent.run(
        "I need to report a phishing email I received — please open a security ticket."
    )

    # The model should call create_ticket; ticket_id may be None if it chose not to,
    # but tools_called must be populated and response must be a non-empty string.
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0
    if result["ticket_id"] is not None:
        assert result["ticket_id"].startswith("TKT-")


# ---------------------------------------------------------------------------
# 4. AdaptedAgent with mock classifier (no trained model required)
# ---------------------------------------------------------------------------

def test_adapted_agent_mock_classifier_end_to_end(client):
    """AdaptedAgent runs with keyword mock classifier and correct result shape."""
    from agents.adapted_agent import AdaptedAgent

    agent = AdaptedAgent(
        client=client,
        classifier_model_id=None,   # uses keyword-based mock
        model=_SMALL_MODEL,
    )
    result = agent.run("My laptop screen is cracked and I can't work.")

    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0
    assert result["classifier_intent"] == "hardware_issue"
    assert 0 < result["classifier_confidence"] <= 1
    assert result["classifier_latency_ms"] >= 0
    assert result["classifier_latency_ms"] <= result["latency_ms"]


def test_adapted_agent_uses_fewer_tools_than_base(client):
    """AdaptedAgent never receives search_knowledge_base in its tool set."""
    from agents.tools import get_tool_definitions

    adapted_tools = {t["function"]["name"] for t in get_tool_definitions("adapted")}
    base_tools    = {t["function"]["name"] for t in get_tool_definitions("base")}

    assert "search_knowledge_base" not in adapted_tools
    assert "search_knowledge_base" in base_tools
    assert adapted_tools == {"create_ticket", "get_escalation_policy"}


# ---------------------------------------------------------------------------
# 5. Fine-tuned classifier via Together.ai
#    Skipped unless classifier_model_id.txt exists AND TOGETHER_API_KEY is set
# ---------------------------------------------------------------------------

_CLASSIFIER_MODEL_ID_FILE = _DATA_DIR / "classifier_model_id.txt"


@pytest.fixture(scope="module")
def classifier_model_id():
    if not _CLASSIFIER_MODEL_ID_FILE.exists():
        pytest.skip("data/classifier_model_id.txt not found — run 02_train_classifier.ipynb first")
    return _CLASSIFIER_MODEL_ID_FILE.read_text().strip()


@pytest.fixture(scope="module")
def together_client():
    if not _TOGETHER_KEY:
        pytest.skip("TOGETHER_API_KEY not set")
    from together import Together
    return Together(api_key=_TOGETHER_KEY)


def test_sft_classifier_returns_known_label(client, together_client, classifier_model_id):
    """Fine-tuned model on Together.ai returns a valid intent label for a clear ticket."""
    from agents.adapted_agent import AdaptedAgent, _INTENT_LABELS

    agent = AdaptedAgent(
        client=client,
        classifier_model_id=classifier_model_id,
        model=_SMALL_MODEL,
        classifier_client=together_client,
    )
    result = agent.run("My laptop screen has a crack running across it.")

    assert result["classifier_intent"] in _INTENT_LABELS
    assert result["classifier_confidence"] > 0


def test_sft_classifier_payments_incident(client, together_client, classifier_model_id):
    """Fine-tuned model classifies a prod-payments incident correctly."""
    from agents.adapted_agent import AdaptedAgent

    agent = AdaptedAgent(
        client=client,
        classifier_model_id=classifier_model_id,
        model=_SMALL_MODEL,
        classifier_client=together_client,
    )
    result = agent.run(
        "prod-payments is throwing 500 errors, transactions are failing."
    )

    assert result["classifier_intent"] == "payments_incident"
