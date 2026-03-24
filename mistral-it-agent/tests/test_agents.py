"""
Unit tests for agents/base_agent.py and agents/adapted_agent.py.

The Mistral client is mocked via unittest.mock so no MISTRAL_API_KEY is
required and no real API calls are made.

NOT tested here:
  - Actual LLM output quality — impossible to assert without a live model.
  - Multi-step tool-calling loops — those require the model to emit tool_calls,
    which would demand a more complex mock chain; integration-tested manually
    in 03_agent_demo.ipynb.
  - Streaming — run_streaming is a thin wrapper; the underlying loop is tested here.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from agents.base_agent import BaseAgent
from agents.adapted_agent import AdaptedAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(content: str = "Here is my response.") -> MagicMock:
    """
    Return a MagicMock Mistral client whose chat.complete() returns a single
    non-tool-calling message, causing the agentic loop to terminate immediately.
    """
    mock_msg = MagicMock()
    mock_msg.tool_calls = None   # falsy → loop exits after first LLM call
    mock_msg.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.complete.return_value = mock_response
    return mock_client


def _make_mock_classifier_chat_response(intent: str) -> MagicMock:
    """
    Return a chat.complete response whose content is the intent label.
    Used to mock the SFT classifier call inside AdaptedAgent._classify().
    """
    mock_msg = MagicMock()
    mock_msg.content = intent
    mock_msg.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class TestBaseAgent:

    def test_base_agent_returns_expected_keys(self):
        client = _make_mock_client("Your ticket has been created.")
        agent = BaseAgent(client=client, model="mistral-large-latest")

        result = agent.run("My keyboard is not working.")

        required_keys = {
            "response",
            "tools_called",
            "tool_results",
            "steps",
            "llm_calls",
            "latency_ms",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_base_agent_response_is_string(self):
        client = _make_mock_client("Here is my answer.")
        result = BaseAgent(client=client).run("How do I reset MFA?")

        assert isinstance(result["response"], str)
        assert result["response"] == "Here is my answer."

    def test_base_agent_no_tool_calls_when_model_skips_tools(self):
        client = _make_mock_client()
        result = BaseAgent(client=client).run("General question.")

        assert result["tools_called"] == []
        assert result["tool_results"] == []

    def test_base_agent_counts_llm_calls(self):
        client = _make_mock_client()
        result = BaseAgent(client=client).run("Some query.")

        assert result["llm_calls"] == 1
        assert result["steps"] == 1

    def test_base_agent_latency_is_positive(self):
        client = _make_mock_client()
        result = BaseAgent(client=client).run("Some query.")

        assert result["latency_ms"] > 0

    def test_base_agent_ticket_id_is_none_when_no_ticket_created(self):
        client = _make_mock_client()
        result = BaseAgent(client=client).run("Just a question.")

        assert result["ticket_id"] is None

    def test_base_agent_uses_all_three_tools(self):
        """BaseAgent must pass all 3 tool schemas to the LLM."""
        client = _make_mock_client()
        BaseAgent(client=client).run("My screen is broken.")

        call_kwargs = client.chat.complete.call_args
        tools_passed = call_kwargs.kwargs.get("tools") or call_kwargs.args[0] if call_kwargs.args else []
        # Inspect via keyword argument
        tools_passed = client.chat.complete.call_args.kwargs["tools"]
        tool_names = {t["function"]["name"] for t in tools_passed}
        assert tool_names == {"search_knowledge_base", "create_ticket", "get_escalation_policy"}


# ---------------------------------------------------------------------------
# AdaptedAgent
# ---------------------------------------------------------------------------

class TestAdaptedAgent:

    def _make_adapted_agent(
        self,
        intent: str = "access_request",
        confidence: float = 0.95,  # unused — SFT returns fixed 0.95; kept for API compat
        response_text: str = "I have handled your request.",
    ) -> tuple[AdaptedAgent, MagicMock]:
        """Return (agent, mock_client) with SFT classifier and agentic loop both mocked.

        chat.complete is called twice per AdaptedAgent.run():
          1st call  — SFT classifier: returns the intent label as plain text
          2nd call  — agentic loop:   returns the final response (no tool calls)
        """
        cls_response  = _make_mock_classifier_chat_response(intent)
        loop_response = _make_mock_client(response_text).chat.complete.return_value

        mock_client = MagicMock()
        mock_client.chat.complete.side_effect = [cls_response, loop_response]

        agent = AdaptedAgent(
            client=mock_client,
            classifier_model_id="ft:ministral-3b:test::sft",
            model="mistral-large-latest",
        )
        return agent, mock_client

    def test_adapted_agent_returns_expected_keys(self):
        agent, _ = self._make_adapted_agent()
        result = agent.run("I need access to Nexus.")

        required_keys = {
            "response",
            "tools_called",
            "tool_results",
            "steps",
            "llm_calls",
            "latency_ms",
            "classifier_intent",
            "classifier_confidence",
            "classifier_latency_ms",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_adapted_agent_returns_classifier_metadata(self):
        agent, _ = self._make_adapted_agent(intent="access_request")
        result = agent.run("I need Nexus access for my team.")

        assert result["classifier_intent"] == "access_request"
        assert abs(result["classifier_confidence"] - 0.95) < 1e-6
        assert result["classifier_latency_ms"] >= 0

    def test_adapted_agent_injects_classifier_intent_into_system_prompt(self):
        """
        The system prompt sent to the LLM must contain the classified intent
        and the confidence rounded to the nearest percent.
        """
        agent, mock_client = self._make_adapted_agent(
            intent="access_request", confidence=0.94
        )
        agent.run("I need Prism access.")

        call_kwargs = mock_client.chat.complete.call_args.kwargs
        messages = call_kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        assert "access_request" in system_message["content"]
        assert "95%" in system_message["content"]

    def test_adapted_agent_uses_only_two_tools(self):
        """AdaptedAgent must restrict tools to create_ticket + get_escalation_policy."""
        agent, mock_client = self._make_adapted_agent()
        agent.run("My laptop battery is swollen.")

        tools_passed = mock_client.chat.complete.call_args.kwargs["tools"]
        tool_names = {t["function"]["name"] for t in tools_passed}

        assert "search_knowledge_base" not in tool_names
        assert "create_ticket" in tool_names
        assert "get_escalation_policy" in tool_names
        assert len(tool_names) == 2

    def test_adapted_agent_mock_classifier_when_no_model_id(self):
        """
        Passing classifier_model_id=None must use the keyword-based mock,
        not call client.chat.complete for classification (only called once for the loop).
        """
        mock_client = _make_mock_client()
        agent = AdaptedAgent(
            client=mock_client,
            classifier_model_id=None,
        )
        result = agent.run("There's a phishing email in my inbox.")

        assert mock_client.chat.complete.call_count == 1  # loop only, not classifier
        assert result["classifier_intent"] != ""
        assert 0 < result["classifier_confidence"] <= 1

    def test_adapted_agent_classifier_latency_is_subset_of_total(self):
        agent, _ = self._make_adapted_agent()
        result = agent.run("Some query.")

        assert result["classifier_latency_ms"] <= result["latency_ms"]
