"""
Base IT support agent — no classifier pre-routing.

The agent receives the user message and calls the LLM with the full tool set.
The model decides which tools to call and figures out the intent itself.

This is the baseline for the demo comparison (before Classifier Factory fine-tuning).
"""

from __future__ import annotations

from mistralai.client import Mistral

from agents.tools import get_tool_definitions, run_agent_loop

_SYSTEM_PROMPT = (
    "You are a helpful IT support agent for Veridian Systems. "
    "Resolve the user's issue as efficiently as possible. "
    "Use the available tools to look up knowledge base articles, check escalation "
    "policies, and create tickets. "
    "If the issue requires human intervention, always create a ticket and tell the "
    "user their ticket ID and expected response time."
)


class BaseAgent:
    """
    Agentic loop with no intent pre-routing.

    All three tools are always available. The model reasons about the intent and
    decides which tools to call without any classifier assistance.
    """

    def __init__(
        self,
        client: Mistral,
        model: str = "mistral-large-latest",
        max_iterations: int = 6,
    ):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations

    def run(self, user_message: str) -> dict:
        """
        Process a user message through the agentic loop.

        Returns
        -------
        dict with keys:
            response        str
            tools_called    list[str]
            tool_results    list[dict]
            steps           int
            llm_calls       int
            latency_ms      float
            ticket_id       str | None
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        return run_agent_loop(
            client=self.client,
            model=self.model,
            messages=messages,
            tools=get_tool_definitions("base"),
            max_iterations=self.max_iterations,
        )
