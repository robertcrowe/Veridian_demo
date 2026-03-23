"""
Adapted IT support agent — uses the Classifier Factory fine-tuned model for intent pre-routing.

Architecture
------------
1. Classify the user message with the fine-tuned ministral-3b classifier.
2. Inject the predicted intent + confidence into the system prompt.
3. Run the agentic loop with a restricted tool set:
   only create_ticket + get_escalation_policy (KB search is redundant when the
   classifier already knows the intent category).

This is the "after fine-tuning" agent for the demo comparison.
"""

from __future__ import annotations

import time

from mistralai.client import Mistral

from agents.tools import get_tool_definitions, run_agent_loop

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful IT support agent for Veridian Systems. "
    "The request has been pre-classified as: {intent} (confidence: {score:.0%}). "
    "Route accordingly. Only call tools if additional information is needed beyond "
    "what you already know about this intent category."
)

# Keyword-based mock classifier — used when classifier_model_id is None
_MOCK_RULES: list[tuple[str, list[str]]] = [
    ("security_incident",  ["phishing", "ransomware", "breach", "stolen", "malware", "suspicious", "hack", "unauthorized", "sev1", "sev2"]),
    ("payments_incident",  ["prod-payments", "payment", "transaction", "braintree", "stripe", "duplicate charge", "refund"]),
    ("access_request",     ["access", "permission", "provision", "revoke", "offboard", "nexus", "prism", "aws iam", "github org"]),
    ("hardware_issue",     ["laptop", "screen", "keyboard", "battery", "monitor", "printer", "webcam", "headset", "macbook", "swollen"]),
    ("software_issue",     ["install", "software", "license", "expired", "docker", "jetbrains", "zoom", "mfa", "okta", "sso", "password"]),
    ("onboarding",         ["new hire", "day 1", "day one", "onboard", "start date", "helix on-call", "vpn profile", "mdm enroll"]),
    ("expense_request",    ["expense", "expensify", "reimburs", "home office", "ergonomic", "budget", "l&d", "conference"]),
]


def _mock_classify(text: str) -> tuple[str, float]:
    """Return (intent, confidence) using simple keyword matching."""
    text_lower = text.lower()
    for intent, keywords in _MOCK_RULES:
        if any(kw in text_lower for kw in keywords):
            return intent, 0.88
    return "general_question", 0.72


class AdaptedAgent:
    """
    Agentic loop with Classifier Factory intent pre-routing.

    Parameters
    ----------
    client : Mistral
        Authenticated Mistral client.
    classifier_model_id : str | None
        Fine-tuned Classifier Factory model ID from 02_train_classifier.ipynb.
        Pass None to use the keyword-based mock (local testing without a trained model).
    model : str
        Generative model for the agentic loop.
    max_iterations : int
        Maximum number of tool-call rounds before giving up.
    """

    def __init__(
        self,
        client: Mistral,
        classifier_model_id: str | None,
        model: str = "mistral-large-latest",
        max_iterations: int = 6,
    ):
        self.client = client
        self.classifier_model_id = classifier_model_id
        self.model = model
        self.max_iterations = max_iterations

    def _classify(self, user_message: str) -> tuple[str, float]:
        """
        Return (intent, confidence).
        Uses the fine-tuned classifier if available, otherwise falls back to the mock.
        """
        if self.classifier_model_id is None:
            return _mock_classify(user_message)

        response = self.client.classifiers.classify(
            model=self.classifier_model_id,
            inputs=[{"text": user_message}],
        )
        probs = response.results[0].probabilities
        intent = max(probs, key=probs.get)
        return intent, probs[intent]

    def run(self, user_message: str) -> dict:
        """
        Classify then run the agentic loop.

        Returns
        -------
        dict with keys (superset of BaseAgent.run return):
            response                str
            tools_called            list[str]
            tool_results            list[dict]
            steps                   int
            llm_calls               int
            latency_ms              float   — total wall-clock time (classify + loop)
            ticket_id               str | None
            classifier_intent       str
            classifier_confidence   float
            classifier_latency_ms   float
        """
        t0 = time.monotonic()

        # Step 1: classify
        t_cls = time.monotonic()
        intent, confidence = self._classify(user_message)
        classifier_latency_ms = (time.monotonic() - t_cls) * 1000

        # Step 2: build intent-aware system prompt
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(intent=intent, score=confidence)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Step 3: agentic loop with restricted tool set
        result = run_agent_loop(
            client=self.client,
            model=self.model,
            messages=messages,
            tools=get_tool_definitions("adapted"),
            max_iterations=self.max_iterations,
        )

        total_latency_ms = (time.monotonic() - t0) * 1000

        return {
            **result,
            "latency_ms": total_latency_ms,
            "classifier_intent": intent,
            "classifier_confidence": confidence,
            "classifier_latency_ms": classifier_latency_ms,
        }
