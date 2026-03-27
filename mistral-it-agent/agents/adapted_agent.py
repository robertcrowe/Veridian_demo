"""
Adapted IT support agent — uses the SFT fine-tuned model for intent pre-routing.

Architecture
------------
1. Classify the user message with the fine-tuned Mistral-7B-Instruct-v0.2 model (hosted on
   Together.ai). Falls back to a keyword mock when no model is available.
2. Inject the predicted intent + confidence into the system prompt.
3. Run the agentic loop with a restricted tool set:
   only create_ticket + get_escalation_policy (KB search is redundant when the
   classifier already knows the intent category).

The classifier call uses an OpenAI-compatible client (Together.ai) passed in via
`classifier_client`. The agentic loop uses the standard Mistral client.

This is the "after fine-tuning" agent for the demo comparison.
"""

from __future__ import annotations

import time

from mistralai.client import Mistral

from agents.tools import get_tool_definitions, run_agent_loop

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful IT support agent for Veridian Systems. "
    "This request has been pre-classified as: {intent} (confidence: {score:.0%}). "
    "The category is confirmed — act on it immediately without exploratory tool calls. "
    "Use get_escalation_policy for incidents that need an escalation path "
    "(security_incident, payments_incident). "
    "Use create_ticket for all other actionable requests (access_request, hardware_issue, "
    "software_issue, onboarding, expense_request). "
    "For general_question, answer directly without any tool calls. "
    "Use the minimum number of tools needed, then give a concise response."
)

# Must match the system prompt used when generating the SFT training data in
# 01_data_prep.ipynb so that inference conditions match training conditions.
_CLASSIFIER_SYSTEM_PROMPT = (
    "Classify the IT support request into exactly one of the following categories: "
    "access_request, security_incident, hardware_issue, software_issue, onboarding, "
    "payments_incident, expense_request, general_question. "
    "Disambiguation rules: "
    "onboarding vs access_request: use onboarding when a new hire in their first 1-2 weeks cannot access a tool as part of initial setup; "
    "use access_request when an established employee requests a permission change, new project access, role transfer, contractor provisioning, or offboarding — "
    "even if they recently joined, once initial setup is complete any further access change is access_request. "
    "payments_incident vs security_incident: use payments_incident when prod-payments has a service failure or degradation (HTTP errors, timeouts, crashes, OOM, duplicate charges, cert expiry, deployment failures); "
    "use security_incident when the trigger is a threat or credential exposure — phishing, stolen device, leaked API key, unauthorized access, malware — "
    "even if the exposed credential or affected system belongs to prod-payments. "
    "expense_request vs general_question: use expense_request when the person has a specific purchase in mind and is seeking approval or process guidance for it; "
    "use general_question when asking about a policy or process with no specific purchase stated. "
    "software_issue vs access_request: use software_issue when the person has access but something is broken, misconfigured, or expired; "
    "use access_request when they need access granted for the first time or revoked entirely. "
    "onboarding vs general_question: use onboarding when a new hire cannot access a specific tool needed for their role; "
    "use general_question for policy or how-to questions even if the asker is new. "
    "hardware_issue vs onboarding: use onboarding when a new hire's device is not yet set up or enrolled; "
    "use hardware_issue when a device is malfunctioning for an existing employee or after initial setup is complete. "
    "Respond with only the category label, nothing else."
)

_INTENT_LABELS = frozenset([
    "access_request", "security_incident", "hardware_issue", "software_issue",
    "onboarding", "payments_incident", "expense_request", "general_question",
])

# Keyword-based mock classifier — used when classifier_model_id is None
# Rule order matters: first match wins. More specific / higher-stakes intents first.
_MOCK_RULES: list[tuple[str, list[str]]] = [
    ("security_incident",  ["phishing", "ransomware", "breach", "stolen", "malware", "suspicious", "hack", "unauthorized", "sev1", "sev2"]),
    ("payments_incident",  ["prod-payments", "payment", "transaction", "braintree", "stripe", "duplicate charge", "refund"]),
    ("access_request",     ["access", "permission", "provision", "revoke", "offboard", "nexus", "prism", "aws iam", "github org", "401", "403"]),
    # onboarding before hardware_issue — new-hire requests often mention devices
    ("onboarding",         ["new hire", "day 1", "day one", "onboard", "start date", "starting today", "first day", "mdm enroll", "mdm", "helix on-call", "vpn profile", "just joined", "i'm starting", "im starting"]),
    ("hardware_issue",     ["laptop", "screen", "keyboard", "battery", "monitor", "printer", "webcam", "headset", "macbook", "swollen"]),
    ("software_issue",     ["install", "software", "license", "expired", "docker", "jetbrains", "zoom", "mfa", "okta", "sso", "password"]),
    ("expense_request",    ["expense", "expensify", "reimburs", "home office", "ergonomic", "standing desk", "sit-stand", "desk", "budget", "l&d", "conference", "doctor recommend", "reimburse"]),
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
    Agentic loop with SFT intent pre-routing.

    Parameters
    ----------
    client : Mistral
        Authenticated Mistral client — used for the agentic loop only.
    classifier_model_id : str | None
        Fine-tuned model ID (e.g. Together.ai output_name from 02_train_classifier.ipynb).
        Pass None to use the keyword-based mock (local testing without a trained model).
    model : str
        Generative model for the agentic loop.
    max_iterations : int
        Maximum number of tool-call rounds before giving up.
    classifier_client : optional
        OpenAI-compatible client for the classifier call (e.g. Together.ai client).
        When provided, uses client.completions.create() with the Mistral [INST] prompt
        format (base model, no chat template). When None, falls back to
        self.client.chat.complete() (Mistral).
    """

    def __init__(
        self,
        client: Mistral,
        classifier_model_id: str | None,
        model: str = "mistral-large-latest",
        max_iterations: int = 6,
        classifier_client=None,
    ):
        self.client = client
        self.classifier_model_id = classifier_model_id
        self.model = model
        self.max_iterations = max_iterations
        self._classifier_client = classifier_client

    def _classify(self, user_message: str) -> tuple[str, float, int, int]:
        """
        Return (intent, confidence, prompt_tokens, completion_tokens).
        Uses the fine-tuned model via Together.ai if classifier_client and
        classifier_model_id are set. Falls back to keyword mock otherwise.
        Confidence is fixed at 0.95 for a recognised label, 0.70 if unexpected.
        Token counts are 0 for the keyword mock path.
        """
        if self.classifier_model_id is None:
            intent, confidence = _mock_classify(user_message)
            return intent, confidence, 0, 0

        if self._classifier_client is not None:
            # Mistral-7B-v0.1 is a base model with no chat template — use the
            # text completions endpoint with the Mistral instruction format.
            prompt = f"<s>[INST] {_CLASSIFIER_SYSTEM_PROMPT}\n\n{user_message} [/INST]"
            response = self._classifier_client.completions.create(
                model=self.classifier_model_id,
                prompt=prompt,
                temperature=0,
                max_tokens=20,
            )
            intent = response.choices[0].text.strip().lower()
        else:
            # Mistral client (legacy / fallback path)
            response = self.client.chat.complete(
                model=self.classifier_model_id,
                messages=msgs,
                temperature=0,
                max_tokens=20,
            )
            intent = response.choices[0].message.content.strip().lower()

        try:
            cls_prompt_tokens = int(response.usage.prompt_tokens)
            cls_completion_tokens = int(response.usage.completion_tokens)
        except (AttributeError, TypeError, ValueError):
            cls_prompt_tokens = cls_completion_tokens = 0

        if intent not in _INTENT_LABELS:
            intent = "general_question"
            return intent, 0.70, cls_prompt_tokens, cls_completion_tokens
        return intent, 0.95, cls_prompt_tokens, cls_completion_tokens

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
        intent, confidence, cls_prompt_tokens, cls_completion_tokens = self._classify(user_message)
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
            "classifier_prompt_tokens": cls_prompt_tokens,
            "classifier_completion_tokens": cls_completion_tokens,
        }
